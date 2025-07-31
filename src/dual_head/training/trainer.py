"""
Custom trainer for Dual-Head model training.

This trainer extends HuggingFace Trainer to handle:
1. Multi-objective loss computation
2. Preference data training
3. Dual-head specific metrics and logging
4. Efficient mixed training (SFT + preference)
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Union, Any, List
import wandb
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import EvalPrediction
from transformers.integrations import WandbCallback

from ..dual_head_model import DualHeadModel
from .losses import DualHeadLoss, DualHeadLossConfig


class DualHeadTrainingArguments(TrainingArguments):
    """Extended training arguments for Dual-Head training."""
    
    def __init__(
        self,
        # Dual-Head specific arguments
        lambda_r: float = 1.0,
        lambda_g: float = 0.01,
        beta_r: float = 1.0,
        label_smoothing: float = 0.0,
        length_normalization: bool = False,
        
        # Preference training arguments
        preference_data_ratio: float = 1.0,  # Ratio of preference data in mixed training
        
        # Logging and evaluation
        log_gating_stats: bool = True,
        eval_preference_accuracy: bool = True,
        
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # Store Dual-Head specific arguments
        self.lambda_r = lambda_r
        self.lambda_g = lambda_g
        self.beta_r = beta_r
        self.label_smoothing = label_smoothing
        self.length_normalization = length_normalization
        self.preference_data_ratio = preference_data_ratio
        self.log_gating_stats = log_gating_stats
        self.eval_preference_accuracy = eval_preference_accuracy


class DualHeadTrainer(Trainer):
    """
    Trainer for Dual-Head models with multi-objective optimization.
    
    This trainer handles:
    - Multi-objective loss computation (LM + preference + gating)
    - Preference data training with ARM-style loss
    - Dual-head specific metrics and logging
    - Memory-efficient training with mixed precision
    """
    
    def __init__(
        self,
        model: DualHeadModel,
        args: DualHeadTrainingArguments,
        loss_config: Optional[DualHeadLossConfig] = None,
        **kwargs
    ):
        super().__init__(model=model, args=args, **kwargs)
        
        # Initialize loss function
        if loss_config is None:
            loss_config = DualHeadLossConfig(
                lambda_r=args.lambda_r,
                lambda_g=args.lambda_g,
                beta_r=args.beta_r,
                label_smoothing=args.label_smoothing,
                length_normalization=args.length_normalization
            )
        
        self.loss_fn = DualHeadLoss(loss_config)
        self.loss_config = loss_config
        
        # Training state
        self.current_epoch = 0
        self.preference_data_ratio = args.preference_data_ratio
        
        # Metrics tracking
        self.gating_stats_history = []
        self.preference_accuracy_history = []
    
    def compute_loss(
        self,
        model: DualHeadModel,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch=None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Compute loss for Dual-Head training.
        
        This method handles both SFT and preference training depending on
        the input format.
        """
        # Check if this is preference data
        has_preference_data = 'chosen_input_ids' in inputs and 'rejected_input_ids' in inputs
        
        if has_preference_data:
            return self._compute_preference_loss(model, inputs, return_outputs)
        else:
            return self._compute_sft_loss(model, inputs, return_outputs)
    
    def _compute_sft_loss(
        self,
        model: DualHeadModel,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Compute loss for supervised fine-tuning data."""
        # Forward pass
        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs.get('attention_mask'),
            labels=inputs.get('labels'),
            return_dict=True
        )
        
        # Prepare model outputs for loss computation
        model_outputs = {
            'fused_logits': outputs.logits,
            'gating_coefficients': getattr(outputs, 'gating_coefficients', None),
            'hidden_states': outputs.hidden_states
        }
        
        # Compute loss
        loss, metrics = self.loss_fn(
            model_outputs=model_outputs,
            batch=inputs,
            return_metrics=True
        )
        
        # Log metrics if training
        if self.state.is_world_process_zero and metrics:
            self._log_metrics(metrics, prefix="train/sft")
        
        if return_outputs:
            return loss, outputs
        else:
            return loss
    
    def _compute_preference_loss(
        self,
        model: DualHeadModel,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Compute loss for preference data."""
        # Forward pass for chosen sequence
        chosen_outputs = model(
            input_ids=inputs['chosen_input_ids'],
            attention_mask=inputs.get('chosen_attention_mask'),
            labels=inputs.get('chosen_labels'),
            return_dict=True
        )
        
        # Forward pass for rejected sequence
        rejected_outputs = model(
            input_ids=inputs['rejected_input_ids'],
            attention_mask=inputs.get('rejected_attention_mask'),
            labels=inputs.get('rejected_labels'),
            return_dict=True
        )
        
        # Get dual head outputs
        dual_heads = model.dual_heads
        
        # Ensure hidden states have correct dtype
        chosen_hidden = chosen_outputs.hidden_states.to(dtype=torch.bfloat16)
        rejected_hidden = rejected_outputs.hidden_states.to(dtype=torch.bfloat16)
        
        chosen_head_outputs = dual_heads(chosen_hidden, return_dict=True)
        rejected_head_outputs = dual_heads(rejected_hidden, return_dict=True)
        
        # Prepare model outputs for loss computation
        model_outputs = {
            'fused_logits': chosen_outputs.logits,  # Use chosen for LM loss
            'gating_coefficients': getattr(chosen_outputs, 'gating_coefficients', None),
            'hidden_states': chosen_outputs.hidden_states,
            'chosen_rm_logits': chosen_head_outputs['rm_logits'],
            'rejected_rm_logits': rejected_head_outputs['rm_logits']
        }
        
        # Compute loss
        loss, metrics = self.loss_fn(
            model_outputs=model_outputs,
            batch=inputs,
            return_metrics=True
        )
        
        # Log metrics if training
        if self.state.is_world_process_zero and metrics:
            self._log_metrics(metrics, prefix="train/preference")
            
            # Track preference accuracy
            if 'reward_accuracy' in metrics:
                self.preference_accuracy_history.append(metrics['reward_accuracy'].item())
        
        # Return chosen outputs for consistency
        if return_outputs:
            return loss, chosen_outputs
        else:
            return loss
    
    def evaluation_loop(
        self,
        dataloader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval"
    ):
        """Custom evaluation loop with Dual-Head specific metrics."""
        # Run standard evaluation
        output = super().evaluation_loop(
            dataloader=dataloader,
            description=description,
            prediction_loss_only=prediction_loss_only,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix
        )
        
        # Add custom metrics
        if hasattr(output, 'metrics') and self.args.log_gating_stats:
            # Compute gating statistics on evaluation data
            gating_stats = self._compute_evaluation_gating_stats(dataloader)
            output.metrics.update({f"{metric_key_prefix}_{k}": v for k, v in gating_stats.items()})
        
        return output
    
    def _compute_evaluation_gating_stats(self, dataloader) -> Dict[str, float]:
        """Compute gating statistics on evaluation data."""
        model = self.model
        model.eval()
        
        all_gating_coeffs = []
        all_attention_masks = []
        
        with torch.no_grad():
            for batch in dataloader:
                # Move to device
                batch = {k: v.to(self.args.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                if 'chosen_input_ids' in batch:
                    # Preference data - use chosen sequence
                    outputs = model(
                        input_ids=batch['chosen_input_ids'],
                        attention_mask=batch.get('chosen_attention_mask'),
                        return_dict=True
                    )
                    attention_mask = batch.get('chosen_attention_mask')
                else:
                    # SFT data
                    outputs = model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch.get('attention_mask'),
                        return_dict=True
                    )
                    attention_mask = batch.get('attention_mask')
                
                # Collect gating coefficients
                if hasattr(outputs, 'gating_coefficients'):
                    all_gating_coeffs.append(outputs.gating_coefficients)
                    if attention_mask is not None:
                        all_attention_masks.append(attention_mask)
        
        if not all_gating_coeffs:
            return {}
        
        # Concatenate all coefficients
        gating_coeffs = torch.cat(all_gating_coeffs, dim=0)
        attention_masks = torch.cat(all_attention_masks, dim=0) if all_attention_masks else None
        
        # Compute statistics using loss function
        stats = self.loss_fn.compute_gating_statistics(gating_coeffs, attention_masks)
        
        # Convert to float for logging
        return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in stats.items()}
    
    def _log_metrics(self, metrics: Dict[str, torch.Tensor], prefix: str = "train"):
        """Log metrics to wandb and console."""
        if not self.state.is_world_process_zero:
            return
        
        # Convert tensors to scalars
        log_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                log_metrics[f"{prefix}/{key}"] = value.item()
            else:
                log_metrics[f"{prefix}/{key}"] = value
        
        # Log to wandb if available
        if self.args.report_to and "wandb" in self.args.report_to:
            wandb.log(log_metrics, step=self.state.global_step)
        
        # Log gating statistics for monitoring
        if prefix == "train/preference" and self.args.log_gating_stats:
            gating_keys = [k for k in log_metrics.keys() if 'gating' in k or 'contribution' in k]
            if gating_keys:
                gating_stats = {k: log_metrics[k] for k in gating_keys}
                self.gating_stats_history.append(gating_stats)
    
    def on_epoch_begin(self):
        """Called at the beginning of each epoch."""
        super().on_epoch_begin()
        self.current_epoch += 1
        
        # Log epoch information
        if self.state.is_world_process_zero:
            print(f"\n=== Epoch {self.current_epoch} ===")
            
            # Log recent gating statistics
            if self.gating_stats_history:
                recent_stats = self.gating_stats_history[-1]
                print(f"Recent gating stats: {recent_stats}")
            
            # Log recent preference accuracy
            if self.preference_accuracy_history:
                recent_acc = self.preference_accuracy_history[-10:]  # Last 10 batches
                avg_acc = sum(recent_acc) / len(recent_acc)
                print(f"Recent preference accuracy: {avg_acc:.3f}")
    
    def on_train_end(self):
        """Called at the end of training."""
        super().on_train_end()
        
        if self.state.is_world_process_zero:
            print("\n=== Training Complete ===")
            
            # Log final statistics
            if self.gating_stats_history:
                final_stats = self.gating_stats_history[-1]
                print(f"Final gating statistics: {final_stats}")
            
            if self.preference_accuracy_history:
                final_acc = sum(self.preference_accuracy_history[-100:]) / min(100, len(self.preference_accuracy_history))
                print(f"Final preference accuracy (last 100 batches): {final_acc:.3f}")
            
            # Save final model statistics
            param_stats = self.model.get_parameter_statistics()
            print(f"\nModel parameter statistics:")
            for key, value in param_stats.items():
                print(f"  {key}: {value}")
    
    def create_optimizer(self):
        """Create optimizer with different learning rates for different components."""
        if self.optimizer is None:
            # Separate parameters by component
            backbone_params = []
            head_params = []
            gating_params = []
            
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                
                if 'backbone' in name:
                    backbone_params.append(param)
                elif 'dual_heads' in name:
                    head_params.append(param)
                elif 'gating' in name or 'fusion' in name:
                    gating_params.append(param)
                else:
                    head_params.append(param)  # Default to head params
            
            # Create parameter groups with different learning rates
            param_groups = []
            
            if backbone_params:
                param_groups.append({
                    'params': backbone_params,
                    'lr': self.args.learning_rate * 0.1,  # Lower LR for backbone
                    'weight_decay': self.args.weight_decay
                })
            
            if head_params:
                param_groups.append({
                    'params': head_params,
                    'lr': self.args.learning_rate,  # Standard LR for heads
                    'weight_decay': self.args.weight_decay
                })
            
            if gating_params:
                param_groups.append({
                    'params': gating_params,
                    'lr': self.args.learning_rate * 1.5,  # Higher LR for gating
                    'weight_decay': self.args.weight_decay * 0.5  # Lower weight decay
                })
            
            # Create optimizer
            from torch.optim import AdamW
            self.optimizer = AdamW(param_groups)
        
        return self.optimizer
    
    def get_train_dataloader(self):
        """Get training dataloader with mixed data sampling if needed."""
        return super().get_train_dataloader()
    
    def compute_metrics(self, eval_pred: EvalPrediction) -> Dict[str, float]:
        """Compute custom metrics for evaluation."""
        predictions, labels = eval_pred.predictions, eval_pred.label_ids
        
        metrics = {}
        
        # Standard perplexity
        if predictions.ndim == 3:  # [batch_size, seq_len, vocab_size]
            # Reshape for cross entropy
            predictions = predictions.reshape(-1, predictions.shape[-1])
            labels = labels.reshape(-1)
            
            # Compute loss only on non-ignored tokens
            valid_mask = labels != -100
            if valid_mask.sum() > 0:
                valid_predictions = predictions[valid_mask]
                valid_labels = labels[valid_mask]
                
                loss = nn.CrossEntropyLoss()(valid_predictions, valid_labels)
                perplexity = torch.exp(loss)
                
                metrics['perplexity'] = perplexity.item()
                metrics['eval_loss'] = loss.item()
        
        return metrics
    
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """Save model with dual-head specific information."""
        super().save_model(output_dir, _internal_call)
        
        if output_dir is not None and self.state.is_world_process_zero:
            # Save training statistics
            stats = {
                'parameter_statistics': self.model.get_parameter_statistics(),
                'loss_config': {
                    'lambda_r': self.loss_config.lambda_r,
                    'lambda_g': self.loss_config.lambda_g,
                    'beta_r': self.loss_config.beta_r,
                    'length_normalization': self.loss_config.length_normalization
                },
                'gating_stats_history': self.gating_stats_history[-10:],  # Last 10 entries
                'preference_accuracy_history': self.preference_accuracy_history[-100:]  # Last 100 entries
            }
            
            import json
            with open(f"{output_dir}/dual_head_stats.json", "w") as f:
                json.dump(stats, f, indent=2, default=str)  # default=str for tensor serialization