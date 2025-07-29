"""
Loss functions for Dual-Head model training.

This module implements the multi-objective loss function combining:
1. Language modeling loss for fluency
2. Preference loss for alignment (ARM-style)
3. Gating regularization for balanced head usage
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass


@dataclass
class DualHeadLossConfig:
    """Configuration for Dual-Head loss computation."""
    lambda_r: float = 1.0      # Preference loss weight
    lambda_g: float = 0.01     # Gating regularization weight
    beta_r: float = 1.0        # Reward model temperature
    label_smoothing: float = 0.0  # Label smoothing for LM loss
    length_normalization: bool = False  # Whether to normalize by sequence length


class DualHeadLoss(nn.Module):
    """
    Multi-objective loss function for Dual-Head training.
    
    The total loss combines three components:
    L_total = L_LM + λ_R * L_pref + λ_G * L_α
    
    Where:
    - L_LM: Standard language modeling loss on fused logits
    - L_pref: Preference loss using autoregressive reward modeling
    - L_α: Entropy regularization to prevent gating collapse
    """
    
    def __init__(self, config: DualHeadLossConfig):
        super().__init__()
        self.config = config
        self.lambda_r = config.lambda_r
        self.lambda_g = config.lambda_g
        self.beta_r = config.beta_r
        self.label_smoothing = config.label_smoothing
        self.length_normalization = config.length_normalization
    
    def forward(
        self,
        model_outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        return_metrics: bool = True
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Compute the total multi-objective loss.
        
        Args:
            model_outputs: Dictionary containing model outputs:
                - 'fused_logits': Fused logits [batch_size, seq_len, vocab_size]
                - 'lm_logits': LM head logits [batch_size, seq_len, vocab_size]
                - 'rm_logits': RM head logits [batch_size, seq_len, vocab_size]
                - 'gating_coefficients': Gating weights [batch_size, seq_len, 1]
                - 'hidden_states': Backbone hidden states [batch_size, seq_len, hidden_size]
            batch: Training batch containing:
                - 'input_ids': Input token IDs [batch_size, seq_len]
                - 'attention_mask': Attention mask [batch_size, seq_len]
                - 'labels': Target labels [batch_size, seq_len]
                - (Optional) 'chosen_input_ids', 'rejected_input_ids' for preference training
            return_metrics: Whether to return detailed metrics
            
        Returns:
            Tuple of (total_loss, metrics_dict)
        """
        losses = {}
        metrics = {}
        
        # 1. Language Modeling Loss
        lm_loss = self.compute_language_modeling_loss(
            logits=model_outputs['fused_logits'],
            labels=batch['labels'],
            attention_mask=batch.get('attention_mask')
        )
        losses['lm_loss'] = lm_loss
        
        # 2. Gating Regularization Loss
        gating_loss = self.compute_gating_regularization_loss(
            gating_coefficients=model_outputs['gating_coefficients'],
            attention_mask=batch.get('attention_mask')
        )
        losses['gating_loss'] = gating_loss
        
        # 3. Preference Loss (if preference data is available)
        pref_loss = torch.tensor(0.0, device=lm_loss.device)
        if 'chosen_input_ids' in batch and 'rejected_input_ids' in batch:
            pref_loss, pref_metrics = self.compute_preference_loss(
                model_outputs=model_outputs,
                chosen_input_ids=batch['chosen_input_ids'],
                rejected_input_ids=batch['rejected_input_ids'],
                chosen_attention_mask=batch.get('chosen_attention_mask'),
                rejected_attention_mask=batch.get('rejected_attention_mask')
            )
            losses['preference_loss'] = pref_loss
            metrics.update(pref_metrics)
        
        # Total loss
        total_loss = (
            lm_loss + 
            self.lambda_r * pref_loss + 
            self.lambda_g * gating_loss
        )
        
        # Compile metrics
        if return_metrics:
            metrics.update({
                'total_loss': total_loss.detach(),
                'lm_loss': lm_loss.detach(),
                'preference_loss': pref_loss.detach(),
                'gating_loss': gating_loss.detach(),
                'gating_entropy': -gating_loss.detach(),  # Entropy is negative of regularization loss
                'lambda_r': self.lambda_r,
                'lambda_g': self.lambda_g,
            })
            
            # Add gating statistics
            gating_stats = self.compute_gating_statistics(
                model_outputs['gating_coefficients'],
                batch.get('attention_mask')
            )
            metrics.update(gating_stats)
        
        return total_loss, metrics if return_metrics else None
    
    def compute_language_modeling_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute standard causal language modeling loss.
        
        Args:
            logits: Model logits [batch_size, seq_len, vocab_size]
            labels: Target labels [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Language modeling loss (scalar)
        """
        # Shift logits and labels for causal language modeling
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Flatten for loss computation
        vocab_size = shift_logits.size(-1)
        shift_logits = shift_logits.view(-1, vocab_size)
        shift_labels = shift_labels.view(-1)
        
        # Compute cross-entropy loss with optional label smoothing
        if self.label_smoothing > 0:
            # Label smoothing implementation
            log_probs = F.log_softmax(shift_logits, dim=-1)
            nll_loss = F.nll_loss(log_probs, shift_labels, reduction='none', ignore_index=-100)
            smooth_loss = -log_probs.mean(dim=-1)
            
            # Apply label smoothing
            loss = (1 - self.label_smoothing) * nll_loss + self.label_smoothing * smooth_loss
            
            # Mask out ignored tokens
            valid_mask = (shift_labels != -100)
            loss = loss * valid_mask
            loss = loss.sum() / valid_mask.sum()
        else:
            loss = F.cross_entropy(shift_logits, shift_labels, reduction='mean', ignore_index=-100)
        
        return loss
    
    def compute_preference_loss(
        self,
        model_outputs: Dict[str, torch.Tensor],
        chosen_input_ids: torch.Tensor,
        rejected_input_ids: torch.Tensor,
        chosen_attention_mask: Optional[torch.Tensor] = None,
        rejected_attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute preference loss using autoregressive reward modeling.
        
        This implements the ARM loss:
        L_pref = -E[log σ(β_r * (R_chosen - R_rejected))]
        
        where R(x, Y) = Σ_t log π_r(y_t | x, y_{<t})
        
        Args:
            model_outputs: Model outputs containing RM logits and hidden states
            chosen_input_ids: Chosen response token IDs [batch_size, chosen_len]
            rejected_input_ids: Rejected response token IDs [batch_size, rejected_len]
            chosen_attention_mask: Attention mask for chosen [batch_size, chosen_len]
            rejected_attention_mask: Attention mask for rejected [batch_size, rejected_len]
            
        Returns:
            Tuple of (preference_loss, metrics_dict)
        """
        device = chosen_input_ids.device
        
        # Compute rewards for chosen and rejected sequences
        chosen_rewards = self.compute_sequence_rewards(
            input_ids=chosen_input_ids,
            rm_logits=model_outputs.get('chosen_rm_logits'),
            attention_mask=chosen_attention_mask
        )
        
        rejected_rewards = self.compute_sequence_rewards(
            input_ids=rejected_input_ids,
            rm_logits=model_outputs.get('rejected_rm_logits'),
            attention_mask=rejected_attention_mask
        )
        
        # Compute preference loss: -log σ(β_r * (R_chosen - R_rejected))
        reward_diff = chosen_rewards - rejected_rewards
        pref_loss = -F.logsigmoid(self.beta_r * reward_diff).mean()
        
        # Compute metrics
        with torch.no_grad():
            accuracy = (chosen_rewards > rejected_rewards).float().mean()
            reward_margin = reward_diff.mean()
            chosen_mean = chosen_rewards.mean()
            rejected_mean = rejected_rewards.mean()
            
            # Compute win rate (how often chosen > rejected)
            win_rate = (reward_diff > 0).float().mean()
        
        metrics = {
            'chosen_rewards': chosen_mean,
            'rejected_rewards': rejected_mean,
            'reward_accuracy': accuracy,
            'reward_margin': reward_margin,
            'reward_win_rate': win_rate,
        }
        
        return pref_loss, metrics
    
    def compute_sequence_rewards(
        self,
        input_ids: torch.Tensor,
        rm_logits: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute sequence-level rewards from token-level RM logits.
        
        R(x, Y) = Σ_t log π_r(y_t | x, y_{<t})
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            rm_logits: RM head logits [batch_size, seq_len, vocab_size]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Sequence rewards [batch_size]
        """
        if rm_logits is None:
            # If RM logits not provided, assume we need to compute them
            # This would require model forward pass, handled externally
            raise ValueError("RM logits must be provided for reward computation")
        
        # Convert logits to log probabilities with temperature scaling
        log_probs = F.log_softmax(rm_logits / self.beta_r, dim=-1)
        
        # Shift inputs to align with predictions (ignore first token)
        target_ids = input_ids[:, 1:]  # [batch_size, seq_len-1]
        log_probs = log_probs[:, :-1, :]  # [batch_size, seq_len-1, vocab_size]
        
        # Gather log probabilities for target tokens
        token_rewards = torch.gather(
            log_probs, 
            dim=-1, 
            index=target_ids.unsqueeze(-1)
        ).squeeze(-1)  # [batch_size, seq_len-1]
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Shift attention mask to align with token rewards
            mask = attention_mask[:, 1:]  # [batch_size, seq_len-1]
            token_rewards = token_rewards * mask
            
            if self.length_normalization:
                # Normalize by number of valid tokens
                valid_lengths = mask.sum(dim=-1, keepdim=True).clamp(min=1)
                sequence_rewards = token_rewards.sum(dim=-1) / valid_lengths.squeeze(-1)
            else:
                sequence_rewards = token_rewards.sum(dim=-1)
        else:
            if self.length_normalization:
                sequence_rewards = token_rewards.mean(dim=-1)
            else:
                sequence_rewards = token_rewards.sum(dim=-1)
        
        return sequence_rewards
    
    def compute_gating_regularization_loss(
        self,
        gating_coefficients: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute entropy regularization loss to prevent gating collapse.
        
        L_α = -λ_G * E_t[α_t * log(α_t) + (1-α_t) * log(1-α_t)]
        
        This encourages balanced utilization of both heads.
        
        Args:
            gating_coefficients: Gating coefficients [batch_size, seq_len, 1]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Gating regularization loss (scalar)
        """
        # Ensure coefficients are in valid range to avoid log(0)
        eps = 1e-8
        alpha = torch.clamp(gating_coefficients.squeeze(-1), eps, 1-eps)  # [batch_size, seq_len]
        
        # Compute binary entropy: -[α*log(α) + (1-α)*log(1-α)]
        entropy = -(alpha * torch.log(alpha) + (1 - alpha) * torch.log(1 - alpha))
        
        # Apply attention mask if provided
        if attention_mask is not None:
            entropy = entropy * attention_mask
            # Normalize by number of valid tokens
            entropy_loss = entropy.sum() / attention_mask.sum()
        else:
            entropy_loss = entropy.mean()
        
        # Return negative entropy as loss (we want to maximize entropy)
        return -entropy_loss
    
    def compute_gating_statistics(
        self,
        gating_coefficients: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute statistics about gating behavior for monitoring.
        
        Args:
            gating_coefficients: Gating coefficients [batch_size, seq_len, 1]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Dictionary of gating statistics
        """
        alpha = gating_coefficients.squeeze(-1)  # [batch_size, seq_len]
        
        with torch.no_grad():
            if attention_mask is not None:
                # Only consider valid (non-padded) positions
                valid_alpha = alpha * attention_mask
                valid_positions = attention_mask.sum()
                
                mean_alpha = valid_alpha.sum() / valid_positions
                mean_lm_contrib = (attention_mask - valid_alpha).sum() / valid_positions
                
                # Compute standard deviation over valid positions
                alpha_var = ((alpha - mean_alpha) ** 2 * attention_mask).sum() / valid_positions
                alpha_std = torch.sqrt(alpha_var)
            else:
                mean_alpha = alpha.mean()
                mean_lm_contrib = (1 - alpha).mean()
                alpha_std = alpha.std()
        
        return {
            'mean_rm_contribution': mean_alpha,
            'mean_lm_contribution': mean_lm_contrib,
            'rm_dominance_ratio': (alpha > 0.5).float().mean(),
            'lm_dominance_ratio': (alpha < 0.5).float().mean(),
            'balanced_positions': ((alpha > 0.4) & (alpha < 0.6)).float().mean(),
            'gating_std': alpha_std,
            'gating_min': alpha.min(),
            'gating_max': alpha.max(),
        }


class ARMLoss(nn.Module):
    """
    Standalone ARM (Autoregressive Reward Modeling) loss for comparison with GenARM.
    
    This implements the preference loss component used in GenARM training.
    """
    
    def __init__(self, beta: float = 1.0, length_normalization: bool = False):
        super().__init__()
        self.beta = beta
        self.length_normalization = length_normalization
    
    def forward(
        self,
        chosen_logps: torch.Tensor,
        rejected_logps: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute ARM loss for a batch of policy model log probabilities.
        
        Args:
            chosen_logps: Log probabilities for chosen responses [batch_size]
            rejected_logps: Log probabilities for rejected responses [batch_size]
            
        Returns:
            Tuple of (losses, chosen_rewards, rejected_rewards)
        """
        # Compute log ratios
        pi_logratios = chosen_logps - rejected_logps
        
        # ARM uses simplified loss without reference model
        logits = pi_logratios
        
        # Compute loss: -log σ(β * logits)
        losses = -F.logsigmoid(self.beta * logits)
        
        # Compute rewards
        chosen_rewards = self.beta * chosen_logps.detach()
        rejected_rewards = self.beta * rejected_logps.detach()
        
        return losses, chosen_rewards, rejected_rewards


class WeightedLoss(nn.Module):
    """
    Utility class for combining multiple loss components with learnable weights.
    
    This can be used for adaptive loss weighting during training.
    """
    
    def __init__(
        self,
        loss_names: list,
        initial_weights: Optional[Dict[str, float]] = None,
        learnable: bool = False
    ):
        super().__init__()
        self.loss_names = loss_names
        
        # Initialize weights
        if initial_weights is None:
            initial_weights = {name: 1.0 for name in loss_names}
        
        if learnable:
            # Learnable loss weights (log-parameterized for numerical stability)
            self.log_weights = nn.ParameterDict({
                name: nn.Parameter(torch.log(torch.tensor(initial_weights.get(name, 1.0))))
                for name in loss_names
            })
        else:
            # Fixed weights
            self.register_buffer('weights', torch.tensor([
                initial_weights.get(name, 1.0) for name in loss_names
            ]))
            self.log_weights = None
    
    def forward(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Combine losses with weighted sum.
        
        Args:
            losses: Dictionary of loss components
            
        Returns:
            Weighted total loss
        """
        if self.log_weights is not None:
            # Learnable weights
            total_loss = sum(
                torch.exp(self.log_weights[name]) * losses[name]
                for name in self.loss_names if name in losses
            )
        else:
            # Fixed weights
            total_loss = sum(
                weight * losses[name]
                for weight, name in zip(self.weights, self.loss_names)
                if name in losses
            )
        
        return total_loss
    
    def get_weights(self) -> Dict[str, float]:
        """Get current loss weights."""
        if self.log_weights is not None:
            return {name: torch.exp(self.log_weights[name]).item() for name in self.loss_names}
        else:
            return {name: weight.item() for name, weight in zip(self.loss_names, self.weights)}