"""
Main Dual-Head model implementation.

This module implements the complete Dual-Head architecture with:
- Frozen backbone for shared representations
- Compact dual heads (LM + RM)
- Context-aware gating mechanism
- Efficient single-pass inference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple, Union
from transformers import (
    AutoModel, 
    AutoModelForCausalLM,
    AutoConfig,
    PreTrainedModel,
    PretrainedConfig,
    GenerationConfig
)
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from dataclasses import dataclass

from .heads import DualHead
from .gating_mechanism import ContextAwareGating, AdaptiveFusion


@dataclass
class DualHeadOutputWithPast(CausalLMOutputWithPast):
    """
    Output class for Dual-Head model with additional fields.
    """
    gating_coefficients: Optional[torch.Tensor] = None
    lm_logits: Optional[torch.Tensor] = None
    rm_logits: Optional[torch.Tensor] = None


class DualHeadConfig(PretrainedConfig):
    """Configuration class for Dual-Head model."""
    
    def __init__(
        self,
        # Backbone configuration
        backbone_name_or_path: str = "meta-llama/Llama-2-7b-hf",
        freeze_backbone: bool = True,
        
        # Head configuration
        lm_bias: bool = False,
        rm_bias: bool = False,
        rm_intermediate_size: Optional[int] = None,
        rm_activation: str = "silu",
        rm_dropout: float = 0.1,
        
        # Gating configuration
        gating_num_heads: int = 8,
        gating_intermediate_size: Optional[int] = None,
        gating_activation: str = "gelu",
        gating_dropout: float = 0.1,
        gating_use_layer_norm: bool = True,
        
        # Fusion configuration
        learnable_temperature: bool = False,
        initial_temperature: float = 1.0,
        
        # Training configuration
        lambda_r: float = 1.0,  # Preference loss weight
        lambda_g: float = 0.01,  # Gating regularization weight
        beta_r: float = 1.0,     # Reward model temperature
        
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.backbone_name_or_path = backbone_name_or_path
        self.freeze_backbone = freeze_backbone
        
        self.lm_bias = lm_bias
        self.rm_bias = rm_bias
        self.rm_intermediate_size = rm_intermediate_size
        self.rm_activation = rm_activation
        self.rm_dropout = rm_dropout
        
        self.gating_num_heads = gating_num_heads
        self.gating_intermediate_size = gating_intermediate_size
        self.gating_activation = gating_activation
        self.gating_dropout = gating_dropout
        self.gating_use_layer_norm = gating_use_layer_norm
        
        self.learnable_temperature = learnable_temperature
        self.initial_temperature = initial_temperature
        
        self.lambda_r = lambda_r
        self.lambda_g = lambda_g
        self.beta_r = beta_r
        
        # Store any additional kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)


class DualHeadModel(PreTrainedModel):
    """
    Dual-Head model for efficient test-time alignment.
    
    This model implements the complete Dual-Head architecture with:
    1. Frozen backbone providing shared contextual representations
    2. Compact dual heads (LM + RM) with 131M parameters each
    3. Context-aware gating mechanism for dynamic fusion
    4. Single forward pass inference efficiency
    """
    
    config_class = DualHeadConfig
    
    def __init__(self, config: DualHeadConfig):
        super().__init__(config)
        self.config = config
        
        # Load and configure backbone model
        self.backbone = self._load_backbone()
        
        # Get backbone configuration
        backbone_config = self.backbone.config
        self.hidden_size = backbone_config.hidden_size
        self.vocab_size = backbone_config.vocab_size
        
        # Get backbone dtype
        backbone_dtype = next(self.backbone.parameters()).dtype
        
        # Initialize dual heads with same dtype as backbone
        self.dual_heads = DualHead(
            hidden_size=self.hidden_size,
            vocab_size=self.vocab_size,
            lm_bias=config.lm_bias,
            rm_bias=config.rm_bias,
            rm_intermediate_size=config.rm_intermediate_size,
            rm_activation=config.rm_activation,
            rm_dropout=config.rm_dropout,
            dtype=backbone_dtype,
        )
        
        # Initialize context-aware gating
        self.gating = ContextAwareGating(
            hidden_size=self.hidden_size,
            num_attention_heads=config.gating_num_heads,
            intermediate_size=config.gating_intermediate_size,
            activation=config.gating_activation,
            dropout=config.gating_dropout,
            bias=True,
            use_layer_norm=config.gating_use_layer_norm,
            dtype=backbone_dtype,
        )
        
        # Initialize adaptive fusion
        self.fusion = AdaptiveFusion(
            learnable_temperature=config.learnable_temperature,
            initial_temperature=config.initial_temperature
        )
        
        # Training configuration
        self.lambda_r = config.lambda_r
        self.lambda_g = config.lambda_g
        self.beta_r = config.beta_r
        
        # Initialize weights
        self.post_init()
    
    def _load_backbone(self) -> PreTrainedModel:
        """Load and configure the backbone model."""
        backbone = AutoModelForCausalLM.from_pretrained(
            self.config.backbone_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map=None,  # We handle device placement manually
            low_cpu_mem_usage=True,  # Optimize memory usage during loading
        )
        
        # Freeze backbone parameters if specified
        if self.config.freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False
            backbone.eval()  # Set to eval mode
        
        return backbone
    
    def get_backbone_hidden_states(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple] = None,
        use_cache: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[Tuple]]:
        """
        Extract hidden states from the frozen backbone model.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            position_ids: Position IDs [batch_size, seq_len]
            past_key_values: Cached key-value pairs for efficient generation
            use_cache: Whether to use/return cache
            
        Returns:
            Tuple of (hidden_states, past_key_values)
        """
        with torch.set_grad_enabled(not self.config.freeze_backbone):
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_hidden_states=True,
                return_dict=True,
                **kwargs
            )
        
        # Get last hidden states
        hidden_states = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_size]
        past_key_values = outputs.past_key_values if use_cache else None
        
        return hidden_states, past_key_values
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        return_dict: bool = True,
        **kwargs
    ) -> Union[Tuple, DualHeadOutputWithPast]:
        """
        Forward pass through the Dual-Head model.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            position_ids: Position IDs [batch_size, seq_len]
            past_key_values: Cached key-value pairs
            labels: Target labels for loss computation [batch_size, seq_len]
            use_cache: Whether to use/return cache
            output_attentions: Whether to return attention weights
            return_dict: Whether to return dict or tuple
            
        Returns:
            Model outputs including logits, loss, and optional attention weights
        """
        # Get hidden states from backbone
        hidden_states, past_key_values = self.get_backbone_hidden_states(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs
        )
        
        # Get outputs from dual heads
        head_outputs = self.dual_heads(hidden_states, return_dict=True)
        lm_logits = head_outputs['lm_logits']  # [batch_size, seq_len, vocab_size]
        rm_logits = head_outputs['rm_logits']  # [batch_size, seq_len, vocab_size]
        
        # Compute context-aware gating coefficients
        gating_coefficients, attention_weights = self.gating(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            return_attention_weights=output_attentions
        )
        
        # Fuse logits using adaptive fusion
        fused_logits = self.fusion(
            lm_logits=lm_logits,
            rm_logits=rm_logits,
            gating_coefficients=gating_coefficients
        )
        
        # Compute loss if labels are provided
        loss = None
        if labels is not None:
            loss = self.compute_loss(
                fused_logits=fused_logits,
                lm_logits=lm_logits,
                rm_logits=rm_logits,
                gating_coefficients=gating_coefficients,
                labels=labels,
                attention_mask=attention_mask,
                input_ids=input_ids
            )
        
        if not return_dict:
            output = (fused_logits,)
            if past_key_values is not None:
                output += (past_key_values,)
            if attention_weights is not None:
                output += (attention_weights,)
            return ((loss,) + output) if loss is not None else output
        
        return DualHeadOutputWithPast(
            loss=loss,
            logits=fused_logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states,
            attentions=attention_weights if output_attentions else None,
            gating_coefficients=gating_coefficients,
            lm_logits=lm_logits,
            rm_logits=rm_logits,
        )
    
    def compute_loss(
        self,
        fused_logits: torch.Tensor,
        lm_logits: torch.Tensor,
        rm_logits: torch.Tensor,
        gating_coefficients: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute the multi-objective loss function.
        
        L_total = L_LM + λ_R * L_pref + λ_G * L_α
        
        Args:
            fused_logits: Fused model logits [batch_size, seq_len, vocab_size]
            lm_logits: LM head logits [batch_size, seq_len, vocab_size]
            rm_logits: RM head logits [batch_size, seq_len, vocab_size]
            gating_coefficients: Gating coefficients [batch_size, seq_len, 1]
            labels: Target labels [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            input_ids: Input token IDs [batch_size, seq_len]
            
        Returns:
            Total loss (scalar)
        """
        # 1. Language Modeling Loss
        lm_loss = self.compute_language_modeling_loss(
            logits=fused_logits,
            labels=labels,
            attention_mask=attention_mask
        )
        
        # 2. Gating Regularization Loss
        gating_loss = self.gating.compute_entropy_regularization(
            gating_coefficients=gating_coefficients,
            attention_mask=attention_mask
        )
        
        # 3. Total Loss (preference loss handled separately during training)
        total_loss = lm_loss + self.lambda_g * gating_loss
        
        return total_loss
    
    def compute_language_modeling_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute standard cross-entropy language modeling loss."""
        # Shift logits and labels for causal LM
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Flatten for loss computation
        shift_logits = shift_logits.view(-1, self.vocab_size)
        shift_labels = shift_labels.view(-1)
        
        # Compute cross-entropy loss (ignores -100 labels)
        loss = F.cross_entropy(shift_logits, shift_labels, reduction='mean')
        
        return loss
    
    def compute_preference_loss(
        self,
        chosen_input_ids: torch.Tensor,
        rejected_input_ids: torch.Tensor,
        chosen_attention_mask: Optional[torch.Tensor] = None,
        rejected_attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute preference loss for ARM training.
        
        This implements the autoregressive reward modeling loss:
        L_pref = -E log σ(β_r * (R_chosen - R_rejected))
        
        Args:
            chosen_input_ids: Chosen response token IDs [batch_size, seq_len]
            rejected_input_ids: Rejected response token IDs [batch_size, seq_len]
            chosen_attention_mask: Attention mask for chosen [batch_size, seq_len]
            rejected_attention_mask: Attention mask for rejected [batch_size, seq_len]
            
        Returns:
            Tuple of (preference_loss, metrics_dict)
        """
        # Get hidden states for both sequences
        chosen_hidden, _ = self.get_backbone_hidden_states(
            input_ids=chosen_input_ids,
            attention_mask=chosen_attention_mask
        )
        
        rejected_hidden, _ = self.get_backbone_hidden_states(
            input_ids=rejected_input_ids,
            attention_mask=rejected_attention_mask
        )
        
        # Compute sequence rewards using RM head
        chosen_rewards = self.dual_heads.rm_head.compute_sequence_reward(
            hidden_states=chosen_hidden,
            input_ids=chosen_input_ids,
            attention_mask=chosen_attention_mask,
            temperature=self.beta_r
        )
        
        rejected_rewards = self.dual_heads.rm_head.compute_sequence_reward(
            hidden_states=rejected_hidden,
            input_ids=rejected_input_ids,
            attention_mask=rejected_attention_mask,
            temperature=self.beta_r
        )
        
        # Compute preference loss: -log σ(β_r * (R_chosen - R_rejected))
        reward_diff = chosen_rewards - rejected_rewards
        pref_loss = -F.logsigmoid(self.beta_r * reward_diff).mean()
        
        # Compute metrics
        with torch.no_grad():
            accuracy = (chosen_rewards > rejected_rewards).float().mean()
            reward_margin = reward_diff.mean()
        
        metrics = {
            'chosen_rewards': chosen_rewards.mean(),
            'rejected_rewards': rejected_rewards.mean(),
            'reward_accuracy': accuracy,
            'reward_margin': reward_margin,
        }
        
        return pref_loss, metrics
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 0,
        do_sample: bool = True,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate text using the Dual-Head model.
        
        This method implements efficient autoregressive generation with
        single forward pass per token using the fused logits.
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Set default token IDs
        if pad_token_id is None:
            pad_token_id = self.backbone.config.pad_token_id or self.backbone.config.eos_token_id
        if eos_token_id is None:
            eos_token_id = self.backbone.config.eos_token_id
        
        # Initialize generation
        generated_ids = input_ids.clone()
        past_key_values = None
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for _ in range(max_new_tokens):
            # Forward pass
            outputs = self.forward(
                input_ids=generated_ids if past_key_values is None else generated_ids[:, -1:],
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True
            )
            
            # Get next token logits
            next_token_logits = outputs.logits[:, -1, :]  # [batch_size, vocab_size]
            past_key_values = outputs.past_key_values
            
            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float('inf')
            
            # Apply top-p filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # Scatter indices to original positions
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = -float('inf')
            
            # Sample next token
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)
            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Update generated sequence
            generated_ids = torch.cat([generated_ids, next_tokens], dim=-1)
            
            # Update attention mask
            if attention_mask is not None:
                attention_mask = torch.cat([
                    attention_mask, 
                    torch.ones(batch_size, 1, dtype=attention_mask.dtype, device=device)
                ], dim=-1)
            
            # Check for EOS tokens
            finished = finished | (next_tokens.squeeze(-1) == eos_token_id)
            if finished.all():
                break
        
        return generated_ids
    
    def get_parameter_statistics(self) -> Dict[str, Any]:
        """Get detailed parameter statistics for the model."""
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        backbone_trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        
        head_stats = self.dual_heads.get_parameter_count()
        gating_params = sum(p.numel() for p in self.gating.parameters())
        fusion_params = sum(p.numel() for p in self.fusion.parameters())
        
        total_params = backbone_params + head_stats['total'] + gating_params + fusion_params
        total_trainable = backbone_trainable + head_stats['total'] + gating_params + fusion_params
        
        return {
            'backbone_params': backbone_params,
            'backbone_trainable': backbone_trainable,
            'lm_head_params': head_stats['lm_head'],
            'rm_head_params': head_stats['rm_head'],
            'dual_heads_params': head_stats['total'],
            'gating_params': gating_params,
            'fusion_params': fusion_params,
            'total_params': total_params,
            'total_trainable': total_trainable,
            'trainable_percentage': (total_trainable / total_params) * 100,
            'parameter_efficiency': backbone_params / head_stats['total'],  # Backbone/Heads ratio
        }
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        config: Optional[DualHeadConfig] = None,
        **kwargs
    ):
        """Load a pre-trained Dual-Head model."""
        if config is None:
            config = DualHeadConfig()
        
        # Update config with any provided kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        model = cls(config)
        
        # Load weights if available
        try:
            state_dict = torch.load(f"{pretrained_model_name_or_path}/pytorch_model.bin")
            model.load_state_dict(state_dict, strict=False)
        except FileNotFoundError:
            print(f"No saved weights found at {pretrained_model_name_or_path}")
        
        return model