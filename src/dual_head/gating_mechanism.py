"""
Context-aware gating mechanism for the Dual-Head architecture.

This module implements the attention-based gating network that dynamically
balances the contributions of language modeling and reward modeling heads
based on the current sequence context.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from transformers.activations import ACT2FN


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism for context-aware gating.
    
    This attention module processes the sequence history to compute
    context-dependent gating weights that determine the fusion of
    LM and RM head outputs.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        bias: bool = True,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        
        if self.head_dim * num_attention_heads != hidden_size:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by "
                f"num_attention_heads ({num_attention_heads})"
            )
        
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Query, Key, Value projections
        self.query = nn.Linear(hidden_size, hidden_size, bias=bias, dtype=dtype)
        self.key = nn.Linear(hidden_size, hidden_size, bias=bias, dtype=dtype)
        self.value = nn.Linear(hidden_size, hidden_size, bias=bias, dtype=dtype)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias, dtype=dtype)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with proper scaling."""
        for module in [self.query, self.key, self.value, self.out_proj]:
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through multi-head attention.
        
        Args:
            query: Query tensor [batch_size, seq_len, hidden_size]
            key: Key tensor [batch_size, seq_len, hidden_size]  
            value: Value tensor [batch_size, seq_len, hidden_size]
            attention_mask: Mask to prevent attention to certain positions
            return_attention_weights: Whether to return attention weights
            
        Returns:
            Tuple of (attended_output, attention_weights)
        """
        batch_size, seq_len, _ = query.shape
        
        # Linear projections and reshape for multi-head attention
        Q = self.query(query).view(batch_size, seq_len, self.num_attention_heads, self.head_dim)
        K = self.key(key).view(batch_size, seq_len, self.num_attention_heads, self.head_dim)
        V = self.value(value).view(batch_size, seq_len, self.num_attention_heads, self.head_dim)
        
        # Transpose for attention computation [batch_size, num_heads, seq_len, head_dim]
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        # Softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended_values = torch.matmul(attention_weights, V)
        
        # Reshape and project output
        attended_values = attended_values.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.hidden_size
        )
        output = self.out_proj(attended_values)
        
        if return_attention_weights:
            return output, attention_weights
        else:
            return output, None


class ContextAwareGating(nn.Module):
    """
    Context-aware gating mechanism that dynamically computes fusion weights.
    
    This module uses multi-head attention over the sequence history to compute
    context-dependent gating coefficients α_t that balance LM and RM head outputs:
    
    z_t = (1 - α_t) * z_LM,t + α_t * z_RM,t
    
    where α_t = σ(W_g · MultiHeadAttention(h_t, H_{1:t}, H_{1:t}) + b_g)
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int = 8,
        intermediate_size: Optional[int] = None,
        activation: str = "gelu",
        dropout: float = 0.1,
        bias: bool = True,
        use_layer_norm: bool = True,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size or hidden_size // 4
        
        # Multi-head attention for context processing
        self.attention = MultiHeadAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
            bias=bias,
            dtype=dtype
        )
        
        # Optional layer normalization
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(hidden_size, dtype=dtype)
        else:
            self.layer_norm = None
        
        # Feed-forward network for gating computation
        self.gate_ffn = nn.Sequential(
            nn.Linear(hidden_size, self.intermediate_size, bias=bias, dtype=dtype),
            ACT2FN[activation],
            nn.Dropout(dropout),
            nn.Linear(self.intermediate_size, 1, bias=bias, dtype=dtype),  # Single gating coefficient
            nn.Sigmoid()  # Ensure α_t ∈ [0,1]
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with proper scaling."""
        for module in self.gate_ffn:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    # Initialize bias to produce α ≈ 0.5 initially for balanced heads
                    if module.out_features == 1:  # Final layer
                        nn.init.zeros_(module.bias)
                    else:
                        nn.init.zeros_(module.bias)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute context-aware gating coefficients.
        
        Args:
            hidden_states: Hidden states from backbone [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]
            return_attention_weights: Whether to return attention weights
            
        Returns:
            Tuple of (gating_coefficients, attention_weights)
            gating_coefficients: [batch_size, seq_len, 1] with values in [0,1]
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Prepare attention mask for multi-head attention
        if attention_mask is not None:
            # Convert to attention mask format (large negative values for masked positions)
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)  # [batch_size, 1, 1, seq_len]
            attention_mask = (1.0 - attention_mask) * -10000.0
        
        # Apply self-attention to capture sequence context
        attended_states, attention_weights = self.attention(
            query=hidden_states,
            key=hidden_states,
            value=hidden_states,
            attention_mask=attention_mask,
            return_attention_weights=return_attention_weights
        )
        
        # Optional layer normalization
        if self.layer_norm is not None:
            attended_states = self.layer_norm(attended_states)
        
        # Compute gating coefficients through feed-forward network
        gating_coefficients = self.gate_ffn(attended_states)  # [batch_size, seq_len, 1]
        
        return gating_coefficients, attention_weights
    
    def compute_entropy_regularization(
        self,
        gating_coefficients: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute entropy regularization loss to prevent gating collapse.
        
        L_α = -λ_G * E_t[α_t * log(α_t) + (1-α_t) * log(1-α_t)]
        
        Args:
            gating_coefficients: Gating coefficients [batch_size, seq_len, 1]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Entropy regularization loss (scalar)
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
        
        return entropy_loss


class AdaptiveFusion(nn.Module):
    """
    Adaptive fusion module that combines LM and RM head outputs using gating coefficients.
    
    This module implements the core fusion operation:
    z_t = (1 - α_t) * z_LM,t + α_t * z_RM,t
    """
    
    def __init__(self, learnable_temperature: bool = False, initial_temperature: float = 1.0):
        super().__init__()
        self.learnable_temperature = learnable_temperature
        
        if learnable_temperature:
            self.temperature = nn.Parameter(torch.tensor(initial_temperature))
        else:
            self.register_buffer('temperature', torch.tensor(initial_temperature))
    
    def forward(
        self,
        lm_logits: torch.Tensor,
        rm_logits: torch.Tensor,
        gating_coefficients: torch.Tensor,
        temperature: Optional[float] = None
    ) -> torch.Tensor:
        """
        Fuse LM and RM logits using gating coefficients.
        
        Args:
            lm_logits: Language model logits [batch_size, seq_len, vocab_size]
            rm_logits: Reward model logits [batch_size, seq_len, vocab_size]  
            gating_coefficients: Gating coefficients [batch_size, seq_len, 1]
            temperature: Optional temperature override
            
        Returns:
            Fused logits [batch_size, seq_len, vocab_size]
        """
        # Use provided temperature or default
        temp = temperature if temperature is not None else self.temperature
        
        # Apply temperature scaling
        lm_logits_scaled = lm_logits / temp
        rm_logits_scaled = rm_logits / temp
        
        # Adaptive fusion: z_t = (1 - α_t) * z_LM,t + α_t * z_RM,t
        fused_logits = (1 - gating_coefficients) * lm_logits_scaled + gating_coefficients * rm_logits_scaled
        
        return fused_logits
    
    def get_head_contributions(
        self,
        gating_coefficients: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> dict:
        """
        Analyze the contribution of each head across the sequence.
        
        Args:
            gating_coefficients: Gating coefficients [batch_size, seq_len, 1]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Dictionary with statistics about head contributions
        """
        alpha = gating_coefficients.squeeze(-1)  # [batch_size, seq_len]
        
        if attention_mask is not None:
            # Only consider valid (non-padded) positions
            valid_alpha = alpha * attention_mask
            valid_positions = attention_mask.sum()
            
            mean_alpha = valid_alpha.sum() / valid_positions
            mean_lm_contrib = (1 - valid_alpha).sum() / valid_positions
        else:
            mean_alpha = alpha.mean()
            mean_lm_contrib = (1 - alpha).mean()
        
        return {
            'mean_rm_contribution': mean_alpha.item(),
            'mean_lm_contribution': mean_lm_contrib.item(),
            'rm_dominance_ratio': (alpha > 0.5).float().mean().item(),
            'gating_std': alpha.std().item(),
            'balanced_positions': ((alpha > 0.4) & (alpha < 0.6)).float().mean().item()
        }