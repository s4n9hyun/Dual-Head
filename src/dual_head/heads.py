"""
Implementation of Language Modeling and Reward Modeling heads for the Dual-Head architecture.

These compact heads (131M parameters each) are attached to the frozen backbone to provide
dual capabilities for language modeling and reward estimation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from transformers.activations import ACT2FN


class LanguageModelingHead(nn.Module):
    """
    Language modeling head that produces standard next-token logits.
    
    This head maintains the original language modeling capability of the base model
    while being compact (131M parameters for LLaMA-7B with vocab_size=32000).
    """
    
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        bias: bool = False,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        
        # Linear projection from hidden states to vocabulary logits
        self.lm_head = nn.Linear(
            hidden_size, 
            vocab_size, 
            bias=bias,
            dtype=dtype
        )
        
        # Initialize weights similar to standard LM heads
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with proper scaling for stable training."""
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)
        if self.lm_head.bias is not None:
            nn.init.zeros_(self.lm_head.bias)
    
    def forward(
        self, 
        hidden_states: torch.Tensor,
        return_logits: bool = True
    ) -> torch.Tensor:
        """
        Forward pass through the language modeling head.
        
        Args:
            hidden_states: Hidden representations from backbone [batch_size, seq_len, hidden_size]
            return_logits: Whether to return raw logits or log probabilities
            
        Returns:
            Tensor of shape [batch_size, seq_len, vocab_size] containing logits or log probs
        """
        # Project to vocabulary space
        logits = self.lm_head(hidden_states)  # [batch_size, seq_len, vocab_size]
        
        if return_logits:
            return logits
        else:
            return F.log_softmax(logits, dim=-1)


class RewardModelingHead(nn.Module):
    """
    Reward modeling head that assigns alignment-oriented scores to tokens.
    
    This head learns to predict token-level rewards that aggregate to trajectory-level
    preferences, following the autoregressive reward modeling approach from GenARM.
    """
    
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        bias: bool = False,
        dtype: Optional[torch.dtype] = None,
        intermediate_size: Optional[int] = None,
        activation: str = "silu",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.intermediate_size = intermediate_size or hidden_size
        
        # Optional intermediate layer for more expressivity
        if intermediate_size and intermediate_size != hidden_size:
            self.intermediate_dense = nn.Linear(
                hidden_size, 
                intermediate_size, 
                bias=bias,
                dtype=dtype
            )
            self.activation = ACT2FN[activation]
            self.dropout = nn.Dropout(dropout)
            
            # Final projection layer
            self.rm_head = nn.Linear(
                intermediate_size, 
                vocab_size, 
                bias=bias,
                dtype=dtype
            )
        else:
            self.intermediate_dense = None
            # Direct projection to vocabulary space
            self.rm_head = nn.Linear(
                hidden_size, 
                vocab_size, 
                bias=bias,
                dtype=dtype
            )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with proper scaling for stable training."""
        if self.intermediate_dense is not None:
            nn.init.normal_(self.intermediate_dense.weight, mean=0.0, std=0.02)
            if self.intermediate_dense.bias is not None:
                nn.init.zeros_(self.intermediate_dense.bias)
        
        nn.init.normal_(self.rm_head.weight, mean=0.0, std=0.02)
        if self.rm_head.bias is not None:
            nn.init.zeros_(self.rm_head.bias)
    
    def forward(
        self, 
        hidden_states: torch.Tensor,
        return_logits: bool = True,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Forward pass through the reward modeling head.
        
        Args:
            hidden_states: Hidden representations from backbone [batch_size, seq_len, hidden_size]
            return_logits: Whether to return raw logits or probabilities
            temperature: Temperature for softmax (used in reward computation)
            
        Returns:
            Tensor of shape [batch_size, seq_len, vocab_size] containing reward logits or probs
        """
        # Optional intermediate transformation
        if self.intermediate_dense is not None:
            hidden_states = self.intermediate_dense(hidden_states)
            hidden_states = self.activation(hidden_states)
            hidden_states = self.dropout(hidden_states)
        
        # Project to vocabulary space for token-level rewards
        logits = self.rm_head(hidden_states)  # [batch_size, seq_len, vocab_size]
        
        if return_logits:
            return logits
        else:
            # Return log probabilities scaled by temperature
            return F.log_softmax(logits / temperature, dim=-1)
    
    def compute_sequence_reward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Compute sequence-level reward by aggregating token-level rewards.
        
        This implements the autoregressive reward modeling approach where:
        R(x, Y) = sum_t log π_r(y_t | x, y_{<t})
        
        Args:
            hidden_states: Hidden representations [batch_size, seq_len, hidden_size]
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            temperature: Temperature parameter β_r
            
        Returns:
            Sequence rewards [batch_size]
        """
        # Get reward probabilities
        log_probs = self.forward(
            hidden_states, 
            return_logits=False, 
            temperature=temperature
        )  # [batch_size, seq_len, vocab_size]
        
        # Gather probabilities for actual tokens
        # Shift input_ids to align with predictions (ignore first token for reward)
        target_ids = input_ids[:, 1:]  # [batch_size, seq_len-1]
        log_probs = log_probs[:, :-1, :]  # [batch_size, seq_len-1, vocab_size]
        
        # Gather log probabilities for target tokens
        token_rewards = torch.gather(
            log_probs, 
            dim=-1, 
            index=target_ids.unsqueeze(-1)
        ).squeeze(-1)  # [batch_size, seq_len-1]
        
        # Apply attention mask if provided (ignore padded tokens)
        if attention_mask is not None:
            mask = attention_mask[:, 1:]  # [batch_size, seq_len-1]
            token_rewards = token_rewards * mask
        
        # Sum token-level rewards to get sequence-level reward
        sequence_rewards = token_rewards.sum(dim=-1)  # [batch_size]
        
        return sequence_rewards


class DualHead(nn.Module):
    """
    Combined dual-head module containing both LM and RM heads.
    
    This module manages both heads and provides a unified interface
    for accessing their outputs during training and inference.
    """
    
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        lm_bias: bool = False,
        rm_bias: bool = False,
        rm_intermediate_size: Optional[int] = None,
        rm_activation: str = "silu",
        rm_dropout: float = 0.1,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        
        # Language modeling head
        self.lm_head = LanguageModelingHead(
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            bias=lm_bias,
            dtype=dtype
        )
        
        # Reward modeling head
        self.rm_head = RewardModelingHead(
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            bias=rm_bias,
            intermediate_size=rm_intermediate_size,
            activation=rm_activation,
            dropout=rm_dropout,
            dtype=dtype
        )
        
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        return_dict: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through both heads.
        
        Args:
            hidden_states: Hidden representations from backbone [batch_size, seq_len, hidden_size]
            return_dict: Whether to return a dictionary or tuple
            
        Returns:
            Tuple of (lm_logits, rm_logits) or dictionary with keys 'lm_logits', 'rm_logits'
        """
        # Get outputs from both heads
        lm_logits = self.lm_head(hidden_states, return_logits=True)
        rm_logits = self.rm_head(hidden_states, return_logits=True)
        
        if return_dict:
            return {
                'lm_logits': lm_logits,
                'rm_logits': rm_logits
            }
        else:
            return lm_logits, rm_logits
    
    def get_parameter_count(self) -> dict:
        """Get parameter counts for each head."""
        lm_params = sum(p.numel() for p in self.lm_head.parameters())
        rm_params = sum(p.numel() for p in self.rm_head.parameters())
        total_params = lm_params + rm_params
        
        return {
            'lm_head': lm_params,
            'rm_head': rm_params,  
            'total': total_params,
            'lm_head_mb': lm_params * 4 / (1024**2),  # Assuming fp32
            'rm_head_mb': rm_params * 4 / (1024**2),
            'total_mb': total_params * 4 / (1024**2)
        }