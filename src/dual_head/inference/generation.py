"""
Advanced generation utilities for Dual-Head models.

This module provides specialized generation methods including:
1. Batch generation with dynamic gating
2. Streaming generation for real-time applications
3. Constrained generation with alignment control
4. Multi-objective generation balancing different criteria
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any, Generator
from transformers import PreTrainedTokenizer, StoppingCriteria, StoppingCriteriaList
from dataclasses import dataclass
import json
import time

from ..dual_head_model import DualHeadModel


@dataclass 
class GenerationConfig:
    """Extended generation configuration for Dual-Head models."""
    
    # Standard generation parameters
    max_new_tokens: int = 128
    min_new_tokens: int = 1
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 0
    do_sample: bool = True
    repetition_penalty: float = 1.0
    length_penalty: float = 1.0
    
    # Dual-Head specific parameters
    alignment_weight: Optional[float] = None  # Override gating if provided
    dynamic_alignment: bool = True  # Whether to use adaptive gating
    
    # Advanced generation features
    use_cache: bool = True
    early_stopping: bool = True
    num_beams: int = 1
    num_return_sequences: int = 1
    
    # Streaming and real-time features
    stream_output: bool = False
    stream_interval: int = 1  # Tokens between streaming updates
    
    # Safety and control
    max_length: int = 2048
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    forced_eos_token_id: Optional[int] = None
    
    # Logging and analysis
    return_dict_in_generate: bool = False
    output_attentions: bool = False
    output_hidden_states: bool = False
    output_scores: bool = False


class AlignmentControlledStopping(StoppingCriteria):
    """Stopping criteria that considers alignment behavior."""
    
    def __init__(
        self,
        max_new_tokens: int,
        min_alignment_ratio: float = 0.3,
        max_alignment_ratio: float = 0.8,
        eos_token_id: Optional[int] = None
    ):
        self.max_new_tokens = max_new_tokens
        self.min_alignment_ratio = min_alignment_ratio
        self.max_alignment_ratio = max_alignment_ratio
        self.eos_token_id = eos_token_id
        self.step_count = 0
        self.alignment_history = []
    
    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        **kwargs
    ) -> bool:
        self.step_count += 1
        
        # Check for EOS token
        if self.eos_token_id is not None:
            if input_ids[0, -1].item() == self.eos_token_id:
                return True
        
        # Check max length
        if self.step_count >= self.max_new_tokens:
            return True
        
        # Check alignment behavior (requires gating info from model)
        gating_info = kwargs.get('gating_coefficients')
        if gating_info is not None:
            current_alignment = gating_info.mean().item()
            self.alignment_history.append(current_alignment)
            
            # Stop if alignment is consistently too high or too low
            if len(self.alignment_history) >= 10:
                recent_alignment = sum(self.alignment_history[-10:]) / 10
                if recent_alignment < self.min_alignment_ratio or recent_alignment > self.max_alignment_ratio:
                    return True
        
        return False


class DualHeadGenerator:
    """
    Advanced generator for Dual-Head models with specialized generation methods.
    
    This generator provides:
    1. Batch generation with efficiency optimizations
    2. Streaming generation for real-time applications  
    3. Constrained generation with alignment control
    4. Multi-objective optimization during generation
    """
    
    def __init__(
        self,
        model: DualHeadModel,
        tokenizer: PreTrainedTokenizer,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or next(model.parameters()).device
        
        # Move model to device and set to eval mode
        self.model.to(self.device)
        self.model.eval()
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate sequences using Dual-Head model.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            generation_config: Generation configuration
            stopping_criteria: Custom stopping criteria
            **kwargs: Additional generation parameters
            
        Returns:
            Dictionary containing generated sequences and metadata
        """
        # Set up configuration
        config = generation_config or GenerationConfig()
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        # Set up stopping criteria
        if stopping_criteria is None:
            stopping_criteria = StoppingCriteriaList()
        
        # Add alignment-controlled stopping if requested
        if hasattr(config, 'use_alignment_stopping') and config.use_alignment_stopping:
            alignment_stopping = AlignmentControlledStopping(
                max_new_tokens=config.max_new_tokens,
                eos_token_id=config.eos_token_id
            )
            stopping_criteria.append(alignment_stopping)
        
        # Choose generation method
        if config.num_beams > 1:
            return self._beam_search_generate(input_ids, attention_mask, config, stopping_criteria)
        elif config.stream_output:
            return self._streaming_generate(input_ids, attention_mask, config, stopping_criteria)
        else:
            return self._greedy_or_sample_generate(input_ids, attention_mask, config, stopping_criteria)
    
    def _greedy_or_sample_generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        config: GenerationConfig,
        stopping_criteria: StoppingCriteriaList,
    ) -> Dict[str, Any]:
        """Standard greedy or sampling generation."""
        batch_size, prompt_length = input_ids.shape
        device = input_ids.device
        
        # Initialize generation state
        generated_ids = input_ids.clone()
        past_key_values = None
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        # Storage for analysis
        generation_metadata = {
            'gating_history': [],
            'score_history': [],
            'timing_info': {},
        }
        
        start_time = time.time()
        
        # Generation loop
        for step in range(config.max_new_tokens):
            # Prepare inputs
            if past_key_values is None:
                current_input_ids = generated_ids
                current_attention_mask = attention_mask
            else:
                current_input_ids = generated_ids[:, -1:]
                current_attention_mask = attention_mask
            
            # Forward pass
            outputs = self.model(
                input_ids=current_input_ids,
                attention_mask=current_attention_mask,
                past_key_values=past_key_values,
                use_cache=config.use_cache,
                return_dict=True,
                output_attentions=config.output_attentions,
                output_hidden_states=config.output_hidden_states
            )
            
            # Get logits and update cache
            next_token_logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values if config.use_cache else None
            
            # Store gating information
            if hasattr(outputs, 'gating_coefficients'):
                gating_coeffs = outputs.gating_coefficients[:, -1, :].squeeze(-1)
                generation_metadata['gating_history'].append(gating_coeffs.cpu())
            
            # Apply alignment weight override if specified
            if config.alignment_weight is not None and not config.dynamic_alignment:
                # Manually fuse heads with specified alignment weight
                head_outputs = self.model.dual_heads(outputs.hidden_states, return_dict=True)
                lm_logits = head_outputs['lm_logits'][:, -1, :]
                rm_logits = head_outputs['rm_logits'][:, -1, :]
                
                alpha = config.alignment_weight
                next_token_logits = (1 - alpha) * lm_logits + alpha * rm_logits
            
            # Apply generation controls
            next_token_logits = self._apply_generation_controls(
                next_token_logits, generated_ids, config
            )
            
            # Sample next tokens
            if config.do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)
            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Update sequences
            generated_ids = torch.cat([generated_ids, next_tokens], dim=-1)
            
            # Update attention mask
            if attention_mask is not None:
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones(batch_size, 1, dtype=attention_mask.dtype, device=device)
                ], dim=-1)
            
            # Store scores if requested
            if config.output_scores:
                generation_metadata['score_history'].append(next_token_logits.cpu())
            
            # Check stopping criteria
            should_stop = False
            for criteria in stopping_criteria:
                if criteria(generated_ids, next_token_logits, gating_coefficients=gating_coeffs if 'gating_coeffs' in locals() else None):
                    should_stop = True
                    break
            
            # Check for EOS tokens
            if config.eos_token_id is not None:
                finished = finished | (next_tokens.squeeze(-1) == config.eos_token_id)
            
            if should_stop or finished.all():
                break
        
        # Record timing
        generation_metadata['timing_info']['total_time'] = time.time() - start_time
        generation_metadata['timing_info']['tokens_per_second'] = (
            (generated_ids.shape[1] - prompt_length) / generation_metadata['timing_info']['total_time']
        )
        
        # Prepare output
        result = {
            'sequences': generated_ids,
            'metadata': generation_metadata
        }
        
        # Add additional outputs if requested
        if config.output_attentions and hasattr(outputs, 'attentions'):
            result['attentions'] = outputs.attentions
        
        if config.output_hidden_states and hasattr(outputs, 'hidden_states'):
            result['hidden_states'] = outputs.hidden_states
        
        return result
    
    def _streaming_generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        config: GenerationConfig,
        stopping_criteria: StoppingCriteriaList,
    ) -> Generator[Dict[str, Any], None, None]:
        """Streaming generation that yields partial results."""
        batch_size, prompt_length = input_ids.shape
        device = input_ids.device
        
        # Initialize generation state
        generated_ids = input_ids.clone()
        past_key_values = None
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        # Generation loop with streaming
        for step in range(config.max_new_tokens):
            # Prepare inputs
            if past_key_values is None:
                current_input_ids = generated_ids
                current_attention_mask = attention_mask
            else:
                current_input_ids = generated_ids[:, -1:]
                current_attention_mask = attention_mask
            
            # Forward pass
            outputs = self.model(
                input_ids=current_input_ids,
                attention_mask=current_attention_mask,
                past_key_values=past_key_values,
                use_cache=config.use_cache,
                return_dict=True
            )
            
            # Get next token
            next_token_logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values if config.use_cache else None
            
            # Apply controls and sample
            next_token_logits = self._apply_generation_controls(
                next_token_logits, generated_ids, config
            )
            
            if config.do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)
            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Update sequences
            generated_ids = torch.cat([generated_ids, next_tokens], dim=-1)
            
            # Update attention mask
            if attention_mask is not None:
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones(batch_size, 1, dtype=attention_mask.dtype, device=device)
                ], dim=-1)
            
            # Yield partial results at specified intervals
            if (step + 1) % config.stream_interval == 0:
                # Decode current sequence
                current_text = self.tokenizer.decode(
                    generated_ids[0, prompt_length:], 
                    skip_special_tokens=True
                )
                
                stream_result = {
                    'partial_text': current_text,
                    'step': step + 1,
                    'finished': False,
                    'current_tokens': generated_ids.clone()
                }
                
                # Add gating info if available
                if hasattr(outputs, 'gating_coefficients'):
                    gating_coeffs = outputs.gating_coefficients[:, -1, :].squeeze(-1)
                    stream_result['current_alignment'] = gating_coeffs.mean().item()
                
                yield stream_result
            
            # Check stopping criteria
            should_stop = False
            for criteria in stopping_criteria:
                if criteria(generated_ids, next_token_logits):
                    should_stop = True
                    break
            
            # Check for EOS tokens
            if config.eos_token_id is not None:
                finished = finished | (next_tokens.squeeze(-1) == config.eos_token_id)
            
            if should_stop or finished.all():
                break
        
        # Yield final result
        final_text = self.tokenizer.decode(
            generated_ids[0, prompt_length:], 
            skip_special_tokens=True
        )
        
        yield {
            'partial_text': final_text,
            'step': step + 1,
            'finished': True,
            'final_tokens': generated_ids,
            'total_new_tokens': generated_ids.shape[1] - prompt_length
        }
    
    def _beam_search_generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        config: GenerationConfig,
        stopping_criteria: StoppingCriteriaList,
    ) -> Dict[str, Any]:
        """Beam search generation with Dual-Head scoring."""
        # This is a simplified beam search implementation
        # For production, consider using HuggingFace's beam search
        
        batch_size, prompt_length = input_ids.shape
        num_beams = config.num_beams
        device = input_ids.device
        
        # Expand input for beam search
        expanded_input_ids = input_ids.unsqueeze(1).expand(batch_size, num_beams, -1)
        expanded_input_ids = expanded_input_ids.reshape(batch_size * num_beams, -1)
        
        if attention_mask is not None:
            expanded_attention_mask = attention_mask.unsqueeze(1).expand(batch_size, num_beams, -1)
            expanded_attention_mask = expanded_attention_mask.reshape(batch_size * num_beams, -1)
        else:
            expanded_attention_mask = None
        
        # Initialize beam search state
        beam_scores = torch.zeros(batch_size, num_beams, device=device)
        beam_tokens = expanded_input_ids.clone()
        past_key_values = None
        
        # Beam search loop
        for step in range(config.max_new_tokens):
            # Forward pass
            outputs = self.model(
                input_ids=beam_tokens if past_key_values is None else beam_tokens[:, -1:],
                attention_mask=expanded_attention_mask,
                past_key_values=past_key_values,
                use_cache=config.use_cache,
                return_dict=True
            )
            
            # Get logits and scores
            next_token_logits = outputs.logits[:, -1, :]  # [batch_size * num_beams, vocab_size]
            past_key_values = outputs.past_key_values if config.use_cache else None
            
            # Apply controls
            next_token_logits = self._apply_generation_controls(
                next_token_logits, beam_tokens, config
            )
            
            # Get log probabilities
            next_token_log_probs = F.log_softmax(next_token_logits, dim=-1)
            
            # Reshape for beam operations
            next_token_log_probs = next_token_log_probs.reshape(batch_size, num_beams, -1)
            
            # Add beam scores
            vocab_size = next_token_log_probs.shape[-1]
            next_token_log_probs = next_token_log_probs + beam_scores.unsqueeze(-1)
            
            # Reshape and get top candidates
            next_token_log_probs = next_token_log_probs.reshape(batch_size, num_beams * vocab_size)
            
            # Select top 2*num_beams candidates
            top_log_probs, top_indices = torch.topk(
                next_token_log_probs, 2 * num_beams, dim=-1, sorted=True
            )
            
            # Decode indices to beam and token indices
            beam_indices = top_indices // vocab_size
            token_indices = top_indices % vocab_size
            
            # Update beams
            new_beam_tokens = []
            new_beam_scores = []
            
            for batch_idx in range(batch_size):
                batch_beam_tokens = []
                batch_beam_scores = []
                
                for i in range(num_beams):
                    beam_idx = beam_indices[batch_idx, i]
                    token_idx = token_indices[batch_idx, i]
                    score = top_log_probs[batch_idx, i]
                    
                    # Get original beam
                    original_beam_idx = beam_idx + batch_idx * num_beams
                    original_beam = beam_tokens[original_beam_idx]
                    
                    # Append new token
                    new_beam = torch.cat([original_beam, token_idx.unsqueeze(0)])
                    
                    batch_beam_tokens.append(new_beam)
                    batch_beam_scores.append(score)
                
                new_beam_tokens.extend(batch_beam_tokens)
                new_beam_scores.extend(batch_beam_scores)
            
            # Update state
            beam_tokens = torch.stack(new_beam_tokens)
            beam_scores = torch.tensor(new_beam_scores, device=device).reshape(batch_size, num_beams)
            
            # Update attention mask
            if expanded_attention_mask is not None:
                expanded_attention_mask = torch.cat([
                    expanded_attention_mask,
                    torch.ones(batch_size * num_beams, 1, dtype=expanded_attention_mask.dtype, device=device)
                ], dim=-1)
            
            # Check stopping criteria (simplified)
            if config.eos_token_id is not None:
                # Check if all beams ended
                eos_mask = (beam_tokens[:, -1] == config.eos_token_id)
                if eos_mask.all():
                    break
        
        # Select best beams
        best_beam_indices = beam_scores.argmax(dim=-1)
        best_sequences = []
        
        for batch_idx in range(batch_size):
            best_beam_idx = best_beam_indices[batch_idx] + batch_idx * num_beams
            best_sequences.append(beam_tokens[best_beam_idx])
        
        final_sequences = torch.stack(best_sequences)
        
        return {
            'sequences': final_sequences,
            'beam_scores': beam_scores,
            'metadata': {
                'num_beams': num_beams,
                'best_beam_scores': beam_scores.max(dim=-1)[0]
            }
        }
    
    def _apply_generation_controls(
        self,
        logits: torch.Tensor,
        generated_ids: torch.Tensor,
        config: GenerationConfig
    ) -> torch.Tensor:
        """Apply various generation controls to logits."""
        # Temperature
        if config.temperature != 1.0:
            logits = logits / config.temperature
        
        # Repetition penalty
        if config.repetition_penalty != 1.0:
            logits = self._apply_repetition_penalty(logits, generated_ids, config.repetition_penalty)
        
        # Top-k filtering
        if config.top_k > 0:
            logits = self._apply_top_k_filtering(logits, config.top_k)
        
        # Top-p filtering
        if config.top_p < 1.0:
            logits = self._apply_top_p_filtering(logits, config.top_p)
        
        return logits
    
    def _apply_repetition_penalty(
        self,
        logits: torch.Tensor,
        generated_ids: torch.Tensor,
        penalty: float
    ) -> torch.Tensor:
        """Apply repetition penalty."""
        batch_size = logits.shape[0]
        
        for i in range(batch_size):
            for token in generated_ids[i].unique():
                if logits[i, token] < 0:
                    logits[i, token] *= penalty
                else:
                    logits[i, token] /= penalty
        
        return logits
    
    def _apply_top_k_filtering(self, logits: torch.Tensor, top_k: int) -> torch.Tensor:
        """Apply top-k filtering."""
        top_k = min(top_k, logits.size(-1))
        top_k_logits, _ = torch.topk(logits, top_k)
        indices_to_remove = logits < top_k_logits[..., -1, None]
        logits[indices_to_remove] = -float('inf')
        return logits
    
    def _apply_top_p_filtering(self, logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """Apply top-p filtering."""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = -float('inf')
        
        return logits
    
    def generate_with_alignment_control(
        self,
        prompts: Union[str, List[str]],
        alignment_schedule: Optional[List[float]] = None,
        **generation_kwargs
    ) -> Dict[str, Any]:
        """
        Generate text with explicit alignment control.
        
        Args:
            prompts: Input prompts
            alignment_schedule: List of alignment weights for each generation step
            **generation_kwargs: Additional generation parameters
            
        Returns:
            Generation results with alignment analysis
        """
        if isinstance(prompts, str):
            prompts = [prompts]
        
        # Tokenize prompts
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        
        # Set up generation config
        config = GenerationConfig(**generation_kwargs)
        config.dynamic_alignment = alignment_schedule is None
        
        results = []
        
        for i, prompt in enumerate(prompts):
            # Generate with controlled alignment
            if alignment_schedule is not None:
                # Step-by-step generation with alignment control
                result = self._generate_with_schedule(
                    inputs['input_ids'][i:i+1],
                    inputs.get('attention_mask', None)[i:i+1] if inputs.get('attention_mask') is not None else None,
                    alignment_schedule,
                    config
                )
            else:
                # Standard generation
                result = self.generate(
                    inputs['input_ids'][i:i+1],
                    inputs.get('attention_mask', None)[i:i+1] if inputs.get('attention_mask') is not None else None,
                    config
                )
            
            results.append(result)
        
        return {
            'results': results[0] if len(results) == 1 else results,
            'prompts': prompts[0] if len(prompts) == 1 else prompts
        }
    
    def _generate_with_schedule(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        alignment_schedule: List[float],
        config: GenerationConfig
    ) -> Dict[str, Any]:
        """Generate with a predetermined alignment schedule."""
        batch_size, prompt_length = input_ids.shape
        device = input_ids.device
        
        generated_ids = input_ids.clone()
        past_key_values = None
        alignment_history = []
        
        max_steps = min(len(alignment_schedule), config.max_new_tokens)
        
        for step in range(max_steps):
            # Set alignment weight for this step
            current_alignment_weight = alignment_schedule[step]
            
            # Forward pass
            outputs = self.model(
                input_ids=generated_ids if past_key_values is None else generated_ids[:, -1:],
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=config.use_cache,
                return_dict=True
            )
            
            # Manual alignment control
            head_outputs = self.model.dual_heads(outputs.hidden_states, return_dict=True)
            lm_logits = head_outputs['lm_logits'][:, -1, :]
            rm_logits = head_outputs['rm_logits'][:, -1, :]
            
            # Fuse with scheduled alignment weight
            next_token_logits = (1 - current_alignment_weight) * lm_logits + current_alignment_weight * rm_logits
            
            # Apply controls and sample
            next_token_logits = self._apply_generation_controls(
                next_token_logits, generated_ids, config
            )
            
            if config.do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)
            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Update sequences
            generated_ids = torch.cat([generated_ids, next_tokens], dim=-1)
            past_key_values = outputs.past_key_values if config.use_cache else None
            
            # Record alignment weight
            alignment_history.append(current_alignment_weight)
            
            # Update attention mask
            if attention_mask is not None:
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones(batch_size, 1, dtype=attention_mask.dtype, device=device)
                ], dim=-1)
            
            # Check for EOS
            if config.eos_token_id is not None and next_tokens.squeeze(-1).item() == config.eos_token_id:
                break
        
        return {
            'sequences': generated_ids,
            'alignment_schedule': alignment_schedule[:len(alignment_history)],
            'alignment_history': alignment_history,
            'metadata': {
                'controlled_generation': True,
                'steps_taken': len(alignment_history)
            }
        }