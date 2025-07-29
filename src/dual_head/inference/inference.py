"""
Efficient inference system for Dual-Head models.

This module provides optimized inference capabilities with:
1. Single forward pass per token
2. KV-cache optimization
3. Batch processing support
4. Dynamic gating analysis
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any
from transformers import PreTrainedTokenizer
from dataclasses import dataclass
import time

from ..dual_head_model import DualHeadModel


@dataclass
class InferenceConfig:
    """Configuration for Dual-Head inference."""
    
    # Generation parameters
    max_new_tokens: int = 128
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 0
    do_sample: bool = True
    repetition_penalty: float = 1.0
    
    # Efficiency settings
    use_cache: bool = True
    batch_size: int = 1
    
    # Analysis settings
    return_gating_analysis: bool = False
    return_timing_info: bool = False
    
    # Safety settings
    max_length: int = 2048
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None


class DualHeadInference:
    """
    High-level inference interface for Dual-Head models.
    
    This class provides an easy-to-use interface for text generation
    with automatic handling of tokenization, generation, and analysis.
    """
    
    def __init__(
        self,
        model: DualHeadModel,
        tokenizer: PreTrainedTokenizer,
        config: Optional[InferenceConfig] = None,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or InferenceConfig()
        self.device = device or (torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu'))
        
        # Move model to device
        self.model.to(self.device)
        self.model.eval()
        
        # Set default token IDs
        if self.config.pad_token_id is None:
            self.config.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        if self.config.eos_token_id is None:
            self.config.eos_token_id = tokenizer.eos_token_id
    
    @torch.no_grad()
    def generate_text(
        self,
        prompts: Union[str, List[str]],
        **generation_kwargs
    ) -> Union[str, List[str], Dict[str, Any]]:
        """
        Generate text from prompts with optional analysis.
        
        Args:
            prompts: Single prompt or list of prompts
            **generation_kwargs: Override default generation parameters
            
        Returns:
            Generated texts or detailed results dictionary
        """
        # Handle single prompt
        if isinstance(prompts, str):
            prompts = [prompts]
            single_prompt = True
        else:
            single_prompt = False
        
        # Update config with any provided kwargs
        config = self._update_config(generation_kwargs)
        
        # Tokenize prompts
        start_time = time.time()
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config.max_length - config.max_new_tokens
        ).to(self.device)
        
        tokenization_time = time.time() - start_time
        
        # Generate
        start_time = time.time()
        results = self._generate_batch(inputs, config)
        generation_time = time.time() - start_time
        
        # Decode generated sequences
        start_time = time.time()
        generated_texts = []
        for i, generated_ids in enumerate(results['generated_ids']):
            # Remove prompt tokens
            prompt_length = inputs['input_ids'][i].shape[0]
            new_tokens = generated_ids[prompt_length:]
            
            # Decode
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            generated_texts.append(text)
        
        decoding_time = time.time() - start_time
        
        # Prepare output
        if single_prompt:
            generated_texts = generated_texts[0]
        
        # Return detailed results if requested
        if config.return_gating_analysis or config.return_timing_info:
            output = {
                'generated_texts': generated_texts,
            }
            
            if config.return_gating_analysis and 'gating_analysis' in results:
                output['gating_analysis'] = results['gating_analysis'][0] if single_prompt else results['gating_analysis']
            
            if config.return_timing_info:
                output['timing_info'] = {
                    'tokenization_time': tokenization_time,
                    'generation_time': generation_time,
                    'decoding_time': decoding_time,
                    'total_time': tokenization_time + generation_time + decoding_time,
                    'tokens_per_second': len(results['generated_ids'][0]) / generation_time if generation_time > 0 else 0
                }
            
            return output
        else:
            return generated_texts
    
    def _update_config(self, kwargs: Dict[str, Any]) -> InferenceConfig:
        """Update configuration with provided kwargs."""
        config = InferenceConfig(
            max_new_tokens=kwargs.get('max_new_tokens', self.config.max_new_tokens),
            temperature=kwargs.get('temperature', self.config.temperature),
            top_p=kwargs.get('top_p', self.config.top_p),
            top_k=kwargs.get('top_k', self.config.top_k),
            do_sample=kwargs.get('do_sample', self.config.do_sample),
            repetition_penalty=kwargs.get('repetition_penalty', self.config.repetition_penalty),
            use_cache=kwargs.get('use_cache', self.config.use_cache),
            batch_size=kwargs.get('batch_size', self.config.batch_size),
            return_gating_analysis=kwargs.get('return_gating_analysis', self.config.return_gating_analysis),
            return_timing_info=kwargs.get('return_timing_info', self.config.return_timing_info),
            max_length=kwargs.get('max_length', self.config.max_length),
            pad_token_id=kwargs.get('pad_token_id', self.config.pad_token_id),
            eos_token_id=kwargs.get('eos_token_id', self.config.eos_token_id),
        )
        return config
    
    def _generate_batch(
        self,
        inputs: Dict[str, torch.Tensor],
        config: InferenceConfig
    ) -> Dict[str, Any]:
        """Generate text for a batch of inputs."""
        input_ids = inputs['input_ids']
        attention_mask = inputs.get('attention_mask')
        batch_size, prompt_length = input_ids.shape
        
        # Initialize generation state
        generated_ids = input_ids.clone()
        past_key_values = None
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        
        # Gating analysis storage
        gating_history = [] if config.return_gating_analysis else None
        
        # Generation loop
        for step in range(config.max_new_tokens):
            # Prepare input for current step
            if past_key_values is None:
                current_input_ids = generated_ids
                current_attention_mask = attention_mask
            else:
                current_input_ids = generated_ids[:, -1:]
                if attention_mask is not None:
                    current_attention_mask = attention_mask
                else:
                    current_attention_mask = None
            
            # Forward pass
            outputs = self.model(
                input_ids=current_input_ids,
                attention_mask=current_attention_mask,
                past_key_values=past_key_values,
                use_cache=config.use_cache,
                return_dict=True
            )
            
            # Get next token logits
            next_token_logits = outputs.logits[:, -1, :]  # [batch_size, vocab_size]
            past_key_values = outputs.past_key_values if config.use_cache else None
            
            # Store gating information
            if config.return_gating_analysis and hasattr(outputs, 'gating_coefficients'):
                gating_coeffs = outputs.gating_coefficients[:, -1, :].squeeze(-1)  # [batch_size]
                gating_history.append(gating_coeffs.cpu())
            
            # Apply repetition penalty
            if config.repetition_penalty != 1.0:
                next_token_logits = self._apply_repetition_penalty(
                    next_token_logits, generated_ids, config.repetition_penalty
                )
            
            # Apply temperature
            if config.temperature != 1.0:
                next_token_logits = next_token_logits / config.temperature
            
            # Apply top-k filtering
            if config.top_k > 0:
                next_token_logits = self._apply_top_k_filtering(next_token_logits, config.top_k)
            
            # Apply top-p filtering
            if config.top_p < 1.0:
                next_token_logits = self._apply_top_p_filtering(next_token_logits, config.top_p)
            
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
                    torch.ones(batch_size, 1, dtype=attention_mask.dtype, device=self.device)
                ], dim=-1)
            
            # Check for EOS tokens
            finished = finished | (next_tokens.squeeze(-1) == config.eos_token_id)
            if finished.all():
                break
        
        # Prepare results
        results = {
            'generated_ids': generated_ids,
        }
        
        # Add gating analysis if requested
        if config.return_gating_analysis and gating_history:
            gating_analysis = self._analyze_gating_behavior(gating_history, batch_size)
            results['gating_analysis'] = gating_analysis
        
        return results
    
    def _apply_repetition_penalty(
        self,
        logits: torch.Tensor,
        generated_ids: torch.Tensor,
        penalty: float
    ) -> torch.Tensor:
        """Apply repetition penalty to logits."""
        batch_size, vocab_size = logits.shape
        
        for i in range(batch_size):
            # Get unique tokens in sequence
            unique_tokens = generated_ids[i].unique()
            
            # Apply penalty
            for token in unique_tokens:
                if logits[i, token] < 0:
                    logits[i, token] *= penalty
                else:
                    logits[i, token] /= penalty
        
        return logits
    
    def _apply_top_k_filtering(self, logits: torch.Tensor, top_k: int) -> torch.Tensor:
        """Apply top-k filtering to logits."""
        top_k = min(top_k, logits.size(-1))
        top_k_logits, _ = torch.topk(logits, top_k)
        indices_to_remove = logits < top_k_logits[..., -1, None]
        logits[indices_to_remove] = -float('inf')
        return logits
    
    def _apply_top_p_filtering(self, logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """Apply top-p (nucleus) filtering to logits."""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # Scatter sorted indices to original order
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = -float('inf')
        
        return logits
    
    def _analyze_gating_behavior(
        self,
        gating_history: List[torch.Tensor],
        batch_size: int
    ) -> List[Dict[str, Any]]:
        """Analyze gating behavior across generation."""
        # Stack gating coefficients [num_steps, batch_size]
        gating_tensor = torch.stack(gating_history, dim=0)
        
        analyses = []
        for i in range(batch_size):
            sequence_gating = gating_tensor[:, i]  # [num_steps]
            
            analysis = {
                'mean_rm_contribution': sequence_gating.mean().item(),
                'mean_lm_contribution': (1 - sequence_gating).mean().item(),
                'rm_dominance_steps': (sequence_gating > 0.5).sum().item(),
                'lm_dominance_steps': (sequence_gating < 0.5).sum().item(),
                'balanced_steps': ((sequence_gating > 0.4) & (sequence_gating < 0.6)).sum().item(),
                'gating_std': sequence_gating.std().item(),
                'gating_trajectory': sequence_gating.tolist(),
                'total_steps': len(sequence_gating)
            }
            
            analyses.append(analysis)
        
        return analyses
    
    def compare_with_baseline(
        self,
        prompts: Union[str, List[str]],
        baseline_model: torch.nn.Module,
        baseline_tokenizer: PreTrainedTokenizer,
        **generation_kwargs
    ) -> Dict[str, Any]:
        """
        Compare Dual-Head generation with a baseline model.
        
        Args:
            prompts: Input prompts
            baseline_model: Baseline model for comparison
            baseline_tokenizer: Baseline tokenizer
            **generation_kwargs: Generation parameters
            
        Returns:
            Comparison results
        """
        # Generate with Dual-Head
        dual_head_results = self.generate_text(
            prompts,
            return_timing_info=True,
            return_gating_analysis=True,
            **generation_kwargs
        )
        
        # Generate with baseline
        if isinstance(prompts, str):
            prompts = [prompts]
        
        baseline_inputs = baseline_tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        
        start_time = time.time()
        with torch.no_grad():
            baseline_outputs = baseline_model.generate(
                **baseline_inputs,
                max_new_tokens=generation_kwargs.get('max_new_tokens', self.config.max_new_tokens),
                temperature=generation_kwargs.get('temperature', self.config.temperature),
                top_p=generation_kwargs.get('top_p', self.config.top_p),
                do_sample=generation_kwargs.get('do_sample', self.config.do_sample),
                pad_token_id=baseline_tokenizer.pad_token_id,
                eos_token_id=baseline_tokenizer.eos_token_id,
            )
        baseline_time = time.time() - start_time
        
        # Decode baseline outputs
        baseline_texts = []
        for i, generated_ids in enumerate(baseline_outputs):
            prompt_length = baseline_inputs['input_ids'][i].shape[0]
            new_tokens = generated_ids[prompt_length:]
            text = baseline_tokenizer.decode(new_tokens, skip_special_tokens=True)
            baseline_texts.append(text)
        
        if len(baseline_texts) == 1:
            baseline_texts = baseline_texts[0]
        
        return {
            'dual_head_results': dual_head_results,
            'baseline_results': {
                'generated_texts': baseline_texts,
                'generation_time': baseline_time,
                'tokens_per_second': len(baseline_outputs[0]) / baseline_time if baseline_time > 0 else 0
            },
            'speedup': baseline_time / dual_head_results['timing_info']['generation_time'] if dual_head_results['timing_info']['generation_time'] > 0 else float('inf')
        }
    
    def analyze_alignment_behavior(
        self,
        prompts: Union[str, List[str]],
        **generation_kwargs
    ) -> Dict[str, Any]:
        """
        Analyze how the model's alignment behavior changes during generation.
        
        This method provides detailed insights into how the gating mechanism
        adapts the balance between fluency and alignment throughout generation.
        """
        results = self.generate_text(
            prompts,
            return_gating_analysis=True,
            return_timing_info=True,
            **generation_kwargs
        )
        
        if isinstance(prompts, str):
            gating_analyses = [results['gating_analysis']]
            generated_texts = [results['generated_texts']]
        else:
            gating_analyses = results['gating_analysis']
            generated_texts = results['generated_texts']
        
        alignment_analysis = {
            'overall_statistics': {
                'avg_rm_contribution': sum(analysis['mean_rm_contribution'] for analysis in gating_analyses) / len(gating_analyses),
                'avg_lm_contribution': sum(analysis['mean_lm_contribution'] for analysis in gating_analyses) / len(gating_analyses),
                'avg_balanced_ratio': sum(analysis['balanced_steps'] / analysis['total_steps'] for analysis in gating_analyses) / len(gating_analyses),
            },
            'per_sequence_analysis': []
        }
        
        for i, (text, analysis) in enumerate(zip(generated_texts, gating_analyses)):
            sequence_analysis = {
                'generated_text': text,
                'gating_statistics': analysis,
                'alignment_insights': self._interpret_gating_pattern(analysis['gating_trajectory'])
            }
            alignment_analysis['per_sequence_analysis'].append(sequence_analysis)
        
        return alignment_analysis
    
    def _interpret_gating_pattern(self, gating_trajectory: List[float]) -> Dict[str, Any]:
        """Interpret the gating pattern to provide alignment insights."""
        trajectory = torch.tensor(gating_trajectory)
        
        # Identify different phases
        high_rm_mask = trajectory > 0.7
        high_lm_mask = trajectory < 0.3
        balanced_mask = (trajectory >= 0.3) & (trajectory <= 0.7)
        
        # Calculate phase transitions
        diff = trajectory[1:] - trajectory[:-1]
        increasing_trend = (diff > 0.1).sum().item()
        decreasing_trend = (diff < -0.1).sum().item()
        
        insights = {
            'alignment_dominant_steps': high_rm_mask.sum().item(),
            'fluency_dominant_steps': high_lm_mask.sum().item(),
            'balanced_steps': balanced_mask.sum().item(),
            'trend_towards_alignment': increasing_trend,
            'trend_towards_fluency': decreasing_trend,
            'gating_volatility': trajectory.std().item(),
            'final_alignment_weight': trajectory[-1].item() if len(trajectory) > 0 else 0.0,
            'interpretation': self._generate_interpretation(trajectory)
        }
        
        return insights
    
    def _generate_interpretation(self, trajectory: torch.Tensor) -> str:
        """Generate human-readable interpretation of gating behavior."""
        mean_val = trajectory.mean().item()
        std_val = trajectory.std().item()
        final_val = trajectory[-1].item() if len(trajectory) > 0 else 0.0
        
        if mean_val > 0.6:
            base = "The model heavily favors alignment over fluency"
        elif mean_val < 0.4:
            base = "The model heavily favors fluency over alignment"
        else:
            base = "The model maintains a balanced approach"
        
        if std_val > 0.2:
            volatility = " with high adaptive behavior throughout generation"
        elif std_val > 0.1:
            volatility = " with moderate adaptive adjustments"
        else:
            volatility = " with consistent behavior"
        
        if abs(final_val - mean_val) > 0.2:
            if final_val > mean_val:
                trend = ". The model becomes more alignment-focused towards the end"
            else:
                trend = ". The model becomes more fluency-focused towards the end"
        else:
            trend = ""
        
        return base + volatility + trend