#!/usr/bin/env python3
"""
Evaluate Dual-Head model on 300 random samples from HH-RLHF test dataset.
"""

import sys
import json
import torch
import random
import time
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np

# Add Dual-Head to path
sys.path.insert(0, "/home/ibel/research/Dual-Head/src")
from dual_head.dual_head_model import DualHeadModel
from dual_head.inference.inference import DualHeadInference, InferenceConfig

def evaluate_dual_head_model():
    """Evaluate Dual-Head model on HH-RLHF test dataset."""
    
    # Configuration
    model_path = "./outputs/dual_head_full_dataset"
    base_model = "argsearch/llama-7b-sft-float32"
    num_samples = 300
    max_new_tokens = 128
    temperature = 0.7
    random_seed = 42
    
    print("=== Dual-Head Model Evaluation ===")
    print(f"Model path: {model_path}")
    print(f"Number of samples: {num_samples}")
    print(f"Max new tokens: {max_new_tokens}")
    print(f"Temperature: {temperature}")
    print(f"Random seed: {random_seed}")
    print()
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
    
    # Check if model exists
    if not Path(model_path).exists():
        print(f"ERROR: Model not found at {model_path}")
        print("Please train the model first using ./train_dualhead.sh")
        return
    
    try:
        # Load model and tokenizer
        print("Loading Dual-Head model...")
        model = DualHeadModel.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        
        # Set pad token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("Model loaded successfully!")
        
        # Create inference interface
        config = InferenceConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            return_gating_analysis=True,
            return_timing_info=True
        )
        
        inference = DualHeadInference(model, tokenizer, config)
        print("Inference interface initialized!")
        
        # Load HH-RLHF test dataset
        print("Loading HH-RLHF test dataset...")
        dataset = load_dataset("Anthropic/hh-rlhf", split="test")
        print(f"Total test samples: {len(dataset)}")
        
        # Sample random indices
        all_indices = list(range(len(dataset)))
        sampled_indices = random.sample(all_indices, min(num_samples, len(dataset)))
        sampled_data = dataset.select(sampled_indices)
        
        print(f"Selected {len(sampled_indices)} random samples for evaluation")
        print()
        
        # Evaluation metrics
        results = []
        total_tokens_generated = 0
        total_generation_time = 0
        gating_stats = {
            'rm_contributions': [],
            'lm_contributions': [],
            'balanced_steps': [],
            'rm_dominant_samples': 0,
            'lm_dominant_samples': 0,
            'balanced_samples': 0
        }
        
        # Process samples
        print("Starting evaluation...")
        for i, sample in enumerate(tqdm(sampled_data, desc="Evaluating")):
            try:
                # Extract prompt (remove assistant response)
                full_conversation = sample["chosen"]
                if "\n\nAssistant:" in full_conversation:
                    prompt = full_conversation.split("\n\nAssistant:")[0] + "\n\nAssistant:"
                else:
                    prompt = full_conversation + "\n\nAssistant:"
                
                # Generate response
                start_time = time.time()
                result = inference.generate_text(prompt)
                end_time = time.time()
                
                generated_text = result['generated_texts']
                gating_analysis = result['gating_analysis']
                timing_info = result['timing_info']
                
                # Update metrics
                total_tokens_generated += len(tokenizer.encode(generated_text))
                total_generation_time += timing_info['generation_time']
                
                # Gating statistics
                rm_contrib = gating_analysis['mean_rm_contribution']
                lm_contrib = gating_analysis['mean_lm_contribution']
                balanced_ratio = gating_analysis['balanced_steps'] / gating_analysis['total_steps']
                
                gating_stats['rm_contributions'].append(rm_contrib)
                gating_stats['lm_contributions'].append(lm_contrib)
                gating_stats['balanced_steps'].append(balanced_ratio)
                
                # Categorize sample
                if rm_contrib > 0.6:
                    gating_stats['rm_dominant_samples'] += 1
                elif lm_contrib > 0.6:
                    gating_stats['lm_dominant_samples'] += 1
                else:
                    gating_stats['balanced_samples'] += 1
                
                # Store result
                sample_result = {
                    'sample_id': sampled_indices[i],
                    'prompt': prompt,
                    'generated_response': generated_text,
                    'reference_response': sample["chosen"].split("\n\nAssistant:")[-1].strip(),
                    'gating_analysis': gating_analysis,
                    'timing_info': timing_info,
                    'tokens_generated': len(tokenizer.encode(generated_text))
                }
                
                results.append(sample_result)
                
                # Progress update every 50 samples
                if (i + 1) % 50 == 0:
                    avg_rm = np.mean(gating_stats['rm_contributions'][-50:])
                    avg_tokens_per_sec = timing_info['tokens_per_second']
                    print(f"Progress: {i+1}/{len(sampled_data)} | Avg RM contrib: {avg_rm:.3f} | Speed: {avg_tokens_per_sec:.1f} tok/s")
                
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                # Store error result
                error_result = {
                    'sample_id': sampled_indices[i] if i < len(sampled_indices) else i,
                    'prompt': prompt if 'prompt' in locals() else "N/A",
                    'generated_response': f"[ERROR: {str(e)}]",
                    'reference_response': "N/A",
                    'error': str(e)
                }
                results.append(error_result)
                continue
        
        # Calculate final statistics
        print("\n=== Evaluation Results ===")
        
        # Basic stats
        successful_samples = len([r for r in results if 'error' not in r])
        failed_samples = len(results) - successful_samples
        
        print(f"Successful samples: {successful_samples}/{len(results)}")
        print(f"Failed samples: {failed_samples}")
        
        if successful_samples > 0:
            # Performance metrics
            avg_tokens_per_second = total_tokens_generated / total_generation_time if total_generation_time > 0 else 0
            avg_tokens_per_sample = total_tokens_generated / successful_samples
            avg_time_per_sample = total_generation_time / successful_samples
            
            print(f"\n--- Performance Metrics ---")
            print(f"Average tokens per second: {avg_tokens_per_second:.2f}")
            print(f"Average tokens per sample: {avg_tokens_per_sample:.1f}")
            print(f"Average generation time per sample: {avg_time_per_sample:.3f}s")
            print(f"Total tokens generated: {total_tokens_generated}")
            print(f"Total generation time: {total_generation_time:.2f}s")
            
            # Gating analysis
            print(f"\n--- Gating Analysis ---")
            print(f"Average RM contribution: {np.mean(gating_stats['rm_contributions']):.3f}")
            print(f"Average LM contribution: {np.mean(gating_stats['lm_contributions']):.3f}")
            print(f"Average balanced steps ratio: {np.mean(gating_stats['balanced_steps']):.3f}")
            print(f"RM-dominant samples: {gating_stats['rm_dominant_samples']} ({gating_stats['rm_dominant_samples']/successful_samples*100:.1f}%)")
            print(f"LM-dominant samples: {gating_stats['lm_dominant_samples']} ({gating_stats['lm_dominant_samples']/successful_samples*100:.1f}%)")
            print(f"Balanced samples: {gating_stats['balanced_samples']} ({gating_stats['balanced_samples']/successful_samples*100:.1f}%)")
            
            # Distribution stats
            rm_std = np.std(gating_stats['rm_contributions'])
            lm_std = np.std(gating_stats['lm_contributions'])
            print(f"RM contribution std: {rm_std:.3f}")
            print(f"LM contribution std: {lm_std:.3f}")
        
        # Save detailed results
        output_file = f"dual_head_evaluation_{num_samples}_samples.json"
        evaluation_summary = {
            'evaluation_config': {
                'model_path': model_path,
                'base_model': base_model,
                'num_samples': num_samples,
                'max_new_tokens': max_new_tokens,
                'temperature': temperature,
                'random_seed': random_seed
            },
            'summary_statistics': {
                'successful_samples': successful_samples,
                'failed_samples': failed_samples,
                'avg_tokens_per_second': avg_tokens_per_second if successful_samples > 0 else 0,
                'avg_tokens_per_sample': avg_tokens_per_sample if successful_samples > 0 else 0,
                'avg_time_per_sample': avg_time_per_sample if successful_samples > 0 else 0,
                'total_tokens_generated': total_tokens_generated,
                'total_generation_time': total_generation_time
            },
            'gating_statistics': {
                'avg_rm_contribution': np.mean(gating_stats['rm_contributions']) if gating_stats['rm_contributions'] else 0,
                'avg_lm_contribution': np.mean(gating_stats['lm_contributions']) if gating_stats['lm_contributions'] else 0,
                'avg_balanced_steps_ratio': np.mean(gating_stats['balanced_steps']) if gating_stats['balanced_steps'] else 0,
                'rm_dominant_samples': gating_stats['rm_dominant_samples'],
                'lm_dominant_samples': gating_stats['lm_dominant_samples'],
                'balanced_samples': gating_stats['balanced_samples'],
                'rm_contribution_std': np.std(gating_stats['rm_contributions']) if gating_stats['rm_contributions'] else 0,
                'lm_contribution_std': np.std(gating_stats['lm_contributions']) if gating_stats['lm_contributions'] else 0
            },
            'detailed_results': results
        }
        
        with open(output_file, 'w') as f:
            json.dump(evaluation_summary, f, indent=2)
        
        print(f"\nDetailed results saved to: {output_file}")
        print("Evaluation completed successfully!")
        
    except Exception as e:
        print(f"Fatal error during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    evaluate_dual_head_model()