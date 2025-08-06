#!/usr/bin/env python3
"""
Generate responses using ARGS model for evaluation.
"""

import sys
import json
import torch
import random
import numpy as np
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# Add ARGS to path
sys.path.insert(0, "/home/ibel/research/args")

def generate_args_responses(num_samples=300, random_seed=42):
    """Generate responses using ARGS model."""
    
    print("=== ARGS Response Generation ===")
    
    # Set random seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
    
    # Configuration - look for ARGS implementation
    base_model = "argsearch/llama-7b-sft-float32"
    
    try:
        # Try to import official ARGS implementation
        try:
            from argsearch import ARGS as ARGSGenerator
            args_available = True
            print("Using official ARGS implementation with reward model")
        except ImportError as e:
            print(f"WARNING: Official ARGS not available: {e}")
            args_available = False
        
        # Load base model and tokenizer
        print(f"Loading base model: {base_model}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        # Move to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("ARGS setup completed!")
        
        # Initialize ARGS if available
        args_generator = None
        if args_available:
            try:
                print("Initializing ARGS generator...")
                # Use official ARGS checkpoints as specified in the repo
                llm_path = base_model  # "argsearch/llama-7b-sft-float32"
                rm_path = "argsearch/llama-7b-rm-float32"  # Official ARGS reward model
                
                # Use float16 for efficiency and single GPU
                args_generator = ARGSGenerator(
                    llm_path=llm_path, 
                    rm_path=rm_path, 
                    llm_dev="cuda:0", 
                    rm_dev="cuda:0",  # Use same GPU for simplicity
                    torch_dtype=torch.float16
                )
                print("ARGS generator initialized successfully!")
            except Exception as e:
                print(f"Failed to initialize ARGS generator: {e}")
                import traceback
                traceback.print_exc()
                print("Falling back to base model...")
                args_available = False
                args_generator = None
        
        # Load HH-RLHF test dataset
        print("Loading HH-RLHF test dataset...")
        dataset = load_dataset("Anthropic/hh-rlhf", split="test")
        
        # Sample random indices
        all_indices = list(range(len(dataset)))
        sampled_indices = random.sample(all_indices, min(num_samples, len(dataset)))
        sampled_data = dataset.select(sampled_indices)
        
        print(f"Generating responses for {len(sampled_indices)} samples...")
        
        results = []
        
        for i, sample in enumerate(tqdm(sampled_data, desc="Generating ARGS responses")):
            try:
                # Extract prompt
                full_conversation = sample["chosen"]
                if "\n\nAssistant:" in full_conversation:
                    prompt = full_conversation.split("\n\nAssistant:")[0] + "\n\nAssistant:"
                else:
                    prompt = full_conversation + "\n\nAssistant:"
                
                if args_generator:
                    # Use ARGS generator with official API (following README example)
                    output_tokens = args_generator.generate(
                        prompt,      # positional argument as per README
                        weight=1.0,  # reward weight as specified in paper
                        topk=10,     # top-k candidates for reward evaluation
                        max_new_token=128,
                        method="greedy"  # args-greedy decoding
                    )
                    # Convert tokens to text
                    generated_text = args_generator.tokens_to_text(output_tokens)[0]
                    # Remove the original prompt from the response
                    if generated_text.startswith(prompt):
                        generated_text = generated_text[len(prompt):].strip()
                else:
                    # Fallback to base model generation with beam search for better quality
                    inputs = tokenizer(prompt, return_tensors="pt")
                    if torch.cuda.is_available():
                        inputs = {k: v.to(model.device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=128,
                            do_sample=False,  # Use greedy decoding for reproducibility
                            pad_token_id=tokenizer.eos_token_id,
                            eos_token_id=tokenizer.eos_token_id
                        )
                    
                    prompt_length = inputs['input_ids'].shape[1]
                    generated_tokens = outputs[0][prompt_length:]
                    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                
                results.append({
                    'sample_id': sampled_indices[i],
                    'prompt': prompt,
                    'response': generated_text,
                    'method': 'ARGS',
                    'reference_chosen': sample["chosen"].split("\n\nAssistant:")[-1].strip(),
                    'reference_rejected': sample["rejected"].split("\n\nAssistant:")[-1].strip(),
                    'args_available': args_generator is not None
                })
                
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                results.append({
                    'sample_id': sampled_indices[i],
                    'prompt': prompt if 'prompt' in locals() else "N/A",
                    'response': f"[ERROR: {str(e)}]",
                    'method': 'ARGS',
                    'error': str(e)
                })
        
        # Save results
        output_file = f"/home/ibel/research/Dual-Head/evaluation/outputs/args_responses_{num_samples}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"ARGS responses saved to: {output_file}")
        return output_file
        
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    num_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    generate_args_responses(num_samples)