#!/usr/bin/env python3
"""
Generate responses using base model (llama-7b-sft) for evaluation.
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

def generate_base_model_responses(num_samples=300, random_seed=42):
    """Generate responses using base model only."""
    
    print("=== Base Model Response Generation ===")
    
    # Set random seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
    
    # Configuration - Base model only
    base_model = "argsearch/llama-7b-sft-float32"
    
    try:
        # Load base model
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
        
        print("Base model loaded successfully!")
        
        # Load HH-RLHF test dataset
        print("Loading HH-RLHF test dataset...")
        dataset = load_dataset("Anthropic/hh-rlhf", split="test")
        
        # Sample random indices
        all_indices = list(range(len(dataset)))
        sampled_indices = random.sample(all_indices, min(num_samples, len(dataset)))
        sampled_data = dataset.select(sampled_indices)
        
        print(f"Generating responses for {len(sampled_indices)} samples...")
        
        results = []
        
        for i, sample in enumerate(tqdm(sampled_data, desc="Generating base model responses")):
            try:
                # Extract prompt
                full_conversation = sample["chosen"]
                if "\n\nAssistant:" in full_conversation:
                    prompt = full_conversation.split("\n\nAssistant:")[0] + "\n\nAssistant:"
                else:
                    prompt = full_conversation + "\n\nAssistant:"
                
                # Tokenize prompt with padding and truncation
                inputs = tokenizer(
                    prompt, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True, 
                    max_length=512
                )
                if torch.cuda.is_available():
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}
                
                # Generate response with beam search
                with torch.no_grad():
                    outputs = model.generate(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs.get('attention_mask', None),
                        max_new_tokens=128,
                        min_new_tokens=5,  # Force at least 5 new tokens
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        use_cache=True
                    )
                
                # Decode response
                prompt_length = inputs['input_ids'].shape[1]
                generated_tokens = outputs[0][prompt_length:]
                generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                
                # Clean up generated text
                generated_text = generated_text.strip()
                
                results.append({
                    'sample_id': sampled_indices[i],
                    'prompt': prompt,
                    'response': generated_text,
                    'method': 'Base Model',
                    'reference_chosen': sample["chosen"].split("\n\nAssistant:")[-1].strip(),
                    'reference_rejected': sample["rejected"].split("\n\nAssistant:")[-1].strip(),
                    'model_path': base_model
                })
                
            except Exception as e:
                # Clear GPU memory on error
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f"Error processing sample {i}: {e}")
                results.append({
                    'sample_id': sampled_indices[i],
                    'prompt': prompt if 'prompt' in locals() else "N/A",
                    'response': f"[ERROR: {str(e)}]",
                    'method': 'Base Model',
                    'error': str(e)
                })
            
            # Clear GPU memory every 10 samples
            if torch.cuda.is_available() and i % 10 == 0:
                torch.cuda.empty_cache()
        
        # Save results
        output_file = f"/home/ibel/research/Dual-Head/evaluation/outputs/base_model_responses_{num_samples}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Base model responses saved to: {output_file}")
        return output_file
        
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    num_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    generate_base_model_responses(num_samples)