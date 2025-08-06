#!/usr/bin/env python3
"""
Generate responses using DPO model for evaluation.
"""

import sys
import json
import torch
import random
import numpy as np
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm

def generate_dpo_responses(num_samples=300, random_seed=42):
    """Generate responses using DPO model."""
    
    print("=== DPO Response Generation ===")
    
    # Set random seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
    
    # Configuration - DPO model paths (LoRA adapters)
    base_model = "argsearch/llama-7b-sft-float32"
    dpo_adapter_paths = [
        "/home/ibel/research/DPO/dpo_final_model",
        "/home/ibel/research/DPO/dpo-llama-7b-results"
    ]
    
    adapter_path = None
    for path in dpo_adapter_paths:
        if Path(path).exists():
            adapter_path = path
            break
    
    try:
        # Check GPU availability
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Load base model first
        print(f"Loading base model: {base_model}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True
        )
        
        # Load DPO adapter if available
        if adapter_path:
            print(f"Loading DPO adapter from: {adapter_path}")
            model = PeftModel.from_pretrained(model, adapter_path)
        else:
            print("WARNING: No DPO adapter found. Using base model only.")
        
        # Ensure model is on correct device
        if not torch.cuda.is_available():
            model = model.to(device)
        
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print(f"Tokenizer info:")
        print(f"  EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
        print(f"  PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
        print(f"  UNK token: {tokenizer.unk_token} (ID: {tokenizer.unk_token_id})")
        
        print("DPO model loaded successfully!")
        
        # Load HH-RLHF test dataset
        print("Loading HH-RLHF test dataset...")
        dataset = load_dataset("Anthropic/hh-rlhf", split="test")
        
        # Sample random indices
        all_indices = list(range(len(dataset)))
        sampled_indices = random.sample(all_indices, min(num_samples, len(dataset)))
        sampled_data = dataset.select(sampled_indices)
        
        print(f"Generating responses for {len(sampled_indices)} samples...")
        
        results = []
        
        # Process individually to avoid batch padding issues
        for i in tqdm(range(len(sampled_data)), desc="Processing samples"):
            sample = sampled_data[i]
            sample_id = sampled_indices[i]
            
            try:
                # Prepare prompt
                if isinstance(sample, dict) and "chosen" in sample:
                    full_conversation = sample["chosen"]
                else:
                    print(f"Sample type: {type(sample)}, sample: {sample}")
                    continue
                    
                if "\n\nAssistant:" in full_conversation:
                    prompt = full_conversation.split("\n\nAssistant:")[0] + "\n\nAssistant:"
                else:
                    prompt = full_conversation + "\n\nAssistant:"
                
                # Tokenize single prompt (no padding needed)
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                )
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                
                prompt_length = inputs['input_ids'].shape[1]
                
                # Generate response
                with torch.no_grad():
                    outputs = model.generate(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        max_new_tokens=128,
                        min_new_tokens=5,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        use_cache=True
                    )
                
                # Extract generated tokens
                generated_tokens = outputs[0][prompt_length:]
                
                # Find actual generated part (stop at EOS)
                if tokenizer.eos_token_id in generated_tokens:
                    eos_pos = (generated_tokens == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
                    if len(eos_pos) > 0:
                        generated_tokens = generated_tokens[:eos_pos[0]]
                
                generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                
                results.append({
                    'sample_id': sample_id,
                    'prompt': prompt,
                    'response': generated_text,
                    'method': 'DPO',
                    'reference_chosen': sample["chosen"].split("\n\nAssistant:")[-1].strip(),
                    'reference_rejected': sample["rejected"].split("\n\nAssistant:")[-1].strip(),
                    'model_path': f"{base_model} + {adapter_path}" if adapter_path else base_model
                })
                        
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                results.append({
                    'sample_id': sample_id,
                    'prompt': prompt if 'prompt' in locals() else "N/A",
                    'response': f"[ERROR: {str(e)}]",
                    'method': 'DPO',
                    'error': str(e)
                })
        
        # Save results
        output_file = f"/home/ibel/research/Dual-Head/evaluation/outputs/dpo_responses_{num_samples}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"DPO responses saved to: {output_file}")
        return output_file
        
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    num_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    generate_dpo_responses(num_samples)