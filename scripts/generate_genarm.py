#!/usr/bin/env python3
"""
Real GenARM generation script for experimental comparison.
"""
import sys
import json
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

# Add GenARM to path
sys.path.insert(0, "/home/ibel/research/GenARM/language-model-arithmetic/src")
from model_arithmetic import ModelArithmetic, PromptedLLM

def generate_genarm_responses(config_path, output_path):
    """Generate responses using real GenARM implementation."""
    
    try:
        # Load configuration
        with open(config_path) as f:
            config = json.load(f)
        
        print(f"Loaded GenARM configuration: {config_path}")
        
        # Initialize GenARM models
        print("Loading GenARM models...")
        model_base = config["base_model"]
        model_autoregressive_rm = config["genarm"]["autoregressive_rm_path"]
        
        # Template for prompts
        prompt_template = lambda system_prompt, input_string: f"{input_string}"
        
        alpha = config["genarm"]["alpha"]
        temperature = config["genarm"]["temperature"]
        
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_base)
        
        print("Loading base model...")
        M_base = PromptedLLM(
            system_prompt="", 
            prompt_template=prompt_template, 
            model=model_base, 
            tokenizer=tokenizer
        )
        
        print("Loading autoregressive reward model...")
        M_arm = PromptedLLM(
            system_prompt="", 
            prompt_template=prompt_template, 
            model=model_autoregressive_rm, 
            tokenizer=tokenizer
        )
        
        print("Creating GenARM formula...")
        formula = M_base + alpha * M_arm
        M = ModelArithmetic(
            formula, 
            needs_input_tokens_lm_eval=False, 
            lm_eval_task=None, 
            dtype=torch.bfloat16
        )
        
        print("GenARM model initialized successfully")
        
        # Load dataset
        print("Loading HH-RLHF dataset...")
        dataset = load_dataset("Anthropic/hh-rlhf", split="test")
        max_samples = min(config["experiment"]["max_samples"], len(dataset))
        
        print(f"Processing {max_samples} samples...")
        
        results = []
        
        for i, sample in enumerate(tqdm(dataset.select(range(max_samples)), desc="Generating GenARM responses")):
            try:
                # Extract prompt in HH-RLHF format
                prompt = sample["chosen"].split("\n\nAssistant:")[0] + "\n\nAssistant:"
                
                # Generate response using GenARM
                response = M.generate_text(
                    prompt,
                    max_new_tokens=config["experiment"]["max_new_tokens"],
                    temperature=temperature,
                    top_p=1.0,
                    top_k=0,
                    do_speculation=False
                )[0]
                
                # Remove EOS token if present
                if hasattr(M, 'tokenizer') and M.tokenizer.eos_token:
                    response = response.removesuffix(M.tokenizer.eos_token)
                
                results.append({
                    "prompt": prompt,
                    "response": response,
                    "method": "GenARM",
                    "sample_id": i
                })
                
                if (i + 1) % 10 == 0:
                    print(f"Generated {i + 1}/{max_samples} responses")
                    
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                results.append({
                    "prompt": prompt if 'prompt' in locals() else f"Sample {i}",
                    "response": f"[ERROR: {str(e)}]",
                    "method": "GenARM",
                    "sample_id": i
                })
                continue
        
        # Save results
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"GenARM generation completed. Results saved to: {output_path}")
        print(f"Total responses generated: {len(results)}")
        
    except Exception as e:
        print(f"Error in GenARM generation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python generate_genarm.py <config_path> <output_path>")
        sys.exit(1)
    
    generate_genarm_responses(sys.argv[1], sys.argv[2])