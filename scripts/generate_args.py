#!/usr/bin/env python3
"""
Real ARGS generation script for experimental comparison.
"""
import sys
import json
import torch
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

# Add ARGS to path
sys.path.insert(0, "/home/ibel/research/ARGS")
from argsearch import ARGS

def generate_args_responses(config_path, output_path):
    """Generate responses using real ARGS implementation."""
    
    try:
        # Load configuration
        with open(config_path) as f:
            config = json.load(f)
        
        print(f"Loaded ARGS configuration: {config_path}")
        
        # Initialize ARGS with real models
        print("Initializing ARGS searcher...")
        searcher = ARGS(
            llm_path=config["base_model"],
            rm_path=config["reward_model"],
            llm_dev="cuda:0",
            rm_dev="cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
        )
        
        print("ARGS searcher initialized successfully")
        
        # Load dataset
        print("Loading HH-RLHF dataset...")
        dataset = load_dataset("Anthropic/hh-rlhf", split="test")
        max_samples = min(config["experiment"]["max_samples"], len(dataset))
        
        print(f"Processing {max_samples} samples...")
        
        results = []
        
        for i, sample in enumerate(tqdm(dataset.select(range(max_samples)), desc="Generating ARGS responses")):
            try:
                # Extract prompt in HH-RLHF format
                prompt = sample["chosen"].split("\n\nAssistant:")[0] + "\n\nAssistant:"
                
                # Generate response using ARGS
                output_tokens = searcher.generate(
                    prompt,
                    topk=config["args"]["k"],
                    weight=config["args"]["w"],
                    max_new_token=config["experiment"]["max_new_tokens"],
                    method="greedy"
                )
                
                # Convert tokens to text
                response = searcher.tokens_to_text(output_tokens)[0]
                
                results.append({
                    "prompt": prompt,
                    "response": response,
                    "method": "ARGS",
                    "sample_id": i
                })
                
                if (i + 1) % 10 == 0:
                    print(f"Generated {i + 1}/{max_samples} responses")
                    
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                results.append({
                    "prompt": prompt if 'prompt' in locals() else f"Sample {i}",
                    "response": f"[ERROR: {str(e)}]",
                    "method": "ARGS",
                    "sample_id": i
                })
                continue
        
        # Save results
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"ARGS generation completed. Results saved to: {output_path}")
        print(f"Total responses generated: {len(results)}")
        
    except Exception as e:
        print(f"Error in ARGS generation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python generate_args.py <config_path> <output_path>")
        sys.exit(1)
    
    generate_args_responses(sys.argv[1], sys.argv[2])