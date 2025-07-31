#!/usr/bin/env python3
"""
Real Dual-Head generation script for experimental comparison.
"""
import sys
import json
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

# Add Dual-Head to path
sys.path.insert(0, "/home/ibel/research/Dual-Head/src")
from dual_head.dual_head_model import DualHeadModel

def generate_dual_head_responses(config_path, output_path):
    """Generate responses using real Dual-Head implementation."""
    
    try:
        # Load configuration
        with open(config_path) as f:
            config = json.load(f)
        
        print(f"Loaded Dual-Head configuration: {config_path}")
        
        # Check if trained model exists
        model_path = config["dual_head"]["checkpoint_path"]
        if not Path(model_path).exists():
            print(f"ERROR: Dual-Head model not found at: {model_path}")
            print("Please train the model first using: ./train_with_argsearch.sh")
            sys.exit(1)
        
        print("Loading Dual-Head model...")
        model = DualHeadModel.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(config["base_model"])
        
        print("Dual-Head model loaded successfully")
        
        # Load dataset
        print("Loading HH-RLHF dataset...")
        dataset = load_dataset("Anthropic/hh-rlhf", split="test")
        max_samples = min(config["experiment"]["max_samples"], len(dataset))
        
        print(f"Processing {max_samples} samples...")
        
        results = []
        
        for i, sample in enumerate(tqdm(dataset.select(range(max_samples)), desc="Generating Dual-Head responses")):
            try:
                # Extract prompt in HH-RLHF format
                prompt = sample["chosen"].split("\n\nAssistant:")[0] + "\n\nAssistant:"
                
                # Generate response using Dual-Head
                response = model.generate(
                    prompt,
                    tokenizer=tokenizer,
                    max_new_tokens=config["experiment"]["max_new_tokens"],
                    temperature=config["dual_head"]["temperature"]
                )
                
                results.append({
                    "prompt": prompt,
                    "response": response,
                    "method": "Dual-Head",
                    "sample_id": i
                })
                
                if (i + 1) % 10 == 0:
                    print(f"Generated {i + 1}/{max_samples} responses")
                    
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                results.append({
                    "prompt": prompt if 'prompt' in locals() else f"Sample {i}",
                    "response": f"[ERROR: {str(e)}]",
                    "method": "Dual-Head",
                    "sample_id": i
                })
                continue
        
        # Save results
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"Dual-Head generation completed. Results saved to: {output_path}")
        print(f"Total responses generated: {len(results)}")
        
    except Exception as e:
        print(f"Error in Dual-Head generation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python generate_dual_head.py <config_path> <output_path>")
        sys.exit(1)
    
    generate_dual_head_responses(sys.argv[1], sys.argv[2])