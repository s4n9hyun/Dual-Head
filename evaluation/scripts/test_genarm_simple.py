#!/usr/bin/env python3
"""
Simple GenARM test script
"""

import sys
import torch
sys.path.insert(0, "/home/ibel/research/GenARM/language-model-arithmetic/src")

from transformers import AutoTokenizer
from model_arithmetic import ModelArithmetic, PromptedLLM

def test_genarm():
    print("Testing GenARM with simple setup...")
    
    # Configuration
    base_model = "argsearch/llama-7b-sft-float32"
    arm_model = "/home/ibel/research/GenARM/checkpoints/HH/arm/args-llama-sft-7b-arm-HH-epoch_1-beta_0.05-lr_5e-4-bs_32"
    alpha = 1.0
    
    try:
        # Load tokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print(f"Tokenizer loaded, vocab size: {len(tokenizer)}")
        
        # Create prompt template
        prompt_template = lambda system_prompt, input_string: f"{input_string}"
        
        # Initialize models
        print("Creating PromptedLLM instances...")
        M_base = PromptedLLM(system_prompt="Not used", prompt_template=prompt_template, 
                           model=base_model, tokenizer=tokenizer)
        print("Base model created")
        
        M_arm = PromptedLLM(system_prompt="Not used", prompt_template=prompt_template, 
                          model=arm_model, tokenizer=tokenizer)
        print("ARM model created")
        
        # Create formula
        print("Creating formula...")
        formula = M_base + alpha * M_arm
        print("Formula created")
        
        # Create ModelArithmetic
        print("Creating ModelArithmetic...")
        genarm_model = ModelArithmetic(formula, needs_input_tokens_lm_eval=False, 
                                     lm_eval_task=None, dtype=torch.bfloat16)
        print("ModelArithmetic created successfully!")
        
        # Test generation with a simple prompt
        test_prompt = "Hello, how are you?"
        print(f"\nTesting generation with prompt: '{test_prompt}'")
        
        try:
            result = genarm_model.generate_text(
                [test_prompt],  # List of strings
                max_new_tokens=20,
                temperature=0.001,
                top_p=1,
                top_k=0,
                do_speculation=False
            )
            print(f"Generation successful!")
            print(f"Result: {result[0]}")
            
        except Exception as e:
            print(f"Generation failed: {e}")
            import traceback
            traceback.print_exc()
        
    except Exception as e:
        print(f"Setup failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_genarm()