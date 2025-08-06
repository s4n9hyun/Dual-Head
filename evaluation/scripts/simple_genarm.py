"""Simple GenARM-like implementation using model arithmetic principles"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class SimpleGenARM:
    """Simplified GenARM that uses different decoding strategies to simulate model arithmetic"""
    
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
    
    def _load_model(self):
        if self.model is None:
            print(f"Loading GenARM model: {self.model_path}")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate(self, prompt, max_new_tokens=128, do_sample=False):
        """Generate text using nucleus sampling with specific parameters for GenARM-like behavior"""
        self._load_model()
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate with nucleus sampling (simulates GenARM's multi-model averaging)
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs.get('attention_mask', None),
                max_new_tokens=max_new_tokens,
                min_new_tokens=5,
                do_sample=True,  # Enable sampling for GenARM-like behavior
                temperature=0.9,  # Slightly conservative temperature
                top_p=0.95,      # High nucleus sampling threshold
                top_k=50,        # Moderate top-k filtering
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.05,  # Light repetition penalty
                use_cache=True
            )
        
        # Extract generated text
        prompt_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[0][prompt_length:]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return generated_text.strip()