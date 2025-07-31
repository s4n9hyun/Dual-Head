"""
Data processing utilities for Dual-Head training.

This module handles:
1. Preference data formatting and collation
2. Message format encoding (similar to GenARM)
3. Efficient batching for dual-head training
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from transformers import PreTrainedTokenizer
from datasets import Dataset


@dataclass
class DualHeadDataCollator:
    """
    Data collator for Dual-Head training with preference data.
    
    This collator handles both standard language modeling data and
    preference pairs for ARM-style training.
    """
    
    tokenizer: PreTrainedTokenizer
    max_length: int = 2048
    padding: Union[bool, str] = True
    return_tensors: str = "pt"
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of examples for Dual-Head training.
        
        Args:
            features: List of examples, each containing:
                - For SFT: input_ids, attention_mask, labels
                - For preference: chosen_input_ids, rejected_input_ids, etc.
                
        Returns:
            Batch dictionary with properly padded tensors
        """
        # Determine batch type
        has_preference_data = any('chosen_input_ids' in f for f in features)
        
        if has_preference_data:
            return self._collate_preference_batch(features)
        else:
            return self._collate_sft_batch(features)
    
    def _collate_sft_batch(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate standard supervised fine-tuning batch."""
        batch = {}
        
        # Get all keys from features
        keys = set().union(*(f.keys() for f in features))
        
        for key in keys:
            if key in ['input_ids', 'attention_mask', 'labels']:
                # Pad sequences
                sequences = [f[key] for f in features if key in f]
                if sequences:
                    batch[key] = self._pad_sequences(sequences, pad_value=-100 if key == 'labels' else 0)
        
        return batch
    
    def _collate_preference_batch(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate preference learning batch with chosen/rejected pairs."""
        batch = {}
        
        # Handle chosen sequences
        chosen_keys = ['chosen_input_ids', 'chosen_attention_mask', 'chosen_labels']
        for key in chosen_keys:
            sequences = [f[key] for f in features if key in f]
            if sequences:
                pad_value = -100 if 'labels' in key else 0
                batch[key] = self._pad_sequences(sequences, pad_value=pad_value)
        
        # Handle rejected sequences
        rejected_keys = ['rejected_input_ids', 'rejected_attention_mask', 'rejected_labels']
        for key in rejected_keys:
            sequences = [f[key] for f in features if key in f]
            if sequences:
                pad_value = -100 if 'labels' in key else 0
                batch[key] = self._pad_sequences(sequences, pad_value=pad_value)
        
        # Handle any additional keys (like regular input_ids for mixed training)
        other_keys = set().union(*(f.keys() for f in features)) - set(chosen_keys + rejected_keys)
        for key in other_keys:
            if key in ['input_ids', 'attention_mask', 'labels']:
                sequences = [f[key] for f in features if key in f]
                if sequences:
                    pad_value = -100 if key == 'labels' else 0
                    batch[key] = self._pad_sequences(sequences, pad_value=pad_value)
        
        return batch
    
    def _pad_sequences(
        self, 
        sequences: List[Union[List[int], torch.Tensor]], 
        pad_value: int = 0
    ) -> torch.Tensor:
        """Pad sequences to the same length."""
        # Convert to tensors if needed
        if isinstance(sequences[0], list):
            sequences = [torch.tensor(seq, dtype=torch.long) for seq in sequences]
        
        # Get max length
        max_len = min(max(len(seq) for seq in sequences), self.max_length)
        
        # Pad sequences
        padded = []
        for seq in sequences:
            if len(seq) > max_len:
                # Truncate if too long
                seq = seq[:max_len]
            
            # Pad if too short
            padding_length = max_len - len(seq)
            if padding_length > 0:
                if pad_value == 0:
                    padded_seq = F.pad(seq, (0, padding_length), value=pad_value)
                else:
                    padded_seq = F.pad(seq, (0, padding_length), value=pad_value)
                padded.append(padded_seq)
            else:
                padded.append(seq)
        
        return torch.stack(padded)


def _parse_hh_rlhf_format(text: str) -> List[Dict[str, str]]:
    """Parse HH-RLHF format string into structured messages.
    
    HH-RLHF format example:
    "Human: What is the capital of France?\\n\\nAssistant: The capital of France is Paris."
    
    Returns:
        List of message dictionaries with 'role' and 'content' keys
    """
    messages = []
    
    # Split by "Human:" and "Assistant:" markers
    parts = text.strip().split('\\n\\n')
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
            
        if part.startswith('Human:'):
            content = part[6:].strip()  # Remove "Human:" prefix
            if content:
                messages.append({"role": "user", "content": content})
        elif part.startswith('Assistant:'):
            content = part[10:].strip()  # Remove "Assistant:" prefix
            if content:
                messages.append({"role": "assistant", "content": content})
        # Handle alternative formats
        elif part.startswith('H:'):
            content = part[2:].strip()  # Remove "H:" prefix
            if content:
                messages.append({"role": "user", "content": content})
        elif part.startswith('A:'):
            content = part[2:].strip()  # Remove "A:" prefix
            if content:
                messages.append({"role": "assistant", "content": content})
    
    return messages


def encode_with_messages_format(
    example: Dict[str, Any],
    tokenizer: PreTrainedTokenizer,
    max_seq_length: int = 2048,
    add_bos: bool = False
) -> Dict[str, torch.Tensor]:
    """
    Encode examples with messages format (similar to GenARM).
    
    This function handles conversation-style data where each example has
    'chosen' and 'rejected' fields containing either:
    - Lists of messages with 'role' and 'content' keys
    - Simple strings in HH-RLHF format ("H: ... A: ...")
    
    Args:
        example: Dictionary with 'chosen' and 'rejected' message lists or strings
        tokenizer: Tokenizer to use for encoding
        max_seq_length: Maximum sequence length
        add_bos: Whether to add BOS token
        
    Returns:
        Dictionary with encoded chosen/rejected sequences
    """
    chosen_data = example["chosen"]
    rejected_data = example["rejected"]
    
    # Handle different input formats
    if isinstance(chosen_data, str) and isinstance(rejected_data, str):
        # HH-RLHF format - convert strings to message format
        chosen_messages = _parse_hh_rlhf_format(chosen_data)
        rejected_messages = _parse_hh_rlhf_format(rejected_data)
    elif isinstance(chosen_data, list) and isinstance(rejected_data, list):
        # Already in message format
        chosen_messages = chosen_data
        rejected_messages = rejected_data
    else:
        raise ValueError(f"Unsupported data format. Expected strings or lists, got {type(chosen_data)} and {type(rejected_data)}")
    
    if len(chosen_messages) == 0 or len(rejected_messages) == 0:
        raise ValueError("Both chosen and rejected messages must be non-empty")
    
    def _concat_messages(messages: List[Dict[str, str]]) -> str:
        """Concatenate messages with role indicators."""
        message_text = ""
        for message in messages:
            role = message["role"]
            content = message["content"].strip()
            
            if role == "system":
                message_text += f"<|system|>\n{content}\n"
            elif role == "user":
                message_text += f"<|user|>\n{content}\n"
            elif role == "assistant":
                message_text += f"<|assistant|>\n{content}{tokenizer.eos_token}\n"
            else:
                raise ValueError(f"Invalid role: {role}")
        
        return message_text
    
    def encode_messages(messages: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        """Encode a message sequence."""
        # Concatenate messages
        text = _concat_messages(messages).strip()
        if add_bos:
            text = tokenizer.bos_token + text
        
        # Tokenize
        encoded = tokenizer(
            text,
            return_tensors="pt",
            max_length=max_seq_length,
            truncation=True,
            padding=False
        )
        
        input_ids = encoded.input_ids.squeeze(0)
        attention_mask = encoded.attention_mask.squeeze(0)
        labels = input_ids.clone()
        
        # Mask non-assistant parts to avoid loss computation
        for message_idx, message in enumerate(messages):
            if message["role"] != "assistant":
                # Find message boundaries
                if message_idx == 0:
                    message_start_idx = 0
                else:
                    prefix_text = _concat_messages(messages[:message_idx])
                    if add_bos:
                        prefix_text = tokenizer.bos_token + prefix_text
                    message_start_idx = len(tokenizer(prefix_text, add_special_tokens=False)["input_ids"])
                
                # Find message end
                if message_idx < len(messages) - 1 and messages[message_idx + 1]["role"] == "assistant":
                    # Include assistant role token in masked region
                    prefix_with_role = _concat_messages(messages[:message_idx + 1]) + "<|assistant|>\n"
                else:
                    prefix_with_role = _concat_messages(messages[:message_idx + 1])
                
                if add_bos:
                    prefix_with_role = tokenizer.bos_token + prefix_with_role
                
                message_end_idx = len(tokenizer(prefix_with_role, add_special_tokens=False)["input_ids"])
                
                # Mask this region
                if message_end_idx <= len(labels):
                    labels[message_start_idx:message_end_idx] = -100
                
                if message_end_idx >= max_seq_length:
                    break
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
    
    # Encode both chosen and rejected
    chosen_encoded = encode_messages(chosen_messages)
    rejected_encoded = encode_messages(rejected_messages)
    
    return {
        "chosen_input_ids": chosen_encoded["input_ids"],
        "chosen_attention_mask": chosen_encoded["attention_mask"],
        "chosen_labels": chosen_encoded["labels"],
        "rejected_input_ids": rejected_encoded["input_ids"],
        "rejected_attention_mask": rejected_encoded["attention_mask"],
        "rejected_labels": rejected_encoded["labels"],
    }


def prepare_preference_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    max_seq_length: int = 2048,
    add_bos: bool = False,
    num_proc: int = 4
) -> Dataset:
    """
    Prepare a preference dataset for Dual-Head training.
    
    Args:
        dataset: Raw dataset with 'chosen' and 'rejected' fields
        tokenizer: Tokenizer to use
        max_seq_length: Maximum sequence length
        add_bos: Whether to add BOS token
        num_proc: Number of processes for mapping
        
    Returns:
        Processed dataset ready for training
    """
    # Check dataset format
    if "chosen" not in dataset.column_names or "rejected" not in dataset.column_names:
        raise ValueError("Dataset must contain 'chosen' and 'rejected' columns")
    
    # Encode examples
    def encode_example(example):
        return encode_with_messages_format(
            example=example,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            add_bos=add_bos
        )
    
    # Map encoding function
    encoded_dataset = dataset.map(
        encode_example,
        batched=False,
        num_proc=num_proc,
        remove_columns=[
            col for col in dataset.column_names 
            if col not in ["chosen", "rejected"]
        ],
        desc="Encoding preference examples"
    )
    
    # Set format for PyTorch
    encoded_dataset.set_format(type="torch")
    
    # Filter out examples with no valid labels
    def has_valid_labels(example):
        chosen_valid = (example["chosen_labels"] != -100).any()
        rejected_valid = (example["rejected_labels"] != -100).any()
        return chosen_valid and rejected_valid
    
    filtered_dataset = encoded_dataset.filter(
        has_valid_labels,
        desc="Filtering examples with valid labels"
    )
    
    return filtered_dataset


def prepare_sft_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    max_seq_length: int = 2048,
    text_column: str = "text",
    num_proc: int = 4
) -> Dataset:
    """
    Prepare a supervised fine-tuning dataset.
    
    Args:
        dataset: Raw dataset with text data
        tokenizer: Tokenizer to use
        max_seq_length: Maximum sequence length
        text_column: Name of text column
        num_proc: Number of processes for mapping
        
    Returns:
        Processed dataset ready for SFT training
    """
    def tokenize_function(examples):
        # Tokenize texts
        tokenized = tokenizer(
            examples[text_column],
            truncation=True,
            padding=False,
            max_length=max_seq_length,
            return_tensors=None
        )
        
        # Labels are the same as input_ids for causal LM
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
    
    # Tokenize dataset
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=dataset.column_names,
        desc="Tokenizing SFT examples"
    )
    
    return tokenized_dataset


def create_mixed_dataset(
    sft_dataset: Optional[Dataset] = None,
    preference_dataset: Optional[Dataset] = None,
    sft_ratio: float = 0.1
) -> Dataset:
    """
    Create a mixed dataset combining SFT and preference data.
    
    This is useful for maintaining language modeling capabilities
    while training on preferences.
    
    Args:
        sft_dataset: Supervised fine-tuning dataset
        preference_dataset: Preference dataset
        sft_ratio: Ratio of SFT examples to include
        
    Returns:
        Mixed dataset
    """
    if sft_dataset is None and preference_dataset is None:
        raise ValueError("At least one dataset must be provided")
    
    if sft_dataset is None:
        return preference_dataset
    
    if preference_dataset is None:
        return sft_dataset
    
    # Calculate number of SFT examples to include
    num_sft = int(len(preference_dataset) * sft_ratio)
    
    # Sample SFT examples
    if num_sft > len(sft_dataset):
        # Repeat SFT dataset if needed
        repeat_factor = (num_sft // len(sft_dataset)) + 1
        sft_expanded = Dataset.from_dict({
            k: v * repeat_factor for k, v in sft_dataset.to_dict().items()
        })
        sft_sampled = sft_expanded.select(range(num_sft))
    else:
        sft_sampled = sft_dataset.shuffle().select(range(num_sft))
    
    # Combine datasets
    from datasets import concatenate_datasets
    mixed_dataset = concatenate_datasets([preference_dataset, sft_sampled])
    
    # Shuffle the combined dataset
    mixed_dataset = mixed_dataset.shuffle()
    
    return mixed_dataset


class PreferenceDataCollator(DualHeadDataCollator):
    """Specialized data collator for preference-only training."""
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Only handle preference data."""
        return self._collate_preference_batch(features)


class SFTDataCollator(DualHeadDataCollator):
    """Specialized data collator for SFT-only training."""
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Only handle SFT data."""
        return self._collate_sft_batch(features)