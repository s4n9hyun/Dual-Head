#!/usr/bin/env python3
"""
Dual-Head model training script with full dataset support.

Ensures complete HH-RLHF dataset usage for fair comparison.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import wandb

# Fix PyTorch 2.6 weights_only issue for checkpoint loading
import builtins
original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)
torch.load = patched_torch_load
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
)
from datasets import load_dataset
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed as accelerate_set_seed

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from dual_head import DualHeadModel
from dual_head.dual_head_model import DualHeadConfig
from dual_head.training import (
    DualHeadTrainer,
    DualHeadTrainingArguments,
    DualHeadDataCollator,
    DualHeadLossConfig,
    prepare_preference_dataset,
    prepare_sft_dataset,
    create_mixed_dataset,
)

logger = get_logger(__name__)

EXPECTED_DATASET_SIZES = {
    "Anthropic/hh-rlhf": {"train": 160800, "test": 8552}
}

def validate_training_config(args):
    """Validate training configuration."""
    if args.max_train_samples is not None and args.max_train_samples < 1000:
        print(f"WARNING: max_train_samples={args.max_train_samples} is very small!")
        response = input("Use FULL dataset for fair comparison? (y/N): ")
        if response.lower() in ['y', 'yes']:
            args.max_train_samples = None
    
    expected_size = EXPECTED_DATASET_SIZES.get(args.dataset_name, {}).get("train", "unknown")
    if expected_size != "unknown":
        effective_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps
        expected_steps = (expected_size // effective_batch_size) * args.num_train_epochs
        print(f"Expected steps: {expected_steps:,}")
        if expected_steps < 100:
            print("ERROR: Expected steps too low")
            sys.exit(1)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Dual-Head model")
    
    # Model arguments
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models"
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name"
    )
    
    # Dataset arguments
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="Anthropic/hh-rlhf",
        help="The name of the dataset to use (via the datasets library)"
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use"
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default=None,
        help="A csv or a json file containing the training data"
    )
    parser.add_argument(
        "--validation_file",
        type=str,
        default=None,
        help="A csv or a json file containing the validation data"
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="For debugging - truncate training dataset"
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="For debugging - truncate validation dataset"
    )
    
    # Data processing arguments
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
        help="The maximum total sequence length"
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=8,
        help="Number of processes to use for data preprocessing"
    )
    parser.add_argument(
        "--add_bos_token",
        action="store_true",
        help="Whether to add BOS token to the beginning of sequences"
    )
    
    # Training arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written"
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Total number of training epochs to perform"
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size per GPU/TPU core/CPU for training"
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=16,
        help="Batch size per GPU/TPU core/CPU for evaluation"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Number of updates steps to accumulate before performing a backward/update pass"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="The initial learning rate for AdamW"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay for AdamW if we apply some"
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=500,
        help="Linear warmup over warmup_steps"
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="Linear warmup over warmup_ratio * total_steps"
    )
    
    # Dual-Head specific arguments
    parser.add_argument(
        "--lambda_r",
        type=float,
        default=1.0,
        help="Weight for preference loss"
    )
    parser.add_argument(
        "--lambda_g",
        type=float,
        default=0.01,
        help="Weight for gating regularization loss"
    )
    parser.add_argument(
        "--beta_r",
        type=float,
        default=1.0,
        help="Temperature parameter for reward model"
    )
    parser.add_argument(
        "--freeze_backbone",
        action="store_true",
        help="Whether to freeze the backbone model parameters"
    )
    parser.add_argument(
        "--gating_num_heads",
        type=int,
        default=8,
        help="Number of attention heads in gating mechanism"
    )
    
    # LoRA arguments
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Whether to use LoRA fine-tuning"
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=16,
        help="LoRA rank"
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha"
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,
        help="LoRA dropout"
    )
    
    # Mixed training arguments
    parser.add_argument(
        "--sft_dataset_name",
        type=str,
        default=None,
        help="SFT dataset name for mixed training"
    )
    parser.add_argument(
        "--sft_ratio",
        type=float,
        default=0.1,
        help="Ratio of SFT data in mixed training"
    )
    parser.add_argument(
        "--preference_data_ratio",
        type=float,
        default=1.0,
        help="Ratio of preference data to use"
    )
    
    # Evaluation arguments
    parser.add_argument(
        "--evaluation_strategy",
        type=str,
        default="steps",
        choices=["no", "steps", "epoch"],
        help="The evaluation strategy to use"
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=1000,
        help="Run evaluation every X steps"
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=50,
        help="Log every X updates steps"
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=1000,
        help="Save checkpoint every X updates steps"
    )
    parser.add_argument(
        "--save_strategy",
        type=str,
        default="steps",
        choices=["no", "steps", "epoch"],
        help="The checkpoint save strategy to use"
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=3,
        help="Limit the total amount of checkpoints"
    )
    parser.add_argument(
        "--load_best_model_at_end",
        action="store_true",
        help="Whether to load the best model at the end of training"
    )
    parser.add_argument(
        "--metric_for_best_model",
        type=str,
        default="eval_loss",
        help="The metric to use to compare two different models"
    )
    parser.add_argument(
        "--greater_is_better",
        action="store_true",
        help="Whether the metric_for_best_model should be maximized or not"
    )
    
    # Monitoring arguments
    parser.add_argument(
        "--report_to",
        type=str,
        default="",
        help="The list of integrations to report the results and logs to"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="dual-head-training",
        help="Wandb project name"
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="Wandb entity name"
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Wandb run name"
    )
    
    # System arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for initialization"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision instead of 32-bit"
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Whether to use bf16 (mixed) precision instead of 32-bit"
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=8,
        help="Number of subprocesses to use for data loading"
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="The path to a folder with a valid checkpoint for your model"
    )
    
    args = parser.parse_args()
    validate_training_config(args)
    return args


def setup_logging(args):
    """Setup logging configuration."""
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    # Suppress verbose transformers logging
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)


def load_datasets(args):
    """Load and prepare datasets for training."""
    print("Loading datasets...")
    
    if args.dataset_name:
        if args.dataset_name == "Anthropic/hh-rlhf":
            preference_dataset = load_dataset(args.dataset_name, split="train")
        else:
            preference_dataset = load_dataset(
                args.dataset_name, args.dataset_config_name, split="train"
            )
    elif args.train_file:
        data_files = {"train": args.train_file}
        if args.validation_file:
            data_files["validation"] = args.validation_file
        preference_dataset = load_dataset("json", data_files=data_files, split="train")
    else:
        raise ValueError("Must provide either dataset_name or train_file")
    
    sft_dataset = None
    if args.sft_dataset_name:
        print(f"Loading SFT dataset: {args.sft_dataset_name}")
        sft_dataset = load_dataset(args.sft_dataset_name, split="train")
    
    # Validate dataset size
    original_size = len(preference_dataset)
    expected_size = EXPECTED_DATASET_SIZES.get(args.dataset_name, {}).get("train", "unknown")
    print(f"Dataset size: {original_size:,} (expected: {expected_size})")
    
    if args.max_train_samples:
        print(f"WARNING: Truncating to {args.max_train_samples:,} samples")
        preference_dataset = preference_dataset.select(range(args.max_train_samples))
        if sft_dataset:
            sft_samples = int(args.max_train_samples * args.sft_ratio)
            sft_dataset = sft_dataset.select(range(min(sft_samples, len(sft_dataset))))
    
    print(f"Final dataset size: {len(preference_dataset):,}")
    return preference_dataset, sft_dataset


def prepare_datasets(preference_dataset, sft_dataset, tokenizer, args):
    """Prepare datasets for training."""
    print("Preparing datasets...")
    
    # Prepare preference dataset
    processed_preference_dataset = prepare_preference_dataset(
        preference_dataset,
        tokenizer,
        max_seq_length=args.max_seq_length,
        add_bos=args.add_bos_token,
        num_proc=args.preprocessing_num_workers
    )
    
    # Prepare SFT dataset if available
    processed_sft_dataset = None
    if sft_dataset:
        processed_sft_dataset = prepare_sft_dataset(
            sft_dataset,
            tokenizer,
            max_seq_length=args.max_seq_length,
            num_proc=args.preprocessing_num_workers
        )
    
    # Create mixed dataset if both are available
    if processed_sft_dataset:
        train_dataset = create_mixed_dataset(
            processed_sft_dataset,
            processed_preference_dataset,
            sft_ratio=args.sft_ratio
        )
        print(f"Mixed dataset size: {len(train_dataset):,}")
    else:
        train_dataset = processed_preference_dataset
    
    # Create validation dataset (use small subset of training data)
    val_size = min(1000, len(train_dataset) // 10)
    eval_dataset = train_dataset.select(range(val_size))
    
    print(f"Training dataset size: {len(train_dataset):,}")
    print(f"Evaluation dataset size: {len(eval_dataset):,}")
    
    return train_dataset, eval_dataset


def create_model_and_tokenizer(args):
    """Create Dual-Head model and tokenizer."""
    print("Creating model and tokenizer...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        trust_remote_code=True,
    )
    
    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create Dual-Head configuration
    dual_head_config = DualHeadConfig(
        backbone_name_or_path=args.model_name_or_path,
        freeze_backbone=args.freeze_backbone,
        
        # Head configuration
        lm_bias=False,
        rm_bias=False,
        
        # Gating configuration
        gating_num_heads=args.gating_num_heads,
        
        # Training configuration
        lambda_r=args.lambda_r,
        lambda_g=args.lambda_g,
        beta_r=args.beta_r,
    )
    
    # Create Dual-Head model
    model = DualHeadModel(dual_head_config)
    
    param_stats = model.module.get_parameter_statistics() if hasattr(model, 'module') else model.get_parameter_statistics()
    print("Model parameters:")
    print(f"  Total: {param_stats['total_params']:,}")
    print(f"  Trainable: {param_stats['total_trainable']:,} ({param_stats['trainable_percentage']:.1f}%)")
    
    return model, tokenizer


def setup_training_arguments(args):
    """Setup training arguments."""
    training_args = DualHeadTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        warmup_ratio=args.warmup_ratio,
        
        # Dual-Head specific arguments
        lambda_r=args.lambda_r,
        lambda_g=args.lambda_g,
        beta_r=args.beta_r,
        preference_data_ratio=args.preference_data_ratio,
        
        # Evaluation arguments
        evaluation_strategy=args.evaluation_strategy,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_strategy=args.save_strategy,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=args.greater_is_better,
        
        # System arguments
        fp16=args.fp16,
        bf16=args.bf16,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=True,
        
        # Monitoring arguments
        report_to=args.report_to.split(",") if args.report_to and args.report_to.strip() else [],
        run_name=args.wandb_run_name,
        
        # Other arguments
        remove_unused_columns=False,  # Important for dual-head training
        seed=args.seed,
    )
    
    return training_args


def setup_wandb(args, training_args):
    """Setup Weights & Biases logging."""
    if hasattr(training_args, 'report_to') and training_args.report_to and "wandb" in training_args.report_to:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_run_name,
                config={
                    "model_name": args.model_name_or_path,
                    "dataset_name": args.dataset_name,
                    "learning_rate": args.learning_rate,
                    "lambda_r": args.lambda_r,
                    "lambda_g": args.lambda_g,
                    "beta_r": args.beta_r,
                }
            )
        except Exception as e:
            print(f"Warning: Failed to initialize wandb: {e}")


def main():
    """Main training function."""
    print("Starting Dual-Head Training")
    print("=" * 40)
    
    args = parse_args()
    
    # Setup logging
    setup_logging(args)
    
    # Set seed for reproducibility
    set_seed(args.seed)
    accelerate_set_seed(args.seed)
    
    print(f"Random seed: {args.seed}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save arguments
    with open(os.path.join(args.output_dir, "training_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    
    # Load datasets
    preference_dataset, sft_dataset = load_datasets(args)
    
    # Create model and tokenizer
    model, tokenizer = create_model_and_tokenizer(args)
    
    # Prepare datasets
    train_dataset, eval_dataset = prepare_datasets(
        preference_dataset, sft_dataset, tokenizer, args
    )
    
    # Setup training arguments
    training_args = setup_training_arguments(args)
    
    # Setup wandb
    setup_wandb(args, training_args)
    
    # Create data collator
    data_collator = DualHeadDataCollator(
        tokenizer=tokenizer,
        max_length=args.max_seq_length,
        padding=True,
        return_tensors="pt"
    )
    
    # Create loss configuration
    loss_config = DualHeadLossConfig(
        lambda_r=args.lambda_r,
        lambda_g=args.lambda_g,
        beta_r=args.beta_r,
    )
    
    # Create trainer
    trainer = DualHeadTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        loss_config=loss_config,
    )
    
    effective_batch_size = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
    expected_steps = (len(train_dataset) // effective_batch_size) * training_args.num_train_epochs
    print(f"\nTraining setup:")
    print(f"  Examples: {len(train_dataset):,}")
    print(f"  Expected steps: {expected_steps:,}")
    
    print("\nStarting training (expected: 2-4 hours)...")
    
    if args.resume_from_checkpoint:
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    else:
        trainer.train()
    
    # Save final model
    trainer.save_model()
    
    final_stats = {
        "parameter_statistics": model.get_parameter_statistics(),
        "training_completed": True,
        "final_training_state": {
            "global_step": trainer.state.global_step,
            "epoch": trainer.state.epoch,
        },
        "gating_stats_history": getattr(trainer, 'gating_stats_history', []),
        "preference_accuracy_history": getattr(trainer, 'preference_accuracy_history', []),
    }
    
    with open(os.path.join(args.output_dir, "final_stats.json"), "w") as f:
        json.dump(final_stats, f, indent=2, default=str)
    
    print(f"\nTraining completed!")
    print(f"Steps: {trainer.state.global_step:,}, Epochs: {trainer.state.epoch:.2f}")
    print(f"Model saved to: {args.output_dir}")
    
    if trainer.state.global_step < 100:
        print("WARNING: Very few training steps - check configuration!")
    
    if hasattr(training_args, 'report_to') and "wandb" in training_args.report_to:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()