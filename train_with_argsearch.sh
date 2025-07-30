#!/bin/bash

# Quick training script using argsearch model and new config
set -e

echo "Training Dual-Head with argsearch/llama-7b-sft-float32"
echo "Using configuration: configs/argsearch_llama7b_sft_hh_rlhf.yaml"

# Set Python path
export PYTHONPATH="${PYTHONPATH}:$(dirname $(realpath $0))/src"

# Train using the argsearch-specific configuration
python scripts/training/train_dual_head.py \
    --model_name_or_path "argsearch/llama-7b-sft-float32" \
    --dataset_name "Anthropic/hh-rlhf" \
    --output_dir "./outputs/dual_head_argsearch_llama7b_sft_hh_rlhf" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 16 \
    --learning_rate 5e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --max_seq_length 2048 \
    --lambda_r 1.0 \
    --lambda_g 0.01 \
    --beta_r 1.0 \
    --freeze_backbone \
    --gating_num_heads 8 \
    --evaluation_strategy steps \
    --eval_steps 500 \
    --save_steps 500 \
    --logging_steps 10 \
    --save_total_limit 3 \
    --seed 42 \
    --bf16 \
    --dataloader_num_workers 4 \
    --preprocessing_num_workers 4 \
    --report_to wandb \
    --wandb_project dual-head-iclr2026 \
    --wandb_run_name argsearch_llama7b_sft_hh_rlhf_main

echo "Training completed! Model saved to ./outputs/dual_head_argsearch_llama7b_sft_hh_rlhf"