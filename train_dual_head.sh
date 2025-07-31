#!/bin/bash

# Dual-Head Training Script - Optimized for H100 94GB GPU
# Efficient parameter training: 333M trainable out of 7B total (4.7%)

# Set CUDA environment
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Suppress token counting warnings (cosmetic only)
export TRANSFORMERS_VERBOSITY=error

echo "Training Dual-Head Model - H100 Optimized"
echo "Model: 7B parameters (333M trainable), HH-RLHF dataset"

# Use regular python instead of torchrun for single GPU
python scripts/training/train_dual_head.py \
    --model_name_or_path "argsearch/llama-7b-sft-float32" \
    --dataset_name "Anthropic/hh-rlhf" \
    --output_dir "./outputs/dual_head_single_h100" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --max_seq_length 2048 \
    --bf16 \
    --dataloader_num_workers 8 \
    --logging_steps 10 \
    --save_steps 500 \
    --eval_steps 500 \
    --evaluation_strategy steps \
    --save_strategy steps \
    --load_best_model_at_end \
    --metric_for_best_model "eval_loss" \
    --freeze_backbone \
    --lambda_r 1.0 \
    --lambda_g 0.01 \
    --beta_r 1.0 \
    --gating_num_heads 8 \
    --preprocessing_num_workers 8 \
    --seed 42 \
    --report_to ""