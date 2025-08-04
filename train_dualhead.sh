#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRANSFORMERS_VERBOSITY=error

LATEST_CHECKPOINT=$(find ./outputs/dual_head_full_dataset -name "checkpoint-*" -type d 2>/dev/null | sort -V | tail -1)
RESUME_ARG=""
if [ -n "$LATEST_CHECKPOINT" ]; then
    RESUME_ARG="--resume_from_checkpoint $LATEST_CHECKPOINT"
fi
mkdir -p ./outputs/dual_head_full_dataset

python scripts/training/train_dual_head_full.py \
    --model_name_or_path "argsearch/llama-7b-sft-float32" \
    --dataset_name "Anthropic/hh-rlhf" \
    --output_dir "./outputs/dual_head_full_dataset" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --max_seq_length 2048 \
    --bf16 \
    --dataloader_num_workers 8 \
    --logging_steps 50 \
    --save_steps 200 \
    --eval_steps 500 \
    --evaluation_strategy no \
    --save_strategy steps \
    --freeze_backbone \
    --lambda_r 1.0 \
    --lambda_g 0.01 \
    --beta_r 1.0 \
    --gating_num_heads 8 \
    --preprocessing_num_workers 8 \
    --seed 42 \
    --report_to "" \
    --save_total_limit 3 \
    $RESUME_ARG