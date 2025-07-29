#!/bin/bash

# Dual-Head Training Script
# This script provides optimized hyperparameters for training Dual-Head models

set -e

# Default values
MODEL_NAME="meta-llama/Llama-2-7b-hf"
DATASET_NAME="Anthropic/hh-rlhf"
OUTPUT_DIR="./outputs/dual_head_llama2_7b"
NUM_EPOCHS=3
BATCH_SIZE=4
LEARNING_RATE=5e-5
MAX_SEQ_LENGTH=2048

# Dual-Head specific parameters
LAMBDA_R=1.0
LAMBDA_G=0.01
BETA_R=1.0
FREEZE_BACKBONE=true
GATING_NUM_HEADS=8

# System parameters
SEED=42
GRADIENT_ACCUMULATION_STEPS=4
EVAL_STEPS=500
SAVE_STEPS=1000
LOGGING_STEPS=50

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_name_or_path)
            MODEL_NAME="$2"
            shift 2
            ;;
        --dataset_name)
            DATASET_NAME="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --num_train_epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --per_device_train_batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --max_seq_length)
            MAX_SEQ_LENGTH="$2"
            shift 2
            ;;
        --lambda_r)
            LAMBDA_R="$2"
            shift 2
            ;;
        --lambda_g)
            LAMBDA_G="$2"
            shift 2
            ;;
        --beta_r)
            BETA_R="$2"
            shift 2
            ;;
        --no_freeze_backbone)
            FREEZE_BACKBONE=false
            shift
            ;;
        --gating_num_heads)
            GATING_NUM_HEADS="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --gradient_accumulation_steps)
            GRADIENT_ACCUMULATION_STEPS="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --model_name_or_path MODEL_NAME    Model name or path (default: $MODEL_NAME)"
            echo "  --dataset_name DATASET_NAME        Dataset name (default: $DATASET_NAME)"
            echo "  --output_dir OUTPUT_DIR             Output directory (default: $OUTPUT_DIR)"
            echo "  --num_train_epochs NUM_EPOCHS       Number of training epochs (default: $NUM_EPOCHS)"
            echo "  --per_device_train_batch_size BATCH_SIZE  Batch size per device (default: $BATCH_SIZE)"
            echo "  --learning_rate LEARNING_RATE       Learning rate (default: $LEARNING_RATE)"
            echo "  --max_seq_length MAX_SEQ_LENGTH     Maximum sequence length (default: $MAX_SEQ_LENGTH)"
            echo "  --lambda_r LAMBDA_R                 Preference loss weight (default: $LAMBDA_R)"
            echo "  --lambda_g LAMBDA_G                 Gating regularization weight (default: $LAMBDA_G)"
            echo "  --beta_r BETA_R                     Reward model temperature (default: $BETA_R)"
            echo "  --no_freeze_backbone                Don't freeze backbone parameters"
            echo "  --gating_num_heads NUM_HEADS        Number of gating attention heads (default: $GATING_NUM_HEADS)"
            echo "  --seed SEED                         Random seed (default: $SEED)"
            echo "  --gradient_accumulation_steps STEPS  Gradient accumulation steps (default: $GRADIENT_ACCUMULATION_STEPS)"
            echo "  --help                              Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

echo "==============================================="
echo "Starting Dual-Head Model Training"
echo "==============================================="
echo "Model: $MODEL_NAME"
echo "Dataset: $DATASET_NAME"
echo "Output Directory: $OUTPUT_DIR"
echo "Training Configuration:"
echo "  - Epochs: $NUM_EPOCHS"
echo "  - Batch Size: $BATCH_SIZE"
echo "  - Learning Rate: $LEARNING_RATE"
echo "  - Max Sequence Length: $MAX_SEQ_LENGTH"
echo "  - Gradient Accumulation Steps: $GRADIENT_ACCUMULATION_STEPS"
echo ""
echo "Dual-Head Configuration:"
echo "  - Lambda R (preference weight): $LAMBDA_R"
echo "  - Lambda G (gating weight): $LAMBDA_G"
echo "  - Beta R (reward temperature): $BETA_R"
echo "  - Freeze Backbone: $FREEZE_BACKBONE"
echo "  - Gating Attention Heads: $GATING_NUM_HEADS"
echo ""
echo "System Configuration:"
echo "  - Seed: $SEED"
echo "  - Eval Steps: $EVAL_STEPS"
echo "  - Save Steps: $SAVE_STEPS"
echo "  - Logging Steps: $LOGGING_STEPS"
echo "==============================================="

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Set up Python path
export PYTHONPATH="${PYTHONPATH}:$(dirname $(dirname $(realpath $0)))/src"

# Prepare training command
TRAINING_CMD="python $(dirname $0)/train_dual_head.py \
    --model_name_or_path $MODEL_NAME \
    --dataset_name $DATASET_NAME \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --learning_rate $LEARNING_RATE \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --max_seq_length $MAX_SEQ_LENGTH \
    --lambda_r $LAMBDA_R \
    --lambda_g $LAMBDA_G \
    --beta_r $BETA_R \
    --gating_num_heads $GATING_NUM_HEADS \
    --evaluation_strategy steps \
    --eval_steps $EVAL_STEPS \
    --save_steps $SAVE_STEPS \
    --logging_steps $LOGGING_STEPS \
    --save_total_limit 3 \
    --seed $SEED \
    --bf16 \
    --dataloader_num_workers 4 \
    --preprocessing_num_workers 4 \
    --report_to wandb \
    --wandb_project dual-head-training \
    --wandb_run_name dual_head_$(basename $MODEL_NAME)_$(date +%Y%m%d_%H%M%S)"

# Add freeze backbone flag if needed
if [ "$FREEZE_BACKBONE" = true ]; then
    TRAINING_CMD="$TRAINING_CMD --freeze_backbone"
fi

echo "Training Command:"
echo "$TRAINING_CMD"
echo ""

# Check if accelerate is available for multi-GPU training
if command -v accelerate &> /dev/null; then
    echo "Accelerate detected. Checking for multi-GPU setup..."
    
    # Check number of GPUs
    NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l || echo "0")
    
    if [ "$NUM_GPUS" -gt 1 ]; then
        echo "Multi-GPU training with $NUM_GPUS GPUs"
        
        # Create or update accelerate config
        ACCELERATE_CONFIG_DIR="$HOME/.cache/huggingface/accelerate"
        mkdir -p "$ACCELERATE_CONFIG_DIR"
        
        cat > "$ACCELERATE_CONFIG_DIR/default_config.yaml" << EOF
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: $NUM_GPUS
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF
        
        echo "Accelerate config created at $ACCELERATE_CONFIG_DIR/default_config.yaml"
        
        # Launch with accelerate
        exec accelerate launch --config_file "$ACCELERATE_CONFIG_DIR/default_config.yaml" $TRAINING_CMD
    else
        echo "Single GPU training"
        exec $TRAINING_CMD
    fi
else
    echo "Accelerate not found. Running single GPU training..."
    exec $TRAINING_CMD
fi