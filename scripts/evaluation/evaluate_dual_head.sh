#!/bin/bash

# Dual-Head Model Evaluation Script
# This script provides easy evaluation of Dual-Head models with predefined configurations

set -e

# Default values
MODEL_PATH=""
OUTPUT_DIR="./eval_results"
EVAL_COMPONENTS="all"
MAX_SAMPLES=500
BATCH_SIZE=4
DEVICE="auto"
BASELINE_MODELS=""
BASELINE_NAMES=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --eval_components)
            EVAL_COMPONENTS="$2"
            shift 2
            ;;
        --max_samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --baseline_models)
            BASELINE_MODELS="$2"
            shift 2
            ;;
        --baseline_names)
            BASELINE_NAMES="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --model_path MODEL_PATH       Path to Dual-Head model (required)"
            echo "  --output_dir OUTPUT_DIR       Output directory (default: ./eval_results)"
            echo "  --eval_components COMPONENTS  Evaluation components (all, hh_rlhf, multi_objective, efficiency, pairwise)"
            echo "  --max_samples MAX_SAMPLES     Maximum samples to evaluate (default: 500)"
            echo "  --batch_size BATCH_SIZE       Batch size (default: 4)"
            echo "  --device DEVICE               Device (auto, cuda, cpu) (default: auto)"
            echo "  --baseline_models MODELS      Space-separated paths to baseline models"
            echo "  --baseline_names NAMES        Space-separated names for baseline models"
            echo "  --help                        Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$MODEL_PATH" ]; then
    echo "Error: --model_path is required"
    exit 1
fi

if [ ! -d "$MODEL_PATH" ] && [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model path does not exist: $MODEL_PATH"
    exit 1
fi

# Display configuration
echo "==============================================="
echo "Dual-Head Model Evaluation"
echo "==============================================="
echo "Model Path: $MODEL_PATH"
echo "Output Directory: $OUTPUT_DIR"
echo "Evaluation Components: $EVAL_COMPONENTS"
echo "Max Samples: $MAX_SAMPLES"
echo "Batch Size: $BATCH_SIZE"
echo "Device: $DEVICE"
if [ -n "$BASELINE_MODELS" ]; then
    echo "Baseline Models: $BASELINE_MODELS"
    echo "Baseline Names: $BASELINE_NAMES"
fi
echo "==============================================="

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Set Python path
export PYTHONPATH="${PYTHONPATH}:$(dirname $(dirname $(realpath $0)))/src"

# Build evaluation command
EVAL_CMD="python $(dirname $0)/run_evaluation.py \
    --model_path \"$MODEL_PATH\" \
    --output_dir \"$OUTPUT_DIR\" \
    --max_samples $MAX_SAMPLES \
    --batch_size $BATCH_SIZE \
    --device $DEVICE \
    --generate_plots \
    --generate_report"

# Add evaluation components
case $EVAL_COMPONENTS in
    "all")
        EVAL_CMD="$EVAL_CMD --eval_all"
        ;;
    "hh_rlhf")
        EVAL_CMD="$EVAL_CMD --eval_hh_rlhf"
        ;;
    "multi_objective")
        EVAL_CMD="$EVAL_CMD --eval_multi_objective"
        ;;
    "efficiency")
        EVAL_CMD="$EVAL_CMD --eval_efficiency"
        ;;
    "pairwise")
        EVAL_CMD="$EVAL_CMD --eval_pairwise"
        ;;
    *)
        # Parse comma-separated components
        IFS=',' read -ra COMPONENTS <<< "$EVAL_COMPONENTS"
        for component in "${COMPONENTS[@]}"; do
            case $component in
                "hh_rlhf")
                    EVAL_CMD="$EVAL_CMD --eval_hh_rlhf"
                    ;;
                "multi_objective")
                    EVAL_CMD="$EVAL_CMD --eval_multi_objective"
                    ;;
                "efficiency")
                    EVAL_CMD="$EVAL_CMD --eval_efficiency"
                    ;;
                "pairwise")
                    EVAL_CMD="$EVAL_CMD --eval_pairwise"
                    ;;
                *)
                    echo "Warning: Unknown evaluation component: $component"
                    ;;
            esac
        done
        ;;
esac

# Add baseline models if provided
if [ -n "$BASELINE_MODELS" ]; then
    EVAL_CMD="$EVAL_CMD --baseline_models $BASELINE_MODELS"
    if [ -n "$BASELINE_NAMES" ]; then
        EVAL_CMD="$EVAL_CMD --baseline_names $BASELINE_NAMES"
    fi
fi

echo "Evaluation Command:"
echo "$EVAL_CMD"
echo ""

# Run evaluation
echo "Starting evaluation..."
eval $EVAL_CMD

# Check if evaluation completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "==============================================="
    echo "Evaluation completed successfully!"
    echo "==============================================="
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    echo "Generated files:"
    find "$OUTPUT_DIR" -name "*.json" -o -name "*.md" -o -name "*.png" | sort
    echo ""
    
    # Display quick summary if report exists
    if [ -f "$OUTPUT_DIR/evaluation_report.md" ]; then
        echo "Quick Summary:"
        echo "=============="
        head -n 20 "$OUTPUT_DIR/evaluation_report.md"
        echo ""
        echo "Full report available at: $OUTPUT_DIR/evaluation_report.md"
    fi
else
    echo ""
    echo "==============================================="
    echo "Evaluation failed!"
    echo "==============================================="
    echo "Check the logs for details."
    exit 1
fi