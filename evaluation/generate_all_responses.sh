#!/bin/bash

# Generate responses from all models for comparison evaluation
# Usage: ./generate_all_responses.sh [num_samples] (default: 300)

NUM_SAMPLES=${1:-300}
SCRIPTS_DIR="/home/ibel/research/Dual-Head/evaluation/scripts"
OUTPUTS_DIR="/home/ibel/research/Dual-Head/evaluation/outputs"

echo "=== Generating Responses for All Models ==="
echo "Number of samples: $NUM_SAMPLES"
echo "Scripts directory: $SCRIPTS_DIR"
echo "Outputs directory: $OUTPUTS_DIR"
echo ""

# Make sure output directory exists
mkdir -p $OUTPUTS_DIR

# Set CUDA environment
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Track start time
START_TIME=$(date +%s)

echo "=== 1/5: Generating Dual-Head responses ==="
cd /home/ibel/research/Dual-Head/evaluation
python scripts/generate_dualhead.py $NUM_SAMPLES
if [ $? -eq 0 ]; then
    echo "✅ Dual-Head generation completed"
else
    echo "❌ Dual-Head generation failed"
fi
echo ""

echo "=== 2/5: Generating DPO responses ==="
python scripts/generate_dpo.py $NUM_SAMPLES
if [ $? -eq 0 ]; then
    echo "✅ DPO generation completed"
else
    echo "❌ DPO generation failed"
fi
echo ""

echo "=== 3/5: Generating SimPO responses ==="
python scripts/generate_simpo.py $NUM_SAMPLES
if [ $? -eq 0 ]; then
    echo "✅ SimPO generation completed"
else
    echo "❌ SimPO generation failed"
fi
echo ""

# echo "=== 4/5: Generating ARGS responses ==="
# python scripts/generate_args.py $NUM_SAMPLES
# if [ $? -eq 0 ]; then
#     echo "✅ ARGS generation completed"
# else
#     echo "❌ ARGS generation failed"
# fi
# echo ""

# echo "=== 5/5: Generating GenARM responses ==="
# python scripts/generate_genarm.py $NUM_SAMPLES
# if [ $? -eq 0 ]; then
#     echo "✅ GenARM generation completed"
# else
#     echo "❌ GenARM generation failed"
# fi
# echo ""

# Calculate total time
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
HOURS=$((TOTAL_TIME / 3600))
MINUTES=$(((TOTAL_TIME % 3600) / 60))
SECONDS=$((TOTAL_TIME % 60))

echo "=== Generation Summary ==="
echo "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo ""

# List generated files
echo "Generated files:"
ls -la $OUTPUTS_DIR/*_responses_${NUM_SAMPLES}.json 2>/dev/null || echo "No response files found"
echo ""

echo "=== Ready for Evaluation ==="
echo "Next step: Run evaluation script to compare all models"
echo "Command: python scripts/evaluate_models.py"