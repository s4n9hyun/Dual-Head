#!/bin/bash

# Generate responses from Dual-Head model only
# Usage: ./generate_dualhead_only.sh [num_samples] (default: 300)

NUM_SAMPLES=${1:-300}
SCRIPTS_DIR="/home/ibel/research/Dual-Head/evaluation/scripts"
OUTPUTS_DIR="/home/ibel/research/Dual-Head/evaluation/outputs"

echo "=== Generating Dual-Head Responses Only ==="
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

echo "=== Generating Dual-Head responses ==="
cd /home/ibel/research/Dual-Head/evaluation
python scripts/generate_dualhead.py $NUM_SAMPLES
if [ $? -eq 0 ]; then
    echo "✅ Dual-Head generation completed"
else
    echo "❌ Dual-Head generation failed"
    exit 1
fi
echo ""

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
ls -la $OUTPUTS_DIR/dualhead_responses_${NUM_SAMPLES}.json 2>/dev/null || echo "No Dual-Head response file found"
echo ""

echo "=== Dual-Head Generation Complete ==="
echo "File location: $OUTPUTS_DIR/dualhead_responses_${NUM_SAMPLES}.json"