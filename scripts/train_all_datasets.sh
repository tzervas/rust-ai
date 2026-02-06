#!/bin/bash
# Train Tritter 100M model with all 371GB of datasets
#
# Dataset composition:
# - Code (GitHub):        204GB at /data/datasets/tritter/pretrain/code/
# - IaC Enriched:         93MB  at /data/datasets/tritter/iac/combined/
# - Alignment:           764MB  at /data/datasets/tritter/alignment/
# - Instruction:          13GB  at /data/datasets/tritter/instruction/
#
# Usage:
#   ./scripts/train_all_datasets.sh [--steps N] [--blend sequential|roundrobin|weighted]

set -euo pipefail

# Default values
STEPS="${1:-100000}"
BLEND_STRATEGY="${2:-roundrobin}"
RUNS_DIR="./runs"
BATCH_SIZE=4
SEQ_LENGTH=2048
LEARNING_RATE="3e-4"

# Dataset paths (adjust these for your system)
CODE_DATASET="/data/datasets/tritter/pretrain/code/"
IAC_DATASET="/data/datasets/tritter/iac/combined/"
ALIGNMENT_DATASET="/data/datasets/tritter/alignment/"
INSTRUCTION_DATASET="/data/datasets/tritter/instruction/"

echo "=== Tritter 100M Training with All Datasets ==="
echo ""
echo "Configuration:"
echo "  Steps:          $STEPS"
echo "  Blend Strategy: $BLEND_STRATEGY"
echo "  Batch Size:     $BATCH_SIZE"
echo "  Sequence Length: $SEQ_LENGTH"
echo "  Learning Rate:  $LEARNING_RATE"
echo ""
echo "Datasets:"

# Check which datasets exist
DATASET_ARGS=""
total_size=0

check_dataset() {
    local path="$1"
    local name="$2"
    if [ -e "$path" ]; then
        if [ -d "$path" ]; then
            size=$(du -sh "$path" 2>/dev/null | cut -f1 || echo "?")
        else
            size=$(ls -lh "$path" 2>/dev/null | awk '{print $5}' || echo "?")
        fi
        echo "  [OK] $name: $path ($size)"
        DATASET_ARGS="$DATASET_ARGS --dataset $path"
    else
        echo "  [--] $name: $path (not found, skipping)"
    fi
}

check_dataset "$CODE_DATASET" "Code (GitHub)"
check_dataset "$INSTRUCTION_DATASET" "Instruction"
check_dataset "$ALIGNMENT_DATASET" "Alignment"
check_dataset "$IAC_DATASET" "IaC Enriched"

if [ -z "$DATASET_ARGS" ]; then
    echo ""
    echo "ERROR: No datasets found. Please check the dataset paths."
    echo "       You can modify this script to point to your data locations."
    exit 1
fi

echo ""
echo "Starting training..."
echo ""

# Build command
CMD="cargo run -p training-tools --release --features cuda --bin train -- \
    --start 100m \
    --end 100m \
    --steps $STEPS \
    --batch-size $BATCH_SIZE \
    --seq-length $SEQ_LENGTH \
    --learning-rate $LEARNING_RATE \
    --blend-strategy $BLEND_STRATEGY \
    --runs-dir $RUNS_DIR \
    --checkpoint-every 5000 \
    --log-every 100 \
    $DATASET_ARGS"

echo "Command: $CMD"
echo ""

# Execute
exec $CMD
