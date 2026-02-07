#!/bin/bash
# VRAM Management Validation Script
#
# This script runs validation tests to verify VRAM management is working correctly.
# Requires: CUDA-capable GPU with nvidia-smi

set -e

echo "========================================="
echo "VRAM Management Validation"
echo "========================================="
echo ""

# Check for CUDA
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ nvidia-smi not found. GPU validation requires CUDA."
    echo "   Running unit tests only..."
    echo ""
    cargo test --test vram_validation
    exit 0
fi

# Display GPU info
echo "📊 GPU Information:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Run unit tests first
echo "1️⃣ Running VramManager unit tests..."
cargo test --test vram_validation --release
echo "✅ Unit tests passed"
echo ""

# Check if GPT-2 example exists
if [ ! -f "examples/gpt2_small_hybrid.rs" ]; then
    echo "⚠️  GPT-2 example not found. Skipping GPU validation."
    exit 0
fi

echo "2️⃣ Running 50-step GPT-2 Small validation..."
echo "   This will take 2-5 minutes depending on GPU..."
echo ""

# Get initial VRAM
INITIAL_VRAM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
echo "   Initial VRAM: ${INITIAL_VRAM} MB"

# Run GPT-2 example with monitoring
echo "   Starting training..."
cargo run --release --example gpt2_small_hybrid 2>&1 | tee /tmp/vram_validation.log &
TRAIN_PID=$!

# Monitor VRAM every 5 seconds
PEAK_VRAM=$INITIAL_VRAM
while kill -0 $TRAIN_PID 2>/dev/null; do
    sleep 5
    CURRENT_VRAM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    if [ "$CURRENT_VRAM" -gt "$PEAK_VRAM" ]; then
        PEAK_VRAM=$CURRENT_VRAM
    fi
    echo "   Current VRAM: ${CURRENT_VRAM} MB (Peak: ${PEAK_VRAM} MB)"
done

# Wait for training to complete
wait $TRAIN_PID
TRAIN_EXIT=$?

# Get final VRAM
FINAL_VRAM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)

echo ""
echo "========================================="
echo "Validation Results"
echo "========================================="
echo ""
echo "VRAM Usage:"
echo "  Initial: ${INITIAL_VRAM} MB"
echo "  Peak:    ${PEAK_VRAM} MB"
echo "  Final:   ${FINAL_VRAM} MB"
echo "  Growth:  $((PEAK_VRAM - INITIAL_VRAM)) MB"
echo ""

# Check against targets
TARGET_PEAK=10000  # 10 GB target
BASELINE_GROWTH=10200  # Baseline: 10.2 GB growth

ACTUAL_GROWTH=$((PEAK_VRAM - INITIAL_VRAM))

if [ "$PEAK_VRAM" -lt "$TARGET_PEAK" ]; then
    echo "✅ VRAM Peak: PASS (${PEAK_VRAM} MB < ${TARGET_PEAK} MB target)"
else
    echo "⚠️  VRAM Peak: HIGH (${PEAK_VRAM} MB > ${TARGET_PEAK} MB target)"
fi

if [ "$ACTUAL_GROWTH" -lt "$BASELINE_GROWTH" ]; then
    REDUCTION=$((100 - (ACTUAL_GROWTH * 100 / BASELINE_GROWTH)))
    echo "✅ VRAM Growth: IMPROVED (${REDUCTION}% reduction vs baseline)"
else
    echo "❌ VRAM Growth: WORSE than baseline"
fi

# Check log for VRAM events
echo ""
echo "VRAM Events in Training Log:"
grep -E "VRAM|Emergency|Cleanup|Phase transition" /tmp/vram_validation.log | head -20

echo ""
if [ "$TRAIN_EXIT" -eq 0 ] && [ "$PEAK_VRAM" -lt "$TARGET_PEAK" ]; then
    echo "✅ Overall: VALIDATION PASSED"
    exit 0
else
    echo "⚠️  Overall: Review results above"
    exit 1
fi
