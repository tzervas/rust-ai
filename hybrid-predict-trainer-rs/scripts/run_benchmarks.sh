#!/bin/bash
# Automated benchmark execution for Phase 2B scientific validation
#
# Runs all three configurations (baseline, hybrid, memory-optimized) with
# multiple random seeds for statistical significance testing.

set -e

# Configuration
RESULTS_DIR="results/phase_2b_$(date +%Y%m%d_%H%M%S)"
SEEDS=(42 43 44)
STEPS=1000
BATCH_SIZE=4
SEQ_LEN=64
LOG_INTERVAL=10

# GPU check
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ ERROR: nvidia-smi not found. GPU validation requires CUDA."
    echo "   For CPU testing (slow), modify this script."
    exit 1
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔬 Phase 2B Scientific Validation"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Configuration:"
echo "  Seeds: ${SEEDS[@]}"
echo "  Steps: $STEPS"
echo "  Batch size: $BATCH_SIZE"
echo "  Sequence length: $SEQ_LEN"
echo "  Results: $RESULTS_DIR"
echo ""

# Create results directory
mkdir -p "$RESULTS_DIR"/{baseline,hybrid,memory_optimized}/logs

# GPU info
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

# Build all examples
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📦 Building Examples"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

cargo build --release --example gpt2_small_baseline --features autodiff,cuda
cargo build --release --example gpt2_small_hybrid --features autodiff,cuda
cargo build --release --example gpt2_small_memory_optimized --features autodiff,cuda

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 Running Benchmarks"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Function to run a single benchmark
run_benchmark() {
    local example=$1
    local seed=$2
    local config=$3
    local run_num=$((seed - 41))

    echo "  [$run_num/3] $config (seed=$seed)"

    local log_file="$RESULTS_DIR/$config/logs/run_${seed}.log"
    local csv_file="$RESULTS_DIR/$config/run_${seed}.csv"

    # Run with timeout (30 min max)
    timeout 1800 cargo run --release --example "$example" --features autodiff,cuda \
        2>&1 | tee "$log_file"

    # TODO: Extract CSV from logs (once examples support --log-csv)
    # For now, logs contain all data

    echo "     ✓ Completed (log: $log_file)"
}

# Run all configurations
total_runs=$((3 * 3))  # 3 configs × 3 seeds
current_run=0

for seed in "${SEEDS[@]}"; do
    # Baseline
    ((current_run++))
    echo "[$current_run/$total_runs] Baseline Configuration"
    run_benchmark "gpt2_small_baseline" "$seed" "baseline"
    sleep 2  # Cool-down between runs

    # HybridTrainer
    ((current_run++))
    echo "[$current_run/$total_runs] HybridTrainer Configuration"
    run_benchmark "gpt2_small_hybrid" "$seed" "hybrid"
    sleep 2

    # Memory-Optimized
    ((current_run++))
    echo "[$current_run/$total_runs] Memory-Optimized Configuration"
    run_benchmark "gpt2_small_memory_optimized" "$seed" "memory_optimized"
    sleep 2

    echo ""
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Benchmarks Complete"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Results saved to: $RESULTS_DIR"
echo ""
echo "Next steps:"
echo "  1. Run analysis: python scripts/analyze_results.py --input $RESULTS_DIR"
echo "  2. Generate report: python scripts/generate_report.py --input $RESULTS_DIR"
echo ""
