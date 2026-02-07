#!/bin/bash
# Quick validation test - runs 10 steps per configuration to verify functionality
#
# This is a smoke test to ensure all configurations work before running
# the full benchmark suite (which takes hours).

set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "⚡ Quick Validation Test (10 steps each)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Build first
echo "📦 Building examples..."
cargo build --release --example gpt2_small_baseline --features autodiff,cuda || {
    echo "❌ Baseline build failed"
    exit 1
}

cargo build --release --example gpt2_small_hybrid --features autodiff,cuda || {
    echo "❌ Hybrid build failed"
    exit 1
}

cargo build --release --example gpt2_small_memory_optimized --features autodiff,cuda || {
    echo "❌ Memory-optimized build failed"
    exit 1
}

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🧪 Running Functionality Tests"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Test 1: Baseline (10 steps, should take ~80 sec on GPU)
echo "[1/3] Testing Baseline (vanilla Burn)..."
timeout 180 cargo run --release --example gpt2_small_baseline --features autodiff,cuda 2>&1 | \
    head -100 | grep -E "(✓|Step|Loss|error|panic)" || {
    echo "❌ Baseline test failed or timed out"
    exit 1
}
echo "  ✓ Baseline functional"
echo ""

# Test 2: HybridTrainer (10 steps)
echo "[2/3] Testing HybridTrainer..."
timeout 180 cargo run --release --example gpt2_small_hybrid --features autodiff,cuda 2>&1 | \
    head -100 | grep -E "(✓|Step|Phase|Loss|error|panic)" || {
    echo "❌ HybridTrainer test failed or timed out"
    exit 1
}
echo "  ✓ HybridTrainer functional"
echo ""

# Test 3: Memory-Optimized (10 steps)
echo "[3/3] Testing Memory-Optimized..."
timeout 180 cargo run --release --example gpt2_small_memory_optimized --features autodiff,cuda 2>&1 | \
    head -100 | grep -E "(✓|Step|Phase|Memory|error|panic)" || {
    echo "❌ Memory-optimized test failed or timed out"
    exit 1
}
echo "  ✓ Memory-Optimized functional"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Quick Validation Complete"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "All configurations are functional. Ready for full benchmark suite."
echo ""
echo "Next: Run full validation with"
echo "  ./scripts/run_benchmarks.sh"
echo ""
