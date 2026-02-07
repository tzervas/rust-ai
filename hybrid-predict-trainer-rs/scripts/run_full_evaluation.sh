#!/bin/bash
# Master orchestration script for Phase 2B scientific validation
#
# Executes the complete evaluation pipeline:
# 1. Quick validation (smoke tests)
# 2. Full benchmark suite (3 configs × 3 seeds × 1000 steps)
# 3. Statistical analysis
# 4. Report generation
#
# Total estimated time: 6-8 hours on RTX 5080

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔬 Phase 2B Complete Scientific Evaluation"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Repository: $REPO_ROOT"
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Check prerequisites
echo "Checking prerequisites..."

if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ ERROR: nvidia-smi not found. GPU required for validation."
    exit 1
fi

if ! command -v python3 &> /dev/null; then
    echo "❌ ERROR: python3 not found. Required for analysis scripts."
    exit 1
fi

# Check Python dependencies
python3 -c "import numpy, pandas, scipy" 2>/dev/null || {
    echo "❌ ERROR: Missing Python dependencies. Install with:"
    echo "   pip install numpy pandas scipy matplotlib"
    exit 1
}

echo "  ✓ GPU available"
echo "  ✓ Python3 available"
echo "  ✓ Required Python packages installed"
echo ""

# Stage 1: Quick Validation
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Stage 1: Quick Validation (10 steps each)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

bash "$SCRIPT_DIR/run_quick_validation.sh" || {
    echo "❌ Quick validation failed. Aborting full evaluation."
    exit 1
}

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Stage 2: Full Benchmark Suite (1000 steps × 9 runs)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "⚠️  This will take 4-6 hours. Press Ctrl+C to abort."
echo ""
sleep 5

bash "$SCRIPT_DIR/run_benchmarks.sh" || {
    echo "❌ Benchmark suite failed."
    exit 1
}

# Find the most recent results directory
RESULTS_DIR=$(ls -td results/phase_2b_* 2>/dev/null | head -1)

if [ -z "$RESULTS_DIR" ]; then
    echo "❌ ERROR: No results directory found."
    exit 1
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Stage 3: Statistical Analysis"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

python3 "$SCRIPT_DIR/analyze_results.py" \
    --input "$RESULTS_DIR" \
    --output report || {
    echo "❌ Statistical analysis failed."
    exit 1
}

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Phase 2B Evaluation Complete"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Results:"
echo "  Raw data: $RESULTS_DIR"
echo "  Analysis: report/summary.json"
echo ""
echo "Review the analysis output above for pass/fail status."
echo ""
echo "Next steps:"
echo "  1. Review report/summary.json for detailed metrics"
echo "  2. If goals met, proceed to GPT-2 Medium (350M params)"
echo "  3. If goals not met, analyze failure modes and iterate"
echo ""
