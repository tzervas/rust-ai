#!/bin/bash
# GPU Test Automation Script for axolotl-rs
#
# Provides tiered GPU testing from quick sanity checks to full validation.
#
# Usage:
#   ./scripts/gpu-test.sh [command] [options]
#
# Commands:
#   quick      - 10 steps with SmolLM2-135M (~1 minute)
#   convergence - 100 steps with SmolLM2-135M (~5 minutes)
#   memory     - 50 steps with TinyLlama-1.1B (~10 minutes)
#   extended   - 500 steps with TinyLlama-1.1B (~30 minutes)
#   full       - 1000 steps with LLaMA-7B (~2 hours)
#   all        - Run quick → convergence → memory in sequence
#
# Options:
#   --check-vram   - Check VRAM before running (requires nvidia-smi)
#   --verbose      - Show detailed test output
#   --continue     - Continue on test failure
#
# Examples:
#   ./scripts/gpu-test.sh quick
#   ./scripts/gpu-test.sh memory --check-vram
#   ./scripts/gpu-test.sh all --verbose
#

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
FEATURES="qlora,cuda"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Parse arguments
COMMAND="${1:-help}"
CHECK_VRAM=false
VERBOSE=false
CONTINUE_ON_ERROR=false

while [[ $# -gt 1 ]]; do
  case "$2" in
    --check-vram) CHECK_VRAM=true ;;
    --verbose) VERBOSE=true ;;
    --continue) CONTINUE_ON_ERROR=true ;;
    *) echo "Unknown option: $2"; exit 1 ;;
  esac
  shift
done

# =============================================================================
# Helper Functions
# =============================================================================

log_info() {
  echo -e "${BLUE}ℹ${NC} $*"
}

log_success() {
  echo -e "${GREEN}✓${NC} $*"
}

log_warning() {
  echo -e "${YELLOW}⚠${NC} $*"
}

log_error() {
  echo -e "${RED}✗${NC} $*"
}

# Check if CUDA is available
check_cuda() {
  if command -v nvidia-smi &> /dev/null; then
    log_success "CUDA detected via nvidia-smi"
    nvidia-smi --query-gpu=name --format=csv,noheader | head -1
  else
    log_warning "nvidia-smi not found - assuming CUDA is available"
  fi
}

# Check GPU VRAM and print warning if insufficient
check_vram() {
  if ! command -v nvidia-smi &> /dev/null; then
    log_warning "Cannot check VRAM without nvidia-smi"
    return 0
  fi

  local free_vram_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  local required_mb=$1

  log_info "GPU VRAM Status: ${free_vram_mb} MB free"
  
  if (( free_vram_mb < required_mb )); then
    log_error "Insufficient VRAM: need ${required_mb} MB, have ${free_vram_mb} MB"
    return 1
  else
    local padding=$((required_mb - free_vram_mb))
    log_success "Sufficient VRAM (${padding} MB buffer)"
    return 0
  fi
}

# Run a test and measure time
run_test() {
  local test_name=$1
  local test_path=$2
  local required_vram_mb=${3:-256}

  echo ""
  log_info "═══════════════════════════════════════════════════════════"
  log_info "Starting: $test_name"
  log_info "═══════════════════════════════════════════════════════════"

  if $CHECK_VRAM; then
    check_vram "$required_vram_mb" || {
      if $CONTINUE_ON_ERROR; then
        log_warning "Skipping due to insufficient VRAM"
        return 1
      else
        log_error "Insufficient VRAM to run test"
        exit 1
      fi
    }
  fi

  local start_time=$(date +%s)
  local cmd="cd '$PROJECT_ROOT' && cargo test --features '$FEATURES' -- --ignored $test_path"

  if $VERBOSE; then
    log_info "Running: $cmd"
    bash -c "$cmd" || {
      if $CONTINUE_ON_ERROR; then
        log_error "$test_name failed (continuing...)"
        return 1
      else
        log_error "$test_name failed"
        exit 1
      fi
    }
  else
    bash -c "$cmd" > /tmp/axolotl_gpu_test.log 2>&1 || {
      if $CONTINUE_ON_ERROR; then
        log_error "$test_name failed (continuing...)"
        tail -20 /tmp/axolotl_gpu_test.log
        return 1
      else
        log_error "$test_name failed"
        echo "--- Test output (last 50 lines) ---"
        tail -50 /tmp/axolotl_gpu_test.log
        exit 1
      fi
    }
  fi

  local end_time=$(date +%s)
  local elapsed=$((end_time - start_time))
  
  log_success "$test_name passed in ${elapsed}s"
  return 0
}

# =============================================================================
# Test Definitions
# =============================================================================

test_quick() {
  run_test "GPU Quick Iteration (10 steps)" "test_gpu_quick_iteration" 256
}

test_convergence() {
  run_test "GPU Loss Convergence (100 steps)" "test_gpu_loss_convergence_100_steps" 256
}

test_memory() {
  run_test "GPU TinyLlama Memory Validation (50 steps)" "test_gpu_tinyllama_memory_validation" 2048
}

test_extended() {
  run_test "GPU TinyLlama Extended Training (500 steps)" "test_gpu_tinyllama_extended_training" 2048
}

test_full() {
  run_test "GPU LLaMA-7B Full Validation (1000 steps)" "test_gpu_llama7b_full_validation" 12288
}

test_all() {
  log_info "Running full GPU test suite: quick → convergence → memory"
  test_quick || true
  test_convergence || true
  test_memory || true
}

# =============================================================================
# Help and Info
# =============================================================================

show_help() {
  cat << 'EOF'
axolotl-rs GPU Test Automation

Usage: ./scripts/gpu-test.sh [command] [options]

Commands:
  quick      Quick iteration test (SmolLM2-135M, 10 steps, ~1 min)
             Validates CUDA device initialization and basic training
             
  convergence
             Loss convergence test (SmolLM2-135M, 100 steps, ~5 min)
             Validates gradients flow and loss decreases
             
  memory     Memory validation test (TinyLlama-1.1B, 50 steps, ~10 min)
             Validates 1.1B model fits in ~2GB VRAM
             
  extended   Extended training test (TinyLlama-1.1B, 500 steps, ~30 min)
             Validates sustained training and memory stability
             
  full       Full validation test (LLaMA-7B, 1000 steps, ~2 hours)
             Production-ready fine-tuning validation
             
  all        Run quick → convergence → memory in sequence
             Good for CI/CD pipelines

Options:
  --check-vram  Verify sufficient VRAM before running (requires nvidia-smi)
  --verbose     Show full test output (otherwise shows summary only)
  --continue    Continue on test failure instead of exiting

Environment:
  HF_TOKEN      HuggingFace API token for private models (optional)
  HF_HOME       HuggingFace cache directory (default: ~/.cache/huggingface)

Requirements:
  - CUDA Toolkit 12.0+ with cuDNN
  - NVIDIA GPU (RTX 3060+ recommended for extended tests)
  - Rust 1.70+ with cargo

Model Setup:
  # SmolLM2-135M
  huggingface-cli download HuggingFaceTB/SmolLM2-135M
  
  # TinyLlama-1.1B
  huggingface-cli download TinyLlama/TinyLlama-1.1B-Chat-v1.0
  
  # LLaMA-7B (requires access)
  huggingface-cli download meta-llama/Llama-2-7b-hf

Examples:
  # Quick sanity check
  ./scripts/gpu-test.sh quick
  
  # Memory validation with VRAM check
  ./scripts/gpu-test.sh memory --check-vram
  
  # Full test suite with verbose output
  ./scripts/gpu-test.sh all --verbose
  
  # Individual extended test
  ./scripts/gpu-test.sh extended --check-vram
  
  # Full validation (takes ~2 hours)
  ./scripts/gpu-test.sh full --check-vram --verbose

Exit Codes:
  0  - All tests passed
  1  - Test failed (or insufficient VRAM with --check-vram)
  2  - Invalid command or missing dependencies

For more information, see: tests/gpu_training.rs
EOF
}

show_info() {
  echo ""
  log_info "axolotl-rs GPU Test Suite"
  echo ""
  
  if check_cuda; then
    echo ""
    log_info "Test Tiers Available:"
    echo "  1. Quick (10 steps, 1 min) - SmolLM2-135M"
    echo "  2. Convergence (100 steps, 5 min) - SmolLM2-135M"
    echo "  3. Memory (50 steps, 10 min) - TinyLlama-1.1B"
    echo "  4. Extended (500 steps, 30 min) - TinyLlama-1.1B"
    echo "  5. Full (1000 steps, 2 hours) - LLaMA-7B"
    echo ""
    log_info "Start with: ./scripts/gpu-test.sh quick"
  else
    log_warning "CUDA not detected - GPU tests may not work"
  fi
  
  echo ""
}

# =============================================================================
# Main
# =============================================================================

case "$COMMAND" in
  quick)
    test_quick
    ;;
  convergence)
    test_convergence
    ;;
  memory)
    test_memory
    ;;
  extended)
    test_extended
    ;;
  full)
    test_full
    ;;
  all)
    test_all
    ;;
  info)
    show_info
    ;;
  help)
    show_help
    ;;
  *)
    log_error "Unknown command: $COMMAND"
    echo ""
    show_help
    exit 2
    ;;
esac

log_success "GPU test suite completed"
