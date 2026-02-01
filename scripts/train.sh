#!/bin/bash
# Rust-AI Training Wrapper
# Automatically sets CUDA_COMPUTE_CAP for Blackwell GPUs (RTX 50 series)

# Detect GPU and set compute capability
detect_compute_cap() {
    if command -v nvidia-smi &> /dev/null; then
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)

        # RTX 50 series (Blackwell) - use sm_90 compatibility
        if [[ "$GPU_NAME" == *"RTX 50"* ]] || [[ "$GPU_NAME" == *"5080"* ]] || [[ "$GPU_NAME" == *"5090"* ]]; then
            echo "90"  # sm_90 compatibility mode for Blackwell
        # RTX 40 series (Ada Lovelace)
        elif [[ "$GPU_NAME" == *"RTX 40"* ]] || [[ "$GPU_NAME" == *"4090"* ]] || [[ "$GPU_NAME" == *"4080"* ]]; then
            echo "89"
        # RTX 30 series (Ampere)
        elif [[ "$GPU_NAME" == *"RTX 30"* ]] || [[ "$GPU_NAME" == *"3090"* ]] || [[ "$GPU_NAME" == *"3080"* ]]; then
            echo "86"
        # A100/H100
        elif [[ "$GPU_NAME" == *"A100"* ]] || [[ "$GPU_NAME" == *"H100"* ]]; then
            echo "90"
        else
            echo "80"  # Default to sm_80
        fi
    else
        echo "80"
    fi
}

# Set compute capability if not already set
if [ -z "$CUDA_COMPUTE_CAP" ]; then
    export CUDA_COMPUTE_CAP=$(detect_compute_cap)
    echo "Auto-detected CUDA_COMPUTE_CAP=$CUDA_COMPUTE_CAP"
fi

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Run the training binary
exec "$PROJECT_ROOT/target/release/train" "$@"
