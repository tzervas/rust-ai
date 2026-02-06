#!/bin/bash
# E2E Validation Setup Script for axolotl-rs
# Downloads test models and datasets for local validation

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}/hub"
DATA_DIR="$SCRIPT_DIR/data"

echo "=== axolotl-rs E2E Validation Setup ==="
echo "Cache directory: $CACHE_DIR"
echo "Data directory: $DATA_DIR"
echo

# Check for huggingface-cli
if ! command -v huggingface-cli &> /dev/null; then
    echo "Installing huggingface_hub..."
    pip install huggingface_hub
fi

# Create data directory
mkdir -p "$DATA_DIR"

# Function to download model
download_model() {
    local model_id="$1"
    local model_name=$(echo "$model_id" | sed 's/\//_/g')
    local target_dir="$CACHE_DIR/models--${model_id//\//_}"
    
    if [ -d "$target_dir" ] && [ -f "$target_dir/snapshots/"*"/model.safetensors" ]; then
        echo "✓ $model_id already cached"
    else
        echo "Downloading $model_id..."
        huggingface-cli download "$model_id" --cache-dir "$CACHE_DIR"
    fi
}

# Download models based on argument
case "${1:-all}" in
    smollm2)
        echo "=== Downloading SmolLM2-135M (Development Model) ==="
        download_model "HuggingFaceTB/SmolLM2-135M"
        ;;
    tinyllama)
        echo "=== Downloading TinyLlama-1.1B (Validation Model) ==="
        download_model "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        ;;
    all)
        echo "=== Downloading All Test Models ==="
        download_model "HuggingFaceTB/SmolLM2-135M"
        download_model "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        ;;
    *)
        echo "Usage: $0 [smollm2|tinyllama|all]"
        exit 1
        ;;
esac

echo
echo "=== Downloading Alpaca Dataset ==="

# Download and create subsets
ALPACA_FILE="$DATA_DIR/alpaca_full.jsonl"
if [ ! -f "$ALPACA_FILE" ]; then
    echo "Downloading tatsu-lab/alpaca..."
    python3 << 'EOF'
import json
from datasets import load_dataset

print("Loading alpaca dataset...")
ds = load_dataset("tatsu-lab/alpaca", split="train")

# Save full dataset as JSONL
with open("data/alpaca_full.jsonl", "w") as f:
    for item in ds:
        f.write(json.dumps({
            "instruction": item["instruction"],
            "input": item.get("input", ""),
            "output": item["output"]
        }) + "\n")

print(f"Saved {len(ds)} examples to data/alpaca_full.jsonl")

# Create subsets
for size, name in [(100, "100"), (1000, "1k"), (5000, "5k")]:
    subset = ds.select(range(min(size, len(ds))))
    filename = f"data/alpaca_{name}.jsonl"
    with open(filename, "w") as f:
        for item in subset:
            f.write(json.dumps({
                "instruction": item["instruction"],
                "input": item.get("input", ""),
                "output": item["output"]
            }) + "\n")
    print(f"Created {filename} with {len(subset)} examples")
EOF
else
    echo "✓ Alpaca dataset already downloaded"
fi

echo
echo "=== Setup Complete ==="
echo
echo "Available models:"
ls -la "$CACHE_DIR"/models--* 2>/dev/null | head -5 || echo "  (none yet)"
echo
echo "Available datasets:"
ls -la "$DATA_DIR"/*.jsonl 2>/dev/null || echo "  (none yet)"
echo
echo "Next steps:"
echo "  1. Run CPU smoke tests: cargo test --features qlora -- --ignored"
echo "  2. Run GPU validation: cargo test --features 'qlora cuda' -- --ignored"
