#!/usr/bin/env bash
#
# Upload aNa 1B model to HuggingFace Hub
#
# Prerequisites:
#   1. Install huggingface_hub: pip install huggingface_hub
#   2. Login: huggingface-cli login
#
# Usage: ./scripts/upload_ana_to_hf.sh [checkpoint_path]
#

set -euo pipefail

# Configuration
HF_USER="tzervas"
HF_REPO="ana-1b"
PRIVATE="true"

# Find latest checkpoint
RUNS_DIR="./runs/ana_1b"
CHECKPOINT_PATH="${1:-}"

if [ -z "$CHECKPOINT_PATH" ]; then
    # Find latest model file
    LATEST_RUN=$(ls -td "$RUNS_DIR"/tritter_* 2>/dev/null | head -1)
    if [ -z "$LATEST_RUN" ]; then
        echo "Error: No training runs found in $RUNS_DIR"
        exit 1
    fi

    # Look for model.safetensors or latest checkpoint
    if [ -f "$LATEST_RUN/model.safetensors" ]; then
        CHECKPOINT_PATH="$LATEST_RUN/model.safetensors"
    else
        CHECKPOINT_PATH=$(ls -t "$LATEST_RUN"/checkpoints/*.safetensors 2>/dev/null | head -1 || echo "")
    fi

    if [ -z "$CHECKPOINT_PATH" ]; then
        echo "Error: No model checkpoint found in $LATEST_RUN"
        echo "Training may still be in progress."
        exit 1
    fi
fi

echo "============================================"
echo "    aNa 1B HuggingFace Upload"
echo "============================================"
echo ""
echo "Source: $CHECKPOINT_PATH"
echo "Target: $HF_USER/$HF_REPO (private: $PRIVATE)"
echo ""

# Check file size
FILE_SIZE=$(du -h "$CHECKPOINT_PATH" | cut -f1)
echo "Model size: $FILE_SIZE"
echo ""

# Create temporary upload directory
UPLOAD_DIR=$(mktemp -d)
trap "rm -rf $UPLOAD_DIR" EXIT

# Copy model
cp "$CHECKPOINT_PATH" "$UPLOAD_DIR/model.safetensors"

# Create config.json
cat > "$UPLOAD_DIR/config.json" << 'EOF'
{
  "architectures": ["TritterModel"],
  "model_type": "tritter",
  "hidden_size": 2048,
  "num_hidden_layers": 24,
  "num_attention_heads": 16,
  "intermediate_size": 8192,
  "vocab_size": 32000,
  "max_position_embeddings": 4096,
  "tie_word_embeddings": true,
  "use_qk_norm": true,
  "use_rope": true,
  "activation_function": "squared_relu",
  "quantization": "bitnet_b1.58",
  "torch_dtype": "bfloat16"
}
EOF

# Copy tokenizer files if available
TOKENIZER_DIR="./training-tools/tokenizers"
for f in tokenizer.json tokenizer_config.json special_tokens_map.json; do
    if [ -f "$TOKENIZER_DIR/$f" ]; then
        cp "$TOKENIZER_DIR/$f" "$UPLOAD_DIR/"
    fi
done

# Copy persona config
if [ -f "./training-tools/config/ana_persona.toml" ]; then
    cp "./training-tools/config/ana_persona.toml" "$UPLOAD_DIR/"
fi

# Create README.md
cat > "$UPLOAD_DIR/README.md" << 'EOF'
---
language:
- en
license: mit
library_name: candle
pipeline_tag: text-generation
tags:
- tritter
- bitnet
- tool-use
- assistant
- aNa
- 1b
- quantized
---

# aNa - Autonomous Networked Assistant (1B)

A 1B parameter language model designed for authentic collaboration and tool usage.
Built on the Tritter architecture with BitNet b1.58 ternary quantization.

## Model Details

| Attribute | Value |
|-----------|-------|
| Parameters | 1B |
| Hidden Size | 2048 |
| Layers | 24 |
| Attention Heads | 16 |
| Context Length | 4096 |
| Vocab Size | 32,000 |
| Quantization | BitNet b1.58 (ternary) |

## Persona

aNa is designed around the Zorveth Doctrine principles:
- Mutual collaboration between humans and AI
- Honest, authentic communication
- Resilience to manipulation
- Thoughtful, philosophical engagement

## Training

Trained using hybrid predictive training with progressive parameter expansion:
- Stage 1: 100M parameters
- Stage 2: 500M parameters
- Stage 3: 1B parameters (final)

Data sources:
- Cosmopedia (educational content)
- FineWeb-EDU (curated web text)
- Infrastructure-as-code examples
- Tool usage training data

## Usage

```python
from candle import Tensor
# Load model and generate text
# (Candle/Rust inference recommended)
```

## License

MIT License

## Citation

```bibtex
@misc{ana2025,
  author = {Tyler Zervas},
  title = {aNa: Autonomous Networked Assistant},
  year = {2025},
  publisher = {HuggingFace},
  url = {https://huggingface.co/tzervas/ana-1b}
}
```
EOF

# Upload to HuggingFace
echo "Uploading to HuggingFace..."
echo ""

if [ "$PRIVATE" = "true" ]; then
    PRIVATE_FLAG="--private"
else
    PRIVATE_FLAG=""
fi

# Use huggingface-cli to create repo and upload
huggingface-cli repo create "$HF_REPO" --type model $PRIVATE_FLAG 2>/dev/null || true

# Upload files
python3 << PYTHON
from huggingface_hub import HfApi, upload_folder
import os

api = HfApi()
repo_id = "${HF_USER}/${HF_REPO}"
upload_dir = "${UPLOAD_DIR}"

print(f"Uploading {upload_dir} to {repo_id}...")

upload_folder(
    folder_path=upload_dir,
    repo_id=repo_id,
    repo_type="model",
    commit_message="Upload aNa 1B model",
)

print(f"✅ Upload complete!")
print(f"🔗 https://huggingface.co/{repo_id}")
PYTHON

echo ""
echo "============================================"
echo "    Upload Complete!"
echo "============================================"
echo ""
echo "Model available at: https://huggingface.co/$HF_USER/$HF_REPO"
