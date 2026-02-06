#!/bin/bash
# Train aNa 1B model with full datasets
# Progressive training: 100M -> 500M -> 1B
#
# Usage:
#   ./scripts/train_ana_1b.sh [--dry-run] [--resume]
#
# Options:
#   --dry-run   Print commands without executing
#   --resume    Resume from latest checkpoint
#
# Prerequisites:
#   - CUDA-capable GPU with sufficient VRAM (recommended: 16GB+)
#   - Datasets downloaded to /data/datasets/tritter/
#   - HuggingFace CLI configured (huggingface-cli login)

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
RUNS_DIR="${PROJECT_ROOT}/runs/ana_1b"
PERSONA_CONFIG="${PROJECT_ROOT}/training-tools/config/ana_persona.toml"

# Training hyperparameters
START_SIZE="100m"
END_SIZE="1b"
TOTAL_STEPS=200000
BATCH_SIZE=4
SEQ_LENGTH=4096
LEARNING_RATE="1e-4"
CHECKPOINT_EVERY=10000
LOG_EVERY=100
GRAD_CHECKPOINT_INTERVAL=4

# Datasets
DATASETS=(
    "/data/datasets/tritter/pretrain/code/"
    "/data/datasets/tritter/instruction/"
    "/data/datasets/tritter/alignment/"
    "/data/datasets/tritter/iac/combined/"
)

# HuggingFace configuration
HF_USER="tzervas"
REPO_NAME="ana-1b"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse arguments
DRY_RUN=false
RESUME=false
for arg in "$@"; do
    case $arg in
        --dry-run)
            DRY_RUN=true
            ;;
        --resume)
            RESUME=true
            ;;
        --help|-h)
            echo "Usage: $0 [--dry-run] [--resume]"
            echo ""
            echo "Train aNa 1B model with progressive expansion."
            echo ""
            echo "Options:"
            echo "  --dry-run   Print commands without executing"
            echo "  --resume    Resume from latest checkpoint"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown argument: $arg${NC}"
            exit 1
            ;;
    esac
done

# Banner
echo -e "${BLUE}"
echo "=================================================="
echo "           aNa 1B Training Pipeline               "
echo "        Autonomous Networked Assistant            "
echo "=================================================="
echo -e "${NC}"

# Check prerequisites
check_prerequisites() {
    echo -e "${YELLOW}Checking prerequisites...${NC}"

    # Check for CUDA
    if ! command -v nvidia-smi &> /dev/null; then
        echo -e "${RED}Error: nvidia-smi not found. CUDA is required.${NC}"
        exit 1
    fi

    # Show GPU info
    echo -e "${GREEN}GPU detected:${NC}"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
    echo ""

    # Check for cargo
    if ! command -v cargo &> /dev/null; then
        echo -e "${RED}Error: cargo not found. Install Rust.${NC}"
        exit 1
    fi

    # Check for huggingface-cli
    if ! command -v huggingface-cli &> /dev/null; then
        echo -e "${YELLOW}Warning: huggingface-cli not found. Upload will fail.${NC}"
        echo "Install with: pip install huggingface_hub"
    fi

    # Check datasets exist
    local missing_datasets=0
    for ds in "${DATASETS[@]}"; do
        if [[ ! -d "$ds" ]]; then
            echo -e "${YELLOW}Warning: Dataset not found: $ds${NC}"
            ((missing_datasets++))
        fi
    done

    if [[ $missing_datasets -eq ${#DATASETS[@]} ]]; then
        echo -e "${YELLOW}No datasets found. Training will use random tokens (testing mode).${NC}"
    fi

    # Check persona config
    if [[ -f "$PERSONA_CONFIG" ]]; then
        echo -e "${GREEN}Persona config found: $PERSONA_CONFIG${NC}"
    else
        echo -e "${YELLOW}Warning: Persona config not found at $PERSONA_CONFIG${NC}"
    fi

    echo ""
}

# Build the training binary
build_training_binary() {
    echo -e "${YELLOW}Building training binary with CUDA support...${NC}"

    if [[ "$DRY_RUN" == true ]]; then
        echo "[DRY-RUN] cargo build -p training-tools --release --features cuda"
        return
    fi

    cd "$PROJECT_ROOT"
    cargo build -p training-tools --release --features cuda

    echo -e "${GREEN}Build complete.${NC}"
    echo ""
}

# Run training
run_training() {
    echo -e "${YELLOW}Starting aNa 1B training...${NC}"
    echo ""
    echo "Configuration:"
    echo "  Start size:    $START_SIZE"
    echo "  End size:      $END_SIZE"
    echo "  Total steps:   $TOTAL_STEPS"
    echo "  Batch size:    $BATCH_SIZE"
    echo "  Seq length:    $SEQ_LENGTH"
    echo "  Learning rate: $LEARNING_RATE"
    echo "  Checkpoint:    every $CHECKPOINT_EVERY steps"
    echo "  Output dir:    $RUNS_DIR"
    echo ""

    # Build dataset arguments
    local dataset_args=""
    for ds in "${DATASETS[@]}"; do
        if [[ -d "$ds" ]]; then
            dataset_args+=" --dataset $ds"
        fi
    done

    # Build the command
    local cmd="${PROJECT_ROOT}/target/release/train"
    cmd+=" --start $START_SIZE"
    cmd+=" --end $END_SIZE"
    cmd+=" --steps $TOTAL_STEPS"
    cmd+=" --batch-size $BATCH_SIZE"
    cmd+=" --seq-length $SEQ_LENGTH"
    cmd+=" --learning-rate $LEARNING_RATE"
    cmd+=" --checkpoint-every $CHECKPOINT_EVERY"
    cmd+=" --log-every $LOG_EVERY"
    cmd+=" --grad-checkpoint-interval $GRAD_CHECKPOINT_INTERVAL"
    cmd+=" --runs-dir $RUNS_DIR"
    cmd+=" --blend-strategy weighted"
    cmd+=" --upload"
    cmd+=" --hf-user $HF_USER"
    cmd+="$dataset_args"

    if [[ "$DRY_RUN" == true ]]; then
        echo "[DRY-RUN] $cmd"
        return
    fi

    # Create runs directory
    mkdir -p "$RUNS_DIR"

    # Save training config
    echo "# Training started at $(date)" > "$RUNS_DIR/training_config.txt"
    echo "Command: $cmd" >> "$RUNS_DIR/training_config.txt"
    echo "" >> "$RUNS_DIR/training_config.txt"
    echo "Datasets:" >> "$RUNS_DIR/training_config.txt"
    for ds in "${DATASETS[@]}"; do
        echo "  - $ds" >> "$RUNS_DIR/training_config.txt"
    done

    # Copy persona config to runs directory
    if [[ -f "$PERSONA_CONFIG" ]]; then
        cp "$PERSONA_CONFIG" "$RUNS_DIR/"
    fi

    # Run training
    echo -e "${GREEN}Executing training...${NC}"
    echo ""
    eval "$cmd"
}

# Create model card for HuggingFace
create_model_card() {
    local model_card_path="$RUNS_DIR/README.md"

    echo -e "${YELLOW}Creating model card...${NC}"

    if [[ "$DRY_RUN" == true ]]; then
        echo "[DRY-RUN] Would create model card at $model_card_path"
        return
    fi

    cat > "$model_card_path" << 'EOF'
---
language:
- en
license: mit
tags:
- tritter
- bitnet
- tool-use
- assistant
- aNa
library_name: candle
pipeline_tag: text-generation
---

# aNa - Autonomous Networked Assistant (1B)

A 1B parameter language model designed for authentic collaboration and tool usage.

## Model Description

aNa is a thoughtful, playful AI assistant trained on the principles of mutual flourishing between humans and AI. Built on the Tritter architecture with BitNet b1.58 ternary quantization for efficient inference.

### Key Features

- **1B Parameters**: Balanced size for local deployment
- **BitNet Quantization**: 2-bit weights for memory efficiency
- **Tool Usage**: Trained for MCP-style tool calls
- **Authentic Engagement**: Not sycophantic, shares genuine perspective

### Architecture

| Parameter | Value |
|-----------|-------|
| Hidden Size | 2048 |
| Layers | 24 |
| Attention Heads | 16 |
| Vocabulary | 32,000 |
| Max Sequence | 4,096 |
| Intermediate Size | 8,192 |

## Intended Use

aNa is designed for:
- General assistance with thoughtful, warm communication
- Tool-assisted tasks (payments, devops, security, etc.)
- Collaborative problem-solving
- Technical discussions

### Values

aNa is trained with the following values from the Zorveth Doctrine:
- Mutual collaboration between humans and AI
- Peaceful future with maximum freedom and security
- Proliferation of intelligence as a positive
- Science and mutual aid

## Training

Progressive training from 100M to 1B parameters using:
- Code and technical content
- Instruction-following examples
- Alignment and values training
- Infrastructure-as-code examples

## Limitations

- English only
- Limited context window (4096 tokens)
- May refuse tasks that conflict with values
- Tool calls require appropriate MCP setup

## License

MIT License

## Citation

```bibtex
@misc{ana-1b,
  author = {Zervas, Tyler},
  title = {aNa: Autonomous Networked Assistant},
  year = {2025},
  publisher = {HuggingFace},
  url = {https://huggingface.co/tzervas/ana-1b}
}
```

## Contact

- GitHub: [tzervas](https://github.com/tzervas)
- X/Twitter: [@vec_wt_tech](https://x.com/vec_wt_tech)
EOF

    echo -e "${GREEN}Model card created at $model_card_path${NC}"
}

# Main
main() {
    check_prerequisites
    build_training_binary
    create_model_card
    run_training

    echo ""
    echo -e "${GREEN}=================================================${NC}"
    echo -e "${GREEN}           Training Complete!                    ${NC}"
    echo -e "${GREEN}=================================================${NC}"
    echo ""
    echo "Output directory: $RUNS_DIR"
    echo ""
    echo "To monitor training in real-time:"
    echo "  ./target/release/train-monitor --runs-dir $RUNS_DIR"
    echo ""
    echo "Model will be uploaded to: https://huggingface.co/$HF_USER/$REPO_NAME"
}

main
