# axolotl-rs

YAML-driven configurable fine-tuning toolkit for LLMs in Rust.

[![Crates.io](https://img.shields.io/crates/v/axolotl-rs.svg)](https://crates.io/crates/axolotl-rs)
[![Documentation](https://docs.rs/axolotl-rs/badge.svg)](https://docs.rs/axolotl-rs)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE-MIT)

## Overview

`axolotl-rs` is a Rust port of the Python [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl) project, providing a framework for fine-tuning language models.

**Features:**
- **YAML Configuration** - Parse and validate training configuration files
- **Dataset Handling** - Load datasets in Alpaca, ShareGPT, completion, and custom formats
- **CLI Interface** - Commands for `validate`, `train`, `merge`, `init`
- **Configuration Presets** - Templates for LLaMA-2, Mistral, and Phi-3 models
- **Adapter Integration** - LoRA and QLoRA via peft-rs and qlora-rs
- **Training Loop** - Forward/backward passes with checkpoint support

## Installation

```bash
# From crates.io
cargo install axolotl-rs

# Or from source
git clone https://github.com/tzervas/axolotl-rs
cd axolotl-rs
cargo build --release
```

## Quick Start

### 1. Generate a Configuration

```bash
# Create a config for LLaMA-2 7B with QLoRA
axolotl init config.yaml --preset llama2-7b
```

### 2. Prepare Your Dataset

Create a JSONL file in Alpaca format:

```json
{"instruction": "Explain quantum computing", "input": "", "output": "Quantum computing uses..."}
{"instruction": "Write a haiku about Rust", "input": "", "output": "Memory safe code\n..."}
```

### 3. Validate Configuration

```bash
axolotl validate config.yaml
```

### 4. Start Training

```bash
axolotl train config.yaml
```

### 5. Merge Adapters (Optional)

```bash
axolotl merge --config config.yaml --output ./merged-model
```

## Configuration

### Full Example

```yaml
# config.yaml
base_model: meta-llama/Llama-2-7b-hf
adapter: qlora

# LoRA settings
lora:
  r: 64
  alpha: 16
  dropout: 0.05
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj

# Quantization (for QLoRA)
quantization:
  bits: 4
  quant_type: nf4
  double_quant: true

# Dataset
dataset:
  path: ./data/train.jsonl
  format: alpaca
  max_length: 2048
  val_split: 0.05

# Training
training:
  epochs: 3
  batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 2.0e-4
  lr_scheduler: cosine
  warmup_ratio: 0.03
  save_steps: 500
  gradient_checkpointing: true

output_dir: ./outputs/my-model
seed: 42
```

### Dataset Formats

| Format | Description | Fields |
|--------|-------------|--------|
| `alpaca` | Standard Alpaca | `instruction`, `input`, `output` |
| `sharegpt` | Conversation format | `conversations[{from, value}]` |
| `completion` | Raw text | `text` |
| `custom` | User-defined | Configure `input_field`, `output_field` |

### Available Presets

- `llama2-7b` - LLaMA-2 7B with QLoRA
- `mistral-7b` - Mistral 7B with QLoRA  
- `phi3-mini` - Phi-3 Mini with LoRA

## CLI Commands

```bash
# Validate configuration
axolotl validate <config.yaml>

# Start training
axolotl train <config.yaml>
axolotl train <config.yaml> --resume ./checkpoint-1000

# Merge adapter into base model
axolotl merge --config <config.yaml> --output <path>

# Generate sample config
axolotl init <output.yaml> --preset <preset>
```

## Architecture

```
axolotl-rs
├── config     - YAML parsing & validation
├── dataset    - Data loading & preprocessing
├── model      - Model loading & adapter management
└── trainer    - Training loop & optimization

Dependencies:
├── candle-*   - Tensor operations and transformer models
├── tokenizers - HuggingFace tokenizer bindings
├── peft-rs    - LoRA/DoRA adapter support (optional)
├── qlora-rs   - 4-bit quantization (optional)
└── unsloth-rs - Optimized kernels (optional)
```

## Feature Flags

| Flag | Description |
|------|-------------|
| `download` | Enable model downloading from HF Hub (default) |
| `peft` | Enable peft-rs for LoRA/DoRA adapters |
| `qlora` | Enable qlora-rs for 4-bit quantization |
| `unsloth` | Enable unsloth-rs optimized kernels |
| `cuda` | Enable CUDA GPU acceleration |

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) - Technical architecture details
- [CONTRIBUTING.md](CONTRIBUTING.md) - Contribution guidelines
- [TEST_COVERAGE_PLAN.md](TEST_COVERAGE_PLAN.md) - Test coverage goals

**Porting from Python:** This is a Rust port of the Python [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl) project, designed for better performance and efficiency.

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Licensed under the [MIT License](LICENSE-MIT).
