# axolotl-rs - YAML-Driven Fine-Tuning Toolkit

## Overview

High-level fine-tuning orchestration layer. Rust port of Python Axolotl, providing YAML-driven configuration for LLM training.

**Status**: 1.0.0 - Configuration and CLI functional, training loop in development.

## Architecture

```
src/
├── lib.rs           # Public API exports
├── main.rs          # CLI entry point
├── cli.rs           # Clap CLI definitions
├── config.rs        # YAML configuration parsing
├── dataset.rs       # Dataset loaders (Alpaca, ShareGPT, etc.)
├── model.rs         # Model loading and architecture
├── lora_llama.rs    # LLaMA with LoRA integration
├── llama_common.rs  # Shared LLaMA utilities
├── error.rs         # Error types
├── normalization.rs # Normalization utilities
├── adapters/        # Adapter integration layer
│   └── mod.rs       # Feature-gated adapter loading
└── mocks/           # Mock implementations for testing
    └── mod.rs

tests/
├── cli_tests.rs           # CLI validation tests
├── e2e_qlora.rs          # End-to-end QLoRA tests
├── gpu_checkpoint.rs     # Checkpoint save/load tests
├── gpu_lora_targets.rs   # LoRA target selection tests
├── gpu_training.rs       # GPU training loop tests
├── gpu_utils.rs          # GPU test utilities
└── lora_llama_tests.rs   # LLaMA+LoRA integration tests
```

## Feature Flags

```toml
[features]
default = ["download"]
download = ["reqwest"]           # Model downloads from HuggingFace

# Real adapter integrations
peft = ["peft-rs"]              # Enable peft-rs adapters
qlora = ["qlora-rs", "peft"]    # QLoRA (requires peft)
unsloth = ["unsloth-rs"]        # Optimized kernels

# Testing without real deps
mock-peft = []
mock-qlora = []
mock-unsloth = []

# GPU support
cuda = ["candle-core/cuda"]
```

## Key Components

### Configuration (`config.rs`)
```rust
#[derive(Debug, Deserialize)]
pub struct AxolotlConfig {
    pub model: ModelConfig,
    pub dataset: DatasetConfig,
    pub training: TrainingConfig,
    pub adapter: Option<AdapterConfig>,
    // ... extensive config options
}
```

### Dataset Loading (`dataset.rs`)
Supports formats:
- Alpaca (instruction/input/output)
- ShareGPT (conversations)
- Completion (raw text)
- Custom with column mapping

### Model Integration (`model.rs`)
Feature-gated integration with peft-rs for LoRA/QLoRA adapters.

## Development Commands

```bash
# Check
cargo check -p axolotl-rs

# Check with features
cargo check -p axolotl-rs --features "peft,qlora,unsloth"

# Test
cargo test -p axolotl-rs

# GPU tests
cargo test -p axolotl-rs --features cuda -- --ignored

# CLI validation
cargo run -p axolotl-rs -- validate config.yaml

# Build CLI
cargo build -p axolotl-rs --release
```

## CLI Commands

```bash
# Validate configuration
axolotl validate config.yaml

# Initialize new config
axolotl init --preset llama-2-7b

# Train
axolotl train config.yaml

# Merge adapters
axolotl merge config.yaml --output merged_model/
```

## Testing Strategy

- CLI tests: Validate command parsing
- Config tests: YAML parsing and validation
- Integration: End-to-end with mock adapters
- GPU tests: Real training loops (ignored without CUDA)

## 1.0 Checklist

- [x] YAML configuration parsing
- [x] Dataset loaders (4 formats)
- [x] CLI interface
- [x] Configuration presets
- [x] Clean compilation (no warnings)
- [x] CI/CD pipeline with GitHub Actions
- [ ] Working training loop
- [ ] Checkpoint save/load
- [ ] Adapter merging
- [ ] Multi-GPU support
- [ ] Progress reporting
- [ ] Metrics logging (TensorBoard/W&B)
- [ ] Examples directory
- [ ] 100% doc coverage

## Integration Points

### With peft-rs (feature: `peft`)
```rust
#[cfg(feature = "peft")]
use peft_rs::{LoraAdapter, LoraConfig, AdapterRegistry};
```

### With qlora-rs (feature: `qlora`)
```rust
#[cfg(feature = "qlora")]
use qlora_rs::{QLoraLayer, Nf4Quantizer};
```

### With unsloth-rs (feature: `unsloth`)
```rust
#[cfg(feature = "unsloth")]
use unsloth_rs::{MultiHeadAttention, apply_rotary_embedding};
```

## Common Issues

### Config validation fails
Check YAML syntax and required fields. Use `axolotl validate` for diagnostics.

### Dataset loading slow
Large datasets should use streaming. Check `dataset.streaming` config option.

### Feature flags
When using adapter features, ensure corresponding dependencies are available:
- `peft` requires peft-rs
- `qlora` requires both qlora-rs and peft-rs
- `unsloth` requires unsloth-rs

## Configuration Example

```yaml
model:
  name: meta-llama/Llama-2-7b-hf
  dtype: bfloat16

dataset:
  path: ./data/alpaca.json
  format: alpaca

training:
  epochs: 3
  batch_size: 4
  learning_rate: 2e-4
  gradient_accumulation: 4

adapter:
  type: lora
  rank: 16
  alpha: 32
  target_modules:
    - q_proj
    - v_proj
```
