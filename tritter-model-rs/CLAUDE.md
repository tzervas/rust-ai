# CLAUDE.md - Tritter Model RS Development Context

Pure Rust implementation of the Tritter transformer model with BitNet quantization.

## Overview

This crate provides:
- **TritterModel**: Transformer architecture with QK-Norm, RoPE, and Squared ReLU
- **BitNet integration**: Ternary weight quantization via `bitnet-quantize`
- **Hybrid training**: Integration with `hybrid-predict-trainer-rs` for predictive training

## Architecture

```
tritter-model-rs/
├── src/
│   ├── lib.rs         # Public API exports
│   ├── config.rs      # TritterConfig (100M to 70B presets)
│   ├── model.rs       # TritterModel (embeddings, layers, LM head)
│   ├── attention.rs   # Multi-head attention with QK-Norm, RoPE, GQA
│   ├── mlp.rs         # SwiGLU MLP with Squared ReLU
│   ├── layer.rs       # Transformer layer (attention + MLP + norms)
│   ├── trainer.rs     # Integration with hybrid-predict-trainer-rs
│   └── error.rs       # Error types
└── examples/
    └── train.rs       # Training example
```

## Key Components

### Model Configuration
```rust
// Presets
TritterConfig::test()        // Minimal for unit tests
TritterConfig::small_100m()  // 100M parameters
TritterConfig::medium_500m() // 500M parameters
TritterConfig::large_1b()    // 1B parameters
TritterConfig::huge_7b()     // 7B parameters
```

### Trainer Integration
```rust
// Create trainer with hybrid predictive training
let trainer = create_trainer_with_config(
    &model_config,
    trainer_config,
    learning_rate,
    &device,
)?;

// Training step (handles phase selection automatically)
let result = trainer.step(&batch)?;
```

## Dependencies

| Crate | Purpose |
|-------|---------|
| `candle-core` | Tensor operations |
| `candle-nn` | Neural network layers |
| `bitnet-quantize` | Ternary weight quantization |
| `trit-vsa` | Packed ternary storage |
| `hybrid-predict-trainer-rs` | Predictive training framework |

## Development Commands

```bash
# Build
cargo build -p tritter-model-rs

# Test
cargo test -p tritter-model-rs

# Run training example
cargo run -p tritter-model-rs --example train --release

# With CUDA
cargo run -p tritter-model-rs --example train --release --features cuda

# Benchmark
cargo bench -p tritter-model-rs
```

## Model Sizes

| Size | Hidden | Layers | Heads | Parameters |
|------|--------|--------|-------|------------|
| Test | 64 | 2 | 2 | ~100K |
| 100M | 768 | 12 | 12 | ~100M |
| 500M | 1024 | 24 | 16 | ~500M |
| 1B | 2048 | 24 | 16 | ~1B |
| 3B | 2560 | 26 | 20 | ~3B |
| 7B | 4096 | 32 | 32 | ~7B |

## TODO

- [ ] Integrate BitNet quantization in forward pass
- [ ] Add model checkpoint save/load
- [ ] Implement gradient checkpointing for memory efficiency
- [ ] Add data loading utilities (JSONL, parquet)
- [ ] Benchmark against Python implementation
- [ ] Add tokenizer integration
