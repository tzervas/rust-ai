# tritter-model-rs

Pure Rust implementation of the Tritter transformer model with BitNet 1.58-bit ternary quantization.

## Features

- **Transformer Architecture**: QK-Norm attention with RoPE positional embeddings
- **BitNet Integration**: Ternary weight quantization ({-1, 0, +1}) via `bitnet-quantize`
- **Squared ReLU**: `x * ReLU(x)` activation for ternary training stability
- **Hybrid Predictive Training**: WARMUP → FULL → PREDICT → CORRECT phase cycling
- **Streaming Data Loading**: Memory-efficient JSONL/Parquet iterators with dynamic padding
- **Model Presets**: 100M, 500M, 1B, 3B, 7B parameter configurations

## Installation

```toml
[dependencies]
tritter-model-rs = "0.1"
```

With CUDA support:
```toml
[dependencies]
tritter-model-rs = { version = "0.1", features = ["cuda"] }
```

## Quick Start

```rust
use tritter_model_rs::{TritterConfig, TritterModel};
use candle_core::Device;

// Create a 100M parameter model
let config = TritterConfig::small_100m();
let device = Device::Cpu;
let mut model = TritterModel::new(&config, &device)?;

// Enable BitNet quantization
let mut config = TritterConfig::medium_500m();
config.use_bitnet = true;
let model = TritterModel::new(&config, &device)?;
```

## Training

```rust
use tritter_model_rs::{TritterConfig, trainer::create_trainer_with_config};
use hybrid_predict_trainer_rs::HybridTrainerConfig;

let model_config = TritterConfig::medium_500m();
let trainer_config = HybridTrainerConfig::builder()
    .warmup_steps(100)
    .full_steps(20)
    .max_predict_steps(80)
    .build();

let mut trainer = create_trainer_with_config(
    &model_config,
    trainer_config,
    3e-4, // learning rate
    &device,
)?;

// Training loop
for batch in data_loader {
    let result = trainer.step(&batch?)?;
    println!("Loss: {:.4}, Phase: {:?}", result.loss, result.phase);
}
```

## Model Sizes

| Size | Hidden | Layers | Heads | Parameters | Memory (BitNet) |
|------|--------|--------|-------|------------|-----------------|
| Test | 64 | 2 | 2 | ~100K | ~1 MB |
| 100M | 768 | 12 | 12 | ~100M | ~33 MB |
| 500M | 1024 | 24 | 16 | ~500M | ~131 MB |
| 1B | 2048 | 24 | 16 | ~1B | ~262 MB |
| 3B | 2560 | 26 | 20 | ~3B | ~786 MB |
| 7B | 4096 | 32 | 32 | ~7B | ~1.8 GB |

## Dependencies

- [candle-core](https://github.com/huggingface/candle) - Tensor operations
- [bitnet-quantize](https://crates.io/crates/bitnet-quantize) - BitNet 1.58-bit quantization
- [trit-vsa](https://crates.io/crates/trit-vsa) - Balanced ternary arithmetic
- [hybrid-predict-trainer-rs](https://github.com/tzervas/rust-ai) - Predictive training framework

## License

MIT
