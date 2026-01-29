# tritter-accel

Rust acceleration for AI training and inference, with both Rust and Python APIs.

[![CI](https://github.com/tzervas/rust-ai/actions/workflows/tritter-accel.yml/badge.svg)](https://github.com/tzervas/rust-ai/actions/workflows/tritter-accel.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE-MIT)

## Overview

**tritter-accel** provides high-performance operations for both ternary (BitNet-style) and conventional neural network workloads. It offers:

- **Dual API**: Both Rust and Python interfaces
- **Ternary Operations**: BitNet b1.58 quantization and inference
- **VSA Gradient Compression**: 10-100x compression for distributed training
- **GPU Acceleration**: Optional CUDA support via CubeCL

## Features

| Feature | Description | Benefit |
|---------|-------------|---------|
| Ternary Quantization | AbsMean/AbsMax to {-1, 0, +1} | 16x memory reduction |
| Packed Storage | 2-bit per trit (4 values/byte) | Efficient storage |
| Ternary Matmul | Addition-only arithmetic | 2-4x speedup |
| VSA Operations | Bind/bundle/similarity | Hyperdimensional computing |
| Gradient Compression | Random projection | 10-100x compression |
| Mixed Precision | BF16 utilities | Training efficiency |

## Installation

### Rust

Add to your `Cargo.toml`:

```toml
[dependencies]
tritter-accel = "0.2"

# With GPU support
tritter-accel = { version = "0.2", features = ["cuda"] }
```

### Python

Build with maturin:

```bash
cd tritter-accel
pip install maturin numpy
maturin develop --release

# With CUDA support
maturin develop --release --features cuda
```

## Usage

### Rust API

```rust
use tritter_accel::core::{
    quantization::{quantize_absmean, QuantizeConfig},
    ternary::{PackedTernary, matmul},
    training::{GradientCompressor, TrainingConfig},
    vsa::{VsaOps, VsaConfig},
};
use candle_core::{Device, Tensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;

    // Quantize weights to ternary
    let weights = Tensor::randn(0f32, 1f32, (512, 512), &device)?;
    let result = quantize_absmean(&weights, &QuantizeConfig::default())?;
    let packed = result.to_packed()?;

    // Ternary matmul (no multiplications!)
    let input = Tensor::randn(0f32, 1f32, (1, 512), &device)?;
    let output = matmul(&input, &packed, None)?;

    // VSA hyperdimensional computing
    let ops = VsaOps::new(VsaConfig::default());
    let a = ops.random(10000, 42)?;
    let b = ops.random(10000, 43)?;
    let bound = ops.bind(&a, &b)?;

    // Compress gradients for distributed training
    let config = TrainingConfig::default().with_compression_ratio(0.1);
    let compressor = GradientCompressor::new(config);
    let gradients: Vec<f32> = vec![0.1, -0.2, 0.3];
    let compressed = compressor.compress(&gradients, None)?;

    Ok(())
}
```

### Python API

```python
import numpy as np
from tritter_accel import (
    quantize_weights_absmean,
    pack_ternary_weights,
    ternary_matmul,
    compress_gradients_vsa,
    decompress_gradients_vsa,
)

# Quantize float weights to ternary {-1, 0, +1}
weights = np.random.randn(512, 512).astype(np.float32)
ternary_weights, scales = quantize_weights_absmean(weights)

# Pack for efficient storage (16x compression)
packed, scales = pack_ternary_weights(ternary_weights, scales)

# Efficient matmul with packed weights
input_data = np.random.randn(4, 512).astype(np.float32)
output = ternary_matmul(input_data, packed, scales, (512, 512))

# VSA gradient compression for distributed training
gradients = np.random.randn(1000000).astype(np.float32)
compressed, seed = compress_gradients_vsa(gradients, 0.1, 42)
print(f"Compression: {len(gradients) / len(compressed):.1f}x")
```

## Module Structure

```
tritter_accel
├── core                    # Pure Rust API
│   ├── ternary            # PackedTernary, matmul, dot
│   ├── quantization       # quantize_absmean, quantize_absmax
│   ├── vsa                # VsaOps (bind, bundle, similarity)
│   ├── training           # GradientCompressor, mixed_precision
│   └── inference          # InferenceEngine, TernaryLayer, KVCache
├── bitnet                  # Re-exports from bitnet-quantize
├── ternary                 # Re-exports from trit-vsa
└── vsa                     # Re-exports from vsa-optim-rs
```

## API Reference

### Python Functions

| Function | Description |
|----------|-------------|
| `quantize_weights_absmean(weights)` | Quantize float weights to ternary using AbsMean scaling |
| `pack_ternary_weights(weights, scales)` | Pack ternary weights into 2-bit representation |
| `unpack_ternary_weights(packed, scales, shape)` | Unpack ternary weights to float |
| `ternary_matmul(input, packed, scales, shape)` | Matrix multiply with packed ternary weights |
| `compress_gradients_vsa(gradients, ratio, seed)` | Compress gradients using VSA |
| `decompress_gradients_vsa(compressed, dim, seed)` | Decompress gradients from VSA |
| `version()` | Get library version |
| `cuda_available_py()` | Check if CUDA is available |

### Rust Types

| Type | Description |
|------|-------------|
| `PackedTernary` | Packed ternary weight storage with scales |
| `QuantizationResult` | Result of quantization with values, scales, shape |
| `VsaOps` | VSA operations handler with device dispatch |
| `GradientCompressor` | Gradient compression/decompression |
| `InferenceEngine` | Batched inference with device management |
| `TernaryLayer` | Pre-quantized layer for fast inference |

## Performance

| Operation | vs FP32 | Memory |
|-----------|---------|--------|
| Ternary matmul (CPU) | 2x speedup | 16x reduction |
| Ternary matmul (GPU) | 4x speedup | 16x reduction |
| Weight packing | N/A | 16x reduction |
| VSA gradient compression | N/A | 10-100x reduction |

Run benchmarks:
```bash
cargo bench -p tritter-accel
```

## Examples

See the `examples/` directory:

- `basic_quantization.py` - Weight quantization demo
- `ternary_inference.py` - Inference with packed weights
- `gradient_compression.py` - VSA gradient compression
- `vsa_operations.py` - Hyperdimensional computing
- `benchmark_comparison.py` - Performance comparisons

## Dependencies

This crate delegates to specialized sister crates:

| Crate | Description |
|-------|-------------|
| [trit-vsa](../trit-vsa) | Balanced ternary arithmetic & VSA |
| [bitnet-quantize](../bitnet-quantize) | BitNet b1.58 quantization |
| [vsa-optim-rs](../vsa-optim-rs) | Gradient optimization |
| [rust-ai-core](../rust-ai-core) | GPU dispatch & device management |

## Feature Flags

| Feature | Description |
|---------|-------------|
| `default` | CPU-only build |
| `cuda` | Enable CUDA GPU acceleration |

## License

MIT License - see [LICENSE-MIT](LICENSE-MIT)

## Contributing

Contributions welcome! Please read:
- [CLAUDE.md](CLAUDE.md) - Development guide
- [SPEC.md](SPEC.md) - Specification and roadmap
