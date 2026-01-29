# bitnet-quantize Development Guide

Microsoft BitNet b1.58 implementation for the rust-ai workspace.

## Architecture Overview

```
bitnet-quantize/
├── src/
│   ├── lib.rs              # Public API, crate docs
│   ├── error.rs            # BitNetError enum
│   ├── config.rs           # BitNetConfig with builder
│   ├── adapter.rs          # peft-rs integration
│   ├── quantization/
│   │   ├── mod.rs          # Re-exports
│   │   ├── weight.rs       # TernaryWeight, AbsMean quantization
│   │   └── activation.rs   # QuantizedActivations, AbsMax
│   ├── layer/
│   │   ├── mod.rs          # Re-exports
│   │   ├── bitlinear.rs    # BitLinear layer
│   │   └── ste.rs          # Straight-Through Estimator
│   ├── kernels/
│   │   ├── mod.rs          # Kernel dispatch
│   │   └── cubecl.rs       # CubeCL GPU kernels (stub)
│   └── export/
│       ├── mod.rs          # Re-exports
│       └── gguf.rs         # GGUF format export
├── benches/
│   └── bitnet_ops.rs       # Criterion benchmarks
├── examples/               # Usage examples
└── tests/                  # Integration tests
```

## Key Algorithms

### Weight Quantization (AbsMean)

```rust
// For each group of weights:
let gamma = weights.abs().mean();
let quantized = (weights / gamma).round().clamp(-1, 1);
// Store: quantized values (2 bits each) + gamma scale (f32)
```

### Activation Quantization (AbsMax)

```rust
// Per-token (or per-tensor):
let gamma = activations.abs().max();
let quantized = (activations * 127.0 / gamma).round().clamp(-127, 127);
// Store: i8 values + gamma scale
```

### Straight-Through Estimator

Forward pass: apply quantization
Backward pass: gradient passes through unchanged (as if identity function)

```rust
// Conceptually:
fn ste_forward(x: Tensor) -> Tensor {
    quantize(x)  // Forward: quantized
}
fn ste_backward(grad: Tensor) -> Tensor {
    grad  // Backward: unchanged
}
```

## Development Commands

```bash
# Run all tests
cargo test -p bitnet-quantize

# Test with peft feature
cargo test -p bitnet-quantize --features peft

# Test with CUDA (requires GPU)
cargo test -p bitnet-quantize --features cuda -- --ignored

# Check for warnings
cargo clippy -p bitnet-quantize -- -W clippy::pedantic

# Run benchmarks
cargo bench -p bitnet-quantize

# Generate docs
cargo doc -p bitnet-quantize --no-deps --open
```

## Feature Flags

| Feature | Description | Dependencies |
|---------|-------------|--------------|
| `cuda` | GPU acceleration | candle-core/cuda |
| `peft` | Adapter trait impl | peft-rs |
| `gguf-export` | GGUF file export | qlora-rs |

## Adding New Features

### New Quantization Method

1. Create new module in `src/quantization/`
2. Implement quantize/dequantize functions
3. Add to `src/quantization/mod.rs` exports
4. Add config options to `BitNetConfig`
5. Add tests and benchmarks

### New Layer Type

1. Create module in `src/layer/`
2. Implement `candle_nn::Module` trait
3. Support both 2D and 3D input tensors
4. Handle bias correctly

## Integration Points

### With trit-vsa

Uses `trit-vsa` for low-level ternary storage and arithmetic.
The `TernaryWeight` struct wraps packed ternary data.

### With peft-rs

When `peft` feature is enabled:
- `BitNetAdapter` implements `peft_rs::Adapter`
- Can be used in PEFT training pipelines
- Supports freeze/unfreeze for fine-tuning

### With qlora-rs

When `gguf-export` feature is enabled:
- Uses qlora-rs GGUF infrastructure
- Exports quantized weights for llama.cpp

## Tensor Shape Conventions

- Weight: `[out_features, in_features]`
- Bias: `[out_features]`
- Input 2D: `[batch_size, in_features]`
- Input 3D: `[batch_size, seq_len, in_features]`
- Output: matches input dimensionality

## Performance Guidelines

- Group size of 64-256 balances accuracy and efficiency
- Per-token scaling is more accurate but slower
- Use forward_quantized() for explicit quantization control
- GPU kernels provide significant speedup for large matrices

## Common Pitfalls

1. **Shape mismatch**: Weight is `[out, in]`, not `[in, out]`
2. **3D input**: Must reshape for matmul, then reshape back
3. **Scale storage**: Scales are f32, not quantized
4. **Gradient flow**: STE only works during training with autograd enabled
5. **Feature gates**: Some functionality requires optional features

## Compatibility

- Minimum Rust: 1.92
- Candle version: 0.9.x
- Workspace dependencies: trit-vsa, peft-rs (optional), qlora-rs (optional)
