# tritter-accel Development Guide

Python acceleration bindings for the Tritter AI training stack.

## Subagent Context Management

When working on this crate with subagents:

- **Autocompact at 98%**: Compact context automatically when usage reaches 98%
- **Focused scope**: Each subagent should handle one specific task
- **Concise output**: Return summaries, not raw file contents
- **Parallel execution**: Independent tasks should run in parallel

## Architecture Overview

```
tritter-accel/
├── src/
│   ├── lib.rs           # PyO3 module definition + main functions
│   ├── device.rs        # Device management (planned)
│   ├── quantization.rs  # Wrapper for bitnet-quantize (planned)
│   ├── packing.rs       # Wrapper for trit-vsa packing (planned)
│   ├── matmul.rs        # Ternary matmul with GPU dispatch (planned)
│   ├── vsa.rs           # Wrapper for vsa-optim-rs + GPU ops (planned)
│   └── error.rs         # Python-compatible errors (planned)
├── benches/
│   └── accel_bench.rs   # Criterion benchmarks (placeholder)
├── examples/            # Python examples (planned)
└── tests/               # Integration tests (planned)
```

## Current State

**Version**: 0.1.1
**Status**: Inline implementations, needs refactoring

Current `lib.rs` has inline implementations that should delegate to:
- `trit-vsa` for ternary packing/unpacking and VSA ops
- `bitnet-quantize` for weight/activation quantization
- `vsa-optim-rs` for gradient compression

## Development Commands

```bash
# Build with maturin
cd tritter-accel
maturin develop --release

# Run Rust tests
cargo test -p tritter-accel

# Build with CUDA (when implemented)
maturin develop --release --features cuda

# Run benchmarks (when implemented)
cargo bench -p tritter-accel

# Check for warnings
cargo clippy -p tritter-accel -- -D warnings
```

## Key Dependencies

| Crate | Purpose |
|-------|---------|
| `pyo3` | Python bindings |
| `numpy` | NumPy array integration |
| `trit-vsa` | Ternary arithmetic, VSA, GPU ops |
| `bitnet-quantize` | Weight/activation quantization |
| `vsa-optim-rs` | Gradient compression |
| `candle-core` | Tensor operations |

## API Design Principles

1. **Delegate, don't reimplement**: All ops should call sister crates
2. **Zero-copy where possible**: Use `PyReadonlyArray` for input
3. **GPU-first with CPU fallback**: Check device, dispatch accordingly
4. **Deterministic by default**: Seeded RNG for reproducibility

## Implementation Roadmap

See `SPEC.md` for detailed phases:

1. **Phase 1 (v0.2.0)**: Delegate to sister crates
2. **Phase 2 (v0.3.0)**: GPU acceleration
3. **Phase 3 (v0.4.0)**: Production polish
4. **Phase 4 (v0.5.0)**: Ecosystem integration
5. **v1.0.0**: Stable release

## Testing Strategy

- **Unit tests**: Verify delegation to sister crates
- **Integration tests**: Python workflows end-to-end
- **GPU tests**: Marked `#[ignore]`, require CUDA
- **Benchmarks**: Criterion for performance regression

## Common Pitfalls

1. **PyO3 lifetime management**: Use `Bound<'py, T>` for owned Python objects
2. **NumPy array ownership**: Use `PyReadonlyArray` to avoid copies
3. **GPU memory**: Ensure tensors are dropped after use
4. **Feature gates**: CUDA ops require `--features cuda`

## Compatibility

- Minimum Rust: 1.92
- Python: 3.9, 3.10, 3.11, 3.12
- Candle version: 0.9.x
- CubeCL version: 0.8.1 (for GPU)
