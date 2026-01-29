# tritter-accel Specification v1.0

## Executive Summary

**tritter-accel** is the acceleration layer for the Tritter AI training stack, providing high-performance operations for both Rust and Python developers. It accelerates both ternary (BitNet-style) and conventional neural network workloads.

### Mission

Provide dual Rust/Python APIs for GPU-accelerated AI operations:
- **2-4x inference speedup** via ternary matmul (addition-only arithmetic)
- **16x memory reduction** via 2-bit weight packing
- **10-100x gradient compression** for distributed training
- **Zero-copy data transfer** between Python and Rust
- **Rust-first design** to entice Rust adoption while supporting Python users

### Dual API Philosophy

tritter-accel exposes **two equivalent APIs**:
1. **Rust API** (`tritter_accel::core`): For Rust developers building native applications
2. **Python API** (PyO3 bindings): For Python developers wanting Rust performance

---

## Architecture

### Module Structure

```
tritter-accel/
├── src/
│   ├── lib.rs              # Crate root: exports + Python module
│   ├── core/               # Pure Rust API
│   │   ├── mod.rs          # Core module exports
│   │   ├── ternary.rs      # PackedTernary, matmul
│   │   ├── quantization.rs # quantize_absmean, quantize_absmax
│   │   ├── vsa.rs          # VsaOps (bind, bundle, similarity)
│   │   ├── training.rs     # GradientCompressor, mixed precision
│   │   └── inference.rs    # InferenceEngine, TernaryLayer, KVCache
│   ├── bitnet.rs           # Re-exports from bitnet-quantize
│   ├── ternary.rs          # Re-exports from trit-vsa
│   ├── vsa.rs              # Re-exports from vsa-optim-rs
│   └── gpu.rs              # GPU bindings (cuda feature)
```

### Dependency Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│                    External Consumers                            │
├──────────────────────────┬──────────────────────────────────────┤
│  Python (NumPy/PyTorch)  │        Rust Applications             │
└──────────┬───────────────┴────────────────┬─────────────────────┘
           │ PyO3 bindings                  │ Rust API (rlib)
┌──────────▼────────────────────────────────▼─────────────────────┐
│                    tritter-accel v0.2.0                         │
│                   (cdylib + rlib hybrid)                        │
├─────────────────────────────────────────────────────────────────┤
│  Rust API (core module)          │  Python API (PyO3 bindings)  │
│  ─────────────────────────       │  ───────────────────────────  │
│  • PackedTernary                 │  • pack_ternary_weights()    │
│  • quantize_absmean()            │  • quantize_weights_absmean()│
│  • GradientCompressor            │  • compress_gradients_vsa()  │
│  • InferenceEngine               │  • ternary_matmul()          │
│  • VsaOps                        │  • vsa_bind/bundle/etc       │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────────┐         │
│  │  trit-vsa   │  │   bitnet-   │  │   vsa-optim-rs   │         │
│  │  (ternary)  │  │  quantize   │  │  (compression)   │         │
│  └──────┬──────┘  └──────┬──────┘  └────────┬─────────┘         │
│         │                │                   │                   │
│         └────────────────┼───────────────────┘                   │
│                          │                                       │
│              ┌───────────▼───────────┐                          │
│              │    rust-ai-core       │                          │
│              │  (GpuDispatchable)    │                          │
│              └───────────┬───────────┘                          │
│                          │                                       │
│              ┌───────────▼───────────┐                          │
│              │       CubeCL          │                          │
│              │  (GPU abstraction)    │                          │
│              └───────────────────────┘                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Dual crate-type `[cdylib, rlib]` | Expose both Rust API and Python bindings |
| `core` module for Rust API | Pure Rust, no PyO3 dependencies |
| `trit-vsa` as canonical ternary lib | Avoid code duplication with unsloth-rs |
| CubeCL for GPU kernels | Cross-platform (CUDA/ROCm/Metal/Vulkan) |
| PyO3 for Python bindings | Zero-copy NumPy integration, safe FFI |
| Delegate to sister crates | Thin wrapper, not reimplementation |

---

## API Surface

### Rust API (`tritter_accel::core`)

The Rust API is organized into focused modules:

#### Core Types

```rust
use tritter_accel::core::{
    // Ternary operations
    ternary::{PackedTernary, TernaryMatmulConfig, matmul},

    // Quantization
    quantization::{QuantizeConfig, QuantizationResult, quantize_absmean, quantize_absmax},

    // VSA operations
    vsa::{VsaOps, VsaConfig, DevicePreference},

    // Training acceleration
    training::{GradientCompressor, TrainingConfig, mixed_precision::LossScaler},

    // Inference acceleration
    inference::{InferenceEngine, InferenceConfig, TernaryLayer, KVCache},
};
```

#### Example: Ternary Inference

```rust
use tritter_accel::core::{quantization::quantize_absmean, ternary::matmul};
use candle_core::{Device, Tensor};

fn ternary_forward(input: &Tensor, weight: &Tensor) -> Result<Tensor, Box<dyn std::error::Error>> {
    // Quantize weights to ternary
    let result = quantize_absmean(weight, &Default::default())?;
    let packed = result.to_packed()?;

    // Efficient ternary matmul
    let output = matmul(input, &packed, None)?;
    Ok(output)
}
```

#### Example: Gradient Compression

```rust
use tritter_accel::core::training::{GradientCompressor, TrainingConfig};

fn compress_gradients(gradients: Vec<f32>) -> Vec<f32> {
    let config = TrainingConfig::default().with_compression_ratio(0.1);
    let compressor = GradientCompressor::new(config);

    let compressed = compressor.compress(&gradients, None).unwrap();
    compressor.decompress(&compressed).unwrap()
}
```

#### Example: VSA Operations

```rust
use tritter_accel::core::vsa::{VsaOps, VsaConfig, DevicePreference};

fn vsa_associative_memory() {
    let ops = VsaOps::new(VsaConfig::default().with_device(DevicePreference::Auto));

    // Create random hypervectors
    let apple = ops.random(10000, 1).unwrap();
    let red = ops.random(10000, 2).unwrap();

    // Bind: create "red apple" association
    let red_apple = ops.bind(&apple, &red).unwrap();

    // Query: what color is the apple?
    let color = ops.unbind(&red_apple, &apple).unwrap();

    // Similarity should be high
    let similarity = ops.cosine_similarity(&color, &red).unwrap();
    assert!(similarity > 0.9);
}
```

---

### Python API (`tritter_accel` module)

#### Quantization Operations

```python
def quantize_weights_absmean(
    weights: np.ndarray,  # shape: (out_features, in_features), dtype: float32
) -> tuple[np.ndarray, np.ndarray]:
    """
    Quantize weights to ternary {-1, 0, +1} using BitNet AbsMean method.

    Returns:
        ternary_weights: shape (out_features, in_features), dtype: float32 (values in {-1, 0, 1})
        scales: shape (out_features,), dtype: float32

    Delegates to: bitnet_quantize::quantize_weights()
    """

def quantize_activations_absmax(
    activations: np.ndarray,  # shape: (batch, features), dtype: float32
) -> tuple[np.ndarray, np.ndarray]:
    """
    Quantize activations to INT8 using AbsMax scaling.

    Returns:
        quantized: shape (batch, features), dtype: int8
        scales: shape (batch,), dtype: float32

    Delegates to: bitnet_quantize::quantize_activations()
    """
```

#### Packing Operations

```python
def pack_ternary_weights(
    weights: np.ndarray,  # shape: (rows, cols), dtype: float32, values in {-1, 0, 1}
    scales: np.ndarray,   # shape: (rows,), dtype: float32
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pack ternary weights into 2-bit representation (4 trits per byte).

    Returns:
        packed: shape (rows * ceil(cols/4),), dtype: uint8
        scales: shape (rows,), dtype: float32

    Delegates to: trit_vsa::PackedTritVec
    """

def unpack_ternary_weights(
    packed: np.ndarray,   # shape: (packed_size,), dtype: uint8
    scales: np.ndarray,   # shape: (rows,), dtype: float32
    shape: tuple[int, int],  # (rows, cols)
) -> np.ndarray:
    """
    Unpack ternary weights and apply scales.

    Returns:
        weights: shape (rows, cols), dtype: float32

    Delegates to: trit_vsa::PackedTritVec
    """
```

#### Matrix Operations

```python
def ternary_matmul(
    input: np.ndarray,         # shape: (batch, in_features), dtype: float32
    packed_weights: np.ndarray, # packed ternary weights
    scales: np.ndarray,        # shape: (out_features,), dtype: float32
    weight_shape: tuple[int, int],  # (out_features, in_features)
    device: str = "auto",      # "cpu", "cuda", or "auto"
) -> np.ndarray:
    """
    Efficient matrix multiplication with packed ternary weights.
    Uses only addition/subtraction (no multiplications).

    Returns:
        output: shape (batch, out_features), dtype: float32

    Delegates to: trit_vsa GPU/CPU dispatch via GpuDispatchable
    """
```

#### VSA Operations

```python
def compress_gradients_vsa(
    gradients: np.ndarray,     # shape: (num_params,), dtype: float32
    compression_ratio: float,  # target ratio (0.01 to 1.0)
    seed: int,                 # for reproducibility
) -> tuple[np.ndarray, int]:
    """
    Compress gradients using VSA random projection.

    Returns:
        compressed: shape (compressed_dim,), dtype: float32
        seed: int (for decompression)

    Delegates to: vsa_optim_rs::VSAGradientCompressor
    """

def decompress_gradients_vsa(
    compressed: np.ndarray,  # shape: (compressed_dim,), dtype: float32
    original_dim: int,
    seed: int,
) -> np.ndarray:
    """
    Decompress gradients (approximate reconstruction).

    Returns:
        gradients: shape (original_dim,), dtype: float32

    Delegates to: vsa_optim_rs::VSAGradientCompressor
    """
```

#### GPU VSA Operations (NEW)

```python
def vsa_bind(
    a: np.ndarray,  # shape: (dim,), ternary values
    b: np.ndarray,  # shape: (dim,), ternary values
    device: str = "auto",
) -> np.ndarray:
    """
    Ternary bind operation (association).

    Delegates to: trit_vsa::gpu::GpuBind
    """

def vsa_unbind(
    bound: np.ndarray,
    key: np.ndarray,
    device: str = "auto",
) -> np.ndarray:
    """
    Ternary unbind operation (recovery).

    Delegates to: trit_vsa::gpu::GpuUnbind
    """

def vsa_bundle(
    vectors: list[np.ndarray],  # list of ternary vectors
    device: str = "auto",
) -> np.ndarray:
    """
    Bundle (superposition via majority voting).

    Delegates to: trit_vsa::gpu::GpuBundle
    """

def vsa_similarity(
    a: np.ndarray,
    b: np.ndarray,
    metric: str = "cosine",  # "cosine", "dot", or "hamming"
    device: str = "auto",
) -> float:
    """
    Compute similarity between ternary vectors.

    Delegates to: trit_vsa::gpu::GpuCosineSimilarity / GpuDotSimilarity / GpuHammingDistance
    """
```

#### Device Management

```python
def get_device() -> str:
    """Return current device: 'cpu' or 'cuda:N'"""

def set_device(device: str) -> None:
    """Set default device for operations."""

def cuda_available() -> bool:
    """Check if CUDA is available."""

def version() -> str:
    """Return tritter-accel version."""
```

---

## Success Criteria

### Performance Targets

| Operation | Metric | Target | Measurement |
|-----------|--------|--------|-------------|
| Ternary matmul (CPU) | vs FP32 matmul | 2x speedup | Criterion benchmark |
| Ternary matmul (GPU) | vs FP32 matmul | 4x speedup | Criterion benchmark |
| Weight packing | Memory reduction | 16x | sizeof(packed) / sizeof(float) |
| VSA compression | Compression ratio | 10-100x | compressed_size / original_size |
| VSA compression | Accuracy loss | <5% | cosine_similarity(original, reconstructed) |
| GPU bind/bundle | vs CPU | 10x speedup @ 10K dims | Criterion benchmark |

### Quality Targets

| Criterion | Target | Verification |
|-----------|--------|--------------|
| Test coverage | >80% line coverage | `cargo tarpaulin` |
| Documentation | 100% public API | `cargo doc --no-deps` |
| Clippy warnings | Zero warnings | `cargo clippy -- -D warnings` |
| Python examples | 5+ runnable examples | `python examples/*.py` |
| Benchmarks | All operations covered | `cargo bench` |

### Compatibility Targets

| Platform | Status | Notes |
|----------|--------|-------|
| Linux x86_64 + CUDA | Primary | RTX 3090/4090/5080 tested |
| Linux x86_64 CPU | Supported | AVX2/SSE4.2 acceleration |
| macOS ARM64 | Future | Metal backend via CubeCL |
| Windows x86_64 | Future | CUDA backend |
| Linux + ROCm | Future | AMD GPU via CubeCL |

---

## Implementation Phases

### Phase 1: Foundation (v0.2.0)

**Goal**: Replace inline implementations with calls to sister crates.

| Task | Effort | Dependency |
|------|--------|------------|
| Refactor `quantize_weights_absmean` to call `bitnet_quantize` | S | None |
| Refactor `pack/unpack` to call `trit_vsa::PackedTritVec` | M | None |
| Refactor `compress/decompress_gradients` to call `vsa_optim_rs` | M | None |
| Add `trit-vsa` and `bitnet-quantize` as dependencies | S | None |
| Update tests to verify delegation | S | Above tasks |
| Remove stub modules (bitnet.rs, ternary.rs, vsa.rs) or populate | S | Above tasks |

**Deliverables**:
- All functions delegate to sister crates
- Zero inline implementations
- Tests verify correctness

### Phase 2: GPU Acceleration (v0.3.0)

**Goal**: Expose GPU operations via Python bindings.

| Task | Effort | Dependency |
|------|--------|------------|
| Add `cuda` feature flag | S | None |
| Expose `GpuBind`, `GpuUnbind`, `GpuBundle` from trit-vsa | M | Phase 1 |
| Expose `GpuDotSimilarity`, `GpuCosineSimilarity` | M | Phase 1 |
| Add `ternary_matmul` GPU path via CubeCL | L | Phase 1 |
| Add device management functions | S | None |
| GPU integration tests (marked `#[ignore]`) | M | Above tasks |

**Deliverables**:
- GPU-accelerated VSA operations callable from Python
- `device` parameter on all ops
- GPU benchmarks

### Phase 3: Production Polish (v0.4.0)

**Goal**: Production-ready quality and documentation.

| Task | Effort | Dependency |
|------|--------|------------|
| Implement `accel_bench.rs` with real benchmarks | M | Phase 2 |
| Create `examples/` directory with Python scripts | M | Phase 2 |
| Add integration tests for Python workflows | M | Phase 2 |
| Achieve 100% public API documentation | M | None |
| Zero clippy warnings | S | None |
| Add CI/CD (GitHub Actions) | M | None |

**Deliverables**:
- Benchmark results in README
- 5+ runnable Python examples
- CI passing on every PR

### Phase 4: Ecosystem Integration (v0.5.0)

**Goal**: Integrate with broader training stack.

| Task | Effort | Dependency |
|------|--------|------------|
| Expose `DeterministicPhaseTrainer` from vsa-optim-rs | L | Phase 3 |
| Add `BitLinear` layer wrapper | M | Phase 3 |
| Integration with axolotl-rs training loop | L | Phase 3 |
| Distributed training example (multi-GPU) | L | Phase 3 |

**Deliverables**:
- End-to-end training example
- Integration tests with full stack

### Phase 5: v1.0 Release

**Goal**: Stable public API.

| Criterion | Required |
|-----------|----------|
| All Phase 1-4 complete | Yes |
| Semantic versioning | Yes |
| CHANGELOG.md | Yes |
| Published to PyPI | Yes |
| Published to crates.io | Yes |

---

## Unification Plan: unsloth-rs Ternary

The `unsloth-rs/src/kernels/ternary/` module duplicates `trit-vsa` functionality.

### Recommended Approach

1. **Keep both for now** - they serve different purposes:
   - `trit-vsa`: VSA/hyperdimensional computing primitives
   - `unsloth-rs/ternary`: Transformer-specific kernels (attention, linear)

2. **Share core types** - unsloth-rs should depend on trit-vsa for:
   - `PackedTritVec` (storage)
   - `Trit`, `Tryte3`, `Word6` (arithmetic)

3. **Unique to unsloth-rs**:
   - `TernaryLinear` (layer)
   - `ternary_attention_cpu` (transformer kernel)
   - Model quantization utilities

4. **Migration path**:
   ```rust
   // unsloth-rs/src/kernels/ternary/types.rs
   // Before:
   pub struct TernaryPlanes { plus: Vec<u64>, minus: Vec<u64> }

   // After:
   pub use trit_vsa::PackedTritVec as TernaryPlanes;
   ```

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| CubeCL API instability | Medium | High | Pin version, test on update |
| GPU kernel numerical issues | Medium | Medium | Extensive CPU/GPU equivalence tests |
| PyO3 memory leaks | Low | Medium | Use `PyReadonlyArray`, avoid raw pointers |
| Performance regression | Medium | Medium | Benchmark CI gate |
| Python version compatibility | Low | Low | Test on 3.9, 3.10, 3.11, 3.12 |

---

## Appendix: File Structure (Current)

```
tritter-accel/
├── Cargo.toml
├── README.md
├── SPEC.md                    # This document
├── CLAUDE.md                  # Development guide
├── CHANGELOG.md
├── LICENSE-MIT
├── pyproject.toml             # maturin build config
├── src/
│   ├── lib.rs                 # Crate root: exports + Python bindings
│   ├── core/                  # Pure Rust API
│   │   ├── mod.rs             # Core module exports
│   │   ├── ternary.rs         # PackedTernary, matmul, dot
│   │   ├── quantization.rs    # quantize_absmean, quantize_absmax
│   │   ├── vsa.rs             # VsaOps with bind/bundle/similarity
│   │   ├── training.rs        # GradientCompressor, mixed_precision
│   │   └── inference.rs       # InferenceEngine, TernaryLayer, KVCache
│   ├── bitnet.rs              # Re-exports from bitnet-quantize
│   ├── ternary.rs             # Re-exports from trit-vsa
│   ├── vsa.rs                 # Re-exports from vsa-optim-rs
│   └── gpu.rs                 # GPU VSA bindings (cuda feature)
├── benches/
│   └── accel_bench.rs         # Criterion benchmarks
├── examples/
│   ├── basic_quantization.py
│   ├── ternary_inference.py
│   ├── gradient_compression.py
│   ├── vsa_operations.py
│   └── gpu_acceleration.py
└── tests/
    ├── test_quantization.py
    ├── test_matmul.py
    ├── test_vsa.py
    └── integration_tests.rs
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 0.1.0 | 2025-01 | Initial release with inline implementations |
| 0.1.1 | 2025-01 | Bump to Rust 1.92, fix dependencies |
| 0.2.0 | 2025-01 | Phase 1: Delegate to sister crates |
| 0.2.1 | 2026-01 | Add Rust API (`core` module), dual crate-type |
| 0.3.0 | TBD | Phase 2: GPU acceleration |
| 0.4.0 | TBD | Phase 3: Production polish |
| 0.5.0 | TBD | Phase 4: Ecosystem integration |
| 1.0.0 | TBD | Stable release |
