# Changelog

All notable changes to tritter-accel will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.1] - 2026-01-28

### Changed
- Removed `rust-ai-core` dependency
- GPU types now imported from `trit_vsa::gpu` instead of `rust_ai_core`
- CUDA feature simplified to only enable `trit-vsa/cuda`
- Updated to PyO3 0.27 API: `into_py()` → `into_pyobject().unwrap().into_any().unbind()`
- Updated to numpy 0.27 API: `to_pyarray_bound()` → `to_pyarray()`
- Replaced deprecated `PyObject` with `Py<PyAny>`

## [0.2.0] - 2026-01-25

### Added

- **Rust API** (`tritter_accel::core` module)
  - `core::ternary`: `PackedTernary`, `matmul()`, `dot()`, packed bytes conversion
  - `core::quantization`: `quantize_absmean()`, `quantize_absmax()`, `QuantizationResult`
  - `core::vsa`: `VsaOps` with bind/unbind/bundle/similarity, device dispatch
  - `core::training`: `GradientCompressor`, `TernaryCompressedGradient`, `GradientAccumulator`, `mixed_precision::LossScaler`
  - `core::inference`: `InferenceEngine`, `TernaryLayer`, `KVCache`

- **Dual crate-type** (`cdylib` + `rlib`)
  - Exposes Rust API for native integration
  - Maintains Python bindings via PyO3

- **Comprehensive benchmarks** (550+ lines)
  - Quantization benchmarks (absmean, group sizes)
  - Packing benchmarks (pack/unpack)
  - Matmul benchmarks (ternary vs float comparison)
  - VSA compression benchmarks
  - VSA operations benchmarks (bind, bundle, similarity)

- **CI/CD** (GitHub Actions)
  - Automated check, test, docs, and Python wheel build

### Changed

- Refactored to delegate all operations to sister crates
- `quantize_weights_absmean` now uses `bitnet_quantize::quantize_weights()`
- Pack/unpack functions now use `trit_vsa::PackedTritVec`
- Updated documentation with Rust API examples

### Dependencies

- Added `candle-nn` for inference utilities

## [0.1.1] - 2026-01-24

### Added
- Documentation and architecture diagram cleanup

### Changed
- Bumped minimum Rust version to 1.92

## [0.1.0] - 2026-01-24

### Added

- **Ternary Weight Operations**
  - `pack_ternary_weights`: Pack ternary {-1, 0, +1} weights into 2-bit representation
  - `unpack_ternary_weights`: Unpack to dequantized float weights
  - `ternary_matmul`: Efficient matrix multiply with packed ternary weights
  - `ternary_matmul_simple`: Simple matmul with float ternary weights

- **Quantization**
  - `quantize_weights_absmean`: AbsMean quantization (BitNet b1.58 style)

- **VSA Gradient Compression**
  - `compress_gradients_vsa`: Compress gradients using Vector Symbolic Architecture
  - `decompress_gradients_vsa`: Decompress gradients from VSA representation

- **Python Bindings**
  - PyO3-based bindings for all operations
  - NumPy array integration
  - Zero-copy where possible

### Dependencies

- `bitnet-quantize` v0.1.0 for BitNet operations
- `trit-vsa` v0.1.0 for ternary VSA primitives
- `vsa-optim-rs` v0.1.0 for gradient compression
- `candle-core` v0.9 for tensor operations
- `pyo3` v0.22 for Python bindings
- `numpy` v0.22 for array interface
