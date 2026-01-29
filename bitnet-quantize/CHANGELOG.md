# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.1] - 2026-01-28

### Changed
- Removed `rust-ai-core` dependency
- CUDA feature now enables `trit-vsa/cuda` for GPU types

## [0.2.0] - 2026-01-25

### Changed
- Migrated CubeCL kernels to 0.9 API
- Fixed Result type collision by using std::result::Result explicitly
- Replace early returns with `terminate!()` macro
- Update math functions to use trait methods (`F::floor()`, `F::sqrt()`)
- Added `usize` suffix to SharedMemory::new() calls
- Added proper usize casts at array index sites
- Wrapped kernel launches in unsafe blocks with SAFETY comments

### Known Limitations
- GPU kernel launches not yet integrated into public API (CPU fallback used)
- Kernel definitions ready for future GPU integration

## [0.1.1] - 2026-01-24

### Added
- CPU fallback warning when CUDA is unavailable

### Changed
- Bumped minimum Rust version to 1.92
- Documentation link fixes and formatting cleanup

## [0.1.0] - 2026-01-24

### Added

- `BitNetConfig`: Configuration for BitNet quantization
  - Configurable group size for weight quantization
  - Per-token or per-tensor activation scaling
  - Training mode with STE support
- `TernaryWeight`: Packed ternary weight storage
  - AbsMean quantization: `W_q = round(W / mean(|W|))`
  - Per-group scale factors
  - Compression tracking and sparsity metrics
- `QuantizedActivations`: INT8 activation quantization
  - AbsMax quantization: `X_q = round(X * 127 / max(|X|))`
  - Per-token scaling for sequence models
  - Efficient dequantization
- `BitLinear`: Drop-in replacement for `nn::Linear`
  - Compatible with candle-nn Module trait
  - Supports 2D and 3D input tensors
  - Optional bias term
  - Forward pass with automatic dequantization
  - `forward_quantized` for explicit quantization control
- Straight-Through Estimator (STE) functions
  - `ternary_ste`: Forward quantization with gradient passthrough
  - `int8_ste`: INT8 quantization with gradient passthrough
- peft-rs adapter integration (optional, `peft` feature)
  - `BitNetAdapter` implementing `Adapter` trait
  - Configuration via `BitNetAdapterConfig`
- GGUF export support (optional, `gguf-export` feature)
- CubeCL GPU kernel stubs (optional, `cuda` feature)
- Comprehensive test suite (35 unit tests)
- Criterion benchmarks for quantization and forward pass

### Technical Details

- Built on candle 0.9.x tensor library
- Minimum Rust version: 1.92
- Optional dependencies gated behind feature flags
- Integration with rust-ai workspace

### References

- BitNet b1.58: "The Era of 1-bit LLMs" (Ma et al., 2024)
- Original BitNet: "Scaling 1-bit Transformers" (Wang et al., 2023)

[Unreleased]: https://github.com/tzervas/bitnet-quantize/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/tzervas/bitnet-quantize/releases/tag/v0.1.0
