# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-01-28

### Added
- Local `GpuDispatchable` trait for GPU/CPU kernel dispatch
- Local `GpuError` enum for GPU operation errors
- Local `GpuResult<T>` type alias
- `warn_if_cpu()` function for CPU fallback warnings
- New `gpu::traits` module with all GPU dispatch abstractions
- Re-exports of GPU types from crate root when `cuda` feature enabled

### Changed
- **Breaking**: Removed `rust-ai-core` dependency
- GPU operations now use local `GpuDispatchable` instead of `rust_ai_core::GpuDispatchable`
- `TernaryError::GpuError` now wraps local `GpuError` instead of `rust_ai_core::CoreError`
- Updated all GPU kernel code to use local error types
- Added `tracing` dependency for CPU fallback warnings

### Removed
- `rust-ai-core` dependency (types now defined locally)

## [0.2.0] - 2026-01-25

### Added
- Modular kernel architecture (`TernaryBackend` trait)
- `CpuBackend` with optional SIMD acceleration
- `CubeclBackend` for CUDA GPU acceleration via CubeCL
- `BurnBackend` stub for future Burn framework integration
- `DynamicBackend` wrapper for runtime dispatch
- `BackendConfig` for flexible backend selection
- Smart dispatch with automatic GPU/CPU selection based on vector dimensions
- Convenience GPU wrapper functions in `gpu::mod.rs`

### Changed
- Migrated all CubeCL kernels to 0.9 API
- Fixed bind/unbind kernel formulas (bind uses subtraction, unbind uses addition)
- Added dimension checks to all GPU dispatch operations
- Improved error handling with proper `CoreError` integration

### Fixed
- Kernel position variables now use correct types for CubeCL 0.9
- SharedMemory initialization with proper usize suffix
- Array indexing with explicit usize casts
- `sync_cube()` replaces deprecated `sync_units()`

## [0.1.1] - 2026-01-24

### Changed
- Bumped minimum Rust version to 1.92
- Documentation link fixes and formatting cleanup

## [0.1.0] - 2026-01-24

### Added

- Core `Trit` type with balanced ternary values {-1, 0, +1}
- `Tryte3` type: 3 trits packed in a single byte (range -13 to +13)
- `Word6` type: 6 trits packed in u16 (range -364 to +364)
- Full arithmetic operations: add, multiply, negate for all types
- `PackedTritVec`: Bitsliced vector storage using plus/minus planes
  - Efficient storage: 2 bits per trit
  - SIMD-friendly memory layout
  - Support for dot product, addition, negation
- `SparseVec`: COO (Coordinate) format for high-sparsity vectors
  - Separate positive and negative index lists
  - Efficient for vectors with >90% zeros
- Vector Symbolic Architecture (VSA) operations:
  - `bind`: Association via subtraction mod 3
  - `unbind`: Recovery via addition mod 3
  - `bundle`: Superposition via majority voting
  - `cosine_similarity`: Similarity metric
  - `hamming_distance`: Distance metric
- SIMD acceleration stubs for AVX2 and NEON
- Comprehensive test suite (84 unit tests, 32 doc tests)
- Criterion benchmarks for performance-critical operations
- Serialization support via serde

### Technical Details

- Minimum Rust version: 1.92
- Zero unsafe code in core types
- Thread-safe (all types are Send + Sync)
- No-std compatible (with alloc)

[Unreleased]: https://github.com/tzervas/trit-vsa/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/tzervas/trit-vsa/releases/tag/v0.1.0
