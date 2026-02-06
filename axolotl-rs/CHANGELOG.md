# Changelog

All notable changes to axolotl-rs will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.1.1] - 2026-01-24

### Added
- CUDA-first device selection with explicit CPU fallback warnings
- Environment overrides: `AXOLOTL_FORCE_CPU`, `AXOLOTL_CUDA_DEVICE`

### Changed
- Bumped minimum Rust version to 1.92
- README badge alignment cleanup

## [1.1.0] - 2026-01-27

### Added
- **VSA-Accelerated Training**: Integrated `vsa-optim-rs` for deterministic gradient prediction
- `VSAAccelerator` wrapper with configurable training phases (WARMUP → FULL → PREDICT → CORRECT)
- Deterministic phase training with closed-form weighted least squares gradient prediction
- `VSAConfig` for fine-grained control over VSA dimensions, prediction windows, and memory budgets
- Ternary gradient accumulation using balanced `{-1, 0, +1}` representation
- Hyperdimensional bind/bundle/unbind operations for gradient compression
- Comprehensive integration tests for VSA acceleration
- Documentation for `vsa_accel` module with architecture overview

### Changed
- Enhanced `TrainingConfig` with optional `vsa_config` field
- Improved memory efficiency through VSA gradient compression

## [1.0.1] - 2026-01-24

### Fixed
- Fixed `std::path::Path` import missing when `peft` feature enabled
- Fixed `lora_params` variable reference in feature-gated code block
- Compilation now succeeds with `--features "peft,qlora,unsloth"`

## [1.0.0] - 2026-01-24

### Added
- Dynamic CI dependency configuration for sister projects (peft-rs, qlora-rs, unsloth-rs)
- GitHub-based dependency strategy with branch pinning for CI builds
- Comprehensive LoRA target injection tests (per-layer configuration)
- QLoRA training integration tests
- GPU checkpoint save/load tests

### Changed
- Resolved all clippy warnings for production quality
- Updated dependencies to use GitHub branches by default for development
- Improved code organization with dead code annotations for future use

### Fixed
- Unused import and variable warnings cleaned up
- All compilation warnings resolved

---

### Added (from 0.1.0-dev)
- Initial project scaffold with Rust port of Axolotl
- YAML configuration parsing with 3 presets (LLaMA-2, Mistral, Phi-3)
- Dataset loaders for 4 formats: Alpaca, ShareGPT, Completion, Custom
- CLI interface with commands: `validate`, `train`, `merge`, `init`
- Error handling with comprehensive error types
- Mock implementations for PEFT, QLoRA, and Unsloth
- CI/CD pipeline with GitHub Actions
- GPU testing support (CUDA and ROCm)
- Codecov integration with 75% target coverage
- 9 comprehensive benchmarks for config parsing
- Extensive test suite:
  - 18 error handling tests
  - 28 config validation tests  
  - Tests for all dataset formats
  - Trainer lifecycle tests
- MIT license
- Documentation with early development status disclosure
- Contributing guidelines
- Test coverage plan targeting 80%

### Changed
- Updated from candle 0.4 to candle 0.8
- Fixed 54 clippy warnings
- Improved error messages and context
- Enhanced configuration validation

### Fixed
- Compilation issues in initial scaffold
- Workspace manifest configuration
- TemplateError handling in progress bars

## [0.1.0-dev] - 2026-01-10

### Status
**Early Development - Framework Scaffold**

This is an initial development release. The configuration system, CLI, and dataset loaders are functional. Core training functionality (model loading, actual training loops, adapter management, checkpoint handling) is planned for future releases.

**What Works:**
-  YAML configuration parsing and validation
-  Dataset loading (all 4 formats)
-  CLI argument parsing
-  Configuration presets

**What's Planned:**
-  Model loading from HuggingFace Hub
-  LoRA/QLoRA adapter implementation
-  Actual training loop with forward/backward passes
-  Checkpoint saving and loading
-  Adapter merging
-  Multi-GPU distributed training

### Project Metrics
- **Lines of Code**: ~1,500 Rust LOC
- **Test Coverage**: ~60-70% (48+ tests)
- **Dependencies**: Candle 0.8, Tokenizers 0.20, Serde 1.0
- **Platform Support**: Linux, macOS (Windows untested)
- **License**: MIT

### Development Team
- Tyler Zervas (@tzervas) - Primary author

---

## Release History

### Version 0.1.0-dev (January 10, 2026)
Initial development release establishing project structure and core scaffolding.

---

[Unreleased]: https://github.com/tzervas/axolotl-rs/compare/v1.0.1...HEAD
[1.0.1]: https://github.com/tzervas/axolotl-rs/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/tzervas/axolotl-rs/compare/v0.1.0-dev...v1.0.0
[0.1.0-dev]: https://github.com/tzervas/axolotl-rs/releases/tag/v0.1.0-dev
