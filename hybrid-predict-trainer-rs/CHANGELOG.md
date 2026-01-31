# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-01-30

### Added
- Complete hybrid predictive training framework with four phases: Warmup, Full, Predict, Correct
- RSSM-lite dynamics model for training trajectory prediction
- GRU cell implementation with Xavier initialization
- Multi-signal divergence detection (loss spikes, gradient explosion, NaN detection)
- LinUCB bandit for adaptive phase selection
- Comprehensive metrics collection and JSON export
- Configuration builder pattern with validation
- Error types with recovery action suggestions
- 70+ unit tests and 28 integration tests
- GitHub Actions CI/CD workflows (ci.yml, security.yml, release.yml)
- Security policy (SECURITY.md) and dependency audit config (deny.toml)

### Documentation
- Complete API documentation with examples
- README with quick start guide
- CLAUDE.md development context
- Research documentation in docs/research/

### Features
- `cuda` - GPU acceleration via CubeCL
- `candle` - Candle tensor integration
- `async` - Tokio async support
- `full` - All features enabled

## [0.0.1] - 2026-01-30

### Added
- Initial release with project structure and boilerplate
- Core module stubs for all major components:
  - `config`: Training configuration and serialization
  - `error`: Comprehensive error types with recovery actions
  - `phases`: Phase state machine and execution control
  - `warmup`: Warmup phase implementation
  - `full_train`: Full training phase implementation
  - `predictive`: Forward/backward predictive training
  - `residuals`: Residual extraction and storage
  - `corrector`: Prediction correction via residual application
  - `state`: Training state encoding and management
  - `dynamics`: RSSM-lite dynamics model for prediction
  - `divergence`: Multi-signal divergence detection
  - `metrics`: Training metrics collection and reporting
  - `gpu`: CubeCL and Burn GPU acceleration kernels
- MIT license
- README with architecture overview
- CLAUDE.md development context document
- Benchmark scaffolding
- Example programs

[Unreleased]: https://github.com/tzervas/hybrid-predict-trainer-rs/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/tzervas/hybrid-predict-trainer-rs/compare/v0.0.1...v0.1.0
[0.0.1]: https://github.com/tzervas/hybrid-predict-trainer-rs/releases/tag/v0.0.1
