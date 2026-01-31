# CLAUDE.md - Development Context for hybrid-predict-trainer-rs

This document provides context for Claude Code and other AI assistants working on this crate.

## Project Overview

**hybrid-predict-trainer-rs** is a Rust crate implementing hybridized predictive training that achieves significant training speedups by intelligently predicting training steps instead of computing full forward/backward passes.

### Core Concept

The training loop cycles through four phases:
1. **Warmup**: Collect baseline statistics (loss, gradient norms)
2. **Full Train**: Standard training + dynamics model learning
3. **Predict**: Skip backward passes using learned dynamics
4. **Correct**: Apply residual corrections to predictions

Target: **5-10x training speedup** with <2% loss quality degradation.

## Architecture

### Module Structure

```
src/
├── lib.rs          # Main entry, HybridTrainer struct
├── config.rs       # HybridTrainerConfig, PredictorConfig, DivergenceConfig
├── error.rs        # Error types, RecoveryAction, DivergenceLevel
├── state.rs        # TrainingState, RingBuffer, StateEncoder, WeightDelta
├── phases.rs       # Phase enum, PhaseController, PhaseOutcome
├── warmup.rs       # WarmupExecutor, WarmupStatistics
├── full_train.rs   # FullTrainExecutor, GradientObservation
├── predictive.rs   # PredictiveExecutor, PhasePrediction
├── residuals.rs    # Residual, ResidualStore, CompressedResidual
├── corrector.rs    # ResidualCorrector, Correction, CorrectionExecutor
├── dynamics.rs     # RSSMLite, LatentState, DynamicsModel trait
├── divergence.rs   # DivergenceMonitor, SignalResult
├── bandit.rs       # BanditSelector, Arm, LinUCB implementation
├── metrics.rs      # MetricsCollector, StepMetrics, TrainingStatistics
└── gpu.rs          # GpuAccelerator, CubeCL kernels (feature-gated)
```

### Key Traits

- `PhaseController`: Decides when to transition between phases
- `DynamicsModel`: Predicts training trajectories
- `StateEncoder`: Encodes training state to features
- `ResidualExtractor`: Extracts residuals from predictions vs reality
- `CorrectionStrategy`: Computes corrections from residuals

### Key Types

```rust
// Main trainer
HybridTrainer<M, O> where M: Model, O: Optimizer

// Phase state machine
enum Phase { Warmup, Full, Predict, Correct }

// Training state representation
TrainingState { step, loss, gradient_norm, history... }

// Prediction output
PhasePrediction { weight_delta, predicted_final_loss, confidence, bounds }

// Correction output
Correction { loss_correction, weight_correction, confidence }
```

## Implementation Status

### Completed (Boilerplate)

- [x] Cargo.toml with dependencies
- [x] All module stubs with types and traits
- [x] Documentation and tests for each module
- [x] Error handling with recovery actions
- [x] Configuration with builder pattern

### TODO (Implementation)

- [ ] Integrate with actual Burn model/optimizer types
- [ ] Implement CubeCL CUDA kernels
- [ ] Wire up full training loop in `HybridTrainer::step()`
- [ ] Add checkpoint save/restore
- [ ] Benchmarks with real models
- [ ] Integration tests

## Dependencies

| Crate | Version | Purpose |
|-------|---------|---------|
| burn | 0.20.1 | Deep learning framework |
| cubecl | 0.9.0 | GPU compute |
| serde | 1.0.228 | Serialization |
| thiserror | 2.0.18 | Error types |
| rand | (latest) | Random number generation |
| half | 2.7.1 | f16/bf16 support |
| tokio | 1.49.0 | Async (optional) |

## Development Commands

```bash
# Build
cargo build

# Build with CUDA
cargo build --features cuda

# Test
cargo test

# Test specific module
cargo test divergence::tests

# Documentation
cargo doc --open

# Benchmarks
cargo bench

# Lint
cargo clippy --all-features
```

## Code Style Guidelines

1. **Documentation**: Every public item needs doc comments with examples
2. **Error Handling**: Use `HybridResult<T>` which includes recovery suggestions
3. **Traits**: Define traits for extensibility (new predictors, backends)
4. **Feature Gates**: GPU code behind `cuda` feature
5. **Tests**: Unit tests in each module, integration tests in `tests/`

## Phase Transition Logic

```
Warmup → Full:
  - warmup_steps completed
  - Statistics baseline established

Full → Predict:
  - min_full_steps completed
  - Predictor confidence > threshold
  - No recent divergence

Predict → Correct:
  - prediction_horizon reached OR
  - Divergence detected OR
  - Confidence dropped

Correct → Full:
  - Validation samples processed
  - Corrections applied
  - (cycle continues)
```

## Divergence Detection

Multiple signals monitored:
- Loss deviation (σ from EMA)
- Gradient norm explosion (>10x baseline)
- Gradient vanishing (<0.01x baseline)
- NaN/Inf values
- Prediction error (>20% relative)
- Loss oscillation (sign change frequency)

## Research References

1. **RSSM Architecture**: DreamerV3 (Hafner 2023)
2. **Gradient Compression**: PowerSGD (Vogels 2019)
3. **Bandit Algorithms**: LinUCB (Li 2010)
4. **Online Learning**: Follow the Regularized Leader

## Common Tasks

### Adding a New Predictor

1. Implement `DynamicsModel` trait in `dynamics.rs`
2. Add variant to `PredictorConfig` enum
3. Update `HybridTrainer::new()` to construct it
4. Add tests

### Adding a New Phase

1. Add variant to `Phase` enum
2. Implement executor in new module
3. Update `PhaseController` trait
4. Update `HybridTrainer::step()` state machine

### Adding GPU Kernel

1. Define kernel in `gpu::kernels`
2. Implement in CubeCL syntax
3. Add wrapper in `GpuAccelerator`
4. Gate with `#[cfg(feature = "cuda")]`

## Testing Strategy

- **Unit tests**: Each module tests its types in isolation
- **Integration tests**: End-to-end training scenarios
- **Benchmarks**: Criterion benchmarks for performance-critical paths
- **Property tests**: Consider proptest for invariants

## Contact

- **Author**: Tyler Zervas (tzervas)
- **Email**: tz-dev@vectorweight.com
- **License**: MIT
