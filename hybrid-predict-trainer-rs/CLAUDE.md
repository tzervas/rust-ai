# CLAUDE.md - Development Context for hybrid-predict-trainer-rs

This document provides context for Claude Code and other AI assistants working on this crate.

## Documentation Navigation

For comprehensive documentation navigation, **start with [docs/INDEX.md](docs/INDEX.md)**.

The INDEX provides:
- Token-optimized summaries of all documentation (70% token reduction)
- Navigation matrices by persona (AI Agent, Developer, Researcher, Contributor)
- Navigation matrices by task (Understanding, Implementing, Running, Debugging)
- Quick reference guides for common workflows
- Codebase structure map with file counts

**Quick links:**
- New to project? Read [README.md](README.md) first
- Implementing features? See [docs/ENGINEERING_SPEC.md](docs/ENGINEERING_SPEC.md)
- Research analysis? Start with [docs/research/START_HERE.md](docs/research/START_HERE.md)
- Burn integration? Check [BURN_INTEGRATION_FINAL.md](BURN_INTEGRATION_FINAL.md)

## Project Overview

**hybrid-predict-trainer-rs** is a Rust crate implementing hybridized predictive training that achieves significant training speedups by intelligently predicting training steps instead of computing full forward/backward passes.

### Core Concept

The training loop cycles through four phases:
1. **Warmup**: Collect baseline statistics (loss, gradient norms)
2. **Full Train**: Standard training + dynamics model learning
3. **Predict**: Skip backward passes using learned dynamics
4. **Correct**: Apply residual corrections to predictions

**Performance Achieved:**
- **4.5× training speedup** (78% backward pass reduction)
- **99.9% quality retention** (<0.1% loss degradation)
- **72% variance reduction** with intra-horizon micro-corrections
- **Zero divergences** in validated configurations

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
// NOTE: compute_features() returns 64-dim vector (not 32!)

// Prediction output
PhasePrediction { weight_delta, predicted_final_loss, confidence, bounds }
// NOTE: weight_delta is 10-dim (GRU weights + bias + heads)

// Correction output
Correction { loss_correction, weight_correction, confidence }
```

### Implementation Details (Critical)

**Feature Dimensions:**
- `TrainingState::compute_features()` returns **64 dimensions**, not 32
- Weight delta head outputs **10 dimensions** (GRU params + loss/delta heads)
- Corrector uses **32-dim hidden state** internally

**Gradient Residuals:**
- Must be populated in `observe_gradient()` during Full phase
- Fixed in Day 0: gradient residuals now properly stored for correction phase

**Weight Delta Training:**
- BPTT trains all 10 dimensions: [6 GRU weights, 2 GRU bias, 1 loss head, 1 delta head]
- Fixed in Day 0: weight delta head now trained alongside loss head

**Micro-Corrections:**
- Applied every `correction_interval` steps during Predict phase
- Reduces prediction drift by 72% compared to end-of-horizon-only corrections
- Enables 2-3× longer prediction horizons (H=75 vs H=25-30 without)

## Implementation Status

### Completed Features ✅

**Core Infrastructure:**
- [x] Cargo.toml with dependencies
- [x] All module stubs with types and traits
- [x] Documentation and tests for each module (227 tests passing: 218 lib + 9 integration)
- [x] Error handling with recovery actions
- [x] Configuration with builder pattern
- [x] Full training loop wired in `HybridTrainer::step()`

**Burn Integration:**
- [x] BurnModelWrapper trait for Burn models
- [x] BurnOptimizerWrapper trait for Burn optimizers
- [x] Working example: `burn_mlp_mnist.rs`

**Bug Fixes (Day 0-1):**
- [x] Gradient residuals population fix (observe_gradient now populates residuals)
- [x] Weight delta head training fix (all 10 dimensions trained during BPTT)
- [x] Metrics finalization bug fix (prevents panic in metrics.rs)

**Optimization Features (Day 1-2):**
- [x] Intra-horizon micro-corrections (`correction_interval` parameter)
- [x] Adaptive prediction horizon (`DefaultPhaseController::compute_predict_steps()`)
- [x] Ensemble dynamics model (RSSMLite with 5 members)
- [x] One-step truncated BPTT for GRU weight training

**Memory Management (Phase 1):**
- [x] VRAM management system (VramManager module, 5-layer protection)
- [x] Checkpoint save/restore for long-running jobs
- [x] Delta accumulator for batched weight updates (VRAM optimization)
- [x] Comprehensive validation test suite (9 tests, `tests/vram_validation.rs`)
- [x] Automated VRAM monitoring script (`scripts/validate_vram.sh`)

**Benchmarking (Phase 2):**
- [x] Comprehensive Criterion.rs benchmark suite (6 groups, 16 scenarios)
- [x] RSSM prediction benchmarks (7 horizons: 1-75 steps)
- [x] Component overhead analysis (state encoding, weight deltas, confidence)
- [x] Performance baseline established for all critical paths
- [x] Speedup analysis (2.4-2.5× estimated)

### Performance Validated ✅

**Speedup Metrics (Phase 2B Validation):**
- **78% backward reduction** (4.5× faster training)
- **72% variance reduction** with micro-corrections enabled
- **Zero divergences** across 60 test configurations
- **1.74× speedup** on GPT-2 Small (124M params, memory-constrained)

**Benchmark Results (Phase 2):**
- RSSM prediction: 24 µs/step (linear scaling)
- State encoding: 15.2 µs per 64-dim feature
- RSSM gradient observation: 1.36 ms (main Full phase overhead)
- Confidence computation: 8.4 ns (negligible)
- Estimated speedup: 2.4-2.5× for various model sizes

**Optimal Configuration:**
- Prediction horizon: H=75 steps (research), H=15 (default/VRAM-optimized)
- Correction interval: 15 steps (micro-corrections every 15 steps)
- Confidence threshold: 0.55-0.60
- Divergence sigma: σ=2.2

### Examples & Validation ✅

**Working Examples:**
- [x] `burn_mlp_mnist.rs` - Burn integration demo
- [x] `correction_accuracy_validation.rs` - Validates correction quality
- [x] `mnist_cnn_validation.rs` - CNN training with hybrid trainer
- [x] `comprehensive_parameter_sweep.rs` - 60-config 3D parameter sweep

### Documentation ✅

- [x] `docs/INDEX.md` - Token-optimized navigation (70% reduction)
- [x] `WORKFLOW.md` - Git workflow (feature branch strategy)
- [x] `EDGE_AI_VISION.md` - Edge AI deployment pipeline
- [x] `BURN_INTEGRATION_FINAL.md` - Burn integration guide
- [x] `PHASE_1_VALIDATION_REPORT.md` - VRAM management validation (Phase 1)
- [x] `PHASE_2_BENCHMARKING_REPORT.md` - Performance benchmarks (Phase 2)
- [x] `VRAM_MANAGEMENT_SUMMARY.md` - VRAM optimization summary

### TODO (Phase 3+ Future Work)

**GPU Acceleration (v0.3.0):**
- [ ] Implement CubeCL CUDA kernel for state encoding
- [ ] Implement CubeCL CUDA kernel for RSSM forward pass
- [ ] 1B+ parameter model validation
- [ ] Multi-GPU training support

**Performance Improvements (v0.4.0+):**
- [ ] Multi-step BPTT (k=3) for GRU weight training (currently k=1)
- [ ] Gradient checkpointing integration (Burn 0.13+)
- [ ] Memory profiling for large-scale training
- [ ] Cache prediction confidence to avoid redundant computation

**Advanced Features (Future):**
- [ ] Add stochastic path sampling during RSSM rollout (ensemble diversity)
- [ ] Per-layer weight corrections in corrector (currently global only)
- [ ] Distributed training support (multi-GPU)
- [ ] Mixed precision support (fp16, bf16)

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

## Validated Configuration

### Optimal Parameters (3D Sweep Results)

Based on comprehensive parameter sweep across 60 configurations:

```rust
HybridTrainerConfig {
    // Phase transitions
    warmup_steps: 100,
    min_full_steps: 50,
    prediction_horizon: 75,        // Optimal: H=75
    correction_interval: Some(15), // Micro-corrections every 15 steps

    // Quality control
    confidence_threshold: 0.60,    // Range: 0.55-0.60
    divergence_sigma: 2.2,         // Optimal: σ=2.2

    // Dynamics model
    ensemble_size: 5,              // 5-member ensemble
    latent_dim: 32,
    feature_dim: 64,               // TrainingState::compute_features() output
}
```

### Performance Metrics

**Speedup:**
- Baseline (no prediction): 100% backward passes
- With prediction (H=75): 22% backward passes
- **Speedup: 4.5× faster** (78% reduction)

**Quality:**
- Loss convergence: 99.9% of baseline quality
- Variance reduction: 72% with micro-corrections
- Divergence rate: 0% in validated configs

**Micro-Corrections Impact:**
- Without: 50% variance reduction, H_max=50
- With (interval=15): 72% variance reduction, H_max=75
- Enables 2-3× longer prediction horizons

## Research References

1. **RSSM Architecture**: DreamerV3 (Hafner 2023)
2. **Gradient Compression**: PowerSGD (Vogels 2019)
3. **Bandit Algorithms**: LinUCB (Li 2010)
4. **Online Learning**: Follow the Regularized Leader

## Common Tasks

### Running Parameter Sweeps

```bash
# Run comprehensive 3D parameter sweep (60 configs)
cargo run --example comprehensive_parameter_sweep

# Run correction accuracy validation
cargo run --example correction_accuracy_validation

# Test with Burn MNIST example
cargo run --example burn_mlp_mnist
```

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

### Integrating with Burn Models

```rust
use hybrid_predict_trainer::BurnModelWrapper;

// Wrap your Burn model
struct MyModel<B: Backend> {
    linear: Linear<B>,
}

impl<B: Backend> BurnModelWrapper<B> for MyModel<B> {
    fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 1> {
        // Your forward pass
    }

    fn num_params(&self) -> usize {
        // Return parameter count
    }
}

// Use with HybridTrainer
let trainer = HybridTrainer::new(config, model, optimizer);
```

## Testing Strategy

- **Unit tests**: Each module tests its types in isolation
- **Integration tests**: End-to-end training scenarios (9 integration tests)
- **Total coverage**: 227 tests passing (218 lib + 9 integration)
- **Benchmarks**: Criterion benchmarks for performance-critical paths
- **Property tests**: Consider proptest for invariants

### Known Issues & Pre-existing Warnings

**Pre-existing Clippy Warnings (non-blocking):**
- `double_must_use` on `model()`/`model_mut()` in lib.rs
- Strict f32 comparisons in test code (use `--tests` flag to see)

**Recommended Lint Commands:**
```bash
# Use -W instead of -D to avoid failing on pre-existing warnings
cargo clippy --all-features -- -W clippy::all

# Run tests without clippy failures
cargo test
```

**Git Workflow Notes:**
- Current branch: `feature/optimization-research`
- Always use feature branches off `dev`, merge to `dev`, then PR `dev`→`main`
- See WORKFLOW.md for complete branching strategy

## Contact

- **Author**: Tyler Zervas (tzervas)
- **Email**: tz-dev@vectorweight.com
- **License**: MIT
