# Hybrid Predictive Training: Engineering Specification

**Version 0.2.0 | Test-Driven Design & Implementation**

---

## 1. System Overview

### 1.1 Design Philosophy

This project follows **Test-Driven Development (TDD)** principles:

1. **Define invariants first**: Mathematical properties that must hold
2. **Write tests before code**: Property-based and unit tests specify behavior
3. **Incremental implementation**: Each feature validated before proceeding
4. **Continuous verification**: CI/CD enforces all invariants

### 1.2 Core Invariants

| ID | Invariant | Test Strategy |
|----|-----------|---------------|
| INV-1 | Loss monotonically decreases (on average) | Rolling average comparison |
| INV-2 | Gradient norms bounded (no explosion/vanishing) | Threshold assertions |
| INV-3 | Phase transitions are deterministic | State machine tests |
| INV-4 | Residual corrections improve loss | Before/after comparison |
| INV-5 | Predictions bounded by confidence | σ-interval tests |
| INV-6 | Memory usage bounded | Peak memory assertions |
| INV-7 | No NaN/Inf propagation | Value checks on all tensors |

### 1.3 Component Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                    hybrid-predict-trainer-rs                    │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│  │   Config    │    │   State     │    │   Phases    │        │
│  │  (Serde)    │───▶│ (Encoder)   │───▶│  (Machine)  │        │
│  └─────────────┘    └─────────────┘    └─────────────┘        │
│        │                  │                  │                 │
│        ▼                  ▼                  ▼                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│  │  Dynamics   │◀──▶│  Residuals  │◀──▶│  Corrector  │        │
│  │  (RSSM)     │    │  (Store)    │    │  (Online)   │        │
│  └─────────────┘    └─────────────┘    └─────────────┘        │
│        │                  │                  │                 │
│        ▼                  ▼                  ▼                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│  │ Divergence  │    │   Metrics   │    │   Bandit    │        │
│  │  (Monitor)  │    │ (Collector) │    │  (LinUCB)   │        │
│  └─────────────┘    └─────────────┘    └─────────────┘        │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

## 2. Test-Driven Development Workflow

### 2.1 Test Categories

```rust
// Level 1: Unit Tests (per module)
#[cfg(test)]
mod tests {
    #[test]
    fn gru_cell_output_shape() { ... }

    #[test]
    fn phase_transition_deterministic() { ... }
}

// Level 2: Property Tests (invariants)
#[cfg(test)]
mod property_tests {
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn loss_bounded(loss in 0.0f32..1000.0) {
            let state = TrainingState::with_loss(loss);
            prop_assert!(state.loss.is_finite());
            prop_assert!(state.loss >= 0.0);
        }
    }
}

// Level 3: Integration Tests (end-to-end)
#[test]
fn training_loop_reduces_loss() {
    let trainer = create_test_trainer();
    let initial_loss = trainer.step(&batch).loss;
    for _ in 0..100 {
        trainer.step(&batch);
    }
    assert!(trainer.state.loss < initial_loss);
}

// Level 4: Benchmark Tests (performance)
#[bench]
fn predict_step_latency(b: &mut Bencher) {
    b.iter(|| trainer.execute_predict_step(&batch));
}
```

### 2.2 Test Coverage Requirements

| Module | Line Coverage | Branch Coverage | Required |
|--------|---------------|-----------------|----------|
| config | 90% | 85% | All defaults tested |
| state | 95% | 90% | All encodings tested |
| phases | 100% | 95% | All transitions tested |
| dynamics | 85% | 80% | GRU, loss head, weight delta |
| residuals | 90% | 85% | Store operations |
| corrector | 85% | 80% | Correction computation |
| divergence | 95% | 90% | All signals tested |
| metrics | 80% | 75% | Export formats |

### 2.3 CI/CD Pipeline

```yaml
# .github/workflows/ci.yml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Run tests
        run: cargo test --all-features

      - name: Property tests
        run: cargo test --features proptest

      - name: Coverage
        run: cargo tarpaulin --out Xml

      - name: Clippy
        run: cargo clippy --all-features -- -D warnings

      - name: Benchmarks
        run: cargo bench --no-run

  gpu-test:
    runs-on: [self-hosted, gpu]
    steps:
      - name: GPU integration tests
        run: cargo test --features cuda -- --ignored
```

---

## 3. Formal Specifications

### 3.1 Phase State Machine

```rust
/// Phase state machine specification.
///
/// INVARIANT: Transitions are deterministic given (state, config).
/// INVARIANT: No phase lasts forever (bounded by config limits).
/// INVARIANT: WARMUP always precedes other phases.
pub enum Phase {
    /// Initial phase: collect baseline statistics
    /// ENTRY: step = 0
    /// EXIT: step >= config.warmup_steps
    Warmup,

    /// Full training: forward + backward + optimizer
    /// ENTRY: from Warmup OR Correct
    /// EXIT: predictor.confidence > config.predict_threshold
    Full,

    /// Predictive training: forward only + apply Δθ
    /// ENTRY: confidence > threshold AND no divergence
    /// EXIT: horizon reached OR divergence detected
    Predict,

    /// Correction: apply residual corrections
    /// ENTRY: from Predict (always)
    /// EXIT: corrections applied
    Correct,
}

impl Phase {
    /// Compute next phase given current state.
    ///
    /// # Guarantees
    /// - Pure function: same inputs → same output
    /// - Terminates: no infinite loops possible
    /// - Valid: returns only reachable phases
    pub fn next(
        &self,
        state: &TrainingState,
        config: &PhaseConfig,
        predictor: &DynamicsModel,
        divergence: &DivergenceMonitor,
    ) -> Phase { ... }
}
```

### 3.2 Dynamics Model Contract

```rust
/// Dynamics model contract (RSSM-lite).
///
/// INVARIANT: Predictions are bounded by confidence intervals.
/// INVARIANT: Confidence ∈ [0, 1].
/// INVARIANT: Weight deltas have same structure as model weights.
pub trait DynamicsModel {
    /// Predict Y steps forward.
    ///
    /// # Guarantees
    /// - predicted_loss ∈ [0, ∞) (finite, non-negative)
    /// - confidence ∈ [0, 1]
    /// - weight_delta.layers matches model.layers
    /// - uncertainty scales with Y (longer predictions = higher uncertainty)
    fn predict_y_steps(
        &self,
        state: &TrainingState,
        y: usize,
    ) -> (PhasePrediction, UncertaintyEstimate);

    /// Update model from observed gradient.
    ///
    /// # Guarantees
    /// - Monotonic improvement: prediction error decreases on average
    /// - Bounded update: ‖Δweight‖ ≤ learning_rate × ‖gradient‖
    fn observe_gradient(
        &mut self,
        state: &TrainingState,
        grad_info: &GradientInfo,
    );

    /// Current prediction confidence.
    ///
    /// # Guarantees
    /// - confidence ∈ [0, 1]
    /// - confidence increases with training steps (generally)
    /// - confidence decreases after prediction errors
    fn prediction_confidence(&self, state: &TrainingState) -> f32;
}
```

### 3.3 Residual Correction Contract

```rust
/// Residual correction contract.
///
/// INVARIANT: Corrections reduce prediction error on average.
/// INVARIANT: Corrections are bounded (no explosion).
/// INVARIANT: Empty residual store → zero correction.
pub trait ResidualCorrector {
    /// Compute correction from residual history.
    ///
    /// # Guarantees
    /// - correction.loss_correction bounded by max observed residual
    /// - correction.weight_correction bounded by max observed Δθ
    /// - confidence reflects residual consistency
    fn compute_correction(
        &self,
        state: &TrainingState,
        residuals: &ResidualStore,
        predicted_loss: f32,
    ) -> Correction;

    /// Update corrector from new residual.
    ///
    /// # Guarantees
    /// - Online learning: O(1) time complexity
    /// - Bounded memory: residual store has fixed capacity
    fn update_from_residual(
        &mut self,
        residual: &Residual,
        state: &TrainingState,
    );
}
```

---

## 4. Test Specifications

### 4.1 Unit Test Examples

```rust
// dynamics.rs tests
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gru_step_preserves_hidden_dim() {
        let weights = GRUWeights::new(64, 32);
        let hidden = vec![0.0; 64];
        let input = vec![0.0; 32];

        let output = RSSMLite::gru_step(&weights, &hidden, &input);

        assert_eq!(output.len(), 64);
        assert!(output.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn weight_delta_has_correct_layers() {
        let model = RSSMLite::new(&test_config()).unwrap();
        let state = TrainingState::default();

        let (pred, _) = model.predict_y_steps(&state, 10);

        assert!(pred.weight_delta.deltas.contains_key("embed"));
        assert!(pred.weight_delta.deltas.contains_key("attention.q"));
        assert!(pred.weight_delta.deltas.contains_key("mlp.up"));
    }

    #[test]
    fn confidence_bounded() {
        let model = RSSMLite::new(&test_config()).unwrap();

        for _ in 0..1000 {
            let state = random_training_state();
            let conf = model.prediction_confidence(&state);
            assert!(conf >= 0.0 && conf <= 1.0);
        }
    }
}
```

### 4.2 Property Tests

```rust
// property_tests.rs
use proptest::prelude::*;

proptest! {
    #[test]
    fn phase_transition_deterministic(
        step in 0usize..100000,
        loss in 0.0f32..100.0,
        confidence in 0.0f32..1.0,
    ) {
        let state = TrainingState { step, loss, ..Default::default() };
        let config = PhaseConfig::default();

        let phase1 = Phase::compute_next(&state, &config);
        let phase2 = Phase::compute_next(&state, &config);

        prop_assert_eq!(phase1, phase2);
    }

    #[test]
    fn residual_store_bounded(
        residuals in prop::collection::vec(any::<Residual>(), 0..1000)
    ) {
        let mut store = ResidualStore::new(100);

        for r in residuals {
            store.add(r);
        }

        prop_assert!(store.len() <= 100);
    }

    #[test]
    fn prediction_error_finite(
        y in 1usize..100,
        loss in 0.001f32..100.0,
    ) {
        let model = RSSMLite::new(&test_config()).unwrap();
        let state = TrainingState { loss, ..Default::default() };

        let (pred, _) = model.predict_y_steps(&state, y);

        prop_assert!(pred.predicted_final_loss.is_finite());
        prop_assert!(pred.predicted_final_loss >= 0.0);
    }
}
```

### 4.3 Integration Tests

```rust
// tests/end_to_end.rs
#[test]
fn full_training_cycle() {
    let config = HybridTrainerConfig::builder()
        .warmup_steps(10)
        .min_full_steps(10)
        .predict_threshold(0.5)
        .build();

    let mut trainer = create_test_trainer(config);
    let mut phase_counts = HashMap::new();

    for _ in 0..1000 {
        let result = trainer.step(&test_batch()).unwrap();
        *phase_counts.entry(result.phase).or_insert(0) += 1;
    }

    // All phases should be visited
    assert!(phase_counts.contains_key(&Phase::Warmup));
    assert!(phase_counts.contains_key(&Phase::Full));
    assert!(phase_counts.contains_key(&Phase::Predict));
    assert!(phase_counts.contains_key(&Phase::Correct));

    // Loss should decrease
    assert!(trainer.state.loss < trainer.initial_loss);
}

#[test]
fn divergence_triggers_recovery() {
    let mut trainer = create_test_trainer_with_unstable_model();

    // Inject divergent gradients
    trainer.inject_gradient_spike(1000.0);

    let result = trainer.step(&test_batch());

    match result {
        Err((err, Some(recovery))) => {
            assert!(recovery.can_continue());
            assert!(matches!(err, HybridTrainingError::PredictionDivergence { .. }));
        }
        _ => panic!("Expected divergence error with recovery"),
    }
}
```

---

## 5. Benchmark Specifications

### 5.1 Performance Targets

#### Achieved Results (Phase 2 Benchmarking - 2026-02-07)

| Component | Measured Performance | Target | Status |
|-----------|---------------------|--------|--------|
| RSSM prediction (1 step) | 50 µs | < 100 µs | ✅ Exceeded |
| RSSM prediction (15 steps) | 400 µs | < 1 ms | ✅ Exceeded |
| RSSM prediction (75 steps) | 1.8 ms | < 10 ms | ✅ Exceeded |
| State encoding (64-dim) | 15.2 µs | < 50 µs | ✅ Exceeded |
| RSSM gradient observation | 1.36 ms | < 5 ms | ✅ Exceeded |
| Confidence computation | 8.4 ns | < 1 µs | ✅ Exceeded |
| State history update | 2.4 ns | < 100 ns | ✅ Exceeded |
| Weight delta clone/scale | < 1 µs | < 10 µs | ✅ Exceeded |

**Performance Characteristics:**
- RSSM prediction scales linearly: ~24 µs per step
- Overhead reduction: 70% (predict vs full training step)
- Estimated training speedup: 2.4-2.5× for various model sizes
- Validated speedup: 1.74× on GPT-2 Small (124M params, memory-constrained)

**VRAM Management (Phase 1):**
- Baseline: 3.9 GB → 14.1 GB over 50 steps (before mitigation)
- Optimized: <10 GB over 50 steps (with VramManager)
- Improvement: ~40-60% VRAM reduction

See `PHASE_2_BENCHMARKING_REPORT.md` for detailed analysis.

### 5.2 Benchmark Implementation

```rust
// benches/predictor_performance.rs
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};

fn predict_step_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("predict_step");

    for model_size in [100_000_000, 500_000_000, 1_000_000_000] {
        let config = test_config_for_size(model_size);
        let trainer = create_trainer(&config);
        let batch = create_batch(&config);

        group.bench_with_input(
            BenchmarkId::new("predict", model_size),
            &(trainer, batch),
            |b, (trainer, batch)| {
                b.iter(|| trainer.execute_predict_step(batch))
            },
        );
    }

    group.finish();
}

criterion_group!(benches, predict_step_benchmark);
criterion_main!(benches);
```

---

## 6. Verification Checklist

### 6.1 Pre-Commit Checklist

- [x] All unit tests pass: `cargo test` (227 tests passing)
- [ ] Property tests pass: `cargo test --features proptest` (not yet implemented)
- [x] No clippy warnings: `cargo clippy --all-features -W clippy::all`
- [x] Documentation builds: `cargo doc --no-deps`
- [x] Benchmarks compile: `cargo bench --no-run`
- [x] MSRV check: `cargo +1.92 build`

### 6.2 Pre-Release Checklist

- [ ] Integration tests pass: `cargo test --test '*'`
- [ ] GPU tests pass: `cargo test --features cuda -- --ignored`
- [ ] Coverage > 80%: `cargo tarpaulin`
- [ ] No security advisories: `cargo audit`
- [ ] CHANGELOG updated
- [ ] Version bumped (semver)
- [ ] README examples verified

### 6.3 Performance Regression Checklist

- [ ] Benchmark comparison: `cargo bench -- --save-baseline new`
- [ ] No regressions > 5%
- [ ] Memory usage unchanged
- [ ] GPU memory unchanged (CUDA tests)

---

## 7. Documentation Requirements

### 7.1 Code Documentation

```rust
/// Short summary (one line).
///
/// Longer description that explains the purpose and behavior.
///
/// # Arguments
///
/// * `arg1` - Description of first argument
/// * `arg2` - Description of second argument
///
/// # Returns
///
/// Description of return value.
///
/// # Errors
///
/// Description of error conditions.
///
/// # Examples
///
/// ```rust
/// use hybrid_predict_trainer_rs::Example;
///
/// let result = Example::new(42);
/// assert_eq!(result.value(), 42);
/// ```
///
/// # Panics
///
/// Description of panic conditions (if any).
///
/// # Safety
///
/// Safety requirements (if unsafe).
pub fn documented_function(arg1: Type1, arg2: Type2) -> Result<ReturnType> {
    // Implementation
}
```

### 7.2 Module Documentation

Each module must have:
- Module-level doc comment with overview
- Public API documented with examples
- Internal implementation notes as code comments
- Test cases that serve as usage documentation

---

## 8. Implementation Schedule

### Phase 1: Core Infrastructure (Week 1-2)
- [x] Configuration with builder pattern
- [x] Error types with recovery actions
- [x] State encoding and management
- [x] Phase state machine

### Phase 2: Dynamics Model (Week 3-4)
- [x] GRU cell implementation
- [x] Loss prediction head
- [x] Weight delta prediction
- [ ] Confidence calibration

### Phase 3: Prediction Integration (Week 5-6)
- [x] Predict phase execution (scaffolded)
- [x] Residual extraction
- [x] Residual store
- [ ] Weight-level correction (corrector produces None for weight corrections)
- [ ] RSSM dynamics training (GRU weights frozen, only loss head trained)
- [ ] Multi-step phase prediction (currently 1-step only)
- [ ] Confidence calibration with caching

### Phase 4: Validation (Week 7-8)
- [ ] Integration tests with real models
- [ ] Performance benchmarks
- [ ] GPU testing
- [ ] Documentation review

### Phase 5: Release (Week 9)
- [ ] Version 1.0.0 release
- [ ] crates.io publication
- [ ] Research paper draft

---

## 9. Gap Analysis (February 2026)

A thorough code analysis of the predict and correct phases identified eight critical
gaps between the specification and the current implementation. These must be resolved
before validation experiments can produce meaningful results.

### 9.1 Critical Gaps

**Gap 1: RSSM GRU weights never trained after initialization**
- **Severity**: Critical
- **Location**: `dynamics.rs` -- `observe_gradient()` only performs SGD on the loss head weights; the GRU cell weights (`W_z`, `W_r`, `W_h`, `U_z`, `U_r`, `U_h`) are initialized and then frozen.
- **Impact**: The deterministic path of the RSSM cannot learn training dynamics. All predictions rely on the randomly initialized GRU, producing effectively random latent states.
- **Fix**: Implement truncated backpropagation through time (BPTT-lite) in `observe_gradient()`. Accumulate GRU gradients over a short window (e.g., 5-10 steps), then apply a single SGD update to all GRU weight matrices.

**Gap 2: Weight delta head weights never trained**
- **Severity**: Critical
- **Location**: `dynamics.rs` -- the weight delta head (`delta_weights`, `delta_bias`) has no gradient computation or update logic in `observe_gradient()`.
- **Impact**: Weight delta predictions are entirely heuristic (lr * grad_norm * learned_scale), where `learned_scale` is never actually learned.
- **Fix**: Compute weight delta prediction error against observed weight changes during full training, then apply SGD updates to the delta head alongside the loss head.

**Gap 3: Stochastic path is passive during rollout**
- **Severity**: High
- **Location**: `dynamics.rs` -- the stochastic sampler `z` exists in the architecture but is never sampled during `predict_y_steps()`. Only the deterministic GRU path is used.
- **Impact**: The model has no way to represent uncertainty from batch sampling variance, producing overconfident predictions.
- **Fix**: Sample from the stochastic posterior during rollout and combine with the deterministic state before feeding into the loss and weight delta heads.

**Gap 4: Corrector never produces weight corrections**
- **Severity**: Critical
- **Location**: `corrector.rs` -- `compute_correction()` always returns `weight_correction: None`. The correction is loss-only.
- **Impact**: The correction phase cannot repair accumulated weight drift from the predict phase, violating Section 2.5 of the theory (residual correction over both loss and weights).
- **Fix**: Implement per-layer weight corrections using stored gradient directions from the full training phase, scaled by the loss residual magnitude.

### 9.2 High-Priority Gaps

**Gap 5: No gradient direction tracking during full training**
- **Severity**: High
- **Location**: `full_train.rs` -- gradient observations record norms and loss but do not store gradient direction vectors (or compressed representations thereof).
- **Impact**: The correction phase has no gradient direction information to construct weight corrections, even if the corrector were implemented.
- **Fix**: Store per-layer gradient direction vectors (or low-rank approximations via PowerSGD) during the full training phase, making them available to the corrector.

**Gap 6: Feature dimension mismatch**
- **Severity**: High
- **Location**: `state.rs` `compute_features()` returns a 64-dimensional feature vector; `corrector.rs` initializes its linear model with input dimension 32.
- **Impact**: Runtime dimension mismatch between the state encoder output and the corrector input. This would cause index-out-of-bounds or silent truncation depending on the execution path.
- **Fix**: Align dimensions -- either reduce `compute_features()` to 32 dimensions or increase the corrector linear model to 64. The corrector should derive its input dimension from the state encoder configuration.

**Gap 7: Predict phase uses 1-step predictions instead of multi-step phase prediction**
- **Severity**: High
- **Location**: `predictive.rs` -- the predict phase executor calls the dynamics model once per step rather than predicting an aggregate Y-step phase outcome as specified in Section 2.4 of the theory.
- **Impact**: No computational savings from the predict phase; each step still requires a full dynamics model forward pass, and prediction errors compound without the stabilizing effect of aggregate prediction.
- **Fix**: Implement Y-step aggregate prediction: call `predict_y_steps(state, Y)` once at the start of the predict phase, then apply the predicted weight delta incrementally (or all at once) over Y steps.

### 9.3 Medium-Priority Gaps

**Gap 8: prediction_confidence() redundantly recomputes ensemble predictions**
- **Severity**: Medium
- **Location**: `dynamics.rs` -- `prediction_confidence()` calls `predict_y_steps(state, 10)` internally, performing a full ensemble forward pass each time it is queried. This is called at every step to evaluate phase transitions.
- **Impact**: Significant computational overhead. Each confidence check runs 10 GRU steps across the ensemble, which is wasteful when the underlying state has not changed since the last prediction.
- **Fix**: Cache the confidence value and invalidate only when `observe_gradient()` is called or the training state changes materially (e.g., new step observed).

### 9.4 Updated Success Criteria

| Metric | Current | Target | Priority |
|--------|---------|--------|----------|
| RSSM loss prediction R-squared | ~0 (untrained) | > 0.85 | Critical |
| Weight delta cosine similarity | ~random | > 0.8 | Critical |
| Correction loss improvement | 0% (no weight corrections) | > 5% per correction | High |
| Confidence computation overhead | O(ensemble x 10 GRU steps) per step | O(1) cached | Medium |
| Predict phase granularity | 1-step | Y-step aggregate | High |
