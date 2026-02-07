# Hypothesis Validation Plan

**A Data-First Scientific Approach to Predictive Training**

---

## 1. Core Hypothesis

### H1: Training Dynamics are Predictable

**Statement**: Neural network training dynamics during gradient descent evolve
on low-dimensional manifolds, making aggregate phase outcomes (loss, weight
changes) predictable from training history.

**Falsifiable Prediction**: Given N training steps of history, a learned
dynamics model can predict the loss at step N+Y with error < 5% for Y ≤ 50.

**Evidence Required**:
1. Gradient cosine similarity > 0.8 for adjacent steps
2. PCA of weight updates explains > 80% variance with 10 components
3. RSSM model achieves prediction R² > 0.9 on held-out training trajectories

### H2: Phase Prediction Enables Speedup

**Statement**: By predicting aggregate training outcomes for entire phases,
we can skip the backward pass for those phases, achieving 3-5× speedup in
compute while maintaining training quality.

**Falsifiable Prediction**: Training with 50% predicted steps achieves final
loss within 2% of conventional training.

**Evidence Required**:
1. Predicted weight deltas correlate with actual (R > 0.9)
2. Model trained with prediction phases achieves equivalent downstream task performance
3. Wall-clock time reduced by ≥ 50%

### H3: Residual Correction Maintains Quality

**Statement**: Online residual correction compensates for prediction errors,
ensuring training quality is maintained even when predictions diverge.

**Falsifiable Prediction**: Correction phases reduce accumulated prediction
error by > 50%.

**Evidence Required**:
1. Loss before correction > loss after correction (statistical significance)
2. Correction frequency correlates with prediction error magnitude
3. Total correction maintains final model quality

---

## 2. Experimental Design

### Experiment 1: Gradient Structure Analysis

**Objective**: Validate that training gradients have exploitable structure.

**Protocol**:
1. Train 100M model on TinyStories for 10K steps (conventional)
2. Record all gradients at each step
3. Compute:
   - Cosine similarity between g_t and g_{t+1}
   - PCA of gradient history
   - Temporal autocorrelation

**Success Criteria**:
- Mean cosine similarity > 0.7
- Top 10 PCA components explain > 70% variance
- Autocorrelation significant at lag 50

**Code**:
```rust
#[test]
fn gradient_structure_analysis() {
    let mut trainer = conventional_trainer();
    let mut gradients = Vec::new();

    for _ in 0..10000 {
        let grad = trainer.step_and_record_gradient(&batch);
        gradients.push(grad);
    }

    let similarities = compute_adjacent_cosine_similarities(&gradients);
    let pca_variance = compute_pca_explained_variance(&gradients, 10);
    let autocorr = compute_autocorrelation(&gradients, 50);

    assert!(similarities.mean() > 0.7);
    assert!(pca_variance > 0.7);
    assert!(autocorr.is_significant(0.05));
}
```

### Experiment 2: Loss Prediction Accuracy

**Objective**: Validate that RSSM-lite can predict future loss.

**Protocol**:
1. Train 100M model on TinyStories for 5K steps (conventional)
2. Train RSSM dynamics model on this trajectory
3. Hold out final 1K steps
4. Predict loss for held-out steps
5. Compute prediction error

**Success Criteria**:
- Mean absolute error < 5% of loss range
- R² > 0.85 for held-out predictions
- Prediction degrades gracefully with horizon Y

**Code**:
```rust
#[test]
fn loss_prediction_accuracy() {
    let trajectory = collect_training_trajectory(5000);
    let (train, test) = trajectory.split_at(4000);

    let dynamics = RSSMLite::train_on_trajectory(&train);

    let predictions = test.iter().map(|state| {
        dynamics.predict_loss_at_step(state.step)
    }).collect::<Vec<_>>();

    let actual = test.iter().map(|s| s.loss).collect::<Vec<_>>();

    let mae = mean_absolute_error(&predictions, &actual);
    let r2 = r_squared(&predictions, &actual);

    assert!(mae < (actual.max() - actual.min()) * 0.05);
    assert!(r2 > 0.85);
}
```

### Experiment 3: Weight Delta Prediction

**Objective**: Validate that weight changes are predictable.

**Protocol**:
1. Train 100M model for 10K steps
2. Record weight snapshots every 50 steps
3. Train dynamics model to predict Δθ given history
4. Evaluate prediction accuracy

**Success Criteria**:
- Cosine similarity between predicted and actual Δθ > 0.8
- L2 norm error < 20% of actual norm
- Per-layer prediction accuracy > 75%

**Code**:
```rust
#[test]
fn weight_delta_prediction() {
    let snapshots = collect_weight_snapshots(10000, 50);

    let dynamics = RSSMLite::train_on_weight_trajectory(&snapshots);

    for i in (200..snapshots.len()).step_by(50) {
        let history = &snapshots[..i];
        let actual_delta = &snapshots[i].weights - &snapshots[i-1].weights;
        let predicted_delta = dynamics.predict_weight_delta(history, 50);

        let cosine = cosine_similarity(&predicted_delta, &actual_delta);
        assert!(cosine > 0.8, "Step {}: cosine = {}", i, cosine);
    }
}
```

### Experiment 4: Prediction Phase Training

**Objective**: Validate that prediction phases maintain training quality.

**Protocol**:
1. Train 100M model conventionally for 20K steps (baseline)
2. Train 100M model with 25% prediction phases
3. Train 100M model with 50% prediction phases
4. Compare final loss and downstream task performance

**Success Criteria**:
- 25% prediction: loss within 1% of baseline
- 50% prediction: loss within 2% of baseline
- Downstream task delta < 1%

**Code**:
```rust
#[test]
fn prediction_phase_training() {
    // Baseline
    let baseline = train_conventional(20000);

    // 25% prediction
    let config_25 = HybridTrainerConfig::builder()
        .prediction_ratio(0.25)
        .build();
    let result_25 = train_hybrid(20000, config_25);

    // 50% prediction
    let config_50 = HybridTrainerConfig::builder()
        .prediction_ratio(0.50)
        .build();
    let result_50 = train_hybrid(20000, config_50);

    assert!((result_25.loss - baseline.loss).abs() / baseline.loss < 0.01);
    assert!((result_50.loss - baseline.loss).abs() / baseline.loss < 0.02);
}
```

### Experiment 5: Residual Correction Effectiveness

**Objective**: Validate that correction phases improve prediction accuracy.

**Protocol**:
1. Train with 50% prediction phases
2. Record loss before/after each correction phase
3. Analyze correction effectiveness

**Success Criteria**:
- Mean loss reduction after correction > 5%
- Correction reduces cumulative prediction error
- Training remains stable (no divergence)

**Code**:
```rust
#[test]
fn residual_correction_effectiveness() {
    let mut corrections = Vec::new();

    for event in train_with_logging(20000) {
        if event.phase == Phase::Correct {
            corrections.push(CorrectionEvent {
                loss_before: event.loss_before,
                loss_after: event.loss_after,
            });
        }
    }

    let improvements: Vec<f32> = corrections.iter()
        .map(|c| (c.loss_before - c.loss_after) / c.loss_before)
        .collect();

    assert!(improvements.mean() > 0.05);
    assert!(improvements.iter().filter(|&&x| x > 0.0).count() > corrections.len() / 2);
}
```

---

## 3. Data Collection Plan

### 3.1 Datasets

| Dataset | Size | Purpose | Priority |
|---------|------|---------|----------|
| TinyStories | 2GB | Fast iteration, hypothesis validation | **Immediate** |
| FineWeb-Edu 10B | 28GB | Production training validation | High |
| FineMath | 20GB | Math domain validation | Medium |

### 3.2 Metrics to Collect

| Metric | Collection Point | Storage |
|--------|------------------|---------|
| Loss per step | Every step | JSONL (metrics.jsonl) |
| Gradient norm | Every step | JSONL |
| Gradient vectors | Every N steps | Compressed tensors |
| Weight snapshots | Every 100 steps | Safetensors |
| Prediction vs actual | Each predict phase | JSONL |
| Correction events | Each correct phase | JSONL |
| Phase transitions | State changes | JSONL |

### 3.3 Statistical Analysis

All experiments will use:
- 5 random seeds for reproducibility
- Bootstrap confidence intervals (95%)
- Paired t-tests for before/after comparisons
- Bonferroni correction for multiple comparisons

---

## 4. Validation Milestones

### Milestone 1: Gradient Structure (Week 1)

**Goal**: Confirm that training dynamics have exploitable structure.

**Deliverables**:
- Gradient cosine similarity analysis
- PCA decomposition results
- Go/No-Go decision for prediction feasibility

**Success**: Mean cosine similarity > 0.7, PCA variance > 70%

### Milestone 2: Loss Prediction (Week 2)

**Goal**: Confirm that RSSM-lite can predict loss trajectories.

**Deliverables**:
- Trained dynamics model on TinyStories
- Prediction accuracy analysis
- Horizon sensitivity analysis

**Success**: R² > 0.85, MAE < 5%

### Milestone 3: Weight Delta Prediction (Week 3)

**Goal**: Confirm that weight changes are predictable.

**Deliverables**:
- Weight delta prediction model
- Layer-wise accuracy analysis
- Scaling analysis (100M → 500M)

**Success**: Cosine similarity > 0.8

### Milestone 4: Training Integration (Week 4)

**Goal**: Validate full hybrid training loop.

**Deliverables**:
- End-to-end training with prediction phases
- Quality comparison with baseline
- Speedup measurements

**Success**: Loss within 2%, speedup > 2×

### Milestone 5: Production Validation (Week 5-6)

**Goal**: Validate on production-scale training.

**Deliverables**:
- 100M model trained on FineWeb-Edu
- Downstream task evaluation
- Energy consumption comparison

**Success**: Equivalent downstream performance, 3× speedup

---

## 5. Risk Mitigation

### Risk 1: Gradients Not Predictable

**Mitigation**: If cosine similarity < 0.5, explore:
- Different architecture (skip connections affect gradient structure)
- Different optimization (LAMB vs AdamW)
- Shorter prediction horizons

### Risk 2: Weight Deltas Not Predictable

**Mitigation**: If weight prediction fails:
- Predict only loss, not weights
- Use loss prediction for early stopping only
- Focus on curriculum optimization

### Risk 3: Prediction Errors Compound

**Mitigation**: If divergence is common:
- Increase correction frequency
- Reduce prediction horizon dynamically
- Add confidence-based fallback to full training

### Risk 4: Speedup < 2×

**Mitigation**: If speedup insufficient:
- Combine with other techniques (mixed precision, checkpointing)
- Focus on memory efficiency instead of compute
- Target different use case (hyperparameter search)

---

## 6. Decision Points

### Decision 1: Proceed with Full Implementation?

**After Milestone 2**:
- If loss prediction R² > 0.85 → Proceed
- If R² ∈ [0.7, 0.85] → Refine dynamics model
- If R² < 0.7 → Pivot or abandon

### Decision 2: Scale to Production?

**After Milestone 4**:
- If speedup > 2× and quality within 2% → Scale up
- If speedup < 2× → Optimize implementation
- If quality > 2% degradation → Increase correction frequency

### Decision 3: Publish Results?

**After Milestone 5**:
- If all success criteria met → Prepare paper
- If partial success → Document learnings, open source
- If failure → Document negative results for community

---

## 7. Appendix: Experimental Logs Template

```json
{
  "experiment_id": "exp_001",
  "hypothesis": "H1",
  "date": "2026-01-31",
  "seed": 42,
  "config": {
    "model_size": "100M",
    "dataset": "TinyStories",
    "steps": 10000
  },
  "results": {
    "cosine_similarity_mean": 0.78,
    "cosine_similarity_std": 0.12,
    "pca_variance_10": 0.73
  },
  "conclusion": "PASS - gradient structure sufficient for prediction",
  "next_steps": "Proceed to Experiment 2"
}
```

---

## 8. References

1. Jacot, A. et al. (2018). Neural Tangent Kernel: Convergence and Generalization.
2. Vogels, T. et al. (2019). PowerSGD: Practical Low-Rank Gradient Compression.
3. Hafner, D. et al. (2023). Mastering Diverse Domains through World Models.
4. Chizat, L. & Bach, F. (2019). On the Global Convergence of Gradient Descent.

---

## 9. Implementation Readiness Assessment

### 9.1 Current State

As of February 2026, the validation experiments defined in Sections 2-5 cannot
produce meaningful results due to critical gaps in the predict and correct phase
implementations. This section documents the specific blockers and provides a
pre-validation checklist.

### 9.2 Blockers for Validation

**Experiments 2 and 3 (Loss and Weight Delta Prediction)** require a functioning
RSSM dynamics model that can learn from training data. The current implementation
has two critical defects:

1. **GRU weights are never trained**: The RSSM GRU cell weights are initialized
   randomly and never updated during `observe_gradient()`. Only the loss head
   receives SGD updates. As a result, the latent states produced by the GRU carry
   no learned information, and loss predictions will have near-zero R-squared
   regardless of training duration.

2. **Weight delta head is never trained**: The weight delta prediction head has no
   gradient computation or update path. Weight delta predictions are dominated by
   the `lr * grad_norm` heuristic with a random scaling factor, producing
   predictions that will show near-zero cosine similarity with actual weight
   changes.

**Experiment 4 (Prediction Phase Training)** additionally requires:

3. **Multi-step phase prediction**: The predict phase currently executes
   one-step-at-a-time rather than predicting aggregate Y-step outcomes. This
   provides no computational savings and introduces compounding single-step errors.

**Experiment 5 (Residual Correction Effectiveness)** additionally requires:

4. **Weight-level corrections**: The corrector always returns `None` for weight
   corrections. Correction effectiveness measurements will show only loss-level
   adjustments, which cannot compensate for accumulated weight drift from the
   predict phase.

### 9.3 Pre-validation Checklist

All items must be completed before running validation experiments:

- [ ] RSSM GRU weights update during observe_gradient()
- [ ] Weight delta head weights update during observe_gradient()
- [ ] Stochastic path samples during prediction rollout
- [ ] Corrector produces per-layer weight corrections
- [ ] Multi-step prediction used in predict phase
- [ ] Gradient directions stored during full training

### 9.4 Recommended Validation Order

Once the pre-validation checklist is satisfied:

1. **First**: Re-run Experiment 2 (loss prediction) with the trained RSSM. If
   R-squared < 0.7, stop and investigate dynamics model capacity before
   proceeding.
2. **Second**: Run Experiment 3 (weight delta prediction) to confirm learned
   dynamics transfer to weight space.
3. **Third**: Run Experiment 5 (correction effectiveness) in isolation before
   attempting full hybrid training.
4. **Last**: Run Experiment 4 (full prediction phase training) only after
   individual components are validated.

---

*This document follows the scientific method: hypothesis -> prediction -> experiment -> conclusion.*
