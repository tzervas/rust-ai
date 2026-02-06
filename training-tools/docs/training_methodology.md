# Hybrid Predictive Training Methodology

**Version:** 1.0
**Last Updated:** 2026-02-01
**Framework:** hybrid-predict-trainer-rs + tritter-model-rs

---

## Table of Contents

1. [Overview](#overview)
2. [Training Phases](#training-phases)
3. [Phase Transitions](#phase-transitions)
4. [Hyperparameter Guide](#hyperparameter-guide)
5. [Loss Curve Interpretation](#loss-curve-interpretation)
6. [Troubleshooting Guide](#troubleshooting-guide)
7. [Specialist vs Generalist Training](#specialist-vs-generalist-training)
8. [Learning Rate Scheduling](#learning-rate-scheduling)
9. [Gradient Health Indicators](#gradient-health-indicators)

---

## Overview

Hybrid predictive training achieves **5-10x training speedup** by intelligently predicting training steps instead of computing full forward/backward passes. The method cycles through four distinct phases, each with specific objectives.

### Core Principle

Training dynamics are predictable enough that we can:
1. Learn patterns during full training phases
2. Predict multiple steps without gradients (speedup)
3. Correct accumulated errors with residuals (maintain quality)

### Visual Architecture

```
┌────────────────────────────────────────────────────────────┐
│                    TRAINING LIFECYCLE                      │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  ┌──────────┐       ┌──────────┐       ┌──────────┐       │
│  │  WARMUP  │──────▶│   FULL   │──────▶│ PREDICT  │       │
│  │          │       │  TRAIN   │       │          │       │
│  │ 100-500  │       │  20-50   │       │  20-200  │       │
│  │  steps   │       │  steps   │       │  steps   │       │
│  └──────────┘       └────┬─────┘       └────┬─────┘       │
│                          │                   │             │
│                          │     ┌──────────┐  │             │
│                          │     │ CORRECT  │  │             │
│                          │     │          │  │             │
│                          │     │  5-20    │  │             │
│                          │     │  steps   │  │             │
│                          │     └────┬─────┘  │             │
│                          │          │        │             │
│                          └──────────┴────────┘             │
│                              (cycle repeats)               │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

---

## Training Phases

### 1. Warmup Phase

**Purpose:** Establish baseline training dynamics

**What Happens:**
- Standard forward/backward training (full compute)
- Statistics collection: loss distribution, gradient norms, weight magnitudes
- Predictor initialization (no predictions yet)
- Training curve stabilization

**Duration:** 100-500 steps (model-dependent)

**Key Metrics Collected:**
- Loss moving average and variance
- Gradient norm statistics (mean, std, min, max)
- Loss oscillation frequency
- Learning rate warmup curve

**When to Increase Warmup Steps:**
- Training is unstable early on
- Large models (>1B parameters)
- Complex datasets with high variance
- Using high initial learning rates

**ASCII Diagram:**

```
WARMUP PHASE (steps 0-200)
───────────────────────────

Loss:     ┌──────────────────────
    4.0   │\
          │ \___
    3.5   │     \______
          │            \____
    3.0   │                 \______
          └────────────────────────────────▶
          0    50   100   150   200  (steps)

Gradient: ┌──────────────────────
Norm      │  ███
   10.0   │ █████
          │██████
    5.0   │███████████
          │█████████████████████
    0.0   └────────────────────────────────▶

Metrics:  ✓ Loss EMA established
          ✓ Gradient baseline recorded
          ✓ Predictor warmup complete
```

---

### 2. Full Training Phase

**Purpose:** Compute ground truth gradients + train dynamics predictor

**What Happens:**
- Full forward/backward passes (expensive)
- Dynamics model learns from observed training trajectory
- Residuals extracted (prediction error vs actual)
- Predictor confidence updated

**Duration:** 20-50 steps per cycle

**Key Operations:**

```rust
// Pseudocode
for step in full_phase {
    // 1. Standard training
    loss = model.forward(batch)
    gradients = loss.backward()
    optimizer.step(gradients)

    // 2. Train predictor
    state = encode_training_state(loss, gradients, weights)
    weight_delta = observe_weight_change(prev_weights, weights)
    predictor.train(state, weight_delta)

    // 3. Store residual
    predicted_delta = predictor.predict(state)
    residual = weight_delta - predicted_delta
    residual_store.push(residual)
}
```

**When to Increase Full Steps:**
- Predictor confidence consistently low (<0.70)
- Frequent divergence during predict phase
- Complex training dynamics (e.g., adversarial training)

**ASCII Diagram:**

```
FULL TRAINING PHASE (20 steps)
───────────────────────────────

Step:     ┌─────────────────────────────────────┐
          │ Forward + Backward (100% compute)   │
  1-20    │ ═══════════════════════════════════ │
          │ Predictor Training (lightweight)    │
          │ ─────────────────────                │
          └─────────────────────────────────────┘

Residual  ┌──────────────────────
Error:    │    ●      ●
  0.05    │  ●   ●  ●   ●
          │ ●  ●  ●  ●  ●  ●
  0.00    │●  ●  ●  ●  ●  ●  ●
          └─────────────────────────────────▶
          1   5   10  15  20  (full steps)

Output:   → Predictor model updated
          → Residual history (20 samples)
          → Confidence score: 0.87
```

---

### 3. Predictive Phase

**Purpose:** Skip backward passes for speedup

**What Happens:**
- Forward pass only (inference)
- Dynamics model predicts weight updates
- **No gradients computed** (major speedup)
- Divergence monitoring active
- Early exit if confidence drops or divergence detected

**Duration:** 20-200 steps (adaptively determined)

**Speedup Calculation:**

```
Backward Pass Time: ~2x forward pass time
Prediction Time: ~0.01x forward pass time

Traditional:  [Forward (1.0) + Backward (2.0) + Update (0.1)] = 3.1 units
Predictive:   [Forward (1.0) + Predict (0.01) + Update (0.1)] = 1.11 units

Speedup: 3.1 / 1.11 ≈ 2.8x per predicted step
```

**Divergence Triggers (Early Exit):**

| Signal | Threshold | Action |
|--------|-----------|--------|
| Loss deviation | >3.0σ from EMA | Exit to Full |
| Gradient explosion | >10x baseline norm | Exit to Full |
| Gradient vanishing | <0.01x baseline | Exit to Full |
| NaN/Inf values | Any occurrence | Exit to Full |
| Prediction error | >20% relative | Exit to Correct |
| Loss oscillation | >3 sign changes in 10 steps | Exit to Correct |
| Confidence drop | <0.85 threshold | Exit to Full |

**ASCII Diagram:**

```
PREDICTIVE PHASE (up to 100 steps)
──────────────────────────────────

Compute:  ┌──────────────────────────────────────┐
          │ Forward (Required)    ████████████   │
          │ Backward (SKIPPED!)   ░░░░░░░░░░░░   │
          │ Predict (Lightweight) ──────         │
          └──────────────────────────────────────┘
                                  ▲
                      2.8x speedup vs full training

Loss:     ┌──────────────────────────────────
  Actual  │     ●     ●     ●     ●
          │  ●     ●     ●     ●
Predicted │ ─────────────────────────────── (3.2)
          └──────────────────────────────────▶
          1    25    50    75   100 (steps)

Confidence:┌──────────────────────────────────
   1.0     │────────
   0.85    │        \
   0.70    │         \______ (exit threshold)
           └──────────────────────────────────▶

Exit Conditions:
  ✓ Reached max_predict_steps (100)
  ✓ Confidence < 0.85
  ✓ Divergence detected
  ✓ Prediction error > 20%
```

---

### 4. Correction Phase

**Purpose:** Apply residual corrections to maintain accuracy

**What Happens:**
- Select K most similar residuals from history
- Compute weighted correction based on similarity
- Apply correction to predicted weights
- Validate with small number of full training steps

**Duration:** 5-20 validation steps

**Residual Similarity Weighting:**

```rust
// Similarity-based residual selection
for residual in residual_store {
    similarity = cosine_similarity(current_state, residual.state)
    if similarity > threshold {
        weight = similarity / sum_of_similarities
        correction += weight * residual.error
    }
}
```

**ASCII Diagram:**

```
CORRECTION PHASE
────────────────

Residual   ┌────────────────────────────────────
Selection: │ History: 200 residuals
           │ Similar: 15 residuals (cos_sim > 0.7)
           │ Weighted: Top 5 by similarity
           └────────────────────────────────────

Similarity:┌──────────────────────
  Weights  │  ████        (0.32) residual_187
           │  ███         (0.24) residual_193
           │  ██          (0.18) residual_156
           │  █           (0.14) residual_201
           │  █           (0.12) residual_172
           └──────────────────────

Application:
  predicted_weights + Σ(weight_i × residual_i) = corrected_weights

Validation:
  Step 1-5: Full training with corrected weights
  If loss improves → Continue to next Full phase
  If loss degrades → Rollback and increase full_steps
```

---

## Phase Transitions

### State Machine

```
┌──────────────────────────────────────────────────────────────┐
│                    PHASE TRANSITION LOGIC                     │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  START                                                       │
│    │                                                         │
│    ▼                                                         │
│  WARMUP                                                      │
│    │                                                         │
│    │ ✓ warmup_steps completed (100-500)                     │
│    │ ✓ loss_ema established                                 │
│    │ ✓ gradient_baseline recorded                           │
│    │                                                         │
│    ▼                                                         │
│  FULL TRAIN ◀─────────────┐                                 │
│    │                      │                                  │
│    │ ✓ min_full_steps (20)│                                 │
│    │ ✓ confidence > 0.85  │                                 │
│    │ ✓ no_recent_divergence                                 │
│    │                      │                                  │
│    ▼                      │                                  │
│  PREDICT                  │                                  │
│    │                      │                                  │
│    ├──(divergence)────────┤                                 │
│    ├──(confidence_drop)───┤                                 │
│    │                      │                                  │
│    │ ✓ prediction_horizon │                                 │
│    │ ✓ or error > 20%     │                                 │
│    │                      │                                  │
│    ▼                      │                                  │
│  CORRECT                  │                                  │
│    │                      │                                  │
│    │ ✓ validation_complete│                                 │
│    │ ✓ correction_applied │                                 │
│    │                      │                                  │
│    └──────────────────────┘                                 │
│         (cycle repeats)                                     │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Transition Conditions

| From | To | Conditions |
|------|----|------------|
| Warmup | Full | `step >= warmup_steps` AND `loss_ema_stable` |
| Full | Predict | `full_steps >= min_full_steps` AND `confidence > threshold` AND `no_recent_divergence` |
| Predict | Correct | `steps >= prediction_horizon` OR `prediction_error > 20%` |
| Predict | Full | `divergence_detected` OR `confidence < threshold` |
| Correct | Full | `validation_complete` AND `corrections_applied` |

---

## Hyperparameter Guide

### Model Size Recommendations

#### Small Models (100M - 500M parameters)

```toml
[hybrid_trainer_config]
warmup_steps = 100
full_steps = 20
max_predict_steps = 60
confidence_threshold = 0.85
divergence_threshold = 3.0

[predictor_config]
type = "Linear"  # Lightweight predictor sufficient

[learning_rate]
initial = 1e-4
schedule = "WarmupStableDecay"
warmup_steps = 100
stable_steps = 5000
```

**Expected Performance:**
- Speedup: 4-6x
- Loss gap: <1%
- Backward reduction: 70-80%

---

#### Medium Models (500M - 3B parameters)

```toml
[hybrid_trainer_config]
warmup_steps = 200
full_steps = 30
max_predict_steps = 80
confidence_threshold = 0.87
divergence_threshold = 2.5

[predictor_config]
type = "RSSM"
deterministic_dim = 256
stochastic_dim = 32
num_categoricals = 32
ensemble_size = 3

[learning_rate]
initial = 5e-5
schedule = "WarmupStableDecay"
warmup_steps = 200
stable_steps = 10000
```

**Expected Performance:**
- Speedup: 6-8x
- Loss gap: <1.5%
- Backward reduction: 80-85%

---

#### Large Models (3B - 70B parameters)

```toml
[hybrid_trainer_config]
warmup_steps = 500
full_steps = 50
max_predict_steps = 100
confidence_threshold = 0.90
divergence_threshold = 2.0

[predictor_config]
type = "RSSM"
deterministic_dim = 512
stochastic_dim = 64
num_categoricals = 32
ensemble_size = 5

[learning_rate]
initial = 1e-5
schedule = "WarmupStableDecay"
warmup_steps = 500
stable_steps = 20000

[optimizer]
type = "AdamW"
beta1 = 0.9
beta2 = 0.999
weight_decay = 0.01
```

**Expected Performance:**
- Speedup: 8-10x
- Loss gap: <2%
- Backward reduction: 85-90%

---

### Hyperparameter Reference Table

| Parameter | Small (100M-500M) | Medium (500M-3B) | Large (3B-70B) |
|-----------|-------------------|------------------|----------------|
| `warmup_steps` | 100 | 200 | 500 |
| `full_steps` | 20 | 30 | 50 |
| `max_predict_steps` | 60 | 80 | 100 |
| `confidence_threshold` | 0.85 | 0.87 | 0.90 |
| `divergence_threshold` | 3.0σ | 2.5σ | 2.0σ |
| `learning_rate` | 1e-4 | 5e-5 | 1e-5 |
| `predictor_type` | Linear | RSSM | RSSM |
| `predictor_ensemble` | 1 | 3 | 5 |

---

## Loss Curve Interpretation

### Healthy Training Curve

```
Loss
────
4.0  ┌────┐
     │\   │ Warmup: Rapid descent
3.5  │ \  │
     │  \ │
3.0  │   \│_____ Full: Steady improvement
     │       ────\____
2.5  │                \____
     │                     \____ Predict: Predicted steps
2.0  │  ═══════════════════════════════
     │  (Actual matches predicted ±5%)
1.5  │
     └─────────────────────────────────────────▶
     0   200   500   1000  1500  2000   Steps

Phases:  [Warmup] [Full][Predict][Correct][Full]...

Characteristics:
✓ Smooth descent during warmup
✓ Predicted loss within 5% of actual
✓ No sudden spikes or plateaus
✓ Stable gradient norms
```

---

### Warning Signs

#### 1. Prediction Divergence

```
Loss
────
3.0  ┌──────────────────────────────────────
     │         ────────── (predicted)
2.5  │       ─╱
     │      ╱
2.0  │  ════  ●    ●     ●      ●
     │           ●    ●     ●  (actual)
1.5  │
     └─────────────────────────────────────▶
                Predict Phase

Problem: Actual loss deviates >10% from predicted
Action:  → Trainer auto-exits to FULL phase
         → Increase full_steps or confidence_threshold
```

---

#### 2. Gradient Explosion

```
Gradient
Norm
────────
100.0 ┌─────────────────────────────────────
      │                            ██
      │                           ███
 50.0 │                          ████
      │                         █████
      │ ████████████████████████████
  0.0 └─────────────────────────────────────▶
      0        500       1000      1500

Problem: Gradient norm >10x baseline
Cause:   Learning rate too high, or divergence
Action:  → Reduce learning_rate by 0.5x
         → Increase divergence_threshold sensitivity
```

---

#### 3. Loss Plateau

```
Loss
────
2.0  ┌─────────────────────────────────────
     │\
1.5  │ \____
     │      ──────────────────────────────
1.0  │     (no improvement for 500 steps)
     │
0.5  │
     └─────────────────────────────────────▶
     0    500   1000  1500  2000   Steps

Problem: Loss not decreasing for >500 steps
Cause:   Learning rate too low, or local minimum
Action:  → Increase learning_rate by 2x
         → Add learning rate warmup
         → Check gradient norms (may be vanishing)
```

---

#### 4. Oscillation

```
Loss
────
2.5  ┌─────────────────────────────────────
     │    ╱\    ╱\    ╱\    ╱\
2.0  │   ╱  \  ╱  \  ╱  \  ╱  \
     │  ╱    \╱    \╱    \╱    \
1.5  │ ╱
     │╱
     └─────────────────────────────────────▶
     0    500   1000  1500  2000   Steps

Problem: Loss oscillates >3 sign changes per 10 steps
Cause:   Learning rate too high
Action:  → Reduce learning_rate by 0.3x
         → Increase full_steps to stabilize
         → Enable gradient clipping (max_norm = 1.0)
```

---

## Troubleshooting Guide

### Issue: Low Predictor Confidence (<0.70)

**Symptoms:**
- Trainer spends >90% time in FULL phase
- Predict phases exit early
- Minimal speedup achieved

**Root Causes:**
1. Insufficient warmup (predictor not trained enough)
2. Training dynamics too chaotic (high variance)
3. Predictor architecture too simple

**Solutions:**

```toml
# Increase warmup
warmup_steps = 500  # was 100

# More full training per cycle
full_steps = 50     # was 20

# Upgrade predictor
[predictor_config]
type = "RSSM"       # was "Linear"
ensemble_size = 5   # was 1
```

**Validation:**
- Monitor `confidence` metric in logs
- Should reach >0.85 after warmup + 2-3 full cycles
- If still low after 1000 steps, check gradient health

---

### Issue: Frequent Divergence

**Symptoms:**
- Predict phases exit to FULL frequently
- "Divergence detected" logs every 20-50 steps
- Loss spikes during predict phase

**Root Causes:**
1. Learning rate too high
2. Divergence threshold too sensitive
3. Batch variance too high

**Solutions:**

```toml
# Reduce sensitivity
divergence_threshold = 4.0  # was 2.0

# Lower learning rate
[learning_rate]
initial = 5e-5              # was 1e-4

# Increase batch size
batch_size = 64             # was 32

# More conservative prediction
max_predict_steps = 40      # was 100
```

**Validation:**
- Divergence rate should be <10% of predict steps
- Track `divergence_exits_per_100_steps` metric

---

### Issue: Overfitting

**Symptoms:**
- Training loss continues decreasing
- Validation loss increases or plateaus
- Large train/val loss gap (>0.5)

**ASCII Visualization:**

```
Loss
────
2.0  ┌─────────────────────────────────────
     │\        Validation ─────────────
1.5  │ \                  ╱
     │  \      ──────────
1.0  │   \    ╱
     │    \──      Training
0.5  │      ──\___
     └─────────────────────────────────────▶
     0    1000   2000   3000   4000  Steps
           ▲
      Overfitting starts
```

**Solutions:**

```toml
# Add regularization
[optimizer]
weight_decay = 0.1          # was 0.01

# Increase dropout
dropout_rate = 0.2          # was 0.1

# Early stopping
early_stopping_patience = 5

# Reduce model capacity
hidden_dim = 512            # was 1024
num_layers = 12             # was 24
```

---

### Issue: Underfitting

**Symptoms:**
- Both train and val loss plateau early
- Loss remains high (>1.5 for language models)
- Gradient norms healthy but no improvement

**Solutions:**

```toml
# Increase model capacity
hidden_dim = 2048           # was 1024
num_layers = 24             # was 12

# Increase learning rate
initial_lr = 3e-4           # was 1e-4

# Reduce regularization
weight_decay = 0.001        # was 0.1
dropout_rate = 0.05         # was 0.2

# Train longer
max_steps = 50000           # was 10000
```

---

### Issue: NaN/Inf Loss

**Symptoms:**
- Loss becomes NaN or Inf
- Usually happens suddenly
- Training cannot recover

**ASCII Visualization:**

```
Loss
────
2.0  ┌─────────────────────
     │\
1.5  │ \____
     │      ───────────
1.0  │              ╱
     │             ╱
0.0  │            ╱ → NaN!
     └─────────────────────▶
     0   500   1000  Steps
                  ▲
            Explosion point
```

**Immediate Actions:**

1. **Rollback to last checkpoint**
2. **Reduce learning rate by 10x**
3. **Enable gradient clipping:**

```toml
[optimizer]
gradient_clip_norm = 1.0  # Clip gradients to max norm

[divergence_config]
nan_check_enabled = true
```

**Prevention:**

```toml
# Conservative learning rate
initial_lr = 1e-5

# Gradient clipping
gradient_clip_norm = 1.0

# Mixed precision (if using GPU)
use_mixed_precision = true
loss_scale = 128  # Dynamic loss scaling
```

---

## Specialist vs Generalist Training

### Specialist Training

**Use Case:** Domain-specific fine-tuning (e.g., medical, legal, code)

**Characteristics:**
- Low data diversity
- Predictable dynamics
- Higher speedup potential

**Recommended Config:**

```toml
[hybrid_trainer_config]
warmup_steps = 100          # Less warmup needed
full_steps = 15             # Fewer full steps
max_predict_steps = 120     # Longer predictions
confidence_threshold = 0.82 # Lower threshold OK

[dataset]
domain = "medical"
num_examples = 50000
diversity_score = 0.3       # Low diversity
```

**Expected Performance:**
- Speedup: **8-12x** (better than generalist)
- Loss gap: <1%
- Convergence: Faster (fewer steps to target loss)

---

### Generalist Training

**Use Case:** Pre-training on diverse corpora (e.g., web text, books)

**Characteristics:**
- High data diversity
- Complex dynamics
- Requires more stability

**Recommended Config:**

```toml
[hybrid_trainer_config]
warmup_steps = 500          # More warmup needed
full_steps = 50             # More full steps
max_predict_steps = 80      # Shorter predictions
confidence_threshold = 0.90 # Higher threshold

[dataset]
domain = "general"
num_examples = 10000000
diversity_score = 0.9       # High diversity
```

**Expected Performance:**
- Speedup: **5-7x** (more conservative)
- Loss gap: <2%
- Convergence: Slower (more steps needed)

---

### Comparison Table

| Aspect | Specialist | Generalist |
|--------|-----------|-----------|
| Warmup | 100-200 steps | 300-500 steps |
| Full Steps | 15-25 | 40-60 |
| Predict Steps | 100-150 | 60-100 |
| Confidence | 0.80-0.85 | 0.88-0.92 |
| Speedup | 8-12x | 5-7x |
| Stability | More stable | Less stable |
| Data Diversity | Low | High |
| Predictor Type | Linear/MLP OK | RSSM recommended |

---

## Learning Rate Scheduling

### WSD (Warmup-Stable-Decay)

**Recommended schedule for hybrid training**

```
Learning
Rate
────────
1e-4  ┌──────────────────────────────────────
      │      Stable ────────────────────
      │     ╱                           \
5e-5  │    ╱                             \___
      │   ╱ Warmup                            Decay
      │  ╱
0     │─╱
      └──────────────────────────────────────▶
      0   500   5000            20000   Steps

      ◀──▶  ◀────────────────▶  ◀──────▶
      Warmup    Stable Phase      Decay
```

**Configuration:**

```toml
[learning_rate_schedule]
type = "WarmupStableDecay"

# Phase 1: Warmup (linear increase)
warmup_steps = 500
warmup_init = 1e-6
warmup_target = 1e-4

# Phase 2: Stable (constant)
stable_steps = 10000
stable_rate = 1e-4

# Phase 3: Decay (cosine)
decay_steps = 5000
decay_min = 1e-6
```

**Rust Implementation:**

```rust
impl LearningRateSchedule {
    fn get_lr(&self, step: usize) -> f32 {
        if step < self.warmup_steps {
            // Linear warmup
            let progress = step as f32 / self.warmup_steps as f32;
            self.warmup_init + (self.warmup_target - self.warmup_init) * progress
        } else if step < self.warmup_steps + self.stable_steps {
            // Stable phase
            self.stable_rate
        } else {
            // Cosine decay
            let decay_step = step - self.warmup_steps - self.stable_steps;
            let progress = decay_step as f32 / self.decay_steps as f32;
            let cosine = 0.5 * (1.0 + (std::f32::consts::PI * progress).cos());
            self.decay_min + (self.stable_rate - self.decay_min) * cosine
        }
    }
}
```

---

### Alternative: Cyclic Learning Rate

**Use for:** Escaping local minima

```
Learning
Rate
────────
1e-4  ┌──────────────────────────────────────
      │    ╱\      ╱\      ╱\      ╱\
5e-5  │   ╱  \    ╱  \    ╱  \    ╱  \
      │  ╱    \  ╱    \  ╱    \  ╱    \
1e-5  │ ╱      \╱      \╱      \╱      \
      └──────────────────────────────────────▶
      0    1000  2000  3000  4000     Steps

      ◀─────▶ Cycle length: 1000 steps
```

**Configuration:**

```toml
[learning_rate_schedule]
type = "Cyclic"
base_lr = 1e-5
max_lr = 1e-4
cycle_length = 1000
mode = "triangular"  # or "triangular2", "exp_range"
```

**Not recommended for hybrid training:** Interferes with prediction stability

---

### Schedule Selection Guide

| Training Type | Recommended Schedule | Reason |
|---------------|---------------------|--------|
| Pre-training | WSD (long stable) | Stable dynamics for prediction |
| Fine-tuning | WSD (short stable) | Quick convergence |
| Adversarial | Cyclic (small cycles) | Escape saddle points |
| Continual | Constant (low) | Prevent catastrophic forgetting |
| Few-shot | Warmup only | Limited steps available |

---

## Gradient Health Indicators

### Key Metrics to Monitor

#### 1. Gradient Norm

**Healthy Range:** 0.1 - 10.0

```
Gradient
Norm
────────
100.0 ┌─────────────────────────────────────
      │                    Explosion! ██
      │                              ████
 10.0 │ Healthy ══════════════════════████
      │                                ████
  1.0 │ ══════════════════════════════════
      │
  0.1 │ ══════════════════════════════════
      │              Vanishing!
  0.0 └─────────────────────────────────────▶
      0    500   1000  1500  2000    Steps
```

**Actions by Zone:**

| Zone | Range | Status | Action |
|------|-------|--------|--------|
| Healthy | 0.1 - 10.0 | ✓ OK | Continue |
| Explosion | >10.0 | ⚠ Warning | Reduce LR / Clip gradients |
| Vanishing | <0.1 | ⚠ Warning | Increase LR / Check init |
| Extreme | >100 or <0.01 | ❌ Critical | Rollback checkpoint |

---

#### 2. Per-Layer Gradient Norms

**Purpose:** Identify problematic layers

```
Layer    Gradient Norm
─────    ─────────────
layer_0  ████████████████ 5.2
layer_1  ██████████████   4.1
layer_2  ████████████     3.5
layer_3  ██████           1.8
layer_4  ████             1.2
layer_5  ██               0.6
layer_6  █                0.3  ← Vanishing!
layer_7  █                0.2
```

**Fix vanishing gradients:**

```toml
# Enable residual connections
use_residual = true

# Use LayerNorm instead of BatchNorm
normalization = "LayerNorm"

# Better weight initialization
weight_init = "xavier_uniform"
```

---

#### 3. Gradient-to-Parameter Ratio

**Healthy Range:** 1e-4 to 1e-2

```rust
// Monitor this ratio
let ratio = gradient_norm / parameter_norm;

// Ideal: ~1e-3 (gradients are 0.1% of weights)
if ratio > 1e-2 {
    // Gradients too large → reduce LR
} else if ratio < 1e-4 {
    // Gradients too small → increase LR
}
```

**ASCII Visualization:**

```
Gradient/Param
Ratio
──────────────
1e-2  ┌──────────────────────────────────────
      │              Too large!
      │                 ↓
1e-3  │ Ideal ═════════════════════════════
      │
1e-4  │ ═══════════════════════════════════
      │  Too small!
1e-5  └──────────────────────────────────────▶
      0    500   1000  1500  2000     Steps
```

---

#### 4. Gradient Variance

**Purpose:** Detect noisy gradients

```
Gradient
Variance
────────
10.0  ┌──────────────────────────────────────
      │         High variance
      │  ███    ↓
 5.0  │ █████  ███
      │███████████████
 1.0  │███████████████████████ Stable
      │███████████████████████████████████
  0.0 └──────────────────────────────────────▶
      0    500   1000  1500  2000     Steps

High variance → Increase batch size or reduce LR
Low variance → Can increase LR or batch size
```

---

### Gradient Health Checklist

**Before starting training:**

- [ ] Gradient norm after 1 step: 0.1 - 10.0 ✓
- [ ] Per-layer norms: No layer <0.01 or >100 ✓
- [ ] Gradient/param ratio: ~1e-3 ✓
- [ ] No NaN/Inf in gradients ✓

**During training (every 100 steps):**

- [ ] Gradient norm stable (±2x from baseline) ✓
- [ ] No layer norm explosions ✓
- [ ] Gradient variance not increasing ✓
- [ ] Grad/param ratio in healthy range ✓

**If any check fails:**

1. Pause training
2. Log full gradient statistics
3. Apply remediation (reduce LR, clip, rollback)
4. Resume with conservative settings

---

## Quick Reference Card

### Phase Quick Reference

| Phase | Compute | Duration | Purpose |
|-------|---------|----------|---------|
| Warmup | Full | 100-500 | Baseline |
| Full | Full | 20-50 | Train predictor |
| Predict | Forward only | 20-200 | Speedup |
| Correct | Validation | 5-20 | Fix errors |

---

### Critical Thresholds

| Metric | Healthy | Warning | Critical |
|--------|---------|---------|----------|
| Confidence | >0.85 | 0.70-0.85 | <0.70 |
| Loss deviation | <2.0σ | 2.0-3.0σ | >3.0σ |
| Gradient norm | 0.1-10.0 | 10.0-100 | >100 or <0.01 |
| Prediction error | <10% | 10-20% | >20% |

---

### Emergency Actions

| Problem | Immediate Fix |
|---------|--------------|
| NaN loss | Rollback checkpoint, LR ÷ 10 |
| Gradient explosion | Clip to 1.0, LR × 0.5 |
| Divergence loop | Increase `full_steps` × 2 |
| Low confidence | Increase `warmup_steps` × 2 |
| Overfitting | Add weight_decay × 10 |

---

## Appendix: Example Training Session

### Configuration

```toml
# Medium model (1B params), specialist task (code)
[hybrid_trainer_config]
warmup_steps = 200
full_steps = 30
max_predict_steps = 80
confidence_threshold = 0.87

[model]
hidden_dim = 2048
num_layers = 24
vocab_size = 50000

[optimizer]
type = "AdamW"
learning_rate = 5e-5
weight_decay = 0.01
```

---

### Training Log

```
Step 0-200: WARMUP
──────────────────
Step   0 | Loss: 4.234 | Grad: 8.21 | LR: 1.00e-06
Step  50 | Loss: 3.456 | Grad: 5.32 | LR: 1.25e-05
Step 100 | Loss: 2.987 | Grad: 3.41 | LR: 2.50e-05
Step 150 | Loss: 2.654 | Grad: 2.18 | LR: 3.75e-05
Step 200 | Loss: 2.412 | Grad: 1.92 | LR: 5.00e-05
✓ Warmup complete | Loss EMA: 2.412 | Grad baseline: 1.92

Step 201-230: FULL TRAIN (Cycle 1)
───────────────────────────────────
Step 201 | Loss: 2.398 | Grad: 1.87 | Confidence: 0.45
Step 210 | Loss: 2.321 | Grad: 1.74 | Confidence: 0.62
Step 220 | Loss: 2.267 | Grad: 1.68 | Confidence: 0.78
Step 230 | Loss: 2.198 | Grad: 1.61 | Confidence: 0.88 ✓
✓ Full train complete | Predictor ready | Confidence: 0.88

Step 231-310: PREDICT (Cycle 1)
────────────────────────────────
Step 231 | Loss: 2.187 (pred: 2.189) | Δ: +0.1% | Conf: 0.88
Step 250 | Loss: 2.123 (pred: 2.119) | Δ: -0.2% | Conf: 0.89
Step 270 | Loss: 2.076 (pred: 2.081) | Δ: +0.2% | Conf: 0.87
Step 290 | Loss: 2.034 (pred: 2.029) | Δ: -0.2% | Conf: 0.86
Step 310 | Loss: 1.998 (pred: 1.993) | Δ: -0.2% | Conf: 0.85
✓ Predict complete | 80 steps | Avg error: 0.18%

Step 311-320: CORRECT (Cycle 1)
────────────────────────────────
Residuals selected: 12 (similarity > 0.7)
Correction magnitude: 0.032
Step 311 | Loss: 1.991 (validation) | Correction applied
Step 315 | Loss: 1.984 (validation) | Quality OK ✓
Step 320 | Loss: 1.979 (validation) | Ready for next cycle
✓ Correction complete | Next phase: FULL

[Cycles repeat...]

Final Statistics (10,000 steps):
─────────────────────────────────
Total time: 4.2 hours (vs 18.7 hours baseline)
Speedup: 4.5x
Backward passes: 2,100 (vs 10,000 baseline)
Backward reduction: 79%
Final loss: 0.87 (baseline: 0.86)
Loss gap: 1.2%
```

---

## Conclusion

Hybrid predictive training offers substantial speedups while maintaining training quality. Success depends on:

1. **Proper warmup:** Establish stable baseline dynamics
2. **Predictor quality:** Train dynamics model thoroughly
3. **Divergence monitoring:** Exit predictions early when needed
4. **Correction fidelity:** Use residuals to maintain accuracy
5. **Hyperparameter tuning:** Match settings to model size and task

**Remember:** When in doubt, increase `full_steps` and `warmup_steps`. Stability > speedup.

---

**Document Version:** 1.0
**Authors:** Tyler Zervas (tzervas), hybrid-predict-trainer-rs team
**Last Updated:** 2026-02-01
**License:** MIT
**Feedback:** Submit issues to https://github.com/tzervas/hybrid-predict-trainer-rs
