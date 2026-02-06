# Loss Velocity/Acceleration Metrics Implementation

## Overview

This document describes the implementation of loss velocity and acceleration computation in the training-tools crate. These metrics track the rate of change and acceleration of loss during training, enabling real-time monitoring of training dynamics.

## What Was Changed

### 1. **New `LossDynamicsTracker` Struct** (`src/training_state.rs`)

A reusable tracker that maintains a sliding window of loss values and computes velocity/acceleration on demand:

```rust
pub struct LossDynamicsTracker {
    loss_history: Vec<f32>,
    window_size: usize,
}

impl LossDynamicsTracker {
    pub fn new(window_size: usize) -> Self { ... }
    pub fn update(&mut self, loss: f32) -> (f32, f32) { ... }
    pub fn history(&self) -> &[f32] { ... }
    pub fn reset(&mut self) { ... }
}
```

**Key Features:**
- Maintains circular buffer of recent loss values (default window: 30 steps)
- Computes velocity (first derivative) using linear regression
- Computes acceleration (second derivative) by comparing velocity in first/second halves
- Returns `(velocity, acceleration)` tuple on each update

### 2. **Updated `ProgressiveTrainer`** (`src/progressive.rs`)

The main training loop now uses actual computed metrics instead of hardcoded 0.0 values:

**Before:**
```rust
let metrics = StepMetrics {
    // ...
    loss_velocity: 0.0,      // Hardcoded!
    loss_acceleration: 0.0,  // Hardcoded!
    // ...
};
```

**After:**
```rust
let (loss_velocity, loss_acceleration) =
    self.loss_dynamics_tracker.update(result.loss);

let metrics = StepMetrics {
    // ...
    loss_velocity,
    loss_acceleration,
    // ...
};
```

**Integration Points:**
- Added `loss_dynamics_tracker: LossDynamicsTracker` field to `ProgressiveTrainer`
- Initialized in `new()` with window size 30
- Updated at each training step before creating metrics

### 3. **Public API Export** (`src/lib.rs`)

Added `LossDynamicsTracker` to public exports:
```rust
pub use training_state::{
    // ...
    LossDynamicsTracker,
    // ...
};
```

### 4. **Comprehensive Tests** (`src/training_state.rs`)

Added 5 new test cases:
- `test_loss_dynamics_tracker_creation` - Basic initialization
- `test_loss_dynamics_tracker_decreasing_loss` - Improving loss scenario
- `test_loss_dynamics_tracker_window_size` - Window size enforcement
- `test_loss_dynamics_tracker_reset` - Reset functionality
- `test_loss_dynamics_tracker_acceleration` - Acceleration computation

### 5. **Usage Example** (`examples/loss_dynamics_tracking.rs`)

Demonstrates real-world scenarios:
- **Scenario 1:** Ideal training (steady improvement)
- **Scenario 2:** Converging training (slowing improvement)
- **Scenario 3:** Plateau (stuck at constant loss)
- **Scenario 4:** Diverging (loss getting worse)

Run with:
```bash
cargo run --example loss_dynamics_tracking
```

## Metrics Interpretation

### Loss Velocity
- **Negative:** Loss is decreasing (improving training)
- **Zero:** Loss is stable (plateau)
- **Positive:** Loss is increasing (diverging)

### Loss Acceleration
- **Negative:** Improvement is accelerating
- **Zero:** Improvement is linear/stable
- **Positive:** Improvement is slowing (convergence nearing)

### Common Patterns

| Pattern | Velocity | Acceleration | Interpretation |
|---------|----------|--------------|-----------------|
| Healthy training | Negative | Near 0-negative | Good progress |
| Converging | Negative | Positive | Approaching minimum |
| Plateau | ~0 | ~0 | Stuck, may need LR adjust |
| Diverging | Positive | Variable | Training failing, check LR |
| Oscillating | Alternating | Alternating | Unstable, too high LR |

## Implementation Details

### Velocity Calculation (Linear Regression)

Computes slope of recent losses using least-squares regression:

```
velocity = Σ(x - x_mean)(y - y_mean) / Σ(x - x_mean)²

where:
  x = step index [0, 1, 2, ...]
  y = loss values
  window = last N losses
```

### Acceleration Calculation (Velocity Difference)

Splits recent window in half and compares velocities:

```
v1 = velocity(first_half)
v2 = velocity(second_half)
acceleration = v2 - v1
```

- If v2 > v1 (less negative): improvement slowing → positive acceleration
- If v2 < v1 (more negative): improvement accelerating → negative acceleration

### Window Size

Default: 30 steps (configurable)
- Larger window: smoother metrics, slower response
- Smaller window: noisier metrics, faster response

## Usage in Training Loop

**Basic Usage:**
```rust
let mut tracker = LossDynamicsTracker::new(30);

for step in 0..num_steps {
    // ... forward/backward pass ...
    let loss = compute_loss();

    // Update tracker and get dynamics
    let (velocity, acceleration) = tracker.update(loss);

    // Create metrics with computed values
    let metrics = StepMetrics {
        step,
        loss,
        loss_velocity: velocity,
        loss_acceleration: acceleration,
        // ... other fields ...
    };

    // Log metrics
    save_metrics(&metrics);
}
```

## Where Hardcoded Values Still Exist

The following files have hardcoded 0.0 for loss_velocity/loss_acceleration in **test fixtures**:

- `src/adaptive/controller.rs` - Test helper function (intentional for testing)
- `src/adaptive/integration.rs` - Test helper function (intentional for testing)
- `src/parquet_export.rs` - Test data generation (intentional for testing)
- `tests/integration_tests.rs` - Test cases (intentional for testing)
- `tests/visual_tests.rs` - Visual test data (intentional for testing)

These are kept as-is since they are test/fixture data. Only production training code has been updated.

## Benefits

1. **Real-time Training Monitoring:** Track learning dynamics at each step
2. **Early Problem Detection:** Identify divergence, plateau, instability early
3. **Learning Rate Optimization:** Use velocity to adjust LR dynamically
4. **Convergence Analysis:** Monitor acceleration to detect convergence point
5. **Better Logging:** Export meaningful metrics for analysis and visualization

## File Changes Summary

| File | Changes |
|------|---------|
| `src/training_state.rs` | Added `LossDynamicsTracker` struct + tests |
| `src/progressive.rs` | Use tracker in training loop, initialize in new() |
| `src/lib.rs` | Export `LossDynamicsTracker` |
| `examples/loss_dynamics_tracking.rs` | NEW: Demonstration examples |

## Performance Impact

- **Memory:** ~240 bytes per tracker (30 f32 values)
- **CPU:** Linear regression O(N) per update, negligible overhead
- **No GPU impact:** All computation on CPU

## Future Enhancements

1. Exponential moving average variant for faster response
2. Multi-window analysis (compare 10-step vs 30-step windows)
3. Anomaly detection (detect NaN/Inf loss early)
4. Automatic LR adjustment based on dynamics
5. Integration with LR advisor for dynamic scheduling
