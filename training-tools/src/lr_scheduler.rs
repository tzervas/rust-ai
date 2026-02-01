//! Learning Rate Schedulers for training
//!
//! This module provides learning rate scheduling strategies for neural network training.
//! The primary implementation is the Warmup-Stable-Decay (WSD) scheduler, which provides
//! a simple yet effective learning rate schedule used in many modern training pipelines.
//!
//! # WSD Schedule Phases
//!
//! 1. **Warmup** (default 1%): Linear ramp from 0 to peak learning rate
//! 2. **Stable** (default ~80%): Constant at peak learning rate
//! 3. **Decay** (default ~19%): Cosine decay from peak to minimum learning rate
//!
//! # Example
//!
//! ```rust
//! use training_tools::lr_scheduler::{LRScheduler, WSDScheduler};
//!
//! let scheduler = WSDScheduler::builder()
//!     .total_steps(10000)
//!     .peak_lr(1e-4)
//!     .min_lr(1e-6)
//!     .warmup_fraction(0.01)
//!     .decay_fraction(0.19)
//!     .build()
//!     .unwrap();
//!
//! // Get learning rate at step 5000 (stable phase)
//! let lr = scheduler.get_lr(5000);
//! assert!((lr - 1e-4).abs() < 1e-10);
//! ```

use std::f32::consts::PI;

/// Trait for learning rate schedulers.
///
/// Implementations provide a learning rate value for each training step,
/// allowing for various scheduling strategies (warmup, decay, cyclical, etc.).
pub trait LRScheduler: Send + Sync {
    /// Get the learning rate for the given step.
    ///
    /// # Arguments
    ///
    /// * `step` - The current training step (0-indexed)
    ///
    /// # Returns
    ///
    /// The learning rate to use at this step.
    fn get_lr(&self, step: u64) -> f32;

    /// Get the total number of steps this scheduler is configured for.
    fn total_steps(&self) -> u64;

    /// Get the current phase name for debugging/logging.
    fn phase_name(&self, step: u64) -> &'static str;
}

/// Warmup-Stable-Decay (WSD) learning rate scheduler.
///
/// This scheduler implements a three-phase learning rate schedule:
///
/// 1. **Warmup**: Linear ramp from 0 to `peak_lr`
/// 2. **Stable**: Constant at `peak_lr`
/// 3. **Decay**: Cosine decay from `peak_lr` to `min_lr`
///
/// The WSD schedule is widely used in LLM training (e.g., GPT-3, LLaMA) due to
/// its simplicity and effectiveness.
#[derive(Debug, Clone)]
pub struct WSDScheduler {
    /// Peak learning rate (used during stable phase)
    peak_lr: f32,
    /// Minimum learning rate (reached at end of decay)
    min_lr: f32,
    /// Number of warmup steps
    warmup_steps: u64,
    /// Number of stable steps
    stable_steps: u64,
    /// Number of decay steps
    decay_steps: u64,
}

impl WSDScheduler {
    /// Create a new WSD scheduler with explicit step counts.
    ///
    /// For easier configuration, prefer using [`WSDScheduler::builder()`].
    ///
    /// # Arguments
    ///
    /// * `peak_lr` - Peak learning rate
    /// * `min_lr` - Minimum learning rate at end of decay
    /// * `warmup_steps` - Number of warmup steps
    /// * `stable_steps` - Number of stable steps
    /// * `decay_steps` - Number of decay steps
    pub fn new(
        peak_lr: f32,
        min_lr: f32,
        warmup_steps: u64,
        stable_steps: u64,
        decay_steps: u64,
    ) -> Self {
        Self {
            peak_lr,
            min_lr,
            warmup_steps,
            stable_steps,
            decay_steps,
        }
    }

    /// Create a builder for configuring a WSD scheduler.
    pub fn builder() -> WSDSchedulerBuilder {
        WSDSchedulerBuilder::default()
    }

    /// Get the step at which warmup ends (exclusive).
    #[inline]
    pub fn warmup_end(&self) -> u64 {
        self.warmup_steps
    }

    /// Get the step at which stable phase ends (exclusive).
    #[inline]
    pub fn stable_end(&self) -> u64 {
        self.warmup_steps + self.stable_steps
    }

    /// Get the step at which decay ends (exclusive) - same as total steps.
    #[inline]
    pub fn decay_end(&self) -> u64 {
        self.warmup_steps + self.stable_steps + self.decay_steps
    }

    /// Get the number of warmup steps.
    #[inline]
    pub fn warmup_steps(&self) -> u64 {
        self.warmup_steps
    }

    /// Get the number of stable steps.
    #[inline]
    pub fn stable_steps(&self) -> u64 {
        self.stable_steps
    }

    /// Get the number of decay steps.
    #[inline]
    pub fn decay_steps(&self) -> u64 {
        self.decay_steps
    }

    /// Get the peak learning rate.
    #[inline]
    pub fn peak_lr(&self) -> f32 {
        self.peak_lr
    }

    /// Get the minimum learning rate.
    #[inline]
    pub fn min_lr(&self) -> f32 {
        self.min_lr
    }
}

impl LRScheduler for WSDScheduler {
    fn get_lr(&self, step: u64) -> f32 {
        // Clamp step to valid range
        let step = step.min(self.total_steps().saturating_sub(1));

        if step < self.warmup_steps {
            // Warmup phase: linear ramp from 0 to peak_lr
            if self.warmup_steps == 0 {
                self.peak_lr
            } else {
                let progress = step as f32 / self.warmup_steps as f32;
                self.peak_lr * progress
            }
        } else if step < self.warmup_steps + self.stable_steps {
            // Stable phase: constant at peak_lr
            self.peak_lr
        } else {
            // Decay phase: cosine decay from peak_lr to min_lr
            if self.decay_steps == 0 {
                self.min_lr
            } else {
                let decay_start = self.warmup_steps + self.stable_steps;
                let progress = (step - decay_start) as f32 / self.decay_steps as f32;
                // Cosine decay: starts at peak_lr, ends at min_lr
                let cosine_factor = (1.0 + (PI * progress).cos()) / 2.0;
                self.min_lr + (self.peak_lr - self.min_lr) * cosine_factor
            }
        }
    }

    fn total_steps(&self) -> u64 {
        self.warmup_steps + self.stable_steps + self.decay_steps
    }

    fn phase_name(&self, step: u64) -> &'static str {
        if step < self.warmup_steps {
            "warmup"
        } else if step < self.warmup_steps + self.stable_steps {
            "stable"
        } else {
            "decay"
        }
    }
}

/// Builder for [`WSDScheduler`].
///
/// Allows configuring the scheduler using either:
/// - Explicit step counts (`warmup_steps`, `stable_steps`, `decay_steps`)
/// - Fractions of total steps (`warmup_fraction`, `decay_fraction`)
///
/// # Defaults
///
/// - `peak_lr`: 1e-4
/// - `min_lr`: 1e-6 (1% of peak)
/// - `warmup_fraction`: 0.01 (1% of total steps)
/// - `decay_fraction`: 0.19 (19% of total steps)
/// - `stable_fraction`: 0.80 (80% of total steps, computed automatically)
#[derive(Debug, Clone)]
pub struct WSDSchedulerBuilder {
    total_steps: Option<u64>,
    peak_lr: f32,
    min_lr: f32,
    warmup_steps: Option<u64>,
    stable_steps: Option<u64>,
    decay_steps: Option<u64>,
    warmup_fraction: f32,
    decay_fraction: f32,
}

impl Default for WSDSchedulerBuilder {
    fn default() -> Self {
        Self {
            total_steps: None,
            peak_lr: 1e-4,
            min_lr: 1e-6,
            warmup_steps: None,
            stable_steps: None,
            decay_steps: None,
            warmup_fraction: 0.01, // 1% warmup
            decay_fraction: 0.19,  // 19% decay
        }
    }
}

impl WSDSchedulerBuilder {
    /// Set the total number of training steps.
    ///
    /// Required if using fraction-based configuration.
    pub fn total_steps(mut self, steps: u64) -> Self {
        self.total_steps = Some(steps);
        self
    }

    /// Set the peak learning rate.
    ///
    /// Default: 1e-4
    pub fn peak_lr(mut self, lr: f32) -> Self {
        self.peak_lr = lr;
        self
    }

    /// Set the minimum learning rate (at end of decay).
    ///
    /// Default: 1e-6
    pub fn min_lr(mut self, lr: f32) -> Self {
        self.min_lr = lr;
        self
    }

    /// Set the warmup fraction (0.0 to 1.0).
    ///
    /// This is the fraction of total steps spent in warmup.
    /// Default: 0.01 (1%)
    pub fn warmup_fraction(mut self, fraction: f32) -> Self {
        self.warmup_fraction = fraction.clamp(0.0, 1.0);
        self
    }

    /// Set the decay fraction (0.0 to 1.0).
    ///
    /// This is the fraction of total steps spent in decay.
    /// Default: 0.19 (19%)
    pub fn decay_fraction(mut self, fraction: f32) -> Self {
        self.decay_fraction = fraction.clamp(0.0, 1.0);
        self
    }

    /// Set explicit warmup steps (overrides fraction).
    pub fn warmup_steps(mut self, steps: u64) -> Self {
        self.warmup_steps = Some(steps);
        self
    }

    /// Set explicit stable steps (overrides computed value).
    pub fn stable_steps(mut self, steps: u64) -> Self {
        self.stable_steps = Some(steps);
        self
    }

    /// Set explicit decay steps (overrides fraction).
    pub fn decay_steps(mut self, steps: u64) -> Self {
        self.decay_steps = Some(steps);
        self
    }

    /// Build the scheduler.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `total_steps` is not set when using fractions
    /// - Learning rates are invalid (negative, NaN, or min > peak)
    /// - Fractions sum to more than 1.0
    pub fn build(self) -> Result<WSDScheduler, SchedulerError> {
        // Validate learning rates
        if self.peak_lr <= 0.0 || self.peak_lr.is_nan() {
            return Err(SchedulerError::InvalidLearningRate {
                name: "peak_lr",
                value: self.peak_lr,
            });
        }
        if self.min_lr < 0.0 || self.min_lr.is_nan() {
            return Err(SchedulerError::InvalidLearningRate {
                name: "min_lr",
                value: self.min_lr,
            });
        }
        if self.min_lr > self.peak_lr {
            return Err(SchedulerError::MinExceedsPeak {
                min_lr: self.min_lr,
                peak_lr: self.peak_lr,
            });
        }

        // Validate fractions
        if self.warmup_fraction + self.decay_fraction > 1.0 {
            return Err(SchedulerError::FractionsExceedOne {
                warmup: self.warmup_fraction,
                decay: self.decay_fraction,
            });
        }

        // Compute step counts
        let (warmup_steps, stable_steps, decay_steps) = if let (Some(w), Some(s), Some(d)) =
            (self.warmup_steps, self.stable_steps, self.decay_steps)
        {
            // All explicit - use as-is
            (w, s, d)
        } else {
            // Need total_steps for fraction-based computation
            let total = self.total_steps.ok_or(SchedulerError::MissingTotalSteps)?;

            let warmup = self
                .warmup_steps
                .unwrap_or_else(|| (total as f32 * self.warmup_fraction) as u64);
            let decay = self
                .decay_steps
                .unwrap_or_else(|| (total as f32 * self.decay_fraction) as u64);
            let stable = self
                .stable_steps
                .unwrap_or_else(|| total.saturating_sub(warmup).saturating_sub(decay));

            (warmup, stable, decay)
        };

        Ok(WSDScheduler::new(
            self.peak_lr,
            self.min_lr,
            warmup_steps,
            stable_steps,
            decay_steps,
        ))
    }
}

/// Errors that can occur when building a scheduler.
#[derive(Debug, thiserror::Error)]
pub enum SchedulerError {
    /// Missing total_steps when using fraction-based configuration.
    #[error("total_steps must be set when using fraction-based configuration")]
    MissingTotalSteps,

    /// Invalid learning rate value.
    #[error("invalid {name}: {value} (must be positive and finite)")]
    InvalidLearningRate {
        /// Name of the parameter
        name: &'static str,
        /// Invalid value
        value: f32,
    },

    /// Minimum LR exceeds peak LR.
    #[error("min_lr ({min_lr}) cannot exceed peak_lr ({peak_lr})")]
    MinExceedsPeak {
        /// Minimum learning rate
        min_lr: f32,
        /// Peak learning rate
        peak_lr: f32,
    },

    /// Warmup + decay fractions exceed 1.0.
    #[error("warmup ({warmup}) + decay ({decay}) fractions exceed 1.0")]
    FractionsExceedOne {
        /// Warmup fraction
        warmup: f32,
        /// Decay fraction
        decay: f32,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-6;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < EPSILON
    }

    #[test]
    fn test_warmup_phase() {
        let scheduler = WSDScheduler::builder()
            .total_steps(1000)
            .peak_lr(1.0)
            .min_lr(0.0)
            .warmup_fraction(0.1) // 100 steps warmup
            .decay_fraction(0.0) // No decay for simpler test
            .build()
            .unwrap();

        // Step 0: should be 0
        assert!(approx_eq(scheduler.get_lr(0), 0.0));

        // Step 50: should be 0.5 (halfway through warmup)
        assert!(approx_eq(scheduler.get_lr(50), 0.5));

        // Step 99: should be close to 1.0
        assert!(approx_eq(scheduler.get_lr(99), 0.99));

        // Phase check
        assert_eq!(scheduler.phase_name(0), "warmup");
        assert_eq!(scheduler.phase_name(50), "warmup");
        assert_eq!(scheduler.phase_name(99), "warmup");
    }

    #[test]
    fn test_stable_phase() {
        let scheduler = WSDScheduler::builder()
            .total_steps(1000)
            .peak_lr(1.0)
            .min_lr(0.0)
            .warmup_fraction(0.1) // 100 steps warmup
            .decay_fraction(0.1) // 100 steps decay
            .build()
            .unwrap();

        // Stable phase: steps 100-899
        assert!(approx_eq(scheduler.get_lr(100), 1.0));
        assert!(approx_eq(scheduler.get_lr(500), 1.0));
        assert!(approx_eq(scheduler.get_lr(899), 1.0));

        // Phase check
        assert_eq!(scheduler.phase_name(100), "stable");
        assert_eq!(scheduler.phase_name(500), "stable");
        assert_eq!(scheduler.phase_name(899), "stable");
    }

    #[test]
    fn test_decay_phase() {
        let scheduler = WSDScheduler::builder()
            .total_steps(1000)
            .peak_lr(1.0)
            .min_lr(0.0)
            .warmup_fraction(0.0) // No warmup
            .decay_fraction(1.0) // All decay
            .build()
            .unwrap();

        // Step 0: peak LR (start of cosine)
        assert!(approx_eq(scheduler.get_lr(0), 1.0));

        // Step 500: halfway through cosine decay = 0.5
        assert!(approx_eq(scheduler.get_lr(500), 0.5));

        // Step 999: end of decay, should be very close to min_lr
        // Note: At step 999 of 1000, cos(pi * 0.999) ≈ -0.99995, so LR ≈ 0.00025
        let lr_999 = scheduler.get_lr(999);
        assert!(
            lr_999 < 0.001,
            "LR at step 999 should be < 0.001, got {}",
            lr_999
        );

        // Phase check
        assert_eq!(scheduler.phase_name(0), "decay");
        assert_eq!(scheduler.phase_name(999), "decay");
    }

    #[test]
    fn test_default_fractions() {
        // Default: 1% warmup, 80% stable, 19% decay
        let scheduler = WSDScheduler::builder()
            .total_steps(10000)
            .peak_lr(1e-4)
            .min_lr(1e-6)
            .build()
            .unwrap();

        assert_eq!(scheduler.warmup_steps(), 100); // 1%
        assert_eq!(scheduler.stable_steps(), 8000); // 80%
        assert_eq!(scheduler.decay_steps(), 1900); // 19%
        assert_eq!(scheduler.total_steps(), 10000);
    }

    #[test]
    fn test_cosine_decay_values() {
        let scheduler = WSDScheduler::new(
            1.0, // peak_lr
            0.1, // min_lr
            0,   // no warmup
            0,   // no stable
            100, // 100 decay steps
        );

        // At step 0: should be peak_lr
        assert!(approx_eq(scheduler.get_lr(0), 1.0));

        // At step 25 (25%): cos(π*0.25) = cos(45°) ≈ 0.707
        // factor = (1 + 0.707) / 2 ≈ 0.854
        // lr = 0.1 + (1.0 - 0.1) * 0.854 ≈ 0.868
        let lr_25 = scheduler.get_lr(25);
        assert!(lr_25 > 0.8 && lr_25 < 0.9, "lr at 25%: {}", lr_25);

        // At step 50 (50%): cos(π*0.5) = 0
        // factor = (1 + 0) / 2 = 0.5
        // lr = 0.1 + (1.0 - 0.1) * 0.5 = 0.55
        assert!(approx_eq(scheduler.get_lr(50), 0.55));

        // At step 75 (75%): cos(π*0.75) = cos(135°) ≈ -0.707
        // factor = (1 - 0.707) / 2 ≈ 0.146
        // lr = 0.1 + (1.0 - 0.1) * 0.146 ≈ 0.232
        let lr_75 = scheduler.get_lr(75);
        assert!(lr_75 > 0.2 && lr_75 < 0.3, "lr at 75%: {}", lr_75);

        // At step 99: should be close to min_lr
        let lr_99 = scheduler.get_lr(99);
        assert!(lr_99 < 0.15, "lr at 99: {}", lr_99);
    }

    #[test]
    fn test_full_wsd_schedule() {
        let scheduler = WSDScheduler::builder()
            .total_steps(10000)
            .peak_lr(1e-4)
            .min_lr(1e-6)
            .warmup_fraction(0.01) // 100 steps
            .decay_fraction(0.19) // 1900 steps
            .build()
            .unwrap();

        // Warmup: 0-99
        assert!(scheduler.get_lr(0) < 1e-6);
        assert!(scheduler.get_lr(50) < 1e-4);
        assert!(scheduler.get_lr(50) > 0.0);

        // Stable: 100-8099
        assert!(approx_eq(scheduler.get_lr(100), 1e-4));
        assert!(approx_eq(scheduler.get_lr(5000), 1e-4));
        assert!(approx_eq(scheduler.get_lr(8099), 1e-4));

        // Decay: 8100-9999
        let lr_decay_start = scheduler.get_lr(8100);
        let lr_decay_mid = scheduler.get_lr(9050);
        let lr_decay_end = scheduler.get_lr(9999);

        assert!(lr_decay_start > lr_decay_mid);
        assert!(lr_decay_mid > lr_decay_end);
        assert!(lr_decay_end > 1e-6 - EPSILON);
    }

    #[test]
    fn test_step_clamping() {
        let scheduler = WSDScheduler::builder()
            .total_steps(100)
            .peak_lr(1.0)
            .min_lr(0.0)
            .warmup_fraction(0.0)
            .decay_fraction(1.0)
            .build()
            .unwrap();

        // Steps beyond total should clamp to last step
        let lr_at_99 = scheduler.get_lr(99);
        let lr_at_100 = scheduler.get_lr(100);
        let lr_at_1000 = scheduler.get_lr(1000);

        assert!(approx_eq(lr_at_99, lr_at_100));
        assert!(approx_eq(lr_at_99, lr_at_1000));
    }

    #[test]
    fn test_zero_warmup() {
        let scheduler = WSDScheduler::builder()
            .total_steps(100)
            .peak_lr(1.0)
            .min_lr(0.0)
            .warmup_fraction(0.0)
            .decay_fraction(0.5)
            .build()
            .unwrap();

        // Should start at peak_lr
        assert!(approx_eq(scheduler.get_lr(0), 1.0));
        assert_eq!(scheduler.phase_name(0), "stable");
    }

    #[test]
    fn test_zero_decay() {
        let scheduler = WSDScheduler::builder()
            .total_steps(100)
            .peak_lr(1.0)
            .min_lr(0.0)
            .warmup_fraction(0.1)
            .decay_fraction(0.0)
            .build()
            .unwrap();

        // Should end at peak_lr (no decay)
        assert!(approx_eq(scheduler.get_lr(99), 1.0));
        assert_eq!(scheduler.phase_name(99), "stable");
    }

    #[test]
    fn test_explicit_steps() {
        let scheduler = WSDScheduler::builder()
            .warmup_steps(50)
            .stable_steps(100)
            .decay_steps(50)
            .peak_lr(1.0)
            .min_lr(0.0)
            .build()
            .unwrap();

        assert_eq!(scheduler.total_steps(), 200);
        assert_eq!(scheduler.warmup_steps(), 50);
        assert_eq!(scheduler.stable_steps(), 100);
        assert_eq!(scheduler.decay_steps(), 50);
    }

    #[test]
    fn test_builder_errors() {
        // Missing total_steps
        let result = WSDScheduler::builder().peak_lr(1.0).build();
        assert!(matches!(result, Err(SchedulerError::MissingTotalSteps)));

        // Negative peak_lr
        let result = WSDScheduler::builder()
            .total_steps(100)
            .peak_lr(-1.0)
            .build();
        assert!(matches!(
            result,
            Err(SchedulerError::InvalidLearningRate { .. })
        ));

        // min_lr > peak_lr
        let result = WSDScheduler::builder()
            .total_steps(100)
            .peak_lr(1e-4)
            .min_lr(1e-3)
            .build();
        assert!(matches!(result, Err(SchedulerError::MinExceedsPeak { .. })));

        // Fractions exceed 1.0
        let result = WSDScheduler::builder()
            .total_steps(100)
            .warmup_fraction(0.6)
            .decay_fraction(0.6)
            .build();
        assert!(matches!(
            result,
            Err(SchedulerError::FractionsExceedOne { .. })
        ));
    }

    #[test]
    fn test_phase_boundaries() {
        let scheduler = WSDScheduler::new(
            1.0, // peak_lr
            0.0, // min_lr
            100, // warmup
            800, // stable
            100, // decay
        );

        // Warmup: 0-99
        assert_eq!(scheduler.phase_name(0), "warmup");
        assert_eq!(scheduler.phase_name(99), "warmup");

        // Stable: 100-899
        assert_eq!(scheduler.phase_name(100), "stable");
        assert_eq!(scheduler.phase_name(899), "stable");

        // Decay: 900-999
        assert_eq!(scheduler.phase_name(900), "decay");
        assert_eq!(scheduler.phase_name(999), "decay");
    }

    #[test]
    fn test_warmup_end_stable_end() {
        let scheduler = WSDScheduler::new(1.0, 0.0, 100, 800, 100);

        assert_eq!(scheduler.warmup_end(), 100);
        assert_eq!(scheduler.stable_end(), 900);
        assert_eq!(scheduler.decay_end(), 1000);
    }
}
