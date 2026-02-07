//! Gradient accumulation for memory-efficient training with large effective batch sizes.
//!
//! Gradient accumulation allows training with effective batch sizes larger than
//! what fits in VRAM by:
//! 1. Computing gradients on small micro-batches
//! 2. Accumulating gradients across multiple micro-batches
//! 3. Updating weights once after accumulation completes
//!
//! # Memory Savings
//!
//! Without gradient accumulation:
//! - Batch size 32: 32× model memory for activations
//!
//! With gradient accumulation (4 steps):
//! - Micro-batch size 8: 8× model memory for activations
//! - Effective batch size: 32 (4 × 8)
//! - Memory savings: 4× reduction in activation memory
//!
//! Expected savings: **30-40% total VRAM usage**
//!
//! # Benefits
//!
//! - **Larger effective batches**: Better gradient estimates, faster convergence
//! - **Memory efficiency**: Train large models with limited VRAM
//! - **No quality loss**: Mathematically equivalent to large-batch training
//!
//! # Usage
//!
//! ```rust
//! use hybrid_predict_trainer_rs::gradient_accumulation::GradientAccumulationConfig;
//!
//! let config = GradientAccumulationConfig {
//!     enabled: true,
//!     accumulation_steps: 4,
//!     scale_lr: true,
//! };
//!
//! // Effective batch = micro_batch × accumulation_steps
//! // Example: 8 × 4 = 32 effective batch size
//! ```
//!
//! # Integration with HybridTrainer
//!
//! Gradient accumulation integrates seamlessly with hybrid training:
//!
//! - **Warmup/Full**: Accumulate gradients, update every N steps
//! - **Predict**: No gradients, no accumulation needed
//! - **Correct**: Accumulate correction gradients
//!
//! This provides maximum memory savings during Full/Correct phases where
//! gradients are computed.

use serde::{Deserialize, Serialize};

/// Configuration for gradient accumulation.
///
/// Controls how gradients are accumulated across micro-batches before
/// applying weight updates.
///
/// # Example
///
/// ```rust
/// use hybrid_predict_trainer_rs::gradient_accumulation::GradientAccumulationConfig;
///
/// // Accumulate over 4 steps (4× memory savings in activations)
/// let config = GradientAccumulationConfig {
///     enabled: true,
///     accumulation_steps: 4,
///     scale_lr: true, // Automatically scale LR by 1/4
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientAccumulationConfig {
    /// Whether gradient accumulation is enabled.
    ///
    /// If false, weight updates happen every step (standard training).
    pub enabled: bool,

    /// Number of micro-batches to accumulate before updating weights.
    ///
    /// Effective batch size = micro_batch_size × accumulation_steps
    ///
    /// **Recommended values**:
    /// - 2-4: Moderate memory savings
    /// - 8-16: Aggressive memory savings
    /// - 32+: Maximum savings for very large models
    ///
    /// **Tradeoff**: Higher values = more memory savings but slower updates
    pub accumulation_steps: usize,

    /// Whether to automatically scale learning rate.
    ///
    /// When enabled, effective LR = base_lr / accumulation_steps
    ///
    /// This maintains gradient magnitude consistency:
    /// - Without scaling: Gradients × N, learning too fast
    /// - With scaling: Gradients × N, LR ÷ N, effective LR same
    ///
    /// **Recommendation**: Enable for most use cases
    pub scale_lr: bool,

    /// Normalize gradients by accumulation steps.
    ///
    /// When enabled, accumulated gradients are divided by accumulation_steps
    /// before applying optimizer update. This keeps gradient magnitudes
    /// consistent with non-accumulated training.
    ///
    /// **Note**: Either use `scale_lr` OR `normalize_gradients`, not both.
    /// They achieve the same effect through different mechanisms.
    pub normalize_gradients: bool,
}

impl GradientAccumulationConfig {
    /// Creates a new gradient accumulation config with defaults.
    ///
    /// Default: Disabled (standard training)
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a disabled config (standard training).
    #[must_use]
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            accumulation_steps: 1,
            scale_lr: false,
            normalize_gradients: false,
        }
    }

    /// Creates a moderate config (2× savings).
    #[must_use]
    pub fn moderate() -> Self {
        Self {
            enabled: true,
            accumulation_steps: 2,
            scale_lr: true,
            normalize_gradients: false,
        }
    }

    /// Creates an aggressive config (4× savings).
    #[must_use]
    pub fn aggressive() -> Self {
        Self {
            enabled: true,
            accumulation_steps: 4,
            scale_lr: true,
            normalize_gradients: false,
        }
    }

    /// Creates a maximum savings config (8× savings).
    #[must_use]
    pub fn maximum() -> Self {
        Self {
            enabled: true,
            accumulation_steps: 8,
            scale_lr: true,
            normalize_gradients: false,
        }
    }

    /// Returns effective learning rate given base LR.
    ///
    /// If `scale_lr` is enabled, returns base_lr / accumulation_steps.
    /// Otherwise returns base_lr unchanged.
    #[must_use]
    pub fn effective_learning_rate(&self, base_lr: f32) -> f32 {
        if self.enabled && self.scale_lr {
            base_lr / self.accumulation_steps as f32
        } else {
            base_lr
        }
    }

    /// Returns effective batch size given micro-batch size.
    ///
    /// Effective batch = micro_batch × accumulation_steps
    #[must_use]
    pub fn effective_batch_size(&self, micro_batch_size: usize) -> usize {
        if self.enabled {
            micro_batch_size * self.accumulation_steps
        } else {
            micro_batch_size
        }
    }

    /// Returns gradient scaling factor.
    ///
    /// When `normalize_gradients` is enabled, gradients should be
    /// divided by this factor before optimizer update.
    #[must_use]
    pub fn gradient_scale_factor(&self) -> f32 {
        if self.enabled && self.normalize_gradients {
            1.0 / self.accumulation_steps as f32
        } else {
            1.0
        }
    }

    /// Estimates memory savings vs non-accumulated training.
    ///
    /// Returns a multiplier between 0.0 and 1.0 indicating approximate
    /// VRAM usage relative to non-accumulated training.
    ///
    /// # Formula
    ///
    /// Activation memory is proportional to batch size. With accumulation:
    /// - Original: batch_size × model_memory
    /// - Accumulated: (batch_size / accumulation_steps) × model_memory
    /// - Savings: 1 / accumulation_steps
    ///
    /// Gradient memory remains constant (accumulated in-place).
    ///
    /// # Arguments
    ///
    /// * `activation_fraction` - Fraction of total memory used by activations (typically 0.4-0.6)
    ///
    /// # Example
    ///
    /// ```rust
    /// use hybrid_predict_trainer_rs::gradient_accumulation::GradientAccumulationConfig;
    ///
    /// let config = GradientAccumulationConfig::aggressive(); // 4 steps
    ///
    /// // If activations are 50% of memory:
    /// let savings = config.estimate_memory_savings(0.5);
    /// // Result: ~0.625 (37.5% savings)
    /// // Calculation: 50% activations ÷ 4 + 50% other = 12.5% + 50% = 62.5%
    /// ```
    #[must_use]
    pub fn estimate_memory_savings(&self, activation_fraction: f32) -> f32 {
        if !self.enabled || self.accumulation_steps <= 1 {
            return 1.0; // No savings
        }

        // Activation memory is reduced by accumulation factor
        let activation_memory = activation_fraction / self.accumulation_steps as f32;

        // Other memory (model, gradients, optimizer state) unchanged
        let other_memory = 1.0 - activation_fraction;

        activation_memory + other_memory
    }

    /// Validates the configuration.
    ///
    /// # Returns
    ///
    /// `Ok(())` if valid, `Err(String)` with error message if invalid.
    pub fn validate(&self) -> Result<(), String> {
        if self.enabled {
            if self.accumulation_steps == 0 {
                return Err("accumulation_steps must be > 0".to_string());
            }

            if self.scale_lr && self.normalize_gradients {
                return Err(
                    "Cannot enable both scale_lr and normalize_gradients (they're redundant)"
                        .to_string(),
                );
            }

            if self.accumulation_steps == 1 {
                return Err(
                    "accumulation_steps=1 is equivalent to disabled; set enabled=false instead"
                        .to_string(),
                );
            }
        }

        Ok(())
    }
}

impl Default for GradientAccumulationConfig {
    /// Creates a disabled config (standard training).
    fn default() -> Self {
        Self::disabled()
    }
}

/// Gradient accumulation state tracker.
///
/// Tracks current position in accumulation cycle and whether
/// an optimizer step should be performed.
///
/// # Example
///
/// ```rust
/// use hybrid_predict_trainer_rs::gradient_accumulation::{
///     GradientAccumulationConfig, GradientAccumulationState
/// };
///
/// let config = GradientAccumulationConfig::aggressive();
/// let mut state = GradientAccumulationState::new(&config);
///
/// // Process 4 micro-batches
/// for _ in 0..4 {
///     state.accumulate_step();
///     if state.should_update_weights() {
///         // Apply optimizer update
///         println!("Update weights!");
///         state.reset();
///     }
/// }
/// ```
#[derive(Debug, Clone)]
pub struct GradientAccumulationState {
    /// Number of gradients accumulated so far.
    pub accumulated_count: usize,

    /// Target accumulation steps.
    pub target_steps: usize,

    /// Whether accumulation is enabled.
    pub enabled: bool,
}

impl GradientAccumulationState {
    /// Creates a new accumulation state from config.
    #[must_use]
    pub fn new(config: &GradientAccumulationConfig) -> Self {
        Self {
            accumulated_count: 0,
            target_steps: config.accumulation_steps,
            enabled: config.enabled,
        }
    }

    /// Records one gradient accumulation step.
    pub fn accumulate_step(&mut self) {
        if self.enabled {
            self.accumulated_count += 1;
        }
    }

    /// Returns true if weights should be updated this step.
    ///
    /// Weights should update when:
    /// - Accumulation is disabled (every step), OR
    /// - Accumulated count reaches target steps
    #[must_use]
    pub fn should_update_weights(&self) -> bool {
        if !self.enabled {
            return true; // Always update when disabled
        }

        self.accumulated_count >= self.target_steps
    }

    /// Resets accumulation counter after weight update.
    pub fn reset(&mut self) {
        self.accumulated_count = 0;
    }

    /// Returns progress through current accumulation cycle (0.0-1.0).
    #[must_use]
    pub fn progress(&self) -> f32 {
        if !self.enabled || self.target_steps == 0 {
            return 1.0;
        }

        (self.accumulated_count as f32 / self.target_steps as f32).min(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_disabled() {
        let config = GradientAccumulationConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.accumulation_steps, 1);
    }

    #[test]
    fn test_moderate_config() {
        let config = GradientAccumulationConfig::moderate();
        assert!(config.enabled);
        assert_eq!(config.accumulation_steps, 2);
        assert!(config.scale_lr);
    }

    #[test]
    fn test_aggressive_config() {
        let config = GradientAccumulationConfig::aggressive();
        assert_eq!(config.accumulation_steps, 4);
    }

    #[test]
    fn test_effective_learning_rate() {
        let config = GradientAccumulationConfig::aggressive();
        let base_lr = 0.001;
        let effective_lr = config.effective_learning_rate(base_lr);
        assert!((effective_lr - 0.00025).abs() < 1e-6);
    }

    #[test]
    fn test_effective_batch_size() {
        let config = GradientAccumulationConfig::aggressive();
        let micro_batch = 8;
        let effective_batch = config.effective_batch_size(micro_batch);
        assert_eq!(effective_batch, 32); // 8 × 4
    }

    #[test]
    fn test_memory_savings() {
        let config = GradientAccumulationConfig::aggressive(); // 4 steps

        // If activations are 50% of memory
        let savings = config.estimate_memory_savings(0.5);

        // Expected: 50% / 4 + 50% = 12.5% + 50% = 62.5%
        assert!((savings - 0.625).abs() < 0.01);
    }

    #[test]
    fn test_accumulation_state() {
        let config = GradientAccumulationConfig::aggressive();
        let mut state = GradientAccumulationState::new(&config);

        // First 3 steps: accumulate only
        for i in 0..3 {
            state.accumulate_step();
            assert!(!state.should_update_weights(), "Step {i}");
        }

        // 4th step: should update
        state.accumulate_step();
        assert!(state.should_update_weights());

        // Reset for next cycle
        state.reset();
        assert_eq!(state.accumulated_count, 0);
        assert!(!state.should_update_weights());
    }

    #[test]
    fn test_disabled_always_updates() {
        let config = GradientAccumulationConfig::disabled();
        let mut state = GradientAccumulationState::new(&config);

        for _ in 0..10 {
            state.accumulate_step();
            assert!(state.should_update_weights());
            state.reset();
        }
    }

    #[test]
    fn test_validation() {
        let mut config = GradientAccumulationConfig::aggressive();
        assert!(config.validate().is_ok());

        // Invalid: both scale_lr and normalize_gradients
        config.scale_lr = true;
        config.normalize_gradients = true;
        assert!(config.validate().is_err());

        // Invalid: accumulation_steps = 0
        config.normalize_gradients = false;
        config.accumulation_steps = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_progress() {
        let config = GradientAccumulationConfig::aggressive(); // 4 steps
        let mut state = GradientAccumulationState::new(&config);

        assert_eq!(state.progress(), 0.0);

        state.accumulate_step();
        assert!((state.progress() - 0.25).abs() < 0.01);

        state.accumulate_step();
        assert!((state.progress() - 0.5).abs() < 0.01);

        state.accumulate_step();
        state.accumulate_step();
        assert!((state.progress() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_gradient_scale_factor() {
        let mut config = GradientAccumulationConfig::aggressive();
        config.normalize_gradients = true;
        config.scale_lr = false;

        let scale = config.gradient_scale_factor();
        assert!((scale - 0.25).abs() < 1e-6); // 1/4
    }
}
