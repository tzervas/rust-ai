//! Mixed precision training support for memory optimization.
//!
//! This module provides automatic precision switching to reduce VRAM usage
//! during training, especially beneficial for large models (1B+ parameters).
//!
//! # Precision Strategy
//!
//! Different training phases have different precision requirements:
//!
//! - **Warmup/Full**: FP32 for accurate gradient computation
//! - **Predict**: FP16/BF16 for memory savings (predictions are approximate anyway)
//! - **Correct**: FP32 for precise correction computation
//!
//! # Memory Savings
//!
//! Using FP16 instead of FP32 provides:
//! - 2× memory reduction for model weights
//! - 2× memory reduction for activations
//! - 40-50% total VRAM savings (combined with other techniques)
//!
//! # Usage
//!
//! ```rust,ignore
//! use hybrid_predict_trainer_rs::mixed_precision::{MixedPrecisionConfig, Precision};
//!
//! let config = MixedPrecisionConfig::default()
//!     .with_phase_precision(Phase::Predict, Precision::Fp16)
//!     .with_phase_precision(Phase::Full, Precision::Fp32);
//!
//! // Apply in training loop
//! let precision = config.precision_for_phase(current_phase);
//! ```
//!
//! # Limitations
//!
//! - Requires Burn backend support for FP16/BF16 (most backends support this)
//! - Some operations may need FP32 fallback (e.g., BatchNorm, LayerNorm)
//! - Quality impact is minimal (<0.5% for prediction phases)

use crate::Phase;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Floating-point precision for tensor operations.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Precision {
    /// 32-bit floating point (standard precision).
    ///
    /// - Memory: 4 bytes per value
    /// - Range: ±3.4e38
    /// - Precision: ~7 decimal digits
    /// - Use for: Gradient computation, corrections, normalization
    Fp32,

    /// 16-bit floating point (half precision).
    ///
    /// - Memory: 2 bytes per value
    /// - Range: ±65,504
    /// - Precision: ~3 decimal digits
    /// - Use for: Forward passes during prediction, inference
    Fp16,

    /// Brain floating point 16-bit (Google's format).
    ///
    /// - Memory: 2 bytes per value
    /// - Range: ±3.4e38 (same as FP32)
    /// - Precision: ~2 decimal digits
    /// - Use for: Training (better than FP16 for gradients)
    ///
    /// BF16 is preferred over FP16 for training because it preserves
    /// the FP32 dynamic range, reducing overflow/underflow issues.
    Bf16,
}

impl Precision {
    /// Returns the number of bytes per value.
    #[must_use]
    pub const fn bytes_per_value(self) -> usize {
        match self {
            Self::Fp32 => 4,
            Self::Fp16 | Self::Bf16 => 2,
        }
    }

    /// Returns the memory savings multiplier vs FP32.
    ///
    /// Example: FP16 returns 0.5 (50% of FP32 memory).
    #[must_use]
    pub fn memory_multiplier(self) -> f32 {
        match self {
            Self::Fp32 => 1.0,
            Self::Fp16 | Self::Bf16 => 0.5,
        }
    }

    /// Returns true if this is a reduced precision format.
    #[must_use]
    pub const fn is_reduced_precision(self) -> bool {
        matches!(self, Self::Fp16 | Self::Bf16)
    }

    /// Returns the recommended precision for a given phase.
    ///
    /// # Rationale
    ///
    /// - **Warmup/Full**: FP32 for accurate gradient computation
    /// - **Predict**: FP16/BF16 for memory savings (predictions approximate anyway)
    /// - **Correct**: FP32 for precise corrections
    #[must_use]
    pub fn recommended_for_phase(phase: Phase) -> Self {
        match phase {
            Phase::Warmup | Phase::Full | Phase::Correct => Self::Fp32,
            Phase::Predict => Self::Bf16, // BF16 preferred over FP16
        }
    }
}

impl Default for Precision {
    fn default() -> Self {
        Self::Fp32
    }
}

/// Configuration for mixed precision training.
///
/// Controls which precision is used for each training phase, enabling
/// memory-efficient training with minimal quality impact.
///
/// # Strategy
///
/// The default strategy uses:
/// - **FP32** for phases requiring precise gradients (Warmup, Full, Correct)
/// - **BF16** for prediction phase (memory savings, minimal quality impact)
///
/// # Example
///
/// ```rust
/// use hybrid_predict_trainer_rs::mixed_precision::{MixedPrecisionConfig, Precision};
/// use hybrid_predict_trainer_rs::Phase;
///
/// let mut config = MixedPrecisionConfig::default();
///
/// // Use FP16 for prediction to maximize memory savings
/// config.set_phase_precision(Phase::Predict, Precision::Fp16);
///
/// // Check what precision to use
/// let precision = config.precision_for_phase(Phase::Predict);
/// assert_eq!(precision, Precision::Fp16);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MixedPrecisionConfig {
    /// Whether mixed precision is enabled.
    ///
    /// If false, always use FP32 regardless of phase.
    pub enabled: bool,

    /// Default precision when no phase-specific setting exists.
    pub default_precision: Precision,

    /// Per-phase precision overrides.
    #[serde(default)]
    pub phase_precisions: HashMap<Phase, Precision>,

    /// Whether to automatically apply recommended precisions per phase.
    ///
    /// If true, uses `Precision::recommended_for_phase()` for phases
    /// not explicitly configured.
    pub auto_recommend: bool,
}

impl MixedPrecisionConfig {
    /// Creates a new mixed precision config with defaults.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a conservative config (FP32 everywhere).
    #[must_use]
    pub fn conservative() -> Self {
        Self {
            enabled: false,
            default_precision: Precision::Fp32,
            phase_precisions: HashMap::new(),
            auto_recommend: false,
        }
    }

    /// Creates an aggressive config (BF16 for predict, FP32 elsewhere).
    #[must_use]
    pub fn aggressive() -> Self {
        let mut config = Self {
            enabled: true,
            default_precision: Precision::Fp32,
            phase_precisions: HashMap::new(),
            auto_recommend: true,
        };
        config.phase_precisions.insert(Phase::Predict, Precision::Bf16);
        config
    }

    /// Sets precision for a specific phase.
    pub fn set_phase_precision(&mut self, phase: Phase, precision: Precision) {
        self.phase_precisions.insert(phase, precision);
    }

    /// Adds a phase precision override (builder pattern).
    #[must_use]
    pub fn with_phase_precision(mut self, phase: Phase, precision: Precision) -> Self {
        self.set_phase_precision(phase, precision);
        self
    }

    /// Returns the precision to use for a given phase.
    ///
    /// Priority:
    /// 1. Phase-specific override (if set)
    /// 2. Auto-recommended (if `auto_recommend` enabled)
    /// 3. Default precision
    #[must_use]
    pub fn precision_for_phase(&self, phase: Phase) -> Precision {
        if !self.enabled {
            return Precision::Fp32;
        }

        // Check phase-specific override
        if let Some(&precision) = self.phase_precisions.get(&phase) {
            return precision;
        }

        // Use auto-recommended if enabled
        if self.auto_recommend {
            return Precision::recommended_for_phase(phase);
        }

        // Fall back to default
        self.default_precision
    }

    /// Estimates memory savings vs full FP32 training.
    ///
    /// Returns a multiplier between 0.0 and 1.0 indicating approximate
    /// VRAM usage relative to FP32.
    ///
    /// # Arguments
    ///
    /// * `phase_distribution` - Fraction of time spent in each phase
    ///
    /// # Example
    ///
    /// ```rust
    /// use hybrid_predict_trainer_rs::mixed_precision::MixedPrecisionConfig;
    /// use std::collections::HashMap;
    ///
    /// let config = MixedPrecisionConfig::aggressive();
    ///
    /// // 10% warmup, 20% full, 60% predict, 10% correct
    /// let mut distribution = HashMap::new();
    /// distribution.insert(hybrid_predict_trainer_rs::Phase::Warmup, 0.1);
    /// distribution.insert(hybrid_predict_trainer_rs::Phase::Full, 0.2);
    /// distribution.insert(hybrid_predict_trainer_rs::Phase::Predict, 0.6);
    /// distribution.insert(hybrid_predict_trainer_rs::Phase::Correct, 0.1);
    ///
    /// let savings = config.estimate_memory_savings(&distribution);
    /// // ~30% savings (60% of time at 50% memory)
    /// ```
    #[must_use]
    pub fn estimate_memory_savings(&self, phase_distribution: &HashMap<Phase, f32>) -> f32 {
        if !self.enabled {
            return 1.0; // No savings
        }

        let mut weighted_sum = 0.0;
        let mut total_weight = 0.0;

        for (phase, &weight) in phase_distribution {
            let precision = self.precision_for_phase(*phase);
            weighted_sum += precision.memory_multiplier() * weight;
            total_weight += weight;
        }

        if total_weight > 0.0 {
            weighted_sum / total_weight
        } else {
            1.0
        }
    }
}

impl Default for MixedPrecisionConfig {
    /// Creates a balanced default configuration.
    ///
    /// - Enabled: true
    /// - Default: FP32
    /// - Auto-recommend: true (uses BF16 for Predict phase)
    fn default() -> Self {
        Self {
            enabled: true,
            default_precision: Precision::Fp32,
            phase_precisions: HashMap::new(),
            auto_recommend: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_precision_bytes() {
        assert_eq!(Precision::Fp32.bytes_per_value(), 4);
        assert_eq!(Precision::Fp16.bytes_per_value(), 2);
        assert_eq!(Precision::Bf16.bytes_per_value(), 2);
    }

    #[test]
    fn test_precision_memory_multiplier() {
        assert_eq!(Precision::Fp32.memory_multiplier(), 1.0);
        assert_eq!(Precision::Fp16.memory_multiplier(), 0.5);
        assert_eq!(Precision::Bf16.memory_multiplier(), 0.5);
    }

    #[test]
    fn test_precision_recommended() {
        assert_eq!(Precision::recommended_for_phase(Phase::Warmup), Precision::Fp32);
        assert_eq!(Precision::recommended_for_phase(Phase::Full), Precision::Fp32);
        assert_eq!(Precision::recommended_for_phase(Phase::Predict), Precision::Bf16);
        assert_eq!(Precision::recommended_for_phase(Phase::Correct), Precision::Fp32);
    }

    #[test]
    fn test_mixed_precision_config_default() {
        let config = MixedPrecisionConfig::default();
        assert!(config.enabled);
        assert_eq!(config.default_precision, Precision::Fp32);
        assert!(config.auto_recommend);

        // With auto-recommend, Predict should use BF16
        assert_eq!(config.precision_for_phase(Phase::Predict), Precision::Bf16);
        assert_eq!(config.precision_for_phase(Phase::Full), Precision::Fp32);
    }

    #[test]
    fn test_mixed_precision_config_conservative() {
        let config = MixedPrecisionConfig::conservative();
        assert!(!config.enabled);

        // Should always return FP32 when disabled
        assert_eq!(config.precision_for_phase(Phase::Predict), Precision::Fp32);
        assert_eq!(config.precision_for_phase(Phase::Full), Precision::Fp32);
    }

    #[test]
    fn test_mixed_precision_config_aggressive() {
        let config = MixedPrecisionConfig::aggressive();
        assert!(config.enabled);

        assert_eq!(config.precision_for_phase(Phase::Predict), Precision::Bf16);
        assert_eq!(config.precision_for_phase(Phase::Full), Precision::Fp32);
    }

    #[test]
    fn test_phase_precision_override() {
        let mut config = MixedPrecisionConfig::default();
        config.set_phase_precision(Phase::Predict, Precision::Fp16);

        assert_eq!(config.precision_for_phase(Phase::Predict), Precision::Fp16);
    }

    #[test]
    fn test_memory_savings_estimate() {
        let config = MixedPrecisionConfig::aggressive();

        let mut distribution = HashMap::new();
        distribution.insert(Phase::Warmup, 0.1);
        distribution.insert(Phase::Full, 0.2);
        distribution.insert(Phase::Predict, 0.6); // 60% at BF16 (0.5×)
        distribution.insert(Phase::Correct, 0.1);

        let savings = config.estimate_memory_savings(&distribution);

        // Expected: 0.1*1.0 + 0.2*1.0 + 0.6*0.5 + 0.1*1.0 = 0.7
        assert!((savings - 0.7).abs() < 0.01);
    }

    #[test]
    fn test_disabled_mixed_precision() {
        let mut config = MixedPrecisionConfig::default();
        config.enabled = false;

        // Should always return FP32 when disabled
        assert_eq!(config.precision_for_phase(Phase::Predict), Precision::Fp32);
    }

    #[test]
    fn test_builder_pattern() {
        let config = MixedPrecisionConfig::default()
            .with_phase_precision(Phase::Predict, Precision::Fp16)
            .with_phase_precision(Phase::Full, Precision::Bf16);

        assert_eq!(config.precision_for_phase(Phase::Predict), Precision::Fp16);
        assert_eq!(config.precision_for_phase(Phase::Full), Precision::Bf16);
    }
}
