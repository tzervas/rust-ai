//! Gradient range tuning and adaptive gradient clipping.
//!
//! This module implements automatic gradient range tuning with phase-specific thresholds
//! and per-layer adaptive gradient clipping (AGC). The tuner tracks gradient statistics
//! across training phases and computes appropriate clipping factors to maintain gradient
//! health and training stability.
//!
//! # Why Adaptive Gradient Clipping?
//!
//! Gradient magnitude varies significantly across:
//! - Training phases (warmup vs mid-training vs convergence)
//! - Network layers (early layers have larger gradients than later ones)
//! - Training steps (transient spikes need different handling than sustained issues)
//!
//! Traditional fixed-threshold gradient clipping either:
//! - Clips too aggressively, slowing convergence
//! - Clips too conservatively, allowing spikes that destabilize training
//!
//! This module uses phase-aware thresholds and per-layer statistics to adapt
//! clipping dynamically, maintaining gradient health while minimizing unnecessary clipping.
//!
//! # Phase-Specific Thresholds
//!
//! Different training phases require different gradient ranges:
//! - **Warmup**: (0.01, 0.5) - gradients are noisy, allow higher clipping threshold
//! - **Early**: (0.1, 1.0) - aggressive learning, moderate clipping
//! - **Mid**: (0.3, 0.8) - stable regime, tight control
//! - **Late**: (0.1, 0.5) - fine-tuning, precision-focused
//!
//! # Per-Layer AGC
//!
//! Adaptive Gradient Clipping (AGC) from [AGC: Adaptive Gradient Clipping](https://arxiv.org/abs/2102.06171)
//! clips gradients relative to the ratio of weight and gradient norms:
//!
//! ```text
//! clipping_coeff = lambda * (weight_norm + epsilon) / (gradient_norm + epsilon)
//! clipped_grad = min(clipping_coeff, 1.0) * gradient
//! ```
//!
//! This prevents the weight-to-gradient ratio from growing unbounded,
//! which can cause training instability.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Phase-specific gradient thresholds.
///
/// Defines the acceptable gradient range (min, max) for different training phases.
/// These thresholds guide the tuner in deciding when to apply clipping.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PhaseGradientThresholds {
    /// Minimum acceptable gradient norm.
    pub min: f32,
    /// Maximum acceptable gradient norm.
    pub max: f32,
}

impl PhaseGradientThresholds {
    /// Creates new thresholds with the given min and max values.
    #[must_use]
    pub fn new(min: f32, max: f32) -> Self {
        Self { min, max }
    }

    /// Checks if a gradient norm is within the acceptable range.
    #[must_use]
    pub fn contains(&self, gradient_norm: f32) -> bool {
        gradient_norm >= self.min && gradient_norm <= self.max
    }

    /// Computes how far a gradient norm deviates from the acceptable range.
    ///
    /// Returns:
    /// - Negative value if gradient is too small (magnitude of deviation)
    /// - 0.0 if within range
    /// - Positive value if gradient is too large (magnitude of deviation)
    #[must_use]
    pub fn deviation(&self, gradient_norm: f32) -> f32 {
        if gradient_norm < self.min {
            gradient_norm - self.min
        } else if gradient_norm > self.max {
            gradient_norm - self.max
        } else {
            0.0
        }
    }
}

/// Per-layer gradient statistics for adaptive gradient clipping.
///
/// Tracks exponential moving averages (EMA) of gradient and weight norms
/// for each layer, enabling per-layer AGC clipping factor computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerGradientStats {
    /// Layer name for identification.
    pub name: String,

    /// EMA of L2 norm of gradients for this layer.
    grad_norm_ema: f32,

    /// EMA of L2 norm of weights for this layer.
    weight_norm_ema: f32,

    /// Ratio of clipped to total gradients (for monitoring).
    pub clip_ratio: f32,

    /// Total number of times this layer's gradients were clipped.
    pub clip_count: u64,

    /// Number of updates applied to this layer.
    update_count: u64,
}

impl LayerGradientStats {
    /// Creates new statistics for a layer.
    #[must_use]
    pub fn new(name: String) -> Self {
        Self {
            name,
            grad_norm_ema: 0.0,
            weight_norm_ema: 0.0,
            clip_ratio: 0.0,
            clip_count: 0,
            update_count: 0,
        }
    }

    /// Updates the EMAs with new gradient and weight norms.
    ///
    /// Uses an EMA decay of 0.99 to smooth out transient spikes
    /// while responding to sustained changes in gradient magnitude.
    ///
    /// # Arguments
    ///
    /// * `grad_norm` - L2 norm of gradients for this layer
    /// * `weight_norm` - L2 norm of weights for this layer
    pub fn update(&mut self, grad_norm: f32, weight_norm: f32) {
        const EMA_DECAY: f32 = 0.99;

        if self.update_count == 0 {
            self.grad_norm_ema = grad_norm;
            self.weight_norm_ema = weight_norm;
        } else {
            self.grad_norm_ema = EMA_DECAY * self.grad_norm_ema + (1.0 - EMA_DECAY) * grad_norm;
            self.weight_norm_ema =
                EMA_DECAY * self.weight_norm_ema + (1.0 - EMA_DECAY) * weight_norm;
        }

        self.update_count += 1;
    }

    /// Computes the adaptive gradient clipping factor using AGC.
    ///
    /// The clipping coefficient is computed as:
    /// ```text
    /// clipping_coeff = lambda * (weight_norm_ema + epsilon) / (grad_norm_ema + epsilon)
    /// clipped_coeff = min(clipping_coeff, 1.0)
    /// ```
    ///
    /// A return value of 1.0 means no clipping; values < 1.0 indicate the
    /// gradient should be scaled down by that factor.
    ///
    /// # Arguments
    ///
    /// * `lambda` - AGC control parameter (default 0.04)
    ///
    /// # Returns
    ///
    /// The clipping factor to multiply gradients by. Values <= 1.0 indicate clipping.
    #[must_use]
    pub fn compute_agc_clip_factor(&self, lambda: f32) -> f32 {
        const EPSILON: f32 = 1e-6;

        let weight_norm = self.weight_norm_ema.max(EPSILON);
        let grad_norm = self.grad_norm_ema.max(EPSILON);

        let clipping_coeff = lambda * weight_norm / grad_norm;

        // Clipping factor is at most 1.0 (no amplification allowed)
        clipping_coeff.min(1.0).max(0.0)
    }

    /// Returns the current gradient norm EMA.
    #[must_use]
    pub fn grad_norm_ema(&self) -> f32 {
        self.grad_norm_ema
    }

    /// Returns the current weight norm EMA.
    #[must_use]
    pub fn weight_norm_ema(&self) -> f32 {
        self.weight_norm_ema
    }

    /// Records that gradients for this layer were clipped.
    pub fn record_clip(&mut self, clipped: bool) {
        if clipped {
            self.clip_count += 1;
        }
    }

    /// Updates the clip ratio based on total gradient updates.
    pub fn update_clip_ratio(&mut self, total_updates: u64) {
        self.clip_ratio = if total_updates > 0 {
            self.clip_count as f32 / total_updates as f32
        } else {
            0.0
        };
    }
}

/// Training phase for gradient range tuning.
///
/// Categorizes training progress into distinct phases, each with
/// different gradient characteristics and tuning strategies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TrainingProgressPhase {
    /// Warmup phase (0-25% of training).
    Warmup,
    /// Early phase (25-50% of training).
    Early,
    /// Mid phase (50-75% of training).
    Mid,
    /// Late phase (75-100% of training).
    Late,
}

impl TrainingProgressPhase {
    /// Determines the current phase from training progress percentage.
    ///
    /// # Arguments
    ///
    /// * `progress_pct` - Training progress as percentage (0.0 to 100.0)
    ///
    /// # Returns
    ///
    /// The [`TrainingProgressPhase`] corresponding to the progress level.
    #[must_use]
    pub fn from_progress(progress_pct: f32) -> Self {
        let pct = progress_pct.clamp(0.0, 100.0);
        if pct < 25.0 {
            TrainingProgressPhase::Warmup
        } else if pct < 50.0 {
            TrainingProgressPhase::Early
        } else if pct < 75.0 {
            TrainingProgressPhase::Mid
        } else {
            TrainingProgressPhase::Late
        }
    }

    /// Returns a human-readable name for the phase.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            TrainingProgressPhase::Warmup => "warmup",
            TrainingProgressPhase::Early => "early",
            TrainingProgressPhase::Mid => "mid",
            TrainingProgressPhase::Late => "late",
        }
    }
}

/// Gradient range tuner with phase-aware thresholds and per-layer AGC.
///
/// Maintains gradient statistics across layers and training phases,
/// computing adaptive clipping factors to keep gradients within
/// healthy ranges while minimizing unnecessary clipping.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientRangeTuner {
    /// Per-layer gradient statistics indexed by layer name.
    layer_stats: HashMap<String, LayerGradientStats>,

    /// EMA of global gradient norm (across all parameters).
    global_grad_norm_ema: f32,

    /// Ring buffer of recent gradient norms (capacity 100).
    /// Used for computing stability metrics and entropy.
    grad_norm_history: Vec<f32>,

    /// Maximum capacity of the gradient norm history buffer.
    history_capacity: usize,

    /// AGC lambda parameter (default 0.04).
    agc_lambda: f32,
}

impl Default for GradientRangeTuner {
    fn default() -> Self {
        Self::new()
    }
}

impl GradientRangeTuner {
    /// Creates a new gradient range tuner with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self {
            layer_stats: HashMap::new(),
            global_grad_norm_ema: 0.0,
            grad_norm_history: Vec::with_capacity(100),
            history_capacity: 100,
            agc_lambda: 0.04,
        }
    }

    /// Creates a new tuner with a custom AGC lambda parameter.
    #[must_use]
    pub fn with_agc_lambda(agc_lambda: f32) -> Self {
        Self {
            layer_stats: HashMap::new(),
            global_grad_norm_ema: 0.0,
            grad_norm_history: Vec::with_capacity(100),
            history_capacity: 100,
            agc_lambda,
        }
    }

    /// Updates gradient statistics and computes per-layer clipping factors.
    ///
    /// # Arguments
    ///
    /// * `global_grad_norm` - L2 norm of all gradients combined
    /// * `layer_grads` - `HashMap` mapping layer names to (`grad_norm`, `weight_norm`) tuples
    /// * `progress_pct` - Training progress as percentage (0.0 to 100.0)
    ///
    /// # Returns
    ///
    /// `HashMap` mapping layer names to clipping factors (< 1.0 means clip, = 1.0 means no clip)
    pub fn update(
        &mut self,
        global_grad_norm: f32,
        layer_grads: &HashMap<String, (f32, f32)>,
        _progress_pct: f32,
    ) -> HashMap<String, f32> {
        const EMA_DECAY: f32 = 0.99;

        // Update global EMA
        if self.grad_norm_history.is_empty() {
            self.global_grad_norm_ema = global_grad_norm;
        } else {
            self.global_grad_norm_ema =
                EMA_DECAY * self.global_grad_norm_ema + (1.0 - EMA_DECAY) * global_grad_norm;
        }

        // Add to history (ring buffer)
        if self.grad_norm_history.len() >= self.history_capacity {
            self.grad_norm_history.remove(0);
        }
        self.grad_norm_history.push(global_grad_norm);

        // Update per-layer statistics
        for (layer_name, (grad_norm, weight_norm)) in layer_grads {
            self.layer_stats
                .entry(layer_name.clone())
                .or_insert_with(|| LayerGradientStats::new(layer_name.clone()))
                .update(*grad_norm, *weight_norm);
        }

        // Compute clipping factors for each layer
        let mut clip_factors = HashMap::new();
        for (layer_name, stats) in &mut self.layer_stats {
            let clip_factor = stats.compute_agc_clip_factor(self.agc_lambda);
            clip_factors.insert(layer_name.clone(), clip_factor);
            stats.record_clip(clip_factor < 1.0);
            stats.update_clip_ratio(self.grad_norm_history.len() as u64);
        }

        clip_factors
    }

    /// Returns the current gradient thresholds for the given progress level.
    ///
    /// # Arguments
    ///
    /// * `progress_pct` - Training progress as percentage
    ///
    /// # Returns
    ///
    /// [`PhaseGradientThresholds`] for the current phase
    #[must_use]
    pub fn current_thresholds(&self, progress_pct: f32) -> PhaseGradientThresholds {
        let phase = TrainingProgressPhase::from_progress(progress_pct);
        match phase {
            TrainingProgressPhase::Warmup => PhaseGradientThresholds::new(0.01, 0.5),
            TrainingProgressPhase::Early => PhaseGradientThresholds::new(0.1, 1.0),
            TrainingProgressPhase::Mid => PhaseGradientThresholds::new(0.3, 0.8),
            TrainingProgressPhase::Late => PhaseGradientThresholds::new(0.1, 0.5),
        }
    }

    /// Checks if the current gradient norm is within acceptable range.
    ///
    /// # Arguments
    ///
    /// * `progress_pct` - Training progress as percentage
    ///
    /// # Returns
    ///
    /// `true` if gradient norm is within phase-specific thresholds, `false` otherwise
    #[must_use]
    pub fn gradient_in_range(&self, progress_pct: f32) -> bool {
        let thresholds = self.current_thresholds(progress_pct);
        thresholds.contains(self.global_grad_norm_ema)
    }

    /// Computes gradient stability as 1 minus the coefficient of variation.
    ///
    /// Coefficient of variation (CV) = `std_dev` / mean.
    /// Stability ranges from 0.0 (highly variable) to 1.0 (very stable).
    ///
    /// # Returns
    ///
    /// Stability score between 0.0 and 1.0
    #[must_use]
    pub fn gradient_stability(&self) -> f32 {
        if self.grad_norm_history.len() < 2 {
            return 1.0; // Not enough history for stability computation
        }

        let mean: f32 =
            self.grad_norm_history.iter().sum::<f32>() / self.grad_norm_history.len() as f32;
        if mean < 1e-6 {
            return 0.0; // Avoid division by zero
        }

        let variance: f32 = self
            .grad_norm_history
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>()
            / self.grad_norm_history.len() as f32;

        let std_dev = variance.sqrt();
        let cv = std_dev / mean;

        // Stability is 1 - CV, clamped to [0, 1]
        (1.0 - cv).clamp(0.0, 1.0)
    }

    /// Computes Shannon entropy of the layer gradient distribution.
    ///
    /// Entropy measures how "spread out" gradients are across layers.
    /// High entropy = gradients well-distributed.
    /// Low entropy = gradients concentrated in few layers.
    ///
    /// Useful for detecting layer-wise imbalances.
    ///
    /// # Returns
    ///
    /// Shannon entropy (non-negative). Higher values indicate more balanced distribution.
    #[must_use]
    pub fn gradient_entropy(&self) -> f32 {
        if self.layer_stats.is_empty() {
            return 0.0;
        }

        // Compute total gradient magnitude across all layers
        let total_grad_norm: f32 = self
            .layer_stats
            .values()
            .map(|s| s.grad_norm_ema().max(0.0))
            .sum();

        if total_grad_norm < 1e-6 {
            return 0.0;
        }

        // Compute probabilities and entropy
        let mut entropy = 0.0;
        for stats in self.layer_stats.values() {
            let grad_norm = stats.grad_norm_ema().max(0.0);
            let p = grad_norm / total_grad_norm;

            if p > 1e-6 {
                entropy -= p * p.log2();
            }
        }

        entropy
    }

    /// Returns reference to per-layer statistics.
    #[must_use]
    pub fn layer_stats(&self) -> &HashMap<String, LayerGradientStats> {
        &self.layer_stats
    }

    /// Returns mutable reference to per-layer statistics.
    pub fn layer_stats_mut(&mut self) -> &mut HashMap<String, LayerGradientStats> {
        &mut self.layer_stats
    }

    /// Returns the current global gradient norm EMA.
    #[must_use]
    pub fn global_grad_norm_ema(&self) -> f32 {
        self.global_grad_norm_ema
    }

    /// Returns a copy of the gradient norm history.
    #[must_use]
    pub fn grad_norm_history(&self) -> Vec<f32> {
        self.grad_norm_history.clone()
    }

    /// Returns the AGC lambda parameter.
    #[must_use]
    pub fn agc_lambda(&self) -> f32 {
        self.agc_lambda
    }

    /// Sets the AGC lambda parameter.
    pub fn set_agc_lambda(&mut self, lambda: f32) {
        self.agc_lambda = lambda;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phase_gradient_thresholds() {
        let thresholds = PhaseGradientThresholds::new(0.1, 1.0);

        assert!(thresholds.contains(0.5));
        assert!(!thresholds.contains(0.05));
        assert!(!thresholds.contains(1.5));

        assert_eq!(thresholds.deviation(0.5), 0.0);
        assert_eq!(thresholds.deviation(0.05), 0.05 - 0.1);
        assert_eq!(thresholds.deviation(1.5), 1.5 - 1.0);
    }

    #[test]
    fn test_training_progress_phase() {
        assert_eq!(
            TrainingProgressPhase::from_progress(10.0),
            TrainingProgressPhase::Warmup
        );
        assert_eq!(
            TrainingProgressPhase::from_progress(30.0),
            TrainingProgressPhase::Early
        );
        assert_eq!(
            TrainingProgressPhase::from_progress(60.0),
            TrainingProgressPhase::Mid
        );
        assert_eq!(
            TrainingProgressPhase::from_progress(90.0),
            TrainingProgressPhase::Late
        );
        assert_eq!(
            TrainingProgressPhase::from_progress(150.0),
            TrainingProgressPhase::Late
        );
    }

    #[test]
    fn test_layer_gradient_stats() {
        let mut stats = LayerGradientStats::new("layer_0".to_string());

        stats.update(1.0, 2.0);
        assert_eq!(stats.grad_norm_ema(), 1.0);
        assert_eq!(stats.weight_norm_ema(), 2.0);

        // Second update with EMA decay
        stats.update(1.2, 2.1);
        let expected_grad_ema = 0.99 * 1.0 + 0.01 * 1.2;
        assert!((stats.grad_norm_ema() - expected_grad_ema).abs() < 1e-5);
    }

    #[test]
    fn test_agc_clip_factor() {
        let mut stats = LayerGradientStats::new("layer_0".to_string());
        stats.update(2.0, 0.5);

        // With lambda=0.04, weight_norm=0.5, grad_norm=2.0:
        // clipping_coeff = 0.04 * 0.5 / 2.0 = 0.01
        let clip_factor = stats.compute_agc_clip_factor(0.04);
        assert!((clip_factor - 0.01).abs() < 1e-5);
        assert!(clip_factor <= 1.0);
    }

    #[test]
    fn test_gradient_range_tuner_basic() {
        let mut tuner = GradientRangeTuner::new();

        // Simulate first update
        let mut layer_grads = HashMap::new();
        layer_grads.insert("layer_0".to_string(), (1.0, 2.0));
        layer_grads.insert("layer_1".to_string(), (0.8, 1.5));

        let clip_factors = tuner.update(1.2, &layer_grads, 0.0);

        assert_eq!(clip_factors.len(), 2);
        assert!(clip_factors.contains_key("layer_0"));
        assert!(clip_factors.contains_key("layer_1"));
    }

    #[test]
    fn test_gradient_stability() {
        let mut tuner = GradientRangeTuner::new();

        // Add constant values - should have high stability
        for _ in 0..10 {
            let mut layer_grads = HashMap::new();
            layer_grads.insert("layer_0".to_string(), (1.0, 2.0));
            tuner.update(1.0, &layer_grads, 0.0);
        }

        let stability = tuner.gradient_stability();
        assert!(stability > 0.9); // High stability for constant values
    }

    #[test]
    fn test_gradient_entropy() {
        let mut tuner = GradientRangeTuner::new();

        let mut layer_grads = HashMap::new();
        // Balanced gradients
        layer_grads.insert("layer_0".to_string(), (1.0, 1.0));
        layer_grads.insert("layer_1".to_string(), (1.0, 1.0));
        layer_grads.insert("layer_2".to_string(), (1.0, 1.0));
        layer_grads.insert("layer_3".to_string(), (1.0, 1.0));

        tuner.update(4.0, &layer_grads, 0.0);
        let entropy = tuner.gradient_entropy();

        // With 4 equally likely events, max entropy is log2(4) = 2.0
        assert!(entropy > 1.9 && entropy <= 2.0);
    }

    #[test]
    fn test_phase_thresholds() {
        let tuner = GradientRangeTuner::new();

        let warmup_threshold = tuner.current_thresholds(10.0);
        assert_eq!(warmup_threshold.min, 0.01);
        assert_eq!(warmup_threshold.max, 0.5);

        let early_threshold = tuner.current_thresholds(30.0);
        assert_eq!(early_threshold.min, 0.1);
        assert_eq!(early_threshold.max, 1.0);

        let mid_threshold = tuner.current_thresholds(60.0);
        assert_eq!(mid_threshold.min, 0.3);
        assert_eq!(mid_threshold.max, 0.8);

        let late_threshold = tuner.current_thresholds(90.0);
        assert_eq!(late_threshold.min, 0.1);
        assert_eq!(late_threshold.max, 0.5);
    }

    #[test]
    fn test_gradient_in_range() {
        let mut tuner = GradientRangeTuner::new();

        // Add gradient within warmup range
        let mut layer_grads = HashMap::new();
        layer_grads.insert("layer_0".to_string(), (0.1, 1.0));
        tuner.update(0.1, &layer_grads, 10.0); // Warmup phase

        assert!(tuner.gradient_in_range(10.0));
        // 0.1 is below mid phase minimum (0.3), so should be out of range
        assert!(!tuner.gradient_in_range(60.0)); // Out of range for mid phase
    }

    #[test]
    fn test_ring_buffer_capacity() {
        let mut tuner = GradientRangeTuner::new();

        // Add more than capacity entries
        let mut layer_grads = HashMap::new();
        layer_grads.insert("layer_0".to_string(), (1.0, 1.0));

        for i in 0..150 {
            tuner.update(1.0 + (i as f32 * 0.01), &layer_grads, 0.0);
        }

        // Should not exceed capacity
        assert_eq!(tuner.grad_norm_history().len(), 100);
    }

    #[test]
    fn test_layer_clip_ratio_tracking() {
        let mut tuner = GradientRangeTuner::new();

        let mut layer_grads = HashMap::new();
        layer_grads.insert("layer_0".to_string(), (5.0, 0.1)); // Will clip due to AGC

        for _ in 0..10 {
            tuner.update(5.0, &layer_grads, 0.0);
        }

        let layer_0_stats = &tuner.layer_stats()["layer_0"];
        assert!(layer_0_stats.clip_ratio > 0.0);
        assert!(layer_0_stats.clip_count > 0);
    }

    #[test]
    fn test_serialization() {
        let mut tuner = GradientRangeTuner::new();
        let mut layer_grads = HashMap::new();
        layer_grads.insert("layer_0".to_string(), (1.0, 2.0));
        tuner.update(1.2, &layer_grads, 0.0);

        // Should be serializable
        let json = serde_json::to_string(&tuner).expect("serialization failed");
        let _deserialized: GradientRangeTuner =
            serde_json::from_str(&json).expect("deserialization failed");
    }
}
