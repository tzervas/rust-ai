//! Residual correction for prediction adjustment.
//!
//! The corrector applies accumulated residuals to predicted weight updates
//! to compensate for systematic prediction errors. This maintains training
//! quality without requiring full gradient computation.
//!
//! # Why Residual Correction?
//!
//! Predictions inevitably accumulate error over multiple steps. Rather than
//! abandoning prediction when errors grow, residual correction:
//! - **Extends prediction horizons**: Correct small errors instead of recomputing
//! - **Learns bias patterns**: Systematic under/over-prediction is correctable
//! - **Preserves speedup**: Correction is O(1) vs `O(backward_pass)` for recomputation
//!
//! # Correction Strategy
//!
//! 1. **Find similar states**: Locate past residuals from similar training conditions
//! 2. **Weight by relevance**: More recent and more similar residuals get higher weight
//! 3. **Apply correction**: Adjust predicted weight delta based on residual patterns
//!
//! # Online Learning
//!
//! The corrector learns online from observed residuals, building a model of
//! prediction error as a function of training state. This enables adaptive
//! correction that improves over training.

use crate::config::HybridTrainerConfig;
use crate::phases::PhaseOutcome;
use crate::residuals::{Residual, ResidualStore};
use crate::state::{TrainingState, WeightDelta};
use crate::Phase;

/// Configuration for the residual corrector.
#[derive(Debug, Clone)]
pub struct CorrectorConfig {
    /// Number of similar residuals to consider for correction.
    pub num_similar_residuals: usize,

    /// Maximum correction magnitude (as fraction of prediction).
    pub max_correction_factor: f32,

    /// Decay factor for older residuals.
    pub temporal_decay: f32,

    /// Learning rate for online correction model.
    pub learning_rate: f32,

    /// Whether to use weighted averaging (vs. uniform).
    pub use_weighted_average: bool,
}

impl Default for CorrectorConfig {
    fn default() -> Self {
        Self {
            num_similar_residuals: 10,
            max_correction_factor: 0.2,
            temporal_decay: 0.95,
            learning_rate: 0.01,
            use_weighted_average: true,
        }
    }
}

/// Statistics collected during correction phase.
#[derive(Debug, Clone, Default)]
pub struct CorrectionStatistics {
    /// Number of corrections applied.
    pub corrections_applied: usize,

    /// Total correction magnitude applied.
    pub total_correction_magnitude: f64,

    /// Mean correction factor.
    pub mean_correction_factor: f64,

    /// Maximum correction factor applied.
    pub max_correction_applied: f32,

    /// Number of residuals used for corrections.
    pub residuals_used: usize,

    /// Improvement in prediction after correction.
    pub prediction_improvement: f64,
}

/// A correction to be applied to predictions.
#[derive(Debug, Clone)]
pub struct Correction {
    /// Loss correction (added to predicted loss).
    pub loss_correction: f32,

    /// Weight delta correction.
    pub weight_correction: Option<WeightDelta>,

    /// Confidence in this correction.
    pub confidence: f32,

    /// Number of residuals used to compute this correction.
    pub num_residuals_used: usize,

    /// Weights assigned to each residual.
    pub residual_weights: Vec<f32>,
}

impl Default for Correction {
    fn default() -> Self {
        Self::zero()
    }
}

impl Correction {
    /// Creates a zero correction (no adjustment).
    #[must_use]
    pub fn zero() -> Self {
        Self {
            loss_correction: 0.0,
            weight_correction: None,
            confidence: 1.0,
            num_residuals_used: 0,
            residual_weights: Vec::new(),
        }
    }

    /// Returns whether this correction is significant.
    #[must_use]
    pub fn is_significant(&self, threshold: f32) -> bool {
        self.loss_correction.abs() > threshold
    }
}

/// Residual corrector that learns from past prediction errors.
pub struct ResidualCorrector {
    /// Configuration.
    config: CorrectorConfig,

    /// Statistics about corrections.
    statistics: CorrectionStatistics,

    /// Linear model coefficients for correction prediction.
    linear_model: Vec<f32>,

    /// Feature dimension for the linear model.
    _feature_dim: usize,

    /// Running estimate of loss correction bias.
    loss_bias: f32,

    /// Exponential moving average of residual magnitudes.
    residual_ema: f32,

    /// EMA decay factor.
    ema_decay: f32,
}

impl ResidualCorrector {
    /// Creates a new residual corrector.
    #[must_use]
    pub fn new(_config: &HybridTrainerConfig) -> Self {
        let feature_dim = 32; // From TrainingState::compute_features
        Self {
            config: CorrectorConfig::default(),
            statistics: CorrectionStatistics::default(),
            linear_model: vec![0.0; feature_dim],
            _feature_dim: feature_dim,
            loss_bias: 0.0,
            residual_ema: 0.0,
            ema_decay: 0.9,
        }
    }

    /// Creates a corrector with custom configuration.
    #[must_use]
    pub fn with_config(config: CorrectorConfig) -> Self {
        let feature_dim = 32;
        Self {
            config,
            statistics: CorrectionStatistics::default(),
            linear_model: vec![0.0; feature_dim],
            _feature_dim: feature_dim,
            loss_bias: 0.0,
            residual_ema: 0.0,
            ema_decay: 0.9,
        }
    }

    /// Computes a correction based on similar past residuals.
    ///
    /// # Arguments
    ///
    /// * `state` - Current training state
    /// * `residual_store` - Store of past residuals
    /// * `predicted_loss` - The predicted loss to correct
    ///
    /// # Returns
    ///
    /// A correction to apply to the prediction.
    #[must_use]
    pub fn compute_correction(
        &self,
        state: &TrainingState,
        residual_store: &ResidualStore,
        predicted_loss: f32,
    ) -> Correction {
        if residual_store.is_empty() {
            return Correction::zero();
        }

        // Find similar residuals
        let similar = residual_store.find_similar(state, self.config.num_similar_residuals);

        if similar.is_empty() {
            return Correction::zero();
        }

        // Compute weights based on similarity and recency
        let current_step = state.step;
        let mut weights = Vec::with_capacity(similar.len());
        let mut weight_sum = 0.0f32;

        for residual in &similar {
            // Temporal decay based on step difference
            let step_diff = current_step.saturating_sub(residual.step) as f32;
            let temporal_weight = self.config.temporal_decay.powf(step_diff / 100.0);

            // Confidence weight
            let confidence_weight = residual.prediction_confidence;

            let weight = temporal_weight * confidence_weight;
            weights.push(weight);
            weight_sum += weight;
        }

        // Normalize weights
        if weight_sum > 0.0 {
            for w in &mut weights {
                *w /= weight_sum;
            }
        }

        // Compute weighted average of loss residuals
        let loss_correction: f32 = similar
            .iter()
            .zip(weights.iter())
            .map(|(r, &w)| r.loss_residual * w)
            .sum();

        // Also use linear model prediction
        let features = state.compute_features();
        let model_correction: f32 = features
            .iter()
            .zip(self.linear_model.iter())
            .map(|(&f, &w)| f * w)
            .sum::<f32>()
            + self.loss_bias;

        // Blend historical and model-based correction
        let blended_correction = 0.7 * loss_correction + 0.3 * model_correction;

        // Clamp to maximum correction
        let max_correction = predicted_loss.abs() * self.config.max_correction_factor;
        let final_correction = blended_correction.clamp(-max_correction, max_correction);

        Correction {
            loss_correction: final_correction,
            weight_correction: None, // Would include weight delta corrections
            confidence: if similar.len() >= 5 { 0.9 } else { 0.5 },
            num_residuals_used: similar.len(),
            residual_weights: weights,
        }
    }

    /// Updates the corrector from a new residual observation.
    ///
    /// # Arguments
    ///
    /// * `residual` - The observed residual
    /// * `state` - Training state at time of prediction
    pub fn update_from_residual(&mut self, residual: &Residual, state: &TrainingState) {
        // Update EMA of residual magnitudes
        let magnitude = residual.loss_residual.abs();
        self.residual_ema = self.ema_decay * self.residual_ema + (1.0 - self.ema_decay) * magnitude;

        // Update linear model using online gradient descent
        let features = state.compute_features();
        let prediction: f32 = features
            .iter()
            .zip(self.linear_model.iter())
            .map(|(&f, &w)| f * w)
            .sum::<f32>()
            + self.loss_bias;

        let error = residual.loss_residual - prediction;

        // Update weights
        for (i, &f) in features.iter().enumerate() {
            if i < self.linear_model.len() {
                self.linear_model[i] += self.config.learning_rate * error * f;
            }
        }

        // Update bias
        self.loss_bias += self.config.learning_rate * error;

        // Update statistics
        self.statistics.corrections_applied += 1;
        self.statistics.residuals_used += 1;
    }

    /// Returns correction statistics.
    #[must_use]
    pub fn statistics(&self) -> &CorrectionStatistics {
        &self.statistics
    }

    /// Computes a simple weight delta correction based on current state.
    ///
    /// This is a simplified version that uses only the linear model,
    /// without requiring access to the full residual store.
    ///
    /// # Arguments
    ///
    /// * `state` - Current training state
    ///
    /// # Returns
    ///
    /// An optional weight delta to apply, or None if no correction needed.
    #[must_use]
    pub fn compute_simple_correction(&self, state: &TrainingState) -> Option<WeightDelta> {
        // Only apply corrections if we have enough history
        if self.statistics.corrections_applied < 10 {
            return None;
        }

        // Use linear model to estimate correction magnitude
        let features = state.compute_features();
        let correction_magnitude: f32 = features
            .iter()
            .zip(self.linear_model.iter())
            .map(|(&f, &w)| f * w)
            .sum::<f32>()
            + self.loss_bias;

        // Only apply if correction is significant
        if correction_magnitude.abs() < 0.01 {
            return None;
        }

        // Create a small weight delta in the gradient direction
        // This is a placeholder - real implementation would store gradient directions
        Some(WeightDelta::scaled_identity(correction_magnitude * 0.01))
    }

    /// Returns the current residual magnitude EMA.
    #[must_use]
    pub fn residual_ema(&self) -> f32 {
        self.residual_ema
    }

    /// Resets the corrector state.
    pub fn reset(&mut self) {
        self.linear_model.fill(0.0);
        self.loss_bias = 0.0;
        self.residual_ema = 0.0;
        self.statistics = CorrectionStatistics::default();
    }
}

/// Executor for the correction phase.
pub struct CorrectionExecutor {
    /// Number of validation samples to use.
    validation_samples: usize,

    /// Current step within correction.
    current_step: usize,

    /// Maximum correction magnitude.
    _max_correction_magnitude: f32,

    /// Collected statistics.
    statistics: CorrectionStatistics,

    /// Start time for duration tracking.
    start_time: Option<std::time::Instant>,
}

impl CorrectionExecutor {
    /// Creates a new correction executor.
    #[must_use]
    pub fn new(validation_samples: usize, max_correction_magnitude: f32) -> Self {
        Self {
            validation_samples,
            current_step: 0,
            _max_correction_magnitude: max_correction_magnitude,
            statistics: CorrectionStatistics::default(),
            start_time: None,
        }
    }

    /// Returns whether the correction phase is complete.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.current_step >= self.validation_samples
    }

    /// Records a correction step.
    pub fn record_step(&mut self, correction: &Correction, actual_improvement: f32) {
        if self.start_time.is_none() {
            self.start_time = Some(std::time::Instant::now());
        }

        self.statistics.corrections_applied += 1;
        self.statistics.total_correction_magnitude += f64::from(correction.loss_correction.abs());
        self.statistics.residuals_used += correction.num_residuals_used;
        self.statistics.prediction_improvement += f64::from(actual_improvement);

        if correction.loss_correction.abs() > self.statistics.max_correction_applied {
            self.statistics.max_correction_applied = correction.loss_correction.abs();
        }

        self.current_step += 1;
    }

    /// Finalizes the correction phase.
    #[must_use]
    pub fn finalize(self) -> PhaseOutcome {
        let duration_ms = self
            .start_time
            .map_or(0.0, |t| t.elapsed().as_secs_f64() * 1000.0);

        PhaseOutcome {
            phase: Phase::Correct,
            steps_executed: self.current_step,
            average_loss: 0.0, // Not applicable for correction
            final_loss: 0.0,
            completed_normally: self.is_complete(),
            early_termination_reason: None,
            prediction_error: None,
            duration_ms,
        }
    }
}

/// Trait for correction strategies.
pub trait CorrectionStrategy: Send + Sync {
    /// Computes a correction for the given state and residuals.
    fn compute(
        &self,
        state: &TrainingState,
        residuals: &[&Residual],
        predicted_loss: f32,
    ) -> Correction;

    /// Updates the strategy from observed outcomes.
    fn update(&mut self, correction: &Correction, actual_improvement: f32);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::residuals::ResidualStore;

    #[test]
    fn test_correction_zero() {
        let correction = Correction::zero();
        assert!((correction.loss_correction).abs() < f32::EPSILON);
        assert!(correction.confidence > 0.99);
    }

    #[test]
    fn test_corrector_initialization() {
        let config = HybridTrainerConfig::default();
        let corrector = ResidualCorrector::new(&config);

        assert_eq!(corrector.linear_model.len(), 32);
        assert!((corrector.loss_bias).abs() < f32::EPSILON);
    }

    #[test]
    fn test_empty_residual_store_correction() {
        let config = HybridTrainerConfig::default();
        let corrector = ResidualCorrector::new(&config);
        let store = ResidualStore::new(100);
        let state = TrainingState::new();

        let correction = corrector.compute_correction(&state, &store, 2.5);

        // Should return zero correction with empty store
        assert!((correction.loss_correction).abs() < f32::EPSILON);
    }

    #[test]
    fn test_correction_executor_completion() {
        let mut executor = CorrectionExecutor::new(5, 0.1);

        assert!(!executor.is_complete());

        for _ in 0..5 {
            let correction = Correction::zero();
            executor.record_step(&correction, 0.0);
        }

        assert!(executor.is_complete());
    }
}
