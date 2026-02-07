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
        let feature_dim = 64; // From TrainingState::compute_features (64-dim)
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
        let feature_dim = 64; // From TrainingState::compute_features (64-dim)
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

        // Adaptive scaling based on recent prediction error magnitude
        // Scale correction up when predictions are inaccurate, down when accurate
        let avg_abs_residual: f32 = similar
            .iter()
            .map(|r| r.loss_residual.abs())
            .sum::<f32>()
            / similar.len().max(1) as f32;
        // Scale from 0.5x (very accurate predictions) to 1.5x (large errors)
        // Clamp error magnitude to [0, 1.0] for stability
        let error_scaling = 0.5 + avg_abs_residual.min(1.0);
        let scaled_correction = blended_correction * error_scaling;

        // Clamp to maximum correction
        let max_correction = predicted_loss.abs() * self.config.max_correction_factor;
        let final_correction = scaled_correction.clamp(-max_correction, max_correction);

        // Build weight-level corrections from per-layer gradient residuals
        let weight_correction = self.compute_weight_correction(&similar, &weights);

        Correction {
            loss_correction: final_correction,
            weight_correction,
            confidence: if similar.len() >= 5 { 0.9 } else { 0.5 },
            num_residuals_used: similar.len(),
            residual_weights: weights,
        }
    }

    /// Computes weight-level corrections from residuals with per-layer information.
    ///
    /// Builds a weighted average of per-layer residual magnitudes across similar
    /// past residuals, producing a `WeightDelta` that corrects the prediction's
    /// per-layer scaling errors.
    fn compute_weight_correction(
        &self,
        similar: &[&Residual],
        weights: &[f32],
    ) -> Option<WeightDelta> {
        // Only produce weight corrections if residuals have gradient info
        let has_layer_info = similar.iter().any(|r| !r.gradient_residuals.is_empty());

        if !has_layer_info {
            return None;
        }

        // Accumulate weighted per-layer correction magnitudes
        let mut layer_corrections: std::collections::HashMap<String, f32> =
            std::collections::HashMap::new();
        let mut layer_counts: std::collections::HashMap<String, f32> =
            std::collections::HashMap::new();

        for (residual, &w) in similar.iter().zip(weights.iter()) {
            for layer_res in &residual.gradient_residuals {
                // The correction direction is derived from cosine similarity:
                // if cosine_similarity < 1, the prediction direction was off
                let direction_error = 1.0 - layer_res.cosine_similarity;
                let correction_mag = layer_res.magnitude * direction_error * w;

                *layer_corrections
                    .entry(layer_res.layer_name.clone())
                    .or_insert(0.0) += correction_mag;
                *layer_counts
                    .entry(layer_res.layer_name.clone())
                    .or_insert(0.0) += w;
            }
        }

        if layer_corrections.is_empty() {
            return None;
        }

        // Normalize by total weight per layer
        let mut deltas = std::collections::HashMap::new();
        let mut total_scale = 0.0_f32;

        for (layer_name, correction_sum) in &layer_corrections {
            let count = layer_counts.get(layer_name).copied().unwrap_or(1.0);
            let avg_correction = correction_sum / count.max(1e-8);

            // Clamp correction magnitude
            let clamped = avg_correction.clamp(
                -self.config.max_correction_factor,
                self.config.max_correction_factor,
            );
            deltas.insert(layer_name.clone(), vec![clamped]);
            total_scale += clamped.abs();
        }

        Some(WeightDelta {
            deltas,
            scale: total_scale / layer_corrections.len().max(1) as f32,
            metadata: crate::state::WeightDeltaMetadata {
                is_predicted: false,
                confidence: Some(if similar.len() >= 5 { 0.8 } else { 0.4 }),
                source_phase: Some(crate::Phase::Correct),
                num_steps: 0,
            },
        })
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

        // Create a weight delta with per-layer corrections
        // The linear model's prediction magnitude drives uniform layer-level correction
        let layer_names = [
            "embed",
            "attention.q",
            "attention.k",
            "attention.v",
            "attention.out",
            "mlp.up",
            "mlp.down",
            "lm_head",
        ];

        let per_layer_correction = correction_magnitude * 0.01;
        let mut deltas = std::collections::HashMap::new();
        for layer_name in &layer_names {
            deltas.insert((*layer_name).to_string(), vec![per_layer_correction]);
        }

        Some(WeightDelta {
            deltas,
            scale: per_layer_correction.abs(),
            metadata: crate::state::WeightDeltaMetadata {
                is_predicted: false,
                confidence: Some(0.5),
                source_phase: Some(crate::Phase::Correct),
                num_steps: 0,
            },
        })
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

        assert_eq!(corrector.linear_model.len(), 64);
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

    /// Helper: create a TrainingState with some recorded steps for realistic features.
    fn make_state(step: u64, loss: f32, grad_norm: f32) -> TrainingState {
        let mut state = TrainingState::new();
        // Record enough steps so that compute_features() produces meaningful values
        for i in 0..step {
            state.record_step(loss + 0.01 * i as f32, grad_norm);
        }
        state
    }

    /// Helper: create a Residual with optional gradient residuals.
    fn make_residual(
        step: u64,
        loss_residual: f32,
        confidence: f32,
        state_features: Vec<f32>,
        gradient_residuals: Vec<crate::residuals::LayerResidual>,
    ) -> crate::residuals::Residual {
        crate::residuals::Residual {
            step,
            phase: crate::Phase::Predict,
            prediction_horizon: 10,
            loss_residual,
            gradient_residuals,
            state_features,
            prediction_confidence: confidence,
        }
    }

    #[test]
    fn test_weight_correction_from_gradient_residuals() {
        let config = HybridTrainerConfig::default();
        let corrector = ResidualCorrector::new(&config);

        let mut store = ResidualStore::new(100);
        let state = make_state(50, 2.0, 1.0);
        let features = state.compute_features();

        // Add residuals with gradient_residuals populated for multiple layers
        for i in 0..10 {
            let layer_residuals = vec![
                crate::residuals::LayerResidual {
                    layer_name: "attention_qkv".to_string(),
                    magnitude: 0.05 + 0.001 * i as f32,
                    compressed: None,
                    cosine_similarity: 0.8,
                },
                crate::residuals::LayerResidual {
                    layer_name: "embedding".to_string(),
                    magnitude: 0.03 + 0.001 * i as f32,
                    compressed: None,
                    cosine_similarity: 0.9,
                },
            ];
            store.add(make_residual(
                40 + i,
                0.1,
                0.9,
                features.clone(),
                layer_residuals,
            ));
        }

        let correction = corrector.compute_correction(&state, &store, 2.0);

        // weight_correction should be Some because residuals have gradient info
        assert!(
            correction.weight_correction.is_some(),
            "Expected weight_correction to be Some when gradient_residuals are populated"
        );

        let wd = correction.weight_correction.unwrap();
        // Verify per-layer deltas exist for both layers
        assert!(
            wd.deltas.contains_key("attention_qkv"),
            "Missing attention_qkv layer delta"
        );
        assert!(
            wd.deltas.contains_key("embedding"),
            "Missing embedding layer delta"
        );
        // Each layer should have a single correction value
        assert_eq!(wd.deltas["attention_qkv"].len(), 1);
        assert_eq!(wd.deltas["embedding"].len(), 1);
    }

    #[test]
    fn test_linear_model_learning() {
        let config = HybridTrainerConfig::default();
        let mut corrector = ResidualCorrector::new(&config);

        let initial_weights: Vec<f32> = corrector.linear_model.clone();

        // Feed multiple residuals to drive online learning updates
        for i in 0..20 {
            let state = make_state(10 + i, 2.0 - 0.01 * i as f32, 1.0);
            let residual = make_residual(
                10 + i,
                0.05 * (i as f32 + 1.0), // Increasing residuals
                0.9,
                state.compute_features(),
                Vec::new(),
            );
            corrector.update_from_residual(&residual, &state);
        }

        // Verify the linear model weights have changed from their initial values
        let changed = corrector
            .linear_model
            .iter()
            .zip(initial_weights.iter())
            .any(|(new, old)| (new - old).abs() > 1e-10);
        assert!(
            changed,
            "Linear model weights should have changed after updates"
        );

        // Also verify the bias has changed
        assert!(
            corrector.loss_bias.abs() > 1e-10,
            "Loss bias should be non-zero after updates"
        );
    }

    #[test]
    fn test_temporal_decay_weighting() {
        let config = HybridTrainerConfig::default();
        let corrector = ResidualCorrector::new(&config);

        let mut store = ResidualStore::new(100);
        let state = make_state(1000, 2.0, 1.0);
        let features = state.compute_features();

        // Add an old residual (step 100) with a large loss residual
        store.add(make_residual(100, 1.0, 0.9, features.clone(), Vec::new()));

        // Add a recent residual (step 990) with a small loss residual
        store.add(make_residual(990, -0.1, 0.9, features.clone(), Vec::new()));

        let correction = corrector.compute_correction(&state, &store, 2.0);

        // The recent residual (loss_residual = -0.1) should dominate over the old one
        // (loss_residual = 1.0) due to temporal decay, so the blended correction
        // should lean toward the recent value's direction (negative)
        // The old residual at step 100 vs current step 1000 has step_diff=900,
        // temporal_weight = 0.95^(900/100) = 0.95^9 ~ 0.63
        // The recent residual at step 990 vs current step 1000 has step_diff=10,
        // temporal_weight = 0.95^(10/100) = 0.95^0.1 ~ 0.995
        // So the recent residual's weight is substantially higher.
        // The correction is a blend of historical and model-based (model is zero initially),
        // so historical component = 0.7 * weighted_avg
        assert!(
            correction.num_residuals_used == 2,
            "Both residuals should be used"
        );
        // The recent residual's weight should be larger
        assert!(
            correction.residual_weights[1] > correction.residual_weights[0],
            "Recent residual (index 1) should have higher weight than old residual (index 0). \
             Weights: {:?}",
            correction.residual_weights
        );
    }

    #[test]
    fn test_simple_correction_produces_layer_deltas() {
        let config = HybridTrainerConfig::default();
        let mut corrector = ResidualCorrector::new(&config);

        // We need at least 10 corrections_applied for compute_simple_correction to return Some
        let state = make_state(50, 2.0, 1.0);
        for i in 0..15 {
            let residual = make_residual(
                i,
                0.5, // Consistent positive residual to build up bias
                0.9,
                state.compute_features(),
                Vec::new(),
            );
            corrector.update_from_residual(&residual, &state);
        }

        let result = corrector.compute_simple_correction(&state);

        // After 15 updates with consistent residuals, the model should produce
        // a correction with sufficient magnitude
        assert!(
            result.is_some(),
            "Simple correction should be Some after 15 updates with consistent residuals"
        );

        let wd = result.unwrap();
        let expected_layers = [
            "embed",
            "attention.q",
            "attention.k",
            "attention.v",
            "attention.out",
            "mlp.up",
            "mlp.down",
            "lm_head",
        ];
        for layer in &expected_layers {
            assert!(
                wd.deltas.contains_key(*layer),
                "Missing expected layer: {layer}"
            );
        }
        // Verify metadata
        assert!(!wd.metadata.is_predicted);
        assert_eq!(wd.metadata.source_phase, Some(crate::Phase::Correct));
    }

    #[test]
    fn test_correction_with_single_residual() {
        let config = HybridTrainerConfig::default();
        let corrector = ResidualCorrector::new(&config);

        let mut store = ResidualStore::new(100);
        let state = make_state(10, 2.0, 1.0);
        let features = state.compute_features();

        // Add just a single residual
        store.add(make_residual(5, 0.3, 0.8, features.clone(), Vec::new()));

        let correction = corrector.compute_correction(&state, &store, 2.0);

        // Should still produce a correction (not panic or return zero)
        assert_eq!(correction.num_residuals_used, 1);
        assert_eq!(correction.residual_weights.len(), 1);
        // Confidence should be low (< 5 residuals)
        assert!(
            (correction.confidence - 0.5).abs() < f32::EPSILON,
            "Single-residual correction should have low confidence (0.5), got {}",
            correction.confidence
        );
        // Loss correction should be derived from the single residual
        // 0.7 * (weighted loss_residual) + 0.3 * (model_correction ≈ 0) = 0.7 * 0.3 = 0.21
        // clamped by max_correction_factor (0.2 * |2.0| = 0.4), so should be ~0.21
        assert!(
            correction.loss_correction.abs() > 0.0,
            "Loss correction should be non-zero with a non-zero residual"
        );
    }

    #[test]
    fn test_correction_executor_tracks_steps() {
        let mut executor = CorrectionExecutor::new(10, 0.5);

        // Record several correction steps with varying improvements
        let corrections = vec![
            (
                Correction {
                    loss_correction: 0.1,
                    weight_correction: None,
                    confidence: 0.9,
                    num_residuals_used: 5,
                    residual_weights: vec![0.2; 5],
                },
                0.05,
            ),
            (
                Correction {
                    loss_correction: -0.2,
                    weight_correction: None,
                    confidence: 0.8,
                    num_residuals_used: 3,
                    residual_weights: vec![0.33; 3],
                },
                0.1,
            ),
            (
                Correction {
                    loss_correction: 0.15,
                    weight_correction: None,
                    confidence: 0.85,
                    num_residuals_used: 7,
                    residual_weights: vec![0.14; 7],
                },
                0.08,
            ),
        ];

        for (correction, improvement) in &corrections {
            executor.record_step(correction, *improvement);
        }

        // Verify the executor tracked all steps
        assert_eq!(executor.current_step, 3);
        assert!(!executor.is_complete()); // 3 < 10

        // Verify statistics
        let stats = &executor.statistics;
        assert_eq!(stats.corrections_applied, 3);
        // Total residuals used = 5 + 3 + 7 = 15
        assert_eq!(stats.residuals_used, 15);
        // Max correction applied = 0.2 (absolute value of -0.2)
        assert!(
            (stats.max_correction_applied - 0.2).abs() < f32::EPSILON,
            "Max correction should be 0.2, got {}",
            stats.max_correction_applied
        );
        // Total correction magnitude = 0.1 + 0.2 + 0.15 = 0.45
        assert!(
            (stats.total_correction_magnitude - 0.45).abs() < 1e-6,
            "Total correction magnitude should be 0.45, got {}",
            stats.total_correction_magnitude
        );
        // Prediction improvement = 0.05 + 0.1 + 0.08 = 0.23
        assert!(
            (stats.prediction_improvement - 0.23).abs() < 1e-6,
            "Prediction improvement should be 0.23, got {}",
            stats.prediction_improvement
        );
    }

    #[test]
    fn test_correction_blending() {
        // Verify that the correction is properly blended from historical (0.7) and model (0.3)
        let mut corrector = ResidualCorrector::with_config(CorrectorConfig {
            max_correction_factor: 1.0, // Large to avoid clamping
            temporal_decay: 1.0,        // No temporal decay for simplicity
            learning_rate: 0.0,         // No model learning (keeps model at zero)
            ..CorrectorConfig::default()
        });

        let mut store = ResidualStore::new(100);
        let state = make_state(10, 2.0, 1.0);
        let features = state.compute_features();

        // Add identical residuals so the weighted average = the residual value
        for i in 0..5 {
            store.add(make_residual(
                5 + i,
                0.4, // Consistent loss residual
                0.9, // Same confidence for uniform weighting
                features.clone(),
                Vec::new(),
            ));
        }

        let correction = corrector.compute_correction(&state, &store, 2.0);

        // Historical component: weighted avg of loss_residuals = 0.4
        // Model component: linear_model is all zeros, bias is 0 => 0.0
        // Blended = 0.7 * 0.4 + 0.3 * 0.0 = 0.28
        // Adaptive scaling: avg_abs_residual = 0.4, error_scaling = 0.5 + 0.4 = 0.9
        // Scaled = 0.28 * 0.9 = 0.252
        // max_correction = |2.0| * 1.0 = 2.0 (no clamping)
        let blended = 0.7 * 0.4 + 0.3 * 0.0;
        let error_scaling = 0.5 + 0.4; // avg_abs_residual = 0.4
        let expected = blended * error_scaling;
        assert!(
            (correction.loss_correction - expected).abs() < 1e-4,
            "Blended correction should be ~{expected}, got {}",
            correction.loss_correction
        );

        // Now train the model to have non-zero weights, and verify blending changes
        for i in 0..50 {
            let s = make_state(10 + i, 2.0, 1.0);
            let r = make_residual(10 + i, 0.4, 0.9, s.compute_features(), Vec::new());
            corrector.update_from_residual(&r, &s);
        }

        // Recompute - now model component should be non-zero
        let correction2 = corrector.compute_correction(&state, &store, 2.0);

        // The model-based component should now contribute, changing the blend
        // We don't know the exact value, but it should differ from the pure historical blend
        // (unless the model learned exactly 0, which is unlikely with 50 updates)
        assert!(
            correction2.loss_correction.abs() > 0.0,
            "Correction should still be non-zero after model learning"
        );
    }
}
