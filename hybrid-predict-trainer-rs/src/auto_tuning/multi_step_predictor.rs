//! Multi-step prediction with calibration for batch-level predictions.
//!
//! This module implements intelligent batch prediction that decides whether to
//! skip backward computation entirely, apply predictions with verification, or
//! run full training based on per-horizon confidence estimates and calibration.
//!
//! # How Batch Skipping Works
//!
//! Rather than predicting individual steps, the multi-step predictor evaluates
//! confidence across multiple horizons (1, 2, 4, 8... steps) and recommends:
//!
//! - **`SkipBatch`**: Very high confidence (≥0.85) with low uncertainty (≤0.3)
//! - **`ApplyWithVerification`**: Good confidence (≥0.7), worth trying but monitor closely
//! - **`RunFullBatch`**: Low confidence (<0.7), run all backward passes normally
//!
//! # Calibration
//!
//! Historical prediction errors for each horizon are tracked in a fixed-size history.
//! Per-horizon calibration factors are computed as:
//!
//! ```text
//! calibration_factor = 1.0 / (1.0 + mean_absolute_error)
//! ```
//!
//! This allows the predictor to account for systematic biases in predictions
//! across different prediction horizons.
//!
//! # Streak Tracking
//!
//! An "accurate streak" counter tracks consecutive steps where predictions were
//! correct, enabling adaptive confidence adjustment and early divergence detection.

use std::collections::HashMap;

/// Recommendation for batch processing strategy.
///
/// Indicates the suggested approach for processing a batch of training steps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchPredictionRecommendation {
    /// Skip the entire backward pass and apply predicted deltas.
    ///
    /// Used when confidence is very high (≥0.85) and uncertainty is very low (≤0.3).
    /// This provides maximum speedup with minimal risk.
    SkipBatch,

    /// Apply predictions but verify with forward pass monitoring.
    ///
    /// Used when confidence is good (≥0.7 but <0.85) or uncertainty is moderate.
    /// Allows speedup while catching divergence early.
    ApplyWithVerification,

    /// Run full backward passes on all steps.
    ///
    /// Used when confidence is low (<0.7). Ensures training stability at the cost
    /// of losing potential speedup.
    RunFullBatch,
}

/// Prediction result for a batch of multiple training steps.
///
/// Contains per-horizon confidence estimates, overall confidence, and a recommendation
/// for how to process the batch.
#[derive(Debug, Clone)]
pub struct BatchPrediction {
    /// The final predicted loss after applying all steps.
    pub final_loss: f32,

    /// Per-horizon confidence values (indexed by step horizon: 1, 2, 4, 8, ...).
    pub horizon_confidences: Vec<f32>,

    /// Overall confidence computed as geometric mean of horizon confidences.
    pub overall_confidence: f32,

    /// Number of steps covered by this prediction.
    pub num_steps: usize,

    /// Recommended action for processing this batch.
    pub recommendation: BatchPredictionRecommendation,
}

/// Multi-step batch predictor with calibration and streak tracking.
///
/// Maintains per-horizon error history and calibration factors to make intelligent
/// decisions about when to skip backward computation. Tracks accurate streaks to
/// detect when the predictor becomes less reliable.
#[derive(Debug, Clone)]
pub struct MultiStepPredictor {
    /// Per-horizon error history (key: horizon in steps, value: Vec of errors).
    ///
    /// Each horizon maintains a fixed-capacity history (default 100) of absolute
    /// prediction errors. This enables per-horizon calibration.
    horizon_errors: HashMap<usize, Vec<f32>>,

    /// Per-horizon calibration factors.
    ///
    /// Updated from `horizon_errors` using: factor = 1.0 / (1.0 + `mean_error`).
    /// Applied to adjust confidence estimates for each horizon.
    calibration_factors: HashMap<usize, f32>,

    /// Ring buffer of accuracy observations (recent 100).
    ///
    /// Each entry is 1.0 if the prediction was accurate, 0.0 otherwise.
    /// Used to track streaks and overall accuracy trends.
    accuracy_history: Vec<f32>,

    /// Current position in `accuracy_history` ring buffer.
    accuracy_head: usize,

    /// Consecutive accurate predictions (increments on correct, resets on error).
    accurate_streak: usize,

    /// Total number of batches skipped via `SkipBatch` recommendation.
    batches_skipped: u64,

    /// Total number of steps skipped via `SkipBatch` recommendation.
    steps_skipped: u64,

    /// Confidence threshold for `SkipBatch` recommendation.
    ///
    /// When `overall_confidence` >= this threshold AND uncertainty <= `error_threshold`,
    /// the predictor recommends `SkipBatch`.
    batch_skip_confidence_threshold: f32,

    /// Uncertainty/error threshold for `SkipBatch` recommendation.
    ///
    /// When overall uncertainty <= this threshold AND confidence >= `confidence_threshold`,
    /// the predictor recommends `SkipBatch`.
    batch_skip_error_threshold: f32,

    /// Whether batch skipping is enabled.
    ///
    /// When false, only `ApplyWithVerification` or `RunFullBatch` are recommended.
    enable_batch_skip: bool,

    /// Maximum capacity for per-horizon error history.
    history_capacity: usize,
}

impl MultiStepPredictor {
    /// Creates a new multi-step predictor with the given configuration.
    ///
    /// # Arguments
    ///
    /// * `confidence_threshold` - Minimum confidence for `SkipBatch` (default 0.85)
    /// * `error_threshold` - Maximum error for `SkipBatch` (default 0.3)
    /// * `enable_skip` - Whether batch skipping is enabled
    ///
    /// # Returns
    ///
    /// A new `MultiStepPredictor` configured with the given thresholds.
    pub fn new(confidence_threshold: f32, error_threshold: f32, enable_skip: bool) -> Self {
        Self {
            horizon_errors: HashMap::new(),
            calibration_factors: HashMap::new(),
            accuracy_history: vec![0.0; 100],
            accuracy_head: 0,
            accurate_streak: 0,
            batches_skipped: 0,
            steps_skipped: 0,
            batch_skip_confidence_threshold: confidence_threshold.clamp(0.0, 1.0),
            batch_skip_error_threshold: error_threshold.clamp(0.0, 1.0),
            enable_batch_skip: enable_skip,
            history_capacity: 100,
        }
    }

    /// Predicts the outcome of a batch of steps with adaptive confidence.
    ///
    /// Computes per-horizon confidence estimates using calibration factors,
    /// then combines them into an overall confidence via geometric mean.
    /// Issues a recommendation based on confidence and uncertainty thresholds.
    ///
    /// # Arguments
    ///
    /// * `base_confidence` - Base confidence from the dynamics model (0.0-1.0)
    /// * `base_uncertainty` - Base uncertainty estimate from the dynamics model
    /// * `y_steps` - Number of steps to predict ahead
    ///
    /// # Returns
    ///
    /// A `BatchPrediction` with per-horizon confidence, overall confidence,
    /// and a recommendation for batch processing.
    pub fn predict_batch(
        &self,
        base_confidence: f32,
        base_uncertainty: f32,
        y_steps: usize,
    ) -> BatchPrediction {
        // Validate inputs
        let base_confidence = base_confidence.clamp(0.0, 1.0);
        let base_uncertainty = base_uncertainty.clamp(0.0, 1.0);
        let y_steps = y_steps.max(1);

        // Compute per-horizon confidence estimates
        let mut horizon_confidences = Vec::new();
        let horizons = self.compute_horizons(y_steps);

        for &horizon in &horizons {
            let calibration = self
                .calibration_factors
                .get(&horizon)
                .copied()
                .unwrap_or(1.0);
            let confidence = base_confidence * calibration;
            let confidence = confidence.clamp(0.0, 1.0);
            horizon_confidences.push(confidence);
        }

        // Compute overall confidence as geometric mean of horizon confidences
        let overall_confidence = if horizon_confidences.is_empty() {
            base_confidence
        } else {
            let product: f32 = horizon_confidences.iter().product();
            product.powf(1.0 / horizon_confidences.len() as f32)
        };

        // Determine recommendation based on thresholds
        let recommendation = if !self.enable_batch_skip {
            // Batch skipping disabled - only offer verification or full run
            if overall_confidence >= 0.7 {
                BatchPredictionRecommendation::ApplyWithVerification
            } else {
                BatchPredictionRecommendation::RunFullBatch
            }
        } else if overall_confidence >= self.batch_skip_confidence_threshold
            && base_uncertainty <= self.batch_skip_error_threshold
        {
            BatchPredictionRecommendation::SkipBatch
        } else if overall_confidence >= 0.7 {
            BatchPredictionRecommendation::ApplyWithVerification
        } else {
            BatchPredictionRecommendation::RunFullBatch
        };

        // Compute predicted final loss (placeholder - would come from dynamics model)
        let final_loss = 0.0; // This would be computed by the dynamics model

        BatchPrediction {
            final_loss,
            horizon_confidences,
            overall_confidence,
            num_steps: y_steps,
            recommendation,
        }
    }

    /// Records an observation from a prediction (actual vs predicted loss).
    ///
    /// Updates the horizon error history and recalibrates per-horizon factors.
    /// Also updates the accuracy streak based on whether the prediction was accurate.
    ///
    /// # Arguments
    ///
    /// * `horizon` - The horizon (in steps) for this observation
    /// * `predicted_loss` - The predicted loss value
    /// * `actual_loss` - The actual observed loss value
    pub fn record_observation(&mut self, horizon: usize, predicted_loss: f32, actual_loss: f32) {
        let horizon = horizon.max(1);
        let error = (actual_loss - predicted_loss).abs();

        // Add to horizon errors
        self.horizon_errors.entry(horizon).or_default().push(error);

        // Trim history if it exceeds capacity
        if let Some(errors) = self.horizon_errors.get_mut(&horizon) {
            if errors.len() > self.history_capacity {
                errors.remove(0);
            }
        }

        // Update calibration factor for this horizon
        self.update_calibration_factor(horizon);

        // Update accuracy history
        let is_accurate = error < 0.1; // Threshold for "accurate" (10% relative error)
        self.record_accuracy(is_accurate);
    }

    /// Updates the calibration factor for a given horizon.
    ///
    /// Computes: `calibration_factor` = 1.0 / (1.0 + `mean_absolute_error`)
    fn update_calibration_factor(&mut self, horizon: usize) {
        if let Some(errors) = self.horizon_errors.get(&horizon) {
            if !errors.is_empty() {
                let mean_error: f32 = errors.iter().sum::<f32>() / errors.len() as f32;
                let calibration = 1.0 / (1.0 + mean_error);
                self.calibration_factors.insert(horizon, calibration);
            }
        }
    }

    /// Records accuracy observation in the ring buffer.
    ///
    /// Updates the accuracy history and maintains the `accurate_streak` counter.
    fn record_accuracy(&mut self, is_accurate: bool) {
        let accuracy_value = if is_accurate { 1.0 } else { 0.0 };
        self.accuracy_history[self.accuracy_head] = accuracy_value;
        self.accuracy_head = (self.accuracy_head + 1) % self.accuracy_history.len();

        // Update streak
        if is_accurate {
            self.accurate_streak += 1;
        } else {
            self.accurate_streak = 0;
        }
    }

    /// Returns the current accurate streak length.
    ///
    /// Indicates how many consecutive accurate predictions have been made.
    /// A high streak suggests the predictor is working well.
    #[must_use]
    pub fn accurate_streak(&self) -> usize {
        self.accurate_streak
    }

    /// Returns statistics about batch skipping.
    ///
    /// # Returns
    ///
    /// A tuple of (`batches_skipped`, `steps_skipped`) with aggregate counts.
    #[must_use]
    pub fn skip_stats(&self) -> (u64, u64) {
        (self.batches_skipped, self.steps_skipped)
    }

    /// Records a successful batch skip.
    ///
    /// Called when a `SkipBatch` recommendation was followed and training continued
    /// successfully. Updates the batch and step skip counters.
    ///
    /// # Arguments
    ///
    /// * `num_steps` - The number of steps that were skipped in this batch
    pub fn record_batch_skip(&mut self, num_steps: usize) {
        self.batches_skipped += 1;
        self.steps_skipped += num_steps as u64;
    }

    /// Computes the list of horizons to evaluate for the given number of steps.
    ///
    /// Returns horizons at powers of 2: [1, 2, 4, 8, ...] up to `y_steps`.
    fn compute_horizons(&self, y_steps: usize) -> Vec<usize> {
        let mut horizons = Vec::new();
        let mut horizon = 1;
        while horizon <= y_steps {
            horizons.push(horizon);
            horizon *= 2;
        }
        if horizons.is_empty() {
            horizons.push(1);
        }
        horizons
    }

    /// Returns the mean error for a specific horizon (for monitoring).
    ///
    /// # Arguments
    ///
    /// * `horizon` - The horizon to query
    ///
    /// # Returns
    ///
    /// The mean absolute error for that horizon, or 0.0 if no observations exist.
    #[must_use]
    pub fn mean_error_for_horizon(&self, horizon: usize) -> f32 {
        self.horizon_errors.get(&horizon).map_or(0.0, |errors| {
            if errors.is_empty() {
                0.0
            } else {
                errors.iter().sum::<f32>() / errors.len() as f32
            }
        })
    }

    /// Returns the calibration factor for a specific horizon.
    ///
    /// # Arguments
    ///
    /// * `horizon` - The horizon to query
    ///
    /// # Returns
    ///
    /// The calibration factor, or 1.0 if no factor has been computed.
    #[must_use]
    pub fn calibration_factor_for_horizon(&self, horizon: usize) -> f32 {
        self.calibration_factors
            .get(&horizon)
            .copied()
            .unwrap_or(1.0)
    }

    /// Resets all calibration and accuracy statistics.
    ///
    /// Useful when transitioning phases or recovering from divergence.
    pub fn reset(&mut self) {
        self.horizon_errors.clear();
        self.calibration_factors.clear();
        self.accuracy_history.fill(0.0);
        self.accuracy_head = 0;
        self.accurate_streak = 0;
        // Keep skip_stats as they're cumulative across the session
    }

    /// Returns overall prediction accuracy rate from recent history.
    ///
    /// # Returns
    ///
    /// Fraction of recent predictions that were accurate (0.0 to 1.0).
    #[must_use]
    pub fn accuracy_rate(&self) -> f32 {
        let sum: f32 = self.accuracy_history.iter().sum();
        sum / self.accuracy_history.len() as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_predictor_defaults() {
        let pred = MultiStepPredictor::new(0.85, 0.3, true);
        assert!(pred.enable_batch_skip);
        assert!((pred.batch_skip_confidence_threshold - 0.85).abs() < 1e-6);
        assert!((pred.batch_skip_error_threshold - 0.3).abs() < 1e-6);
        assert_eq!(pred.batches_skipped, 0);
        assert_eq!(pred.steps_skipped, 0);
        assert_eq!(pred.accurate_streak(), 0);
    }

    #[test]
    fn test_predict_batch_high_confidence_skip() {
        let pred = MultiStepPredictor::new(0.85, 0.3, true);
        let batch = pred.predict_batch(0.9, 0.2, 4);

        assert_eq!(batch.num_steps, 4);
        assert!(batch.overall_confidence > 0.8);
        assert_eq!(
            batch.recommendation,
            BatchPredictionRecommendation::SkipBatch
        );
    }

    #[test]
    fn test_predict_batch_medium_confidence_verify() {
        let pred = MultiStepPredictor::new(0.85, 0.3, true);
        let batch = pred.predict_batch(0.75, 0.15, 4);

        assert!(batch.overall_confidence > 0.7);
        assert_eq!(
            batch.recommendation,
            BatchPredictionRecommendation::ApplyWithVerification
        );
    }

    #[test]
    fn test_predict_batch_low_confidence_full() {
        let pred = MultiStepPredictor::new(0.85, 0.3, true);
        let batch = pred.predict_batch(0.6, 0.4, 4);

        assert!(batch.overall_confidence < 0.7);
        assert_eq!(
            batch.recommendation,
            BatchPredictionRecommendation::RunFullBatch
        );
    }

    #[test]
    fn test_predict_batch_skip_disabled() {
        let pred = MultiStepPredictor::new(0.85, 0.3, false);
        let batch = pred.predict_batch(0.9, 0.1, 4);

        // Even with high confidence, should not recommend skip when disabled
        assert_ne!(
            batch.recommendation,
            BatchPredictionRecommendation::SkipBatch
        );
    }

    #[test]
    fn test_record_observation_updates_calibration() {
        let mut pred = MultiStepPredictor::new(0.85, 0.3, true);

        // Record several observations for horizon 1
        pred.record_observation(1, 2.5, 2.4); // error = 0.1
        pred.record_observation(1, 2.4, 2.35); // error = 0.05
        pred.record_observation(1, 2.35, 2.3); // error = 0.05

        // Check that calibration factor was updated
        let cal = pred.calibration_factor_for_horizon(1);
        assert!(cal > 0.0 && cal < 1.0);

        // Mean error = (0.1 + 0.05 + 0.05) / 3 = 0.0667
        // Calibration = 1.0 / (1.0 + 0.0667) ≈ 0.9375
        assert!((cal - 0.9375).abs() < 0.01);
    }

    #[test]
    fn test_accurate_streak() {
        let mut pred = MultiStepPredictor::new(0.85, 0.3, true);

        // Record accurate predictions (error < 0.1)
        pred.record_observation(1, 2.5, 2.505); // error = 0.005 - accurate
        assert_eq!(pred.accurate_streak(), 1);

        pred.record_observation(1, 2.5, 2.509); // error = 0.009 - accurate
        assert_eq!(pred.accurate_streak(), 2);

        pred.record_observation(1, 2.5, 2.61); // error = 0.11 - inaccurate
        assert_eq!(pred.accurate_streak(), 0);
    }

    #[test]
    fn test_skip_stats() {
        let mut pred = MultiStepPredictor::new(0.85, 0.3, true);

        assert_eq!(pred.skip_stats(), (0, 0));

        pred.record_batch_skip(8);
        assert_eq!(pred.skip_stats(), (1, 8));

        pred.record_batch_skip(4);
        assert_eq!(pred.skip_stats(), (2, 12));
    }

    #[test]
    fn test_horizon_computation() {
        let pred = MultiStepPredictor::new(0.85, 0.3, true);

        let horizons_4 = pred.compute_horizons(4);
        assert_eq!(horizons_4, vec![1, 2, 4]);

        let horizons_8 = pred.compute_horizons(8);
        assert_eq!(horizons_8, vec![1, 2, 4, 8]);

        let horizons_1 = pred.compute_horizons(1);
        assert_eq!(horizons_1, vec![1]);
    }

    #[test]
    fn test_accuracy_history_ring_buffer() {
        let mut pred = MultiStepPredictor::new(0.85, 0.3, true);

        // Record 100 accurate predictions (error < 0.1) - fills entire ring buffer
        for _ in 0..100 {
            pred.record_observation(1, 2.5, 2.505); // error = 0.005
        }
        // Ring buffer has 100 accurate predictions
        // Accuracy should be 100/100 = 1.0
        assert!(pred.accuracy_rate() > 0.99);

        // Record 50 inaccurate predictions (error > 0.1) - wraps around and replaces first 50
        for _ in 0..50 {
            pred.record_observation(1, 2.5, 2.8); // error = 0.3
        }
        // Ring buffer now has 50 inaccurate (0.0) at positions 0-49 (wrapped) and 50 accurate at 50-99
        // Accuracy should be 50/100 = 0.5
        assert!((pred.accuracy_rate() - 0.5).abs() < 0.01);

        // Record 50 more inaccurate predictions - fills positions 50-99
        for _ in 0..50 {
            pred.record_observation(1, 2.5, 2.8); // error = 0.3
        }
        // Ring buffer now has all inaccurate predictions
        // Accuracy should be 0/100 = 0.0
        assert!(pred.accuracy_rate() < 0.01);
    }

    #[test]
    fn test_reset() {
        let mut pred = MultiStepPredictor::new(0.85, 0.3, true);

        // Populate with data
        pred.record_observation(1, 2.5, 2.4);
        pred.record_observation(2, 2.4, 2.35);
        pred.record_batch_skip(4);

        assert!(!pred.horizon_errors.is_empty());
        assert_eq!(pred.accurate_streak(), 2);
        assert_eq!(pred.batches_skipped, 1); // Before reset

        pred.reset();

        // Calibration and accuracy should be reset
        assert!(pred.horizon_errors.is_empty());
        assert_eq!(pred.accurate_streak(), 0);
        // But skip_stats should persist
        assert_eq!(pred.batches_skipped, 1);
    }

    #[test]
    fn test_confidence_clamping() {
        let pred = MultiStepPredictor::new(0.85, 0.3, true);

        // Test with out-of-range inputs
        let batch = pred.predict_batch(1.5, -0.1, 4);
        assert!(batch.overall_confidence >= 0.0 && batch.overall_confidence <= 1.0);
    }

    #[test]
    fn test_per_horizon_confidence() {
        let pred = MultiStepPredictor::new(0.85, 0.3, true);
        let batch = pred.predict_batch(0.8, 0.2, 8);

        // Should have horizons [1, 2, 4, 8]
        assert_eq!(batch.horizon_confidences.len(), 4);

        // All should be valid confidences
        for conf in &batch.horizon_confidences {
            assert!(*conf >= 0.0 && *conf <= 1.0);
        }
    }

    #[test]
    fn test_error_threshold_blocking_skip() {
        let pred = MultiStepPredictor::new(0.85, 0.3, true);

        // High confidence but high uncertainty should not recommend skip
        let batch = pred.predict_batch(0.9, 0.5, 4);
        assert_ne!(
            batch.recommendation,
            BatchPredictionRecommendation::SkipBatch
        );
    }

    #[test]
    fn test_multiple_horizons_independent() {
        let mut pred = MultiStepPredictor::new(0.85, 0.3, true);

        // Record very accurate predictions for horizon 1
        for _ in 0..10 {
            pred.record_observation(1, 2.5, 2.50);
        }

        // Record very inaccurate predictions for horizon 4
        for _ in 0..10 {
            pred.record_observation(4, 2.5, 2.9);
        }

        let cal1 = pred.calibration_factor_for_horizon(1);
        let cal4 = pred.calibration_factor_for_horizon(4);

        // Horizon 1 should have much better calibration
        assert!(cal1 > cal4);
    }
}
