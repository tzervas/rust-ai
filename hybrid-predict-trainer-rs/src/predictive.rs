//! Predictive training phase implementation.
//!
//! The predictive phase skips backward passes by using the learned dynamics
//! model to predict weight updates. This provides significant speedup at
//! the cost of some prediction error, which is corrected in the subsequent
//! correction phase.
//!
//! # How Prediction Works
//!
//! 1. The dynamics model takes the current training state as input
//! 2. It predicts the weight changes that would result from Y training steps
//! 3. These predicted deltas are applied to model weights
//! 4. Only forward passes are computed to track actual loss for monitoring
//!
//! # Prediction Confidence
//!
//! The predictor maintains a confidence estimate based on:
//! - Ensemble disagreement (if using ensemble predictor)
//! - Historical prediction error
//! - Training stability indicators
//!
//! If confidence drops below threshold, the phase terminates early and
//! transitions to full training.
//!
//! # Divergence Detection
//!
//! During prediction, the actual loss is monitored against predicted loss.
//! If divergence exceeds threshold, the phase terminates and the trainer
//! can rollback to the last checkpoint if necessary.

use crate::config::HybridTrainerConfig;
use crate::error::{HybridResult, HybridTrainingError, RecoveryAction};
use crate::phases::PhaseOutcome;
use crate::state::{TrainingState, WeightDelta};
use crate::Phase;

/// Statistics collected during predictive phase.
#[derive(Debug, Clone, Default)]
pub struct PredictiveStatistics {
    /// Number of prediction steps completed.
    pub steps_completed: usize,

    /// Sum of predicted losses.
    pub predicted_loss_sum: f64,

    /// Sum of actual losses.
    pub actual_loss_sum: f64,

    /// Sum of squared prediction errors.
    pub prediction_error_sq_sum: f64,

    /// Maximum prediction error observed.
    pub max_prediction_error: f32,

    /// Number of steps where prediction was accurate (within threshold).
    pub accurate_predictions: usize,

    /// Number of early terminations due to divergence.
    pub divergence_terminations: usize,

    /// Total backward passes avoided.
    pub backward_passes_avoided: usize,

    /// Estimated time saved in milliseconds.
    pub estimated_time_saved_ms: f64,
}

impl PredictiveStatistics {
    /// Returns the mean absolute prediction error.
    #[must_use]
    pub fn mean_prediction_error(&self) -> f32 {
        if self.steps_completed == 0 {
            0.0
        } else {
            let mse = self.prediction_error_sq_sum / self.steps_completed as f64;
            mse.sqrt() as f32
        }
    }

    /// Returns the prediction accuracy rate.
    #[must_use]
    pub fn accuracy_rate(&self) -> f32 {
        if self.steps_completed == 0 {
            0.0
        } else {
            self.accurate_predictions as f32 / self.steps_completed as f32
        }
    }

    /// Records a prediction step result.
    pub fn record_step(&mut self, predicted_loss: f32, actual_loss: f32, accuracy_threshold: f32) {
        self.steps_completed += 1;
        self.backward_passes_avoided += 1;

        self.predicted_loss_sum += f64::from(predicted_loss);
        self.actual_loss_sum += f64::from(actual_loss);

        let error = (actual_loss - predicted_loss).abs();
        self.prediction_error_sq_sum += f64::from(error * error);

        if error > self.max_prediction_error {
            self.max_prediction_error = error;
        }

        if error < accuracy_threshold {
            self.accurate_predictions += 1;
        }
    }
}

/// Prediction for a single step or batch of steps.
#[derive(Debug, Clone)]
pub struct StepPrediction {
    /// Predicted weight delta.
    pub weight_delta: WeightDelta,

    /// Predicted loss after applying delta.
    pub predicted_loss: f32,

    /// Confidence in this prediction.
    pub confidence: f32,

    /// Uncertainty bounds (low, high) for predicted loss.
    pub loss_bounds: (f32, f32),
}

/// Prediction for an entire phase (multiple steps).
#[derive(Debug, Clone)]
pub struct PhasePrediction {
    /// Aggregate weight delta for all steps.
    pub weight_delta: WeightDelta,

    /// Predicted final loss after phase.
    pub predicted_final_loss: f32,

    /// Sparse loss trajectory (key points).
    pub loss_trajectory: Vec<f32>,

    /// Overall confidence for the phase prediction.
    pub confidence: f32,

    /// Uncertainty bounds for final loss.
    pub loss_bounds: (f32, f32),

    /// Number of steps this prediction covers.
    pub num_steps: usize,
}

/// Executor for the predictive phase.
pub struct PredictiveExecutor {
    /// Maximum number of prediction steps.
    max_steps: usize,

    /// Current step within phase.
    current_step: usize,

    /// Collected statistics.
    statistics: PredictiveStatistics,

    /// Current confidence level.
    confidence: f32,

    /// Confidence threshold for continuing.
    confidence_threshold: f32,

    /// Fallback loss threshold for early termination.
    fallback_threshold: f32,

    /// Accuracy threshold for "accurate" prediction.
    accuracy_threshold: f32,

    /// Start time for duration tracking.
    start_time: Option<std::time::Instant>,

    /// Reason for early termination (if any).
    termination_reason: Option<String>,
}

impl PredictiveExecutor {
    /// Creates a new predictive phase executor.
    ///
    /// # Arguments
    ///
    /// * `max_steps` - Maximum number of prediction steps
    /// * `confidence` - Initial predictor confidence
    /// * `confidence_threshold` - Minimum confidence to continue
    /// * `fallback_threshold` - Loss threshold for early termination
    #[must_use]
    pub fn new(
        max_steps: usize,
        confidence: f32,
        confidence_threshold: f32,
        fallback_threshold: f32,
    ) -> Self {
        Self {
            max_steps,
            current_step: 0,
            statistics: PredictiveStatistics::default(),
            confidence,
            confidence_threshold,
            fallback_threshold,
            accuracy_threshold: 0.1, // 10% relative error
            start_time: None,
            termination_reason: None,
        }
    }

    /// Creates an executor from configuration.
    #[must_use]
    pub fn from_config(config: &HybridTrainerConfig, confidence: f32) -> Self {
        let loss_threshold = 10.0; // Default, should be computed from history
        Self::new(
            config.max_predict_steps,
            confidence,
            config.confidence_threshold,
            loss_threshold,
        )
    }

    /// Returns whether the phase should terminate.
    #[must_use]
    pub fn should_terminate(&self) -> bool {
        self.current_step >= self.max_steps
            || self.confidence < self.confidence_threshold
            || self.termination_reason.is_some()
    }

    /// Returns the current progress as a fraction [0, 1].
    #[must_use]
    pub fn progress(&self) -> f32 {
        if self.max_steps == 0 {
            1.0
        } else {
            (self.current_step as f32) / (self.max_steps as f32)
        }
    }

    /// Returns the number of remaining steps.
    #[must_use]
    pub fn steps_remaining(&self) -> usize {
        self.max_steps.saturating_sub(self.current_step)
    }

    /// Returns collected statistics.
    #[must_use]
    pub fn statistics(&self) -> &PredictiveStatistics {
        &self.statistics
    }

    /// Returns the current confidence.
    #[must_use]
    pub fn current_confidence(&self) -> f32 {
        self.confidence
    }

    /// Records a prediction step result.
    ///
    /// # Arguments
    ///
    /// * `predicted_loss` - The predicted loss value
    /// * `actual_loss` - The actual loss (from forward pass only)
    /// * `new_confidence` - Updated predictor confidence
    ///
    /// # Returns
    ///
    /// `Ok(())` if prediction is acceptable, or error if divergence detected.
    pub fn record_step(
        &mut self,
        predicted_loss: f32,
        actual_loss: f32,
        new_confidence: f32,
    ) -> HybridResult<()> {
        if self.start_time.is_none() {
            self.start_time = Some(std::time::Instant::now());
        }

        let relative_threshold = self.accuracy_threshold * actual_loss.abs().max(0.1);
        self.statistics
            .record_step(predicted_loss, actual_loss, relative_threshold);

        self.confidence = new_confidence;
        self.current_step += 1;

        // Check for divergence
        if actual_loss > self.fallback_threshold {
            self.termination_reason = Some(format!(
                "Loss {} exceeded threshold {}",
                actual_loss, self.fallback_threshold
            ));
            self.statistics.divergence_terminations += 1;

            return Err((
                HybridTrainingError::PredictionDivergence {
                    actual: actual_loss,
                    predicted: predicted_loss,
                    delta: (actual_loss - predicted_loss).abs(),
                    step: self.current_step as u64,
                },
                Some(RecoveryAction::ForceFullPhase(20)),
            ));
        }

        // Check confidence
        if self.confidence < self.confidence_threshold {
            self.termination_reason = Some(format!(
                "Confidence {} dropped below threshold {}",
                self.confidence, self.confidence_threshold
            ));
        }

        Ok(())
    }

    /// Finalizes the phase and returns the outcome.
    #[must_use]
    pub fn finalize(self) -> PhaseOutcome {
        let duration_ms = self
            .start_time
            .map_or(0.0, |t| t.elapsed().as_secs_f64() * 1000.0);

        let average_actual = if self.statistics.steps_completed > 0 {
            (self.statistics.actual_loss_sum / self.statistics.steps_completed as f64) as f32
        } else {
            0.0
        };

        PhaseOutcome {
            phase: Phase::Predict,
            steps_executed: self.current_step,
            average_loss: average_actual,
            final_loss: average_actual, // Approximate
            completed_normally: self.termination_reason.is_none()
                && self.current_step >= self.max_steps,
            early_termination_reason: self.termination_reason,
            prediction_error: Some(self.statistics.mean_prediction_error()),
            duration_ms,
        }
    }
}

/// Trait for predictive phase executors.
pub trait PredictivePhase: Send + Sync {
    /// Makes a prediction for the next step.
    ///
    /// # Arguments
    ///
    /// * `state` - Current training state
    ///
    /// # Returns
    ///
    /// Prediction for the next step.
    fn predict_step(&self, state: &TrainingState) -> HybridResult<StepPrediction>;

    /// Makes a prediction for multiple steps.
    ///
    /// # Arguments
    ///
    /// * `state` - Current training state
    /// * `num_steps` - Number of steps to predict
    ///
    /// # Returns
    ///
    /// Aggregate prediction for the phase.
    fn predict_phase(
        &self,
        state: &TrainingState,
        num_steps: usize,
    ) -> HybridResult<PhasePrediction>;

    /// Returns the current prediction confidence.
    fn confidence(&self, state: &TrainingState) -> f32;

    /// Updates the predictor from observed training data.
    fn update_from_observation(
        &mut self,
        state_before: &TrainingState,
        state_after: &TrainingState,
        actual_loss_trajectory: &[f32],
    ) -> HybridResult<()>;
}

/// Result of a single predictive step.
#[derive(Debug, Clone)]
pub struct PredictiveStepResult {
    /// The prediction that was applied.
    pub prediction: StepPrediction,

    /// Actual loss (from forward pass).
    pub actual_loss: f32,

    /// Prediction error.
    pub prediction_error: f32,

    /// Updated confidence.
    pub new_confidence: f32,

    /// Step time in milliseconds.
    pub step_time_ms: f64,

    /// Whether to continue or terminate.
    pub should_continue: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_statistics_tracking() {
        let mut stats = PredictiveStatistics::default();

        stats.record_step(2.5, 2.6, 0.2);
        stats.record_step(2.4, 2.5, 0.2);
        stats.record_step(2.3, 2.4, 0.2);

        assert_eq!(stats.steps_completed, 3);
        assert_eq!(stats.backward_passes_avoided, 3);
        assert!(stats.mean_prediction_error() < 0.2);
    }

    #[test]
    fn test_executor_termination() {
        let mut executor = PredictiveExecutor::new(10, 0.9, 0.85, 5.0);

        // Should not terminate initially
        assert!(!executor.should_terminate());

        // Record some steps
        for _ in 0..10 {
            let _ = executor.record_step(2.5, 2.6, 0.9);
        }

        // Should terminate after max steps
        assert!(executor.should_terminate());
    }

    #[test]
    fn test_divergence_detection() {
        let mut executor = PredictiveExecutor::new(10, 0.9, 0.85, 3.0);

        // Record a step that exceeds fallback threshold
        let result = executor.record_step(2.5, 5.0, 0.9);

        assert!(result.is_err());
        assert!(executor.should_terminate());
    }

    #[test]
    fn test_confidence_termination() {
        let mut executor = PredictiveExecutor::new(10, 0.9, 0.85, 10.0);

        // Record step with low confidence
        let _ = executor.record_step(2.5, 2.6, 0.5);

        assert!(executor.should_terminate());
    }
}
