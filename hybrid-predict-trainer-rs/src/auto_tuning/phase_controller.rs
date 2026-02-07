//! Gradient-aware adaptive phase controller.
//!
//! This module implements an intelligent phase controller that adapts prediction
//! length based on gradient stability, prediction confidence, and training health.
//!
//! # Core Concept
//!
//! The `AdaptivePhaseController` dynamically adjusts phase lengths to match current
//! training conditions. When training is healthy and predictions are accurate, we
//! run longer prediction phases. When conditions degrade, we fall back to more
//! frequent full training.
//!
//! # Phase Transitions
//!
//! The controller manages these transitions:
//! - **Warmup → Full**: After `warmup_steps`
//! - **Full/Correct → Predict**: If health >= Moderate AND confidence >= 0.85
//! - **Predict → Correct**: After dynamic prediction steps
//! - **Correct → Full**: Always
//!
//! # Dynamic Predict Steps
//!
//! Prediction length is computed as:
//! ```text
//! combined_confidence = prediction_confidence × gradient_stability
//! predict_steps = base × combined_confidence
//!
//! where base depends on health:
//!   Excellent: max_predict_steps
//!   Good:      0.8 × max_predict_steps
//!   Moderate:  0.5 × max_predict_steps
//!   Poor:      min_predict_steps × 2
//!   Critical:  min_predict_steps
//! ```
//!
//! # Error Trend Monitoring
//!
//! The controller maintains a ring buffer of prediction errors and:
//! - **Extends Full phase** if error trend is increasing (> threshold over last 10)
//! - **Shortens Full phase** if error remains low for a streak (10+ consecutive low)
//!
//! # Example
//!
//! ```rust
//! use hybrid_predict_trainer_rs::auto_tuning::phase_controller::AdaptivePhaseController;
//! use hybrid_predict_trainer_rs::auto_tuning::HealthClassification;
//!
//! let mut controller = AdaptivePhaseController::new(5, 100);
//! controller.record_prediction_error(0.02);
//!
//! // In training loop:
//! // let (phase, steps) = controller.select_next_phase(
//! //     HealthClassification::Good,
//! //     0.75,
//! //     gradient_stability,
//! //     prediction_confidence,
//! //     current_step,
//! //     warmup_complete,
//! // );
//! ```

use crate::auto_tuning::HealthClassification;
use crate::Phase;
use serde::{Deserialize, Serialize};

/// Returns the base prediction step factor for a given health classification.
///
/// Higher factors result in longer prediction phases.
///
/// # Arguments
///
/// * `classification` - The health classification
/// * `max_steps` - Maximum prediction steps allowed
///
/// # Returns
///
/// The base prediction steps for this health level
fn health_classification_base_factor(
    classification: HealthClassification,
    max_steps: usize,
) -> usize {
    match classification {
        HealthClassification::Excellent => max_steps,
        HealthClassification::Good => ((max_steps as f32) * 0.8) as usize,
        HealthClassification::Moderate => ((max_steps as f32) * 0.5) as usize,
        HealthClassification::Poor => 2,
        HealthClassification::Critical => 1,
    }
}

/// Adaptive phase controller that adjusts predictions based on training dynamics.
///
/// This controller maintains a history of prediction errors and training metrics,
/// using them to dynamically adjust how long prediction phases should run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptivePhaseController {
    /// Current training phase.
    current_phase: Phase,

    /// Number of steps to execute in the current predict phase.
    current_predict_steps: usize,

    /// Steps remaining in the current phase.
    steps_remaining: usize,

    /// Number of times the full phase has been extended due to error trend.
    full_phase_extensions: u32,

    /// Ring buffer of recent prediction errors (capacity 50).
    prediction_error_trend: Vec<f32>,

    /// Index in the ring buffer for the next write.
    error_buffer_index: usize,

    /// Step number when the last phase transition occurred.
    last_transition_step: u64,

    /// Minimum prediction steps before considering other conditions.
    min_predict_steps: usize,

    /// Maximum prediction steps allowed.
    max_predict_steps: usize,

    /// Threshold for error trend to trigger full phase extension.
    ///
    /// If error trend exceeds this over the last 10 samples, extend full phase.
    error_trend_extend_threshold: f32,

    /// Number of consecutive low-error steps before shortening full phase.
    accurate_streak_shorten_threshold: usize,

    /// Current streak of low-error steps.
    current_low_error_streak: usize,

    /// Whether warmup has been completed.
    warmup_complete: bool,
}

impl AdaptivePhaseController {
    /// Creates a new adaptive phase controller.
    ///
    /// # Arguments
    ///
    /// * `min_predict_steps` - Minimum prediction steps (e.g., 5)
    /// * `max_predict_steps` - Maximum prediction steps (e.g., 100)
    ///
    /// # Returns
    ///
    /// A new controller with default thresholds and ring buffer.
    #[must_use]
    pub fn new(min_predict_steps: usize, max_predict_steps: usize) -> Self {
        Self {
            current_phase: Phase::Warmup,
            current_predict_steps: min_predict_steps,
            steps_remaining: 0,
            full_phase_extensions: 0,
            prediction_error_trend: vec![0.0; 50],
            error_buffer_index: 0,
            last_transition_step: 0,
            min_predict_steps,
            max_predict_steps,
            error_trend_extend_threshold: 0.05,
            accurate_streak_shorten_threshold: 10,
            current_low_error_streak: 0,
            warmup_complete: false,
        }
    }

    /// Selects the next phase based on health and training metrics.
    ///
    /// # Arguments
    ///
    /// * `health_classification` - Overall training health (Excellent/Good/Moderate/Poor/Critical)
    /// * `health_score` - Numerical health metric (0.0-1.0)
    /// * `gradient_stability` - Gradient norm stability factor (0.0-1.0)
    /// * `prediction_confidence` - Predictor confidence (0.0-1.0)
    /// * `step` - Current training step
    /// * `warmup_complete` - Whether warmup phase is done
    ///
    /// # Returns
    ///
    /// Tuple of (Phase, `duration_in_steps`)
    #[allow(clippy::too_many_arguments)]
    pub fn select_next_phase(
        &mut self,
        health_classification: HealthClassification,
        _health_score: f32,
        gradient_stability: f32,
        prediction_confidence: f32,
        step: u64,
        warmup_complete: bool,
    ) -> (Phase, usize) {
        self.warmup_complete = warmup_complete;

        // Handle warmup completion
        if !self.warmup_complete {
            self.current_phase = Phase::Warmup;
            return (Phase::Warmup, 20);
        }

        // If warmup just completed, transition to Full
        if !self.warmup_complete && step > 0 {
            self.warmup_complete = true;
            self.last_transition_step = step;
            self.current_phase = Phase::Full;
            let extension = if self.should_extend_full_phase() {
                10
            } else {
                0
            };
            return (Phase::Full, 30 + extension);
        }

        match self.current_phase {
            Phase::Warmup => {
                // This case is handled above
                self.current_phase = Phase::Full;
                let extension = if self.should_extend_full_phase() {
                    10
                } else {
                    0
                };
                (Phase::Full, 30 + extension)
            }

            Phase::Full | Phase::Correct => {
                // Decide between Predict and Full based on health and confidence
                if health_classification != HealthClassification::Critical
                    && health_classification != HealthClassification::Poor
                    && prediction_confidence >= 0.85
                {
                    // Conditions favorable for prediction
                    self.current_phase = Phase::Predict;
                    self.last_transition_step = step;

                    let predict_steps = self.compute_dynamic_predict_steps(
                        prediction_confidence,
                        gradient_stability,
                        health_classification,
                    );
                    self.current_predict_steps = predict_steps;
                    (Phase::Predict, predict_steps)
                } else {
                    // Stay in or return to Full training
                    self.current_phase = Phase::Full;
                    self.last_transition_step = step;

                    let extension = if self.should_extend_full_phase() {
                        10
                    } else {
                        0
                    };
                    (Phase::Full, 30 + extension)
                }
            }

            Phase::Predict => {
                // Transition from Predict to Correct
                self.current_phase = Phase::Correct;
                self.last_transition_step = step;
                (Phase::Correct, 5)
            }
        }
    }

    /// Computes dynamic prediction steps based on confidence and stability.
    ///
    /// # Arguments
    ///
    /// * `prediction_confidence` - Predictor confidence (0.0-1.0)
    /// * `gradient_stability` - Gradient stability (0.0-1.0)
    /// * `health_classification` - Current health classification
    ///
    /// # Returns
    ///
    /// Number of prediction steps to execute
    fn compute_dynamic_predict_steps(
        &self,
        prediction_confidence: f32,
        gradient_stability: f32,
        health_classification: HealthClassification,
    ) -> usize {
        // Combined confidence from prediction and gradient metrics
        let combined_confidence = prediction_confidence * gradient_stability;

        // Base steps depend on health
        let base_steps =
            health_classification_base_factor(health_classification, self.max_predict_steps);

        // Scale by combined confidence
        let adaptive_steps = (base_steps as f32 * combined_confidence) as usize;

        // Clamp to valid range
        adaptive_steps
            .max(self.min_predict_steps)
            .min(self.max_predict_steps)
    }

    /// Determines if the full phase should be extended.
    ///
    /// Full phase is extended if the prediction error trend has been increasing
    /// over the last 10 samples, indicating degrading prediction quality.
    ///
    /// # Returns
    ///
    /// `true` if error trend exceeds threshold
    fn should_extend_full_phase(&self) -> bool {
        if self.prediction_error_trend.len() < 10 {
            return false;
        }

        // Check last 10 errors
        let mut error_sum = 0.0;
        for i in 0..10 {
            let idx = if self.error_buffer_index >= i {
                self.error_buffer_index - i
            } else {
                self.prediction_error_trend.len() - (i - self.error_buffer_index)
            };
            error_sum += self.prediction_error_trend[idx];
        }

        let avg_error = error_sum / 10.0;
        avg_error > self.error_trend_extend_threshold
    }

    /// Determines if the full phase should be shortened.
    ///
    /// Full phase can be shortened if we have a long streak of low errors,
    /// indicating stable and accurate predictions.
    ///
    /// # Returns
    ///
    /// `true` if we have a long enough streak of accurate predictions
    pub fn can_shorten_full_phase(&self) -> bool {
        self.current_low_error_streak >= self.accurate_streak_shorten_threshold
    }

    /// Records a prediction error for trend monitoring.
    ///
    /// # Arguments
    ///
    /// * `error` - Prediction error (typically 0.0-1.0)
    pub fn record_prediction_error(&mut self, error: f32) {
        self.prediction_error_trend[self.error_buffer_index] = error;
        self.error_buffer_index = (self.error_buffer_index + 1) % self.prediction_error_trend.len();

        // Update low-error streak
        if error < self.error_trend_extend_threshold {
            self.current_low_error_streak += 1;
        } else {
            self.current_low_error_streak = 0;
        }
    }

    /// Forces a transition to a specific phase.
    ///
    /// # Arguments
    ///
    /// * `phase` - The phase to force
    /// * `steps` - Number of steps for this phase
    pub fn force_phase(&mut self, phase: Phase, steps: usize) {
        self.current_phase = phase;
        self.steps_remaining = steps;
        if phase == Phase::Predict {
            self.current_predict_steps = steps;
        }
    }

    /// Returns the number of prediction steps in the current predict phase.
    ///
    /// # Returns
    ///
    /// Current prediction step count
    #[must_use]
    pub fn current_predict_steps(&self) -> usize {
        self.current_predict_steps
    }

    /// Returns the current phase.
    ///
    /// # Returns
    ///
    /// The current training phase
    #[must_use]
    pub fn current_phase(&self) -> Phase {
        self.current_phase
    }

    /// Returns the step when the last phase transition occurred.
    ///
    /// # Returns
    ///
    /// Last transition step
    #[must_use]
    pub fn last_transition_step(&self) -> u64 {
        self.last_transition_step
    }

    /// Returns the number of times full phase has been extended.
    ///
    /// # Returns
    ///
    /// Extension count
    #[must_use]
    pub fn full_phase_extensions(&self) -> u32 {
        self.full_phase_extensions
    }

    /// Sets the error trend extension threshold.
    ///
    /// # Arguments
    ///
    /// * `threshold` - New threshold value (0.0-1.0)
    pub fn set_error_trend_extend_threshold(&mut self, threshold: f32) {
        self.error_trend_extend_threshold = threshold;
    }

    /// Sets the accurate streak shortening threshold.
    ///
    /// # Arguments
    ///
    /// * `threshold` - New threshold (number of consecutive low errors)
    pub fn set_accurate_streak_shorten_threshold(&mut self, threshold: usize) {
        self.accurate_streak_shorten_threshold = threshold;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_controller_initialization() {
        let controller = AdaptivePhaseController::new(5, 100);
        assert_eq!(controller.current_phase(), Phase::Warmup);
        assert_eq!(controller.current_predict_steps(), 5);
        assert_eq!(controller.full_phase_extensions(), 0);
    }

    #[test]
    fn test_phase_transition_warmup_to_full() {
        let mut controller = AdaptivePhaseController::new(5, 100);
        let (phase, steps) =
            controller.select_next_phase(HealthClassification::Good, 0.8, 0.9, 0.9, 10, true);
        assert_eq!(phase, Phase::Full);
        assert!(steps > 0);
    }

    #[test]
    fn test_phase_transition_full_to_predict() {
        let mut controller = AdaptivePhaseController::new(5, 100);
        controller.warmup_complete = true;
        controller.current_phase = Phase::Full;

        let (phase, steps) = controller.select_next_phase(
            HealthClassification::Excellent,
            0.9,
            0.95,
            0.95,
            50,
            true,
        );
        assert_eq!(phase, Phase::Predict);
        assert!(steps <= 100);
        assert!(steps >= 5);
    }

    #[test]
    fn test_phase_transition_full_to_full_on_poor_health() {
        let mut controller = AdaptivePhaseController::new(5, 100);
        controller.warmup_complete = true;
        controller.current_phase = Phase::Full;

        let (phase, _steps) =
            controller.select_next_phase(HealthClassification::Poor, 0.3, 0.5, 0.9, 50, true);
        assert_eq!(phase, Phase::Full);
    }

    #[test]
    fn test_phase_transition_full_to_full_on_low_confidence() {
        let mut controller = AdaptivePhaseController::new(5, 100);
        controller.warmup_complete = true;
        controller.current_phase = Phase::Full;

        let (phase, _steps) = controller.select_next_phase(
            HealthClassification::Good,
            0.8,
            0.9,
            0.7, // Low confidence
            50,
            true,
        );
        assert_eq!(phase, Phase::Full);
    }

    #[test]
    fn test_phase_transition_predict_to_correct() {
        let mut controller = AdaptivePhaseController::new(5, 100);
        controller.warmup_complete = true;
        controller.current_phase = Phase::Predict;

        let (phase, steps) =
            controller.select_next_phase(HealthClassification::Good, 0.8, 0.9, 0.9, 50, true);
        assert_eq!(phase, Phase::Correct);
        assert_eq!(steps, 5);
    }

    #[test]
    fn test_compute_dynamic_predict_steps_excellent() {
        let controller = AdaptivePhaseController::new(5, 100);
        let steps =
            controller.compute_dynamic_predict_steps(0.9, 0.9, HealthClassification::Excellent);
        assert!(steps >= 80); // 100 * 0.9 * 0.9 = 81
        assert!(steps <= 100);
    }

    #[test]
    fn test_compute_dynamic_predict_steps_moderate() {
        let controller = AdaptivePhaseController::new(5, 100);
        let steps =
            controller.compute_dynamic_predict_steps(0.9, 0.9, HealthClassification::Moderate);
        assert!(steps >= 5); // 50 * 0.9 * 0.9 = 40.5, but clamped
        assert!(steps <= 50);
    }

    #[test]
    fn test_compute_dynamic_predict_steps_critical() {
        let controller = AdaptivePhaseController::new(5, 100);
        let steps =
            controller.compute_dynamic_predict_steps(0.9, 0.9, HealthClassification::Critical);
        assert_eq!(steps, 5); // Critical forces min_predict_steps
    }

    #[test]
    fn test_health_classification_factors() {
        assert_eq!(
            health_classification_base_factor(HealthClassification::Excellent, 100),
            100
        );
        assert_eq!(
            health_classification_base_factor(HealthClassification::Good, 100),
            80
        );
        assert_eq!(
            health_classification_base_factor(HealthClassification::Moderate, 100),
            50
        );
        assert!(health_classification_base_factor(HealthClassification::Poor, 100) <= 5);
        assert!(health_classification_base_factor(HealthClassification::Critical, 100) <= 5);
    }

    #[test]
    fn test_record_prediction_error() {
        let mut controller = AdaptivePhaseController::new(5, 100);
        controller.record_prediction_error(0.02);
        assert_eq!(controller.current_low_error_streak, 1);

        controller.record_prediction_error(0.03);
        assert_eq!(controller.current_low_error_streak, 2);

        controller.record_prediction_error(0.1);
        assert_eq!(controller.current_low_error_streak, 0);
    }

    #[test]
    fn test_can_shorten_full_phase() {
        let mut controller = AdaptivePhaseController::new(5, 100);
        controller.accurate_streak_shorten_threshold = 3;

        controller.record_prediction_error(0.01);
        assert!(!controller.can_shorten_full_phase());

        controller.record_prediction_error(0.01);
        assert!(!controller.can_shorten_full_phase());

        controller.record_prediction_error(0.01);
        assert!(controller.can_shorten_full_phase());
    }

    #[test]
    fn test_force_phase() {
        let mut controller = AdaptivePhaseController::new(5, 100);
        controller.force_phase(Phase::Predict, 25);
        assert_eq!(controller.current_phase(), Phase::Predict);
        assert_eq!(controller.current_predict_steps(), 25);
    }

    #[test]
    fn test_error_buffer_ring_wrap() {
        let mut controller = AdaptivePhaseController::new(5, 100);

        // Fill the buffer and wrap around
        for i in 0..60 {
            controller.record_prediction_error((i % 50) as f32 * 0.01);
        }

        // The buffer should still be valid
        assert!(controller.prediction_error_trend.len() == 50);
    }

    #[test]
    fn test_should_extend_full_phase() {
        let mut controller = AdaptivePhaseController::new(5, 100);
        controller.error_trend_extend_threshold = 0.05;

        // Add high errors
        for _ in 0..15 {
            controller.record_prediction_error(0.08);
        }

        assert!(controller.should_extend_full_phase());
    }

    #[test]
    fn test_should_not_extend_full_phase_low_errors() {
        let mut controller = AdaptivePhaseController::new(5, 100);
        controller.error_trend_extend_threshold = 0.05;

        // Add low errors
        for _ in 0..15 {
            controller.record_prediction_error(0.02);
        }

        assert!(!controller.should_extend_full_phase());
    }

    #[test]
    fn test_last_transition_step() {
        let mut controller = AdaptivePhaseController::new(5, 100);
        controller.last_transition_step = 42;
        assert_eq!(controller.last_transition_step(), 42);
    }

    #[test]
    fn test_serialization_roundtrip() {
        let mut controller = AdaptivePhaseController::new(5, 100);
        controller.record_prediction_error(0.05);
        controller.force_phase(Phase::Predict, 30);

        let serialized = serde_json::to_string(&controller).expect("Serialization failed");
        let deserialized: AdaptivePhaseController =
            serde_json::from_str(&serialized).expect("Deserialization failed");

        assert_eq!(deserialized.current_phase(), controller.current_phase());
        assert_eq!(
            deserialized.current_predict_steps(),
            controller.current_predict_steps()
        );
    }
}
