//! Full training phase implementation.
//!
//! The full training phase executes traditional forward and backward passes,
//! computing actual gradients for model updates. This phase serves multiple
//! purposes:
//!
//! 1. **Ground Truth**: Provides actual gradient information for predictor training
//! 2. **Residual Collection**: Extracts residuals for correction phase
//! 3. **Validation**: Validates predictor accuracy against real gradients
//!
//! # When to Use Full Training
//!
//! Full training is used when:
//! - Predictor confidence is below threshold
//! - After divergence detection requires re-stabilization
//! - Periodically to update predictor and collect residuals
//!
//! # Memory Considerations
//!
//! Full training requires storing activations for the backward pass,
//! which can be memory-intensive for large models. Consider gradient
//! checkpointing for memory-constrained scenarios.

use crate::config::HybridTrainerConfig;
use crate::error::HybridResult;
use crate::phases::PhaseOutcome;
use crate::state::TrainingState;
use crate::Phase;

/// Statistics collected during full training phase.
#[derive(Debug, Clone, Default)]
pub struct FullTrainStatistics {
    /// Number of steps completed.
    pub steps_completed: usize,

    /// Running sum of loss values (for mean computation).
    pub loss_sum: f64,

    /// Sum of squared loss values (for variance).
    pub loss_sq_sum: f64,

    /// Running sum of gradient norms.
    pub gradient_norm_sum: f64,

    /// Sum of squared gradient norms.
    pub gradient_norm_sq_sum: f64,

    /// Maximum gradient norm observed.
    pub max_gradient_norm: f32,

    /// Minimum loss observed.
    pub min_loss: f32,

    /// Final loss at end of phase.
    pub final_loss: f32,

    /// Total wall-clock time in milliseconds.
    pub total_duration_ms: f64,

    /// Number of gradient clipping events.
    pub gradient_clip_count: usize,
}

impl FullTrainStatistics {
    /// Creates new statistics tracker.
    #[must_use]
    pub fn new() -> Self {
        Self {
            min_loss: f32::INFINITY,
            max_gradient_norm: 0.0,
            ..Default::default()
        }
    }

    /// Returns the average loss.
    #[must_use]
    pub fn average_loss(&self) -> f32 {
        if self.steps_completed == 0 {
            0.0
        } else {
            (self.loss_sum / self.steps_completed as f64) as f32
        }
    }

    /// Returns the loss standard deviation.
    #[must_use]
    pub fn loss_std(&self) -> f32 {
        if self.steps_completed < 2 {
            0.0
        } else {
            let n = self.steps_completed as f64;
            let mean = self.loss_sum / n;
            let variance = (self.loss_sq_sum / n) - (mean * mean);
            variance.max(0.0).sqrt() as f32
        }
    }

    /// Returns the average gradient norm.
    #[must_use]
    pub fn average_gradient_norm(&self) -> f32 {
        if self.steps_completed == 0 {
            0.0
        } else {
            (self.gradient_norm_sum / self.steps_completed as f64) as f32
        }
    }

    /// Updates statistics with a new step observation.
    pub fn record_step(&mut self, loss: f32, gradient_norm: f32, was_clipped: bool) {
        self.steps_completed += 1;

        let loss_f64 = f64::from(loss);
        self.loss_sum += loss_f64;
        self.loss_sq_sum += loss_f64 * loss_f64;

        let grad_f64 = f64::from(gradient_norm);
        self.gradient_norm_sum += grad_f64;
        self.gradient_norm_sq_sum += grad_f64 * grad_f64;

        if gradient_norm > self.max_gradient_norm {
            self.max_gradient_norm = gradient_norm;
        }

        if loss < self.min_loss {
            self.min_loss = loss;
        }

        self.final_loss = loss;

        if was_clipped {
            self.gradient_clip_count += 1;
        }
    }
}

/// Executor for the full training phase.
///
/// Manages full forward/backward pass execution, collecting statistics
/// and residuals for predictor training.
pub struct FullTrainExecutor {
    /// Target number of steps for this phase.
    target_steps: usize,

    /// Current step within phase.
    current_step: usize,

    /// Collected statistics.
    statistics: FullTrainStatistics,

    /// Maximum gradient norm for clipping.
    max_grad_norm: f32,

    /// Start time for duration tracking.
    start_time: Option<std::time::Instant>,

    /// Gradient information to send to predictor.
    gradient_observations: Vec<GradientObservation>,
}

/// Observation of gradient information for predictor training.
#[derive(Debug, Clone)]
pub struct GradientObservation {
    /// Training step.
    pub step: u64,

    /// Loss before update.
    pub loss_before: f32,

    /// Loss after update (if available).
    pub loss_after: Option<f32>,

    /// Global gradient norm.
    pub gradient_norm: f32,

    /// Per-layer gradient norms.
    pub layer_norms: Vec<(String, f32)>,

    /// Cosine similarity with previous gradient (if available).
    pub gradient_cosine: Option<f32>,

    /// Learning rate used.
    pub learning_rate: f32,
}

impl FullTrainExecutor {
    /// Creates a new full training executor.
    ///
    /// # Arguments
    ///
    /// * `config` - The hybrid trainer configuration
    /// * `steps` - Number of steps for this phase
    #[must_use]
    pub fn new(_config: &HybridTrainerConfig, steps: usize) -> Self {
        Self {
            target_steps: steps,
            current_step: 0,
            statistics: FullTrainStatistics::new(),
            max_grad_norm: 1.0, // Default max gradient norm
            start_time: None,
            gradient_observations: Vec::with_capacity(steps),
        }
    }

    /// Creates an executor with custom gradient clipping threshold.
    #[must_use]
    pub fn with_max_grad_norm(mut self, max_norm: f32) -> Self {
        self.max_grad_norm = max_norm;
        self
    }

    /// Returns whether the phase is complete.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.current_step >= self.target_steps
    }

    /// Returns the current progress as a fraction [0, 1].
    #[must_use]
    pub fn progress(&self) -> f32 {
        if self.target_steps == 0 {
            1.0
        } else {
            (self.current_step as f32) / (self.target_steps as f32)
        }
    }

    /// Returns the number of remaining steps.
    #[must_use]
    pub fn steps_remaining(&self) -> usize {
        self.target_steps.saturating_sub(self.current_step)
    }

    /// Returns a reference to collected statistics.
    #[must_use]
    pub fn statistics(&self) -> &FullTrainStatistics {
        &self.statistics
    }

    /// Returns gradient observations for predictor training.
    #[must_use]
    pub fn gradient_observations(&self) -> &[GradientObservation] {
        &self.gradient_observations
    }

    /// Records a full training step result.
    ///
    /// # Arguments
    ///
    /// * `loss` - The loss value
    /// * `gradient_norm` - The global gradient norm
    /// * `observation` - Detailed gradient observation for predictor
    pub fn record_step(
        &mut self,
        loss: f32,
        gradient_norm: f32,
        observation: Option<GradientObservation>,
    ) {
        if self.start_time.is_none() {
            self.start_time = Some(std::time::Instant::now());
        }

        let was_clipped = gradient_norm > self.max_grad_norm;
        self.statistics
            .record_step(loss, gradient_norm, was_clipped);

        if let Some(obs) = observation {
            self.gradient_observations.push(obs);
        }

        self.current_step += 1;
    }

    /// Finalizes the phase and returns the outcome.
    #[must_use]
    pub fn finalize(mut self) -> PhaseOutcome {
        let duration_ms = self
            .start_time
            .map_or(0.0, |t| t.elapsed().as_secs_f64() * 1000.0);
        self.statistics.total_duration_ms = duration_ms;

        PhaseOutcome {
            phase: Phase::Full,
            steps_executed: self.current_step,
            average_loss: self.statistics.average_loss(),
            final_loss: self.statistics.final_loss,
            completed_normally: self.is_complete(),
            early_termination_reason: if self.is_complete() {
                None
            } else {
                Some("Full training terminated early".to_string())
            },
            prediction_error: None,
            duration_ms,
        }
    }
}

/// Trait for full training phase executors.
///
/// Allows for different full training strategies (e.g., with or without
/// gradient accumulation, mixed precision, etc.).
pub trait FullTrainPhase: Send + Sync {
    /// Executes a single full training step.
    ///
    /// # Arguments
    ///
    /// * `state` - Current training state
    ///
    /// # Returns
    ///
    /// Result containing step metrics and gradient information.
    fn step(&mut self, state: &mut TrainingState) -> HybridResult<FullTrainStepResult>;

    /// Returns whether the phase is complete.
    fn is_complete(&self) -> bool;

    /// Returns collected statistics.
    fn statistics(&self) -> &FullTrainStatistics;

    /// Returns gradient observations for predictor.
    fn gradient_observations(&self) -> &[GradientObservation];
}

/// Result of a single full training step.
#[derive(Debug, Clone)]
pub struct FullTrainStepResult {
    /// Loss value.
    pub loss: f32,

    /// Gradient norm (before clipping).
    pub gradient_norm: f32,

    /// Gradient norm (after clipping, if applied).
    pub clipped_gradient_norm: f32,

    /// Whether gradient was clipped.
    pub was_clipped: bool,

    /// Detailed gradient observation.
    pub observation: Option<GradientObservation>,

    /// Step time in milliseconds.
    pub step_time_ms: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_statistics_tracking() {
        let mut stats = FullTrainStatistics::new();

        stats.record_step(3.0, 1.0, false);
        stats.record_step(2.5, 0.9, false);
        stats.record_step(2.0, 0.8, false);

        assert_eq!(stats.steps_completed, 3);
        assert!((stats.average_loss() - 2.5).abs() < 0.01);
        assert_eq!(stats.min_loss, 2.0);
        assert_eq!(stats.final_loss, 2.0);
    }

    #[test]
    fn test_executor_progress() {
        let config = HybridTrainerConfig::default();
        let mut executor = FullTrainExecutor::new(&config, 20);

        assert_eq!(executor.progress(), 0.0);
        assert!(!executor.is_complete());

        for _ in 0..10 {
            executor.record_step(2.5, 1.0, None);
        }

        assert!((executor.progress() - 0.5).abs() < 0.01);

        for _ in 0..10 {
            executor.record_step(2.0, 0.8, None);
        }

        assert!(executor.is_complete());
    }

    #[test]
    fn test_gradient_clipping_count() {
        let mut stats = FullTrainStatistics::new();

        stats.record_step(2.5, 0.5, false);
        stats.record_step(2.4, 1.5, true); // Clipped
        stats.record_step(2.3, 2.0, true); // Clipped

        assert_eq!(stats.gradient_clip_count, 2);
    }

    #[test]
    fn test_average_gradient_norm() {
        let mut stats = FullTrainStatistics::new();

        // Record steps with known gradient norms: 1.0, 2.0, 3.0, 4.0
        stats.record_step(1.0, 1.0, false);
        stats.record_step(1.0, 2.0, false);
        stats.record_step(1.0, 3.0, false);
        stats.record_step(1.0, 4.0, false);

        // Average = (1 + 2 + 3 + 4) / 4 = 2.5
        let avg = stats.average_gradient_norm();
        assert!(
            (avg - 2.5).abs() < 1e-6,
            "Expected average gradient norm 2.5, got {}",
            avg
        );

        // Edge case: no steps recorded
        let empty_stats = FullTrainStatistics::new();
        assert!((empty_stats.average_gradient_norm()).abs() < 1e-6);
    }

    #[test]
    fn test_loss_std_computation() {
        let mut stats = FullTrainStatistics::new();

        // Record steps with known loss values: all the same = 0 std
        stats.record_step(5.0, 1.0, false);
        stats.record_step(5.0, 1.0, false);
        stats.record_step(5.0, 1.0, false);

        assert!(
            stats.loss_std().abs() < 1e-6,
            "Std of constant values should be ~0, got {}",
            stats.loss_std()
        );

        // Now test with varying values: 1.0, 3.0
        // Mean = 2.0, population variance = ((1-2)^2 + (3-2)^2)/2 = 1.0, std = 1.0
        let mut stats2 = FullTrainStatistics::new();
        stats2.record_step(1.0, 1.0, false);
        stats2.record_step(3.0, 1.0, false);

        let std = stats2.loss_std();
        // loss_std uses population variance: (loss_sq_sum/n) - (mean*mean)
        // loss_sum = 4.0, loss_sq_sum = 1 + 9 = 10, n = 2
        // variance = 10/2 - (4/2)^2 = 5 - 4 = 1.0
        // std = 1.0
        assert!(
            (std - 1.0).abs() < 1e-5,
            "Expected loss std ~1.0, got {}",
            std
        );

        // Edge case: fewer than 2 steps
        let mut stats_one = FullTrainStatistics::new();
        stats_one.record_step(3.0, 1.0, false);
        assert!(
            (stats_one.loss_std()).abs() < 1e-6,
            "Std with 1 sample should be 0"
        );
    }

    #[test]
    fn test_gradient_observation_collection() {
        let config = HybridTrainerConfig::default();
        let mut executor = FullTrainExecutor::new(&config, 10);

        // Record steps without observations
        executor.record_step(2.5, 1.0, None);
        executor.record_step(2.4, 0.9, None);
        assert_eq!(executor.gradient_observations().len(), 0);

        // Record steps with observations
        let obs1 = GradientObservation {
            step: 3,
            loss_before: 2.3,
            loss_after: Some(2.2),
            gradient_norm: 0.85,
            layer_norms: vec![("layer1".to_string(), 0.5), ("layer2".to_string(), 0.35)],
            gradient_cosine: Some(0.95),
            learning_rate: 0.001,
        };
        let obs2 = GradientObservation {
            step: 4,
            loss_before: 2.2,
            loss_after: None,
            gradient_norm: 0.7,
            layer_norms: vec![],
            gradient_cosine: None,
            learning_rate: 0.001,
        };

        executor.record_step(2.3, 0.85, Some(obs1));
        executor.record_step(2.2, 0.7, Some(obs2));

        let observations = executor.gradient_observations();
        assert_eq!(observations.len(), 2);
        assert_eq!(observations[0].step, 3);
        assert_eq!(observations[1].step, 4);
        assert!((observations[0].gradient_norm - 0.85).abs() < 1e-6);
        assert!(observations[0].loss_after.is_some());
        assert!(observations[1].loss_after.is_none());
    }

    #[test]
    fn test_max_gradient_norm_tracking() {
        let mut stats = FullTrainStatistics::new();

        stats.record_step(2.0, 0.5, false);
        stats.record_step(2.0, 3.2, false);
        stats.record_step(2.0, 1.8, false);
        stats.record_step(2.0, 2.9, false);
        stats.record_step(2.0, 0.1, false);

        assert!(
            (stats.max_gradient_norm - 3.2).abs() < 1e-6,
            "Max gradient norm should be 3.2, got {}",
            stats.max_gradient_norm
        );

        // Adding a new max should update
        stats.record_step(2.0, 5.0, false);
        assert!(
            (stats.max_gradient_norm - 5.0).abs() < 1e-6,
            "Max gradient norm should update to 5.0, got {}",
            stats.max_gradient_norm
        );
    }

    #[test]
    fn test_loss_trajectory_min_max() {
        let mut stats = FullTrainStatistics::new();

        // Record decreasing losses
        stats.record_step(5.0, 1.0, false);
        stats.record_step(4.0, 1.0, false);
        stats.record_step(3.0, 1.0, false);
        stats.record_step(2.0, 1.0, false);
        stats.record_step(1.0, 1.0, false);

        assert!(
            (stats.min_loss - 1.0).abs() < 1e-6,
            "Min loss should be 1.0, got {}",
            stats.min_loss
        );
        assert!(
            (stats.final_loss - 1.0).abs() < 1e-6,
            "Final loss should be last recorded (1.0), got {}",
            stats.final_loss
        );

        // Average should be (5+4+3+2+1)/5 = 3.0
        assert!(
            (stats.average_loss() - 3.0).abs() < 1e-5,
            "Average loss should be 3.0, got {}",
            stats.average_loss()
        );

        // Now record a lower loss
        stats.record_step(0.5, 1.0, false);
        assert!(
            (stats.min_loss - 0.5).abs() < 1e-6,
            "Min loss should update to 0.5"
        );

        // Record a higher loss -- min should NOT change
        stats.record_step(2.5, 1.0, false);
        assert!(
            (stats.min_loss - 0.5).abs() < 1e-6,
            "Min loss should remain 0.5 after recording higher loss"
        );
        assert!(
            (stats.final_loss - 2.5).abs() < 1e-6,
            "Final loss should update to 2.5"
        );
    }

    #[test]
    fn test_zero_target_steps_edge_case() {
        let config = HybridTrainerConfig::default();
        let executor = FullTrainExecutor::new(&config, 0);

        // With 0 target steps, executor should be immediately complete
        assert!(executor.is_complete());

        // Progress should be 1.0 (not divide-by-zero)
        assert!(
            (executor.progress() - 1.0).abs() < 1e-6,
            "Progress with 0 target steps should be 1.0, got {}",
            executor.progress()
        );

        // Steps remaining should be 0
        assert_eq!(executor.steps_remaining(), 0);

        // Finalize should produce a valid outcome
        let outcome = executor.finalize();
        assert_eq!(outcome.phase, Phase::Full);
        assert_eq!(outcome.steps_executed, 0);
        assert!(outcome.completed_normally);
        assert!(outcome.early_termination_reason.is_none());
    }

    #[test]
    fn test_finalize_phase_outcome() {
        let config = HybridTrainerConfig::default();
        let mut executor = FullTrainExecutor::new(&config, 5);

        // Record all 5 steps with known values
        executor.record_step(3.0, 1.5, None);
        executor.record_step(2.8, 1.3, None);
        executor.record_step(2.6, 1.1, None);
        executor.record_step(2.4, 0.9, None);
        executor.record_step(2.2, 0.7, None);

        assert!(executor.is_complete());

        let outcome = executor.finalize();

        // Phase should be Full
        assert_eq!(outcome.phase, Phase::Full);

        // Steps executed should match
        assert_eq!(outcome.steps_executed, 5);

        // Average loss = (3.0 + 2.8 + 2.6 + 2.4 + 2.2) / 5 = 2.6
        assert!(
            (outcome.average_loss - 2.6).abs() < 1e-5,
            "Average loss should be 2.6, got {}",
            outcome.average_loss
        );

        // Final loss should be the last recorded
        assert!(
            (outcome.final_loss - 2.2).abs() < 1e-6,
            "Final loss should be 2.2, got {}",
            outcome.final_loss
        );

        // Should have completed normally
        assert!(outcome.completed_normally);
        assert!(outcome.early_termination_reason.is_none());

        // Prediction error should be None (this is Full phase, not Predict)
        assert!(outcome.prediction_error.is_none());

        // Duration should be non-negative (could be very small in tests)
        assert!(outcome.duration_ms >= 0.0);
    }

    #[test]
    fn test_finalize_early_termination() {
        let config = HybridTrainerConfig::default();
        let mut executor = FullTrainExecutor::new(&config, 10);

        // Only record 3 out of 10 steps
        executor.record_step(3.0, 1.5, None);
        executor.record_step(2.8, 1.3, None);
        executor.record_step(2.6, 1.1, None);

        assert!(!executor.is_complete());
        assert_eq!(executor.steps_remaining(), 7);

        let outcome = executor.finalize();

        assert_eq!(outcome.steps_executed, 3);
        assert!(!outcome.completed_normally);
        assert!(outcome.early_termination_reason.is_some());
        assert_eq!(
            outcome.early_termination_reason.as_deref(),
            Some("Full training terminated early")
        );
    }
}
