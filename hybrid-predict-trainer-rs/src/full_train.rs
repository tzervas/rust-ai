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
}
