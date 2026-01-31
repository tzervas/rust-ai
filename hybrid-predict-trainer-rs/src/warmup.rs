//! Warmup phase implementation.
//!
//! The warmup phase establishes baseline training dynamics by executing
//! full forward and backward passes while the dynamics predictor observes
//! and learns the initial training behavior.
//!
//! # Purpose
//!
//! During warmup:
//! 1. The model trains normally with full gradient computation
//! 2. Loss and gradient statistics are collected to establish baselines
//! 3. The dynamics predictor accumulates training data
//! 4. K-FAC factors (if enabled) are initialized
//!
//! # When Warmup Ends
//!
//! Warmup completes after the configured number of steps, at which point
//! the predictor should have sufficient data to begin making predictions.
//! The transition to full training phase occurs automatically.
//!
//! # Example
//!
//! ```rust,ignore
//! use hybrid_predict_trainer_rs::warmup::WarmupExecutor;
//!
//! let executor = WarmupExecutor::new(&config);
//!
//! while !executor.is_complete() {
//!     let result = executor.step(&mut model, &mut optimizer, &batch)?;
//!     // Predictor observes the result
//! }
//! ```

use crate::config::HybridTrainerConfig;
use crate::error::HybridResult;
use crate::phases::PhaseOutcome;
use crate::state::TrainingState;
use crate::Phase;

/// Statistics collected during warmup for baseline establishment.
#[derive(Debug, Clone, Default)]
pub struct WarmupStatistics {
    /// Number of warmup steps completed.
    pub steps_completed: usize,

    /// Running mean of loss values.
    pub loss_mean: f64,

    /// Running variance of loss values (for std computation).
    pub loss_variance: f64,

    /// Running mean of gradient norms.
    pub gradient_norm_mean: f64,

    /// Running variance of gradient norms.
    pub gradient_norm_variance: f64,

    /// Initial loss (first step).
    pub initial_loss: f32,

    /// Final loss (last step).
    pub final_loss: f32,

    /// Loss improvement ratio (initial / final).
    pub loss_improvement_ratio: f32,

    /// Total wall-clock time in milliseconds.
    pub total_duration_ms: f64,
}

impl WarmupStatistics {
    /// Returns the loss standard deviation.
    #[must_use]
    pub fn loss_std(&self) -> f64 {
        self.loss_variance.sqrt()
    }

    /// Returns the gradient norm standard deviation.
    #[must_use]
    pub fn gradient_norm_std(&self) -> f64 {
        self.gradient_norm_variance.sqrt()
    }

    /// Updates running statistics with a new observation.
    ///
    /// Uses Welford's online algorithm for numerical stability.
    pub fn update(&mut self, loss: f32, gradient_norm: f32) {
        self.steps_completed += 1;
        let n = self.steps_completed as f64;

        // Update loss statistics
        let loss_f64 = f64::from(loss);
        let delta_loss = loss_f64 - self.loss_mean;
        self.loss_mean += delta_loss / n;
        let delta_loss2 = loss_f64 - self.loss_mean;
        self.loss_variance += delta_loss * delta_loss2;

        // Update gradient norm statistics
        let grad_f64 = f64::from(gradient_norm);
        let delta_grad = grad_f64 - self.gradient_norm_mean;
        self.gradient_norm_mean += delta_grad / n;
        let delta_grad2 = grad_f64 - self.gradient_norm_mean;
        self.gradient_norm_variance += delta_grad * delta_grad2;

        // Track initial and final loss
        if self.steps_completed == 1 {
            self.initial_loss = loss;
        }
        self.final_loss = loss;

        // Update improvement ratio
        if self.initial_loss > 0.0 && self.final_loss > 0.0 {
            self.loss_improvement_ratio = self.initial_loss / self.final_loss;
        }
    }

    /// Finalizes variance computation after all updates.
    ///
    /// Must be called after all `update()` calls to get correct
    /// variance/std values.
    pub fn finalize(&mut self) {
        if self.steps_completed > 1 {
            let n = self.steps_completed as f64;
            self.loss_variance /= n - 1.0;
            self.gradient_norm_variance /= n - 1.0;
        }
    }
}

/// Executor for the warmup phase.
///
/// Manages the warmup process, collecting statistics and coordinating
/// with the dynamics predictor for observation.
pub struct WarmupExecutor {
    /// Target number of warmup steps.
    target_steps: usize,

    /// Current step within warmup.
    current_step: usize,

    /// Collected statistics.
    statistics: WarmupStatistics,

    /// Start time for duration tracking.
    start_time: Option<std::time::Instant>,
}

impl WarmupExecutor {
    /// Creates a new warmup executor.
    ///
    /// # Arguments
    ///
    /// * `config` - The hybrid trainer configuration
    #[must_use]
    pub fn new(config: &HybridTrainerConfig) -> Self {
        Self {
            target_steps: config.warmup_steps,
            current_step: 0,
            statistics: WarmupStatistics::default(),
            start_time: None,
        }
    }

    /// Returns whether warmup is complete.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.current_step >= self.target_steps
    }

    /// Returns the current warmup progress as a fraction [0, 1].
    #[must_use]
    pub fn progress(&self) -> f32 {
        if self.target_steps == 0 {
            1.0
        } else {
            (self.current_step as f32) / (self.target_steps as f32)
        }
    }

    /// Returns the number of remaining warmup steps.
    #[must_use]
    pub fn steps_remaining(&self) -> usize {
        self.target_steps.saturating_sub(self.current_step)
    }

    /// Returns a reference to the collected statistics.
    #[must_use]
    pub fn statistics(&self) -> &WarmupStatistics {
        &self.statistics
    }

    /// Records a warmup step result.
    ///
    /// Should be called after each training step during warmup to update
    /// statistics and track progress.
    ///
    /// # Arguments
    ///
    /// * `loss` - The loss value for this step
    /// * `gradient_norm` - The global gradient norm for this step
    pub fn record_step(&mut self, loss: f32, gradient_norm: f32) {
        if self.start_time.is_none() {
            self.start_time = Some(std::time::Instant::now());
        }

        self.statistics.update(loss, gradient_norm);
        self.current_step += 1;
    }

    /// Finalizes the warmup phase and returns the outcome.
    ///
    /// Should be called when warmup completes to get the final statistics
    /// and prepare for transition to the full training phase.
    ///
    /// # Returns
    ///
    /// A `PhaseOutcome` summarizing the warmup phase.
    #[must_use]
    pub fn finalize(mut self) -> PhaseOutcome {
        self.statistics.finalize();

        let duration_ms = self
            .start_time
            .map_or(0.0, |t| t.elapsed().as_secs_f64() * 1000.0);
        self.statistics.total_duration_ms = duration_ms;

        PhaseOutcome {
            phase: Phase::Warmup,
            steps_executed: self.current_step,
            average_loss: self.statistics.loss_mean as f32,
            final_loss: self.statistics.final_loss,
            completed_normally: self.is_complete(),
            early_termination_reason: if self.is_complete() {
                None
            } else {
                Some("Warmup terminated early".to_string())
            },
            prediction_error: None,
            duration_ms,
        }
    }
}

/// Trait for warmup phase executors.
///
/// Allows for different warmup strategies (e.g., adaptive warmup
/// that terminates early if dynamics are stable).
pub trait WarmupPhase: Send + Sync {
    /// Executes a single warmup step.
    ///
    /// # Arguments
    ///
    /// * `state` - Current training state
    ///
    /// # Returns
    ///
    /// Updated state after the warmup step.
    fn step(&mut self, state: &mut TrainingState) -> HybridResult<WarmupStepResult>;

    /// Returns whether warmup is complete.
    fn is_complete(&self) -> bool;

    /// Returns warmup statistics.
    fn statistics(&self) -> &WarmupStatistics;
}

/// Result of a single warmup step.
#[derive(Debug, Clone)]
pub struct WarmupStepResult {
    /// Loss value for this step.
    pub loss: f32,

    /// Gradient norm for this step.
    pub gradient_norm: f32,

    /// Whether this was the final warmup step.
    pub warmup_complete: bool,

    /// Step time in milliseconds.
    pub step_time_ms: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_warmup_statistics_update() {
        let mut stats = WarmupStatistics::default();

        stats.update(3.0, 1.0);
        stats.update(2.5, 0.9);
        stats.update(2.0, 0.8);
        stats.finalize();

        assert_eq!(stats.steps_completed, 3);
        assert!((stats.loss_mean - 2.5).abs() < 0.01);
        assert_eq!(stats.initial_loss, 3.0);
        assert_eq!(stats.final_loss, 2.0);
    }

    #[test]
    fn test_warmup_executor_progress() {
        let config = HybridTrainerConfig {
            warmup_steps: 100,
            ..Default::default()
        };

        let mut executor = WarmupExecutor::new(&config);
        assert_eq!(executor.progress(), 0.0);
        assert!(!executor.is_complete());

        for i in 0..50 {
            executor.record_step(3.0 - i as f32 * 0.01, 1.0);
        }

        assert!((executor.progress() - 0.5).abs() < 0.01);
        assert!(!executor.is_complete());

        for i in 50..100 {
            executor.record_step(3.0 - i as f32 * 0.01, 1.0);
        }

        assert!(executor.is_complete());
    }

    #[test]
    fn test_warmup_finalize() {
        let config = HybridTrainerConfig {
            warmup_steps: 10,
            ..Default::default()
        };

        let mut executor = WarmupExecutor::new(&config);
        for _ in 0..10 {
            executor.record_step(2.5, 1.0);
        }

        let outcome = executor.finalize();
        assert_eq!(outcome.phase, Phase::Warmup);
        assert_eq!(outcome.steps_executed, 10);
        assert!(outcome.completed_normally);
    }
}
