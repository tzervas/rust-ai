//! Early stopping for training with patience and delta thresholds.
//!
//! Monitors a metric (loss or accuracy) and stops training when no improvement
//! is observed for a specified number of steps (patience).
//!
//! # Example
//!
//! ```
//! use training_tools::early_stopping::{EarlyStopping, StoppingMode, StoppingDecision};
//!
//! let mut early_stop = EarlyStopping::new(10, 0.001)
//!     .with_mode(StoppingMode::MinLoss);
//!
//! for step in 0..1000 {
//!     let loss = train_step();
//!
//!     match early_stop.check(loss, step) {
//!         StoppingDecision::NewBest => {
//!             println!("New best loss: {}", loss);
//!             save_checkpoint();
//!         }
//!         StoppingDecision::NoImprovement { count, remaining } => {
//!             println!("No improvement for {} steps, {} remaining", count, remaining);
//!         }
//!         StoppingDecision::Stop => {
//!             println!("Early stopping at step {}", step);
//!             restore_best_checkpoint();
//!             break;
//!         }
//!         StoppingDecision::Continue => {}
//!     }
//! }
//! ```

/// Early stopping state tracker.
#[derive(Debug, Clone)]
pub struct EarlyStopping {
    /// Number of steps with no improvement before stopping.
    patience: usize,
    /// Minimum change to qualify as improvement.
    min_delta: f32,
    /// Whether to minimize (loss) or maximize (accuracy) the metric.
    mode: StoppingMode,
    /// Best metric value observed so far.
    best_value: f32,
    /// Step at which best value was observed.
    best_step: u64,
    /// Number of consecutive steps with no improvement.
    counter: usize,
    /// Whether stopping criterion has been met.
    stopped: bool,
}

/// Metric optimization mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StoppingMode {
    /// Stop when loss stops decreasing (minimize metric).
    MinLoss,
    /// Stop when metric stops increasing (e.g., accuracy, F1 score).
    MaxMetric,
}

/// Result of checking a new metric value.
#[derive(Debug, Clone, PartialEq)]
pub enum StoppingDecision {
    /// Continue training, no significant change.
    Continue,
    /// New best value achieved, should save checkpoint.
    NewBest,
    /// No improvement, but patience not exhausted.
    NoImprovement {
        /// Number of steps without improvement.
        count: usize,
        /// Steps remaining before stopping.
        remaining: usize,
    },
    /// Patience exhausted, should stop training.
    Stop,
}

impl EarlyStopping {
    /// Create a new early stopping tracker with default MinLoss mode.
    ///
    /// # Arguments
    ///
    /// * `patience` - Number of steps with no improvement before stopping
    /// * `min_delta` - Minimum change to qualify as improvement (absolute value)
    ///
    /// # Example
    ///
    /// ```
    /// use training_tools::early_stopping::EarlyStopping;
    ///
    /// // Stop after 10 steps with no improvement > 0.001
    /// let early_stop = EarlyStopping::new(10, 0.001);
    /// ```
    pub fn new(patience: usize, min_delta: f32) -> Self {
        Self {
            patience,
            min_delta,
            mode: StoppingMode::MinLoss,
            best_value: f32::INFINITY,
            best_step: 0,
            counter: 0,
            stopped: false,
        }
    }

    /// Set the optimization mode (builder pattern).
    ///
    /// # Example
    ///
    /// ```
    /// use training_tools::early_stopping::{EarlyStopping, StoppingMode};
    ///
    /// let early_stop = EarlyStopping::new(10, 0.01)
    ///     .with_mode(StoppingMode::MaxMetric);
    /// ```
    pub fn with_mode(mut self, mode: StoppingMode) -> Self {
        self.mode = mode;
        // Initialize best_value based on mode
        self.best_value = match mode {
            StoppingMode::MinLoss => f32::INFINITY,
            StoppingMode::MaxMetric => f32::NEG_INFINITY,
        };
        self
    }

    /// Check a new metric value and update internal state.
    ///
    /// Returns a decision indicating whether to continue, save checkpoint, or stop.
    ///
    /// # Arguments
    ///
    /// * `value` - Current metric value (loss or accuracy)
    /// * `step` - Current training step
    ///
    /// # Example
    ///
    /// ```
    /// use training_tools::early_stopping::{EarlyStopping, StoppingDecision};
    ///
    /// let mut early_stop = EarlyStopping::new(5, 0.001);
    ///
    /// match early_stop.check(0.5, 100) {
    ///     StoppingDecision::NewBest => println!("Save checkpoint"),
    ///     StoppingDecision::Stop => println!("Stop training"),
    ///     _ => {}
    /// }
    /// ```
    pub fn check(&mut self, value: f32, step: u64) -> StoppingDecision {
        if self.stopped {
            return StoppingDecision::Stop;
        }

        let improved = match self.mode {
            StoppingMode::MinLoss => {
                // For minimization: value must be lower by at least min_delta
                self.best_value - value > self.min_delta
            }
            StoppingMode::MaxMetric => {
                // For maximization: value must be higher by at least min_delta
                value - self.best_value > self.min_delta
            }
        };

        if improved {
            // New best value found
            self.best_value = value;
            self.best_step = step;
            self.counter = 0;
            StoppingDecision::NewBest
        } else {
            // No improvement
            self.counter += 1;

            if self.counter >= self.patience {
                // Patience exhausted
                self.stopped = true;
                StoppingDecision::Stop
            } else {
                // Still have patience remaining
                StoppingDecision::NoImprovement {
                    count: self.counter,
                    remaining: self.patience - self.counter,
                }
            }
        }
    }

    /// Check if training should stop (patience exhausted).
    pub fn should_stop(&self) -> bool {
        self.stopped
    }

    /// Get the step at which the best value was observed.
    pub fn best_step(&self) -> u64 {
        self.best_step
    }

    /// Get the best value observed so far.
    pub fn best_value(&self) -> f32 {
        self.best_value
    }

    /// Get current patience counter (steps without improvement).
    pub fn counter(&self) -> usize {
        self.counter
    }

    /// Get remaining patience (steps before stopping).
    pub fn remaining_patience(&self) -> usize {
        if self.stopped {
            0
        } else {
            self.patience.saturating_sub(self.counter)
        }
    }

    /// Reset the early stopping state (useful when resuming training).
    pub fn reset(&mut self) {
        self.best_value = match self.mode {
            StoppingMode::MinLoss => f32::INFINITY,
            StoppingMode::MaxMetric => f32::NEG_INFINITY,
        };
        self.best_step = 0;
        self.counter = 0;
        self.stopped = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_min_loss_new_best() {
        let mut early_stop = EarlyStopping::new(3, 0.01);

        // First value is always best
        assert_eq!(early_stop.check(1.0, 0), StoppingDecision::NewBest);
        assert_eq!(early_stop.best_value(), 1.0);
        assert_eq!(early_stop.best_step(), 0);

        // Improvement by more than min_delta
        assert_eq!(early_stop.check(0.98, 1), StoppingDecision::NewBest);
        assert_eq!(early_stop.best_value(), 0.98);
        assert_eq!(early_stop.counter(), 0);
    }

    #[test]
    fn test_min_loss_patience() {
        let mut early_stop = EarlyStopping::new(3, 0.01);

        early_stop.check(1.0, 0); // best
        early_stop.check(1.0, 1); // no improvement
        early_stop.check(1.0, 2); // no improvement

        let decision = early_stop.check(1.0, 3); // no improvement, patience exhausted
        assert_eq!(decision, StoppingDecision::Stop);
        assert!(early_stop.should_stop());
    }

    #[test]
    fn test_min_loss_no_improvement() {
        let mut early_stop = EarlyStopping::new(5, 0.01);

        early_stop.check(1.0, 0); // best

        match early_stop.check(1.0, 1) {
            StoppingDecision::NoImprovement { count, remaining } => {
                assert_eq!(count, 1);
                assert_eq!(remaining, 4);
            }
            _ => panic!("Expected NoImprovement"),
        }

        match early_stop.check(0.999, 2) {
            // 1.0 - 0.999 = 0.001 < min_delta (0.01), so no improvement
            StoppingDecision::NoImprovement { count, remaining } => {
                assert_eq!(count, 2);
                assert_eq!(remaining, 3);
            }
            _ => panic!("Expected NoImprovement"),
        }
    }

    #[test]
    fn test_max_metric_mode() {
        let mut early_stop = EarlyStopping::new(3, 0.01).with_mode(StoppingMode::MaxMetric);

        // First value is always best
        assert_eq!(early_stop.check(0.5, 0), StoppingDecision::NewBest);

        // Higher value is better
        assert_eq!(early_stop.check(0.52, 1), StoppingDecision::NewBest);
        assert_eq!(early_stop.best_value(), 0.52);

        // Lower value is worse
        match early_stop.check(0.51, 2) {
            StoppingDecision::NoImprovement { count, remaining } => {
                assert_eq!(count, 1);
                assert_eq!(remaining, 2);
            }
            _ => panic!("Expected NoImprovement"),
        }
    }

    #[test]
    fn test_reset() {
        let mut early_stop = EarlyStopping::new(2, 0.01);

        early_stop.check(1.0, 0);
        early_stop.check(1.0, 1);
        early_stop.check(1.0, 2); // stops

        assert!(early_stop.should_stop());

        early_stop.reset();

        assert!(!early_stop.should_stop());
        assert_eq!(early_stop.counter(), 0);
        assert_eq!(early_stop.best_value(), f32::INFINITY);
    }

    #[test]
    fn test_min_delta_threshold() {
        let mut early_stop = EarlyStopping::new(3, 0.1);

        early_stop.check(1.0, 0); // best

        // Improvement of 0.05 < 0.1, so no improvement
        match early_stop.check(0.95, 1) {
            StoppingDecision::NoImprovement { .. } => {}
            _ => panic!("Expected NoImprovement"),
        }

        // Improvement of 0.11 > 0.1, so new best
        assert_eq!(early_stop.check(0.89, 2), StoppingDecision::NewBest);
    }

    #[test]
    fn test_remaining_patience() {
        let mut early_stop = EarlyStopping::new(5, 0.01);

        early_stop.check(1.0, 0);
        assert_eq!(early_stop.remaining_patience(), 5);

        early_stop.check(1.0, 1);
        assert_eq!(early_stop.remaining_patience(), 4);

        early_stop.check(1.0, 2);
        assert_eq!(early_stop.remaining_patience(), 3);

        early_stop.check(0.98, 3); // new best
        assert_eq!(early_stop.remaining_patience(), 5);
    }
}
