//! Plateau detection and warmup restart logic for training optimization.
//!
//! This module provides mechanisms to detect when training has plateaued
//! (stopped making meaningful progress) and trigger warmup restarts to
//! escape local minima or reset learning dynamics.
//!
//! # Overview
//!
//! The plateau detector monitors loss dynamics through:
//! - **Velocity**: Rate of change of loss (dLoss/dStep via linear regression)
//! - **Acceleration**: Rate of change of velocity (dVelocity/dStep)
//! - **Normalized Slope**: Velocity relative to mean loss magnitude
//!
//! Based on these metrics, it classifies training into one of four statuses:
//! - **Normal**: Making progress, continue training
//! - **Monitoring**: Potential plateau detected, watch carefully
//! - **Stuck**: Clear plateau with poor progress, may need restart
//! - **`HealthyConvergence`**: Plateau at low loss (converged successfully)
//!
//! # Warmup Restart Strategy
//!
//! When training is stuck:
//! 1. Increment warmup restart counter
//! 2. Apply learning rate multiplier (default 1.5x) to escape basin
//! 3. Enforce cooldown period (500 steps) before next restart
//! 4. Cap total restarts at 3 to avoid infinite loops
//!
//! # Example
//!
//! ```rust,no_run
//! use hybrid_predict_trainer_rs::auto_tuning::PlateauDetector;
//!
//! let mut detector = PlateauDetector::new(50, 0.001);
//!
//! // In your training loop:
//! for step in 0..1000 {
//!     let loss = 1.0 / (step as f32 + 1.0); // Example decreasing loss
//!     let progress_pct = (step as f32 / 1000.0) * 100.0;
//!
//!     let status = detector.update(loss, step as u64, progress_pct);
//!
//!     if let Some(lr_multiplier) = detector.should_warmup_restart(step as u64) {
//!         println!("Warmup restart! Multiply LR by {}", lr_multiplier);
//!         // Apply lr_multiplier to your optimizer's learning rate
//!     }
//! }
//! ```

use serde::{Deserialize, Serialize};

/// Status of training with respect to plateauing.
///
/// Used to communicate the current training regime and decision points
/// to the hybrid trainer and external monitoring systems.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PlateauStatus {
    /// Training is progressing normally.
    ///
    /// Loss is decreasing at a healthy rate relative to its magnitude.
    /// No intervention needed.
    Normal,

    /// Potential plateau detected, monitoring closely.
    ///
    /// Loss dynamics show signs of slowing but not yet conclusive.
    /// Increased logging/monitoring recommended.
    Monitoring,

    /// Clear plateau detected with insufficient progress.
    ///
    /// Training has stalled significantly despite being early/mid-training.
    /// Warmup restart or learning rate adjustment recommended.
    Stuck,

    /// Plateau at good convergence.
    ///
    /// Loss has plateaued but at a sufficiently low value (based on training
    /// progress percentage). This is a healthy convergence pattern.
    HealthyConvergence,
}

/// Loss dynamics computed from history.
///
/// Tracks velocity and acceleration of loss to detect plateaus and
/// convergence patterns. Uses a ring buffer for memory efficiency.
#[derive(Debug, Clone)]
pub struct LossDynamics {
    /// Ring buffer of loss values (max 256 entries).
    loss_history: Vec<f32>,
    /// Current position in ring buffer (for circular indexing).
    history_pos: usize,
    /// Whether the ring buffer is full.
    history_full: bool,

    /// Velocity of loss (dLoss/dStep).
    ///
    /// Computed via linear regression over the window.
    /// Negative velocity indicates decreasing loss (good).
    pub velocity: f32,

    /// Acceleration of velocity (dVelocity/dStep).
    ///
    /// Indicates whether velocity is increasing (becoming more positive,
    /// which means loss decrease is slowing).
    pub acceleration: f32,

    /// Normalized slope (velocity / `mean_loss`).
    ///
    /// Makes plateau detection scale-invariant. Used for threshold comparison.
    pub normalized_slope: f32,

    /// Window size for computing dynamics (default 50).
    ///
    /// Loss dynamics are computed over this many recent steps.
    pub window_size: usize,
}

impl LossDynamics {
    /// Creates a new loss dynamics tracker.
    ///
    /// # Arguments
    ///
    /// * `window_size` - Number of recent steps to use for dynamics computation.
    ///   Typical range: 30-100. Default: 50.
    pub fn new(window_size: usize) -> Self {
        let window_size = window_size.max(10).min(256); // Clamp to reasonable range
        Self {
            loss_history: Vec::with_capacity(256),
            history_pos: 0,
            history_full: false,
            velocity: 0.0,
            acceleration: 0.0,
            normalized_slope: 0.0,
            window_size,
        }
    }

    /// Adds a new loss value and recomputes dynamics.
    ///
    /// This should be called once per training step with the loss value.
    pub fn update(&mut self, loss: f32) {
        // Add to ring buffer
        if self.loss_history.len() < 256 {
            self.loss_history.push(loss);
        } else {
            self.loss_history[self.history_pos] = loss;
            self.history_pos = (self.history_pos + 1) % 256;
            self.history_full = true;
        }

        // Recompute dynamics
        self.compute_dynamics();
    }

    /// Computes velocity and acceleration from loss history.
    ///
    /// Velocity is computed via linear regression over `window_size` steps.
    /// Acceleration is computed by comparing two halves of the window.
    fn compute_dynamics(&mut self) {
        let history_len = self.loss_history.len();

        // Need at least 2 points to compute velocity
        if history_len < 2 {
            self.velocity = 0.0;
            self.acceleration = 0.0;
            self.normalized_slope = 0.0;
            return;
        }

        let window = history_len.min(self.window_size);

        // Get the window slice (most recent `window` items)
        let window_start = history_len.saturating_sub(window);

        let window_losses: Vec<f32> = self.loss_history[window_start..].to_vec();

        // Compute velocity via linear regression
        self.velocity = self.linear_regression_slope(&window_losses);

        // Compute acceleration via split-window method
        if window >= 4 {
            let mid = window / 2;
            let first_half_velocity = self.linear_regression_slope(&window_losses[..mid]);
            let second_half_velocity = self.linear_regression_slope(&window_losses[mid..]);
            self.acceleration = second_half_velocity - first_half_velocity;
        } else {
            self.acceleration = 0.0;
        }

        // Compute normalized slope
        let mean_loss = window_losses.iter().sum::<f32>() / window_losses.len() as f32;
        if mean_loss.abs() > 1e-6 {
            self.normalized_slope = self.velocity / mean_loss;
        } else {
            self.normalized_slope = 0.0;
        }
    }

    /// Computes the slope of loss via simple linear regression.
    ///
    /// Uses least-squares fitting: slope = sum((x - `mean_x`) * (y - `mean_y`)) / sum((x - `mean_x)^2`)
    fn linear_regression_slope(&self, losses: &[f32]) -> f32 {
        if losses.len() < 2 {
            return 0.0;
        }

        let n = losses.len() as f32;
        let mean_x = (losses.len() - 1) as f32 / 2.0;
        let mean_y = losses.iter().sum::<f32>() / n;

        let mut numerator = 0.0;
        let mut denominator = 0.0;

        for (i, &loss) in losses.iter().enumerate() {
            let x = i as f32;
            let dx = x - mean_x;
            let dy = loss - mean_y;
            numerator += dx * dy;
            denominator += dx * dx;
        }

        if denominator.abs() > 1e-6 {
            numerator / denominator
        } else {
            0.0
        }
    }

    /// Returns the number of loss samples in the history.
    pub fn history_len(&self) -> usize {
        self.loss_history.len()
    }

    /// Returns a reference to the loss history.
    pub fn history(&self) -> &[f32] {
        &self.loss_history
    }
}

/// Detects training plateaus and recommends warmup restarts.
///
/// Monitors loss dynamics and training progress to identify when training
/// has stalled, then suggests learning rate adjustments (warmup restarts)
/// to escape local minima.
#[derive(Debug, Clone)]
pub struct PlateauDetector {
    /// Loss dynamics tracker.
    dynamics: LossDynamics,

    /// Current status (Normal, Monitoring, Stuck, `HealthyConvergence`).
    pub status: PlateauStatus,

    /// Number of consecutive steps in plateau state.
    pub steps_in_plateau: u64,

    /// Number of warmup restarts performed.
    pub warmup_restarts: u32,

    /// Step number of the last warmup restart.
    pub last_restart_step: u64,

    /// Threshold for normalized slope to trigger plateau detection.
    ///
    /// If |`normalized_slope`| < this, training is considered plateaued.
    /// Default: 0.001 (0.1% loss change per step relative to mean loss)
    pub plateau_slope_threshold: f32,

    /// Progress percentage below which plateau is "stuck" (not healthy).
    ///
    /// If training progress < this % and plateau detected, status = Stuck.
    /// Default: 0.50 (50% through training)
    pub stuck_plateau_progress: f32,

    /// Progress percentage above which plateau is "healthy convergence".
    ///
    /// If training progress > this % and plateau detected, status = `HealthyConvergence`.
    /// Default: 0.75 (75% through training)
    pub convergence_plateau_progress: f32,
}

impl PlateauDetector {
    /// Creates a new plateau detector.
    ///
    /// # Arguments
    ///
    /// * `window_size` - Window for computing loss dynamics (default 50)
    /// * `plateau_slope_threshold` - Threshold for plateau detection (default 0.001)
    ///
    /// # Returns
    ///
    /// A new `PlateauDetector` with default thresholds.
    pub fn new(window_size: usize, plateau_slope_threshold: f32) -> Self {
        Self {
            dynamics: LossDynamics::new(window_size),
            status: PlateauStatus::Normal,
            steps_in_plateau: 0,
            warmup_restarts: 0,
            last_restart_step: 0,
            plateau_slope_threshold: plateau_slope_threshold.abs(),
            stuck_plateau_progress: 0.50,
            convergence_plateau_progress: 0.75,
        }
    }

    /// Updates the detector with a new loss observation.
    ///
    /// Should be called once per training step with the current loss.
    ///
    /// # Arguments
    ///
    /// * `loss` - The loss value for this step
    /// * `step` - The current step number
    /// * `progress_pct` - Training progress as percentage (0-100)
    ///
    /// # Returns
    ///
    /// The updated [`PlateauStatus`].
    pub fn update(&mut self, loss: f32, _step: u64, progress_pct: f32) -> PlateauStatus {
        // Update dynamics
        self.dynamics.update(loss);

        // Determine if we're in a plateau
        let is_plateau = self.dynamics.normalized_slope.abs() < self.plateau_slope_threshold
            && self.dynamics.history_len() >= self.dynamics.window_size;

        if is_plateau {
            self.steps_in_plateau += 1;
        } else {
            self.steps_in_plateau = 0;
        }

        // Update status based on plateau and progress
        self.status = if !is_plateau {
            PlateauStatus::Normal
        } else if self.steps_in_plateau < 10 {
            // Just entered plateau, monitor
            PlateauStatus::Monitoring
        } else {
            // Established plateau - classify by progress
            let progress = progress_pct / 100.0;
            if progress < self.stuck_plateau_progress {
                PlateauStatus::Stuck
            } else if progress > self.convergence_plateau_progress {
                PlateauStatus::HealthyConvergence
            } else {
                PlateauStatus::Monitoring
            }
        };

        self.status
    }

    /// Checks if a warmup restart should be triggered at this step.
    ///
    /// Returns the learning rate multiplier (e.g., 1.5) to apply if a restart
    /// is warranted, or None if no restart is needed.
    ///
    /// Restart conditions:
    /// 1. Status is Stuck
    /// 2. At least 100 steps in current plateau
    /// 3. At least 500 steps since last restart
    /// 4. Total restarts < 3
    ///
    /// # Arguments
    ///
    /// * `step` - The current training step
    ///
    /// # Returns
    ///
    /// `Some(lr_multiplier)` if restart should happen, `None` otherwise.
    pub fn should_warmup_restart(&mut self, step: u64) -> Option<f32> {
        // Check all conditions
        let is_stuck = self.status == PlateauStatus::Stuck;
        let steps_in_plateau_ok = self.steps_in_plateau >= 100;
        let cooldown_ok = step.saturating_sub(self.last_restart_step) >= 500;
        let restart_count_ok = self.warmup_restarts < 3;

        if is_stuck && steps_in_plateau_ok && cooldown_ok && restart_count_ok {
            // Trigger restart
            self.warmup_restarts += 1;
            self.last_restart_step = step;
            self.steps_in_plateau = 0; // Reset plateau counter
            Some(1.5) // Return learning rate multiplier
        } else {
            None
        }
    }

    /// Returns the current velocity (dLoss/dStep).
    #[must_use]
    pub fn velocity(&self) -> f32 {
        self.dynamics.velocity
    }

    /// Returns the current acceleration (dVelocity/dStep).
    #[must_use]
    pub fn acceleration(&self) -> f32 {
        self.dynamics.acceleration
    }

    /// Returns the current normalized slope (velocity / `mean_loss`).
    #[must_use]
    pub fn normalized_slope(&self) -> f32 {
        self.dynamics.normalized_slope
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plateau_status_basics() {
        let mut detector = PlateauDetector::new(10, 0.001);

        // Initial status should be Normal
        assert_eq!(detector.status, PlateauStatus::Normal);

        // Add some loss values showing clear decrease
        for i in 0..20 {
            let loss = 2.0 - (i as f32 * 0.05);
            let _status = detector.update(loss, i as u64, (i as f32 / 20.0) * 100.0);
        }

        // With decreasing loss, status should still be Normal
        assert_eq!(detector.status, PlateauStatus::Normal);
    }

    #[test]
    fn test_plateau_detection() {
        let mut detector = PlateauDetector::new(10, 0.01); // Higher threshold for easier detection in test

        // Add values to fill window - completely flat loss
        for i in 0..30 {
            let _loss = 1.0; // Completely flat - should detect plateau
            let status = detector.update(_loss, i as u64, 25.0);

            // By step 20 (after window fills + monitoring threshold), should detect plateau
            if i > 20 {
                assert!(
                    status == PlateauStatus::Monitoring || status == PlateauStatus::Stuck,
                    "Expected plateau detection at step {}, got {:?}",
                    i,
                    status
                );
            }
        }
    }

    #[test]
    fn test_stuck_vs_healthy_convergence() {
        let mut detector = PlateauDetector::new(10, 0.01); // Higher threshold

        // Scenario 1: Plateau at 25% progress -> Stuck
        for i in 0..25 {
            let _loss = 1.0; // Flat loss
            detector.update(_loss, i as u64, 25.0);
        }

        // After enough steps in plateau, should be Stuck at 25% progress
        assert_eq!(
            detector.status,
            PlateauStatus::Stuck,
            "Plateau at 25% progress should be Stuck, but got {:?}",
            detector.status
        );

        // Scenario 2: Plateau at 80% progress -> HealthyConvergence
        let mut detector2 = PlateauDetector::new(10, 0.01); // Higher threshold
        for i in 0..30 {
            let _loss = 0.5; // Flat loss at convergence
            detector2.update(_loss, i as u64, 80.0);
        }

        assert_eq!(
            detector2.status,
            PlateauStatus::HealthyConvergence,
            "Plateau at 80% progress should be HealthyConvergence, but got {:?}",
            detector2.status
        );
    }

    #[test]
    fn test_warmup_restart_conditions() {
        let mut detector = PlateauDetector::new(10, 0.01); // Higher threshold

        // Create a stuck plateau at early progress
        for i in 0..130 {
            let _loss = 1.0;
            let _status = detector.update(_loss, i as u64, 30.0);
        }

        assert_eq!(detector.status, PlateauStatus::Stuck);
        assert!(
            detector.steps_in_plateau >= 100,
            "Should have 100+ steps in plateau"
        );

        // First restart at step 500 (after 100 steps in plateau and cooldown satisfied)
        let restart = detector.should_warmup_restart(500);
        assert_eq!(restart, Some(1.5), "First restart should trigger");
        assert_eq!(detector.warmup_restarts, 1);
        assert_eq!(detector.last_restart_step, 500);

        // Immediately after restart, cooldown should prevent another
        let restart2 = detector.should_warmup_restart(501);
        assert_eq!(restart2, None, "Cooldown should prevent immediate restart");

        // After cooldown (500 steps from last restart), should allow another restart if stuck again
        for i in 130..650 {
            let _loss = 1.0;
            let _status = detector.update(_loss, i as u64, 40.0);
        }

        let restart3 = detector.should_warmup_restart(1000);
        assert_eq!(restart3, Some(1.5), "Restart after cooldown should trigger");
        assert_eq!(detector.warmup_restarts, 2);
    }

    #[test]
    fn test_warmup_restart_max_limit() {
        let mut detector = PlateauDetector::new(10, 0.001);
        detector.warmup_restarts = 3; // Already at max

        // Even if stuck and conditions met, should not allow 4th restart
        for i in 0..120 {
            let _loss = 1.0;
            let _status = detector.update(_loss, i as u64, 30.0);
        }

        let restart = detector.should_warmup_restart(600);
        assert_eq!(restart, None, "Should not allow more than 3 restarts");
    }

    #[test]
    fn test_loss_dynamics_velocity_and_acceleration() {
        let mut dynamics = LossDynamics::new(5);

        // Add linearly decreasing loss
        for i in 0..10 {
            dynamics.update(10.0 - (i as f32) * 0.5);
        }

        // Velocity should be negative (loss decreasing)
        assert!(
            dynamics.velocity < 0.0,
            "Velocity should be negative for decreasing loss"
        );

        // Acceleration should be near zero (constant velocity)
        assert!(
            dynamics.acceleration.abs() < 0.1,
            "Acceleration should be small for linear decrease"
        );
    }

    #[test]
    fn test_normalized_slope_invariance() {
        let mut dynamics1 = LossDynamics::new(5);
        let mut dynamics2 = LossDynamics::new(5);

        // Add same relative changes but different scales
        for i in 0..10 {
            dynamics1.update(10.0 - (i as f32) * 0.05); // Loss from 10 to 9.5
            dynamics2.update(100.0 - (i as f32) * 0.5); // Loss from 100 to 95
        }

        // Normalized slopes should be similar (scale-invariant)
        let ratio = dynamics1.normalized_slope.abs() / dynamics2.normalized_slope.abs();
        assert!(
            (ratio - 1.0).abs() < 0.1,
            "Normalized slopes should be similar: {:.4} vs {:.4}",
            dynamics1.normalized_slope,
            dynamics2.normalized_slope
        );
    }

    #[test]
    fn test_ring_buffer_wrapping() {
        let mut dynamics = LossDynamics::new(10);

        // Fill buffer beyond capacity to test ring wrapping
        for i in 0..500 {
            let loss = 1.0 + (i as f32 % 10.0) * 0.01;
            dynamics.update(loss);
        }

        // Buffer should still be 256 max
        assert_eq!(dynamics.history_len(), 256);

        // Dynamics should still be computable
        assert!(dynamics.velocity.is_finite());
        assert!(dynamics.normalized_slope.is_finite());
    }

    #[test]
    fn test_plateau_detector_reset_on_recovery() {
        let mut detector = PlateauDetector::new(10, 0.001);

        // Create initial plateau
        for i in 0..20 {
            let _loss = 1.0;
            let _status = detector.update(_loss, i as u64, 30.0);
        }

        let initial_steps = detector.steps_in_plateau;
        assert!(initial_steps > 0, "Should detect plateau");

        // Recover with decreasing loss
        for i in 20..30 {
            let loss = 1.0 - ((i - 20) as f32) * 0.05;
            let _status = detector.update(loss, i as u64, 40.0);
        }

        // After recovery, steps_in_plateau should reset
        assert!(
            detector.steps_in_plateau < 5,
            "Plateau counter should reset after recovery"
        );
    }

    #[test]
    fn test_edge_case_empty_history() {
        let mut dynamics = LossDynamics::new(10);

        // Velocity/acceleration should be 0 with empty history
        assert_eq!(dynamics.velocity, 0.0);
        assert_eq!(dynamics.acceleration, 0.0);
        assert_eq!(dynamics.normalized_slope, 0.0);

        // Add one value
        dynamics.update(1.0);
        assert_eq!(dynamics.velocity, 0.0, "One value should have 0 velocity");
    }
}
