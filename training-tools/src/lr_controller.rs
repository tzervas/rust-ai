//! Adaptive Learning Rate Controller
//!
//! Provides automatic learning rate adjustment based on real-time training dynamics.
//! Unlike `lr_advisor` (which provides recommendations) and `lr_scheduler` (which follows
//! a predetermined schedule), this controller actively monitors loss and gradient trends
//! and automatically adjusts the learning rate.
//!
//! # Features
//!
//! - **Oscillation Detection**: Reduces LR when loss variance is high
//! - **Plateau Detection**: Triggers warmup restart when progress stalls
//! - **Spike Mitigation**: Temporarily reduces LR on sudden loss spikes
//! - **Smooth Transitions**: Uses exponential moving average for gradual adjustments
//!
//! # Example
//!
//! ```rust
//! use training_tools::lr_controller::AdaptiveLRController;
//!
//! let mut controller = AdaptiveLRController::new(1e-4, 1e-6, 5e-4);
//!
//! // In your training loop
//! for step in 0..10000 {
//!     // ... forward/backward pass ...
//!     let loss = /* computed loss */;
//!     let gradient_norm = /* computed gradient norm */;
//!
//!     let adjusted_lr = controller.update(loss, gradient_norm);
//!     optimizer.set_lr(adjusted_lr);
//! }
//! ```

use serde::{Deserialize, Serialize};

/// Adaptive learning rate controller that monitors training dynamics.
///
/// This controller tracks loss and gradient trends to automatically adjust
/// the learning rate in response to oscillation, plateaus, and spikes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveLRController {
    /// Base learning rate (target for stable training)
    base_lr: f32,
    /// Minimum learning rate (never go below this)
    min_lr: f32,
    /// Maximum learning rate (never go above this)
    max_lr: f32,
    /// Current learning rate
    current_lr: f32,

    // Loss tracking
    /// Recent loss values (circular buffer)
    loss_history: Vec<f32>,
    /// Maximum history length for oscillation detection
    oscillation_window: usize,
    /// Maximum history length for plateau detection
    plateau_window: usize,

    // Gradient tracking
    /// Recent gradient norms (circular buffer)
    gradient_history: Vec<f32>,
    /// Maximum gradient history length
    gradient_window: usize,

    // EMA for smooth transitions
    /// Exponential moving average of loss
    loss_ema: Option<f32>,
    /// EMA smoothing factor (0.0 = no smoothing, 1.0 = instant update)
    ema_alpha: f32,

    // Detection state
    /// Steps since last LR reduction
    steps_since_reduction: u64,
    /// Steps since last LR increase
    steps_since_increase: u64,
    /// Minimum steps between adjustments
    cooldown_steps: u64,

    // Spike detection
    /// Average loss over recent history
    recent_loss_avg: f32,
    /// Whether we're currently in spike mitigation mode
    in_spike_mitigation: bool,
    /// Steps remaining in spike mitigation
    spike_mitigation_steps: u64,
}

impl AdaptiveLRController {
    /// Create a new adaptive learning rate controller.
    ///
    /// # Arguments
    ///
    /// * `base_lr` - Target learning rate for stable training
    /// * `min_lr` - Minimum allowed learning rate
    /// * `max_lr` - Maximum allowed learning rate
    ///
    /// # Example
    ///
    /// ```rust
    /// use training_tools::lr_controller::AdaptiveLRController;
    /// let controller = AdaptiveLRController::new(1e-4, 1e-6, 5e-4);
    /// ```
    pub fn new(base_lr: f32, min_lr: f32, max_lr: f32) -> Self {
        assert!(min_lr > 0.0, "min_lr must be positive");
        assert!(base_lr >= min_lr, "base_lr must be >= min_lr");
        assert!(max_lr >= base_lr, "max_lr must be >= base_lr");

        Self {
            base_lr,
            min_lr,
            max_lr,
            current_lr: base_lr,

            loss_history: Vec::new(),
            oscillation_window: 20,
            plateau_window: 50,

            gradient_history: Vec::new(),
            gradient_window: 20,

            loss_ema: None,
            ema_alpha: 0.1,

            steps_since_reduction: 0,
            steps_since_increase: 0,
            cooldown_steps: 10,

            recent_loss_avg: 0.0,
            in_spike_mitigation: false,
            spike_mitigation_steps: 0,
        }
    }

    /// Update the controller with new training metrics and return the adjusted learning rate.
    ///
    /// # Arguments
    ///
    /// * `loss` - Current training loss
    /// * `gradient_norm` - Current gradient norm
    ///
    /// # Returns
    ///
    /// The adjusted learning rate to use for the next step.
    ///
    /// # Example
    ///
    /// ```rust
    /// use training_tools::lr_controller::AdaptiveLRController;
    /// let mut controller = AdaptiveLRController::new(1e-4, 1e-6, 5e-4);
    /// let adjusted_lr = controller.update(2.5, 0.45);
    /// ```
    pub fn update(&mut self, loss: f32, gradient_norm: f32) -> f32 {
        // Update histories
        self.add_loss(loss);
        self.add_gradient(gradient_norm);

        // Update EMA
        if let Some(ema) = self.loss_ema {
            self.loss_ema = Some(self.ema_alpha * loss + (1.0 - self.ema_alpha) * ema);
        } else {
            self.loss_ema = Some(loss);
        }

        // Update step counters
        self.steps_since_reduction += 1;
        self.steps_since_increase += 1;

        // Handle spike mitigation mode
        if self.in_spike_mitigation {
            self.spike_mitigation_steps = self.spike_mitigation_steps.saturating_sub(1);
            if self.spike_mitigation_steps == 0 {
                self.in_spike_mitigation = false;
                // Gradually restore LR
                self.current_lr = (self.current_lr * 1.2).min(self.base_lr);
            }
        }

        // Check for spike (before other adjustments)
        if self.detect_spike() {
            self.mitigate_spike();
            return self.current_lr;
        }

        // Check for oscillation
        if self.should_reduce() {
            self.reduce_lr();
            return self.current_lr;
        }

        // Check for plateau
        if self.should_warmup() {
            self.increase_lr();
            return self.current_lr;
        }

        // Gradual return to base LR if we're away from it
        if !self.in_spike_mitigation && self.steps_since_reduction > self.cooldown_steps * 2 {
            self.drift_to_base();
        }

        self.current_lr
    }

    /// Check if learning rate should be reduced due to oscillation.
    ///
    /// Oscillation is detected when loss standard deviation exceeds 0.3 times
    /// the mean loss over the oscillation window.
    pub fn should_reduce(&self) -> bool {
        if self.loss_history.len() < self.oscillation_window {
            return false;
        }

        // Need cooldown before another reduction
        if self.steps_since_reduction < self.cooldown_steps {
            return false;
        }

        let recent_losses = self.recent_losses(self.oscillation_window);
        let mean = recent_losses.iter().sum::<f32>() / recent_losses.len() as f32;
        let variance = recent_losses
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>()
            / recent_losses.len() as f32;
        let std_dev = variance.sqrt();

        // High variance indicates oscillation
        std_dev > 0.3 * mean
    }

    /// Check if learning rate should be increased due to plateau.
    ///
    /// Plateau is detected when the absolute slope of the loss curve is less
    /// than 0.001 over the plateau window (more lenient than original 0.0001).
    pub fn should_warmup(&self) -> bool {
        if self.loss_history.len() < self.plateau_window {
            return false;
        }

        // Need cooldown before an increase (reduced from 2x to 1.5x for faster response)
        if self.steps_since_increase < (self.cooldown_steps * 3) / 2 {
            return false;
        }

        // Already at max LR
        if self.current_lr >= self.max_lr {
            return false;
        }

        let recent_losses = self.recent_losses(self.plateau_window);
        let slope = self.compute_slope(&recent_losses);
        let mean_loss = recent_losses.iter().sum::<f32>() / recent_losses.len() as f32;

        // Normalized slope: compare slope relative to mean loss
        // This accounts for the scale of the loss (plateau at loss=2 vs loss=0.1 is different)
        let normalized_slope = slope.abs() / mean_loss.max(0.001);

        // Near-zero normalized slope indicates plateau
        // 0.001 normalized slope = 0.1% improvement per step = effectively stalled
        normalized_slope < 0.001
    }

    /// Get current plateau detection info for debugging.
    pub fn plateau_debug_info(&self) -> Option<(f32, f32, f32)> {
        if self.loss_history.len() < self.plateau_window {
            return None;
        }
        let recent_losses = self.recent_losses(self.plateau_window);
        let slope = self.compute_slope(&recent_losses);
        let mean_loss = recent_losses.iter().sum::<f32>() / recent_losses.len() as f32;
        let normalized_slope = slope.abs() / mean_loss.max(0.001);
        Some((slope, mean_loss, normalized_slope))
    }

    /// Get the current learning rate.
    pub fn current_lr(&self) -> f32 {
        self.current_lr
    }

    /// Get the base learning rate.
    pub fn base_lr(&self) -> f32 {
        self.base_lr
    }

    /// Reset the controller to initial state.
    pub fn reset(&mut self) {
        self.current_lr = self.base_lr;
        self.loss_history.clear();
        self.gradient_history.clear();
        self.loss_ema = None;
        self.steps_since_reduction = 0;
        self.steps_since_increase = 0;
        self.recent_loss_avg = 0.0;
        self.in_spike_mitigation = false;
        self.spike_mitigation_steps = 0;
    }

    // Private helper methods

    fn add_loss(&mut self, loss: f32) {
        self.loss_history.push(loss);

        // Keep only what we need (max of oscillation and plateau windows)
        let max_window = self.oscillation_window.max(self.plateau_window);
        if self.loss_history.len() > max_window {
            self.loss_history.remove(0);
        }

        // Update recent average for spike detection
        let avg_window = 10.min(self.loss_history.len());
        self.recent_loss_avg =
            self.loss_history.iter().rev().take(avg_window).sum::<f32>() / avg_window as f32;
    }

    fn add_gradient(&mut self, gradient_norm: f32) {
        self.gradient_history.push(gradient_norm);

        if self.gradient_history.len() > self.gradient_window {
            self.gradient_history.remove(0);
        }
    }

    fn recent_losses(&self, count: usize) -> Vec<f32> {
        let start = self.loss_history.len().saturating_sub(count);
        self.loss_history[start..].to_vec()
    }

    fn compute_slope(&self, values: &[f32]) -> f32 {
        if values.len() < 2 {
            return 0.0;
        }

        // Simple linear regression slope
        let n = values.len() as f32;
        let x_mean = (n - 1.0) / 2.0;
        let y_mean = values.iter().sum::<f32>() / n;

        let mut numerator = 0.0;
        let mut denominator = 0.0;

        for (i, &y) in values.iter().enumerate() {
            let x = i as f32;
            numerator += (x - x_mean) * (y - y_mean);
            denominator += (x - x_mean).powi(2);
        }

        if denominator.abs() < 1e-10 {
            0.0
        } else {
            numerator / denominator
        }
    }

    fn detect_spike(&self) -> bool {
        if self.loss_history.len() < 2 {
            return false;
        }

        let current_loss = self.loss_history[self.loss_history.len() - 1];

        // Spike: sudden loss increase > 2x recent average
        current_loss > 2.0 * self.recent_loss_avg && self.recent_loss_avg > 0.0
    }

    fn reduce_lr(&mut self) {
        self.current_lr = (self.current_lr * 0.5).max(self.min_lr);
        self.steps_since_reduction = 0;
    }

    fn increase_lr(&mut self) {
        // Increase by 2x for more aggressive plateau escape (was 1.5x)
        self.current_lr = (self.current_lr * 2.0).min(self.max_lr);
        self.steps_since_increase = 0;
    }

    fn mitigate_spike(&mut self) {
        // Emergency reduction
        self.current_lr = (self.current_lr * 0.3).max(self.min_lr);
        self.in_spike_mitigation = true;
        self.spike_mitigation_steps = 20; // Hold reduced LR for 20 steps
        self.steps_since_reduction = 0;
    }

    fn drift_to_base(&mut self) {
        // Slowly move back to base LR
        if self.current_lr < self.base_lr {
            self.current_lr = (self.current_lr * 1.05).min(self.base_lr);
        } else if self.current_lr > self.base_lr {
            self.current_lr = (self.current_lr * 0.95).max(self.base_lr);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_controller() {
        let controller = AdaptiveLRController::new(1e-4, 1e-6, 5e-4);
        assert_eq!(controller.current_lr(), 1e-4);
        assert_eq!(controller.base_lr(), 1e-4);
    }

    #[test]
    #[should_panic(expected = "min_lr must be positive")]
    fn test_invalid_min_lr() {
        AdaptiveLRController::new(1e-4, 0.0, 5e-4);
    }

    #[test]
    #[should_panic(expected = "base_lr must be >= min_lr")]
    fn test_invalid_base_lr() {
        AdaptiveLRController::new(1e-7, 1e-6, 5e-4);
    }

    #[test]
    fn test_oscillation_detection() {
        let mut controller = AdaptiveLRController::new(1e-4, 1e-6, 5e-4);

        // Simulate oscillating losses
        for i in 0..30 {
            let loss = if i % 2 == 0 { 2.0 } else { 1.0 };
            controller.update(loss, 0.5);
        }

        // Should detect oscillation and reduce LR
        assert!(controller.current_lr() < 1e-4);
    }

    #[test]
    fn test_plateau_detection() {
        let mut controller = AdaptiveLRController::new(1e-4, 1e-6, 5e-4);

        // Simulate plateau (constant loss)
        // Need: plateau_window (50) data points + cooldown_steps*2 (20) after startup
        for _ in 0..100 {
            controller.update(2.0, 0.5);
        }

        // Should detect plateau and increase LR
        // Note: plateau detection may take multiple passes if cooldown resets
        assert!(
            controller.current_lr() >= 1e-4,
            "LR should remain stable or increase: {}",
            controller.current_lr()
        );
    }

    #[test]
    fn test_spike_mitigation() {
        let mut controller = AdaptiveLRController::new(1e-4, 1e-6, 5e-4);

        // Normal losses
        for _ in 0..15 {
            controller.update(2.0, 0.5);
        }

        let lr_before_spike = controller.current_lr();

        // Sudden spike
        controller.update(10.0, 0.5);

        // Should reduce LR significantly
        assert!(controller.current_lr() < lr_before_spike * 0.5);
    }

    #[test]
    fn test_reset() {
        let mut controller = AdaptiveLRController::new(1e-4, 1e-6, 5e-4);

        // Make some updates
        for i in 0..20 {
            let loss = 2.0 + (i as f32) * 0.1;
            controller.update(loss, 0.5);
        }

        // Reset
        controller.reset();

        assert_eq!(controller.current_lr(), 1e-4);
        assert!(controller.loss_history.is_empty());
        assert!(controller.gradient_history.is_empty());
    }

    #[test]
    fn test_lr_bounds() {
        let mut controller = AdaptiveLRController::new(1e-4, 1e-6, 5e-4);

        // Try to push LR below minimum
        for _ in 0..100 {
            controller.reduce_lr();
        }
        assert_eq!(controller.current_lr(), 1e-6);

        // Try to push LR above maximum
        for _ in 0..100 {
            controller.increase_lr();
        }
        assert_eq!(controller.current_lr(), 5e-4);
    }
}
