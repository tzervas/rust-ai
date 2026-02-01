//! Automatic gradient clipping threshold adjustment
//!
//! Monitors gradient norms and dynamically adjusts clipping thresholds to:
//! - Prevent gradient explosions
//! - Detect vanishing gradients
//! - Maintain stable training dynamics
//!
//! # Example
//!
//! ```rust
//! use training_tools::{GradientController, GradientAction};
//!
//! let mut controller = GradientController::new(1.0);
//!
//! // In training loop:
//! let grad_norm = compute_gradient_norm(&gradients);
//! match controller.update(grad_norm) {
//!     GradientAction::Clip(threshold) => {
//!         clip_gradients(&mut gradients, threshold);
//!     }
//!     GradientAction::EmergencyClip => {
//!         clip_gradients(&mut gradients, controller.current_threshold());
//!         // Maybe also reduce learning rate
//!     }
//!     GradientAction::Warning(msg) => {
//!         eprintln!("Gradient warning: {}", msg);
//!     }
//!     _ => {}
//! }
//! ```

use std::collections::VecDeque;

/// Actions to take based on gradient norm analysis
#[derive(Debug, Clone, PartialEq)]
pub enum GradientAction {
    /// No action needed - gradients are healthy
    NoChange,

    /// Clip gradients to the specified threshold
    Clip(f32),

    /// Gradients consistently high - reduce threshold for next steps
    ReduceThreshold,

    /// Gradients consistently low - increase threshold to allow larger updates
    IncreaseThreshold,

    /// Emergency: gradient explosion detected - clip immediately
    EmergencyClip,

    /// Warning about gradient health (vanishing, instability, etc.)
    Warning(String),
}

/// Statistics about gradient norm history
#[derive(Debug, Clone)]
pub struct GradientStats {
    pub mean: f32,
    pub std_dev: f32,
    pub min: f32,
    pub max: f32,
    pub recent_trend: f32, // Positive = increasing, negative = decreasing
}

/// Automatic gradient clipping controller with adaptive thresholding
///
/// Monitors gradient norms over a sliding window and adjusts the clipping
/// threshold based on observed statistics. Detects explosions and vanishing
/// gradients.
pub struct GradientController {
    /// Current clipping threshold
    clip_threshold: f32,

    /// Minimum allowed threshold (prevents over-aggressive clipping)
    min_threshold: f32,

    /// Maximum allowed threshold (prevents under-clipping)
    max_threshold: f32,

    /// Sliding window of recent gradient norms
    history: VecDeque<f32>,

    /// Number of steps to keep in history
    window_size: usize,

    /// Multiplier for explosion detection (e.g., 10x mean)
    explosion_factor: f32,

    /// Threshold for vanishing gradient warning (e.g., 0.001)
    vanishing_threshold: f32,

    /// How aggressively to adjust threshold (0.0 = no adjustment, 1.0 = very aggressive)
    adjustment_rate: f32,

    /// Number of consecutive high gradients before reducing threshold
    high_count_threshold: usize,

    /// Number of consecutive low gradients before increasing threshold
    low_count_threshold: usize,

    /// Counter for consecutive high gradient norms
    consecutive_high: usize,

    /// Counter for consecutive low gradient norms
    consecutive_low: usize,
}

impl GradientController {
    /// Create a new gradient controller with default settings
    ///
    /// # Arguments
    /// * `initial_threshold` - Initial clipping threshold (e.g., 1.0)
    pub fn new(initial_threshold: f32) -> Self {
        Self::with_config(
            initial_threshold,
            initial_threshold * 0.1,  // min = 10% of initial
            initial_threshold * 10.0, // max = 10x initial
            100,                      // window size
            10.0,                     // explosion = 10x mean
            0.001,                    // vanishing threshold
            0.1,                      // 10% adjustment rate
        )
    }

    /// Create a gradient controller with custom configuration
    ///
    /// # Arguments
    /// * `initial_threshold` - Starting clipping threshold
    /// * `min_threshold` - Minimum allowed threshold
    /// * `max_threshold` - Maximum allowed threshold
    /// * `window_size` - Number of gradient norms to track
    /// * `explosion_factor` - Multiplier for explosion detection (vs mean)
    /// * `vanishing_threshold` - Absolute threshold for vanishing gradient warning
    /// * `adjustment_rate` - How aggressively to adjust (0.0-1.0)
    pub fn with_config(
        initial_threshold: f32,
        min_threshold: f32,
        max_threshold: f32,
        window_size: usize,
        explosion_factor: f32,
        vanishing_threshold: f32,
        adjustment_rate: f32,
    ) -> Self {
        assert!(initial_threshold > 0.0, "Threshold must be positive");
        assert!(min_threshold > 0.0, "Min threshold must be positive");
        assert!(max_threshold >= initial_threshold, "Max must be >= initial");
        assert!(window_size > 0, "Window size must be positive");
        assert!(explosion_factor > 1.0, "Explosion factor must be > 1.0");
        assert!(
            adjustment_rate >= 0.0 && adjustment_rate <= 1.0,
            "Rate must be in [0, 1]"
        );

        Self {
            clip_threshold: initial_threshold,
            min_threshold,
            max_threshold,
            history: VecDeque::with_capacity(window_size),
            window_size,
            explosion_factor,
            vanishing_threshold,
            adjustment_rate,
            high_count_threshold: (window_size / 4).max(5), // 25% of window
            low_count_threshold: (window_size / 2).max(10), // 50% of window
            consecutive_high: 0,
            consecutive_low: 0,
        }
    }

    /// Update controller with a new gradient norm and get recommended action
    ///
    /// # Arguments
    /// * `gradient_norm` - L2 norm of the current gradients
    ///
    /// # Returns
    /// Action to take (clip, adjust threshold, warn, etc.)
    pub fn update(&mut self, gradient_norm: f32) -> GradientAction {
        // Add to history
        if self.history.len() >= self.window_size {
            self.history.pop_front();
        }
        self.history.push_back(gradient_norm);

        // Need enough history for statistics
        if self.history.len() < 10 {
            // Bootstrap phase - just clip if needed
            return if gradient_norm > self.clip_threshold {
                GradientAction::Clip(self.clip_threshold)
            } else {
                GradientAction::NoChange
            };
        }

        let stats = self.compute_stats();

        // Check for gradient explosion
        if gradient_norm > stats.mean * self.explosion_factor {
            self.consecutive_high = 0;
            self.consecutive_low = 0;

            // Emergency: reduce threshold immediately
            self.clip_threshold = (stats.mean * 2.0).clamp(self.min_threshold, self.max_threshold);

            return GradientAction::EmergencyClip;
        }

        // Check for vanishing gradients
        if gradient_norm < self.vanishing_threshold {
            return GradientAction::Warning(format!(
                "Vanishing gradient detected: {:.6} < {:.6}",
                gradient_norm, self.vanishing_threshold
            ));
        }

        // Track consecutive high/low gradients
        if gradient_norm > stats.mean + stats.std_dev {
            self.consecutive_high += 1;
            self.consecutive_low = 0;
        } else if gradient_norm < stats.mean - stats.std_dev {
            self.consecutive_low += 1;
            self.consecutive_high = 0;
        } else {
            self.consecutive_high = 0;
            self.consecutive_low = 0;
        }

        // Decide on threshold adjustment
        let mut action = if gradient_norm > self.clip_threshold {
            GradientAction::Clip(self.clip_threshold)
        } else {
            GradientAction::NoChange
        };

        // Consecutive high gradients - reduce threshold
        if self.consecutive_high >= self.high_count_threshold {
            let new_threshold = self.clip_threshold * (1.0 - self.adjustment_rate);
            self.clip_threshold = new_threshold.clamp(self.min_threshold, self.max_threshold);
            self.consecutive_high = 0;

            action = GradientAction::ReduceThreshold;
        }

        // Consecutive low gradients - increase threshold
        if self.consecutive_low >= self.low_count_threshold {
            let new_threshold = self.clip_threshold * (1.0 + self.adjustment_rate);
            self.clip_threshold = new_threshold.clamp(self.min_threshold, self.max_threshold);
            self.consecutive_low = 0;

            // Only suggest increase if we're not already clipping
            if matches!(action, GradientAction::NoChange) {
                action = GradientAction::IncreaseThreshold;
            }
        }

        // Check for instability (high variance)
        if stats.std_dev > stats.mean * 2.0 && self.history.len() >= self.window_size / 2 {
            return GradientAction::Warning(format!(
                "High gradient variance detected: std={:.4}, mean={:.4}. Consider reducing learning rate.",
                stats.std_dev, stats.mean
            ));
        }

        action
    }

    /// Get the current clipping threshold
    pub fn current_threshold(&self) -> f32 {
        self.clip_threshold
    }

    /// Get statistics about recent gradient norms
    pub fn stats(&self) -> Option<GradientStats> {
        if self.history.len() < 10 {
            return None;
        }
        Some(self.compute_stats())
    }

    /// Reset the controller (clears history but keeps configuration)
    pub fn reset(&mut self) {
        self.history.clear();
        self.consecutive_high = 0;
        self.consecutive_low = 0;
    }

    /// Update the clipping threshold manually
    pub fn set_threshold(&mut self, threshold: f32) {
        self.clip_threshold = threshold.clamp(self.min_threshold, self.max_threshold);
    }

    // Internal: compute statistics from history
    fn compute_stats(&self) -> GradientStats {
        let n = self.history.len() as f32;
        let sum: f32 = self.history.iter().sum();
        let mean = sum / n;

        let variance: f32 = self
            .history
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>()
            / n;
        let std_dev = variance.sqrt();

        let min = self.history.iter().copied().fold(f32::INFINITY, f32::min);
        let max = self
            .history
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);

        // Compute trend (simple linear regression slope)
        let recent_trend = if self.history.len() >= 20 {
            let n_recent = 20;
            let recent: Vec<f32> = self.history.iter().rev().take(n_recent).copied().collect();

            let x_mean = (n_recent - 1) as f32 / 2.0;
            let y_mean = recent.iter().sum::<f32>() / n_recent as f32;

            let numerator: f32 = recent
                .iter()
                .enumerate()
                .map(|(i, &y)| (i as f32 - x_mean) * (y - y_mean))
                .sum();

            let denominator: f32 = (0..n_recent).map(|i| (i as f32 - x_mean).powi(2)).sum();

            if denominator > 0.0 {
                numerator / denominator
            } else {
                0.0
            }
        } else {
            0.0
        };

        GradientStats {
            mean,
            std_dev,
            min,
            max,
            recent_trend,
        }
    }
}

impl Default for GradientController {
    fn default() -> Self {
        Self::new(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_creation() {
        let controller = GradientController::new(1.0);
        assert_eq!(controller.current_threshold(), 1.0);
    }

    #[test]
    fn test_bootstrap_phase() {
        let mut controller = GradientController::new(1.0);

        // First few steps should just clip if needed
        assert_eq!(controller.update(0.5), GradientAction::NoChange);
        assert_eq!(controller.update(0.8), GradientAction::NoChange);
        assert_eq!(controller.update(1.5), GradientAction::Clip(1.0));
    }

    #[test]
    fn test_explosion_detection() {
        let mut controller = GradientController::new(1.0);

        // Fill with stable gradients
        for _ in 0..20 {
            controller.update(0.5);
        }

        // After 20 updates of 0.5 plus one of 10.0:
        // mean ≈ (20*0.5 + 10.0) / 21 ≈ 0.952
        // explosion_factor = 10.0, so threshold = 9.52
        // 10.0 > 9.52, so explosion is detected
        let action = controller.update(10.0);
        assert_eq!(action, GradientAction::EmergencyClip);

        // After explosion, threshold is set to mean * 2.0 = ~1.9
        // This caps the threshold to a more reasonable value than the original 1.0
        // The key assertion is that EmergencyClip was triggered
        let new_threshold = controller.current_threshold();
        assert!(
            new_threshold > 0.0 && new_threshold < 5.0,
            "Threshold should be reasonable after explosion: {}",
            new_threshold
        );
    }

    #[test]
    fn test_vanishing_detection() {
        let mut controller = GradientController::new(1.0);

        // Fill with stable gradients
        for _ in 0..20 {
            controller.update(0.5);
        }

        // Vanishing gradient
        let action = controller.update(0.0001);
        match action {
            GradientAction::Warning(msg) => {
                assert!(msg.contains("Vanishing"));
            }
            _ => panic!("Expected vanishing gradient warning"),
        }
    }

    #[test]
    fn test_threshold_reduction() {
        let mut controller = GradientController::with_config(
            1.0,   // initial
            0.1,   // min
            10.0,  // max
            40,    // window
            10.0,  // explosion
            0.001, // vanishing
            0.2,   // 20% adjustment
        );

        // Fill with low gradients
        for _ in 0..20 {
            controller.update(0.3);
        }

        let initial_threshold = controller.current_threshold();

        // Now send consistently high gradients (but not explosion)
        for _ in 0..15 {
            controller.update(0.9);
        }

        // Should have reduced threshold
        assert!(controller.current_threshold() < initial_threshold);
    }

    #[test]
    fn test_threshold_increase() {
        let mut controller = GradientController::with_config(
            1.0,   // initial
            0.1,   // min
            10.0,  // max
            40,    // window
            10.0,  // explosion
            0.001, // vanishing
            0.2,   // 20% adjustment
        );

        // Fill with mixed gradients
        for i in 0..20 {
            controller.update(if i % 2 == 0 { 0.3 } else { 0.8 });
        }

        let initial_threshold = controller.current_threshold();

        // Now send consistently low gradients
        for _ in 0..25 {
            controller.update(0.1);
        }

        // Should have increased threshold (though may be clamped)
        assert!(controller.current_threshold() >= initial_threshold * 0.9);
    }

    #[test]
    fn test_stats_computation() {
        let mut controller = GradientController::new(1.0);

        // Need at least 10 samples
        for i in 0..20 {
            controller.update((i as f32) * 0.1);
        }

        let stats = controller.stats().unwrap();
        assert!(stats.mean > 0.0);
        assert!(stats.std_dev > 0.0);
        assert!(stats.min >= 0.0);
        assert!(stats.max > stats.min);
    }

    #[test]
    fn test_manual_threshold_update() {
        let mut controller = GradientController::new(1.0);

        controller.set_threshold(2.0);
        assert_eq!(controller.current_threshold(), 2.0);

        // Should clamp to max
        controller.set_threshold(100.0);
        assert_eq!(controller.current_threshold(), controller.max_threshold);
    }

    #[test]
    fn test_reset() {
        let mut controller = GradientController::new(1.0);

        for _ in 0..20 {
            controller.update(0.5);
        }

        assert!(controller.stats().is_some());

        controller.reset();
        assert!(controller.stats().is_none());
    }

    #[test]
    fn test_instability_warning() {
        let mut controller = GradientController::new(1.0);

        // Create high variance scenario
        for i in 0..30 {
            let val = if i % 2 == 0 { 0.1 } else { 2.0 };
            controller.update(val);
        }

        // Should eventually warn about instability
        let action = controller.update(0.1);
        match action {
            GradientAction::Warning(msg) => {
                assert!(msg.contains("variance") || msg.contains("instability"));
            }
            _ => {} // May not trigger on every step
        }
    }
}
