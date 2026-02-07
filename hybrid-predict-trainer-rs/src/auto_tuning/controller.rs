//! Automatic tuning controller that orchestrates health scoring, gradient tuning,
//! and plateau detection to provide unified training optimization recommendations.
//!
//! This module integrates the `auto_tuning` subsystems into a single controller
//! that can be integrated into the `HybridTrainer`'s `step()` method.

use super::gradient_tuner::{GradientRangeTuner, PhaseGradientThresholds};
use super::health_scorer::{HealthClassification, HealthRecommendation, HealthScorer};
use super::plateau_detector::{PlateauDetector, PlateauStatus};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for the automatic tuning controller.
///
/// Controls the behavior of health scoring, gradient tuning, and plateau detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoTuningConfig {
    /// Window size for health scoring (number of steps to consider).
    pub health_window: usize,

    /// Window size for plateau detection.
    pub plateau_window: usize,

    /// Minimum velocity threshold for plateau detection (absolute value).
    pub plateau_velocity_threshold: f32,

    /// Whether to enable automatic warmup restarts.
    pub enable_warmup_restart: bool,

    /// Learning rate multiplier for warmup restart (e.g., 1.5).
    pub warmup_restart_lr_multiplier: f32,

    /// Cooldown steps between warmup restarts.
    pub warmup_restart_cooldown: u64,

    /// Maximum number of warmup restarts allowed.
    pub max_warmup_restarts: usize,

    /// Enable adaptive gradient clipping.
    pub enable_gradient_clipping: bool,

    /// AGC lambda parameter for per-layer clipping.
    pub agc_lambda: f32,

    /// Phase-specific gradient thresholds.
    pub phase_thresholds: HashMap<String, PhaseGradientThresholds>,
}

impl Default for AutoTuningConfig {
    fn default() -> Self {
        let mut phase_thresholds = HashMap::new();
        phase_thresholds.insert(
            "warmup".to_string(),
            PhaseGradientThresholds::new(0.01, 0.5),
        );
        phase_thresholds.insert("early".to_string(), PhaseGradientThresholds::new(0.1, 1.0));
        phase_thresholds.insert("mid".to_string(), PhaseGradientThresholds::new(0.3, 0.8));
        phase_thresholds.insert("late".to_string(), PhaseGradientThresholds::new(0.1, 0.5));

        Self {
            health_window: 50,
            plateau_window: 50,
            plateau_velocity_threshold: 0.001,
            enable_warmup_restart: true,
            warmup_restart_lr_multiplier: 1.5,
            warmup_restart_cooldown: 500,
            max_warmup_restarts: 3,
            enable_gradient_clipping: true,
            agc_lambda: 0.01,
            phase_thresholds,
        }
    }
}

/// Update result from the auto-tuning controller.
///
/// Contains recommended actions and state information for the training loop.
#[derive(Debug, Clone)]
pub struct AutoTuningUpdate {
    /// Overall training health classification.
    pub health: HealthClassification,

    /// Recommended actions to improve training.
    pub recommendations: Vec<HealthRecommendation>,

    /// Plateau detection status.
    pub plateau_status: PlateauStatus,

    /// Per-layer gradient clipping factors (if enabled).
    ///
    /// Maps layer name to clipping coefficient. Coefficient of 1.0 means no clipping.
    pub layer_clip_factors: HashMap<String, f32>,

    /// Learning rate multiplier for warmup restart (Some(value) if restart is recommended).
    pub warmup_restart: Option<f32>,

    /// Current training progress percentage (0-100).
    pub progress_pct: f32,

    /// Loss velocity (negative is good).
    pub velocity: f32,

    /// Loss acceleration.
    pub acceleration: f32,

    /// Gradient entropy score [0, 1].
    pub gradient_entropy: f32,

    /// Prediction accuracy score [0, 1].
    pub prediction_accuracy: f32,

    /// Gradient stability score [0, 1].
    pub gradient_stability: f32,

    /// Overall health score [0, 1].
    pub health_score: f32,
}

impl AutoTuningUpdate {
    /// Returns true if a warmup restart is recommended.
    #[must_use]
    pub fn should_restart(&self) -> bool {
        self.warmup_restart.is_some()
    }

    /// Returns true if health is critical and requires immediate action.
    #[must_use]
    pub fn is_critical(&self) -> bool {
        self.health == HealthClassification::Critical
    }

    /// Returns true if any gradient clipping is recommended.
    #[must_use]
    pub fn has_clipping(&self) -> bool {
        self.layer_clip_factors.values().any(|&coeff| coeff < 1.0)
    }
}

/// Automatic tuning controller that orchestrates health monitoring and tuning.
///
/// Integrates health scoring, gradient tuning, and plateau detection to provide
/// unified training optimization recommendations.
pub struct AutoTuningController {
    config: AutoTuningConfig,
    health_scorer: HealthScorer,
    gradient_tuner: GradientRangeTuner,
    plateau_detector: PlateauDetector,
    max_steps: u64,
    warmup_restart_count: usize,
    last_restart_step: Option<u64>,
    current_step: u64,
}

impl AutoTuningController {
    /// Creates a new auto-tuning controller.
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration for auto-tuning behavior
    /// * `max_steps` - Total number of training steps (for progress percentage)
    #[must_use]
    pub fn new(config: AutoTuningConfig, max_steps: u64) -> Self {
        let health_scorer = HealthScorer::new(config.health_window, max_steps as usize);
        let gradient_tuner = GradientRangeTuner::with_agc_lambda(config.agc_lambda);
        let plateau_detector =
            PlateauDetector::new(config.plateau_window, config.plateau_velocity_threshold);

        Self {
            config,
            health_scorer,
            gradient_tuner,
            plateau_detector,
            max_steps,
            warmup_restart_count: 0,
            last_restart_step: None,
            current_step: 0,
        }
    }

    /// Updates the controller with new training step data.
    ///
    /// # Arguments
    ///
    /// * `step` - Current training step
    /// * `loss` - Current loss value
    /// * `gradient_norm` - Global gradient norm
    /// * `layer_gradients` - Per-layer gradient statistics (name, `grad_norm`, `weight_norm`)
    /// * `confidence` - Predictor confidence [0, 1]
    ///
    /// # Returns
    ///
    /// An `AutoTuningUpdate` containing recommendations and state information.
    pub fn update(
        &mut self,
        step: u64,
        loss: f32,
        gradient_norm: f32,
        layer_gradients: &[(String, f32, f32)],
        confidence: f32,
    ) -> AutoTuningUpdate {
        self.current_step = step;
        let progress_pct = (step as f32 / self.max_steps as f32) * 100.0;

        // Update plateau detector
        let plateau_status = self.plateau_detector.update(loss, step, progress_pct);

        // Compute loss dynamics from plateau detector
        let velocity = self.plateau_detector.velocity();
        let acceleration = self.plateau_detector.acceleration();

        // Compute gradient entropy and stability (simplified for now)
        let gradient_entropy = self.compute_gradient_entropy(gradient_norm);
        let gradient_stability = self.compute_gradient_stability(gradient_norm);

        // Compute prediction accuracy from confidence
        let prediction_accuracy = confidence;

        // Update health scorer
        let health_result = self.health_scorer.compute(
            velocity,
            acceleration,
            gradient_entropy,
            prediction_accuracy,
            gradient_stability,
            progress_pct,
        );

        // Update gradient tuner with layer statistics
        let _training_phase = self.determine_training_phase(progress_pct);

        // Convert layer_gradients to HashMap format for gradient_tuner
        let mut layer_grads_map = HashMap::new();
        for (layer_name, grad_norm, weight_norm) in layer_gradients {
            layer_grads_map.insert(layer_name.clone(), (*grad_norm, *weight_norm));
        }

        let clip_factors =
            self.gradient_tuner
                .update(gradient_norm, &layer_grads_map, progress_pct);

        // Layer clip factors come from gradient tuner directly
        let layer_clip_factors = clip_factors;

        // Check if warmup restart is needed
        let warmup_restart = if self.config.enable_warmup_restart
            && plateau_status == PlateauStatus::Stuck
            && self.warmup_restart_count < self.config.max_warmup_restarts
        {
            // Check cooldown
            let can_restart = match self.last_restart_step {
                None => true,
                Some(last_step) => step - last_step >= self.config.warmup_restart_cooldown,
            };

            if can_restart {
                self.warmup_restart_count += 1;
                self.last_restart_step = Some(step);
                Some(self.config.warmup_restart_lr_multiplier)
            } else {
                None
            }
        } else {
            None
        };

        AutoTuningUpdate {
            health: health_result.classification,
            recommendations: health_result.recommendations,
            plateau_status,
            layer_clip_factors,
            warmup_restart,
            progress_pct,
            velocity,
            acceleration,
            gradient_entropy,
            prediction_accuracy,
            gradient_stability,
            health_score: health_result.overall,
        }
    }

    /// Determines the current training phase based on progress percentage.
    fn determine_training_phase(&self, progress_pct: f32) -> &str {
        if progress_pct < 10.0 {
            "warmup"
        } else if progress_pct < 40.0 {
            "early"
        } else if progress_pct < 80.0 {
            "mid"
        } else {
            "late"
        }
    }

    /// Computes gradient entropy score from gradient norm.
    ///
    /// This is a simplified implementation. A more sophisticated version would
    /// track gradient variance and compute actual entropy.
    fn compute_gradient_entropy(&self, gradient_norm: f32) -> f32 {
        // Normalize to [0, 1] assuming reasonable gradient range [0.01, 10.0]
        let log_norm = gradient_norm.max(0.01).ln();
        let log_min = 0.01_f32.ln();
        let log_max = 10.0_f32.ln();

        ((log_norm - log_min) / (log_max - log_min)).clamp(0.0, 1.0)
    }

    /// Computes gradient stability score from gradient norm.
    ///
    /// This is a simplified implementation. A more sophisticated version would
    /// track variance of gradient norms over time.
    fn compute_gradient_stability(&self, gradient_norm: f32) -> f32 {
        // For now, use inverse of gradient magnitude as stability proxy
        // High gradients = low stability, low gradients = high stability
        // Map [0.01, 10.0] to [1.0, 0.0]
        let normalized = (gradient_norm.ln() - 0.01_f32.ln()) / (10.0_f32.ln() - 0.01_f32.ln());
        (1.0 - normalized).clamp(0.0, 1.0)
    }

    /// Returns the current warmup restart count.
    #[must_use]
    pub fn warmup_restart_count(&self) -> usize {
        self.warmup_restart_count
    }

    /// Returns the step at which the last warmup restart occurred.
    #[must_use]
    pub fn last_restart_step(&self) -> Option<u64> {
        self.last_restart_step
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_auto_tuning_config_default() {
        let config = AutoTuningConfig::default();
        assert_eq!(config.health_window, 50);
        assert_eq!(config.plateau_window, 50);
        assert!(config.enable_warmup_restart);
        assert!(config.enable_gradient_clipping);
    }

    #[test]
    fn test_auto_tuning_controller_creation() {
        let config = AutoTuningConfig::default();
        let controller = AutoTuningController::new(config, 1000);
        assert_eq!(controller.warmup_restart_count(), 0);
        assert!(controller.last_restart_step().is_none());
    }

    #[test]
    fn test_auto_tuning_update() {
        let config = AutoTuningConfig::default();
        let mut controller = AutoTuningController::new(config, 1000);

        let layer_grads = vec![
            ("embed".to_string(), 0.5, 10.0),
            ("attention".to_string(), 0.8, 15.0),
        ];

        let update = controller.update(0, 1.5, 0.6, &layer_grads, 0.9);

        assert!(update.progress_pct >= 0.0 && update.progress_pct <= 100.0);
        assert!(!update.should_restart()); // No restart on first update with reasonable loss
    }

    #[test]
    fn test_auto_tuning_update_has_recommendations() {
        let update = AutoTuningUpdate {
            health: HealthClassification::Good,
            recommendations: vec![],
            plateau_status: PlateauStatus::Normal,
            layer_clip_factors: HashMap::new(),
            warmup_restart: None,
            progress_pct: 50.0,
            velocity: -0.01,
            acceleration: 0.001,
            gradient_entropy: 0.5,
            prediction_accuracy: 0.9,
            gradient_stability: 0.8,
            health_score: 0.75,
        };

        assert!(!update.should_restart());
        assert!(!update.is_critical());
        assert!(!update.has_clipping());
    }
}
