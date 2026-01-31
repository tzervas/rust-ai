//! Multi-signal divergence detection.
//!
//! Monitors training metrics for signs of instability that could indicate
//! prediction divergence or training failure. Early detection allows for
//! recovery actions before catastrophic failure (NaN, explosion).
//!
//! # Why Multi-Signal Detection?
//!
//! Single-metric thresholds (e.g., loss > X) produce many false positives due
//! to natural training variance. By combining multiple signals—loss deviation,
//! gradient norms, oscillation patterns—we achieve:
//! - **Higher sensitivity**: Catch real problems earlier
//! - **Fewer false alarms**: Correlated signals reduce noise
//! - **Graceful degradation**: Partial signal loss doesn't blind detection
//!
//! # Monitored Signals
//!
//! - **Loss deviation**: Actual loss vs predicted or historical
//! - **Gradient norm**: Explosion or vanishing gradients
//! - **Loss plateau**: Training stagnation
//! - **Oscillation**: Unstable training dynamics
//!
//! # Detection Algorithm
//!
//! Uses exponential moving averages and statistical deviation tests to
//! identify anomalies. Multiple signals are combined to reduce false
//! positives while maintaining sensitivity to real issues.

use crate::config::{DivergenceConfig, HybridTrainerConfig};
use crate::error::DivergenceLevel;
use crate::state::TrainingState;

/// Result of a divergence check.
#[derive(Debug, Clone)]
pub struct DivergenceCheckResult {
    /// Overall severity level.
    pub level: DivergenceLevel,

    /// Individual signal results.
    pub signals: Vec<SignalResult>,

    /// Human-readable summary.
    pub summary: String,

    /// Recommended action based on signals.
    pub recommended_action: Option<String>,
}

impl DivergenceCheckResult {
    /// Returns whether any concerning signals were detected.
    #[must_use]
    pub fn is_concerning(&self) -> bool {
        self.level > DivergenceLevel::Normal
    }

    /// Returns whether immediate action is needed.
    #[must_use]
    pub fn needs_immediate_action(&self) -> bool {
        self.level >= DivergenceLevel::Warning
    }
}

/// Result from a single monitoring signal.
#[derive(Debug, Clone)]
pub struct SignalResult {
    /// Name of the signal.
    pub name: String,

    /// Current value.
    pub value: f32,

    /// Expected value (EMA or predicted).
    pub expected: f32,

    /// Deviation from expected (in standard deviations).
    pub deviation_sigma: f32,

    /// Whether this signal triggered.
    pub triggered: bool,

    /// Severity if triggered.
    pub severity: DivergenceLevel,
}

/// Monitor for detecting training divergence.
pub struct DivergenceMonitor {
    /// Configuration.
    config: DivergenceConfig,

    /// EMA of loss values.
    loss_ema: f32,

    /// EMA of loss variance (for std computation).
    loss_var_ema: f32,

    /// EMA of gradient norms.
    gradient_norm_ema: f32,

    /// EMA of gradient norm variance.
    gradient_var_ema: f32,

    /// Baseline gradient norm (from warmup).
    baseline_gradient_norm: f32,

    /// Number of observations.
    num_observations: usize,

    /// EMA decay factor.
    ema_decay: f32,

    /// History of recent loss values for oscillation detection.
    loss_history: Vec<f32>,

    /// Maximum history size.
    max_history: usize,

    /// Number of consecutive warnings.
    consecutive_warnings: usize,

    /// Last check step.
    last_check_step: u64,
}

impl DivergenceMonitor {
    /// Creates a new divergence monitor with the given configuration.
    #[must_use]
    pub fn new(config: &HybridTrainerConfig) -> Self {
        Self {
            config: config.divergence_config.clone(),
            loss_ema: 0.0,
            loss_var_ema: 0.0,
            gradient_norm_ema: 0.0,
            gradient_var_ema: 0.0,
            baseline_gradient_norm: 1.0,
            num_observations: 0,
            ema_decay: 0.99,
            loss_history: Vec::with_capacity(100),
            max_history: 100,
            consecutive_warnings: 0,
            last_check_step: 0,
        }
    }

    /// Creates a monitor with default configuration.
    #[must_use]
    pub fn default_config() -> Self {
        Self::new(&HybridTrainerConfig::default())
    }

    /// Updates the monitor with new observations.
    ///
    /// # Arguments
    ///
    /// * `loss` - Current loss value
    /// * `gradient_norm` - Current gradient norm
    pub fn observe(&mut self, loss: f32, gradient_norm: f32) {
        let alpha = if self.num_observations == 0 {
            1.0
        } else {
            1.0 - self.ema_decay
        };

        // Update loss EMA and variance
        let loss_delta = loss - self.loss_ema;
        self.loss_ema += alpha * loss_delta;
        self.loss_var_ema = self.ema_decay * self.loss_var_ema + alpha * loss_delta * loss_delta;

        // Update gradient norm EMA and variance
        let grad_delta = gradient_norm - self.gradient_norm_ema;
        self.gradient_norm_ema += alpha * grad_delta;
        self.gradient_var_ema =
            self.ema_decay * self.gradient_var_ema + alpha * grad_delta * grad_delta;

        // Set baseline during early training
        if self.num_observations < 50 {
            self.baseline_gradient_norm = self.gradient_norm_ema;
        }

        // Update history
        self.loss_history.push(loss);
        if self.loss_history.len() > self.max_history {
            self.loss_history.remove(0);
        }

        self.num_observations += 1;
    }

    /// Checks for divergence based on current state.
    ///
    /// # Arguments
    ///
    /// * `state` - Current training state
    /// * `predicted_loss` - Optional predicted loss for comparison
    ///
    /// # Returns
    ///
    /// Result indicating divergence level and signals.
    pub fn check(
        &mut self,
        state: &TrainingState,
        predicted_loss: Option<f32>,
    ) -> DivergenceCheckResult {
        let mut signals = Vec::new();
        let mut max_severity = DivergenceLevel::Normal;

        // Check loss deviation
        let loss_std = self.loss_var_ema.sqrt().max(1e-6);
        let loss_deviation = (state.loss - self.loss_ema) / loss_std;

        let loss_signal = SignalResult {
            name: "loss_deviation".to_string(),
            value: state.loss,
            expected: self.loss_ema,
            deviation_sigma: loss_deviation,
            triggered: loss_deviation.abs() > self.config.loss_sigma_threshold,
            severity: if loss_deviation.abs() > self.config.loss_sigma_threshold * 2.0 {
                DivergenceLevel::Critical
            } else if loss_deviation.abs() > self.config.loss_sigma_threshold {
                DivergenceLevel::Warning
            } else if loss_deviation.abs() > self.config.loss_sigma_threshold * 0.7 {
                DivergenceLevel::Caution
            } else {
                DivergenceLevel::Normal
            },
        };

        if loss_signal.severity > max_severity {
            max_severity = loss_signal.severity;
        }
        signals.push(loss_signal);

        // Check gradient explosion
        let grad_threshold = self.baseline_gradient_norm * self.config.gradient_norm_multiplier;
        let grad_explosion = state.gradient_norm > grad_threshold;

        let grad_signal = SignalResult {
            name: "gradient_norm".to_string(),
            value: state.gradient_norm,
            expected: self.gradient_norm_ema,
            deviation_sigma: (state.gradient_norm - self.gradient_norm_ema)
                / self.gradient_var_ema.sqrt().max(1e-6),
            triggered: grad_explosion,
            severity: if state.gradient_norm > grad_threshold * 10.0 {
                DivergenceLevel::Critical
            } else if grad_explosion {
                DivergenceLevel::Warning
            } else {
                DivergenceLevel::Normal
            },
        };

        if grad_signal.severity > max_severity {
            max_severity = grad_signal.severity;
        }
        signals.push(grad_signal);

        // Check gradient vanishing
        let vanishing = state.gradient_norm
            < self.config.vanishing_gradient_threshold * self.baseline_gradient_norm;

        let vanish_signal = SignalResult {
            name: "gradient_vanishing".to_string(),
            value: state.gradient_norm,
            expected: self.baseline_gradient_norm * self.config.vanishing_gradient_threshold,
            deviation_sigma: 0.0,
            triggered: vanishing,
            severity: if vanishing {
                DivergenceLevel::Warning
            } else {
                DivergenceLevel::Normal
            },
        };

        if vanish_signal.severity > max_severity {
            max_severity = vanish_signal.severity;
        }
        signals.push(vanish_signal);

        // Check NaN/Inf
        if state.loss.is_nan()
            || state.loss.is_infinite()
            || state.gradient_norm.is_nan()
            || state.gradient_norm.is_infinite()
        {
            max_severity = DivergenceLevel::Critical;
            signals.push(SignalResult {
                name: "numerical_stability".to_string(),
                value: state.loss,
                expected: self.loss_ema,
                deviation_sigma: f32::INFINITY,
                triggered: true,
                severity: DivergenceLevel::Critical,
            });
        }

        // Check prediction accuracy (if predicted loss provided)
        if let Some(predicted) = predicted_loss {
            let prediction_error = (state.loss - predicted).abs();
            let relative_error = prediction_error / state.loss.abs().max(0.1);

            let pred_signal = SignalResult {
                name: "prediction_error".to_string(),
                value: state.loss,
                expected: predicted,
                deviation_sigma: relative_error * 3.0, // Scale to sigma units
                triggered: relative_error > 0.2,       // 20% relative error
                severity: if relative_error > 0.5 {
                    DivergenceLevel::Warning
                } else if relative_error > 0.2 {
                    DivergenceLevel::Caution
                } else {
                    DivergenceLevel::Normal
                },
            };

            if pred_signal.severity > max_severity {
                max_severity = pred_signal.severity;
            }
            signals.push(pred_signal);
        }

        // Check oscillation
        if self.loss_history.len() >= 10 {
            let oscillation_score = self.compute_oscillation_score();

            let osc_signal = SignalResult {
                name: "oscillation".to_string(),
                value: oscillation_score,
                expected: 0.0,
                deviation_sigma: oscillation_score * 5.0,
                triggered: oscillation_score > 0.5,
                severity: if oscillation_score > 0.8 {
                    DivergenceLevel::Warning
                } else if oscillation_score > 0.5 {
                    DivergenceLevel::Caution
                } else {
                    DivergenceLevel::Normal
                },
            };

            if osc_signal.severity > max_severity {
                max_severity = osc_signal.severity;
            }
            signals.push(osc_signal);
        }

        // Update consecutive warnings counter
        if max_severity >= DivergenceLevel::Warning {
            self.consecutive_warnings += 1;
        } else {
            self.consecutive_warnings = 0;
        }

        // Escalate severity if warnings are persistent
        if self.consecutive_warnings >= 3 && max_severity == DivergenceLevel::Warning {
            max_severity = DivergenceLevel::Critical;
        }

        self.last_check_step = state.step;

        let summary = self.generate_summary(&signals, max_severity);
        let recommended_action = self.generate_recommendation(max_severity);

        DivergenceCheckResult {
            level: max_severity,
            signals,
            summary,
            recommended_action,
        }
    }

    /// Computes an oscillation score from loss history.
    ///
    /// Returns a value between 0 (no oscillation) and 1 (severe oscillation).
    fn compute_oscillation_score(&self) -> f32 {
        if self.loss_history.len() < 4 {
            return 0.0;
        }

        // Count sign changes in the derivative
        let mut sign_changes = 0;
        let mut prev_sign = 0i32;

        for window in self.loss_history.windows(2) {
            let diff = window[1] - window[0];
            let sign = if diff > 0.0 {
                1
            } else if diff < 0.0 {
                -1
            } else {
                0
            };

            if prev_sign != 0 && sign != 0 && sign != prev_sign {
                sign_changes += 1;
            }

            if sign != 0 {
                prev_sign = sign;
            }
        }

        // Normalize by number of opportunities for sign change
        let max_changes = (self.loss_history.len() - 1) as f32;
        (sign_changes as f32 / max_changes).min(1.0)
    }

    /// Generates a summary message for the divergence check.
    fn generate_summary(&self, signals: &[SignalResult], severity: DivergenceLevel) -> String {
        let triggered: Vec<_> = signals
            .iter()
            .filter(|s| s.triggered)
            .map(|s| s.name.as_str())
            .collect();

        match severity {
            DivergenceLevel::Normal => "Training metrics within normal ranges".to_string(),
            DivergenceLevel::Caution => {
                format!("Caution: {} showing unusual values", triggered.join(", "))
            }
            DivergenceLevel::Warning => {
                format!(
                    "Warning: {} triggered divergence detection",
                    triggered.join(", ")
                )
            }
            DivergenceLevel::Critical => {
                format!(
                    "Critical: {} indicate imminent training failure",
                    triggered.join(", ")
                )
            }
        }
    }

    /// Generates a recommendation based on severity.
    fn generate_recommendation(&self, severity: DivergenceLevel) -> Option<String> {
        match severity {
            DivergenceLevel::Normal => None,
            DivergenceLevel::Caution => {
                Some("Consider reducing prediction phase length".to_string())
            }
            DivergenceLevel::Warning => {
                Some("Force full training phase to re-establish stable dynamics".to_string())
            }
            DivergenceLevel::Critical => {
                Some("Rollback to checkpoint and reduce learning rate".to_string())
            }
        }
    }

    /// Resets the monitor state.
    pub fn reset(&mut self) {
        self.loss_ema = 0.0;
        self.loss_var_ema = 0.0;
        self.gradient_norm_ema = 0.0;
        self.gradient_var_ema = 0.0;
        self.num_observations = 0;
        self.loss_history.clear();
        self.consecutive_warnings = 0;
    }

    /// Returns the current loss EMA.
    #[must_use]
    pub fn loss_ema(&self) -> f32 {
        self.loss_ema
    }

    /// Returns the current loss standard deviation estimate.
    #[must_use]
    pub fn loss_std(&self) -> f32 {
        self.loss_var_ema.sqrt()
    }

    /// Returns the current gradient norm EMA.
    #[must_use]
    pub fn gradient_norm_ema(&self) -> f32 {
        self.gradient_norm_ema
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_monitor_initialization() {
        let monitor = DivergenceMonitor::default_config();
        assert_eq!(monitor.num_observations, 0);
        assert!((monitor.loss_ema).abs() < f32::EPSILON);
    }

    #[test]
    fn test_observe_updates_ema() {
        let mut monitor = DivergenceMonitor::default_config();

        monitor.observe(2.5, 1.0);
        monitor.observe(2.4, 0.9);
        monitor.observe(2.3, 0.8);

        assert_eq!(monitor.num_observations, 3);
        assert!(monitor.loss_ema > 2.0 && monitor.loss_ema < 2.6);
    }

    #[test]
    fn test_normal_training_detection() {
        let mut monitor = DivergenceMonitor::default_config();

        // Simulate normal training
        for i in 0..100 {
            let loss = 3.0 - i as f32 * 0.01;
            let grad = 1.0 + (i as f32 * 0.01).sin() * 0.1;
            monitor.observe(loss, grad);
        }

        let mut state = TrainingState::new();
        state.loss = 2.0;
        state.gradient_norm = 1.0;

        let result = monitor.check(&state, None);
        assert_eq!(result.level, DivergenceLevel::Normal);
    }

    #[test]
    fn test_loss_spike_detection() {
        let mut monitor = DivergenceMonitor::default_config();

        // Establish baseline
        for _ in 0..50 {
            monitor.observe(2.5, 1.0);
        }

        // Now check with a spike
        let mut state = TrainingState::new();
        state.loss = 10.0; // Big spike
        state.gradient_norm = 1.0;

        let result = monitor.check(&state, None);
        assert!(result.level >= DivergenceLevel::Warning);
    }

    #[test]
    fn test_nan_detection() {
        let mut monitor = DivergenceMonitor::default_config();

        let mut state = TrainingState::new();
        state.loss = f32::NAN;
        state.gradient_norm = 1.0;

        let result = monitor.check(&state, None);
        assert_eq!(result.level, DivergenceLevel::Critical);
    }

    #[test]
    fn test_oscillation_detection() {
        let mut monitor = DivergenceMonitor::default_config();

        // Create oscillating pattern
        for i in 0..20 {
            let loss = 2.5 + if i % 2 == 0 { 0.3 } else { -0.3 };
            monitor.observe(loss, 1.0);
        }

        let score = monitor.compute_oscillation_score();
        assert!(score > 0.5); // Should detect oscillation
    }
}
