//! Training alerts and notification system
//!
//! Monitors training metrics and triggers alerts based on configurable conditions.
//! Supports severity levels (Info, Warning, Critical) and maintains alert history.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Alert severity level
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum AlertSeverity {
    /// Informational alerts (e.g., training phase transitions)
    Info,
    /// Warnings that may require attention (e.g., loss plateau)
    Warning,
    /// Critical issues that require immediate action (e.g., divergence, OOM)
    Critical,
}

impl AlertSeverity {
    /// Returns a colored string representation for terminal output
    pub fn as_colored_str(&self) -> &'static str {
        match self {
            AlertSeverity::Info => "\x1b[34mINFO\x1b[0m",
            AlertSeverity::Warning => "\x1b[33mWARNING\x1b[0m",
            AlertSeverity::Critical => "\x1b[31mCRITICAL\x1b[0m",
        }
    }
}

/// Alert condition types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertCondition {
    /// Trigger when loss spikes beyond threshold multiplier from baseline
    LossSpike {
        /// Multiplier threshold (e.g., 2.0 = 2x baseline)
        threshold: f32,
    },
    /// Trigger when loss doesn't improve for N steps
    Plateau {
        /// Number of steps without improvement
        steps: usize,
    },
    /// Trigger when loss increases continuously (diverging)
    Diverging,
    /// Trigger when memory usage exceeds threshold (0.0-1.0)
    MemoryHigh {
        /// Memory usage threshold (e.g., 0.9 = 90%)
        threshold: f32,
    },
    /// Trigger when gradient norm exceeds threshold
    GradientExploding {
        /// Absolute gradient norm threshold
        threshold: f32,
    },
    /// Trigger when gradient norm falls below threshold
    GradientVanishing {
        /// Absolute gradient norm threshold
        threshold: f32,
    },
}

impl AlertCondition {
    /// Returns default alert conditions for typical training scenarios
    pub fn defaults() -> Vec<AlertCondition> {
        vec![
            AlertCondition::LossSpike { threshold: 2.0 },
            AlertCondition::Plateau { steps: 100 },
            AlertCondition::Diverging,
            AlertCondition::MemoryHigh { threshold: 0.9 },
            AlertCondition::GradientExploding { threshold: 100.0 },
            AlertCondition::GradientVanishing { threshold: 1e-6 },
        ]
    }
}

/// Training alert record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    /// Alert severity level
    pub severity: AlertSeverity,
    /// Human-readable alert message
    pub message: String,
    /// Timestamp when alert was triggered
    pub timestamp: DateTime<Utc>,
    /// Training step when alert occurred
    pub step: u64,
    /// Whether the alert has been acknowledged
    pub acknowledged: bool,
}

impl Alert {
    /// Create a new alert
    pub fn new(severity: AlertSeverity, message: String, step: u64) -> Self {
        Self {
            severity,
            message,
            timestamp: Utc::now(),
            step,
            acknowledged: false,
        }
    }

    /// Format alert for display
    pub fn format(&self) -> String {
        format!(
            "[{}] Step {}: {} ({})",
            self.severity.as_colored_str(),
            self.step,
            self.message,
            self.timestamp.format("%Y-%m-%d %H:%M:%S")
        )
    }
}

/// Training metrics snapshot for alert evaluation
#[derive(Debug, Clone)]
pub struct MetricsSnapshot {
    /// Current training step
    pub step: u64,
    /// Current loss value
    pub loss: f32,
    /// Current gradient norm (if available)
    pub gradient_norm: Option<f32>,
    /// Current memory usage fraction (0.0-1.0, if available)
    pub memory_usage: Option<f32>,
}

/// Alert manager that monitors training and triggers alerts
pub struct AlertManager {
    /// Alert history (bounded by max_history)
    alerts: VecDeque<Alert>,
    /// Active alert conditions
    conditions: Vec<AlertCondition>,
    /// Maximum number of alerts to keep in history
    max_history: usize,
    /// Recent loss values for baseline calculation
    loss_history: VecDeque<f32>,
    /// Step of last improvement (for plateau detection)
    last_improvement_step: u64,
    /// Best loss seen so far
    best_loss: f32,
    /// Whether to send desktop notifications
    desktop_notifications: bool,
}

impl AlertManager {
    /// Create a new AlertManager with default conditions
    pub fn new() -> Self {
        Self::with_conditions(AlertCondition::defaults())
    }

    /// Create a new AlertManager with custom conditions
    pub fn with_conditions(conditions: Vec<AlertCondition>) -> Self {
        Self {
            alerts: VecDeque::new(),
            conditions,
            max_history: 1000,
            loss_history: VecDeque::new(),
            last_improvement_step: 0,
            best_loss: f32::INFINITY,
            desktop_notifications: false,
        }
    }

    /// Set maximum alert history size
    pub fn set_max_history(&mut self, max: usize) {
        self.max_history = max;
        while self.alerts.len() > max {
            self.alerts.pop_front();
        }
    }

    /// Enable or disable desktop notifications
    pub fn set_desktop_notifications(&mut self, enabled: bool) {
        self.desktop_notifications = enabled;
    }

    /// Add a custom alert condition
    pub fn add_condition(&mut self, condition: AlertCondition) {
        self.conditions.push(condition);
    }

    /// Clear all alert conditions
    pub fn clear_conditions(&mut self) {
        self.conditions.clear();
    }

    /// Process metrics snapshot and generate alerts
    pub fn check_metrics(&mut self, metrics: MetricsSnapshot) -> Vec<Alert> {
        let mut new_alerts = Vec::new();

        // Update loss history
        self.loss_history.push_back(metrics.loss);
        if self.loss_history.len() > 200 {
            self.loss_history.pop_front();
        }

        // Track best loss for plateau detection
        if metrics.loss < self.best_loss {
            self.best_loss = metrics.loss;
            self.last_improvement_step = metrics.step;
        }

        // Check each condition
        let conditions = self.conditions.clone();
        for condition in &conditions {
            if let Some(alert) = self.evaluate_condition(condition, &metrics) {
                self.add_alert(alert.clone());
                new_alerts.push(alert);
            }
        }

        new_alerts
    }

    /// Evaluate a single alert condition
    fn evaluate_condition(
        &self,
        condition: &AlertCondition,
        metrics: &MetricsSnapshot,
    ) -> Option<Alert> {
        match condition {
            AlertCondition::LossSpike { threshold } => {
                if self.loss_history.len() < 10 {
                    return None; // Need baseline
                }

                // Calculate baseline as median of recent losses
                let mut sorted: Vec<f32> = self.loss_history.iter().copied().collect();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let baseline = sorted[sorted.len() / 2];

                if metrics.loss > baseline * threshold {
                    return Some(Alert::new(
                        AlertSeverity::Warning,
                        format!(
                            "Loss spike detected: {:.4} ({}x baseline {:.4})",
                            metrics.loss,
                            metrics.loss / baseline,
                            baseline
                        ),
                        metrics.step,
                    ));
                }
            }

            AlertCondition::Plateau { steps } => {
                let steps_without_improvement = metrics.step - self.last_improvement_step;
                if steps_without_improvement >= *steps as u64 {
                    return Some(Alert::new(
                        AlertSeverity::Warning,
                        format!(
                            "Loss plateau: No improvement for {} steps (best: {:.4})",
                            steps_without_improvement, self.best_loss
                        ),
                        metrics.step,
                    ));
                }
            }

            AlertCondition::Diverging => {
                if self.loss_history.len() < 20 {
                    return None;
                }

                // Check if loss is consistently increasing
                let recent: Vec<f32> = self.loss_history.iter().rev().take(20).copied().collect();
                let mut increasing_count = 0;
                for i in 1..recent.len() {
                    if recent[i - 1] > recent[i] {
                        increasing_count += 1;
                    }
                }

                // If >80% of recent steps show increasing loss
                if increasing_count >= 16 {
                    return Some(Alert::new(
                        AlertSeverity::Critical,
                        format!(
                            "Training diverging: Loss increasing for {}/{} recent steps",
                            increasing_count, 20
                        ),
                        metrics.step,
                    ));
                }
            }

            AlertCondition::MemoryHigh { threshold } => {
                if let Some(usage) = metrics.memory_usage {
                    if usage > *threshold {
                        return Some(Alert::new(
                            AlertSeverity::Critical,
                            format!(
                                "High memory usage: {:.1}% (threshold: {:.1}%)",
                                usage * 100.0,
                                threshold * 100.0
                            ),
                            metrics.step,
                        ));
                    }
                }
            }

            AlertCondition::GradientExploding { threshold } => {
                if let Some(grad_norm) = metrics.gradient_norm {
                    if grad_norm > *threshold {
                        return Some(Alert::new(
                            AlertSeverity::Critical,
                            format!(
                                "Gradient explosion: norm={:.2e} (threshold: {:.2e})",
                                grad_norm, threshold
                            ),
                            metrics.step,
                        ));
                    }
                }
            }

            AlertCondition::GradientVanishing { threshold } => {
                if let Some(grad_norm) = metrics.gradient_norm {
                    if grad_norm < *threshold && grad_norm > 0.0 {
                        return Some(Alert::new(
                            AlertSeverity::Warning,
                            format!(
                                "Gradient vanishing: norm={:.2e} (threshold: {:.2e})",
                                grad_norm, threshold
                            ),
                            metrics.step,
                        ));
                    }
                }
            }
        }

        None
    }

    /// Add an alert to history
    fn add_alert(&mut self, alert: Alert) {
        // Send desktop notification if enabled
        if self.desktop_notifications {
            self.send_desktop_notification(&alert);
        }

        self.alerts.push_back(alert);

        // Enforce max history
        while self.alerts.len() > self.max_history {
            self.alerts.pop_front();
        }
    }

    /// Send desktop notification (requires notify-rust feature)
    #[allow(unused_variables)]
    fn send_desktop_notification(&self, alert: &Alert) {
        // This is a stub - desktop notifications would require notify-rust crate
        // which is not currently in dependencies. This can be enabled optionally.
        // For now, we just log the intent.
        tracing::debug!("Desktop notification: {}", alert.message);
    }

    /// Get all alerts
    pub fn get_alerts(&self) -> impl Iterator<Item = &Alert> {
        self.alerts.iter()
    }

    /// Get unacknowledged alerts
    pub fn get_unacknowledged(&self) -> impl Iterator<Item = &Alert> {
        self.alerts.iter().filter(|a| !a.acknowledged)
    }

    /// Get alerts by severity
    pub fn get_by_severity(&self, severity: AlertSeverity) -> impl Iterator<Item = &Alert> {
        self.alerts.iter().filter(move |a| a.severity == severity)
    }

    /// Acknowledge an alert by index
    pub fn acknowledge(&mut self, index: usize) -> bool {
        if let Some(alert) = self.alerts.get_mut(index) {
            alert.acknowledged = true;
            true
        } else {
            false
        }
    }

    /// Acknowledge all alerts
    pub fn acknowledge_all(&mut self) {
        for alert in &mut self.alerts {
            alert.acknowledged = true;
        }
    }

    /// Clear all alerts
    pub fn clear_alerts(&mut self) {
        self.alerts.clear();
    }

    /// Get alert count by severity
    pub fn count_by_severity(&self, severity: AlertSeverity) -> usize {
        self.alerts
            .iter()
            .filter(|a| a.severity == severity)
            .count()
    }

    /// Get total alert count
    pub fn count(&self) -> usize {
        self.alerts.len()
    }

    /// Export alerts to JSON
    pub fn export_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(&self.alerts)
    }

    /// Import alerts from JSON
    pub fn import_json(&mut self, json: &str) -> Result<(), serde_json::Error> {
        let alerts: Vec<Alert> = serde_json::from_str(json)?;
        for alert in alerts {
            self.add_alert(alert);
        }
        Ok(())
    }
}

impl Default for AlertManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alert_creation() {
        let alert = Alert::new(AlertSeverity::Warning, "Test alert".to_string(), 100);
        assert_eq!(alert.severity, AlertSeverity::Warning);
        assert_eq!(alert.message, "Test alert");
        assert_eq!(alert.step, 100);
        assert!(!alert.acknowledged);
    }

    #[test]
    fn test_loss_spike_detection() {
        let mut manager =
            AlertManager::with_conditions(vec![AlertCondition::LossSpike { threshold: 2.0 }]);

        // Build baseline
        for i in 0..20 {
            let metrics = MetricsSnapshot {
                step: i,
                loss: 1.0,
                gradient_norm: None,
                memory_usage: None,
            };
            manager.check_metrics(metrics);
        }

        // Trigger spike
        let metrics = MetricsSnapshot {
            step: 20,
            loss: 3.0, // 3x baseline
            gradient_norm: None,
            memory_usage: None,
        };
        let alerts = manager.check_metrics(metrics);
        assert_eq!(alerts.len(), 1);
        assert!(alerts[0].message.contains("Loss spike"));
    }

    #[test]
    fn test_plateau_detection() {
        let mut manager =
            AlertManager::with_conditions(vec![AlertCondition::Plateau { steps: 10 }]);

        // Establish best loss
        let metrics = MetricsSnapshot {
            step: 0,
            loss: 1.0,
            gradient_norm: None,
            memory_usage: None,
        };
        manager.check_metrics(metrics);

        // No improvement for 10 steps
        for i in 1..11 {
            let metrics = MetricsSnapshot {
                step: i,
                loss: 1.1, // Worse than best
                gradient_norm: None,
                memory_usage: None,
            };
            let alerts = manager.check_metrics(metrics);
            if i == 10 {
                assert_eq!(alerts.len(), 1);
                assert!(alerts[0].message.contains("plateau"));
            }
        }
    }

    #[test]
    fn test_gradient_explosion() {
        let mut manager = AlertManager::with_conditions(vec![AlertCondition::GradientExploding {
            threshold: 100.0,
        }]);

        let metrics = MetricsSnapshot {
            step: 1,
            loss: 1.0,
            gradient_norm: Some(150.0),
            memory_usage: None,
        };
        let alerts = manager.check_metrics(metrics);
        assert_eq!(alerts.len(), 1);
        assert!(alerts[0].message.contains("Gradient explosion"));
    }

    #[test]
    fn test_memory_high() {
        let mut manager =
            AlertManager::with_conditions(vec![AlertCondition::MemoryHigh { threshold: 0.9 }]);

        let metrics = MetricsSnapshot {
            step: 1,
            loss: 1.0,
            gradient_norm: None,
            memory_usage: Some(0.95),
        };
        let alerts = manager.check_metrics(metrics);
        assert_eq!(alerts.len(), 1);
        assert!(alerts[0].message.contains("High memory"));
    }

    #[test]
    fn test_acknowledge() {
        let mut manager = AlertManager::new();
        manager.add_alert(Alert::new(AlertSeverity::Info, "Test".to_string(), 1));

        assert_eq!(manager.get_unacknowledged().count(), 1);
        manager.acknowledge(0);
        assert_eq!(manager.get_unacknowledged().count(), 0);
    }

    #[test]
    fn test_severity_filtering() {
        let mut manager = AlertManager::new();
        manager.add_alert(Alert::new(AlertSeverity::Info, "Info".to_string(), 1));
        manager.add_alert(Alert::new(AlertSeverity::Warning, "Warning".to_string(), 2));
        manager.add_alert(Alert::new(
            AlertSeverity::Critical,
            "Critical".to_string(),
            3,
        ));

        assert_eq!(manager.count_by_severity(AlertSeverity::Info), 1);
        assert_eq!(manager.count_by_severity(AlertSeverity::Warning), 1);
        assert_eq!(manager.count_by_severity(AlertSeverity::Critical), 1);
    }

    #[test]
    fn test_json_export_import() {
        let mut manager = AlertManager::new();
        manager.add_alert(Alert::new(
            AlertSeverity::Warning,
            "Test alert".to_string(),
            100,
        ));

        let json = manager.export_json().unwrap();
        let mut manager2 = AlertManager::new();
        manager2.import_json(&json).unwrap();

        assert_eq!(manager2.count(), 1);
        assert_eq!(manager2.get_alerts().next().unwrap().message, "Test alert");
    }
}
