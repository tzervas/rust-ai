//! Training health analysis.
//!
//! Provides comprehensive health assessment of the training process,
//! including loss trends, gradient health, and actionable recommendations.

use serde::{Deserialize, Serialize};

use crate::curve_analysis::CurveTrend;

/// Overall training health status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Health {
    /// Training is progressing well
    Excellent,
    /// Training is healthy with minor issues
    Good,
    /// Training has some concerning trends
    Fair,
    /// Training has significant issues
    Poor,
    /// Training is failing or diverging
    Critical,
}

impl Health {
    /// Convert from a numeric score (0.0 - 1.0).
    pub fn from_score(score: f32) -> Self {
        if score >= 0.9 {
            Self::Excellent
        } else if score >= 0.7 {
            Self::Good
        } else if score >= 0.5 {
            Self::Fair
        } else if score >= 0.3 {
            Self::Poor
        } else {
            Self::Critical
        }
    }

    /// Convert to a numeric score (0.0 - 1.0).
    pub fn to_score(&self) -> f32 {
        match self {
            Self::Excellent => 0.95,
            Self::Good => 0.80,
            Self::Fair => 0.60,
            Self::Poor => 0.40,
            Self::Critical => 0.15,
        }
    }

    /// Human-readable description.
    pub fn description(&self) -> &'static str {
        match self {
            Self::Excellent => "Training is progressing excellently",
            Self::Good => "Training is healthy with minor issues",
            Self::Fair => "Training has some concerning trends",
            Self::Poor => "Training has significant issues",
            Self::Critical => "Training is failing or diverging",
        }
    }

    /// Whether training should continue.
    pub fn should_continue(&self) -> bool {
        !matches!(self, Self::Critical)
    }

    /// Whether intervention is recommended.
    pub fn needs_intervention(&self) -> bool {
        matches!(self, Self::Poor | Self::Critical)
    }
}

impl std::fmt::Display for Health {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Excellent => write!(f, "EXCELLENT"),
            Self::Good => write!(f, "GOOD"),
            Self::Fair => write!(f, "FAIR"),
            Self::Poor => write!(f, "POOR"),
            Self::Critical => write!(f, "CRITICAL"),
        }
    }
}

/// Gradient health status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GradientHealth {
    /// Gradients are healthy
    Healthy,
    /// Gradients show some instability
    Unstable,
    /// Gradients are vanishing
    Vanishing,
    /// Gradients are exploding
    Exploding,
    /// Mixed gradient issues
    Mixed,
}

impl GradientHealth {
    /// Create from gradient statistics.
    pub fn from_stats(mean: f32, std_dev: f32, min: f32, max: f32) -> Self {
        let variance_ratio = if mean > 1e-8 { std_dev / mean } else { 0.0 };

        // Check for exploding
        if max > 100.0 || (mean > 10.0 && variance_ratio > 2.0) {
            return Self::Exploding;
        }

        // Check for vanishing
        if max < 1e-6 || mean < 1e-5 {
            return Self::Vanishing;
        }

        // Check for instability
        if variance_ratio > 1.0 || (max / min.max(1e-8)) > 100.0 {
            return Self::Unstable;
        }

        // Check for mixed issues (some very small, some large)
        if min < 1e-5 && max > 10.0 {
            return Self::Mixed;
        }

        Self::Healthy
    }

    /// Human-readable description.
    pub fn description(&self) -> &'static str {
        match self {
            Self::Healthy => "Gradients are healthy",
            Self::Unstable => "Gradients show high variance",
            Self::Vanishing => "Gradients are too small",
            Self::Exploding => "Gradients are too large",
            Self::Mixed => "Mixed gradient issues detected",
        }
    }

    /// Whether this status indicates a problem.
    pub fn is_problematic(&self) -> bool {
        !matches!(self, Self::Healthy)
    }

    /// Suggested remediation.
    pub fn remediation(&self) -> Vec<&'static str> {
        match self {
            Self::Healthy => vec![],
            Self::Unstable => vec![
                "Consider reducing learning rate",
                "Try gradient clipping",
                "Increase batch size for stability",
            ],
            Self::Vanishing => vec![
                "Increase learning rate",
                "Check for proper weight initialization",
                "Consider using residual connections",
                "Check layer normalization",
            ],
            Self::Exploding => vec![
                "Reduce learning rate immediately",
                "Apply gradient clipping",
                "Check for NaN/Inf in inputs",
                "Reduce batch size",
            ],
            Self::Mixed => vec![
                "Review model architecture",
                "Check for layer-specific issues",
                "Consider per-layer learning rates",
            ],
        }
    }
}

impl std::fmt::Display for GradientHealth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Healthy => write!(f, "Healthy"),
            Self::Unstable => write!(f, "Unstable"),
            Self::Vanishing => write!(f, "Vanishing"),
            Self::Exploding => write!(f, "Exploding"),
            Self::Mixed => write!(f, "Mixed Issues"),
        }
    }
}

/// Comprehensive training health report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingHealthReport {
    /// Overall training health
    pub overall_health: Health,
    /// Numeric health score (0.0 - 1.0)
    pub health_score: f32,
    /// Current loss trend
    pub loss_trend: CurveTrend,
    /// Gradient health status
    pub gradient_health: GradientHealth,
    /// Recent loss values (for context)
    pub recent_losses: Vec<f32>,
    /// Recent gradient norms (for context)
    pub recent_gradients: Vec<f32>,
    /// Actionable recommendations
    pub recommendations: Vec<String>,
    /// Warnings (less critical than recommendations)
    pub warnings: Vec<String>,
    /// Current step
    pub step: u64,
    /// Steps since last significant improvement
    pub steps_since_improvement: u64,
}

impl TrainingHealthReport {
    /// Create a new health report.
    pub fn new(
        health_score: f32,
        loss_trend: CurveTrend,
        gradient_health: GradientHealth,
        step: u64,
        steps_since_improvement: u64,
    ) -> Self {
        let overall_health = Health::from_score(health_score);
        let mut recommendations = Vec::new();
        let mut warnings = Vec::new();

        // Add recommendations based on gradient health
        if gradient_health.is_problematic() {
            for remedy in gradient_health.remediation() {
                recommendations.push(remedy.to_string());
            }
        }

        // Add recommendations based on loss trend
        match loss_trend {
            CurveTrend::Diverging => {
                recommendations.push("CRITICAL: Loss is diverging! Reduce LR by 75%".to_string());
                recommendations.push("Check for data issues or bugs".to_string());
            }
            CurveTrend::Oscillating => {
                recommendations.push("High variance detected. Reduce LR by 50%".to_string());
                recommendations.push("Consider increasing batch size".to_string());
            }
            CurveTrend::Plateau => {
                if steps_since_improvement > 100 {
                    recommendations.push("Loss has plateaued. Try warmup restart".to_string());
                    recommendations.push("Consider increasing LR by 50%".to_string());
                } else {
                    warnings.push("Loss improvement slowing - monitor closely".to_string());
                }
            }
            CurveTrend::LinearDecay => {
                warnings.push("Linear decay is slower than expected".to_string());
                warnings.push("Consider increasing LR for faster convergence".to_string());
            }
            CurveTrend::HealthyDecay => {
                // No issues
            }
        }

        // Add warnings for overall health
        if overall_health.needs_intervention() {
            warnings.push(format!(
                "Overall health is {}. Consider pausing to investigate.",
                overall_health
            ));
        }

        Self {
            overall_health,
            health_score,
            loss_trend,
            gradient_health,
            recent_losses: Vec::new(),
            recent_gradients: Vec::new(),
            recommendations,
            warnings,
            step,
            steps_since_improvement,
        }
    }

    /// Add recent losses for context.
    pub fn with_recent_losses(mut self, losses: Vec<f32>) -> Self {
        self.recent_losses = losses;
        self
    }

    /// Add recent gradients for context.
    pub fn with_recent_gradients(mut self, gradients: Vec<f32>) -> Self {
        self.recent_gradients = gradients;
        self
    }

    /// Add a custom recommendation.
    pub fn add_recommendation(&mut self, recommendation: &str) {
        self.recommendations.push(recommendation.to_string());
    }

    /// Add a custom warning.
    pub fn add_warning(&mut self, warning: &str) {
        self.warnings.push(warning.to_string());
    }

    /// Check if there are any critical issues.
    pub fn has_critical_issues(&self) -> bool {
        matches!(self.overall_health, Health::Critical)
            || matches!(self.gradient_health, GradientHealth::Exploding)
            || matches!(self.loss_trend, CurveTrend::Diverging)
    }

    /// Check if intervention is needed.
    pub fn needs_intervention(&self) -> bool {
        self.overall_health.needs_intervention()
            || self.gradient_health.is_problematic()
            || !self.recommendations.is_empty()
    }

    /// Format the report as a human-readable string.
    pub fn format(&self) -> String {
        let mut lines = vec![
            format!("Training Health Report (Step {})", self.step),
            format!("═══════════════════════════════════"),
            format!(""),
            format!(
                "Overall Health: {} ({:.2}/1.00)",
                self.overall_health, self.health_score
            ),
            format!("Loss Trend: {}", self.loss_trend.description()),
            format!("Gradient Health: {}", self.gradient_health),
            format!("Steps Since Improvement: {}", self.steps_since_improvement),
        ];

        if !self.recent_losses.is_empty() {
            let recent_loss = self.recent_losses.last().copied().unwrap_or(0.0);
            let min_loss = self
                .recent_losses
                .iter()
                .copied()
                .fold(f32::INFINITY, f32::min);
            lines.push(format!(
                "Current Loss: {:.4} (Best: {:.4})",
                recent_loss, min_loss
            ));
        }

        if !self.recent_gradients.is_empty() {
            let avg_grad =
                self.recent_gradients.iter().sum::<f32>() / self.recent_gradients.len() as f32;
            lines.push(format!("Avg Gradient Norm: {:.4}", avg_grad));
        }

        if !self.recommendations.is_empty() {
            lines.push(format!(""));
            lines.push(format!("Recommendations:"));
            lines.push(format!("─────────────────"));
            for rec in &self.recommendations {
                lines.push(format!("  - {}", rec));
            }
        }

        if !self.warnings.is_empty() {
            lines.push(format!(""));
            lines.push(format!("Warnings:"));
            lines.push(format!("─────────"));
            for warning in &self.warnings {
                lines.push(format!("  - {}", warning));
            }
        }

        lines.join("\n")
    }

    /// Get a compact one-line summary.
    pub fn summary(&self) -> String {
        format!(
            "[{}] Health: {}, Loss: {}, Gradient: {}",
            self.step,
            self.overall_health,
            self.loss_trend.description(),
            self.gradient_health
        )
    }
}

impl Default for TrainingHealthReport {
    fn default() -> Self {
        Self::new(0.5, CurveTrend::Plateau, GradientHealth::Healthy, 0, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health_from_score() {
        assert_eq!(Health::from_score(0.95), Health::Excellent);
        assert_eq!(Health::from_score(0.75), Health::Good);
        assert_eq!(Health::from_score(0.55), Health::Fair);
        assert_eq!(Health::from_score(0.35), Health::Poor);
        assert_eq!(Health::from_score(0.15), Health::Critical);
    }

    #[test]
    fn test_health_properties() {
        assert!(Health::Excellent.should_continue());
        assert!(Health::Good.should_continue());
        assert!(Health::Fair.should_continue());
        assert!(Health::Poor.should_continue());
        assert!(!Health::Critical.should_continue());

        assert!(!Health::Excellent.needs_intervention());
        assert!(!Health::Good.needs_intervention());
        assert!(!Health::Fair.needs_intervention());
        assert!(Health::Poor.needs_intervention());
        assert!(Health::Critical.needs_intervention());
    }

    #[test]
    fn test_gradient_health_from_stats() {
        // Healthy gradients
        assert_eq!(
            GradientHealth::from_stats(0.5, 0.1, 0.3, 0.8),
            GradientHealth::Healthy
        );

        // Exploding gradients
        assert_eq!(
            GradientHealth::from_stats(50.0, 100.0, 0.1, 150.0),
            GradientHealth::Exploding
        );

        // Vanishing gradients
        assert_eq!(
            GradientHealth::from_stats(1e-7, 1e-8, 1e-8, 1e-7),
            GradientHealth::Vanishing
        );

        // Unstable gradients (high variance)
        assert_eq!(
            GradientHealth::from_stats(1.0, 2.0, 0.01, 10.0),
            GradientHealth::Unstable
        );
    }

    #[test]
    fn test_health_report_critical() {
        let report = TrainingHealthReport::new(
            0.15,
            CurveTrend::Diverging,
            GradientHealth::Exploding,
            100,
            50,
        );

        assert!(report.has_critical_issues());
        assert!(report.needs_intervention());
        assert!(!report.recommendations.is_empty());
    }

    #[test]
    fn test_health_report_healthy() {
        let report = TrainingHealthReport::new(
            0.85,
            CurveTrend::HealthyDecay,
            GradientHealth::Healthy,
            100,
            5,
        );

        assert!(!report.has_critical_issues());
        assert!(!report.needs_intervention());
        assert!(report.recommendations.is_empty());
    }

    #[test]
    fn test_health_report_format() {
        let report = TrainingHealthReport::new(
            0.75,
            CurveTrend::LinearDecay,
            GradientHealth::Healthy,
            1000,
            20,
        )
        .with_recent_losses(vec![2.5, 2.4, 2.3, 2.2])
        .with_recent_gradients(vec![0.5, 0.6, 0.55, 0.58]);

        let formatted = report.format();
        assert!(formatted.contains("Training Health Report"));
        assert!(formatted.contains("GOOD"));
        assert!(formatted.contains("Current Loss"));
    }
}
