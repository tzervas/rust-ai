//! Main adaptive training controller.
//!
//! Coordinates learning rate, batch size, gradient clipping, and warmup
//! adjustments based on real-time training dynamics.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::batch_tuner::BatchTuner;
use crate::curve_analysis::{CurveAnalyzer, CurveTrend};
use crate::gradient_control::{GradientAction, GradientController};
use crate::lr_controller::AdaptiveLRController;
use crate::training_state::StepMetrics;

use super::health::{GradientHealth, Health, TrainingHealthReport};
use super::history::AdaptationHistory;
use super::strategy::AdaptationStrategy;

/// Configuration for the adaptive training controller.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveConfig {
    /// Initial learning rate
    pub initial_lr: f32,
    /// Minimum learning rate
    pub min_lr: f32,
    /// Maximum learning rate
    pub max_lr: f32,
    /// Initial batch size
    pub initial_batch_size: usize,
    /// Minimum batch size
    pub min_batch_size: usize,
    /// Maximum batch size
    pub max_batch_size: usize,
    /// Initial gradient clipping threshold
    pub initial_grad_clip: f32,
    /// Minimum gradient clipping threshold
    pub min_grad_clip: f32,
    /// Maximum gradient clipping threshold
    pub max_grad_clip: f32,
    /// Initial warmup steps
    pub initial_warmup_steps: u64,
    /// Initial momentum beta1
    pub initial_momentum_beta1: f32,
    /// Model size in parameters (for curve analysis scaling)
    pub model_size_params: u64,
    /// GPU memory threshold for batch tuning (0.0 - 1.0)
    pub memory_threshold: f32,
    /// Initial adaptation strategy
    pub initial_strategy: AdaptationStrategy,
    /// Enable predictive adjustments based on curve analysis
    pub enable_predictive: bool,
    /// Steps between health checks
    pub health_check_interval: u64,
}

impl Default for AdaptiveConfig {
    fn default() -> Self {
        Self {
            initial_lr: 1e-4,
            min_lr: 1e-7,
            max_lr: 1e-2,
            initial_batch_size: 8,
            min_batch_size: 1,
            max_batch_size: 64,
            initial_grad_clip: 1.0,
            min_grad_clip: 0.1,
            max_grad_clip: 10.0,
            initial_warmup_steps: 100,
            initial_momentum_beta1: 0.9,
            model_size_params: 100_000_000,
            memory_threshold: 0.8,
            initial_strategy: AdaptationStrategy::Balanced,
            enable_predictive: true,
            health_check_interval: 50,
        }
    }
}

/// Parameter types that can be adapted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AdaptedParam {
    /// Learning rate
    LearningRate,
    /// Batch size
    BatchSize,
    /// Gradient clipping threshold
    GradientClip,
    /// Number of warmup steps
    WarmupSteps,
    /// Momentum beta1 coefficient
    MomentumBeta1,
}

impl std::fmt::Display for AdaptedParam {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LearningRate => write!(f, "learning_rate"),
            Self::BatchSize => write!(f, "batch_size"),
            Self::GradientClip => write!(f, "gradient_clip"),
            Self::WarmupSteps => write!(f, "warmup_steps"),
            Self::MomentumBeta1 => write!(f, "momentum_beta1"),
        }
    }
}

/// Record of a single adaptation event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Adaptation {
    /// Step when adaptation occurred
    pub step: u64,
    /// Parameter that was adapted
    pub param: AdaptedParam,
    /// Previous value
    pub old_value: f32,
    /// New value
    pub new_value: f32,
    /// Reason for the adaptation
    pub reason: String,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

impl Adaptation {
    /// Create a new adaptation record.
    pub fn new(
        step: u64,
        param: AdaptedParam,
        old_value: f32,
        new_value: f32,
        reason: &str,
    ) -> Self {
        Self {
            step,
            param,
            old_value,
            new_value,
            reason: reason.to_string(),
            timestamp: Utc::now(),
        }
    }

    /// Get the change factor (new/old).
    pub fn change_factor(&self) -> f32 {
        if self.old_value.abs() < 1e-10 {
            0.0
        } else {
            self.new_value / self.old_value
        }
    }

    /// Check if this is an increase.
    pub fn is_increase(&self) -> bool {
        self.new_value > self.old_value
    }

    /// Format as a log message.
    pub fn format_log(&self) -> String {
        let direction = if self.is_increase() {
            "increased"
        } else {
            "decreased"
        };
        let factor = self.change_factor();
        format!(
            "[Step {}] {} {} {:.2}x: {:.6} -> {:.6} ({})",
            self.step, self.param, direction, factor, self.old_value, self.new_value, self.reason
        )
    }
}

/// Adaptive training controller that coordinates all adaptation components.
pub struct AdaptiveTrainingController {
    /// Configuration
    config: AdaptiveConfig,
    /// Learning rate controller
    lr_controller: AdaptiveLRController,
    /// Batch size tuner
    batch_tuner: BatchTuner,
    /// Gradient clipping controller
    grad_controller: GradientController,
    /// Loss curve analyzer
    curve_analyzer: CurveAnalyzer,
    /// Adaptation history
    adaptation_history: AdaptationHistory,
    /// Current adaptation strategy
    current_strategy: AdaptationStrategy,
    /// Current step
    current_step: u64,
    /// Current warmup steps setting
    current_warmup_steps: u64,
    /// Current momentum beta1
    current_momentum_beta1: f32,
    /// Best loss seen so far
    best_loss: f32,
    /// Step at best loss
    best_loss_step: u64,
    /// Steps since last LR adaptation
    steps_since_lr_change: u64,
    /// Steps since last batch adaptation
    steps_since_batch_change: u64,
    /// Steps since last grad clip adaptation
    steps_since_grad_change: u64,
    /// Recent losses for health analysis
    recent_losses: Vec<f32>,
    /// Recent gradient norms for health analysis
    recent_gradients: Vec<f32>,
    /// Memory usage for batch tuning (externally provided)
    last_memory_usage: f32,
    /// Whether a recovery was triggered
    in_recovery: bool,
    /// Step when recovery started
    recovery_start_step: u64,
}

impl AdaptiveTrainingController {
    /// Create a new adaptive training controller.
    pub fn new(config: AdaptiveConfig) -> Self {
        let lr_controller =
            AdaptiveLRController::new(config.initial_lr, config.min_lr, config.max_lr);

        let batch_tuner = BatchTuner::new(
            config.initial_batch_size,
            config.min_batch_size,
            config.max_batch_size,
            config.memory_threshold,
        );

        let grad_controller = GradientController::with_config(
            config.initial_grad_clip,
            config.min_grad_clip,
            config.max_grad_clip,
            100,   // window size
            10.0,  // explosion factor
            0.001, // vanishing threshold
            0.1,   // adjustment rate
        );

        let curve_analyzer = CurveAnalyzer::new(config.model_size_params);

        Self {
            config: config.clone(),
            lr_controller,
            batch_tuner,
            grad_controller,
            curve_analyzer,
            adaptation_history: AdaptationHistory::new(10000),
            current_strategy: config.initial_strategy,
            current_step: 0,
            current_warmup_steps: config.initial_warmup_steps,
            current_momentum_beta1: config.initial_momentum_beta1,
            best_loss: f32::INFINITY,
            best_loss_step: 0,
            steps_since_lr_change: 0,
            steps_since_batch_change: 0,
            steps_since_grad_change: 0,
            recent_losses: Vec::with_capacity(100),
            recent_gradients: Vec::with_capacity(100),
            last_memory_usage: 0.0,
            in_recovery: false,
            recovery_start_step: 0,
        }
    }

    /// Create with a specific strategy.
    pub fn with_strategy(mut self, strategy: AdaptationStrategy) -> Self {
        self.current_strategy = strategy;
        self
    }

    /// Update the controller with new step metrics and return adaptations.
    ///
    /// This is the main entry point for the controller. Call this after each
    /// training step with the current metrics, and apply the returned adaptations.
    pub fn update(&mut self, metrics: &StepMetrics) -> Vec<Adaptation> {
        self.current_step = metrics.step;
        let mut adaptations = Vec::new();

        // Update internal state
        self.update_tracking(metrics);

        // Update component controllers
        let lr_result = self
            .lr_controller
            .update(metrics.loss, metrics.gradient_norm);
        let grad_action = self.grad_controller.update(metrics.gradient_norm);

        // Process learning rate adaptations
        if let Some(adaptation) = self.process_lr_result(lr_result, metrics) {
            adaptations.push(adaptation);
        }

        // Process gradient adaptations
        if let Some(adaptation) = self.process_grad_action(grad_action, metrics) {
            adaptations.push(adaptation);
        }

        // Process batch size (if memory info available)
        if self.last_memory_usage > 0.0 {
            if let Some(adaptation) = self.process_batch_tuning(metrics) {
                adaptations.push(adaptation);
            }
        }

        // Predictive adjustments based on curve analysis
        if self.config.enable_predictive && self.current_step > 50 {
            if let Some(adaptation) = self.predictive_adjustment(metrics) {
                adaptations.push(adaptation);
            }
        }

        // Check for recovery mode transitions
        self.check_recovery_transitions(metrics);

        // Auto-adjust strategy based on health
        if self.current_step % self.config.health_check_interval == 0 {
            self.auto_adjust_strategy();
        }

        // Record adaptations
        self.adaptation_history.record_batch(adaptations.clone());

        adaptations
    }

    /// Update tracking state with new metrics.
    fn update_tracking(&mut self, metrics: &StepMetrics) {
        // Track losses
        self.recent_losses.push(metrics.loss);
        if self.recent_losses.len() > 100 {
            self.recent_losses.remove(0);
        }

        // Track gradients
        self.recent_gradients.push(metrics.gradient_norm);
        if self.recent_gradients.len() > 100 {
            self.recent_gradients.remove(0);
        }

        // Update curve analyzer
        self.curve_analyzer.add_loss(metrics.loss);

        // Track best loss
        if metrics.loss < self.best_loss {
            self.best_loss = metrics.loss;
            self.best_loss_step = metrics.step;
        }

        // Increment step counters
        self.steps_since_lr_change += 1;
        self.steps_since_batch_change += 1;
        self.steps_since_grad_change += 1;
    }

    /// Process learning rate controller result.
    fn process_lr_result(&mut self, _new_lr: f32, metrics: &StepMetrics) -> Option<Adaptation> {
        let old_lr = self.lr_controller.base_lr();
        let current_lr = self.lr_controller.current_lr();

        // Check if there's a significant change
        let change_ratio = (current_lr - old_lr).abs() / old_lr.max(1e-10);

        // Determine if this is a plateau escape (LR increase)
        let is_plateau_escape = current_lr > old_lr && self.lr_controller.should_warmup();

        // Apply strategy-specific thresholds
        // Use lower threshold for plateau escapes to ensure they're logged
        let threshold = if is_plateau_escape {
            0.01 // Log plateau escapes at just 1% change
        } else {
            match self.current_strategy {
                AdaptationStrategy::Conservative => 0.05,
                AdaptationStrategy::Balanced => 0.10,
                AdaptationStrategy::Aggressive => 0.20,
                AdaptationStrategy::Exploratory => 0.15,
                AdaptationStrategy::Recovery => 0.03,
            }
        };

        if change_ratio < threshold {
            return None;
        }

        // Cooldown check - shorter cooldown for plateau escapes
        let cooldown = if is_plateau_escape {
            self.current_strategy.cooldown_steps() / 2
        } else {
            self.current_strategy.cooldown_steps()
        };

        if self.steps_since_lr_change < cooldown {
            return None;
        }

        self.steps_since_lr_change = 0;

        // Determine reason based on controller state
        let reason: String = if current_lr < old_lr {
            if self.lr_controller.should_reduce() {
                "Oscillation detected".to_string()
            } else {
                "Spike mitigation".to_string()
            }
        } else if is_plateau_escape {
            // Include debug info for plateau escapes
            if let Some((slope, mean, norm_slope)) = self.lr_controller.plateau_debug_info() {
                format!(
                    "Plateau escape (slope={:.2e}, mean={:.3}, norm={:.4})",
                    slope, mean, norm_slope
                )
            } else {
                "Plateau escape".to_string()
            }
        } else {
            "LR adjustment".to_string()
        };

        Some(Adaptation::new(
            metrics.step,
            AdaptedParam::LearningRate,
            old_lr,
            current_lr,
            &reason,
        ))
    }

    /// Process gradient controller action.
    fn process_grad_action(
        &mut self,
        action: GradientAction,
        metrics: &StepMetrics,
    ) -> Option<Adaptation> {
        let old_clip = self.grad_controller.current_threshold();

        match action {
            GradientAction::EmergencyClip => {
                // Emergency: reduce clip threshold significantly
                let new_clip = (old_clip * self.current_strategy.grad_clip_factor())
                    .max(self.config.min_grad_clip);
                self.grad_controller.set_threshold(new_clip);
                self.steps_since_grad_change = 0;
                Some(Adaptation::new(
                    metrics.step,
                    AdaptedParam::GradientClip,
                    old_clip,
                    new_clip,
                    "Gradient explosion emergency",
                ))
            }
            GradientAction::ReduceThreshold => {
                if self.steps_since_grad_change < self.current_strategy.cooldown_steps() {
                    return None;
                }
                let new_clip = self.grad_controller.current_threshold();
                if (new_clip - old_clip).abs() > 0.01 {
                    self.steps_since_grad_change = 0;
                    Some(Adaptation::new(
                        metrics.step,
                        AdaptedParam::GradientClip,
                        old_clip,
                        new_clip,
                        "Consistently high gradients",
                    ))
                } else {
                    None
                }
            }
            GradientAction::IncreaseThreshold => {
                if self.steps_since_grad_change < self.current_strategy.cooldown_steps() {
                    return None;
                }
                let new_clip = self.grad_controller.current_threshold();
                if (new_clip - old_clip).abs() > 0.01 {
                    self.steps_since_grad_change = 0;
                    Some(Adaptation::new(
                        metrics.step,
                        AdaptedParam::GradientClip,
                        old_clip,
                        new_clip,
                        "Consistently low gradients",
                    ))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Process batch size tuning.
    fn process_batch_tuning(&mut self, metrics: &StepMetrics) -> Option<Adaptation> {
        if self.steps_since_batch_change < self.current_strategy.cooldown_steps() {
            return None;
        }

        let old_batch = self.batch_tuner.current_batch_size();
        let suggested = self.batch_tuner.suggest_batch_size(self.last_memory_usage);

        if suggested == old_batch {
            return None;
        }

        // Record the step
        self.batch_tuner
            .record_step(old_batch, self.last_memory_usage, true);
        self.steps_since_batch_change = 0;

        let reason = if suggested > old_batch {
            "Memory headroom available"
        } else {
            "Memory pressure"
        };

        Some(Adaptation::new(
            metrics.step,
            AdaptedParam::BatchSize,
            old_batch as f32,
            suggested as f32,
            reason,
        ))
    }

    /// Make predictive adjustments based on curve analysis.
    fn predictive_adjustment(&mut self, metrics: &StepMetrics) -> Option<Adaptation> {
        let analysis = self.curve_analyzer.analyze();

        // Only act on strong signals
        if analysis.health_score > 0.7 {
            return None;
        }

        match analysis.trend {
            CurveTrend::Diverging => {
                // Critical: reduce LR immediately
                let old_lr = self.lr_controller.current_lr();
                let new_lr = (old_lr * 0.25).max(self.config.min_lr);

                // Force the controller to this LR
                self.force_lr(new_lr);
                self.steps_since_lr_change = 0;

                // Switch to recovery mode
                if !self.in_recovery {
                    self.enter_recovery();
                }

                Some(Adaptation::new(
                    metrics.step,
                    AdaptedParam::LearningRate,
                    old_lr,
                    new_lr,
                    "Predictive: divergence detected",
                ))
            }
            CurveTrend::Plateau if analysis.optimal_stop_step.is_none() => {
                // Try to escape plateau with warmup restart
                if self.steps_since_lr_change > 100 {
                    let old_warmup = self.current_warmup_steps;
                    let new_warmup = (old_warmup + 50).min(500);

                    if new_warmup != old_warmup {
                        self.current_warmup_steps = new_warmup;

                        return Some(Adaptation::new(
                            metrics.step,
                            AdaptedParam::WarmupSteps,
                            old_warmup as f32,
                            new_warmup as f32,
                            "Predictive: plateau escape warmup restart",
                        ));
                    }
                }
                None
            }
            CurveTrend::Oscillating if analysis.noise_ratio < 2.0 => {
                // High noise: reduce momentum for stability
                let old_beta = self.current_momentum_beta1;
                if old_beta > 0.85 {
                    let new_beta = (old_beta - 0.05).max(0.8);
                    self.current_momentum_beta1 = new_beta;

                    return Some(Adaptation::new(
                        metrics.step,
                        AdaptedParam::MomentumBeta1,
                        old_beta,
                        new_beta,
                        "Predictive: reduce momentum for stability",
                    ));
                }
                None
            }
            _ => None,
        }
    }

    /// Check and handle recovery mode transitions.
    fn check_recovery_transitions(&mut self, metrics: &StepMetrics) {
        if self.in_recovery {
            // Check if we should exit recovery
            let steps_in_recovery = metrics.step.saturating_sub(self.recovery_start_step);
            let health = self.get_health_report();

            if steps_in_recovery > 200
                && matches!(health.overall_health, Health::Good | Health::Excellent)
            {
                self.exit_recovery();
            }
        } else {
            // Check if we should enter recovery
            if self.recent_losses.len() >= 20 {
                let recent_avg = self.recent_losses.iter().rev().take(10).sum::<f32>() / 10.0;
                let older_avg = self
                    .recent_losses
                    .iter()
                    .rev()
                    .skip(10)
                    .take(10)
                    .sum::<f32>()
                    / 10.0;

                // Significant regression
                if recent_avg > older_avg * 1.5 && older_avg > 0.0 {
                    self.enter_recovery();
                }
            }
        }
    }

    /// Enter recovery mode.
    fn enter_recovery(&mut self) {
        if !self.in_recovery {
            let old_strategy = self.current_strategy;
            self.current_strategy = AdaptationStrategy::Recovery;
            self.in_recovery = true;
            self.recovery_start_step = self.current_step;

            self.adaptation_history.record_strategy_change(
                self.current_step,
                old_strategy,
                AdaptationStrategy::Recovery,
                "Training instability detected",
            );
        }
    }

    /// Exit recovery mode.
    fn exit_recovery(&mut self) {
        if self.in_recovery {
            let old_strategy = self.current_strategy;
            self.current_strategy = AdaptationStrategy::Conservative; // Be conservative after recovery
            self.in_recovery = false;

            self.adaptation_history.record_strategy_change(
                self.current_step,
                old_strategy,
                AdaptationStrategy::Conservative,
                "Training stabilized - exiting recovery",
            );
        }
    }

    /// Auto-adjust strategy based on training health.
    fn auto_adjust_strategy(&mut self) {
        // Don't auto-adjust during recovery
        if self.in_recovery {
            return;
        }

        let health = self.get_health_report();
        let loss_improving = if self.recent_losses.len() >= 20 {
            let recent = self.recent_losses.iter().rev().take(10).sum::<f32>() / 10.0;
            let older = self
                .recent_losses
                .iter()
                .rev()
                .skip(10)
                .take(10)
                .sum::<f32>()
                / 10.0;
            recent < older
        } else {
            true
        };

        let recommended =
            AdaptationStrategy::recommended_for_health(health.health_score, loss_improving);

        // Only change if significantly different
        if recommended != self.current_strategy {
            let should_change = match (&self.current_strategy, &recommended) {
                // Always switch to Recovery if recommended
                (_, AdaptationStrategy::Recovery) => true,
                // Always exit Recovery when stable
                (AdaptationStrategy::Recovery, _) => true,
                // Otherwise, require consistent signals
                _ => self.current_step % 200 == 0,
            };

            if should_change {
                let old = self.current_strategy;
                self.current_strategy = recommended;
                self.adaptation_history.record_strategy_change(
                    self.current_step,
                    old,
                    recommended,
                    "Auto-adjusted based on training health",
                );
            }
        }
    }

    /// Set GPU memory usage for batch tuning.
    pub fn set_memory_usage(&mut self, usage_pct: f32) {
        self.last_memory_usage = usage_pct;
    }

    /// Get the current learning rate.
    pub fn current_lr(&self) -> f32 {
        self.lr_controller.current_lr()
    }

    /// Get the current batch size.
    pub fn current_batch_size(&self) -> usize {
        self.batch_tuner.current_batch_size()
    }

    /// Get the current gradient clip threshold.
    pub fn current_grad_clip(&self) -> f32 {
        self.grad_controller.current_threshold()
    }

    /// Get current warmup steps.
    pub fn current_warmup_steps(&self) -> u64 {
        self.current_warmup_steps
    }

    /// Get current momentum beta1.
    pub fn current_momentum_beta1(&self) -> f32 {
        self.current_momentum_beta1
    }

    /// Set the adaptation strategy.
    pub fn set_strategy(&mut self, strategy: AdaptationStrategy) {
        if strategy != self.current_strategy {
            self.adaptation_history.record_strategy_change(
                self.current_step,
                self.current_strategy,
                strategy,
                "Manual strategy change",
            );
            self.current_strategy = strategy;
        }
    }

    /// Get the current strategy.
    pub fn current_strategy(&self) -> AdaptationStrategy {
        self.current_strategy
    }

    /// Force a specific learning rate (bypasses controller).
    pub fn force_lr(&mut self, lr: f32) {
        // Reset the controller with new base LR
        self.lr_controller = AdaptiveLRController::new(lr, self.config.min_lr, self.config.max_lr);
    }

    /// Force a specific batch size (bypasses tuner).
    pub fn force_batch_size(&mut self, batch_size: usize) {
        // Recreate tuner with new initial size
        self.batch_tuner = BatchTuner::new(
            batch_size,
            self.config.min_batch_size,
            self.config.max_batch_size,
            self.config.memory_threshold,
        );
    }

    /// Get the adaptation history.
    pub fn get_adaptation_history(&self) -> &AdaptationHistory {
        &self.adaptation_history
    }

    /// Get a comprehensive health report.
    pub fn get_health_report(&self) -> TrainingHealthReport {
        // Analyze curve
        let curve_analysis = self.curve_analyzer.analyze();

        // Compute gradient health
        let grad_health = if let Some(stats) = self.grad_controller.stats() {
            GradientHealth::from_stats(stats.mean, stats.std_dev, stats.min, stats.max)
        } else {
            GradientHealth::Healthy
        };

        // Compute steps since improvement
        let steps_since_improvement = self.current_step.saturating_sub(self.best_loss_step);

        // Create report
        let mut report = TrainingHealthReport::new(
            curve_analysis.health_score,
            curve_analysis.trend,
            grad_health,
            self.current_step,
            steps_since_improvement,
        );

        // Add context
        report.recent_losses = self.recent_losses.clone();
        report.recent_gradients = self.recent_gradients.clone();

        // Add recovery mode warning
        if self.in_recovery {
            report.add_warning(&format!(
                "In recovery mode since step {}",
                self.recovery_start_step
            ));
        }

        // Add strategy info
        report.add_warning(&format!("Current strategy: {}", self.current_strategy));

        report
    }

    /// Reset the controller to initial state.
    pub fn reset(&mut self) {
        self.lr_controller.reset();
        self.batch_tuner.reset();
        self.grad_controller.reset();
        self.curve_analyzer.clear();
        self.adaptation_history.clear();
        self.current_strategy = self.config.initial_strategy;
        self.current_step = 0;
        self.current_warmup_steps = self.config.initial_warmup_steps;
        self.current_momentum_beta1 = self.config.initial_momentum_beta1;
        self.best_loss = f32::INFINITY;
        self.best_loss_step = 0;
        self.steps_since_lr_change = 0;
        self.steps_since_batch_change = 0;
        self.steps_since_grad_change = 0;
        self.recent_losses.clear();
        self.recent_gradients.clear();
        self.last_memory_usage = 0.0;
        self.in_recovery = false;
        self.recovery_start_step = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::training_state::TrainingPhase;

    fn make_metrics(step: u64, loss: f32, grad_norm: f32) -> StepMetrics {
        StepMetrics {
            step,
            loss,
            gradient_norm: grad_norm,
            phase: TrainingPhase::Full,
            was_predicted: false,
            prediction_error: None,
            step_time_ms: 100.0,
            timestamp: Utc::now(),
            tokens_this_step: 4096,
            total_tokens_trained: step * 4096,
            tokens_remaining: 1000000,
            confidence: 0.9,
            learning_rate: 1e-4,
            perplexity: loss.exp(),
            train_val_gap: None,
            loss_velocity: 0.0,
            loss_acceleration: 0.0,
            gradient_entropy: None,
            layer_gradients: None,
            layer_gradient_stats: None,
        }
    }

    #[test]
    fn test_controller_creation() {
        let controller = AdaptiveTrainingController::new(AdaptiveConfig::default());
        assert_eq!(controller.current_lr(), 1e-4);
        assert_eq!(controller.current_batch_size(), 8);
        assert_eq!(controller.current_grad_clip(), 1.0);
    }

    #[test]
    fn test_update_tracking() {
        let mut controller = AdaptiveTrainingController::new(AdaptiveConfig::default());

        for step in 0..50 {
            let loss = 3.0 - (step as f32 * 0.01);
            let metrics = make_metrics(step, loss, 0.5);
            controller.update(&metrics);
        }

        assert_eq!(controller.current_step, 49);
        assert!(controller.best_loss < 3.0);
        assert_eq!(controller.recent_losses.len(), 50);
    }

    #[test]
    fn test_oscillation_triggers_lr_reduction() {
        let mut controller = AdaptiveTrainingController::new(AdaptiveConfig::default());
        let initial_lr = controller.current_lr();

        // Generate heavily oscillating loss pattern
        // Using 4.0 and 1.5 gives std/mean ≈ 0.45 which is > 0.3 threshold
        for step in 0..100 {
            let loss = if step % 2 == 0 { 4.0 } else { 1.5 };
            let metrics = make_metrics(step, loss, 0.5);
            controller.update(&metrics);
        }

        // Oscillation should have caused LR to decrease from initial
        // The lr_controller internally detects high variance and reduces LR
        assert!(
            controller.current_lr() < initial_lr,
            "Final LR {} should be less than initial {} due to oscillation",
            controller.current_lr(),
            initial_lr
        );
    }

    #[test]
    fn test_gradient_explosion_triggers_clip_reduction() {
        let mut controller = AdaptiveTrainingController::new(AdaptiveConfig::default());

        // Normal gradients first
        for step in 0..30 {
            let metrics = make_metrics(step, 2.5, 0.5);
            controller.update(&metrics);
        }

        // Then explosion
        let metrics = make_metrics(30, 2.5, 50.0);
        let adaptations = controller.update(&metrics);

        // Should trigger gradient clip adaptation
        let has_grad_clip = adaptations
            .iter()
            .any(|a| matches!(a.param, AdaptedParam::GradientClip));
        assert!(has_grad_clip);
    }

    #[test]
    fn test_strategy_change() {
        let mut controller = AdaptiveTrainingController::new(AdaptiveConfig::default());

        assert_eq!(controller.current_strategy(), AdaptationStrategy::Balanced);

        controller.set_strategy(AdaptationStrategy::Aggressive);
        assert_eq!(
            controller.current_strategy(),
            AdaptationStrategy::Aggressive
        );

        // Check history recorded
        assert_eq!(controller.adaptation_history.strategy_changes().len(), 1);
    }

    #[test]
    fn test_force_lr() {
        let mut controller = AdaptiveTrainingController::new(AdaptiveConfig::default());

        controller.force_lr(0.001);
        assert!((controller.current_lr() - 0.001).abs() < 1e-6);
    }

    #[test]
    fn test_health_report() {
        let mut controller = AdaptiveTrainingController::new(AdaptiveConfig::default());

        // Generate some training data
        for step in 0..100 {
            let loss = 3.0 - (step as f32 * 0.01);
            let metrics = make_metrics(step, loss, 0.5);
            controller.update(&metrics);
        }

        let report = controller.get_health_report();
        assert!(report.health_score > 0.0);
        assert_eq!(report.step, 99);
    }

    #[test]
    fn test_memory_triggers_batch_adjustment() {
        let mut controller = AdaptiveTrainingController::new(AdaptiveConfig::default());

        // Set high memory usage
        controller.set_memory_usage(0.95);

        // Run a few steps to allow batch adjustment
        for step in 0..50 {
            let metrics = make_metrics(step, 2.5, 0.5);
            let adaptations = controller.update(&metrics);

            for adaptation in &adaptations {
                if matches!(adaptation.param, AdaptedParam::BatchSize) {
                    assert!(adaptation.new_value < adaptation.old_value);
                }
            }
        }
    }

    #[test]
    fn test_reset() {
        let mut controller = AdaptiveTrainingController::new(AdaptiveConfig::default());

        // Make some changes
        for step in 0..50 {
            let metrics = make_metrics(step, 2.5, 0.5);
            controller.update(&metrics);
        }
        controller.set_strategy(AdaptationStrategy::Aggressive);

        // Reset
        controller.reset();

        assert_eq!(controller.current_step, 0);
        assert_eq!(controller.current_strategy(), AdaptationStrategy::Balanced);
        assert!(controller.recent_losses.is_empty());
        assert_eq!(controller.adaptation_history.total_count(), 0);
    }
}
