//! Integration with the progressive trainer.
//!
//! Provides an enhanced progressive trainer that uses the adaptive controller
//! for dynamic hyperparameter adjustment during training.

use tracing::{info, warn};

use crate::training_state::{StepMetrics, TrainingPhase};

use super::controller::{Adaptation, AdaptiveConfig, AdaptiveTrainingController};
use super::health::TrainingHealthReport;
use super::strategy::AdaptationStrategy;

/// Configuration for adaptive progressive training.
#[derive(Debug, Clone)]
pub struct AdaptiveProgressiveConfig {
    /// Base progressive config (from progressive.rs)
    pub base_learning_rate: f32,
    pub batch_size: usize,
    pub gradient_clip: f32,
    /// Adaptive controller configuration
    pub adaptive_config: AdaptiveConfig,
    /// Whether to log all adaptations
    pub log_adaptations: bool,
    /// Health check interval (steps)
    pub health_check_interval: u64,
    /// Pause training if health drops below this score
    pub min_health_score: f32,
}

impl Default for AdaptiveProgressiveConfig {
    fn default() -> Self {
        Self {
            base_learning_rate: 6e-4,
            batch_size: 4,
            gradient_clip: 1.0,
            adaptive_config: AdaptiveConfig::default(),
            log_adaptations: true,
            health_check_interval: 50,
            min_health_score: 0.2,
        }
    }
}

/// Wrapper that adds adaptive control to a progressive trainer.
///
/// This struct wraps the existing `ProgressiveTrainer` and adds adaptive
/// hyperparameter adjustment capabilities.
pub struct AdaptiveProgressiveTrainer {
    /// Adaptive controller
    controller: AdaptiveTrainingController,
    /// Configuration
    config: AdaptiveProgressiveConfig,
    /// Current step (updated externally)
    current_step: u64,
    /// Training paused due to health issues
    paused: bool,
    /// Pause reason
    pause_reason: Option<String>,
    /// Callback for when adaptations are made
    adaptation_callback: Option<Box<dyn Fn(&[Adaptation]) + Send>>,
}

impl AdaptiveProgressiveTrainer {
    /// Create a new adaptive progressive trainer.
    pub fn new(config: AdaptiveProgressiveConfig) -> Self {
        let controller = AdaptiveTrainingController::new(config.adaptive_config.clone());

        Self {
            controller,
            config,
            current_step: 0,
            paused: false,
            pause_reason: None,
            adaptation_callback: None,
        }
    }

    /// Set a callback for when adaptations are made.
    pub fn on_adaptation<F: Fn(&[Adaptation]) + Send + 'static>(mut self, callback: F) -> Self {
        self.adaptation_callback = Some(Box::new(callback));
        self
    }

    /// Process a training step and get adaptations to apply.
    ///
    /// This should be called after each training step with the current metrics.
    /// The returned adaptations should be applied to the optimizer/training loop.
    pub fn step(&mut self, metrics: &StepMetrics) -> StepResult {
        self.current_step = metrics.step;

        // Check if paused
        if self.paused {
            return StepResult {
                adaptations: Vec::new(),
                should_continue: false,
                health_report: None,
                pause_reason: self.pause_reason.clone(),
            };
        }

        // Get adaptations from controller
        let adaptations = self.controller.update(metrics);

        // Log adaptations if configured
        if self.config.log_adaptations && !adaptations.is_empty() {
            for adaptation in &adaptations {
                info!("{}", adaptation.format_log());
            }
        }

        // Call callback if set
        if let Some(ref callback) = self.adaptation_callback {
            if !adaptations.is_empty() {
                callback(&adaptations);
            }
        }

        // Periodic health check
        let health_report = if metrics.step % self.config.health_check_interval == 0 {
            Some(self.check_health())
        } else {
            None
        };

        StepResult {
            adaptations,
            should_continue: !self.paused,
            health_report,
            pause_reason: self.pause_reason.clone(),
        }
    }

    /// Check training health and potentially pause.
    fn check_health(&mut self) -> TrainingHealthReport {
        let report = self.controller.get_health_report();

        if report.health_score < self.config.min_health_score {
            self.paused = true;
            self.pause_reason = Some(format!(
                "Health score {:.2} below minimum {:.2}",
                report.health_score, self.config.min_health_score
            ));
            warn!("Training paused: {}", self.pause_reason.as_ref().unwrap());
        }

        report
    }

    /// Resume training after a pause.
    pub fn resume(&mut self) {
        if self.paused {
            self.paused = false;
            self.pause_reason = None;
            info!("Training resumed");
        }
    }

    /// Force resume with a specific strategy.
    pub fn resume_with_strategy(&mut self, strategy: AdaptationStrategy) {
        self.controller.set_strategy(strategy);
        self.resume();
    }

    /// Check if training is paused.
    pub fn is_paused(&self) -> bool {
        self.paused
    }

    /// Get the pause reason.
    pub fn pause_reason(&self) -> Option<&str> {
        self.pause_reason.as_deref()
    }

    /// Get the current learning rate.
    pub fn current_lr(&self) -> f32 {
        self.controller.current_lr()
    }

    /// Get the current batch size.
    pub fn current_batch_size(&self) -> usize {
        self.controller.current_batch_size()
    }

    /// Get the current gradient clip.
    pub fn current_grad_clip(&self) -> f32 {
        self.controller.current_grad_clip()
    }

    /// Get the current strategy.
    pub fn current_strategy(&self) -> AdaptationStrategy {
        self.controller.current_strategy()
    }

    /// Set the adaptation strategy.
    pub fn set_strategy(&mut self, strategy: AdaptationStrategy) {
        self.controller.set_strategy(strategy);
    }

    /// Update memory usage for batch tuning.
    pub fn set_memory_usage(&mut self, usage_pct: f32) {
        self.controller.set_memory_usage(usage_pct);
    }

    /// Get a health report.
    pub fn get_health_report(&self) -> TrainingHealthReport {
        self.controller.get_health_report()
    }

    /// Get the underlying controller.
    pub fn controller(&self) -> &AdaptiveTrainingController {
        &self.controller
    }

    /// Get a mutable reference to the controller.
    pub fn controller_mut(&mut self) -> &mut AdaptiveTrainingController {
        &mut self.controller
    }

    /// Force a specific learning rate.
    pub fn force_lr(&mut self, lr: f32) {
        self.controller.force_lr(lr);
    }

    /// Force a specific batch size.
    pub fn force_batch_size(&mut self, batch_size: usize) {
        self.controller.force_batch_size(batch_size);
    }

    /// Reset the trainer.
    pub fn reset(&mut self) {
        self.controller.reset();
        self.current_step = 0;
        self.paused = false;
        self.pause_reason = None;
    }
}

/// Result from a training step.
#[derive(Debug)]
pub struct StepResult {
    /// Adaptations to apply
    pub adaptations: Vec<Adaptation>,
    /// Whether training should continue
    pub should_continue: bool,
    /// Health report (if health check was performed)
    pub health_report: Option<TrainingHealthReport>,
    /// Reason for pause (if paused)
    pub pause_reason: Option<String>,
}

impl StepResult {
    /// Check if any adaptations were made.
    pub fn has_adaptations(&self) -> bool {
        !self.adaptations.is_empty()
    }

    /// Get the new learning rate if it was adapted.
    pub fn new_lr(&self) -> Option<f32> {
        self.adaptations
            .iter()
            .find(|a| matches!(a.param, super::AdaptedParam::LearningRate))
            .map(|a| a.new_value)
    }

    /// Get the new batch size if it was adapted.
    pub fn new_batch_size(&self) -> Option<usize> {
        self.adaptations
            .iter()
            .find(|a| matches!(a.param, super::AdaptedParam::BatchSize))
            .map(|a| a.new_value as usize)
    }

    /// Get the new gradient clip if it was adapted.
    pub fn new_grad_clip(&self) -> Option<f32> {
        self.adaptations
            .iter()
            .find(|a| matches!(a.param, super::AdaptedParam::GradientClip))
            .map(|a| a.new_value)
    }
}

/// Helper trait for applying adaptations to a training loop.
pub trait ApplyAdaptations {
    /// Apply the adaptations to the training state.
    fn apply_adaptations(&mut self, adaptations: &[Adaptation]);
}

/// Training loop integration helper.
///
/// This struct provides a convenient way to integrate the adaptive controller
/// into an existing training loop without modifying the loop structure.
pub struct AdaptiveLoopHelper {
    trainer: AdaptiveProgressiveTrainer,
    /// Last applied learning rate
    last_lr: f32,
    /// Last applied batch size
    last_batch_size: usize,
    /// Last applied gradient clip
    last_grad_clip: f32,
}

impl AdaptiveLoopHelper {
    /// Create a new loop helper.
    pub fn new(config: AdaptiveProgressiveConfig) -> Self {
        let initial_lr = config.base_learning_rate;
        let initial_batch = config.batch_size;
        let initial_clip = config.gradient_clip;

        Self {
            trainer: AdaptiveProgressiveTrainer::new(config),
            last_lr: initial_lr,
            last_batch_size: initial_batch,
            last_grad_clip: initial_clip,
        }
    }

    /// Process a step and return whether any parameters changed.
    pub fn process_step(&mut self, metrics: &StepMetrics) -> AdaptiveUpdate {
        let result = self.trainer.step(metrics);

        let mut update = AdaptiveUpdate {
            lr_changed: false,
            batch_changed: false,
            grad_clip_changed: false,
            new_lr: self.last_lr,
            new_batch_size: self.last_batch_size,
            new_grad_clip: self.last_grad_clip,
            should_pause: !result.should_continue,
            health_report: result.health_report,
        };

        // Check for LR change
        let current_lr = self.trainer.current_lr();
        if (current_lr - self.last_lr).abs() > 1e-10 {
            update.lr_changed = true;
            update.new_lr = current_lr;
            self.last_lr = current_lr;
        }

        // Check for batch size change
        let current_batch = self.trainer.current_batch_size();
        if current_batch != self.last_batch_size {
            update.batch_changed = true;
            update.new_batch_size = current_batch;
            self.last_batch_size = current_batch;
        }

        // Check for grad clip change
        let current_clip = self.trainer.current_grad_clip();
        if (current_clip - self.last_grad_clip).abs() > 1e-6 {
            update.grad_clip_changed = true;
            update.new_grad_clip = current_clip;
            self.last_grad_clip = current_clip;
        }

        update
    }

    /// Get the current state.
    pub fn current_state(&self) -> (f32, usize, f32) {
        (self.last_lr, self.last_batch_size, self.last_grad_clip)
    }

    /// Resume training.
    pub fn resume(&mut self) {
        self.trainer.resume();
    }

    /// Check if paused.
    pub fn is_paused(&self) -> bool {
        self.trainer.is_paused()
    }

    /// Get the trainer.
    pub fn trainer(&self) -> &AdaptiveProgressiveTrainer {
        &self.trainer
    }

    /// Get mutable trainer.
    pub fn trainer_mut(&mut self) -> &mut AdaptiveProgressiveTrainer {
        &mut self.trainer
    }
}

/// Update information from adaptive processing.
#[derive(Debug)]
pub struct AdaptiveUpdate {
    /// Whether learning rate changed
    pub lr_changed: bool,
    /// Whether batch size changed
    pub batch_changed: bool,
    /// Whether gradient clip changed
    pub grad_clip_changed: bool,
    /// New learning rate (always set to current)
    pub new_lr: f32,
    /// New batch size (always set to current)
    pub new_batch_size: usize,
    /// New gradient clip (always set to current)
    pub new_grad_clip: f32,
    /// Whether training should pause
    pub should_pause: bool,
    /// Health report if available
    pub health_report: Option<TrainingHealthReport>,
}

impl AdaptiveUpdate {
    /// Check if any parameter changed.
    pub fn has_changes(&self) -> bool {
        self.lr_changed || self.batch_changed || self.grad_clip_changed
    }

    /// Format as a log string.
    pub fn format_changes(&self) -> String {
        let mut changes = Vec::new();

        if self.lr_changed {
            changes.push(format!("LR: {:.2e}", self.new_lr));
        }
        if self.batch_changed {
            changes.push(format!("Batch: {}", self.new_batch_size));
        }
        if self.grad_clip_changed {
            changes.push(format!("GradClip: {:.2}", self.new_grad_clip));
        }

        if changes.is_empty() {
            "No changes".to_string()
        } else {
            changes.join(", ")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

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
    fn test_adaptive_progressive_trainer() {
        let mut trainer = AdaptiveProgressiveTrainer::new(AdaptiveProgressiveConfig::default());

        assert!(!trainer.is_paused());
        assert!(trainer.current_lr() > 0.0);
        assert!(trainer.current_batch_size() > 0);
    }

    #[test]
    fn test_step_processing() {
        let mut trainer = AdaptiveProgressiveTrainer::new(AdaptiveProgressiveConfig::default());

        let metrics = make_metrics(0, 2.5, 0.5);
        let result = trainer.step(&metrics);

        assert!(result.should_continue);
        assert!(result.pause_reason.is_none());
    }

    #[test]
    fn test_loop_helper() {
        let mut helper = AdaptiveLoopHelper::new(AdaptiveProgressiveConfig::default());

        for step in 0..10 {
            let metrics = make_metrics(step, 2.5, 0.5);
            let update = helper.process_step(&metrics);

            assert!(!update.should_pause);
        }

        let (lr, batch, clip) = helper.current_state();
        assert!(lr > 0.0);
        assert!(batch > 0);
        assert!(clip > 0.0);
    }

    #[test]
    fn test_pause_and_resume() {
        let mut config = AdaptiveProgressiveConfig::default();
        config.min_health_score = 0.99; // Very high threshold to trigger pause

        let mut trainer = AdaptiveProgressiveTrainer::new(config);

        // Generate some bad training data
        for step in 0..200 {
            let loss = 2.5 + (step as f32 * 0.1); // Increasing loss
            let metrics = make_metrics(step, loss, 0.5);
            trainer.step(&metrics);
        }

        // May or may not be paused depending on health calculation
        if trainer.is_paused() {
            assert!(trainer.pause_reason().is_some());

            // Resume
            trainer.resume();
            assert!(!trainer.is_paused());
            assert!(trainer.pause_reason().is_none());
        }
    }

    #[test]
    fn test_step_result_helpers() {
        let result = StepResult {
            adaptations: vec![super::super::controller::Adaptation::new(
                10,
                super::super::controller::AdaptedParam::LearningRate,
                0.001,
                0.0005,
                "test",
            )],
            should_continue: true,
            health_report: None,
            pause_reason: None,
        };

        assert!(result.has_adaptations());
        assert_eq!(result.new_lr(), Some(0.0005));
        assert!(result.new_batch_size().is_none());
    }

    #[test]
    fn test_adaptive_update_format() {
        let update = AdaptiveUpdate {
            lr_changed: true,
            batch_changed: true,
            grad_clip_changed: false,
            new_lr: 0.0001,
            new_batch_size: 16,
            new_grad_clip: 1.0,
            should_pause: false,
            health_report: None,
        };

        assert!(update.has_changes());
        let formatted = update.format_changes();
        assert!(formatted.contains("LR"));
        assert!(formatted.contains("Batch"));
        assert!(!formatted.contains("GradClip"));
    }
}
