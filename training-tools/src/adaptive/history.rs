//! Adaptation history tracking.
//!
//! Records all adaptations made during training for analysis and debugging.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::controller::{Adaptation, AdaptedParam};
use super::strategy::AdaptationStrategy;

/// Tracks the history of all adaptations made during training.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptationHistory {
    /// All adaptations in chronological order
    adaptations: Vec<Adaptation>,
    /// Count of adaptations by parameter type
    counts_by_param: HashMap<String, usize>,
    /// Strategy changes over time
    strategy_changes: Vec<StrategyChange>,
    /// Maximum history size (oldest entries are dropped)
    max_size: usize,
}

/// Records a strategy change event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyChange {
    /// Step when the change occurred
    pub step: u64,
    /// Previous strategy
    pub from: AdaptationStrategy,
    /// New strategy
    pub to: AdaptationStrategy,
    /// Reason for the change
    pub reason: String,
}

impl AdaptationHistory {
    /// Create a new adaptation history.
    pub fn new(max_size: usize) -> Self {
        Self {
            adaptations: Vec::with_capacity(max_size.min(1000)),
            counts_by_param: HashMap::new(),
            strategy_changes: Vec::new(),
            max_size,
        }
    }

    /// Record a new adaptation.
    pub fn record(&mut self, adaptation: Adaptation) {
        // Update counts
        let param_key = format!("{:?}", adaptation.param);
        *self.counts_by_param.entry(param_key).or_insert(0) += 1;

        // Add to history
        self.adaptations.push(adaptation);

        // Trim if over max size
        if self.adaptations.len() > self.max_size {
            let to_remove = self.adaptations.len() - self.max_size;
            self.adaptations.drain(0..to_remove);
        }
    }

    /// Record multiple adaptations.
    pub fn record_batch(&mut self, adaptations: Vec<Adaptation>) {
        for adaptation in adaptations {
            self.record(adaptation);
        }
    }

    /// Record a strategy change.
    pub fn record_strategy_change(
        &mut self,
        step: u64,
        from: AdaptationStrategy,
        to: AdaptationStrategy,
        reason: &str,
    ) {
        self.strategy_changes.push(StrategyChange {
            step,
            from,
            to,
            reason: reason.to_string(),
        });
    }

    /// Get all adaptations.
    pub fn all(&self) -> &[Adaptation] {
        &self.adaptations
    }

    /// Get adaptations for a specific parameter.
    pub fn for_param(&self, param: AdaptedParam) -> Vec<&Adaptation> {
        self.adaptations
            .iter()
            .filter(|a| std::mem::discriminant(&a.param) == std::mem::discriminant(&param))
            .collect()
    }

    /// Get recent adaptations (last N).
    pub fn recent(&self, count: usize) -> &[Adaptation] {
        let start = self.adaptations.len().saturating_sub(count);
        &self.adaptations[start..]
    }

    /// Get adaptations in a step range.
    pub fn in_range(&self, start_step: u64, end_step: u64) -> Vec<&Adaptation> {
        self.adaptations
            .iter()
            .filter(|a| a.step >= start_step && a.step <= end_step)
            .collect()
    }

    /// Get the total number of adaptations.
    pub fn total_count(&self) -> usize {
        self.adaptations.len()
    }

    /// Get the count of adaptations for a specific parameter.
    pub fn count_for_param(&self, param: AdaptedParam) -> usize {
        let param_key = format!("{:?}", param);
        self.counts_by_param.get(&param_key).copied().unwrap_or(0)
    }

    /// Get all strategy changes.
    pub fn strategy_changes(&self) -> &[StrategyChange] {
        &self.strategy_changes
    }

    /// Get the last adaptation for a specific parameter.
    pub fn last_for_param(&self, param: AdaptedParam) -> Option<&Adaptation> {
        self.adaptations
            .iter()
            .rev()
            .find(|a| std::mem::discriminant(&a.param) == std::mem::discriminant(&param))
    }

    /// Get the average adjustment magnitude for a parameter.
    pub fn avg_adjustment_for_param(&self, param: AdaptedParam) -> Option<f32> {
        let adaptations = self.for_param(param);
        if adaptations.is_empty() {
            return None;
        }

        let sum: f32 = adaptations
            .iter()
            .map(|a| (a.new_value - a.old_value).abs() / a.old_value.max(1e-8))
            .sum();

        Some(sum / adaptations.len() as f32)
    }

    /// Analyze adaptation frequency (adaptations per 100 steps).
    pub fn adaptation_frequency(&self, total_steps: u64) -> f32 {
        if total_steps == 0 {
            return 0.0;
        }
        (self.adaptations.len() as f32 / total_steps as f32) * 100.0
    }

    /// Check if a parameter was recently adjusted (within last N steps).
    pub fn was_recently_adjusted(
        &self,
        param: AdaptedParam,
        steps_ago: u64,
        current_step: u64,
    ) -> bool {
        self.adaptations
            .iter()
            .rev()
            .take(50) // Only check recent history
            .any(|a| {
                std::mem::discriminant(&a.param) == std::mem::discriminant(&param)
                    && current_step.saturating_sub(a.step) <= steps_ago
            })
    }

    /// Get a summary of the adaptation history.
    pub fn summary(&self) -> AdaptationSummary {
        let lr_count = self.count_for_param(AdaptedParam::LearningRate);
        let batch_count = self.count_for_param(AdaptedParam::BatchSize);
        let grad_clip_count = self.count_for_param(AdaptedParam::GradientClip);
        let warmup_count = self.count_for_param(AdaptedParam::WarmupSteps);
        let momentum_count = self.count_for_param(AdaptedParam::MomentumBeta1);

        // Calculate net changes
        let lr_net_change = self.net_change_for_param(AdaptedParam::LearningRate);
        let batch_net_change = self.net_change_for_param(AdaptedParam::BatchSize);
        let grad_clip_net_change = self.net_change_for_param(AdaptedParam::GradientClip);

        AdaptationSummary {
            total_adaptations: self.adaptations.len(),
            lr_adaptations: lr_count,
            batch_adaptations: batch_count,
            grad_clip_adaptations: grad_clip_count,
            warmup_adaptations: warmup_count,
            momentum_adaptations: momentum_count,
            strategy_changes: self.strategy_changes.len(),
            lr_net_change,
            batch_net_change,
            grad_clip_net_change,
        }
    }

    /// Calculate the net change for a parameter (ratio of final to initial).
    fn net_change_for_param(&self, param: AdaptedParam) -> Option<f32> {
        let adaptations = self.for_param(param);
        if adaptations.is_empty() {
            return None;
        }

        let first = adaptations.first()?;
        let last = adaptations.last()?;

        if first.old_value.abs() < 1e-10 {
            None
        } else {
            Some(last.new_value / first.old_value)
        }
    }

    /// Clear all history.
    pub fn clear(&mut self) {
        self.adaptations.clear();
        self.counts_by_param.clear();
        self.strategy_changes.clear();
    }
}

impl Default for AdaptationHistory {
    fn default() -> Self {
        Self::new(10000)
    }
}

/// Summary statistics for adaptation history.
#[derive(Debug, Clone)]
pub struct AdaptationSummary {
    /// Total number of adaptations
    pub total_adaptations: usize,
    /// Learning rate adaptations
    pub lr_adaptations: usize,
    /// Batch size adaptations
    pub batch_adaptations: usize,
    /// Gradient clip adaptations
    pub grad_clip_adaptations: usize,
    /// Warmup steps adaptations
    pub warmup_adaptations: usize,
    /// Momentum adaptations
    pub momentum_adaptations: usize,
    /// Number of strategy changes
    pub strategy_changes: usize,
    /// Net learning rate change (ratio of final to initial)
    pub lr_net_change: Option<f32>,
    /// Net batch size change (ratio of final to initial)
    pub batch_net_change: Option<f32>,
    /// Net gradient clip change (ratio of final to initial)
    pub grad_clip_net_change: Option<f32>,
}

impl AdaptationSummary {
    /// Format the summary as a human-readable string.
    pub fn format(&self) -> String {
        let mut lines = vec![
            format!("Adaptation Summary"),
            format!("══════════════════"),
            format!("Total adaptations: {}", self.total_adaptations),
            format!("  Learning rate: {}", self.lr_adaptations),
            format!("  Batch size: {}", self.batch_adaptations),
            format!("  Gradient clip: {}", self.grad_clip_adaptations),
            format!("  Warmup steps: {}", self.warmup_adaptations),
            format!("  Momentum: {}", self.momentum_adaptations),
            format!("Strategy changes: {}", self.strategy_changes),
        ];

        if let Some(ratio) = self.lr_net_change {
            lines.push(format!("LR net change: {:.2}x", ratio));
        }
        if let Some(ratio) = self.batch_net_change {
            lines.push(format!("Batch net change: {:.2}x", ratio));
        }
        if let Some(ratio) = self.grad_clip_net_change {
            lines.push(format!("Grad clip net change: {:.2}x", ratio));
        }

        lines.join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn make_adaptation(step: u64, param: AdaptedParam, old: f32, new: f32) -> Adaptation {
        Adaptation {
            step,
            param,
            old_value: old,
            new_value: new,
            reason: "test".to_string(),
            timestamp: Utc::now(),
        }
    }

    #[test]
    fn test_record_adaptations() {
        let mut history = AdaptationHistory::new(100);

        history.record(make_adaptation(
            1,
            AdaptedParam::LearningRate,
            0.001,
            0.0005,
        ));
        history.record(make_adaptation(2, AdaptedParam::BatchSize, 8.0, 16.0));

        assert_eq!(history.total_count(), 2);
        assert_eq!(history.count_for_param(AdaptedParam::LearningRate), 1);
        assert_eq!(history.count_for_param(AdaptedParam::BatchSize), 1);
    }

    #[test]
    fn test_history_max_size() {
        let mut history = AdaptationHistory::new(3);

        for i in 0..5 {
            history.record(make_adaptation(
                i,
                AdaptedParam::LearningRate,
                0.001,
                0.0005,
            ));
        }

        assert_eq!(history.total_count(), 3);
        // Should keep the last 3
        assert_eq!(history.adaptations[0].step, 2);
        assert_eq!(history.adaptations[2].step, 4);
    }

    #[test]
    fn test_for_param() {
        let mut history = AdaptationHistory::new(100);

        history.record(make_adaptation(
            1,
            AdaptedParam::LearningRate,
            0.001,
            0.0005,
        ));
        history.record(make_adaptation(2, AdaptedParam::BatchSize, 8.0, 16.0));
        history.record(make_adaptation(
            3,
            AdaptedParam::LearningRate,
            0.0005,
            0.00025,
        ));

        let lr_adaptations = history.for_param(AdaptedParam::LearningRate);
        assert_eq!(lr_adaptations.len(), 2);
        assert_eq!(lr_adaptations[0].step, 1);
        assert_eq!(lr_adaptations[1].step, 3);
    }

    #[test]
    fn test_recent() {
        let mut history = AdaptationHistory::new(100);

        for i in 0..10 {
            history.record(make_adaptation(
                i,
                AdaptedParam::LearningRate,
                0.001,
                0.0005,
            ));
        }

        let recent = history.recent(3);
        assert_eq!(recent.len(), 3);
        assert_eq!(recent[0].step, 7);
        assert_eq!(recent[2].step, 9);
    }

    #[test]
    fn test_in_range() {
        let mut history = AdaptationHistory::new(100);

        for i in 0..10 {
            history.record(make_adaptation(
                i,
                AdaptedParam::LearningRate,
                0.001,
                0.0005,
            ));
        }

        let in_range = history.in_range(3, 6);
        assert_eq!(in_range.len(), 4);
        assert_eq!(in_range[0].step, 3);
        assert_eq!(in_range[3].step, 6);
    }

    #[test]
    fn test_strategy_changes() {
        let mut history = AdaptationHistory::new(100);

        history.record_strategy_change(
            10,
            AdaptationStrategy::Balanced,
            AdaptationStrategy::Recovery,
            "Training instability detected",
        );

        assert_eq!(history.strategy_changes.len(), 1);
        assert_eq!(
            history.strategy_changes[0].from,
            AdaptationStrategy::Balanced
        );
        assert_eq!(history.strategy_changes[0].to, AdaptationStrategy::Recovery);
    }

    #[test]
    fn test_was_recently_adjusted() {
        let mut history = AdaptationHistory::new(100);

        history.record(make_adaptation(
            50,
            AdaptedParam::LearningRate,
            0.001,
            0.0005,
        ));
        history.record(make_adaptation(60, AdaptedParam::BatchSize, 8.0, 16.0));

        // LR was adjusted at step 50, we're at step 100
        assert!(!history.was_recently_adjusted(AdaptedParam::LearningRate, 40, 100));
        assert!(history.was_recently_adjusted(AdaptedParam::LearningRate, 60, 100));

        // Batch was adjusted at step 60
        assert!(history.was_recently_adjusted(AdaptedParam::BatchSize, 50, 100));
    }

    #[test]
    fn test_summary() {
        let mut history = AdaptationHistory::new(100);

        history.record(make_adaptation(
            1,
            AdaptedParam::LearningRate,
            0.001,
            0.0005,
        ));
        history.record(make_adaptation(
            2,
            AdaptedParam::LearningRate,
            0.0005,
            0.00025,
        ));
        history.record(make_adaptation(3, AdaptedParam::BatchSize, 8.0, 16.0));

        let summary = history.summary();
        assert_eq!(summary.total_adaptations, 3);
        assert_eq!(summary.lr_adaptations, 2);
        assert_eq!(summary.batch_adaptations, 1);

        // LR went from 0.001 to 0.00025 = 0.25x
        assert!(summary.lr_net_change.is_some());
        assert!((summary.lr_net_change.unwrap() - 0.25).abs() < 0.01);
    }
}
