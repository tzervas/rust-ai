//! Automatic batch size tuning based on GPU memory usage.
//!
//! Dynamically adjusts batch size during training to maximize GPU utilization
//! while avoiding OOM conditions. Starts conservatively and gradually increases
//! batch size until memory threshold is reached.

use std::collections::VecDeque;
use tracing::{debug, info, warn};

/// Batch size tuner that monitors GPU memory and adjusts batch size.
#[derive(Debug, Clone)]
pub struct BatchTuner {
    /// Current batch size
    current_batch_size: usize,
    /// Minimum allowed batch size
    min_batch_size: usize,
    /// Maximum allowed batch size
    max_batch_size: usize,
    /// Target memory usage (e.g., 0.8 = use up to 80%)
    memory_threshold: f32,
    /// History of (batch_size, memory_used_pct)
    history: VecDeque<(usize, f32)>,
    /// Maximum history entries to keep
    history_limit: usize,
    /// Number of successful steps before attempting increase
    success_steps_required: usize,
    /// Counter for consecutive successful steps
    consecutive_successes: usize,
    /// Last known safe batch size (for emergency fallback)
    last_safe_batch: usize,
    /// Aggressive mode: faster ramp-up
    aggressive: bool,
}

impl BatchTuner {
    /// Create a new batch tuner.
    ///
    /// # Arguments
    /// * `initial` - Initial batch size to start with
    /// * `min` - Minimum allowed batch size
    /// * `max` - Maximum allowed batch size
    /// * `memory_threshold` - Target memory usage (0.0-1.0, e.g., 0.8 for 80%)
    pub fn new(initial: usize, min: usize, max: usize, memory_threshold: f32) -> Self {
        assert!(initial >= min, "Initial batch size must be >= min");
        assert!(initial <= max, "Initial batch size must be <= max");
        assert!(
            memory_threshold > 0.0 && memory_threshold <= 1.0,
            "Memory threshold must be in range (0.0, 1.0]"
        );

        Self {
            current_batch_size: initial,
            min_batch_size: min,
            max_batch_size: max,
            memory_threshold,
            history: VecDeque::new(),
            history_limit: 100,
            success_steps_required: 5,
            consecutive_successes: 0,
            last_safe_batch: initial,
            aggressive: false,
        }
    }

    /// Create a conservative tuner (slow ramp-up, more headroom).
    pub fn conservative(initial: usize, min: usize, max: usize) -> Self {
        let mut tuner = Self::new(initial, min, max, 0.75);
        tuner.success_steps_required = 10;
        tuner
    }

    /// Create an aggressive tuner (fast ramp-up, higher memory target).
    pub fn aggressive(initial: usize, min: usize, max: usize) -> Self {
        let mut tuner = Self::new(initial, min, max, 0.85);
        tuner.success_steps_required = 3;
        tuner.aggressive = true;
        tuner
    }

    /// Get current batch size.
    pub fn current_batch_size(&self) -> usize {
        self.current_batch_size
    }

    /// Suggest next batch size based on current memory usage.
    ///
    /// # Arguments
    /// * `current_memory_pct` - Current memory usage as percentage (0.0-1.0)
    ///
    /// # Returns
    /// Suggested batch size
    pub fn suggest_batch_size(&self, current_memory_pct: f32) -> usize {
        // If memory is very high, reduce immediately
        if current_memory_pct > 0.95 {
            debug!(
                "Memory critical ({:.1}%), suggesting emergency reduction",
                current_memory_pct * 100.0
            );
            return (self.current_batch_size * 2) / 3; // Reduce by 33%
        }

        // If under memory pressure, reduce
        if current_memory_pct > self.memory_threshold {
            debug!(
                "Memory above threshold ({:.1}% > {:.1}%), suggesting reduction",
                current_memory_pct * 100.0,
                self.memory_threshold * 100.0
            );
            return self
                .current_batch_size
                .saturating_sub(1)
                .max(self.min_batch_size);
        }

        // If well under threshold and have consistent success, suggest increase
        if current_memory_pct < self.memory_threshold - 0.1
            && self.consecutive_successes >= self.success_steps_required
            && self.current_batch_size < self.max_batch_size
        {
            let increase_factor = if self.aggressive { 1.5 } else { 1.2 };
            let suggested = ((self.current_batch_size as f32 * increase_factor) as usize)
                .min(self.max_batch_size);

            debug!(
                "Memory comfortable ({:.1}%), suggesting increase to {}",
                current_memory_pct * 100.0,
                suggested
            );
            return suggested;
        }

        // Otherwise, maintain current batch size
        self.current_batch_size
    }

    /// Record a training step.
    ///
    /// # Arguments
    /// * `batch_size` - Batch size used for this step
    /// * `memory_pct` - Memory usage percentage (0.0-1.0)
    /// * `success` - Whether the step completed successfully (false = OOM or error)
    pub fn record_step(&mut self, batch_size: usize, memory_pct: f32, success: bool) {
        // Add to history
        self.history.push_back((batch_size, memory_pct));
        if self.history.len() > self.history_limit {
            self.history.pop_front();
        }

        if success {
            self.consecutive_successes += 1;
            self.last_safe_batch = batch_size;

            // Auto-adjust batch size based on memory
            let suggested = self.suggest_batch_size(memory_pct);
            if suggested != self.current_batch_size {
                info!(
                    "Adjusting batch size: {} -> {} (memory: {:.1}%)",
                    self.current_batch_size,
                    suggested,
                    memory_pct * 100.0
                );
                self.current_batch_size = suggested;
                self.consecutive_successes = 0; // Reset success counter on change
            }
        } else {
            // Step failed (OOM or error)
            warn!(
                "Step failed with batch_size={}, memory={:.1}%",
                batch_size,
                memory_pct * 100.0
            );

            // Emergency reduction
            let new_batch = if batch_size > self.last_safe_batch {
                // Was trying to increase, fall back to last known safe
                self.last_safe_batch
            } else {
                // Even last safe failed, reduce aggressively
                (batch_size * 3) / 4 // 25% reduction
            };

            let new_batch = new_batch.max(self.min_batch_size);
            warn!(
                "Emergency batch size reduction: {} -> {}",
                batch_size, new_batch
            );

            self.current_batch_size = new_batch;
            self.consecutive_successes = 0;
        }
    }

    /// Get average memory usage from recent history.
    pub fn avg_memory_usage(&self) -> Option<f32> {
        if self.history.is_empty() {
            return None;
        }

        let sum: f32 = self.history.iter().map(|(_, mem)| mem).sum();
        Some(sum / self.history.len() as f32)
    }

    /// Get peak memory usage from history.
    pub fn peak_memory_usage(&self) -> Option<f32> {
        self.history
            .iter()
            .map(|(_, mem)| *mem)
            .max_by(|a, b| a.partial_cmp(b).unwrap())
    }

    /// Get minimum memory usage from history.
    pub fn min_memory_usage(&self) -> Option<f32> {
        self.history
            .iter()
            .map(|(_, mem)| *mem)
            .min_by(|a, b| a.partial_cmp(b).unwrap())
    }

    /// Get memory usage for a specific batch size (from history).
    pub fn memory_for_batch(&self, batch_size: usize) -> Option<f32> {
        self.history
            .iter()
            .filter(|(bs, _)| *bs == batch_size)
            .map(|(_, mem)| *mem)
            .last()
    }

    /// Estimate memory usage for a target batch size using linear extrapolation.
    ///
    /// Uses recent history to predict memory usage. Returns None if insufficient data.
    pub fn estimate_memory(&self, target_batch: usize) -> Option<f32> {
        if self.history.len() < 2 {
            return None;
        }

        // Find two most recent distinct batch sizes
        let mut batch_sizes: Vec<usize> = self.history.iter().map(|(bs, _)| *bs).collect();
        batch_sizes.sort_unstable();
        batch_sizes.dedup();

        if batch_sizes.len() < 2 {
            return None;
        }

        let bs1 = batch_sizes[batch_sizes.len() - 2];
        let bs2 = batch_sizes[batch_sizes.len() - 1];

        let mem1 = self.memory_for_batch(bs1)?;
        let mem2 = self.memory_for_batch(bs2)?;

        // Linear extrapolation: memory = m * batch_size + b
        let slope = (mem2 - mem1) / (bs2 as f32 - bs1 as f32);
        let intercept = mem1 - slope * bs1 as f32;

        let estimated = slope * target_batch as f32 + intercept;

        // Clamp to reasonable range
        Some(estimated.max(0.0).min(1.0))
    }

    /// Get statistics about tuner performance.
    pub fn stats(&self) -> BatchTunerStats {
        BatchTunerStats {
            current_batch: self.current_batch_size,
            min_batch: self.min_batch_size,
            max_batch: self.max_batch_size,
            avg_memory: self.avg_memory_usage(),
            peak_memory: self.peak_memory_usage(),
            consecutive_successes: self.consecutive_successes,
            history_size: self.history.len(),
            last_safe_batch: self.last_safe_batch,
        }
    }

    /// Reset tuner to initial state.
    pub fn reset(&mut self) {
        self.current_batch_size = self.last_safe_batch;
        self.consecutive_successes = 0;
        self.history.clear();
    }
}

/// Statistics about batch tuner performance.
#[derive(Debug, Clone)]
pub struct BatchTunerStats {
    pub current_batch: usize,
    pub min_batch: usize,
    pub max_batch: usize,
    pub avg_memory: Option<f32>,
    pub peak_memory: Option<f32>,
    pub consecutive_successes: usize,
    pub history_size: usize,
    pub last_safe_batch: usize,
}

impl BatchTunerStats {
    /// Format as human-readable string.
    pub fn format(&self) -> String {
        format!(
            "Batch: {} (min: {}, max: {}, safe: {}) | Memory: avg={:.1}% peak={:.1}% | Successes: {}",
            self.current_batch,
            self.min_batch,
            self.max_batch,
            self.last_safe_batch,
            self.avg_memory.unwrap_or(0.0) * 100.0,
            self.peak_memory.unwrap_or(0.0) * 100.0,
            self.consecutive_successes
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_tuner_creation() {
        let tuner = BatchTuner::new(8, 1, 32, 0.8);
        assert_eq!(tuner.current_batch_size(), 8);
        assert_eq!(tuner.min_batch_size, 1);
        assert_eq!(tuner.max_batch_size, 32);
        assert_eq!(tuner.memory_threshold, 0.8);
    }

    #[test]
    #[should_panic]
    fn test_invalid_initial() {
        BatchTuner::new(0, 1, 32, 0.8); // initial < min
    }

    #[test]
    fn test_suggest_emergency_reduction() {
        let tuner = BatchTuner::new(16, 1, 32, 0.8);
        let suggested = tuner.suggest_batch_size(0.96);
        assert!(suggested < 16);
        assert_eq!(suggested, 10); // 16 * 2/3 = 10.66 -> 10
    }

    #[test]
    fn test_suggest_normal_reduction() {
        let tuner = BatchTuner::new(16, 1, 32, 0.8);
        let suggested = tuner.suggest_batch_size(0.85);
        assert_eq!(suggested, 15); // Reduce by 1
    }

    #[test]
    fn test_suggest_increase() {
        let mut tuner = BatchTuner::new(8, 1, 32, 0.8);
        tuner.consecutive_successes = 5;
        let suggested = tuner.suggest_batch_size(0.6);
        assert!(suggested > 8);
        assert_eq!(suggested, 9); // 8 * 1.2 = 9.6 -> 9
    }

    #[test]
    fn test_record_success() {
        let mut tuner = BatchTuner::new(8, 1, 32, 0.8);

        // Record successful steps with low memory
        for _ in 0..6 {
            tuner.record_step(8, 0.6, true);
        }

        // Should increase batch size after 5 successful steps
        assert!(tuner.current_batch_size() > 8);
        // After batch change on step 5, consecutive_successes resets to 0,
        // then step 6 increments it to 1
        assert_eq!(tuner.consecutive_successes, 1);
    }

    #[test]
    fn test_record_failure() {
        let mut tuner = BatchTuner::new(16, 1, 32, 0.8);
        tuner.last_safe_batch = 12;

        // Record failure at batch 16
        tuner.record_step(16, 0.95, false);

        // Should fall back to last safe batch
        assert_eq!(tuner.current_batch_size(), 12);
        assert_eq!(tuner.consecutive_successes, 0);
    }

    #[test]
    fn test_avg_memory() {
        let mut tuner = BatchTuner::new(8, 1, 32, 0.8);
        tuner.record_step(8, 0.6, true);
        tuner.record_step(8, 0.7, true);
        tuner.record_step(8, 0.8, true);

        let avg = tuner.avg_memory_usage().unwrap();
        assert!((avg - 0.7).abs() < 0.01);
    }

    #[test]
    fn test_peak_memory() {
        let mut tuner = BatchTuner::new(8, 1, 32, 0.8);
        tuner.record_step(8, 0.6, true);
        tuner.record_step(8, 0.9, true);
        tuner.record_step(8, 0.7, true);

        let peak = tuner.peak_memory_usage().unwrap();
        assert_eq!(peak, 0.9);
    }

    #[test]
    fn test_estimate_memory() {
        let mut tuner = BatchTuner::new(8, 1, 32, 0.8);

        // Record data points: batch 8 -> 0.6, batch 16 -> 0.8
        tuner.record_step(8, 0.6, true);
        tuner.record_step(16, 0.8, true);

        // Estimate for batch 24: should be ~1.0 (linear extrapolation)
        let estimated = tuner.estimate_memory(24).unwrap();
        assert!((estimated - 1.0).abs() < 0.05);
    }

    #[test]
    fn test_aggressive_tuner() {
        let tuner = BatchTuner::aggressive(8, 1, 32);
        assert_eq!(tuner.memory_threshold, 0.85);
        assert_eq!(tuner.success_steps_required, 3);
        assert!(tuner.aggressive);
    }

    #[test]
    fn test_conservative_tuner() {
        let tuner = BatchTuner::conservative(8, 1, 32);
        assert_eq!(tuner.memory_threshold, 0.75);
        assert_eq!(tuner.success_steps_required, 10);
        assert!(!tuner.aggressive);
    }

    #[test]
    fn test_stats() {
        let mut tuner = BatchTuner::new(8, 1, 32, 0.8);
        tuner.record_step(8, 0.7, true);
        tuner.record_step(8, 0.75, true);

        let stats = tuner.stats();
        assert_eq!(stats.current_batch, 8);
        assert_eq!(stats.history_size, 2);
        assert_eq!(stats.consecutive_successes, 2);
        assert!(stats.avg_memory.is_some());
    }

    #[test]
    fn test_reset() {
        let mut tuner = BatchTuner::new(8, 1, 32, 0.8);
        tuner.record_step(8, 0.7, true);
        tuner.record_step(8, 0.7, true);
        tuner.consecutive_successes = 5;

        tuner.reset();
        assert_eq!(tuner.consecutive_successes, 0);
        assert_eq!(tuner.history.len(), 0);
    }
}
