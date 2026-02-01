//! Memory management and gradient checkpointing utilities.
//!
//! # Gradient Checkpointing
//!
//! Gradient checkpointing trades compute for memory by caching activations at
//! checkpoint boundaries and recomputing intermediates during backward pass.
//!
//! ## Implementation Status
//!
//! The actual gradient checkpointing implementation is in the [`crate::checkpoint`] module.
//! This module provides memory estimation and planning utilities.
//!
//! - **Activation caching**: [`crate::checkpoint::CheckpointStore`] caches activations
//! - **Memory estimation**: [`CheckpointConfig`] and [`TritterConfig`] estimate savings
//! - **Segment recomputation**: [`crate::model::TritterModel::recompute_segment`]
//!
//! ## Memory Savings
//!
//! | Checkpoint Interval | Memory Reduction | Compute Overhead |
//! |---------------------|------------------|------------------|
//! | Every 4 layers      | ~75% reduction   | ~33% overhead    |
//! | Every 8 layers      | ~87.5% reduction | ~75% overhead    |
//!
//! ## Usage
//!
//! Enable checkpointing in model config:
//!
//! ```rust
//! use tritter_model_rs::TritterConfig;
//!
//! let mut config = TritterConfig::large_1b();
//! config.gradient_checkpointing = true;
//! config.checkpoint_every_n_layers = 4;
//!
//! let estimate = config.total_training_memory_estimate(4, 2048);
//! println!("{}", estimate.format());
//! ```
//!
//! ## Memory Planning
//!
//! Use `TritterConfig::total_training_memory_estimate()` to plan memory:
//!
//! ```rust
//! use tritter_model_rs::TritterConfig;
//!
//! let config = TritterConfig::large_1b();
//! let estimate = config.total_training_memory_estimate(4, 2048);
//! println!("{}", estimate.format());
//! ```
//!
//! ## Candle Limitations
//!
//! Note: Candle's autograd doesn't have native checkpoint support. Our implementation:
//! 1. Caches activations at checkpoint boundaries during forward
//! 2. Clears checkpoints after backward to free memory
//! 3. Provides segment recomputation utilities for custom backward implementations
//!
//! The automatic gradient computation uses Candle's standard backward pass.
//! Full recomputation during backward (like PyTorch's `torch.utils.checkpoint`)
//! would require deeper autograd integration.

use crate::config::TritterConfig;

/// Memory pool for tracking allocations during training.
///
/// This is a simple bookkeeping structure that helps estimate and track
/// memory usage without actually managing GPU memory (which is handled by Candle).
#[derive(Debug, Clone)]
pub struct MemoryTracker {
    /// Current tracked allocation in bytes
    current: usize,
    /// Peak tracked allocation in bytes
    peak: usize,
    /// Optional memory limit for warnings
    limit: Option<usize>,
}

impl MemoryTracker {
    /// Create a new memory tracker
    pub fn new() -> Self {
        Self {
            current: 0,
            peak: 0,
            limit: None,
        }
    }

    /// Create a tracker with a memory limit
    pub fn with_limit(limit: usize) -> Self {
        Self {
            current: 0,
            peak: 0,
            limit: Some(limit),
        }
    }

    /// Record an allocation
    pub fn allocate(&mut self, bytes: usize) -> bool {
        self.current += bytes;
        self.peak = self.peak.max(self.current);

        // Return whether we're under the limit
        self.limit.is_none_or(|l| self.current <= l)
    }

    /// Record a deallocation
    pub fn deallocate(&mut self, bytes: usize) {
        self.current = self.current.saturating_sub(bytes);
    }

    /// Get current tracked allocation
    pub fn current(&self) -> usize {
        self.current
    }

    /// Get peak tracked allocation
    pub fn peak(&self) -> usize {
        self.peak
    }

    /// Reset the tracker
    pub fn reset(&mut self) {
        self.current = 0;
        self.peak = 0;
    }

    /// Check if current allocation exceeds limit
    pub fn is_over_limit(&self) -> bool {
        self.limit.is_some_and(|l| self.current > l)
    }
}

impl Default for MemoryTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for gradient checkpointing.
///
/// **Note**: This is currently for planning/estimation only.
/// Actual checkpointing is not implemented due to Candle limitations.
#[derive(Debug, Clone)]
pub struct CheckpointConfig {
    /// Enable checkpointing (for memory estimation)
    pub enabled: bool,
    /// Checkpoint every N layers
    pub checkpoint_every: usize,
}

impl Default for CheckpointConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            checkpoint_every: 1,
        }
    }
}

impl CheckpointConfig {
    /// Create checkpoint config from model config
    pub fn from_model_config(config: &TritterConfig) -> Self {
        Self {
            enabled: config.gradient_checkpointing,
            checkpoint_every: config.checkpoint_every_n_layers,
        }
    }

    /// Calculate memory reduction factor
    ///
    /// Returns a value between 0 and 1, where lower is better.
    pub fn memory_reduction_factor(&self, num_layers: usize) -> f64 {
        if !self.enabled || num_layers == 0 {
            1.0
        } else {
            let stored = num_layers.div_ceil(self.checkpoint_every);
            stored as f64 / num_layers as f64
        }
    }

    /// Calculate extra compute overhead from recomputation
    ///
    /// Returns the factor by which forward compute increases (e.g., 1.33 = 33% more)
    pub fn compute_overhead_factor(&self, num_layers: usize) -> f64 {
        if !self.enabled || num_layers == 0 {
            1.0
        } else {
            // Each non-checkpointed layer is computed twice (forward + recompute in backward)
            let stored = num_layers.div_ceil(self.checkpoint_every);
            let recomputed = num_layers - stored;

            // Original: num_layers forward passes
            // With checkpointing: num_layers forward + recomputed forward during backward
            // Total = num_layers + recomputed
            (num_layers + recomputed) as f64 / num_layers as f64
        }
    }
}

/// Format bytes as human-readable string
pub fn format_bytes(bytes: usize) -> String {
    const KB: usize = 1024;
    const MB: usize = KB * 1024;
    const GB: usize = MB * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} bytes", bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_tracker() {
        let mut tracker = MemoryTracker::with_limit(1000);

        assert!(tracker.allocate(400));
        assert_eq!(tracker.current(), 400);

        assert!(tracker.allocate(400));
        assert_eq!(tracker.current(), 800);

        // This puts us over limit
        assert!(!tracker.allocate(300));
        assert!(tracker.is_over_limit());

        tracker.deallocate(500);
        assert_eq!(tracker.current(), 600);
        assert!(!tracker.is_over_limit());

        // Peak should still be 1100
        assert_eq!(tracker.peak(), 1100);
    }

    #[test]
    fn test_checkpoint_config_memory_factor() {
        let config = CheckpointConfig {
            enabled: true,
            checkpoint_every: 4,
        };

        // 32 layers, checkpoint every 4 = 8 stored = 8/32 = 0.25
        let factor = config.memory_reduction_factor(32);
        assert!((factor - 0.25).abs() < 0.01);
    }

    #[test]
    fn test_checkpoint_config_compute_overhead() {
        let config = CheckpointConfig {
            enabled: true,
            checkpoint_every: 4,
        };

        // 32 layers, 8 checkpointed, 24 recomputed
        // Total compute = 32 + 24 = 56 vs 32 = 1.75x
        let factor = config.compute_overhead_factor(32);
        assert!((factor - 1.75).abs() < 0.01);
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(500), "500 bytes");
        assert_eq!(format_bytes(1024), "1.00 KB");
        assert_eq!(format_bytes(1024 * 1024), "1.00 MB");
        assert_eq!(format_bytes(1024 * 1024 * 1024), "1.00 GB");
    }
}
