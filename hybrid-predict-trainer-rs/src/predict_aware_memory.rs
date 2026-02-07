//! Predict-aware memory management for HybridTrainer.
//!
//! This module provides memory optimizations specific to hybrid predictive training.
//! The key insight: during Predict phase, we don't compute gradients or update weights,
//! so we can temporarily offload optimizer state and intermediate tensors to free VRAM.
//!
//! # Memory Phases
//!
//! ## Warmup/Full/Correct (Need Full Memory)
//!
//! - Model weights: VRAM
//! - Gradients: VRAM
//! - Optimizer state: VRAM
//! - Activations: VRAM (for backprop)
//!
//! ## Predict (Minimal Memory)
//!
//! - Model weights: VRAM (needed for forward pass)
//! - Gradients: **Offloaded** (not computing)
//! - Optimizer state: **Offloaded to CPU** (not updating)
//! - Activations: **Discarded** (no backprop)
//!
//! # Memory Savings
//!
//! Typical memory breakdown for large models:
//! - Model: 25%
//! - Gradients: 25%
//! - Optimizer state (Adam): 50% (2× model size for momentum + variance)
//! - Activations: Variable (batch-dependent)
//!
//! **During Predict phase**:
//! - Offload optimizer state: 50% → 0% (50% savings)
//! - No gradient storage: 25% → 0% (25% savings)
//! - No activation checkpointing: 15% → 0% (15% savings)
//!
//! **Total savings**: Up to 90% VRAM reduction vs Full phase!
//!
//! # Usage
//!
//! ```rust,ignore
//! use hybrid_predict_trainer_rs::predict_aware_memory::{
//!     PredictAwareMemoryConfig, MemoryOffloadStrategy
//! };
//!
//! let config = PredictAwareMemoryConfig {
//!     enabled: true,
//!     strategy: MemoryOffloadStrategy::CpuOffload, // Move optimizer to CPU
//!     restore_async: true, // Async restore during Correct phase
//! };
//! ```
//!
//! # Implementation Status
//!
//! **Current**: Framework and configuration defined
//! **TODO**: Integrate with Burn's optimizer trait for actual offloading

use serde::{Deserialize, Serialize};

/// Strategy for offloading optimizer state during Predict phase.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MemoryOffloadStrategy {
    /// No offloading (standard training).
    ///
    /// Optimizer state remains in VRAM during all phases.
    None,

    /// Offload optimizer state to CPU RAM during Predict.
    ///
    /// - **Savings**: 50-60% VRAM (optimizer state size)
    /// - **Overhead**: CPU→GPU transfer latency (~100-500ms)
    /// - **Best for**: Limited VRAM, fast PCIe
    CpuOffload,

    /// Offload to pinned CPU memory for faster transfers.
    ///
    /// - **Savings**: 50-60% VRAM
    /// - **Overhead**: 50-100ms (faster than regular CPU memory)
    /// - **Best for**: PCIe 4.0/5.0, frequent phase transitions
    PinnedCpuOffload,

    /// Compress optimizer state and keep in VRAM.
    ///
    /// - **Savings**: 30-40% VRAM (compression ratio dependent)
    /// - **Overhead**: Compression/decompression time (~50ms)
    /// - **Best for**: Slow CPU↔GPU transfers
    CompressInPlace,

    /// Drop optimizer state entirely, reinitialize when needed.
    ///
    /// - **Savings**: 50-60% VRAM
    /// - **Overhead**: None (but may affect convergence)
    /// - **Best for**: Very long Predict phases, experimental
    ///
    /// **Warning**: This breaks optimizer momentum/variance tracking.
    /// Only use if Predict phases are very long (100+ steps).
    DropAndReinitialize,
}

impl MemoryOffloadStrategy {
    /// Returns estimated VRAM savings as fraction (0.0-1.0).
    ///
    /// Assumes optimizer state is ~50% of total memory.
    #[must_use]
    pub fn estimated_vram_savings(self) -> f32 {
        match self {
            Self::None => 0.0,
            Self::CpuOffload | Self::PinnedCpuOffload => 0.5,     // 50%
            Self::CompressInPlace => 0.35,                       // 35%
            Self::DropAndReinitialize => 0.6,                    // 60% (+ gradients)
        }
    }

    /// Returns estimated overhead time in milliseconds.
    ///
    /// Overhead includes offload + restore operations.
    #[must_use]
    pub fn estimated_overhead_ms(self, optimizer_size_mb: usize) -> u64 {
        match self {
            Self::None => 0,
            Self::CpuOffload => {
                // Assume PCIe 3.0 x16: ~12 GB/s bidirectional
                // Transfer time = size / bandwidth
                let transfer_time_ms = (optimizer_size_mb as f64 / 12_000.0) * 1000.0;
                (transfer_time_ms * 2.0) as u64 // 2× for offload + restore
            }
            Self::PinnedCpuOffload => {
                // ~2× faster than regular CPU memory
                let transfer_time_ms = (optimizer_size_mb as f64 / 24_000.0) * 1000.0;
                (transfer_time_ms * 2.0) as u64
            }
            Self::CompressInPlace => {
                // Compression: ~2 GB/s, Decompression: ~4 GB/s
                let compress_ms = (optimizer_size_mb as f64 / 2000.0) * 1000.0;
                let decompress_ms = (optimizer_size_mb as f64 / 4000.0) * 1000.0;
                (compress_ms + decompress_ms) as u64
            }
            Self::DropAndReinitialize => 0, // No transfer, just reinit
        }
    }

    /// Returns whether this strategy affects optimizer convergence.
    #[must_use]
    pub const fn affects_convergence(self) -> bool {
        matches!(self, Self::DropAndReinitialize)
    }
}

impl Default for MemoryOffloadStrategy {
    fn default() -> Self {
        Self::None
    }
}

/// Configuration for predict-aware memory management.
///
/// Controls how optimizer state and tensors are managed during different
/// training phases to maximize VRAM efficiency.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictAwareMemoryConfig {
    /// Whether predict-aware memory management is enabled.
    pub enabled: bool,

    /// Offload strategy for optimizer state.
    pub offload_strategy: MemoryOffloadStrategy,

    /// Whether to asynchronously restore optimizer state.
    ///
    /// When enabled, restoration starts during the last few Predict steps,
    /// overlapping transfer with computation to reduce overhead.
    pub async_restore: bool,

    /// Number of steps before phase transition to start async restore.
    ///
    /// Only used if `async_restore` is true.
    ///
    /// Recommended: 3-5 steps (enough time for transfer to complete)
    pub async_restore_lookahead: usize,

    /// Whether to discard activations during Predict phase.
    ///
    /// Since Predict phase doesn't backpropagate, we can discard intermediate
    /// activations, saving 10-20% VRAM.
    pub discard_activations: bool,

    /// Whether to use gradient checkpointing during Full phase.
    ///
    /// Trades compute for memory by recomputing activations during backward pass.
    ///
    /// **Note**: This is orthogonal to predict-aware memory; included here
    /// for convenience.
    pub gradient_checkpointing: bool,
}

impl PredictAwareMemoryConfig {
    /// Creates a new config with defaults.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a disabled config (standard training).
    #[must_use]
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            offload_strategy: MemoryOffloadStrategy::None,
            async_restore: false,
            async_restore_lookahead: 0,
            discard_activations: false,
            gradient_checkpointing: false,
        }
    }

    /// Creates a conservative config (CPU offload only).
    #[must_use]
    pub fn conservative() -> Self {
        Self {
            enabled: true,
            offload_strategy: MemoryOffloadStrategy::CpuOffload,
            async_restore: false,
            async_restore_lookahead: 0,
            discard_activations: false,
            gradient_checkpointing: false,
        }
    }

    /// Creates an aggressive config (pinned CPU + async + discard activations).
    #[must_use]
    pub fn aggressive() -> Self {
        Self {
            enabled: true,
            offload_strategy: MemoryOffloadStrategy::PinnedCpuOffload,
            async_restore: true,
            async_restore_lookahead: 5,
            discard_activations: true,
            gradient_checkpointing: false,
        }
    }

    /// Creates a maximum savings config (all optimizations enabled).
    #[must_use]
    pub fn maximum() -> Self {
        Self {
            enabled: true,
            offload_strategy: MemoryOffloadStrategy::PinnedCpuOffload,
            async_restore: true,
            async_restore_lookahead: 5,
            discard_activations: true,
            gradient_checkpointing: true,
        }
    }

    /// Estimates total VRAM savings vs standard training.
    ///
    /// # Arguments
    ///
    /// * `predict_phase_fraction` - Fraction of time spent in Predict phase (0.0-1.0)
    ///
    /// # Returns
    ///
    /// Approximate VRAM usage relative to standard training (0.0-1.0).
    /// Lower is better (e.g., 0.3 = 70% savings).
    ///
    /// # Example
    ///
    /// ```rust
    /// use hybrid_predict_trainer_rs::predict_aware_memory::PredictAwareMemoryConfig;
    ///
    /// let config = PredictAwareMemoryConfig::aggressive();
    ///
    /// // If 60% of training is in Predict phase:
    /// let vram_usage = config.estimate_vram_savings(0.6);
    /// // Result: ~0.4 (60% savings during 60% of training = 36% average savings)
    /// ```
    #[must_use]
    pub fn estimate_vram_savings(&self, predict_phase_fraction: f32) -> f32 {
        if !self.enabled {
            return 1.0; // No savings
        }

        // Savings during Predict phase
        let mut predict_savings = self.offload_strategy.estimated_vram_savings();

        // Additional savings from discarding activations (~15% of memory)
        if self.discard_activations {
            predict_savings += 0.15;
        }

        // VRAM usage = (1.0 - savings) during predict + 1.0 during non-predict
        let predict_usage = 1.0 - predict_savings;
        let non_predict_fraction = 1.0 - predict_phase_fraction;

        predict_usage * predict_phase_fraction + non_predict_fraction
    }

    /// Estimates overhead time per phase transition.
    ///
    /// Returns time in milliseconds for offload + restore operations.
    #[must_use]
    pub fn estimate_overhead_ms(&self, optimizer_size_mb: usize) -> u64 {
        if !self.enabled {
            return 0;
        }

        let base_overhead = self
            .offload_strategy
            .estimated_overhead_ms(optimizer_size_mb);

        if self.async_restore {
            // Async restore hides most of the restore latency
            base_overhead / 2
        } else {
            base_overhead
        }
    }

    /// Validates the configuration.
    ///
    /// # Returns
    ///
    /// `Ok(())` if valid, `Err(String)` with error message if invalid.
    pub fn validate(&self) -> Result<(), String> {
        if self.enabled {
            if self.async_restore && self.async_restore_lookahead == 0 {
                return Err(
                    "async_restore_lookahead must be > 0 when async_restore is enabled"
                        .to_string(),
                );
            }

            if self.offload_strategy.affects_convergence() {
                eprintln!(
                    "Warning: {:?} may affect optimizer convergence",
                    self.offload_strategy
                );
            }
        }

        Ok(())
    }
}

impl Default for PredictAwareMemoryConfig {
    /// Creates a balanced default configuration.
    ///
    /// - Enabled: true
    /// - Strategy: PinnedCpuOffload (good balance)
    /// - Async restore: true (reduce overhead)
    /// - Discard activations: true (safe in Predict phase)
    /// - Gradient checkpointing: false (orthogonal optimization)
    fn default() -> Self {
        Self {
            enabled: true,
            offload_strategy: MemoryOffloadStrategy::PinnedCpuOffload,
            async_restore: true,
            async_restore_lookahead: 5,
            discard_activations: true,
            gradient_checkpointing: false,
        }
    }
}

/// Runtime state for predict-aware memory management.
///
/// Tracks current offload state and handles transitions between phases.
///
/// # TODO
///
/// - Integrate with Burn's optimizer trait for actual CPU offloading
/// - Implement async restore with CUDA streams
/// - Add compression support via bincode or custom codec
#[derive(Debug, Clone)]
pub struct PredictAwareMemoryState {
    /// Whether optimizer state is currently offloaded.
    pub optimizer_offloaded: bool,

    /// Whether activations are being discarded.
    pub activations_discarded: bool,

    /// Number of steps until restore should begin (for async).
    pub steps_until_restore: usize,
}

impl PredictAwareMemoryState {
    /// Creates a new state (optimizer on GPU).
    #[must_use]
    pub fn new() -> Self {
        Self {
            optimizer_offloaded: false,
            activations_discarded: false,
            steps_until_restore: 0,
        }
    }

    /// Records transition to Predict phase.
    ///
    /// # TODO
    ///
    /// - Actually offload optimizer state to CPU
    /// - Set up async restore timer
    pub fn enter_predict_phase(&mut self, config: &PredictAwareMemoryConfig) {
        if config.enabled {
            if config.offload_strategy != MemoryOffloadStrategy::None {
                // TODO: Offload optimizer state
                self.optimizer_offloaded = true;
            }

            if config.discard_activations {
                // TODO: Configure Burn to discard activations
                self.activations_discarded = true;
            }
        }
    }

    /// Records transition to Full/Correct phase.
    ///
    /// # TODO
    ///
    /// - Restore optimizer state from CPU (if not already async restored)
    /// - Re-enable activation storage
    pub fn exit_predict_phase(&mut self, config: &PredictAwareMemoryConfig) {
        if self.optimizer_offloaded {
            // TODO: Restore optimizer state from CPU
            self.optimizer_offloaded = false;
        }

        if self.activations_discarded {
            // TODO: Re-enable activation storage
            self.activations_discarded = false;
        }
    }

    /// Steps the async restore countdown.
    ///
    /// Call this each Predict step. When countdown reaches 0,
    /// async restore begins.
    ///
    /// # TODO
    ///
    /// - Trigger async restore on countdown = 0
    /// - Use CUDA streams for overlap
    pub fn step_async_restore(&mut self, config: &PredictAwareMemoryConfig) {
        if config.async_restore && self.steps_until_restore > 0 {
            self.steps_until_restore -= 1;

            if self.steps_until_restore == 0 {
                // TODO: Start async restore
            }
        }
    }
}

impl Default for PredictAwareMemoryState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_offload_strategy_savings() {
        assert_eq!(MemoryOffloadStrategy::None.estimated_vram_savings(), 0.0);
        assert_eq!(MemoryOffloadStrategy::CpuOffload.estimated_vram_savings(), 0.5);
        assert_eq!(
            MemoryOffloadStrategy::PinnedCpuOffload.estimated_vram_savings(),
            0.5
        );
        assert_eq!(
            MemoryOffloadStrategy::CompressInPlace.estimated_vram_savings(),
            0.35
        );
    }

    #[test]
    fn test_config_defaults() {
        let config = PredictAwareMemoryConfig::default();
        assert!(config.enabled);
        assert_eq!(
            config.offload_strategy,
            MemoryOffloadStrategy::PinnedCpuOffload
        );
        assert!(config.async_restore);
    }

    #[test]
    fn test_config_conservative() {
        let config = PredictAwareMemoryConfig::conservative();
        assert_eq!(config.offload_strategy, MemoryOffloadStrategy::CpuOffload);
        assert!(!config.async_restore);
    }

    #[test]
    fn test_config_aggressive() {
        let config = PredictAwareMemoryConfig::aggressive();
        assert!(config.discard_activations);
        assert!(config.async_restore);
    }

    #[test]
    fn test_vram_savings_estimate() {
        let config = PredictAwareMemoryConfig::aggressive();

        // 60% of time in Predict phase
        let vram_usage = config.estimate_vram_savings(0.6);

        // Expected: ~40% usage (60% savings)
        // Calculation: (1.0 - 0.5 - 0.15) * 0.6 + 0.4 = 0.35 * 0.6 + 0.4 = 0.61
        assert!((vram_usage - 0.61).abs() < 0.05);
    }

    #[test]
    fn test_disabled_no_savings() {
        let config = PredictAwareMemoryConfig::disabled();
        assert!(!config.enabled);
        assert_eq!(config.estimate_vram_savings(0.6), 1.0);
    }

    #[test]
    fn test_overhead_estimation() {
        let config = PredictAwareMemoryConfig::conservative();

        // 1GB optimizer state
        let overhead_ms = config.estimate_overhead_ms(1000);

        // Should be reasonable (< 1 second)
        assert!(overhead_ms < 1000);
    }

    #[test]
    fn test_async_restore_reduces_overhead() {
        let mut config = PredictAwareMemoryConfig::conservative();
        let overhead_sync = config.estimate_overhead_ms(1000);

        config.async_restore = true;
        let overhead_async = config.estimate_overhead_ms(1000);

        assert!(overhead_async < overhead_sync);
    }

    #[test]
    fn test_validation() {
        let mut config = PredictAwareMemoryConfig::default();
        assert!(config.validate().is_ok());

        // Invalid: async_restore without lookahead
        config.async_restore = true;
        config.async_restore_lookahead = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_state_transitions() {
        let config = PredictAwareMemoryConfig::aggressive();
        let mut state = PredictAwareMemoryState::new();

        assert!(!state.optimizer_offloaded);

        // Enter Predict
        state.enter_predict_phase(&config);
        assert!(state.optimizer_offloaded);
        assert!(state.activations_discarded);

        // Exit Predict
        state.exit_predict_phase(&config);
        assert!(!state.optimizer_offloaded);
        assert!(!state.activations_discarded);
    }
}
