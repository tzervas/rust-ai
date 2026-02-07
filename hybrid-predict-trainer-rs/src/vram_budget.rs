//! VRAM budget management and auto-configuration.
//!
//! This module provides utilities to automatically detect available VRAM,
//! reserve safe buffers for the OS/host, and auto-adjust training configurations
//! to prevent OOM errors.
//!
//! # Why VRAM Management?
//!
//! GPU memory is shared between:
//! - Operating system (drivers, desktop environment)
//! - Running applications (browser, editor, terminals)
//! - Training workload (model, gradients, optimizer state, batches)
//!
//! Without proper management, training can trigger OOM errors that crash
//! the process or even destabilize the system.
//!
//! # Usage
//!
//! ```rust,ignore
//! use hybrid_predict_trainer_rs::vram_budget::{VramBudget, VramConfig};
//!
//! // Detect available VRAM and configure safely
//! let budget = VramBudget::detect()?;
//! let config = budget.recommend_config_for_model(model_size_mb, param_count);
//!
//! println!("Recommended batch size: {}", config.max_batch_size);
//! println!("Use gradient checkpointing: {}", config.use_gradient_checkpointing);
//! ```

use crate::config::HybridTrainerConfig;
use crate::error::HybridResult;
use std::process::Command;

/// VRAM budget information for a GPU.
#[derive(Debug, Clone)]
pub struct VramBudget {
    /// Total VRAM in MB
    pub total_vram_mb: usize,

    /// Currently used VRAM in MB (baseline before training)
    pub baseline_used_mb: usize,

    /// Reserved buffer for OS/host stability (MB)
    pub reserved_buffer_mb: usize,

    /// Available VRAM for training (MB)
    pub available_for_training_mb: usize,
}

impl VramBudget {
    /// Detects current VRAM usage and calculates safe budget.
    ///
    /// # Returns
    ///
    /// A `VramBudget` with detected values and calculated safe limits.
    ///
    /// # Errors
    ///
    /// Returns error if `nvidia-smi` is not available or fails to execute.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let budget = VramBudget::detect()?;
    /// println!("Safe training budget: {} MB", budget.available_for_training_mb);
    /// ```
    pub fn detect() -> HybridResult<Self> {
        // Query nvidia-smi for memory info
        let output = Command::new("nvidia-smi")
            .args(&[
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ])
            .output()
            .map_err(|e| {
                (
                    crate::error::HybridTrainingError::IntegrationError {
                        crate_name: "nvidia-smi".to_string(),
                        detail: format!("Failed to execute: {}", e),
                    },
                    None,
                )
            })?;

        let output_str = String::from_utf8_lossy(&output.stdout);
        let parts: Vec<&str> = output_str.trim().split(',').collect();

        if parts.len() != 2 {
            return Err((
                crate::error::HybridTrainingError::IntegrationError {
                    crate_name: "nvidia-smi".to_string(),
                    detail: format!("Unexpected output: {}", output_str),
                },
                None,
            ));
        }

        let baseline_used_mb = parts[0].trim().parse::<usize>().map_err(|e| {
            (
                crate::error::HybridTrainingError::IntegrationError {
                    crate_name: "nvidia-smi".to_string(),
                    detail: format!("Failed to parse used memory: {}", e),
                },
                None,
            )
        })?;

        let total_vram_mb = parts[1].trim().parse::<usize>().map_err(|e| {
            (
                crate::error::HybridTrainingError::IntegrationError {
                    crate_name: "nvidia-smi".to_string(),
                    detail: format!("Failed to parse total memory: {}", e),
                },
                None,
            )
        })?;

        // Calculate safe reserved buffer:
        // - Current baseline usage (OS + desktop + apps)
        // - Plus 500 MB safety margin for OS spikes
        let reserved_buffer_mb = baseline_used_mb + 500;

        let available_for_training_mb = total_vram_mb.saturating_sub(reserved_buffer_mb);

        Ok(Self {
            total_vram_mb,
            baseline_used_mb,
            reserved_buffer_mb,
            available_for_training_mb,
        })
    }

    /// Recommends training configuration based on model size and available VRAM.
    ///
    /// # Arguments
    ///
    /// * `model_size_mb` - Estimated model parameter size in MB
    /// * `param_count` - Number of trainable parameters
    ///
    /// # Returns
    ///
    /// A `VramConfig` with recommended settings to fit within budget.
    pub fn recommend_config_for_model(
        &self,
        model_size_mb: usize,
        _param_count: usize,
    ) -> VramConfig {
        // Estimate total VRAM requirements
        // Model: model_size_mb (FP16) or 2*model_size_mb (FP32)
        // Gradients: model_size_mb (FP32 for precision)
        // Optimizer (Adam): 2*model_size_mb (momentum + variance)
        // Activations: varies by batch size
        // RSSM dynamics: ~500-800 MB
        // Training state: ~100-200 MB

        let available = self.available_for_training_mb;

        // Conservative estimate: assume FP16 model
        let base_memory = model_size_mb * 4 + 1000; // model + gradients + optimizer + overhead

        if base_memory > available {
            // Model doesn't fit at all - need FP16 + aggressive settings
            return VramConfig {
                max_batch_size: 1,
                use_mixed_precision: true,
                use_gradient_checkpointing: true,
                gradient_accumulation_steps: 4,
                estimated_vram_mb: base_memory,
                fits_in_budget: false,
            };
        }

        let remaining = available - base_memory;

        // Calculate batch size based on remaining VRAM
        // Rough estimate: 200-500 MB per batch depending on model size
        let mb_per_batch = if model_size_mb < 500 {
            200
        } else if model_size_mb < 2000 {
            400
        } else {
            600
        };

        let max_batch_size = (remaining / mb_per_batch).max(1);

        // Decide on optimizations based on model size
        let use_mixed_precision = model_size_mb > 100;
        let use_gradient_checkpointing = model_size_mb > 500;
        let gradient_accumulation_steps = if max_batch_size < 4 { 4 } else { 1 };

        VramConfig {
            max_batch_size,
            use_mixed_precision,
            use_gradient_checkpointing,
            gradient_accumulation_steps,
            estimated_vram_mb: base_memory + (max_batch_size * mb_per_batch),
            fits_in_budget: true,
        }
    }

    /// Prints a human-readable VRAM budget summary.
    pub fn print_summary(&self) {
        println!("╔════════════════════════════════════════════╗");
        println!("║         VRAM Budget Summary                ║");
        println!("╠════════════════════════════════════════════╣");
        println!(
            "║ Total VRAM:        {:>8} MB          ║",
            self.total_vram_mb
        );
        println!(
            "║ Baseline Used:     {:>8} MB          ║",
            self.baseline_used_mb
        );
        println!(
            "║ Reserved Buffer:   {:>8} MB          ║",
            self.reserved_buffer_mb
        );
        println!("║ ────────────────────────────────────────── ║");
        println!(
            "║ Available:         {:>8} MB ({:>5.1}%) ║",
            self.available_for_training_mb,
            (self.available_for_training_mb as f32 / self.total_vram_mb as f32) * 100.0
        );
        println!("╚════════════════════════════════════════════╝");
    }
}

/// Recommended training configuration based on VRAM budget.
#[derive(Debug, Clone)]
pub struct VramConfig {
    /// Maximum safe batch size
    pub max_batch_size: usize,

    /// Whether to use mixed precision (FP16/BF16)
    pub use_mixed_precision: bool,

    /// Whether to enable gradient checkpointing
    pub use_gradient_checkpointing: bool,

    /// Number of gradient accumulation steps
    pub gradient_accumulation_steps: usize,

    /// Estimated total VRAM usage with these settings
    pub estimated_vram_mb: usize,

    /// Whether this configuration fits within budget
    pub fits_in_budget: bool,
}

impl VramConfig {
    /// Applies this VRAM configuration to a `HybridTrainerConfig`.
    ///
    /// # Arguments
    ///
    /// * `config` - The trainer config to modify
    ///
    /// # Returns
    ///
    /// A new `HybridTrainerConfig` with VRAM-safe settings applied.
    ///
    /// # Note
    ///
    /// Currently returns the config unmodified as VRAM-specific fields
    /// will be added to `HybridTrainerConfig` in Phase 2 of Burn integration.
    pub fn apply_to_config(&self, config: HybridTrainerConfig) -> HybridTrainerConfig {
        // TODO: Add these fields to HybridTrainerConfig in Phase 2:
        // - max_batch_size
        // - use_mixed_precision
        // - use_gradient_checkpointing
        // - gradient_accumulation_steps

        // For now, return unmodified config
        config
    }

    /// Prints a human-readable configuration summary.
    pub fn print_summary(&self) {
        println!("╔════════════════════════════════════════════╗");
        println!("║    Recommended Training Configuration      ║");
        println!("╠════════════════════════════════════════════╣");
        println!(
            "║ Max Batch Size:    {:>8}              ║",
            self.max_batch_size
        );
        println!(
            "║ Mixed Precision:   {:>8}              ║",
            if self.use_mixed_precision {
                "Yes"
            } else {
                "No"
            }
        );
        println!(
            "║ Grad Checkpoint:   {:>8}              ║",
            if self.use_gradient_checkpointing {
                "Yes"
            } else {
                "No"
            }
        );
        println!(
            "║ Grad Accumulation: {:>8}              ║",
            self.gradient_accumulation_steps
        );
        println!("║ ────────────────────────────────────────── ║");
        println!(
            "║ Est. VRAM Usage:   {:>8} MB          ║",
            self.estimated_vram_mb
        );
        println!(
            "║ Fits in Budget:    {:>8}              ║",
            if self.fits_in_budget {
                "Yes ✓"
            } else {
                "No ✗"
            }
        );
        println!("╚════════════════════════════════════════════╝");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vram_budget_detection() {
        // This test requires nvidia-smi to be available
        // Skip if not running on a system with NVIDIA GPU
        if let Ok(budget) = VramBudget::detect() {
            assert!(budget.total_vram_mb > 0);
            assert!(budget.baseline_used_mb > 0);
            assert!(budget.reserved_buffer_mb > budget.baseline_used_mb);
            assert!(budget.available_for_training_mb < budget.total_vram_mb);
        }
    }

    #[test]
    fn test_small_model_config() {
        let budget = VramBudget {
            total_vram_mb: 16000,
            baseline_used_mb: 450,
            reserved_buffer_mb: 950,
            available_for_training_mb: 15050,
        };

        // SimpleMLP (100 MB)
        let config = budget.recommend_config_for_model(100, 100_000);
        assert!(config.max_batch_size >= 16);
        assert!(config.fits_in_budget);
    }

    #[test]
    fn test_large_model_config() {
        let budget = VramBudget {
            total_vram_mb: 16000,
            baseline_used_mb: 450,
            reserved_buffer_mb: 950,
            available_for_training_mb: 15050,
        };

        // TinyLlama-1.1B (~2200 MB)
        let config = budget.recommend_config_for_model(2200, 1_100_000_000);
        assert!(config.use_mixed_precision);
        assert!(config.use_gradient_checkpointing);
        assert!(config.fits_in_budget);
    }
}
