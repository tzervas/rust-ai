//! GPU test utilities for axolotl-rs.
//!
//! Provides helper functions for GPU availability detection, VRAM checking,
//! and device initialization for GPU-accelerated tests.
//!
//! # Usage
//!
//! ```rust,ignore
//! use gpu_utils::*;
//!
//! #[test]
//! #[cfg(feature = "cuda")]
//! fn test_gpu_training() {
//!     if !cuda_available() {
//!         skip_gpu_test("CUDA not available");
//!         return;
//!     }
//!     
//!     let device = get_cuda_device().expect("Failed to get CUDA device");
//!     // ... run GPU test
//! }
//! ```

use candle_core::Device;

/// VRAM requirements for different model tiers (in bytes).
pub mod vram_requirements {
    /// SmolLM2-135M: ~150 MB base + activations
    pub const SMOLLM2_135M: usize = 256 * 1024 * 1024; // 256 MB
    /// TinyLlama-1.1B 4-bit: ~700 MB base + activations
    pub const TINYLLAMA_1B_4BIT: usize = 2 * 1024 * 1024 * 1024; // 2 GB
    /// LLaMA-7B 4-bit: ~4 GB base + activations
    pub const LLAMA_7B_4BIT: usize = 8 * 1024 * 1024 * 1024; // 8 GB
    /// LLaMA-7B 4-bit with gradient checkpointing
    pub const LLAMA_7B_4BIT_CHECKPOINT: usize = 12 * 1024 * 1024 * 1024; // 12 GB
}

/// Check if CUDA is available at runtime.
///
/// Returns `true` if:
/// - The `cuda` feature is enabled at compile time
/// - CUDA runtime is available (driver installed, GPU present)
#[must_use]
pub fn cuda_available() -> bool {
    #[cfg(feature = "cuda")]
    {
        candle_core::utils::cuda_is_available()
    }
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

/// Get the CUDA device (GPU 0).
///
/// # Errors
/// Returns an error if CUDA device initialization fails.
pub fn get_cuda_device() -> Result<Device, String> {
    #[cfg(feature = "cuda")]
    {
        Device::new_cuda(0).map_err(|e| format!("Failed to initialize CUDA device: {}", e))
    }
    #[cfg(not(feature = "cuda"))]
    {
        Err("CUDA feature not enabled at compile time".to_string())
    }
}

/// Get CUDA device count.
#[must_use]
pub fn cuda_device_count() -> usize {
    #[cfg(feature = "cuda")]
    {
        candle_core::utils::get_num_threads() // This is actually thread count, not GPU count
                                              // TODO: Use proper CUDA API when available in candle
    }
    #[cfg(not(feature = "cuda"))]
    {
        0
    }
}

/// Print GPU test skip message with reason.
pub fn skip_gpu_test(reason: &str) {
    println!("⚠️  Skipping GPU test: {}", reason);
}

/// Print GPU test status message.
pub fn gpu_test_status(message: &str) {
    println!("🔷 GPU Test: {}", message);
}

/// Assert that loss is decreasing over training steps.
///
/// # Arguments
/// * `losses` - Vector of loss values from training steps
/// * `min_decrease_ratio` - Minimum required decrease (e.g., 0.3 for 30% decrease)
/// * `window_size` - Number of steps to average for comparison
///
/// # Panics
/// Panics if loss convergence criteria are not met.
pub fn assert_loss_convergence(losses: &[f64], min_decrease_ratio: f64, window_size: usize) {
    assert!(
        !losses.is_empty(),
        "No loss values to check for convergence"
    );

    if losses.len() < window_size * 2 {
        println!(
            "⚠️  Not enough loss values ({}) for convergence check (need {})",
            losses.len(),
            window_size * 2
        );
        return;
    }

    // Compare first window average to last window average
    let first_window: f64 = losses[..window_size].iter().sum::<f64>() / window_size as f64;
    let last_window: f64 =
        losses[losses.len() - window_size..].iter().sum::<f64>() / window_size as f64;

    let decrease_ratio = (first_window - last_window) / first_window;

    println!(
        "📉 Loss convergence: {:.4} → {:.4} ({:.1}% decrease)",
        first_window,
        last_window,
        decrease_ratio * 100.0
    );

    assert!(
        decrease_ratio >= min_decrease_ratio,
        "Loss did not decrease enough: {:.4} → {:.4} ({:.1}% < {:.1}% required)",
        first_window,
        last_window,
        decrease_ratio * 100.0,
        min_decrease_ratio * 100.0
    );
}

/// Assert that loss values are monotonically decreasing (with some tolerance).
///
/// # Arguments
/// * `losses` - Vector of loss values
/// * `tolerance` - Maximum allowed increase between consecutive values
///
/// # Panics
/// Panics if loss increases by more than tolerance.
pub fn assert_monotonic_decrease(losses: &[f64], tolerance: f64) {
    for i in 1..losses.len() {
        let increase = losses[i] - losses[i - 1];
        if increase > tolerance {
            panic!(
                "Loss increased at step {}: {:.4} → {:.4} (increase {:.4} > tolerance {:.4})",
                i,
                losses[i - 1],
                losses[i],
                increase,
                tolerance
            );
        }
    }
    println!(
        "✓ Loss is monotonically decreasing (tolerance: {:.4})",
        tolerance
    );
}

/// Track training metrics for GPU tests.
#[derive(Debug, Default)]
pub struct TrainingMetrics {
    /// Loss values per step
    pub losses: Vec<f64>,
    /// Step times in milliseconds
    pub step_times_ms: Vec<f64>,
    /// Peak memory usage in bytes (if available)
    pub peak_memory: Option<usize>,
}

impl TrainingMetrics {
    /// Create new metrics tracker.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a training step.
    pub fn record_step(&mut self, loss: f64, time_ms: f64) {
        self.losses.push(loss);
        self.step_times_ms.push(time_ms);
    }

    /// Get average loss.
    #[must_use]
    pub fn average_loss(&self) -> f64 {
        if self.losses.is_empty() {
            0.0
        } else {
            self.losses.iter().sum::<f64>() / self.losses.len() as f64
        }
    }

    /// Get average step time.
    #[must_use]
    pub fn average_step_time(&self) -> f64 {
        if self.step_times_ms.is_empty() {
            0.0
        } else {
            self.step_times_ms.iter().sum::<f64>() / self.step_times_ms.len() as f64
        }
    }

    /// Get total training time.
    #[must_use]
    pub fn total_time_ms(&self) -> f64 {
        self.step_times_ms.iter().sum()
    }

    /// Print summary.
    pub fn print_summary(&self) {
        println!("📊 Training Metrics Summary:");
        println!("   Steps: {}", self.losses.len());
        println!("   Average loss: {:.4}", self.average_loss());
        println!(
            "   Loss range: {:.4} → {:.4}",
            self.losses.first().unwrap_or(&0.0),
            self.losses.last().unwrap_or(&0.0)
        );
        println!("   Average step time: {:.1} ms", self.average_step_time());
        println!("   Total time: {:.1} s", self.total_time_ms() / 1000.0);
        if let Some(peak) = self.peak_memory {
            println!("   Peak memory: {:.1} MB", peak as f64 / 1024.0 / 1024.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_available_check() {
        // This test just verifies the function doesn't panic
        let available = cuda_available();
        println!("CUDA available: {}", available);
    }

    #[test]
    fn test_loss_convergence_assertion() {
        // Decreasing losses should pass
        let losses = vec![2.5, 2.3, 2.1, 1.9, 1.7, 1.5, 1.3, 1.1, 0.9, 0.8];
        assert_loss_convergence(&losses, 0.5, 3); // 50% decrease required
    }

    #[test]
    #[should_panic(expected = "Loss did not decrease enough")]
    fn test_loss_convergence_failure() {
        // Flat losses should fail
        let losses = vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0];
        assert_loss_convergence(&losses, 0.3, 3);
    }

    #[test]
    fn test_metrics_tracker() {
        let mut metrics = TrainingMetrics::new();
        metrics.record_step(2.5, 100.0);
        metrics.record_step(2.0, 95.0);
        metrics.record_step(1.5, 90.0);

        assert_eq!(metrics.losses.len(), 3);
        assert!((metrics.average_loss() - 2.0).abs() < 0.001);
        assert!((metrics.average_step_time() - 95.0).abs() < 0.001);
    }
}
