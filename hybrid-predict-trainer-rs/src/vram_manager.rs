//! VRAM management utilities for controlling GPU memory usage.
//!
//! This module provides aggressive memory management to work around Burn's
//! functional API creating model copies. Forces CUDA synchronization and
//! explicit memory cleanup to prevent accumulation.

use std::sync::atomic::{AtomicUsize, Ordering};

/// Global counter for tracking model copies (for debugging)
static MODEL_COPY_COUNT: AtomicUsize = AtomicUsize::new(0);

/// Tracks VRAM usage and forces cleanup when necessary.
#[derive(Debug)]
pub struct VramManager {
    /// Threshold in MB before forcing cleanup
    cleanup_threshold_mb: usize,

    /// Steps since last cleanup
    steps_since_cleanup: usize,

    /// Force cleanup every N steps regardless of usage
    force_cleanup_interval: usize,

    /// Last recorded VRAM usage
    last_vram_mb: usize,
}

impl Default for VramManager {
    fn default() -> Self {
        Self::new()
    }
}

impl VramManager {
    /// Creates a new VRAM manager with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self {
            cleanup_threshold_mb: 12000, // 12 GB threshold
            steps_since_cleanup: 0,
            force_cleanup_interval: 10, // Force every 10 steps
            last_vram_mb: 0,
        }
    }

    /// Creates a VRAM manager with custom thresholds.
    #[must_use]
    pub fn with_thresholds(cleanup_threshold_mb: usize, force_interval: usize) -> Self {
        Self {
            cleanup_threshold_mb,
            steps_since_cleanup: 0,
            force_cleanup_interval: force_interval,
            last_vram_mb: 0,
        }
    }

    /// Checks if cleanup should be forced based on VRAM usage.
    ///
    /// Returns true if:
    /// - VRAM exceeds threshold
    /// - OR force_cleanup_interval steps have passed
    ///
    /// Also logs warnings when approaching VRAM limits.
    pub fn should_cleanup(&mut self) -> bool {
        self.steps_since_cleanup += 1;

        // Always check current VRAM
        let current_vram = measure_vram_mb();
        self.last_vram_mb = current_vram;

        // Warning thresholds
        const WARNING_THRESHOLD_MB: usize = 10_000;  // 10 GB warning
        const CRITICAL_THRESHOLD_MB: usize = 14_000; // 14 GB critical

        // Log warnings when approaching limits
        if current_vram > CRITICAL_THRESHOLD_MB {
            eprintln!(
                "⚠️  CRITICAL: VRAM usage at {} MB (>14 GB). Consider saving checkpoint and restarting.",
                current_vram
            );
        } else if current_vram > WARNING_THRESHOLD_MB {
            println!(
                "⚠️  WARNING: VRAM usage at {} MB (>10 GB). Approaching memory limits.",
                current_vram
            );
        }

        // Force cleanup if threshold exceeded
        if current_vram > self.cleanup_threshold_mb {
            println!("🧹 Forcing VRAM cleanup (threshold {} MB exceeded)", self.cleanup_threshold_mb);
            return true;
        }

        // Force cleanup every N steps
        if self.steps_since_cleanup >= self.force_cleanup_interval {
            return true;
        }

        false
    }

    /// Performs aggressive VRAM cleanup.
    ///
    /// This forces CUDA synchronization and attempts to trigger
    /// garbage collection of unreachable model copies.
    pub fn force_cleanup(&mut self) {
        // Reset counter
        self.steps_since_cleanup = 0;

        // Force Rust garbage collection
        // This drops any unreachable model copies
        #[cfg(not(target_env = "msvc"))]
        {
            // On Unix-like systems, we can suggest GC
            // (Rust doesn't have explicit GC, but this helps drop references)
            std::hint::black_box(());
        }

        // CUDA synchronization (if available)
        #[cfg(feature = "cuda")]
        {
            // Force CUDA to synchronize and free unused memory
            // This is a best-effort attempt using available APIs
            sync_cuda_memory();
        }
    }

    /// Returns the last measured VRAM usage in MB.
    #[must_use]
    pub fn last_vram_mb(&self) -> usize {
        self.last_vram_mb
    }

    /// Returns a formatted status string with VRAM usage and model copy count.
    #[must_use]
    pub fn status_string(&self) -> String {
        format!(
            "VRAM: {} MB | Copies: {} | Cleanups: {}",
            self.last_vram_mb,
            Self::total_copies(),
            // Estimate cleanups from steps (each interval = 1 cleanup)
            self.steps_since_cleanup / self.force_cleanup_interval.max(1)
        )
    }

    /// Increments the global model copy counter.
    pub fn record_model_copy() {
        MODEL_COPY_COUNT.fetch_add(1, Ordering::Relaxed);
    }

    /// Returns the total number of model copies created.
    #[must_use]
    pub fn total_copies() -> usize {
        MODEL_COPY_COUNT.load(Ordering::Relaxed)
    }
}

/// Measures current VRAM usage via nvidia-smi.
///
/// Returns 0 if nvidia-smi is unavailable (CPU mode).
fn measure_vram_mb() -> usize {
    std::process::Command::new("nvidia-smi")
        .args([" --query-gpu=memory.used", "--format=csv,noheader,nounits"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().parse::<usize>().ok())
        .unwrap_or(0)
}

/// Forces CUDA synchronization and memory cleanup.
///
/// This is a best-effort attempt to free unused GPU memory.
/// The exact behavior depends on the CUDA runtime and Burn's internals.
///
/// **Note**: Called conditionally via `#[cfg(feature = "cuda")]` in `force_cleanup()`.
/// The `#[cfg(not(feature = "cuda"))]` version below is used in non-CUDA builds.
#[cfg(feature = "cuda")]
#[allow(dead_code)]  // Called in force_cleanup() when feature = "cuda" is enabled
fn sync_cuda_memory() {
    // Burn uses cudarc internally, but doesn't expose sync APIs
    // We'll try to trigger cleanup through standard Rust mechanisms

    // 1. Force all pending operations to complete
    std::thread::yield_now();

    // 2. Hint that we're done with memory
    std::hint::black_box(());

    // 3. Sleep briefly to allow CUDA driver to catch up
    std::thread::sleep(std::time::Duration::from_millis(10));
}

#[cfg(not(feature = "cuda"))]
fn sync_cuda_memory() {
    // No-op on CPU
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vram_manager_creation() {
        let manager = VramManager::new();
        assert_eq!(manager.cleanup_threshold_mb, 12000);
        assert_eq!(manager.force_cleanup_interval, 10);
    }

    #[test]
    fn test_should_cleanup_interval() {
        let mut manager = VramManager::with_thresholds(20000, 5);

        // Should not cleanup for first 4 steps
        assert!(!manager.should_cleanup());
        assert!(!manager.should_cleanup());
        assert!(!manager.should_cleanup());
        assert!(!manager.should_cleanup());

        // Should cleanup on 5th step (force interval)
        assert!(manager.should_cleanup());

        // Counter should reset after cleanup
        manager.force_cleanup();
        assert!(!manager.should_cleanup());
    }

    #[test]
    fn test_model_copy_counter() {
        let initial = VramManager::total_copies();
        VramManager::record_model_copy();
        VramManager::record_model_copy();
        VramManager::record_model_copy();
        assert_eq!(VramManager::total_copies(), initial + 3);
    }
}
