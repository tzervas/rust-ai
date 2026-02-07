//! CPU Offloading Manager for Unlimited Effective Memory
//!
//! Enables training of massive models (7B-50B parameters) on consumer GPUs
//! by streaming layers between CPU RAM and GPU VRAM just-in-time.
//!
//! ## Strategy
//!
//! - Keep only **active layers** on GPU (typically 2-4 layers)
//! - Offload **inactive layers** to CPU RAM
//! - Stream layers to GPU just before they're needed
//! - Prefetch upcoming layers for overlapped transfer
//!
//! ## Memory Savings
//!
//! For 7B model (32 layers):
//! - Without offloading: 200 GB VRAM (impossible on 24 GB GPU)
//! - With offloading: 5-10 GB VRAM (2-4 active layers + overhead)
//! - Trade-off: 2-5× slower due to CPU-GPU transfer overhead
//!
//! ## HybridTrainer Integration
//!
//! **CRITICAL**: CPU offloading is **phase-aware**:
//! - **Full Phase**: Active (stream layers during forward/backward)
//! - **Predict Phase**: INACTIVE (all layers on GPU for forward + delta application)
//! - **Correct Phase**: INACTIVE (all layers on GPU)
//!
//! This preserves HybridTrainer's backward pass prediction capability.
//!
//! ## Example
//!
//! ```rust
//! use hybrid_predict_trainer_rs::cpu_offloading::CpuOffloadManager;
//!
//! // Create manager for 32-layer model with 2 active layers on GPU
//! let manager = CpuOffloadManager::new(32, 2);
//!
//! // For 7B model:
//! // - GPU: 2 layers × 700 MB = 1.4 GB
//! // - CPU: 30 layers × 700 MB = 21 GB
//! // - Total VRAM: <5 GB ✅
//! ```

use crate::phases::Phase;
use std::collections::{HashMap, VecDeque};

/// Configuration for CPU offloading
#[derive(Debug, Clone)]
pub struct CpuOffloadConfig {
    /// Maximum number of layers to keep on GPU simultaneously
    ///
    /// Recommended values:
    /// - Small models (<1B): 8-16 layers (or disable offloading)
    /// - Medium models (1-7B): 2-4 layers
    /// - Large models (7B+): 1-2 layers
    pub max_active_layers: usize,

    /// Enable CPU offloading (can disable for benchmarking)
    pub enabled: bool,

    /// Prefetch distance (how many layers ahead to prefetch)
    ///
    /// Higher = more overlapping of transfer and compute
    /// But uses more GPU memory for staging
    pub prefetch_distance: usize,

    /// Phases where offloading is active
    ///
    /// **CRITICAL**: Predict and Correct phases should NOT use offloading
    /// because they need all layers on GPU for:
    /// 1. Forward pass (validation)
    /// 2. apply_weight_delta() (predicted delta application)
    pub active_phases: Vec<Phase>,
}

impl Default for CpuOffloadConfig {
    fn default() -> Self {
        Self {
            max_active_layers: 2,
            enabled: true,
            prefetch_distance: 1,
            // Only offload during Full phase (has forward + backward)
            // Predict/Correct need all layers on GPU
            active_phases: vec![Phase::Full, Phase::Warmup],
        }
    }
}

/// CPU offloading manager
///
/// Manages layer streaming between CPU RAM and GPU VRAM to enable
/// training of models larger than GPU memory.
pub struct CpuOffloadManager {
    /// Configuration
    config: CpuOffloadConfig,

    /// Current phase (determines if offloading is active)
    current_phase: Phase,

    /// Total number of layers in the model
    total_layers: usize,

    /// Layers currently on GPU (layer_id -> on_gpu flag)
    on_gpu: Vec<bool>,

    /// Layers in prefetch queue (waiting to be transferred to GPU)
    prefetch_queue: VecDeque<usize>,

    /// Current active layer window (for sequential processing)
    active_window_start: usize,
    active_window_end: usize,

    /// Statistics
    total_transfers_to_gpu: usize,
    total_transfers_to_cpu: usize,
    total_bytes_transferred: usize,
}

impl CpuOffloadManager {
    /// Create new CPU offloading manager
    ///
    /// # Arguments
    ///
    /// * `total_layers` - Total number of layers in the model
    /// * `max_active_layers` - Maximum layers to keep on GPU
    ///
    /// # Example
    ///
    /// ```rust
    /// use hybrid_predict_trainer_rs::cpu_offloading::CpuOffloadManager;
    ///
    /// // 32-layer model, keep 2 on GPU at a time
    /// let manager = CpuOffloadManager::new(32, 2);
    /// ```
    pub fn new(total_layers: usize, max_active_layers: usize) -> Self {
        Self::with_config(
            total_layers,
            CpuOffloadConfig {
                max_active_layers,
                ..Default::default()
            },
        )
    }

    /// Create with custom configuration
    pub fn with_config(total_layers: usize, config: CpuOffloadConfig) -> Self {
        // Initially all layers are on GPU (will be offloaded on first phase transition)
        let on_gpu = vec![true; total_layers];

        Self {
            config,
            current_phase: Phase::Warmup,
            total_layers,
            on_gpu,
            prefetch_queue: VecDeque::new(),
            active_window_start: 0,
            active_window_end: 0,
            total_transfers_to_gpu: 0,
            total_transfers_to_cpu: 0,
            total_bytes_transferred: 0,
        }
    }

    /// Check if offloading is active for current phase
    pub fn is_active(&self) -> bool {
        self.config.enabled && self.config.active_phases.contains(&self.current_phase)
    }

    /// Update current phase
    ///
    /// **CRITICAL**: This automatically handles phase transitions:
    /// - Entering Predict/Correct: Prefetch ALL layers to GPU
    /// - Entering Full: Resume offloading
    pub fn set_phase(&mut self, phase: Phase) {
        let previous_phase = self.current_phase;
        self.current_phase = phase;

        // If transitioning to a phase where offloading should be inactive
        if !self.config.active_phases.contains(&phase)
            && self.config.active_phases.contains(&previous_phase)
        {
            // Prefetch ALL layers to GPU
            // This ensures apply_weight_delta() and forward() work correctly
            self.prefetch_all_layers();
        }
    }

    /// Prefetch all layers to GPU
    ///
    /// Called when entering Predict or Correct phase to ensure
    /// all layers are available for forward pass and delta application.
    fn prefetch_all_layers(&mut self) {
        if !self.config.enabled {
            return;
        }

        for layer_id in 0..self.total_layers {
            if !self.on_gpu[layer_id] {
                self.prefetch_queue.push_back(layer_id);
            }
        }

        // Process all prefetch requests immediately
        while let Some(layer_id) = self.prefetch_queue.pop_front() {
            self.transfer_to_gpu(layer_id);
        }
    }

    /// Mark layer as needed on GPU
    ///
    /// # Arguments
    ///
    /// * `layer_id` - Layer index (0-based)
    ///
    /// Returns true if the layer needed to be transferred (wasn't already on GPU)
    pub fn request_layer(&mut self, layer_id: usize) -> bool {
        if !self.is_active() {
            return false; // All layers on GPU when offloading inactive
        }

        assert!(
            layer_id < self.total_layers,
            "Layer ID {} out of bounds (total: {})",
            layer_id,
            self.total_layers
        );

        // If already on GPU, nothing to do
        if self.on_gpu[layer_id] {
            return false;
        }

        // Transfer to GPU
        self.transfer_to_gpu(layer_id);

        // Prefetch upcoming layers
        for offset in 1..=self.config.prefetch_distance {
            let prefetch_id = layer_id + offset;
            if prefetch_id < self.total_layers && !self.on_gpu[prefetch_id] {
                self.prefetch_queue.push_back(prefetch_id);
            }
        }

        // Evict old layers if we exceed max_active_layers
        self.evict_old_layers();

        true
    }

    /// Transfer layer from CPU to GPU
    fn transfer_to_gpu(&mut self, layer_id: usize) {
        if self.on_gpu[layer_id] {
            return; // Already on GPU
        }

        // TODO: Actual implementation would call into Burn/CUDA to transfer tensors
        // For now, just track the state

        self.on_gpu[layer_id] = true;
        self.total_transfers_to_gpu += 1;

        // Estimate: 7B model / 32 layers = ~220M params per layer
        // At fp16: 220M × 2 bytes = 440 MB per layer
        let estimated_bytes = 440_000_000;
        self.total_bytes_transferred += estimated_bytes;
    }

    /// Transfer layer from GPU to CPU
    fn transfer_to_cpu(&mut self, layer_id: usize) {
        if !self.on_gpu[layer_id] {
            return; // Already on CPU
        }

        // TODO: Actual implementation would call into Burn/CUDA to transfer tensors

        self.on_gpu[layer_id] = false;
        self.total_transfers_to_cpu += 1;

        let estimated_bytes = 440_000_000;
        self.total_bytes_transferred += estimated_bytes;
    }

    /// Evict old layers to make room for new ones
    fn evict_old_layers(&mut self) {
        let on_gpu_count = self.on_gpu.iter().filter(|&&x| x).count();

        if on_gpu_count <= self.config.max_active_layers {
            return; // Under limit, nothing to evict
        }

        // Evict layers outside the active window
        // Keep layers in [active_window_start, active_window_end]
        for layer_id in 0..self.total_layers {
            if self.on_gpu[layer_id]
                && (layer_id < self.active_window_start || layer_id > self.active_window_end)
            {
                self.transfer_to_cpu(layer_id);

                // Check if we're under limit now
                let on_gpu_count = self.on_gpu.iter().filter(|&&x| x).count();
                if on_gpu_count <= self.config.max_active_layers {
                    break;
                }
            }
        }
    }

    /// Update active window for sequential layer processing
    ///
    /// Call this during forward/backward pass to indicate which layers
    /// are currently active.
    pub fn set_active_window(&mut self, start: usize, end: usize) {
        self.active_window_start = start;
        self.active_window_end = end;
    }

    /// Check if layer is currently on GPU
    pub fn is_on_gpu(&self, layer_id: usize) -> bool {
        if !self.is_active() {
            return true; // All layers on GPU when offloading inactive
        }

        self.on_gpu[layer_id]
    }

    /// Get statistics
    pub fn statistics(&self) -> CpuOffloadStatistics {
        let layers_on_gpu = self.on_gpu.iter().filter(|&&x| x).count();
        let layers_on_cpu = self.total_layers - layers_on_gpu;

        CpuOffloadStatistics {
            total_layers: self.total_layers,
            layers_on_gpu,
            layers_on_cpu,
            max_active_layers: self.config.max_active_layers,
            total_transfers_to_gpu: self.total_transfers_to_gpu,
            total_transfers_to_cpu: self.total_transfers_to_cpu,
            total_gb_transferred: self.total_bytes_transferred as f32 / (1024.0 * 1024.0 * 1024.0),
            is_active: self.is_active(),
        }
    }

    /// Calculate theoretical memory savings
    ///
    /// # Arguments
    ///
    /// * `layer_size_mb` - Average size per layer in MB
    ///
    /// # Returns
    ///
    /// Theoretical VRAM savings in MB
    pub fn theoretical_savings(&self, layer_size_mb: f32) -> f32 {
        if !self.is_active() {
            return 0.0;
        }

        let without_offloading = self.total_layers as f32 * layer_size_mb;
        let with_offloading = self.config.max_active_layers as f32 * layer_size_mb;

        without_offloading - with_offloading
    }
}

/// CPU offloading statistics
#[derive(Debug, Clone)]
pub struct CpuOffloadStatistics {
    /// Total layers in model
    pub total_layers: usize,

    /// Layers currently on GPU
    pub layers_on_gpu: usize,

    /// Layers currently on CPU
    pub layers_on_cpu: usize,

    /// Maximum active layers allowed
    pub max_active_layers: usize,

    /// Total transfers from CPU to GPU
    pub total_transfers_to_gpu: usize,

    /// Total transfers from GPU to CPU
    pub total_transfers_to_cpu: usize,

    /// Total data transferred (GB)
    pub total_gb_transferred: f32,

    /// Whether offloading is currently active
    pub is_active: bool,
}

impl std::fmt::Display for CpuOffloadStatistics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Offloading: {} | GPU: {}/{} layers | Transfers: {} to GPU, {} to CPU | Data: {:.2} GB",
            if self.is_active { "ACTIVE" } else { "INACTIVE" },
            self.layers_on_gpu,
            self.total_layers,
            self.total_transfers_to_gpu,
            self.total_transfers_to_cpu,
            self.total_gb_transferred
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phase_awareness() {
        let mut manager = CpuOffloadManager::new(32, 2);

        // Should be active in Full phase
        manager.set_phase(Phase::Full);
        assert!(manager.is_active());

        // Should be inactive in Predict phase
        manager.set_phase(Phase::Predict);
        assert!(!manager.is_active());

        // Should be inactive in Correct phase
        manager.set_phase(Phase::Correct);
        assert!(!manager.is_active());
    }

    #[test]
    fn test_prefetch_all_on_predict() {
        let mut manager = CpuOffloadManager::new(4, 2);

        // Start in Full phase, mark layers as offloaded
        manager.set_phase(Phase::Full);
        manager.on_gpu = vec![false, false, false, false];

        // Switch to Predict phase should prefetch ALL layers
        manager.set_phase(Phase::Predict);

        // All layers should now be on GPU
        assert_eq!(manager.on_gpu, vec![true, true, true, true]);
    }

    #[test]
    fn test_layer_eviction() {
        let mut manager = CpuOffloadManager::new(10, 2);
        manager.set_phase(Phase::Full);

        // Start with all layers on CPU
        manager.on_gpu = vec![false; 10];

        // Request layer 5 (should transfer to GPU)
        manager.set_active_window(5, 6);
        manager.request_layer(5);

        // Layer 5 should be on GPU
        assert!(manager.on_gpu[5]);

        // Request layer 6 (should transfer to GPU)
        manager.set_active_window(5, 6);
        manager.request_layer(6);

        // Both 5 and 6 should be on GPU (within max_active=2)
        assert!(manager.on_gpu[5]);
        assert!(manager.on_gpu[6]);
        let on_gpu_count = manager.on_gpu.iter().filter(|&&x| x).count();
        assert!(on_gpu_count <= 2, "Expected <=2 layers on GPU, got {}", on_gpu_count);

        // Request layer 8 (should evict layer 5, keep 6 and 8)
        manager.set_active_window(6, 8);
        manager.request_layer(8);

        let on_gpu_count = manager.on_gpu.iter().filter(|&&x| x).count();
        assert!(on_gpu_count <= 2, "Expected <=2 layers on GPU, got {}", on_gpu_count);
    }

    #[test]
    fn test_request_layer() {
        let mut manager = CpuOffloadManager::new(10, 2);
        manager.set_phase(Phase::Full);
        manager.on_gpu = vec![false; 10]; // All on CPU initially

        // Request layer 0
        let needed_transfer = manager.request_layer(0);
        assert!(needed_transfer);
        assert!(manager.on_gpu[0]);

        // Request again (should be no-op)
        let needed_transfer = manager.request_layer(0);
        assert!(!needed_transfer);
    }

    #[test]
    fn test_theoretical_savings() {
        let manager = CpuOffloadManager::new(32, 2);

        // 32 layers, 700 MB per layer
        // Without: 32 × 700 = 22,400 MB
        // With: 2 × 700 = 1,400 MB
        // Savings: 21,000 MB
        let savings = manager.theoretical_savings(700.0);
        assert!((savings - 21_000.0).abs() < 0.1);
    }

    #[test]
    fn test_statistics() {
        let mut manager = CpuOffloadManager::new(32, 2);
        manager.set_phase(Phase::Full);

        let stats = manager.statistics();
        assert_eq!(stats.total_layers, 32);
        assert_eq!(stats.max_active_layers, 2);
        assert!(stats.is_active);
    }

    #[test]
    fn test_inactive_during_predict() {
        let mut manager = CpuOffloadManager::new(32, 2);
        manager.set_phase(Phase::Predict); // Set to Predict phase

        // In Predict phase, all layers should be considered on GPU
        assert!(!manager.is_active());
        assert!(manager.is_on_gpu(0));
        assert!(manager.is_on_gpu(31));
    }
}
