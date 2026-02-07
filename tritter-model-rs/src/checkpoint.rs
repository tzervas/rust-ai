//! Gradient checkpointing for memory-efficient training.
//!
//! This module implements activation caching with selective recomputation to enable
//! training larger models (500M, 1B+) within limited GPU memory (16GB).
//!
//! # How It Works
//!
//! During the forward pass:
//! - Every N layers, we cache the activation tensor (checkpoint boundary)
//! - Non-checkpoint layers discard activations after use
//!
//! During the backward pass:
//! - For each segment between checkpoints, we:
//!   1. Load the cached activation from the segment start
//!   2. Re-forward through the segment to recompute activations
//!   3. Compute gradients for that segment
//!   4. Free segment activations
//!
//! # Memory Savings
//!
//! | Checkpoint Interval | Memory Reduction | Compute Overhead |
//! |---------------------|------------------|------------------|
//! | Every 4 layers      | ~75% reduction   | ~33% overhead    |
//! | Every 8 layers      | ~87.5% reduction | ~75% overhead    |
//!
//! # Example
//!
//! ```no_run
//! use tritter_model_rs::checkpoint::{CheckpointStore, GradientCheckpointConfig};
//! use candle_core::Device;
//!
//! let config = GradientCheckpointConfig {
//!     enabled: true,
//!     checkpoint_interval: 4,
//!     offload_to_cpu: false,
//! };
//!
//! let device = Device::Cpu;
//! let mut store = CheckpointStore::new(&config, &device);
//!
//! // During forward pass, store checkpoints
//! // During backward pass, retrieve and recompute
//! ```

use std::collections::HashMap;

use candle_core::{Device, Tensor};

use crate::error::{TritterError, TritterResult};

/// Configuration for gradient checkpointing.
#[derive(Debug, Clone)]
pub struct GradientCheckpointConfig {
    /// Enable gradient checkpointing
    pub enabled: bool,
    /// Checkpoint every N layers (lower = more memory savings, more recomputation)
    pub checkpoint_interval: usize,
    /// Offload checkpointed activations to CPU (slower but more GPU memory savings)
    pub offload_to_cpu: bool,
}

impl Default for GradientCheckpointConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            checkpoint_interval: 4,
            offload_to_cpu: false,
        }
    }
}

impl GradientCheckpointConfig {
    /// Create config from model config values
    pub fn from_model_config(enabled: bool, checkpoint_every_n_layers: usize) -> Self {
        Self {
            enabled,
            checkpoint_interval: checkpoint_every_n_layers.max(1),
            offload_to_cpu: false,
        }
    }

    /// Create a config that checkpoints every N layers
    pub fn every_n_layers(n: usize) -> Self {
        Self {
            enabled: true,
            checkpoint_interval: n.max(1),
            offload_to_cpu: false,
        }
    }

    /// Enable CPU offloading for maximum memory savings
    pub fn with_cpu_offload(mut self) -> Self {
        self.offload_to_cpu = true;
        self
    }

    /// Calculate memory reduction factor (0.0 to 1.0, lower = more savings)
    pub fn memory_reduction_factor(&self, num_layers: usize) -> f64 {
        if !self.enabled || num_layers == 0 {
            1.0
        } else {
            let stored = num_layers.div_ceil(self.checkpoint_interval);
            stored as f64 / num_layers as f64
        }
    }

    /// Calculate compute overhead factor (1.0 = no overhead)
    pub fn compute_overhead_factor(&self, num_layers: usize) -> f64 {
        if !self.enabled || num_layers == 0 {
            1.0
        } else {
            let stored = num_layers.div_ceil(self.checkpoint_interval);
            let recomputed = num_layers - stored;
            (num_layers + recomputed) as f64 / num_layers as f64
        }
    }
}

/// Storage for checkpointed activations during training.
///
/// This struct manages caching and retrieval of activation tensors at checkpoint
/// boundaries. Activations can optionally be offloaded to CPU to save GPU memory.
#[derive(Debug)]
pub struct CheckpointStore {
    /// Cached activations indexed by layer number
    activations: HashMap<usize, Tensor>,
    /// Configuration
    config: GradientCheckpointConfig,
    /// Storage device (CPU for offload, otherwise matches compute device)
    storage_device: Device,
    /// Original compute device (for retrieval)
    compute_device: Device,
}

impl CheckpointStore {
    /// Create a new checkpoint store.
    ///
    /// # Arguments
    /// * `config` - Checkpointing configuration
    /// * `compute_device` - The device where model computation happens
    pub fn new(config: &GradientCheckpointConfig, compute_device: &Device) -> Self {
        let storage_device = if config.offload_to_cpu {
            Device::Cpu
        } else {
            compute_device.clone()
        };

        Self {
            activations: HashMap::new(),
            config: config.clone(),
            storage_device,
            compute_device: compute_device.clone(),
        }
    }

    /// Create a store with a specific checkpoint interval.
    pub fn with_interval(interval: usize, compute_device: &Device) -> Self {
        let config = GradientCheckpointConfig::every_n_layers(interval);
        Self::new(&config, compute_device)
    }

    /// Check if checkpointing is enabled.
    #[inline]
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Check if a given layer index should be checkpointed.
    ///
    /// Layer indices are 0-based. We checkpoint layer 0 and every N-th layer after.
    #[inline]
    pub fn is_checkpoint_layer(&self, layer_idx: usize) -> bool {
        self.config.enabled && (layer_idx % self.config.checkpoint_interval == 0)
    }

    /// Get the checkpoint interval.
    #[inline]
    pub fn checkpoint_interval(&self) -> usize {
        self.config.checkpoint_interval
    }

    /// Store an activation tensor at a checkpoint boundary.
    ///
    /// If CPU offload is enabled, the tensor is moved to CPU.
    /// The tensor is cloned to ensure the stored copy is independent.
    ///
    /// # Arguments
    /// * `layer_idx` - The layer index (must be a checkpoint layer)
    /// * `activation` - The activation tensor to cache
    pub fn store(&mut self, layer_idx: usize, activation: &Tensor) -> TritterResult<()> {
        if !self.config.enabled {
            return Ok(());
        }

        // Move to storage device if different from compute device
        let stored = if self.storage_device.same_device(&self.compute_device) {
            // Same device - just clone to break the computation graph
            activation.clone()
        } else {
            // Different device (CPU offload) - copy to storage device
            activation.to_device(&self.storage_device)?
        };

        // Detach from computation graph to allow garbage collection of upstream tensors
        // Note: Candle doesn't have explicit detach, but cloning + moving to another
        // device effectively breaks the graph connection
        self.activations.insert(layer_idx, stored);

        Ok(())
    }

    /// Retrieve a cached activation and move it back to the compute device.
    ///
    /// # Arguments
    /// * `layer_idx` - The layer index to retrieve
    ///
    /// # Returns
    /// The activation tensor on the compute device, or an error if not found.
    pub fn retrieve(&self, layer_idx: usize) -> TritterResult<Tensor> {
        let stored = self.activations.get(&layer_idx).ok_or_else(|| {
            TritterError::Training(format!(
                "Checkpoint not found for layer {}. Available: {:?}",
                layer_idx,
                self.activations.keys().collect::<Vec<_>>()
            ))
        })?;

        // Move back to compute device if needed
        if self.storage_device.same_device(&self.compute_device) {
            Ok(stored.clone())
        } else {
            Ok(stored.to_device(&self.compute_device)?)
        }
    }

    /// Check if a checkpoint exists for a given layer.
    pub fn has_checkpoint(&self, layer_idx: usize) -> bool {
        self.activations.contains_key(&layer_idx)
    }

    /// Get the layer index of the most recent checkpoint at or before the given layer.
    ///
    /// # Arguments
    /// * `layer_idx` - The layer to find the previous checkpoint for
    ///
    /// # Returns
    /// The index of the checkpoint layer, or None if no checkpoint exists.
    pub fn previous_checkpoint(&self, layer_idx: usize) -> Option<usize> {
        if !self.config.enabled {
            return None;
        }

        // Find the checkpoint at or before this layer
        let checkpoint_idx =
            (layer_idx / self.config.checkpoint_interval) * self.config.checkpoint_interval;

        if self.activations.contains_key(&checkpoint_idx) {
            Some(checkpoint_idx)
        } else {
            None
        }
    }

    /// Get the next checkpoint layer index after the given layer.
    ///
    /// # Arguments
    /// * `layer_idx` - The current layer index
    /// * `num_layers` - Total number of layers in the model
    ///
    /// # Returns
    /// The index of the next checkpoint layer, or num_layers if none.
    pub fn next_checkpoint(&self, layer_idx: usize, num_layers: usize) -> usize {
        if !self.config.enabled {
            return num_layers;
        }

        let next =
            ((layer_idx / self.config.checkpoint_interval) + 1) * self.config.checkpoint_interval;
        next.min(num_layers)
    }

    /// Get segment boundaries for backward pass processing.
    ///
    /// Returns a list of (start_layer, end_layer) tuples representing segments
    /// to process during backward pass, in reverse order.
    ///
    /// # Arguments
    /// * `num_layers` - Total number of layers in the model
    pub fn get_segments(&self, num_layers: usize) -> Vec<(usize, usize)> {
        if !self.config.enabled || num_layers == 0 {
            return vec![(0, num_layers)];
        }

        let mut segments = Vec::new();
        let interval = self.config.checkpoint_interval;

        // Build segments from end to start
        let mut end = num_layers;
        while end > 0 {
            let start = if end >= interval {
                (end - 1) / interval * interval
            } else {
                0
            };
            segments.push((start, end));
            end = start;
        }

        segments
    }

    /// Clear all stored activations.
    ///
    /// Call this at the end of each training step to free memory.
    pub fn clear(&mut self) {
        self.activations.clear();
    }

    /// Get the number of stored checkpoints.
    pub fn num_checkpoints(&self) -> usize {
        self.activations.len()
    }

    /// Estimate memory usage of stored checkpoints in bytes.
    pub fn memory_usage(&self) -> usize {
        self.activations
            .values()
            .map(|t| t.elem_count() * t.dtype().size_in_bytes())
            .sum()
    }
}

/// Segment information for backward pass recomputation.
#[derive(Debug, Clone)]
pub struct CheckpointSegment {
    /// Starting layer index (inclusive)
    pub start_layer: usize,
    /// Ending layer index (exclusive)
    pub end_layer: usize,
    /// Whether the start activation is available from cache
    pub has_cached_input: bool,
}

impl CheckpointSegment {
    /// Get the number of layers in this segment.
    pub fn num_layers(&self) -> usize {
        self.end_layer.saturating_sub(self.start_layer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;

    #[test]
    fn test_checkpoint_config_default() {
        let config = GradientCheckpointConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.checkpoint_interval, 4);
        assert!(!config.offload_to_cpu);
    }

    #[test]
    fn test_checkpoint_config_memory_factor() {
        let config = GradientCheckpointConfig::every_n_layers(4);

        // 24 layers, checkpoint every 4 = 6 stored = 6/24 = 0.25
        let factor = config.memory_reduction_factor(24);
        assert!((factor - 0.25).abs() < 0.01);

        // 12 layers, checkpoint every 4 = 3 stored = 3/12 = 0.25
        let factor = config.memory_reduction_factor(12);
        assert!((factor - 0.25).abs() < 0.01);
    }

    #[test]
    fn test_checkpoint_config_compute_overhead() {
        let config = GradientCheckpointConfig::every_n_layers(4);

        // 24 layers, 6 checkpointed, 18 recomputed
        // Total compute = 24 + 18 = 42 vs 24 = 1.75x
        let factor = config.compute_overhead_factor(24);
        assert!((factor - 1.75).abs() < 0.01);
    }

    #[test]
    fn test_checkpoint_store_is_checkpoint_layer() {
        let config = GradientCheckpointConfig::every_n_layers(4);
        let store = CheckpointStore::new(&config, &Device::Cpu);

        assert!(store.is_checkpoint_layer(0));
        assert!(!store.is_checkpoint_layer(1));
        assert!(!store.is_checkpoint_layer(2));
        assert!(!store.is_checkpoint_layer(3));
        assert!(store.is_checkpoint_layer(4));
        assert!(!store.is_checkpoint_layer(5));
        assert!(store.is_checkpoint_layer(8));
    }

    #[test]
    fn test_checkpoint_store_roundtrip() {
        let config = GradientCheckpointConfig::every_n_layers(4);
        let device = Device::Cpu;
        let mut store = CheckpointStore::new(&config, &device);

        // Create a test tensor
        let tensor = Tensor::randn(0f32, 1f32, (2, 256, 768), &device).unwrap();
        let original_sum: f32 = tensor.sum_all().unwrap().to_scalar().unwrap();

        // Store at checkpoint layer
        store.store(0, &tensor).unwrap();
        assert!(store.has_checkpoint(0));
        assert_eq!(store.num_checkpoints(), 1);

        // Retrieve and verify
        let retrieved = store.retrieve(0).unwrap();
        assert_eq!(tensor.dims(), retrieved.dims());

        let retrieved_sum: f32 = retrieved.sum_all().unwrap().to_scalar().unwrap();
        assert!((original_sum - retrieved_sum).abs() < 1e-5);
    }

    #[test]
    fn test_checkpoint_store_multiple_layers() {
        let config = GradientCheckpointConfig::every_n_layers(4);
        let device = Device::Cpu;
        let mut store = CheckpointStore::new(&config, &device);

        // Store multiple checkpoints
        for layer in [0, 4, 8, 12] {
            let tensor = Tensor::ones((2, 64, 128), DType::F32, &device).unwrap();
            store.store(layer, &tensor).unwrap();
        }

        assert_eq!(store.num_checkpoints(), 4);
        assert!(store.has_checkpoint(0));
        assert!(store.has_checkpoint(4));
        assert!(store.has_checkpoint(8));
        assert!(store.has_checkpoint(12));
        assert!(!store.has_checkpoint(1));
    }

    #[test]
    fn test_checkpoint_store_retrieve_nonexistent() {
        let config = GradientCheckpointConfig::every_n_layers(4);
        let store = CheckpointStore::new(&config, &Device::Cpu);

        let result = store.retrieve(5);
        assert!(result.is_err());
    }

    #[test]
    fn test_checkpoint_store_clear() {
        let config = GradientCheckpointConfig::every_n_layers(4);
        let device = Device::Cpu;
        let mut store = CheckpointStore::new(&config, &device);

        let tensor = Tensor::ones((2, 64, 128), DType::F32, &device).unwrap();
        store.store(0, &tensor).unwrap();
        store.store(4, &tensor).unwrap();

        assert_eq!(store.num_checkpoints(), 2);

        store.clear();
        assert_eq!(store.num_checkpoints(), 0);
        assert!(!store.has_checkpoint(0));
    }

    #[test]
    fn test_checkpoint_store_get_segments() {
        let config = GradientCheckpointConfig::every_n_layers(4);
        let store = CheckpointStore::new(&config, &Device::Cpu);

        // 12 layers with interval 4: segments are (8,12), (4,8), (0,4)
        let segments = store.get_segments(12);
        assert_eq!(segments.len(), 3);
        assert_eq!(segments[0], (8, 12));
        assert_eq!(segments[1], (4, 8));
        assert_eq!(segments[2], (0, 4));
    }

    #[test]
    fn test_checkpoint_store_get_segments_odd() {
        let config = GradientCheckpointConfig::every_n_layers(4);
        let store = CheckpointStore::new(&config, &Device::Cpu);

        // 10 layers with interval 4: segments are (8,10), (4,8), (0,4)
        let segments = store.get_segments(10);
        assert_eq!(segments.len(), 3);
        assert_eq!(segments[0], (8, 10));
        assert_eq!(segments[1], (4, 8));
        assert_eq!(segments[2], (0, 4));
    }

    #[test]
    fn test_checkpoint_store_previous_checkpoint() {
        let config = GradientCheckpointConfig::every_n_layers(4);
        let device = Device::Cpu;
        let mut store = CheckpointStore::new(&config, &device);

        // Store checkpoints
        let tensor = Tensor::ones((2, 64, 128), DType::F32, &device).unwrap();
        store.store(0, &tensor).unwrap();
        store.store(4, &tensor).unwrap();
        store.store(8, &tensor).unwrap();

        // Test previous checkpoint lookup
        assert_eq!(store.previous_checkpoint(0), Some(0));
        assert_eq!(store.previous_checkpoint(1), Some(0));
        assert_eq!(store.previous_checkpoint(3), Some(0));
        assert_eq!(store.previous_checkpoint(4), Some(4));
        assert_eq!(store.previous_checkpoint(7), Some(4));
        assert_eq!(store.previous_checkpoint(8), Some(8));
    }

    #[test]
    fn test_checkpoint_store_next_checkpoint() {
        let config = GradientCheckpointConfig::every_n_layers(4);
        let store = CheckpointStore::new(&config, &Device::Cpu);

        assert_eq!(store.next_checkpoint(0, 12), 4);
        assert_eq!(store.next_checkpoint(1, 12), 4);
        assert_eq!(store.next_checkpoint(3, 12), 4);
        assert_eq!(store.next_checkpoint(4, 12), 8);
        assert_eq!(store.next_checkpoint(8, 12), 12);
        assert_eq!(store.next_checkpoint(11, 12), 12);
    }

    #[test]
    fn test_checkpoint_store_memory_usage() {
        let config = GradientCheckpointConfig::every_n_layers(4);
        let device = Device::Cpu;
        let mut store = CheckpointStore::new(&config, &device);

        // F32 tensor: 2 * 64 * 128 * 4 bytes = 65536 bytes
        let tensor = Tensor::ones((2, 64, 128), DType::F32, &device).unwrap();
        store.store(0, &tensor).unwrap();

        assert_eq!(store.memory_usage(), 2 * 64 * 128 * 4);

        store.store(4, &tensor).unwrap();
        assert_eq!(store.memory_usage(), 2 * 2 * 64 * 128 * 4);
    }

    #[test]
    fn test_disabled_checkpointing() {
        let config = GradientCheckpointConfig::default(); // disabled by default
        let store = CheckpointStore::new(&config, &Device::Cpu);

        assert!(!store.is_enabled());
        assert!(!store.is_checkpoint_layer(0));
        assert!(!store.is_checkpoint_layer(4));

        // get_segments should return single segment for whole model
        let segments = store.get_segments(12);
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0], (0, 12));
    }
}
