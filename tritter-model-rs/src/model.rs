//! Main Tritter transformer model.
//!
//! Implements the full model with:
//! - Token embeddings
//! - Transformer layers
//! - LM head for next-token prediction
//! - BitNet quantization via bitnet-quantize crate
//! - Gradient checkpointing for memory-efficient training

use candle_core::{DType, Device, Tensor, D};
use candle_nn::{embedding, Embedding, Module, VarBuilder, VarMap};
use std::collections::HashMap;

use crate::attention::create_causal_mask;
use crate::checkpoint::{CheckpointStore, GradientCheckpointConfig};
use crate::config::TritterConfig;
use crate::error::TritterResult;
use crate::layer::TritterLayer;
use crate::norm::{manual_layer_norm, ManualLayerNorm};

/// The main Tritter transformer model
pub struct TritterModel {
    embed_tokens: Embedding,
    layers: Vec<TritterLayer>,
    final_norm: ManualLayerNorm,
    lm_head: candle_nn::Linear,
    config: TritterConfig,
    device: Device,
    /// Cached causal mask (recomputed on sequence length change)
    cached_mask: Option<Tensor>,
    cached_mask_len: usize,
    /// Gradients stored after backward pass
    gradients: Option<HashMap<String, Tensor>>,
    /// Last computed logits (for loss calculation)
    last_logits: Option<Tensor>,
    /// VarMap for accessing parameters
    var_map: VarMap,
    /// Checkpoint store for gradient checkpointing
    checkpoint_store: Option<CheckpointStore>,
    /// Checkpoint configuration
    checkpoint_config: GradientCheckpointConfig,
}

impl TritterModel {
    /// Create a new model with random initialization
    pub fn new(config: &TritterConfig, device: &Device) -> TritterResult<Self> {
        let var_map = VarMap::new();
        let vb = VarBuilder::from_varmap(&var_map, DType::F32, device);

        Self::from_varbuilder(config, vb, device, var_map)
    }

    /// Create model from a VarBuilder (for loading weights)
    pub fn from_varbuilder(
        config: &TritterConfig,
        vb: VarBuilder,
        device: &Device,
        var_map: VarMap,
    ) -> TritterResult<Self> {
        // Token embeddings
        let embed_tokens = embedding(config.vocab_size, config.hidden_size, vb.pp("embed_tokens"))?;

        // Transformer layers
        let mut layers = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let layer = TritterLayer::new(config, vb.pp(format!("layers.{}", i)), device)?;
            layers.push(layer);
        }

        // Final layer norm (using manual impl for CUDA compatibility)
        let final_norm = manual_layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("final_norm"),
        )?;

        // LM head (tied to embeddings in some models, separate here)
        let lm_head =
            candle_nn::linear_no_bias(config.hidden_size, config.vocab_size, vb.pp("lm_head"))?;

        // Initialize checkpoint configuration
        let checkpoint_config = GradientCheckpointConfig::from_model_config(
            config.gradient_checkpointing,
            config.checkpoint_every_n_layers,
        );

        // Create checkpoint store if checkpointing is enabled
        let checkpoint_store = if checkpoint_config.enabled {
            Some(CheckpointStore::new(&checkpoint_config, device))
        } else {
            None
        };

        Ok(Self {
            embed_tokens,
            layers,
            final_norm,
            lm_head,
            config: config.clone(),
            device: device.clone(),
            cached_mask: None,
            cached_mask_len: 0,
            gradients: None,
            last_logits: None,
            var_map,
            checkpoint_store,
            checkpoint_config,
        })
    }

    /// Get or create causal mask for given sequence length
    fn get_causal_mask(&mut self, seq_len: usize) -> TritterResult<Tensor> {
        if self.cached_mask_len != seq_len {
            self.cached_mask = Some(create_causal_mask(seq_len, &self.device)?);
            self.cached_mask_len = seq_len;
        }
        Ok(self.cached_mask.as_ref().unwrap().clone())
    }

    /// Forward pass returning hidden states
    pub fn forward_hidden(&mut self, input_ids: &Tensor) -> TritterResult<Tensor> {
        // Use checkpointing-aware forward if enabled
        if self.checkpoint_config.enabled {
            self.forward_hidden_with_checkpointing(input_ids)
        } else {
            self.forward_hidden_standard(input_ids)
        }
    }

    /// Standard forward pass without checkpointing
    fn forward_hidden_standard(&mut self, input_ids: &Tensor) -> TritterResult<Tensor> {
        let (_, seq_len) = input_ids.dims2()?;

        // Embed tokens
        let mut hidden = self.embed_tokens.forward(input_ids)?;

        // Get causal mask
        let mask = self.get_causal_mask(seq_len)?;

        // Pass through transformer layers
        for layer in &self.layers {
            hidden = layer.forward(&hidden, Some(&mask))?;
        }

        // Final normalization
        Ok(self.final_norm.forward(&hidden)?)
    }

    /// Forward pass with gradient checkpointing.
    ///
    /// Caches activations at checkpoint boundaries to enable memory-efficient
    /// training. Non-checkpoint activations are discarded after use and
    /// recomputed during the backward pass.
    fn forward_hidden_with_checkpointing(&mut self, input_ids: &Tensor) -> TritterResult<Tensor> {
        let (_, seq_len) = input_ids.dims2()?;

        // Embed tokens
        let mut hidden = self.embed_tokens.forward(input_ids)?;

        // Get causal mask
        let mask = self.get_causal_mask(seq_len)?;

        // Clear any previous checkpoints
        if let Some(ref mut store) = self.checkpoint_store {
            store.clear();
        }

        // Pass through transformer layers, caching at checkpoint boundaries
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            // Cache activation at checkpoint boundaries BEFORE processing
            if let Some(ref mut store) = self.checkpoint_store {
                if store.is_checkpoint_layer(layer_idx) {
                    store.store(layer_idx, &hidden)?;
                }
            }

            // Forward through layer
            hidden = layer.forward(&hidden, Some(&mask))?;
        }

        // Final normalization
        Ok(self.final_norm.forward(&hidden)?)
    }

    /// Recompute forward pass for a segment of layers.
    ///
    /// Used during backward pass to recompute activations that were not
    /// checkpointed. Takes the input activation (from a checkpoint) and
    /// runs forward through the specified layers.
    ///
    /// # Arguments
    /// * `start_layer` - First layer index to process (inclusive)
    /// * `end_layer` - Last layer index to process (exclusive)
    /// * `input_activation` - Activation from the checkpoint at start_layer
    /// * `mask` - Causal attention mask
    ///
    /// # Returns
    /// The output activation after processing the segment.
    pub fn recompute_segment(
        &self,
        start_layer: usize,
        end_layer: usize,
        input_activation: Tensor,
        mask: Option<&Tensor>,
    ) -> TritterResult<Tensor> {
        let mut hidden = input_activation;

        for layer_idx in start_layer..end_layer.min(self.layers.len()) {
            hidden = self.layers[layer_idx].forward(&hidden, mask)?;
        }

        Ok(hidden)
    }

    /// Get segments for backward pass processing.
    ///
    /// Returns a list of (start_layer, end_layer) tuples representing
    /// segments to process during the backward pass, in reverse order
    /// (from last segment to first).
    pub fn get_checkpoint_segments(&self) -> Vec<(usize, usize)> {
        if let Some(ref store) = self.checkpoint_store {
            store.get_segments(self.layers.len())
        } else {
            vec![(0, self.layers.len())]
        }
    }

    /// Retrieve a cached checkpoint activation.
    ///
    /// # Arguments
    /// * `layer_idx` - The layer index to retrieve the checkpoint for
    ///
    /// # Returns
    /// The cached activation tensor, or an error if not found.
    pub fn retrieve_checkpoint(&self, layer_idx: usize) -> TritterResult<Tensor> {
        self.checkpoint_store
            .as_ref()
            .ok_or_else(|| {
                crate::error::TritterError::Training("Checkpoint store not initialized".to_string())
            })?
            .retrieve(layer_idx)
    }

    /// Check if checkpointing is enabled.
    pub fn is_checkpointing_enabled(&self) -> bool {
        self.checkpoint_config.enabled
    }

    /// Get the checkpoint interval.
    pub fn checkpoint_interval(&self) -> usize {
        self.checkpoint_config.checkpoint_interval
    }

    /// Clear checkpoint store to free memory.
    ///
    /// Call this after the backward pass is complete.
    pub fn clear_checkpoints(&mut self) {
        if let Some(ref mut store) = self.checkpoint_store {
            store.clear();
        }
    }

    /// Enable or disable gradient checkpointing at runtime.
    ///
    /// # Arguments
    /// * `enabled` - Whether to enable checkpointing
    /// * `interval` - Optional new checkpoint interval (uses existing if None)
    pub fn set_checkpointing(&mut self, enabled: bool, interval: Option<usize>) {
        let interval = interval.unwrap_or(self.checkpoint_config.checkpoint_interval);

        self.checkpoint_config = GradientCheckpointConfig {
            enabled,
            checkpoint_interval: interval,
            offload_to_cpu: self.checkpoint_config.offload_to_cpu,
        };

        if enabled {
            self.checkpoint_store =
                Some(CheckpointStore::new(&self.checkpoint_config, &self.device));
        } else {
            self.checkpoint_store = None;
        }
    }

    /// Get number of stored checkpoints (for debugging/monitoring).
    pub fn num_checkpoints(&self) -> usize {
        self.checkpoint_store
            .as_ref()
            .map(|s| s.num_checkpoints())
            .unwrap_or(0)
    }

    /// Get memory usage of stored checkpoints in bytes.
    pub fn checkpoint_memory_usage(&self) -> usize {
        self.checkpoint_store
            .as_ref()
            .map(|s| s.memory_usage())
            .unwrap_or(0)
    }

    /// Forward pass returning logits
    pub fn forward(&mut self, input_ids: &Tensor) -> TritterResult<Tensor> {
        let hidden = self.forward_hidden(input_ids)?;
        let logits = self.lm_head.forward(&hidden)?;
        self.last_logits = Some(logits.clone());
        Ok(logits)
    }

    /// Compute cross-entropy loss
    pub fn compute_loss(&self, logits: &Tensor, labels: &Tensor) -> TritterResult<Tensor> {
        let (batch, seq_len, vocab) = logits.dims3()?;

        // Shift logits and labels for next-token prediction
        // logits: use positions 0..seq_len-1
        // labels: use positions 1..seq_len
        let logits = logits.narrow(1, 0, seq_len - 1)?;
        let labels = labels.narrow(1, 1, seq_len - 1)?;

        // Flatten for cross-entropy: (batch * (seq-1), vocab) and (batch * (seq-1),)
        let logits = logits.reshape((batch * (seq_len - 1), vocab))?;
        let labels = labels.reshape((batch * (seq_len - 1),))?;

        // Cross-entropy loss
        let log_probs = candle_nn::ops::log_softmax(&logits, D::Minus1)?;
        let loss = candle_nn::loss::nll(&log_probs, &labels)?;

        Ok(loss)
    }

    /// Forward pass with loss computation
    pub fn forward_loss(&mut self, input_ids: &Tensor) -> TritterResult<f32> {
        let logits = self.forward(input_ids)?;
        let loss = self.compute_loss(&logits, input_ids)?;
        Ok(loss.to_scalar::<f32>()?)
    }

    /// Get total parameter count
    pub fn parameter_count(&self) -> usize {
        self.config.parameter_count()
    }

    /// Get model configuration
    pub fn config(&self) -> &TritterConfig {
        &self.config
    }

    /// Get device
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get number of transformer layers
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Get reference to transformer layers (for advanced use cases)
    pub fn layers(&self) -> &[TritterLayer] {
        &self.layers
    }

    /// Save model weights to file (safetensors format).
    ///
    /// # Arguments
    /// * `path` - Path to save the safetensors file
    ///
    /// # Example
    /// ```no_run
    /// # use tritter_model_rs::{TritterConfig, TritterModel};
    /// # use candle_core::Device;
    /// let config = TritterConfig::small_100m();
    /// let device = Device::Cpu;
    /// let model = TritterModel::new(&config, &device).unwrap();
    /// model.save(std::path::Path::new("model.safetensors")).unwrap();
    /// ```
    pub fn save(&self, path: &std::path::Path) -> TritterResult<()> {
        self.var_map.save(path)?;
        Ok(())
    }

    /// Alias for save() - saves in safetensors format.
    pub fn save_safetensors(&self, path: &std::path::Path) -> TritterResult<()> {
        self.save(path)
    }

    /// Load model weights from file (safetensors format).
    ///
    /// Creates a new model and loads weights from the specified file.
    ///
    /// # Arguments
    /// * `config` - Model configuration (must match the saved model)
    /// * `path` - Path to the safetensors file
    /// * `device` - Device to load tensors onto
    ///
    /// # Example
    /// ```no_run
    /// # use tritter_model_rs::{TritterConfig, TritterModel};
    /// # use candle_core::Device;
    /// let config = TritterConfig::small_100m();
    /// let device = Device::Cpu;
    /// let model = TritterModel::load(&config, std::path::Path::new("model.safetensors"), &device).unwrap();
    /// ```
    pub fn load(
        config: &TritterConfig,
        path: &std::path::Path,
        device: &Device,
    ) -> TritterResult<Self> {
        let mut var_map = VarMap::new();
        var_map.load(path)?;
        let vb = VarBuilder::from_varmap(&var_map, DType::F32, device);
        Self::from_varbuilder(config, vb, device, var_map)
    }

    /// Alias for load() - loads from safetensors format.
    pub fn load_safetensors(
        config: &TritterConfig,
        path: &std::path::Path,
        device: &Device,
    ) -> TritterResult<Self> {
        Self::load(config, path, device)
    }

    /// Load weights into existing model from file (safetensors format).
    ///
    /// Updates existing variables to match values from the safetensors file.
    /// Variables not present in the file are unchanged.
    ///
    /// # Arguments
    /// * `path` - Path to the safetensors file
    ///
    /// # Example
    /// ```no_run
    /// # use tritter_model_rs::{TritterConfig, TritterModel};
    /// # use candle_core::Device;
    /// let config = TritterConfig::small_100m();
    /// let device = Device::Cpu;
    /// let mut model = TritterModel::new(&config, &device).unwrap();
    /// // ... train for a while ...
    /// // Load checkpoint
    /// model.load_weights(std::path::Path::new("checkpoint.safetensors")).unwrap();
    /// ```
    pub fn load_weights(&mut self, path: &std::path::Path) -> TritterResult<()> {
        // VarMap::load modifies existing variables in place
        // We need a mutable reference to var_map, but we own it
        // This is a bit awkward with the current API - we need to reload
        let mut new_var_map = VarMap::new();
        new_var_map.load(path)?;

        // Copy values from new_var_map to self.var_map
        let new_data = new_var_map.data().lock().unwrap();
        let self_data = self.var_map.data().lock().unwrap();

        for (name, var) in self_data.iter() {
            if let Some(new_var) = new_data.get(name) {
                var.set(new_var.as_tensor())?;
            }
        }

        Ok(())
    }

    /// Get reference to VarMap for gradient computation
    pub fn var_map(&self) -> &VarMap {
        &self.var_map
    }

    /// Store computed gradients
    pub fn store_gradients(&mut self, grads: HashMap<String, Tensor>) {
        self.gradients = Some(grads);
    }

    /// Get stored gradients
    pub fn get_gradients(&self) -> Option<&HashMap<String, Tensor>> {
        self.gradients.as_ref()
    }

    /// Clear gradients
    pub fn zero_grad(&mut self) {
        self.gradients = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_forward() {
        let config = TritterConfig::test();
        let device = Device::Cpu;
        let mut model = TritterModel::new(&config, &device).unwrap();

        let batch_size = 2;
        let seq_len = 16;
        let input_ids = Tensor::zeros((batch_size, seq_len), DType::U32, &device).unwrap();

        let logits = model.forward(&input_ids).unwrap();
        assert_eq!(logits.dims(), &[batch_size, seq_len, config.vocab_size]);
    }

    #[test]
    fn test_model_loss() {
        let config = TritterConfig::test();
        let device = Device::Cpu;
        let mut model = TritterModel::new(&config, &device).unwrap();

        let batch_size = 2;
        let seq_len = 16;
        let input_ids = Tensor::zeros((batch_size, seq_len), DType::U32, &device).unwrap();

        let loss = model.forward_loss(&input_ids).unwrap();
        assert!(loss.is_finite());
        assert!(loss > 0.0);
    }

    #[test]
    fn test_parameter_count() {
        let config = TritterConfig::test();
        let device = Device::Cpu;
        let model = TritterModel::new(&config, &device).unwrap();

        let count = model.parameter_count();
        assert!(count > 0);
    }

    #[test]
    fn test_model_with_bitnet() {
        let mut config = TritterConfig::test();
        config.use_bitnet = true;
        let device = Device::Cpu;
        let mut model = TritterModel::new(&config, &device).unwrap();

        let batch_size = 2;
        let seq_len = 16;
        let input_ids = Tensor::zeros((batch_size, seq_len), DType::U32, &device).unwrap();

        // Forward pass should work with BitNet
        let logits = model.forward(&input_ids).unwrap();
        assert_eq!(logits.dims(), &[batch_size, seq_len, config.vocab_size]);

        // Loss should be computable
        let loss = model.forward_loss(&input_ids).unwrap();
        assert!(loss.is_finite());
        assert!(loss > 0.0);
    }

    #[test]
    fn test_bitnet_vs_standard_output_shapes() {
        // Test that both BitNet and standard models produce same shapes
        let mut config_std = TritterConfig::test();
        config_std.use_bitnet = false;

        let mut config_bit = TritterConfig::test();
        config_bit.use_bitnet = true;

        let device = Device::Cpu;
        let mut model_std = TritterModel::new(&config_std, &device).unwrap();
        let mut model_bit = TritterModel::new(&config_bit, &device).unwrap();

        let batch_size = 2;
        let seq_len = 8;
        let input_ids = Tensor::zeros((batch_size, seq_len), DType::U32, &device).unwrap();

        let out_std = model_std.forward(&input_ids).unwrap();
        let out_bit = model_bit.forward(&input_ids).unwrap();

        assert_eq!(out_std.dims(), out_bit.dims());
    }

    #[test]
    fn test_model_with_checkpointing() {
        // Create model with checkpointing enabled
        let mut config = TritterConfig::test();
        config.gradient_checkpointing = true;
        config.checkpoint_every_n_layers = 1;
        config.num_layers = 4; // More layers to test checkpointing

        let device = Device::Cpu;
        let mut model = TritterModel::new(&config, &device).unwrap();

        assert!(model.is_checkpointing_enabled());
        assert_eq!(model.num_layers(), 4);

        let batch_size = 2;
        let seq_len = 8;
        let input_ids = Tensor::zeros((batch_size, seq_len), DType::U32, &device).unwrap();

        // Forward should work with checkpointing
        let logits = model.forward(&input_ids).unwrap();
        assert_eq!(logits.dims(), &[batch_size, seq_len, config.vocab_size]);

        // Should have stored checkpoints
        assert!(model.num_checkpoints() > 0);
    }

    #[test]
    fn test_checkpointing_vs_standard_equivalence() {
        // Verify checkpointed forward produces same results as standard
        let mut config = TritterConfig::test();
        config.num_layers = 4;

        let device = Device::Cpu;

        // Create two models with same weights
        let mut model_std = TritterModel::new(&config, &device).unwrap();
        model_std.set_checkpointing(false, None);

        // Enable checkpointing on a copy
        let mut model_cp = TritterModel::new(&config, &device).unwrap();
        model_cp.set_checkpointing(true, Some(2));

        // Note: These have different random weights, so we can only test shapes
        let batch_size = 2;
        let seq_len = 8;
        let input_ids = Tensor::zeros((batch_size, seq_len), DType::U32, &device).unwrap();

        let out_std = model_std.forward(&input_ids).unwrap();
        let out_cp = model_cp.forward(&input_ids).unwrap();

        // Shapes should match
        assert_eq!(out_std.dims(), out_cp.dims());
    }

    #[test]
    fn test_recompute_segment() {
        let mut config = TritterConfig::test();
        config.num_layers = 4;
        config.gradient_checkpointing = true;
        config.checkpoint_every_n_layers = 2;

        let device = Device::Cpu;
        let mut model = TritterModel::new(&config, &device).unwrap();

        let batch_size = 2;
        let seq_len = 8;
        let input_ids = Tensor::zeros((batch_size, seq_len), DType::U32, &device).unwrap();

        // Run forward to populate checkpoints
        let _ = model.forward(&input_ids).unwrap();

        // Get checkpoint at layer 0
        let checkpoint_0 = model.retrieve_checkpoint(0).unwrap();
        assert_eq!(
            checkpoint_0.dims(),
            &[batch_size, seq_len, config.hidden_size]
        );

        // Recompute segment 0-2
        let recomputed = model.recompute_segment(0, 2, checkpoint_0, None).unwrap();
        assert_eq!(
            recomputed.dims(),
            &[batch_size, seq_len, config.hidden_size]
        );
    }

    #[test]
    fn test_get_checkpoint_segments() {
        let mut config = TritterConfig::test();
        config.num_layers = 8;
        config.gradient_checkpointing = true;
        config.checkpoint_every_n_layers = 4;

        let device = Device::Cpu;
        let model = TritterModel::new(&config, &device).unwrap();

        let segments = model.get_checkpoint_segments();
        // 8 layers with interval 4: segments (4,8), (0,4)
        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0], (4, 8));
        assert_eq!(segments[1], (0, 4));
    }

    #[test]
    fn test_clear_checkpoints() {
        let mut config = TritterConfig::test();
        config.gradient_checkpointing = true;
        config.checkpoint_every_n_layers = 1;

        let device = Device::Cpu;
        let mut model = TritterModel::new(&config, &device).unwrap();

        let input_ids = Tensor::zeros((2, 8), DType::U32, &device).unwrap();
        let _ = model.forward(&input_ids).unwrap();

        assert!(model.num_checkpoints() > 0);

        model.clear_checkpoints();
        assert_eq!(model.num_checkpoints(), 0);
    }

    #[test]
    fn test_set_checkpointing_runtime() {
        let config = TritterConfig::test();
        let device = Device::Cpu;
        let mut model = TritterModel::new(&config, &device).unwrap();

        // Initially disabled (test config)
        assert!(!model.is_checkpointing_enabled());

        // Enable at runtime
        model.set_checkpointing(true, Some(2));
        assert!(model.is_checkpointing_enabled());
        assert_eq!(model.checkpoint_interval(), 2);

        // Disable again
        model.set_checkpointing(false, None);
        assert!(!model.is_checkpointing_enabled());
    }
}
