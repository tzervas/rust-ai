//! Main Tritter transformer model.
//!
//! Implements the full model with:
//! - Token embeddings
//! - Transformer layers
//! - LM head for next-token prediction
//! - BitNet quantization via bitnet-quantize crate

use candle_core::{DType, Device, Tensor, D};
use candle_nn::{embedding, layer_norm, Embedding, LayerNorm, Module, VarBuilder, VarMap};
use std::collections::HashMap;

use crate::attention::create_causal_mask;
use crate::config::TritterConfig;
use crate::error::TritterResult;
use crate::layer::TritterLayer;

/// The main Tritter transformer model
pub struct TritterModel {
    embed_tokens: Embedding,
    layers: Vec<TritterLayer>,
    final_norm: LayerNorm,
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

        // Final layer norm
        let final_norm = layer_norm(config.hidden_size, config.layer_norm_eps, vb.pp("final_norm"))?;

        // LM head (tied to embeddings in some models, separate here)
        let lm_head = candle_nn::linear_no_bias(
            config.hidden_size,
            config.vocab_size,
            vb.pp("lm_head"),
        )?;

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

    /// Save model weights to file
    pub fn save(&self, path: &std::path::Path) -> TritterResult<()> {
        self.var_map.save(path)?;
        Ok(())
    }

    /// Load model weights from file
    pub fn load(config: &TritterConfig, path: &std::path::Path, device: &Device) -> TritterResult<Self> {
        let mut var_map = VarMap::new();
        var_map.load(path)?;
        let vb = VarBuilder::from_varmap(&var_map, DType::F32, device);
        Self::from_varbuilder(config, vb, device, var_map)
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
}
