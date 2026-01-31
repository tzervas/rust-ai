//! Transformer layer combining attention and MLP.
//!
//! Implements:
//! - Pre-normalization before attention
//! - Post-normalization after MLP residual (Chameleon pattern)
//! - Residual connections

use candle_core::{Result, Tensor};
use candle_nn::{Module, VarBuilder};

use crate::attention::TritterAttention;
use crate::config::TritterConfig;
use crate::mlp::TritterMLP;
use crate::norm::{manual_layer_norm, ManualLayerNorm};
use candle_core::Device;

/// Single transformer layer
pub struct TritterLayer {
    attention: TritterAttention,
    mlp: TritterMLP,
    input_norm: ManualLayerNorm,
    post_attention_norm: ManualLayerNorm,
}

impl TritterLayer {
    /// Create transformer layer
    pub fn new(config: &TritterConfig, vb: VarBuilder, device: &Device) -> Result<Self> {
        let hidden = config.hidden_size;
        let eps = config.layer_norm_eps;

        let attention = TritterAttention::new(config, vb.pp("attention"), device)?;
        let mlp = TritterMLP::new(config, vb.pp("mlp"), device)?;

        // Pre-attention normalization (using manual impl for CUDA compatibility)
        let input_norm = manual_layer_norm(hidden, eps, vb.pp("input_norm"))?;
        // Post-MLP normalization (Chameleon style)
        let post_attention_norm = manual_layer_norm(hidden, eps, vb.pp("post_attention_norm"))?;

        Ok(Self {
            attention,
            mlp,
            input_norm,
            post_attention_norm,
        })
    }

    /// Forward pass
    /// Input: (batch, seq_len, hidden_size)
    /// Output: (batch, seq_len, hidden_size)
    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        // Pre-norm + attention + residual
        let normed = self.input_norm.forward(x)?;
        let attn_out = self.attention.forward(&normed, mask)?;
        let x = (x + attn_out)?;

        // Post-norm + MLP + residual (Chameleon asymmetric pattern)
        let normed = self.post_attention_norm.forward(&x)?;
        let mlp_out = self.mlp.forward(&normed)?;
        &x + mlp_out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use candle_nn::VarMap;

    #[test]
    fn test_layer_shape() {
        let config = crate::config::TritterConfig::test();
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);

        let layer = TritterLayer::new(&config, vb, &device).unwrap();
        let x = Tensor::randn(0.0f32, 1.0, (2, 8, config.hidden_size), &device).unwrap();
        let out = layer.forward(&x, None).unwrap();

        assert_eq!(out.dims(), &[2, 8, config.hidden_size]);
    }
}
