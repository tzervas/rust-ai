//! BitNet integration for Tritter model.
//!
//! Provides a unified linear layer interface that can use either:
//! - Standard `candle_nn::Linear` for full-precision weights
//! - `bitnet_quantize::BitLinear` for ternary quantized weights
//!
//! The choice is controlled by `TritterConfig::use_bitnet`.

use candle_core::{Device, Result, Tensor};
use candle_nn::{linear_no_bias, Module, VarBuilder};

use bitnet_quantize::{BitLinear, BitNetConfig};

/// A linear layer that can be either standard or BitNet quantized.
///
/// When `use_bitnet` is true, weights are quantized to ternary {-1, 0, +1}
/// using the AbsMean method from BitNet b1.58.
pub enum TritterLinear {
    /// Standard full-precision linear layer
    Standard(candle_nn::Linear),
    /// BitNet ternary quantized linear layer
    BitNet(BitLinear),
}

impl TritterLinear {
    /// Create a new linear layer, optionally with BitNet quantization.
    ///
    /// # Arguments
    ///
    /// * `in_features` - Input dimension
    /// * `out_features` - Output dimension
    /// * `use_bitnet` - Whether to use BitNet quantization
    /// * `vb` - VarBuilder for weight initialization
    /// * `device` - Device for operations
    ///
    /// # Errors
    ///
    /// Returns error if layer creation fails.
    pub fn new(
        in_features: usize,
        out_features: usize,
        use_bitnet: bool,
        vb: VarBuilder,
        _device: &Device,
    ) -> Result<Self> {
        if use_bitnet {
            // Create initial weights using VarBuilder
            let weight = vb.get((out_features, in_features), "weight")?;

            // Configure BitNet with reasonable defaults for transformer layers
            let config = BitNetConfig::new()
                .with_group_size(64)
                .with_ste(true)
                .with_per_token_activation(true);

            // Quantize to BitLinear
            let bit_linear = BitLinear::from_weight(&weight, None, &config)
                .map_err(|e| candle_core::Error::Msg(format!("BitNet quantization failed: {}", e)))?;

            Ok(Self::BitNet(bit_linear))
        } else {
            // Standard linear layer
            let linear = linear_no_bias(in_features, out_features, vb)?;
            Ok(Self::Standard(linear))
        }
    }

    /// Create from an existing weight tensor.
    ///
    /// This is useful for loading pretrained weights and optionally quantizing them.
    pub fn from_weight(
        weight: &Tensor,
        use_bitnet: bool,
        _device: &Device,
    ) -> Result<Self> {
        if use_bitnet {
            let config = BitNetConfig::new()
                .with_group_size(64)
                .with_ste(true)
                .with_per_token_activation(true);

            let bit_linear = BitLinear::from_weight(weight, None, &config)
                .map_err(|e| candle_core::Error::Msg(format!("BitNet quantization failed: {}", e)))?;

            Ok(Self::BitNet(bit_linear))
        } else {
            // Create standard linear from weight
            // Note: candle_nn::Linear expects weight tensor directly
            Ok(Self::Standard(candle_nn::Linear::new(weight.clone(), None)))
        }
    }

    /// Get input features dimension.
    #[must_use]
    pub fn in_features(&self) -> usize {
        match self {
            Self::Standard(l) => l.weight().dim(1).unwrap_or(0),
            Self::BitNet(l) => l.in_features(),
        }
    }

    /// Get output features dimension.
    #[must_use]
    pub fn out_features(&self) -> usize {
        match self {
            Self::Standard(l) => l.weight().dim(0).unwrap_or(0),
            Self::BitNet(l) => l.out_features(),
        }
    }

    /// Check if this layer is using BitNet quantization.
    #[must_use]
    pub const fn is_bitnet(&self) -> bool {
        matches!(self, Self::BitNet(_))
    }

    /// Get compression ratio (1.0 for standard, >1.0 for BitNet).
    #[must_use]
    pub fn compression_ratio(&self) -> f32 {
        match self {
            Self::Standard(_) => 1.0,
            Self::BitNet(l) => l.compression_ratio(),
        }
    }

    /// Get weight sparsity (0.0 for standard, varies for BitNet).
    #[must_use]
    pub fn sparsity(&self) -> f32 {
        match self {
            Self::Standard(_) => 0.0,
            Self::BitNet(l) => l.sparsity(),
        }
    }
}

impl Module for TritterLinear {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        match self {
            Self::Standard(linear) => linear.forward(input),
            Self::BitNet(bit_linear) => bit_linear.forward(input),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarMap;

    #[test]
    fn test_standard_linear() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let layer = TritterLinear::new(128, 64, false, vb, &device).unwrap();
        assert!(!layer.is_bitnet());
        assert_eq!(layer.in_features(), 128);
        assert_eq!(layer.out_features(), 64);
        assert_eq!(layer.compression_ratio(), 1.0);

        let input = Tensor::randn(0.0f32, 1.0, (2, 128), &device).unwrap();
        let output = layer.forward(&input).unwrap();
        assert_eq!(output.dims(), &[2, 64]);
    }

    #[test]
    fn test_bitnet_linear() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let layer = TritterLinear::new(128, 64, true, vb, &device).unwrap();
        assert!(layer.is_bitnet());
        assert_eq!(layer.in_features(), 128);
        assert_eq!(layer.out_features(), 64);
        assert!(layer.compression_ratio() > 1.0);

        let input = Tensor::randn(0.0f32, 1.0, (2, 128), &device).unwrap();
        let output = layer.forward(&input).unwrap();
        assert_eq!(output.dims(), &[2, 64]);
    }

    #[test]
    fn test_bitnet_3d_input() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let layer = TritterLinear::new(128, 64, true, vb, &device).unwrap();

        // 3D input [batch, seq_len, hidden]
        let input = Tensor::randn(0.0f32, 1.0, (2, 16, 128), &device).unwrap();
        let output = layer.forward(&input).unwrap();
        assert_eq!(output.dims(), &[2, 16, 64]);
    }

    #[test]
    fn test_from_weight() {
        let device = Device::Cpu;

        let weight = Tensor::randn(0.0f32, 1.0, (64, 128), &device).unwrap();

        // Standard
        let layer_std = TritterLinear::from_weight(&weight, false, &device).unwrap();
        assert!(!layer_std.is_bitnet());

        // BitNet
        let layer_bit = TritterLinear::from_weight(&weight, true, &device).unwrap();
        assert!(layer_bit.is_bitnet());

        // Both should produce same shape output
        let input = Tensor::randn(0.0f32, 1.0, (2, 128), &device).unwrap();
        let out_std = layer_std.forward(&input).unwrap();
        let out_bit = layer_bit.forward(&input).unwrap();
        assert_eq!(out_std.dims(), out_bit.dims());
    }
}
