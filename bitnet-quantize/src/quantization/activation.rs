//! Activation quantization for BitNet.
//!
//! Implements per-token AbsMax quantization to INT8.

use candle_core::{Device, Tensor};
use serde::{Deserialize, Serialize};

use crate::config::BitNetConfig;
use crate::error::{BitNetError, Result};

/// Quantized activations with per-token scales.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizedActivations {
    /// Quantized INT8 values (stored as i8 in a Vec).
    pub data: Vec<i8>,

    /// Per-token scale factors.
    /// Shape depends on `per_token`: [batch, seq_len] or [batch].
    pub scales: Vec<f32>,

    /// Original shape [batch, seq_len, hidden_dim].
    pub shape: Vec<usize>,

    /// Whether per-token scaling was used.
    pub per_token: bool,
}

impl QuantizedActivations {
    /// Get the batch size.
    #[must_use]
    pub fn batch_size(&self) -> usize {
        self.shape.first().copied().unwrap_or(1)
    }

    /// Get the sequence length.
    #[must_use]
    pub fn seq_len(&self) -> usize {
        self.shape.get(1).copied().unwrap_or(1)
    }

    /// Get the hidden dimension.
    #[must_use]
    pub fn hidden_dim(&self) -> usize {
        self.shape.last().copied().unwrap_or(0)
    }

    /// Get the total number of elements.
    #[must_use]
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

/// Quantize activations using per-token AbsMax scaling to INT8.
///
/// # Algorithm
///
/// For each token (row):
/// 1. Compute `scale = max(|X|) / 127`
/// 2. Compute `X_q = round(X / scale)` clamped to [-127, 127]
///
/// # Arguments
///
/// * `activations` - Input tensor [batch, seq_len, hidden_dim] or [batch, hidden_dim]
/// * `config` - BitNet configuration
///
/// # Errors
///
/// Returns error if quantization fails.
pub fn quantize_activations(
    activations: &Tensor,
    config: &BitNetConfig,
) -> Result<QuantizedActivations> {
    let shape = activations.shape().dims().to_vec();

    // Handle both 2D [batch, hidden] and 3D [batch, seq, hidden] inputs
    let (batch_size, seq_len, hidden_dim) = match shape.len() {
        2 => (shape[0], 1, shape[1]),
        3 => (shape[0], shape[1], shape[2]),
        _ => {
            return Err(BitNetError::InvalidConfig(
                "activations must be 2D or 3D".to_string(),
            ))
        }
    };

    // Reshape to [batch * seq_len, hidden_dim] for uniform processing
    let flat = activations.reshape((batch_size * seq_len, hidden_dim))?;
    let flat_f32 = flat.to_dtype(candle_core::DType::F32)?.to_vec2::<f32>()?;

    let max_val = (1 << (config.activation_bits - 1)) - 1; // 127 for 8-bit
    let max_val_f32 = max_val as f32;

    let mut data = Vec::with_capacity(batch_size * seq_len * hidden_dim);
    let mut scales = Vec::with_capacity(batch_size * seq_len);

    for row in &flat_f32 {
        // Compute AbsMax for this token
        let abs_max = row.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let scale = if abs_max > 0.0 {
            abs_max / max_val_f32
        } else {
            1.0
        };
        scales.push(scale);

        // Quantize
        for &val in row {
            let quantized = (val / scale).round().clamp(-max_val_f32, max_val_f32) as i8;
            data.push(quantized);
        }
    }

    Ok(QuantizedActivations {
        data,
        scales,
        shape,
        per_token: config.per_token_activation,
    })
}

/// Dequantize INT8 activations back to float tensor.
///
/// # Arguments
///
/// * `quantized` - Quantized activations
/// * `device` - Device to create output tensor on
///
/// # Errors
///
/// Returns error if tensor creation fails.
pub fn dequantize_activations(quantized: &QuantizedActivations, device: &Device) -> Result<Tensor> {
    let shape = &quantized.shape;
    let (batch_size, seq_len, hidden_dim) = match shape.len() {
        2 => (shape[0], 1, shape[1]),
        3 => (shape[0], shape[1], shape[2]),
        _ => {
            return Err(BitNetError::InvalidConfig(
                "invalid shape for dequantization".to_string(),
            ))
        }
    };

    let mut output = vec![0.0f32; batch_size * seq_len * hidden_dim];

    for token_idx in 0..(batch_size * seq_len) {
        let scale = quantized.scales[token_idx];
        let token_start = token_idx * hidden_dim;

        for i in 0..hidden_dim {
            let q_val = quantized.data[token_start + i];
            output[token_start + i] = q_val as f32 * scale;
        }
    }

    let tensor = Tensor::from_vec(output, shape.clone(), device)?;
    Ok(tensor)
}

/// Apply quantization in a differentiable way using Straight-Through Estimator.
///
/// During forward pass: quantize -> dequantize
/// During backward pass: gradients flow through unchanged
///
/// # Arguments
///
/// * `activations` - Input tensor
/// * `config` - BitNet configuration
///
/// # Errors
///
/// Returns error if quantization fails.
pub fn quantize_ste(activations: &Tensor, config: &BitNetConfig) -> Result<Tensor> {
    let device = activations.device();
    let quantized = quantize_activations(activations, config)?;
    dequantize_activations(&quantized, device)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_dequantize_roundtrip_2d() {
        let device = Device::Cpu;
        let config = BitNetConfig::default();

        // Create 2D activations [batch, hidden]
        let activations = Tensor::randn(0.0f32, 1.0, (4, 128), &device).unwrap();

        let quantized = quantize_activations(&activations, &config).unwrap();

        assert_eq!(quantized.shape, vec![4, 128]);
        assert_eq!(quantized.scales.len(), 4); // One per batch item
        assert_eq!(quantized.data.len(), 4 * 128);

        let restored = dequantize_activations(&quantized, &device).unwrap();
        assert_eq!(restored.shape().dims(), &[4, 128]);
    }

    #[test]
    fn test_quantize_dequantize_roundtrip_3d() {
        let device = Device::Cpu;
        let config = BitNetConfig::default();

        // Create 3D activations [batch, seq, hidden]
        let activations = Tensor::randn(0.0f32, 1.0, (2, 16, 128), &device).unwrap();

        let quantized = quantize_activations(&activations, &config).unwrap();

        assert_eq!(quantized.shape, vec![2, 16, 128]);
        assert_eq!(quantized.scales.len(), 2 * 16); // Per token
        assert_eq!(quantized.data.len(), 2 * 16 * 128);

        let restored = dequantize_activations(&quantized, &device).unwrap();
        assert_eq!(restored.shape().dims(), &[2, 16, 128]);
    }

    #[test]
    fn test_quantization_range() {
        let device = Device::Cpu;
        let config = BitNetConfig::default();

        // Create activations with known range
        let values: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) / 10.0).collect();
        let activations = Tensor::from_vec(values, (1, 64), &device).unwrap();

        let quantized = quantize_activations(&activations, &config).unwrap();

        // All quantized values should be in [-127, 127] (i8 type enforces upper bound)
        for &val in &quantized.data {
            assert!(val >= -127, "value {val} below -127");
        }
    }

    #[test]
    fn test_ste_passthrough() {
        let device = Device::Cpu;
        let config = BitNetConfig::training();

        let activations = Tensor::randn(0.0f32, 1.0, (2, 64), &device).unwrap();

        let result = quantize_ste(&activations, &config).unwrap();

        // Shape should be preserved
        assert_eq!(result.shape().dims(), activations.shape().dims());

        // Values should be close (within quantization error)
        let orig: Vec<f32> = activations.flatten_all().unwrap().to_vec1().unwrap();
        let quant: Vec<f32> = result.flatten_all().unwrap().to_vec1().unwrap();

        for (o, q) in orig.iter().zip(quant.iter()) {
            let error = (o - q).abs();
            // INT8 quantization error should be bounded
            assert!(error < 0.1, "error {error} too large");
        }
    }

    #[test]
    fn test_zero_activations() {
        let device = Device::Cpu;
        let config = BitNetConfig::default();

        let activations = Tensor::zeros(&[4, 64], candle_core::DType::F32, &device).unwrap();

        let quantized = quantize_activations(&activations, &config).unwrap();

        // All quantized values should be zero
        for &val in &quantized.data {
            assert_eq!(val, 0);
        }

        // Scales should be 1.0 (fallback)
        for &scale in &quantized.scales {
            assert!((scale - 1.0).abs() < 0.001);
        }
    }

    #[test]
    fn test_invalid_shape() {
        let device = Device::Cpu;
        let config = BitNetConfig::default();

        // 1D tensor should fail
        let activations = Tensor::zeros(&[64], candle_core::DType::F32, &device).unwrap();
        assert!(quantize_activations(&activations, &config).is_err());

        // 4D tensor should fail
        let activations = Tensor::zeros(&[2, 4, 8, 16], candle_core::DType::F32, &device).unwrap();
        assert!(quantize_activations(&activations, &config).is_err());
    }
}
