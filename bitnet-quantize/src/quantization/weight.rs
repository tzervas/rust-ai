//! Weight quantization for BitNet.
//!
//! Implements AbsMean quantization: `W_q = round(W / mean(|W|))` clamped to {-1, 0, +1}.

use candle_core::{Device, Tensor};
use serde::{Deserialize, Serialize};
use trit_vsa::PackedTritVec;

use crate::config::BitNetConfig;
use crate::error::{BitNetError, Result};

/// Ternary weight representation with per-group scales.
///
/// Weights are quantized to {-1, 0, +1} using AbsMean quantization,
/// with a scale factor stored per group.
#[derive(Clone, Serialize, Deserialize)]
pub struct TernaryWeight {
    /// Packed ternary values (bitsliced storage).
    /// Shape: [out_features, in_features] flattened.
    pub data: Vec<PackedTritVec>,

    /// Scale factors per group.
    /// For a weight matrix [out, in], scales has shape [out, in/group_size].
    pub scales: Vec<f32>,

    /// Original shape [out_features, in_features].
    pub shape: (usize, usize),

    /// Group size used for quantization.
    pub group_size: usize,
}

impl std::fmt::Debug for TernaryWeight {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TernaryWeight")
            .field("shape", &self.shape)
            .field("group_size", &self.group_size)
            .field("num_scales", &self.scales.len())
            .field("sparsity", &self.sparsity())
            .finish_non_exhaustive()
    }
}

impl TernaryWeight {
    /// Get the output features dimension.
    #[must_use]
    pub const fn out_features(&self) -> usize {
        self.shape.0
    }

    /// Get the input features dimension.
    #[must_use]
    pub const fn in_features(&self) -> usize {
        self.shape.1
    }

    /// Calculate the sparsity (fraction of zeros).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn sparsity(&self) -> f32 {
        let total_nonzero: usize = self.data.iter().map(PackedTritVec::count_nonzero).sum();
        let total_elements = self.shape.0 * self.shape.1;
        1.0 - (total_nonzero as f32 / total_elements as f32)
    }

    /// Memory size in bytes.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        // Packed trits: 2 bits per trit, so num_words * 4 * 2 per row
        let trit_bytes: usize = self.data.iter().map(|v| v.num_words() * 8).sum();
        // Scales: f32 per group
        let scale_bytes = self.scales.len() * 4;
        trit_bytes + scale_bytes
    }

    /// Compression ratio vs FP32.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn compression_ratio(&self) -> f32 {
        let fp32_bytes = self.shape.0 * self.shape.1 * 4;
        fp32_bytes as f32 / self.memory_bytes() as f32
    }
}

/// Quantize a weight tensor to ternary using AbsMean quantization.
///
/// # Algorithm
///
/// For each group of weights:
/// 1. Compute `scale = mean(|W|)`
/// 2. Compute `W_q = round(W / scale)` clamped to {-1, 0, +1}
///
/// # Arguments
///
/// * `weight` - Input weight tensor [out_features, in_features]
/// * `config` - BitNet configuration
///
/// # Errors
///
/// Returns error if weight has wrong shape or quantization fails.
pub fn quantize_weights(weight: &Tensor, config: &BitNetConfig) -> Result<TernaryWeight> {
    let shape = weight.shape().dims();
    if shape.len() != 2 {
        return Err(BitNetError::InvalidConfig(
            "weight must be 2D [out_features, in_features]".to_string(),
        ));
    }

    let out_features = shape[0];
    let in_features = shape[1];
    let group_size = config.group_size;

    // Ensure in_features is divisible by group_size
    if !in_features.is_multiple_of(group_size) {
        return Err(BitNetError::InvalidConfig(format!(
            "in_features ({in_features}) must be divisible by group_size ({group_size})"
        )));
    }

    let num_groups_per_row = in_features / group_size;
    let mut scales = Vec::with_capacity(out_features * num_groups_per_row);
    let mut data = Vec::with_capacity(out_features);

    // Convert to f32 for processing
    let weight_f32 = weight.to_dtype(candle_core::DType::F32)?.to_vec2::<f32>()?;

    for row in &weight_f32 {
        let mut packed = PackedTritVec::new(in_features);

        for g in 0..num_groups_per_row {
            let start = g * group_size;
            let end = start + group_size;
            let group = &row[start..end];

            // Compute AbsMean scale
            let abs_mean: f32 = group.iter().map(|x| x.abs()).sum::<f32>() / group_size as f32;
            let scale = if abs_mean > 0.0 { abs_mean } else { 1.0 };
            scales.push(scale);

            // Quantize each value in the group
            for (i, &val) in group.iter().enumerate() {
                let normalized = val / scale;
                let quantized = normalized.round().clamp(-1.0, 1.0) as i8;
                let trit = trit_vsa::Trit::from_value(quantized as i32)?;
                packed.set(start + i, trit);
            }
        }

        data.push(packed);
    }

    Ok(TernaryWeight {
        data,
        scales,
        shape: (out_features, in_features),
        group_size,
    })
}

/// Dequantize ternary weights back to float tensor.
///
/// # Arguments
///
/// * `ternary` - Ternary weight to dequantize
/// * `device` - Device to create output tensor on
///
/// # Errors
///
/// Returns error if tensor creation fails.
pub fn dequantize_weights(ternary: &TernaryWeight, device: &Device) -> Result<Tensor> {
    let out_features = ternary.out_features();
    let in_features = ternary.in_features();
    let group_size = ternary.group_size;
    let num_groups_per_row = in_features / group_size;

    let mut output = vec![0.0f32; out_features * in_features];

    for (row_idx, packed) in ternary.data.iter().enumerate() {
        let row_start = row_idx * in_features;

        for g in 0..num_groups_per_row {
            let scale_idx = row_idx * num_groups_per_row + g;
            let scale = ternary.scales[scale_idx];
            let group_start = g * group_size;

            for i in 0..group_size {
                let trit = packed.get(group_start + i);
                let value = trit.value() as f32 * scale;
                output[row_start + group_start + i] = value;
            }
        }
    }

    let tensor = Tensor::from_vec(output, (out_features, in_features), device)?;
    Ok(tensor)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;

    #[test]
    fn test_quantize_dequantize_roundtrip() {
        let device = Device::Cpu;
        let config = BitNetConfig::default();

        // Create a weight tensor
        let weight = Tensor::randn(0.0f32, 1.0, (64, 128), &device).unwrap();

        // Quantize
        let ternary = quantize_weights(&weight, &config).unwrap();

        // Check structure
        assert_eq!(ternary.shape, (64, 128));
        assert_eq!(ternary.data.len(), 64);
        assert_eq!(ternary.scales.len(), 64 * (128 / 64)); // 64 rows * 2 groups

        // Dequantize
        let restored = dequantize_weights(&ternary, &device).unwrap();
        assert_eq!(restored.shape().dims(), &[64, 128]);
    }

    #[test]
    fn test_quantize_preserves_sign() {
        let device = Device::Cpu;
        let config = BitNetConfig::default().with_group_size(4);

        // Create a simple weight with known values
        let values: Vec<f32> = vec![1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 0.1, -0.1];
        let weight = Tensor::from_vec(values, (2, 4), &device).unwrap();

        let ternary = quantize_weights(&weight, &config).unwrap();

        // Check that signs are preserved
        // Row 0: [1, -1, 0.5, -0.5] -> scale = (1+1+0.5+0.5)/4 = 0.75
        // Normalized: [1.33, -1.33, 0.67, -0.67] -> [+1, -1, +1, -1]
        assert_eq!(ternary.data[0].get(0), trit_vsa::Trit::P);
        assert_eq!(ternary.data[0].get(1), trit_vsa::Trit::N);
    }

    #[test]
    fn test_sparsity() {
        let device = Device::Cpu;
        let config = BitNetConfig::default().with_group_size(4);

        // Create a sparse weight (many zeros)
        let values: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0];
        let weight = Tensor::from_vec(values, (2, 4), &device).unwrap();

        let ternary = quantize_weights(&weight, &config).unwrap();

        // Should have high sparsity
        let sparsity = ternary.sparsity();
        assert!(sparsity > 0.5, "expected high sparsity, got {sparsity}");
    }

    #[test]
    fn test_compression_ratio() {
        let device = Device::Cpu;
        let config = BitNetConfig::default();

        let weight = Tensor::randn(0.0f32, 1.0, (1024, 4096), &device).unwrap();
        let ternary = quantize_weights(&weight, &config).unwrap();

        let ratio = ternary.compression_ratio();
        // Should achieve significant compression (typically 8-16x)
        assert!(ratio > 4.0, "expected >4x compression, got {ratio:.2}x");
    }

    #[test]
    fn test_invalid_shape() {
        let device = Device::Cpu;
        let config = BitNetConfig::default();

        // 1D tensor should fail
        let weight = Tensor::zeros(&[64], DType::F32, &device).unwrap();
        assert!(quantize_weights(&weight, &config).is_err());

        // 3D tensor should fail
        let weight = Tensor::zeros(&[2, 64, 64], DType::F32, &device).unwrap();
        assert!(quantize_weights(&weight, &config).is_err());
    }

    #[test]
    fn test_indivisible_group_size() {
        let device = Device::Cpu;
        let config = BitNetConfig::default().with_group_size(64);

        // in_features=100 is not divisible by 64
        let weight = Tensor::zeros(&[32, 100], DType::F32, &device).unwrap();
        assert!(quantize_weights(&weight, &config).is_err());
    }
}
