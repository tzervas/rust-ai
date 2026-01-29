//! Quantization operations for weight and activation compression.
//!
//! Wraps `bitnet-quantize` for ternary quantization with various scaling methods.
//!
//! # Quantization Methods
//!
//! - **AbsMean**: Scale = mean(|W|), round to {-1, 0, +1}
//! - **AbsMax**: Scale = max(|W|), more aggressive outlier handling
//!
//! # Example
//!
//! ```rust,ignore
//! use tritter_accel::core::quantization::{quantize_absmean, QuantizeConfig};
//! use candle_core::{Device, Tensor};
//!
//! let weights = Tensor::randn(0f32, 1f32, (512, 512), &Device::Cpu)?;
//! let config = QuantizeConfig::default();
//! let result = quantize_absmean(&weights, &config)?;
//! println!("Quantized to {} groups", result.scales.len());
//! ```

use bitnet_quantize::{quantize_weights, BitNetConfig, TernaryWeight};
use candle_core::{Device, Tensor};
use thiserror::Error;

use super::ternary::PackedTernary;

/// Errors from quantization operations.
#[derive(Debug, Error)]
pub enum QuantizationError {
    /// Invalid configuration.
    #[error("invalid config: {0}")]
    InvalidConfig(String),

    /// Tensor operation failed.
    #[error("tensor error: {0}")]
    Tensor(#[from] candle_core::Error),

    /// BitNet quantization failed.
    #[error("quantization failed: {0}")]
    Quantize(#[from] bitnet_quantize::BitNetError),
}

/// Configuration for quantization operations.
#[derive(Debug, Clone)]
pub struct QuantizeConfig {
    /// Group size for block-wise quantization.
    /// Smaller groups = more scales = higher accuracy.
    /// 0 = per-tensor, otherwise per-group.
    pub group_size: usize,

    /// Whether to use symmetric quantization.
    pub symmetric: bool,
}

impl Default for QuantizeConfig {
    fn default() -> Self {
        Self {
            group_size: 0, // Per-row by default
            symmetric: true,
        }
    }
}

impl QuantizeConfig {
    /// Set group size.
    pub fn with_group_size(mut self, size: usize) -> Self {
        self.group_size = size;
        self
    }
}

/// Result of quantization operation.
#[derive(Debug, Clone)]
pub struct QuantizationResult {
    /// Quantized ternary values as i8 (-1, 0, +1).
    pub values: Vec<i8>,
    /// Scale factors (one per group or per row).
    pub scales: Vec<f32>,
    /// Original tensor shape.
    pub shape: (usize, usize),
    /// Group size used.
    pub group_size: usize,
}

impl QuantizationResult {
    /// Convert to PackedTernary for efficient storage and matmul.
    pub fn to_packed(&self) -> Result<PackedTernary, super::ternary::TernaryError> {
        // For per-row quantization, scales align with rows
        // For group quantization, we need to expand scales to per-row
        let (rows, cols) = self.shape;

        if self.group_size == 0 || self.group_size >= cols {
            // Per-row quantization
            PackedTernary::from_i8(&self.values, &self.scales, self.shape)
        } else {
            // Group quantization - we need one scale per row for PackedTernary
            // Average the group scales for each row
            let groups_per_row = cols.div_ceil(self.group_size);
            let mut row_scales = Vec::with_capacity(rows);

            for row in 0..rows {
                let start = row * groups_per_row;
                let end = (start + groups_per_row).min(self.scales.len());
                let avg: f32 = self.scales[start..end].iter().sum::<f32>() / (end - start) as f32;
                row_scales.push(avg);
            }

            PackedTernary::from_i8(&self.values, &row_scales, self.shape)
        }
    }

    /// Get values as f32 tensor with scales applied.
    pub fn to_tensor(&self, device: &Device) -> Result<Tensor, QuantizationError> {
        let (rows, cols) = self.shape;
        let mut output = vec![0.0f32; rows * cols];

        if self.group_size == 0 || self.group_size >= cols {
            // Per-row scaling
            for row in 0..rows {
                let scale = self.scales[row];
                for col in 0..cols {
                    let idx = row * cols + col;
                    output[idx] = f32::from(self.values[idx]) * scale;
                }
            }
        } else {
            // Per-group scaling
            let groups_per_row = cols.div_ceil(self.group_size);
            for row in 0..rows {
                for col in 0..cols {
                    let group = col / self.group_size;
                    let scale_idx = row * groups_per_row + group;
                    let idx = row * cols + col;
                    output[idx] = f32::from(self.values[idx]) * self.scales[scale_idx];
                }
            }
        }

        Ok(Tensor::from_vec(output, (rows, cols), device)?)
    }
}

/// Quantize weights using AbsMean scaling (BitNet b1.58 method).
///
/// For each group: scale = mean(|W|), then round W/scale to {-1, 0, +1}.
///
/// # Arguments
///
/// * `weights` - 2D weight tensor
/// * `config` - Quantization configuration
///
/// # Returns
///
/// Quantized weights with scales.
pub fn quantize_absmean(
    weights: &Tensor,
    config: &QuantizeConfig,
) -> Result<QuantizationResult, QuantizationError> {
    let (rows, cols) = weights.dims2()?;

    // Use row-wise quantization if group_size is 0
    let effective_group_size = if config.group_size == 0 {
        cols
    } else {
        config.group_size
    };

    let bitnet_config = BitNetConfig::default().with_group_size(effective_group_size);

    let ternary: TernaryWeight = quantize_weights(weights, &bitnet_config)?;

    // Extract values and scales
    let mut values = Vec::with_capacity(rows * cols);
    for packed in &ternary.data {
        for col in 0..cols {
            values.push(packed.get(col).value());
        }
    }

    Ok(QuantizationResult {
        values,
        scales: ternary.scales,
        shape: (rows, cols),
        group_size: effective_group_size,
    })
}

/// Quantize weights using AbsMax scaling.
///
/// For each group: scale = max(|W|), then round W/scale to {-1, 0, +1}.
/// More robust to outliers than AbsMean.
///
/// # Arguments
///
/// * `weights` - 2D weight tensor
/// * `config` - Quantization configuration
pub fn quantize_absmax(
    weights: &Tensor,
    config: &QuantizeConfig,
) -> Result<QuantizationResult, QuantizationError> {
    let (rows, cols) = weights.dims2()?;
    let data: Vec<f32> = weights.flatten_all()?.to_vec1()?;

    let effective_group_size = if config.group_size == 0 {
        cols
    } else {
        config.group_size
    };

    let groups_per_row = cols.div_ceil(effective_group_size);
    let mut scales = Vec::with_capacity(rows * groups_per_row);
    let mut values = Vec::with_capacity(rows * cols);

    for row in 0..rows {
        for group in 0..groups_per_row {
            let start = group * effective_group_size;
            let end = (start + effective_group_size).min(cols);

            // Find max absolute value in group
            let mut max_abs = 0.0f32;
            for col in start..end {
                let val = data[row * cols + col].abs();
                if val > max_abs {
                    max_abs = val;
                }
            }

            // Avoid division by zero
            let scale = if max_abs > 1e-10 { max_abs } else { 1.0 };
            scales.push(scale);

            // Quantize this group
            for col in start..end {
                let val = data[row * cols + col];
                let normalized = val / scale;
                let quantized = if normalized > 0.5 {
                    1i8
                } else if normalized < -0.5 {
                    -1i8
                } else {
                    0i8
                };
                values.push(quantized);
            }
        }
    }

    Ok(QuantizationResult {
        values,
        scales,
        shape: (rows, cols),
        group_size: effective_group_size,
    })
}

/// Quantize activations for inference.
///
/// Uses AbsMax per-tensor scaling to preserve dynamic range.
pub fn quantize_activations(activations: &Tensor) -> Result<(Tensor, f32), QuantizationError> {
    let data: Vec<f32> = activations.flatten_all()?.to_vec1()?;

    // Find max absolute value
    let max_abs = data.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    let scale = if max_abs > 1e-10 { max_abs } else { 1.0 };

    // Scale to [-1, 1] range
    let scaled: Vec<f32> = data.iter().map(|x| x / scale).collect();

    Ok((
        Tensor::from_vec(scaled, activations.shape(), activations.device())?,
        scale,
    ))
}

/// Dequantize ternary values back to float.
pub fn dequantize(result: &QuantizationResult, device: &Device) -> Result<Tensor, QuantizationError> {
    result.to_tensor(device)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_absmean() {
        let device = Device::Cpu;
        let weights = Tensor::from_vec(
            vec![0.5f32, -0.3, 0.1, 0.8, -0.2, 0.6, -0.7, 0.4],
            (2, 4),
            &device,
        )
        .unwrap();

        let config = QuantizeConfig::default();
        let result = quantize_absmean(&weights, &config).unwrap();

        assert_eq!(result.shape, (2, 4));
        assert_eq!(result.values.len(), 8);
        assert_eq!(result.scales.len(), 2); // Per-row

        // All values should be in {-1, 0, +1}
        for v in &result.values {
            assert!([-1, 0, 1].contains(v));
        }
    }

    #[test]
    fn test_quantize_absmax() {
        let device = Device::Cpu;
        let weights = Tensor::from_vec(
            vec![0.5f32, -0.3, 0.1, 0.8, -0.2, 0.6, -0.7, 0.4],
            (2, 4),
            &device,
        )
        .unwrap();

        let config = QuantizeConfig::default();
        let result = quantize_absmax(&weights, &config).unwrap();

        assert_eq!(result.shape, (2, 4));
        assert_eq!(result.values.len(), 8);

        for v in &result.values {
            assert!([-1, 0, 1].contains(v));
        }
    }

    #[test]
    fn test_quantize_dequantize_roundtrip() {
        let device = Device::Cpu;
        let weights = Tensor::from_vec(
            vec![0.8f32, -0.8, 0.0, 0.8, -0.8, 0.8, -0.8, 0.0],
            (2, 4),
            &device,
        )
        .unwrap();

        let config = QuantizeConfig::default();
        let result = quantize_absmean(&weights, &config).unwrap();
        let dequantized = dequantize(&result, &device).unwrap();

        // Check shape preserved
        assert_eq!(dequantized.dims(), &[2, 4]);

        // Dequantized values should be close to original for saturated values
        let deq_data: Vec<f32> = dequantized.flatten_all().unwrap().to_vec1().unwrap();
        let orig_data: Vec<f32> = weights.flatten_all().unwrap().to_vec1().unwrap();

        // At least the signs should match for non-zero values
        for (d, o) in deq_data.iter().zip(orig_data.iter()) {
            if o.abs() > 0.5 {
                assert_eq!(d.signum(), o.signum());
            }
        }
    }
}
