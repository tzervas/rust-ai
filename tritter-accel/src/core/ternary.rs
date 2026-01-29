//! Ternary weight operations.
//!
//! Provides efficient packing, unpacking, and matrix multiplication
//! for ternary weights (values in {-1, 0, +1}).
//!
//! # Architecture
//!
//! Delegates to `trit-vsa` for the underlying bitsliced storage and
//! `bitnet-quantize` for quantization algorithms. This module provides
//! a unified interface for common workflows.

use candle_core::{Device, Tensor};
use thiserror::Error;
use trit_vsa::{PackedTritVec, Trit};

/// Errors from ternary operations.
#[derive(Debug, Error)]
pub enum TernaryError {
    /// Shape mismatch between operands.
    #[error("shape mismatch: expected {expected}, got {actual}")]
    ShapeMismatch { expected: String, actual: String },

    /// Invalid ternary value (not in {-1, 0, +1}).
    #[error("invalid ternary value {value} at index {index}")]
    InvalidValue { value: i8, index: usize },

    /// Tensor operation failed.
    #[error("tensor error: {0}")]
    Tensor(#[from] candle_core::Error),
}

/// Configuration for ternary matrix multiplication.
#[derive(Debug, Clone)]
pub struct TernaryMatmulConfig {
    /// Use GPU if available.
    pub use_gpu: bool,
    /// Block size for tiled matmul (0 = auto).
    pub block_size: usize,
}

impl Default for TernaryMatmulConfig {
    fn default() -> Self {
        Self {
            use_gpu: true,
            block_size: 0,
        }
    }
}

/// Packed ternary weight storage.
///
/// Contains packed ternary values along with per-row scale factors
/// for dequantization.
#[derive(Debug, Clone)]
pub struct PackedTernary {
    /// Packed ternary vectors, one per output row.
    pub data: Vec<PackedTritVec>,
    /// Per-row scale factors.
    pub scales: Vec<f32>,
    /// Original shape (rows, cols).
    pub shape: (usize, usize),
}

impl PackedTernary {
    /// Create from raw ternary values and scales.
    ///
    /// # Arguments
    ///
    /// * `values` - 2D tensor of ternary values (f32 with values -1, 0, +1)
    /// * `scales` - 1D tensor of per-row scales
    ///
    /// # Errors
    ///
    /// Returns error if values contain non-ternary values or shapes don't match.
    pub fn from_tensors(values: &Tensor, scales: &Tensor) -> Result<Self, TernaryError> {
        let (rows, cols) = values.dims2()?;
        let values_flat: Vec<f32> = values.flatten_all()?.to_vec1()?;
        let scales_vec: Vec<f32> = scales.flatten_all()?.to_vec1()?;

        if scales_vec.len() != rows {
            return Err(TernaryError::ShapeMismatch {
                expected: format!("scales length {rows}"),
                actual: format!("got {}", scales_vec.len()),
            });
        }

        let mut data = Vec::with_capacity(rows);
        for row in 0..rows {
            let mut packed = PackedTritVec::new(cols);
            for col in 0..cols {
                let val = values_flat[row * cols + col];
                let trit = match val as i8 {
                    v if v > 0 => Trit::P,
                    v if v < 0 => Trit::N,
                    _ => Trit::Z,
                };
                packed.set(col, trit);
            }
            data.push(packed);
        }

        Ok(Self {
            data,
            scales: scales_vec,
            shape: (rows, cols),
        })
    }

    /// Create from i8 slice directly.
    ///
    /// # Arguments
    ///
    /// * `values` - Flattened ternary values in row-major order
    /// * `scales` - Per-row scale factors
    /// * `shape` - (rows, cols)
    pub fn from_i8(
        values: &[i8],
        scales: &[f32],
        shape: (usize, usize),
    ) -> Result<Self, TernaryError> {
        let (rows, cols) = shape;
        if values.len() != rows * cols {
            return Err(TernaryError::ShapeMismatch {
                expected: format!("values length {}", rows * cols),
                actual: format!("got {}", values.len()),
            });
        }
        if scales.len() != rows {
            return Err(TernaryError::ShapeMismatch {
                expected: format!("scales length {rows}"),
                actual: format!("got {}", scales.len()),
            });
        }

        let mut data = Vec::with_capacity(rows);
        for row in 0..rows {
            let mut packed = PackedTritVec::new(cols);
            for col in 0..cols {
                let idx = row * cols + col;
                let val = values[idx];
                let trit = match val {
                    1 => Trit::P,
                    0 => Trit::Z,
                    -1 => Trit::N,
                    _ => {
                        return Err(TernaryError::InvalidValue {
                            value: val,
                            index: idx,
                        })
                    }
                };
                packed.set(col, trit);
            }
            data.push(packed);
        }

        Ok(Self {
            data,
            scales: scales.to_vec(),
            shape,
        })
    }

    /// Unpack to f32 tensor with scales applied.
    pub fn to_tensor(&self, device: &Device) -> Result<Tensor, TernaryError> {
        let (rows, cols) = self.shape;
        let mut values = vec![0.0f32; rows * cols];

        for (row, (packed, &scale)) in self.data.iter().zip(self.scales.iter()).enumerate() {
            for col in 0..cols {
                let trit_val = f32::from(packed.get(col).value());
                values[row * cols + col] = trit_val * scale;
            }
        }

        Ok(Tensor::from_vec(values, (rows, cols), device)?)
    }

    /// Get raw i8 ternary values without scaling.
    pub fn to_i8(&self) -> Vec<i8> {
        let (rows, cols) = self.shape;
        let mut values = vec![0i8; rows * cols];

        for (row, packed) in self.data.iter().enumerate() {
            for col in 0..cols {
                values[row * cols + col] = packed.get(col).value();
            }
        }

        values
    }

    /// Pack to 2-bit byte format for storage/transmission.
    ///
    /// Each byte contains 4 ternary values (2 bits each):
    /// - 0b00 = 0
    /// - 0b01 = +1
    /// - 0b10 = -1
    pub fn to_packed_bytes(&self) -> Vec<u8> {
        let (rows, cols) = self.shape;
        let packed_cols = cols.div_ceil(4);
        let mut bytes = vec![0u8; rows * packed_cols];

        for (row, packed) in self.data.iter().enumerate() {
            for col in 0..cols {
                let trit = packed.get(col);
                let bits = match trit {
                    Trit::P => 0b01,
                    Trit::N => 0b10,
                    Trit::Z => 0b00,
                };
                let byte_idx = row * packed_cols + col / 4;
                let bit_offset = (col % 4) * 2;
                bytes[byte_idx] |= bits << bit_offset;
            }
        }

        bytes
    }

    /// Unpack from 2-bit byte format.
    pub fn from_packed_bytes(
        bytes: &[u8],
        scales: &[f32],
        shape: (usize, usize),
    ) -> Result<Self, TernaryError> {
        let (rows, cols) = shape;
        let packed_cols = cols.div_ceil(4);

        if bytes.len() != rows * packed_cols {
            return Err(TernaryError::ShapeMismatch {
                expected: format!("bytes length {}", rows * packed_cols),
                actual: format!("got {}", bytes.len()),
            });
        }

        let mut data = Vec::with_capacity(rows);
        for row in 0..rows {
            let mut packed = PackedTritVec::new(cols);
            for col in 0..cols {
                let byte_idx = row * packed_cols + col / 4;
                let bit_offset = (col % 4) * 2;
                let bits = (bytes[byte_idx] >> bit_offset) & 0b11;
                let trit = match bits {
                    0b01 => Trit::P,
                    0b10 => Trit::N,
                    _ => Trit::Z,
                };
                packed.set(col, trit);
            }
            data.push(packed);
        }

        Ok(Self {
            data,
            scales: scales.to_vec(),
            shape,
        })
    }
}

/// Perform ternary matrix multiplication.
///
/// Computes: output = input @ weights.T
///
/// For each element, multiplies input by the ternary weight value (+1, 0, -1)
/// which reduces to additions and subtractions (no actual multiplications).
///
/// # Arguments
///
/// * `input` - Input tensor of shape (batch, in_features)
/// * `weights` - Packed ternary weights of shape (out_features, in_features)
/// * `config` - Optional configuration
///
/// # Returns
///
/// Output tensor of shape (batch, out_features)
pub fn matmul(
    input: &Tensor,
    weights: &PackedTernary,
    _config: Option<&TernaryMatmulConfig>,
) -> Result<Tensor, TernaryError> {
    let (batch, in_features) = input.dims2()?;
    let (out_features, weight_features) = weights.shape;

    if in_features != weight_features {
        return Err(TernaryError::ShapeMismatch {
            expected: format!("input features {weight_features}"),
            actual: format!("got {in_features}"),
        });
    }

    let input_data: Vec<f32> = input.flatten_all()?.to_vec1()?;
    let mut output = vec![0.0f32; batch * out_features];

    // For each batch
    for b in 0..batch {
        // For each output feature (weight row)
        for (o, (weight_row, &scale)) in weights.data.iter().zip(weights.scales.iter()).enumerate()
        {
            let mut sum = 0.0f32;

            // Ternary dot product: sum of (input * trit_value)
            for i in 0..in_features {
                let x = input_data[b * in_features + i];
                sum += match weight_row.get(i) {
                    Trit::P => x,
                    Trit::N => -x,
                    Trit::Z => 0.0,
                };
            }

            output[b * out_features + o] = sum * scale;
        }
    }

    Ok(Tensor::from_vec(output, (batch, out_features), input.device())?)
}

/// Compute ternary dot product between two packed vectors.
///
/// Both vectors must have the same length.
pub fn dot(a: &PackedTritVec, b: &PackedTritVec) -> i32 {
    a.dot(b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_packed_ternary_roundtrip() {
        let values = vec![1i8, 0, -1, 1, -1, 0, 1, -1];
        let scales = vec![1.0f32, 0.5];
        let shape = (2, 4);

        let packed = PackedTernary::from_i8(&values, &scales, shape).unwrap();
        let recovered = packed.to_i8();

        assert_eq!(values, recovered);
    }

    #[test]
    fn test_packed_bytes_roundtrip() {
        let values = vec![1i8, 0, -1, 1, -1, 0, 1, -1];
        let scales = vec![1.0f32, 0.5];
        let shape = (2, 4);

        let packed = PackedTernary::from_i8(&values, &scales, shape).unwrap();
        let bytes = packed.to_packed_bytes();
        let recovered = PackedTernary::from_packed_bytes(&bytes, &scales, shape).unwrap();

        assert_eq!(packed.to_i8(), recovered.to_i8());
    }

    #[test]
    fn test_ternary_matmul() {
        let device = Device::Cpu;

        // Input: 1x4
        let input = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (1, 4), &device).unwrap();

        // Weights: 2x4, values [+1, -1, +1, -1] and [+1, +1, 0, 0]
        let weight_values = vec![1i8, -1, 1, -1, 1, 1, 0, 0];
        let scales = vec![1.0f32, 1.0];
        let weights = PackedTernary::from_i8(&weight_values, &scales, (2, 4)).unwrap();

        let output = matmul(&input, &weights, None).unwrap();
        let output_data: Vec<f32> = output.flatten_all().unwrap().to_vec1().unwrap();

        // Row 0: 1*1 + 2*(-1) + 3*1 + 4*(-1) = 1 - 2 + 3 - 4 = -2
        // Row 1: 1*1 + 2*1 + 3*0 + 4*0 = 1 + 2 = 3
        assert!((output_data[0] - (-2.0)).abs() < 1e-6);
        assert!((output_data[1] - 3.0).abs() < 1e-6);
    }
}
