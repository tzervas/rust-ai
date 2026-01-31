//! Custom layer normalization using basic tensor operations.
//!
//! This module provides a manual implementation of layer normalization
//! that uses only basic tensor operations (mean, variance, add, mul, div).
//! This ensures CUDA compatibility even when the CUDA layer-norm kernel
//! is not available (e.g., on newer GPU architectures).

use candle_core::{Result, Tensor, D};
use candle_nn::{Module, VarBuilder};

/// Manual LayerNorm implementation using basic tensor ops.
///
/// This implementation is CUDA-compatible as it only uses
/// basic operations that have CUDA implementations.
#[derive(Debug, Clone)]
pub struct ManualLayerNorm {
    weight: Tensor,
    bias: Option<Tensor>,
    eps: f64,
    normalized_shape: usize,
}

impl ManualLayerNorm {
    /// Create a new ManualLayerNorm.
    pub fn new(weight: Tensor, bias: Option<Tensor>, eps: f64) -> Self {
        let normalized_shape = weight.dims()[0];
        Self {
            weight,
            bias,
            eps,
            normalized_shape,
        }
    }

    /// Get the normalized shape.
    pub fn normalized_shape(&self) -> usize {
        self.normalized_shape
    }

    /// Forward pass using basic tensor operations.
    ///
    /// Layer norm: y = (x - mean(x)) / sqrt(var(x) + eps) * weight + bias
    fn forward_impl(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.dims();
        let last_dim = dims.len() - 1;

        // Compute mean along last dimension
        let mean = x.mean_keepdim(D::Minus1)?;

        // Compute variance: E[(x - mean)^2]
        let x_centered = x.broadcast_sub(&mean)?;
        let x_centered_sq = x_centered.sqr()?;
        let var = x_centered_sq.mean_keepdim(D::Minus1)?;

        // Normalize: (x - mean) / sqrt(var + eps)
        let eps_tensor = Tensor::new(self.eps as f32, x.device())?;
        let var_eps = var.broadcast_add(&eps_tensor)?;
        let std = var_eps.sqrt()?;
        let normalized = x_centered.broadcast_div(&std)?;

        // Scale and shift: normalized * weight + bias
        let scaled = normalized.broadcast_mul(&self.weight)?;

        match &self.bias {
            Some(bias) => scaled.broadcast_add(bias),
            None => Ok(scaled),
        }
    }
}

impl Module for ManualLayerNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.forward_impl(x)
    }
}

/// Create a manual layer norm layer.
pub fn manual_layer_norm(size: usize, eps: f64, vb: VarBuilder) -> Result<ManualLayerNorm> {
    let weight = vb.get_with_hints(size, "weight", candle_nn::Init::Const(1.0))?;
    let bias = vb.get_with_hints(size, "bias", candle_nn::Init::Const(0.0))?;
    Ok(ManualLayerNorm::new(weight, Some(bias), eps))
}

/// RMS normalization using basic tensor ops.
///
/// RMS norm: y = x / sqrt(mean(x^2) + eps) * weight
#[derive(Debug, Clone)]
pub struct ManualRmsNorm {
    weight: Tensor,
    eps: f64,
}

impl ManualRmsNorm {
    /// Create a new ManualRmsNorm.
    pub fn new(weight: Tensor, eps: f64) -> Self {
        Self { weight, eps }
    }

    fn forward_impl(&self, x: &Tensor) -> Result<Tensor> {
        // RMS: sqrt(mean(x^2))
        let x_sq = x.sqr()?;
        let mean_sq = x_sq.mean_keepdim(D::Minus1)?;

        let eps_tensor = Tensor::new(self.eps as f32, x.device())?;
        let mean_sq_eps = mean_sq.broadcast_add(&eps_tensor)?;
        let rms = mean_sq_eps.sqrt()?;

        // Normalize and scale
        let normalized = x.broadcast_div(&rms)?;
        normalized.broadcast_mul(&self.weight)
    }
}

impl Module for ManualRmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.forward_impl(x)
    }
}

/// Create a manual RMS norm layer.
pub fn manual_rms_norm(size: usize, eps: f64, vb: VarBuilder) -> Result<ManualRmsNorm> {
    let weight = vb.get_with_hints(size, "weight", candle_nn::Init::Const(1.0))?;
    Ok(ManualRmsNorm::new(weight, eps))
}

/// Manual softmax over last dimension using basic tensor ops.
///
/// softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
///
/// This implementation is CUDA-compatible as it only uses basic operations.
pub fn manual_softmax_last_dim(x: &Tensor) -> Result<Tensor> {
    // Subtract max for numerical stability
    let max = x.max_keepdim(D::Minus1)?;
    let x_shifted = x.broadcast_sub(&max)?;

    // exp(x - max)
    let exp_x = x_shifted.exp()?;

    // sum(exp(x - max))
    let sum_exp = exp_x.sum_keepdim(D::Minus1)?;

    // softmax = exp / sum
    exp_x.broadcast_div(&sum_exp)
}

/// Manual log_softmax over last dimension using basic tensor ops.
///
/// log_softmax(x) = x - max(x) - log(sum(exp(x - max(x))))
pub fn manual_log_softmax_last_dim(x: &Tensor) -> Result<Tensor> {
    // Subtract max for numerical stability
    let max = x.max_keepdim(D::Minus1)?;
    let x_shifted = x.broadcast_sub(&max)?;

    // log(sum(exp(x - max)))
    let exp_x = x_shifted.exp()?;
    let sum_exp = exp_x.sum_keepdim(D::Minus1)?;
    let log_sum_exp = sum_exp.log()?;

    // log_softmax = x - max - log_sum_exp
    x_shifted.broadcast_sub(&log_sum_exp)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, DType};
    use candle_nn::VarMap;

    #[test]
    fn test_manual_layer_norm_shape() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let norm = manual_layer_norm(64, 1e-5, vb).unwrap();

        // Test 2D input
        let x = Tensor::randn(0.0f32, 1.0, (2, 64), &device).unwrap();
        let out = norm.forward(&x).unwrap();
        assert_eq!(out.dims(), &[2, 64]);

        // Test 3D input
        let x = Tensor::randn(0.0f32, 1.0, (2, 8, 64), &device).unwrap();
        let out = norm.forward(&x).unwrap();
        assert_eq!(out.dims(), &[2, 8, 64]);
    }

    #[test]
    fn test_manual_layer_norm_normalized() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let norm = manual_layer_norm(64, 1e-5, vb).unwrap();

        let x = Tensor::randn(0.0f32, 1.0, (1, 64), &device).unwrap();
        let out = norm.forward(&x).unwrap();

        // After normalization, mean should be close to 0 and std close to 1
        let mean = out.mean_all().unwrap().to_scalar::<f32>().unwrap();
        let var = out.sqr().unwrap().mean_all().unwrap().to_scalar::<f32>().unwrap();

        assert!(mean.abs() < 0.5, "Mean should be close to 0, got {}", mean);
        assert!((var - 1.0).abs() < 0.5, "Variance should be close to 1, got {}", var);
    }

    #[test]
    fn test_manual_rms_norm_shape() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let norm = manual_rms_norm(64, 1e-5, vb).unwrap();

        let x = Tensor::randn(0.0f32, 1.0, (2, 8, 64), &device).unwrap();
        let out = norm.forward(&x).unwrap();
        assert_eq!(out.dims(), &[2, 8, 64]);
    }
}
