//! Normalization layer wrappers with GPU support fallback.
//!
//! This module provides wrappers around normalization operations that automatically
//! handle GPU/CPU conversions to support operations that don't have full GPU implementations.
//!
//! # Example
//!
//! ```rust,ignore
//! use candle_core::{Device, Tensor};
//! use axolotl_rs::normalization::RmsNormWrapper;
//!
//! # fn main() -> anyhow::Result<()> {
//! let device = Device::Cpu;
//! let hidden_size = 768;
//! let eps = 1e-5;
//!
//! // Create an RMS normalization layer
//! let rms_norm = RmsNormWrapper::new(hidden_size, eps, &device)?;
//!
//! // Dummy input with shape [batch, seq_len, hidden_size]
//! let x = Tensor::zeros((1, 4, hidden_size), candle_core::DType::F32, &device)?;
//!
//! // Apply normalization
//! let y = rms_norm.forward(&x)?;
//! # Ok(())
//! # }
//! ```

use crate::error::Result;
use candle_core::{Device, Tensor};

#[cfg(feature = "unsloth")]
use unsloth_rs::kernels::RmsNorm;

/// RMS Normalization with automatic GPU/CPU fallback.
///
/// When running on GPU, we use unsloth-rs's optimized implementation.
/// For CPU or when unsloth is unavailable, we fall back to standard operations.
pub struct RmsNormWrapper {
    #[cfg(feature = "unsloth")]
    inner: Option<RmsNorm>,
    #[cfg(not(feature = "unsloth"))]
    weight: Tensor,
    eps: f64,
}

impl RmsNormWrapper {
    /// Create a new RMS normalization layer.
    pub fn new(hidden_size: usize, eps: f64, device: &Device) -> Result<Self> {
        #[cfg(feature = "unsloth")]
        {
            let inner = if device.is_cuda() {
                Some(RmsNorm::new(hidden_size, eps, device).map_err(|e| {
                    crate::error::AxolotlError::Model(format!("Failed to create RmsNorm: {}", e))
                })?)
            } else {
                None
            };
            Ok(Self { inner, eps })
        }

        #[cfg(not(feature = "unsloth"))]
        {
            let weight = Tensor::ones((hidden_size,), candle_core::DType::F32, device)?;
            Ok(Self { weight, eps })
        }
    }

    /// Forward pass with GPU/CPU handling.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "unsloth")]
        {
            if let Some(ref inner) = self.inner {
                inner.forward(x).map_err(|e| {
                    crate::error::AxolotlError::Model(format!("RmsNorm forward failed: {}", e))
                })
            } else {
                self.forward_cpu(x)
            }
        }

        #[cfg(not(feature = "unsloth"))]
        {
            self.forward_cpu(x)
        }
    }

    #[cfg(any(not(feature = "unsloth"), feature = "unsloth"))]
    fn forward_cpu(&self, x: &Tensor) -> Result<Tensor> {
        // Fallback CPU implementation
        let x_sq = x.sqr()?;
        let mean_sq = x_sq.mean_keepdim(x.rank() - 1)?;
        let rms = (mean_sq + self.eps)?.sqrt()?;

        #[cfg(feature = "unsloth")]
        {
            let normalized = x.broadcast_div(&rms)?;
            // Create weight tensor if not using unsloth
            let weight = Tensor::ones(
                (x.shape().dims()[x.rank() - 1],),
                candle_core::DType::F32,
                x.device(),
            )?;
            let output = normalized.broadcast_mul(&weight)?;
            Ok(output)
        }

        #[cfg(not(feature = "unsloth"))]
        {
            let normalized = x.broadcast_div(&rms)?;
            let output = normalized.broadcast_mul(&self.weight)?;
            Ok(output)
        }
    }
}
