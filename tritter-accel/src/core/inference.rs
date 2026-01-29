//! Inference acceleration utilities.
//!
//! Provides tools for accelerating neural network inference:
//! - Batched operations
//! - Device dispatch (CPU/GPU)
//! - Model optimization helpers
//!
//! # Example
//!
//! ```rust,ignore
//! use tritter_accel::core::inference::{InferenceEngine, InferenceConfig};
//! use candle_core::Device;
//!
//! let config = InferenceConfig::default();
//! let engine = InferenceEngine::new(config)?;
//!
//! // Run batched inference
//! let outputs = engine.forward_batch(&inputs)?;
//! ```

use candle_core::{Device, Tensor};
use thiserror::Error;

use super::quantization::{quantize_absmean, QuantizationError, QuantizeConfig};
use super::ternary::{matmul, PackedTernary, TernaryError, TernaryMatmulConfig};

/// Errors from inference operations.
#[derive(Debug, Error)]
pub enum InferenceError {
    /// Configuration error.
    #[error("config error: {0}")]
    Config(String),

    /// Device error.
    #[error("device error: {0}")]
    Device(String),

    /// Shape mismatch.
    #[error("shape mismatch: {0}")]
    Shape(String),

    /// Tensor error.
    #[error("tensor error: {0}")]
    Tensor(#[from] candle_core::Error),

    /// Ternary error.
    #[error("ternary error: {0}")]
    Ternary(#[from] TernaryError),

    /// Quantization error.
    #[error("quantization error: {0}")]
    Quantization(#[from] QuantizationError),
}

/// Configuration for inference engine.
#[derive(Debug, Clone)]
pub struct InferenceConfig {
    /// Preferred device.
    pub device: DeviceType,
    /// Maximum batch size for batched operations.
    pub max_batch_size: usize,
    /// Enable weight quantization.
    pub quantize_weights: bool,
    /// Enable activation caching.
    pub cache_activations: bool,
}

/// Device type for inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceType {
    /// Automatic device selection.
    Auto,
    /// Force CPU.
    Cpu,
    /// Force GPU with optional device index.
    Gpu(Option<usize>),
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            device: DeviceType::Auto,
            max_batch_size: 32,
            quantize_weights: false,
            cache_activations: false,
        }
    }
}

impl InferenceConfig {
    /// Set device type.
    pub fn with_device(mut self, device: DeviceType) -> Self {
        self.device = device;
        self
    }

    /// Set max batch size.
    pub fn with_max_batch_size(mut self, size: usize) -> Self {
        self.max_batch_size = size;
        self
    }

    /// Enable weight quantization.
    pub fn with_quantization(mut self, enabled: bool) -> Self {
        self.quantize_weights = enabled;
        self
    }
}

/// Inference engine for accelerated forward passes.
#[derive(Debug)]
pub struct InferenceEngine {
    config: InferenceConfig,
    device: Device,
}

impl InferenceEngine {
    /// Create a new inference engine.
    pub fn new(config: InferenceConfig) -> Result<Self, InferenceError> {
        let device = match config.device {
            DeviceType::Cpu => Device::Cpu,
            DeviceType::Auto => {
                #[cfg(feature = "cuda")]
                {
                    Device::cuda_if_available(0).unwrap_or(Device::Cpu)
                }
                #[cfg(not(feature = "cuda"))]
                {
                    Device::Cpu
                }
            }
            DeviceType::Gpu(ordinal) => {
                #[cfg(feature = "cuda")]
                {
                    let idx = ordinal.unwrap_or(0);
                    Device::new_cuda(idx)
                        .map_err(|e| InferenceError::Device(format!("CUDA device {idx}: {e}")))?
                }
                #[cfg(not(feature = "cuda"))]
                {
                    let _ = ordinal;
                    return Err(InferenceError::Device(
                        "CUDA not compiled. Rebuild with --features cuda".to_string(),
                    ));
                }
            }
        };

        Ok(Self { config, device })
    }

    /// Get the active device.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Check if running on GPU.
    pub fn is_gpu(&self) -> bool {
        matches!(self.device, Device::Cuda(_))
    }

    /// Linear layer forward pass with optional ternary quantization.
    ///
    /// Computes: output = input @ weight.T + bias
    pub fn linear(
        &self,
        input: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>,
    ) -> Result<Tensor, InferenceError> {
        // Move tensors to device if needed
        let input = input.to_device(&self.device)?;
        let weight = weight.to_device(&self.device)?;

        // Compute matmul
        let output = input.matmul(&weight.t()?)?;

        // Add bias if present
        let output = if let Some(b) = bias {
            let b = b.to_device(&self.device)?;
            output.broadcast_add(&b)?
        } else {
            output
        };

        Ok(output)
    }

    /// Ternary linear layer (quantized weights).
    ///
    /// Quantizes weights to ternary for memory-efficient inference.
    pub fn ternary_linear(
        &self,
        input: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>,
    ) -> Result<Tensor, InferenceError> {
        // Quantize weights
        let quant_config = QuantizeConfig::default();
        let quantized = quantize_absmean(weight, &quant_config)?;
        let packed = quantized.to_packed()?;

        // Move input to device
        let input = input.to_device(&self.device)?;

        // Ternary matmul
        let matmul_config = TernaryMatmulConfig::default();
        let output = matmul(&input, &packed, Some(&matmul_config))?;

        // Add bias
        let output = if let Some(b) = bias {
            let b = b.to_device(&self.device)?;
            output.broadcast_add(&b)?
        } else {
            output
        };

        Ok(output)
    }

    /// Batched inference with automatic chunking.
    ///
    /// Splits large batches into smaller chunks to fit in memory.
    pub fn batched_forward<F>(
        &self,
        inputs: &Tensor,
        forward_fn: F,
    ) -> Result<Tensor, InferenceError>
    where
        F: Fn(&Tensor) -> Result<Tensor, InferenceError>,
    {
        let batch_size = inputs.dim(0)?;

        if batch_size <= self.config.max_batch_size {
            return forward_fn(inputs);
        }

        // Split into chunks
        let mut outputs = Vec::new();
        let mut start = 0;

        while start < batch_size {
            let end = (start + self.config.max_batch_size).min(batch_size);
            let chunk = inputs.narrow(0, start, end - start)?;
            let output = forward_fn(&chunk)?;
            outputs.push(output);
            start = end;
        }

        // Concatenate outputs
        Ok(Tensor::cat(&outputs, 0)?)
    }

    /// Apply softmax along specified dimension.
    pub fn softmax(&self, input: &Tensor, dim: usize) -> Result<Tensor, InferenceError> {
        let input = input.to_device(&self.device)?;
        Ok(candle_nn::ops::softmax(&input, dim)?)
    }

    /// Apply layer normalization.
    pub fn layer_norm(
        &self,
        input: &Tensor,
        weight: &Tensor,
        bias: &Tensor,
        eps: f64,
    ) -> Result<Tensor, InferenceError> {
        let input = input.to_device(&self.device)?;
        let weight = weight.to_device(&self.device)?;
        let bias = bias.to_device(&self.device)?;

        // Compute mean and variance along last dimension
        let dim = input.dims().len() - 1;
        let mean = input.mean_keepdim(dim)?;
        let var = input
            .broadcast_sub(&mean)?
            .sqr()?
            .mean_keepdim(dim)?;

        // Normalize
        let normalized = input
            .broadcast_sub(&mean)?
            .broadcast_div(&(var + eps)?.sqrt()?)?;

        // Scale and shift
        Ok(normalized.broadcast_mul(&weight)?.broadcast_add(&bias)?)
    }
}

/// Pre-computed ternary layer for repeated inference.
///
/// Stores quantized weights to avoid re-quantization overhead.
#[derive(Debug)]
pub struct TernaryLayer {
    /// Packed ternary weights.
    pub weights: PackedTernary,
    /// Bias (optional).
    pub bias: Option<Vec<f32>>,
    /// Input features.
    pub in_features: usize,
    /// Output features.
    pub out_features: usize,
}

impl TernaryLayer {
    /// Create from float weight tensor.
    pub fn from_tensor(
        weight: &Tensor,
        bias: Option<&Tensor>,
    ) -> Result<Self, InferenceError> {
        let (out_features, in_features) = weight.dims2()?;

        // Quantize weights
        let quant_config = QuantizeConfig::default();
        let quantized = quantize_absmean(weight, &quant_config)?;
        let weights = quantized.to_packed()?;

        // Extract bias if present
        let bias = if let Some(b) = bias {
            Some(b.flatten_all()?.to_vec1()?)
        } else {
            None
        };

        Ok(Self {
            weights,
            bias,
            in_features,
            out_features,
        })
    }

    /// Forward pass.
    pub fn forward(&self, input: &Tensor) -> Result<Tensor, InferenceError> {
        let matmul_config = TernaryMatmulConfig::default();
        let output = matmul(input, &self.weights, Some(&matmul_config))?;

        // Add bias
        if let Some(ref bias) = self.bias {
            let bias_tensor = Tensor::from_vec(bias.clone(), self.out_features, input.device())?;
            Ok(output.broadcast_add(&bias_tensor)?)
        } else {
            Ok(output)
        }
    }

    /// Memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        // Packed weights: 2 bits per value
        let weight_bits = self.in_features * self.out_features * 2;
        let weight_bytes = weight_bits.div_ceil(8);

        // Scales: f32 per row
        let scale_bytes = self.out_features * 4;

        // Bias: f32 per output
        let bias_bytes = self.bias.as_ref().map(|b| b.len() * 4).unwrap_or(0);

        weight_bytes + scale_bytes + bias_bytes
    }

    /// Original (unquantized) memory usage for comparison.
    pub fn original_memory_bytes(&self) -> usize {
        // f32 weights + f32 bias
        let weight_bytes = self.in_features * self.out_features * 4;
        let bias_bytes = self.bias.as_ref().map(|b| b.len() * 4).unwrap_or(0);
        weight_bytes + bias_bytes
    }

    /// Compression ratio achieved.
    #[allow(clippy::cast_precision_loss)]
    pub fn compression_ratio(&self) -> f32 {
        self.original_memory_bytes() as f32 / self.memory_bytes() as f32
    }
}

/// KV cache for efficient autoregressive inference.
#[derive(Debug)]
pub struct KVCache {
    /// Cached keys.
    keys: Vec<Tensor>,
    /// Cached values.
    values: Vec<Tensor>,
    /// Maximum sequence length.
    max_seq_len: usize,
    /// Current sequence length.
    seq_len: usize,
}

impl KVCache {
    /// Create a new KV cache.
    pub fn new(max_seq_len: usize) -> Self {
        Self {
            keys: Vec::new(),
            values: Vec::new(),
            max_seq_len,
            seq_len: 0,
        }
    }

    /// Update cache with new key-value pairs.
    pub fn update(
        &mut self,
        new_keys: Tensor,
        new_values: Tensor,
    ) -> Result<(Tensor, Tensor), InferenceError> {
        // Append new KV pairs
        self.keys.push(new_keys);
        self.values.push(new_values);
        self.seq_len += 1;

        // Concatenate all cached KVs
        let all_keys = Tensor::cat(&self.keys, 1)?;
        let all_values = Tensor::cat(&self.values, 1)?;

        // Trim if exceeds max length
        if self.seq_len > self.max_seq_len {
            self.keys.remove(0);
            self.values.remove(0);
            self.seq_len = self.max_seq_len;
        }

        Ok((all_keys, all_values))
    }

    /// Clear the cache.
    pub fn clear(&mut self) {
        self.keys.clear();
        self.values.clear();
        self.seq_len = 0;
    }

    /// Current sequence length.
    pub fn len(&self) -> usize {
        self.seq_len
    }

    /// Check if cache is empty.
    pub fn is_empty(&self) -> bool {
        self.seq_len == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inference_engine_creation() {
        let config = InferenceConfig::default().with_device(DeviceType::Cpu);
        let engine = InferenceEngine::new(config).unwrap();

        assert!(!engine.is_gpu());
    }

    #[test]
    fn test_linear_forward() {
        let config = InferenceConfig::default().with_device(DeviceType::Cpu);
        let engine = InferenceEngine::new(config).unwrap();

        let input = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (1, 4), engine.device()).unwrap();
        let weight =
            Tensor::from_vec(vec![1.0f32; 8], (2, 4), engine.device()).unwrap();

        let output = engine.linear(&input, &weight, None).unwrap();

        assert_eq!(output.dims(), &[1, 2]);
    }

    #[test]
    fn test_ternary_layer() {
        let device = Device::Cpu;
        let weight = Tensor::randn(0f32, 1f32, (16, 32), &device).unwrap();

        let layer = TernaryLayer::from_tensor(&weight, None).unwrap();

        // Check compression
        assert!(layer.compression_ratio() > 10.0);

        // Test forward pass
        let input = Tensor::randn(0f32, 1f32, (1, 32), &device).unwrap();
        let output = layer.forward(&input).unwrap();

        assert_eq!(output.dims(), &[1, 16]);
    }

    #[test]
    fn test_kv_cache() {
        let mut cache = KVCache::new(4);

        assert!(cache.is_empty());

        let device = Device::Cpu;
        let k1 = Tensor::zeros((1, 1, 8), candle_core::DType::F32, &device).unwrap();
        let v1 = Tensor::zeros((1, 1, 8), candle_core::DType::F32, &device).unwrap();

        let (keys, values) = cache.update(k1, v1).unwrap();

        assert_eq!(cache.len(), 1);
        assert_eq!(keys.dim(1).unwrap(), 1);
        assert_eq!(values.dim(1).unwrap(), 1);
    }

    #[test]
    fn test_batched_forward() {
        let config = InferenceConfig::default()
            .with_device(DeviceType::Cpu)
            .with_max_batch_size(2);
        let engine = InferenceEngine::new(config).unwrap();

        // Create input with 5 samples (larger than max_batch_size)
        let input = Tensor::randn(0f32, 1f32, (5, 4), engine.device()).unwrap();

        // Simple identity-like forward function
        let output = engine
            .batched_forward(&input, |x| Ok(x.clone()))
            .unwrap();

        assert_eq!(output.dims(), &[5, 4]);
    }
}
