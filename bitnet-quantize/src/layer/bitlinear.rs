//! BitLinear layer - drop-in replacement for nn::Linear with ternary weights.

use candle_core::{Device, Tensor};
use candle_nn::Module;

use crate::config::BitNetConfig;
use crate::error::Result;
use crate::quantization::{
    dequantize_activations, dequantize_weights, quantize_activations, quantize_weights,
    TernaryWeight,
};

/// Emit a warning when CPU fallback is used.
///
/// GPU (CUDA) is the intended default for BitNet operations.
/// This function warns users once per process when CPU is being used.
fn warn_cpu_fallback(device: &Device) {
    static WARN_ONCE: std::sync::Once = std::sync::Once::new();
    if matches!(device, Device::Cpu) {
        WARN_ONCE.call_once(|| {
            eprintln!(
                "bitnet-quantize: CPU device in use. CUDA is the intended default; \
                 enable the 'cuda' feature and use Device::cuda_if_available(0) when possible."
            );
        });
    }
}

/// BitLinear layer with ternary weights and INT8 activations.
///
/// This is a drop-in replacement for `candle_nn::Linear` that uses:
/// - Ternary weights {-1, 0, +1} with per-group scales
/// - INT8 activation quantization with per-token scales
///
/// # Example
///
/// ```ignore
/// use bitnet_rs::{BitLinear, BitNetConfig};
/// use candle_core::{Device, Tensor};
///
/// let device = Device::Cpu;
/// let config = BitNetConfig::default();
///
/// // Create from existing weights
/// let weight = Tensor::randn(0.0f32, 1.0, (512, 256), &device)?;
/// let layer = BitLinear::from_weight(&weight, None, &config)?;
///
/// // Forward pass
/// let input = Tensor::randn(0.0f32, 1.0, (4, 256), &device)?;
/// let output = layer.forward(&input)?;
/// ```
#[derive(Debug)]
pub struct BitLinear {
    /// Quantized ternary weights.
    weight: TernaryWeight,

    /// Optional bias (not quantized).
    bias: Option<Tensor>,

    /// Configuration.
    config: BitNetConfig,

    /// Device for tensor operations.
    device: Device,
}

impl BitLinear {
    /// Create a new BitLinear layer from a weight tensor.
    ///
    /// # Arguments
    ///
    /// * `weight` - Weight tensor [out_features, in_features]
    /// * `bias` - Optional bias tensor [out_features]
    /// * `config` - BitNet configuration
    ///
    /// # Errors
    ///
    /// Returns error if weight quantization fails.
    pub fn from_weight(
        weight: &Tensor,
        bias: Option<&Tensor>,
        config: &BitNetConfig,
    ) -> Result<Self> {
        config.validate()?;

        let device = weight.device().clone();
        warn_cpu_fallback(&device);
        let quantized_weight = quantize_weights(weight, config)?;

        Ok(Self {
            weight: quantized_weight,
            bias: bias.cloned(),
            config: config.clone(),
            device,
        })
    }

    /// Create a new BitLinear layer from pre-quantized weights.
    ///
    /// # Arguments
    ///
    /// * `weight` - Pre-quantized ternary weight
    /// * `bias` - Optional bias tensor
    /// * `config` - BitNet configuration
    /// * `device` - Device for operations
    #[must_use]
    pub fn from_quantized(
        weight: TernaryWeight,
        bias: Option<Tensor>,
        config: BitNetConfig,
        device: Device,
    ) -> Self {
        warn_cpu_fallback(&device);
        Self {
            weight,
            bias,
            config,
            device,
        }
    }

    /// Get the input features dimension.
    #[must_use]
    pub fn in_features(&self) -> usize {
        self.weight.in_features()
    }

    /// Get the output features dimension.
    #[must_use]
    pub fn out_features(&self) -> usize {
        self.weight.out_features()
    }

    /// Get reference to the quantized weights.
    #[must_use]
    pub const fn quantized_weight(&self) -> &TernaryWeight {
        &self.weight
    }

    /// Get reference to the bias.
    #[must_use]
    pub const fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }

    /// Get reference to the configuration.
    #[must_use]
    pub const fn config(&self) -> &BitNetConfig {
        &self.config
    }

    /// Get the device.
    #[must_use]
    pub const fn device(&self) -> &Device {
        &self.device
    }

    /// Get the weight sparsity.
    #[must_use]
    pub fn sparsity(&self) -> f32 {
        self.weight.sparsity()
    }

    /// Get the compression ratio.
    #[must_use]
    pub fn compression_ratio(&self) -> f32 {
        self.weight.compression_ratio()
    }

    /// Forward pass with explicit activation quantization.
    ///
    /// This method:
    /// 1. Quantizes input activations to INT8
    /// 2. Uses GPU-optimized ternary matmul if available, otherwise dequantizes
    /// 3. Performs the linear transformation
    /// 4. Adds bias if present
    ///
    /// When the `cuda` feature is enabled and a CUDA device is detected,
    /// this uses optimized GPU kernels that exploit ternary weight sparsity.
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor [batch, ..., in_features]
    ///
    /// # Errors
    ///
    /// Returns error if forward pass fails.
    pub fn forward_quantized(&self, input: &Tensor) -> Result<Tensor> {
        // Quantize activations
        let quantized_input = quantize_activations(input, &self.config)?;
        let dequant_input = dequantize_activations(&quantized_input, &self.device)?;

        // Try GPU-optimized ternary matmul if available
        #[cfg(feature = "cuda")]
        let output = {
            if crate::kernels::should_use_gpu(&dequant_input, &self.weight) {
                crate::kernels::ternary_matmul_gpu(&dequant_input, &self.weight)?
            } else {
                // Fallback to standard matmul
                let dequant_weight = dequantize_weights(&self.weight, &self.device)?;
                dequant_input.matmul(&dequant_weight.t()?)?
            }
        };

        #[cfg(not(feature = "cuda"))]
        let output = {
            // Standard dequantize + matmul path
            let dequant_weight = dequantize_weights(&self.weight, &self.device)?;
            dequant_input.matmul(&dequant_weight.t()?)?
        };

        // Add bias
        let output = if let Some(ref bias) = self.bias {
            output.broadcast_add(bias)?
        } else {
            output
        };

        Ok(output)
    }
}

impl Module for BitLinear {
    fn forward(&self, input: &Tensor) -> candle_core::Result<Tensor> {
        let dims = input.dims();

        // Handle 3D input [batch, seq_len, hidden] by flattening
        let (flat_input, original_shape) = if dims.len() == 3 {
            let (batch, seq_len, hidden) = (dims[0], dims[1], dims[2]);
            (
                input.reshape((batch * seq_len, hidden))?,
                Some((batch, seq_len)),
            )
        } else {
            (input.clone(), None)
        };

        // Use GPU-optimized ternary matmul if available
        #[cfg(feature = "cuda")]
        let output = {
            if crate::kernels::cuda_available()
                && crate::kernels::should_use_gpu(&flat_input, &self.weight)
            {
                crate::kernels::ternary_matmul_gpu(&flat_input, &self.weight)
                    .map_err(|e| candle_core::Error::Msg(e.to_string()))?
            } else {
                // Fallback to standard dequantize + matmul
                let dequant_weight = dequantize_weights(&self.weight, &self.device)
                    .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
                flat_input.matmul(&dequant_weight.t()?)?
            }
        };

        #[cfg(not(feature = "cuda"))]
        let output = {
            let dequant_weight = dequantize_weights(&self.weight, &self.device)
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
            flat_input.matmul(&dequant_weight.t()?)?
        };

        // Reshape back to 3D if needed
        let output = if let Some((batch, seq_len)) = original_shape {
            output.reshape((batch, seq_len, self.out_features()))?
        } else {
            output
        };

        // Add bias
        let output = if let Some(ref bias) = self.bias {
            output.broadcast_add(bias)?
        } else {
            output
        };

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bitlinear_creation() {
        let device = Device::Cpu;
        let config = BitNetConfig::default();

        let weight = Tensor::randn(0.0f32, 1.0, (128, 256), &device).unwrap();
        let layer = BitLinear::from_weight(&weight, None, &config).unwrap();

        assert_eq!(layer.in_features(), 256);
        assert_eq!(layer.out_features(), 128);
    }

    #[test]
    fn test_bitlinear_forward() {
        let device = Device::Cpu;
        let config = BitNetConfig::default().with_group_size(64);

        let weight = Tensor::randn(0.0f32, 1.0, (64, 128), &device).unwrap();
        let layer = BitLinear::from_weight(&weight, None, &config).unwrap();

        let input = Tensor::randn(0.0f32, 1.0, (4, 128), &device).unwrap();
        let output = layer.forward(&input).unwrap();

        assert_eq!(output.shape().dims(), &[4, 64]);
    }

    #[test]
    fn test_bitlinear_forward_quantized() {
        let device = Device::Cpu;
        let config = BitNetConfig::default().with_group_size(64);

        let weight = Tensor::randn(0.0f32, 1.0, (64, 128), &device).unwrap();
        let layer = BitLinear::from_weight(&weight, None, &config).unwrap();

        let input = Tensor::randn(0.0f32, 1.0, (4, 128), &device).unwrap();
        let output = layer.forward_quantized(&input).unwrap();

        assert_eq!(output.shape().dims(), &[4, 64]);
    }

    #[test]
    fn test_bitlinear_with_bias() {
        let device = Device::Cpu;
        let config = BitNetConfig::default().with_group_size(64);

        let weight = Tensor::randn(0.0f32, 1.0, (64, 128), &device).unwrap();
        let bias = Tensor::randn(0.0f32, 1.0, (64,), &device).unwrap();
        let layer = BitLinear::from_weight(&weight, Some(&bias), &config).unwrap();

        let input = Tensor::randn(0.0f32, 1.0, (4, 128), &device).unwrap();
        let output = layer.forward(&input).unwrap();

        assert_eq!(output.shape().dims(), &[4, 64]);
    }

    #[test]
    fn test_bitlinear_3d_input() {
        let device = Device::Cpu;
        let config = BitNetConfig::default().with_group_size(64);

        let weight = Tensor::randn(0.0f32, 1.0, (64, 128), &device).unwrap();
        let layer = BitLinear::from_weight(&weight, None, &config).unwrap();

        // 3D input [batch, seq_len, hidden]
        let input = Tensor::randn(0.0f32, 1.0, (2, 16, 128), &device).unwrap();
        let output = layer.forward(&input).unwrap();

        assert_eq!(output.shape().dims(), &[2, 16, 64]);
    }

    #[test]
    fn test_bitlinear_sparsity() {
        let device = Device::Cpu;
        let config = BitNetConfig::default().with_group_size(64);

        let weight = Tensor::randn(0.0f32, 1.0, (64, 128), &device).unwrap();
        let layer = BitLinear::from_weight(&weight, None, &config).unwrap();

        // Ternary quantization typically results in some sparsity
        let sparsity = layer.sparsity();
        assert!(sparsity >= 0.0 && sparsity <= 1.0);
    }

    #[test]
    fn test_bitlinear_compression() {
        let device = Device::Cpu;
        let config = BitNetConfig::default();

        // Larger weight for meaningful compression measurement
        let weight = Tensor::randn(0.0f32, 1.0, (1024, 4096), &device).unwrap();
        let layer = BitLinear::from_weight(&weight, None, &config).unwrap();

        let ratio = layer.compression_ratio();
        assert!(ratio > 1.0, "should achieve some compression");
    }
}
