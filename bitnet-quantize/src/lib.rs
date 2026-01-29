//! Microsoft BitNet b1.58 quantization and inference for Rust.
//!
//! This crate provides an implementation of BitNet, which uses:
//! - **Ternary weights**: {-1, 0, +1} via AbsMean quantization
//! - **INT8 activations**: Per-token AbsMax scaling
//!
//! # Features
//!
//! - `BitLinear`: Drop-in replacement for `nn::Linear`
//! - Efficient ternary weight storage via `trit-vsa`
//! - Straight-Through Estimator (STE) for training
//! - Optional peft-rs adapter integration
//! - Optional GGUF export via qlora-rs
//!
//! # Quick Start
//!
//! ```ignore
//! use bitnet_quantize::{BitLinear, BitNetConfig};
//! use candle_core::{Device, Tensor};
//!
//! let device = Device::Cpu;
//! let config = BitNetConfig::default();
//!
//! // Create BitLinear from existing weights
//! let weight = Tensor::randn(0.0f32, 1.0, (512, 256), &device)?;
//! let layer = BitLinear::from_weight(&weight, None, &config)?;
//!
//! // Forward pass
//! let input = Tensor::randn(0.0f32, 1.0, (4, 256), &device)?;
//! let output = layer.forward(&input)?;
//!
//! println!("Compression ratio: {:.2}x", layer.compression_ratio());
//! ```
//!
//! # Quantization
//!
//! ## Weight Quantization (AbsMean)
//!
//! Weights are quantized using the AbsMean method:
//! ```text
//! scale = mean(|W|)
//! W_q = round(W / scale) clamped to {-1, 0, +1}
//! ```
//!
//! ## Activation Quantization (AbsMax)
//!
//! Activations are quantized to INT8 using per-token AbsMax:
//! ```text
//! scale = max(|X|) / 127
//! X_q = round(X / scale) clamped to [-127, 127]
//! ```
//!
//! # Feature Flags
//!
//! - `default`: CPU-only
//! - `cuda`: Enable CUDA GPU kernels
//! - `peft`: Enable peft-rs adapter integration
//! - `gguf-export`: Enable GGUF export via qlora-rs
//!
//! # References
//!
//! - "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits"
//!   <https://arxiv.org/abs/2402.17764>

#![warn(missing_docs)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::doc_markdown)] // Many technical terms don't need backticks

mod adapter;
mod config;
mod error;
pub mod export;
pub mod kernels;
pub mod layer;
pub mod quantization;

pub use adapter::{BitNetAdapter, BitNetAdapterConfig};
pub use config::BitNetConfig;
pub use error::{BitNetError, Result};
pub use layer::BitLinear;
pub use quantization::{
    dequantize_activations, dequantize_weights, quantize_activations, quantize_weights,
    QuantizedActivations, TernaryWeight,
};

/// Prelude module for convenient imports.
pub mod prelude {
    pub use crate::config::BitNetConfig;
    pub use crate::error::{BitNetError, Result};
    pub use crate::layer::BitLinear;
    pub use crate::quantization::{quantize_activations, quantize_weights};
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use candle_nn::Module;

    #[test]
    fn test_basic_workflow() {
        let device = Device::Cpu;
        let config = BitNetConfig::default().with_group_size(64);

        // Create weight and quantize
        let weight = candle_core::Tensor::randn(0.0f32, 1.0, (64, 128), &device).unwrap();
        let layer = BitLinear::from_weight(&weight, None, &config).unwrap();

        // Forward pass
        let input = candle_core::Tensor::randn(0.0f32, 1.0, (4, 128), &device).unwrap();
        let output = layer.forward(&input).unwrap();

        assert_eq!(output.shape().dims(), &[4, 64]);

        // Check compression
        let ratio = layer.compression_ratio();
        assert!(ratio > 1.0, "should achieve compression");
    }

    #[test]
    fn test_quantization_workflow() {
        let device = Device::Cpu;
        let config = BitNetConfig::default();

        // Quantize weights
        let weight = candle_core::Tensor::randn(0.0f32, 1.0, (64, 128), &device).unwrap();
        let ternary = quantize_weights(&weight, &config).unwrap();

        // Check structure
        assert_eq!(ternary.shape, (64, 128));
        assert!(ternary.sparsity() >= 0.0);
        assert!(ternary.sparsity() <= 1.0);

        // Dequantize
        let restored = dequantize_weights(&ternary, &device).unwrap();
        assert_eq!(restored.shape().dims(), &[64, 128]);
    }

    #[test]
    fn test_activation_quantization() {
        let device = Device::Cpu;
        let config = BitNetConfig::default();

        let activations = candle_core::Tensor::randn(0.0f32, 1.0, (4, 64), &device).unwrap();
        let quantized = quantize_activations(&activations, &config).unwrap();

        assert_eq!(quantized.shape, vec![4, 64]);
        assert_eq!(quantized.scales.len(), 4); // Per-token

        let restored = dequantize_activations(&quantized, &device).unwrap();
        assert_eq!(restored.shape().dims(), &[4, 64]);
    }

    #[test]
    fn test_config_builder() {
        let config = BitNetConfig::new()
            .with_group_size(128)
            .with_activation_bits(4)
            .with_per_token_activation(false)
            .with_rms_norm(false)
            .with_ste(true);

        assert_eq!(config.group_size, 128);
        assert_eq!(config.activation_bits, 4);
        assert!(!config.per_token_activation);
        assert!(!config.use_rms_norm);
        assert!(config.enable_ste);
    }
}
