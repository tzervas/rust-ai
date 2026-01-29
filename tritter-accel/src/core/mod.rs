//! Core Rust API for tritter-accel.
//!
//! This module provides the pure Rust interface to all acceleration features.
//! Python bindings in the `python` module wrap these functions.
//!
//! # Overview
//!
//! - [`ternary`] - Ternary weight packing, unpacking, and matmul
//! - [`quantization`] - BitNet quantization (AbsMean, AbsMax)
//! - [`vsa`] - Vector Symbolic Architecture operations
//! - [`training`] - Training acceleration (gradient compression, mixed precision)
//! - [`inference`] - Inference acceleration (batched ops, device dispatch)
//!
//! # Example
//!
//! ```rust,ignore
//! use tritter_accel::core::{ternary, quantization, vsa};
//! use candle_core::{Device, Tensor};
//!
//! // Quantize weights to ternary
//! let weights = Tensor::randn(0f32, 1f32, (512, 512), &Device::Cpu)?;
//! let ternary_weights = quantization::quantize_absmean(&weights)?;
//!
//! // Pack for efficient storage
//! let packed = ternary::pack(&ternary_weights.values, ternary_weights.scales)?;
//!
//! // Perform ternary matmul
//! let input = Tensor::randn(0f32, 1f32, (1, 512), &Device::Cpu)?;
//! let output = ternary::matmul(&input, &packed)?;
//! ```

pub mod inference;
pub mod quantization;
pub mod ternary;
pub mod training;
pub mod vsa;

// Re-export key types at module level
pub use inference::{InferenceConfig, InferenceEngine};
pub use quantization::{QuantizationResult, QuantizeConfig};
pub use ternary::{PackedTernary, TernaryMatmulConfig};
pub use training::{GradientCompressor, TrainingConfig};
pub use vsa::{VsaConfig, VsaOps};
