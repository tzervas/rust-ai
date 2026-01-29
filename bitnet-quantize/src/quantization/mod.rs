//! Quantization modules for BitNet.
//!
//! This module provides:
//! - **Weight quantization**: AbsMean -> {-1, 0, +1}
//! - **Activation quantization**: Per-token AbsMax -> INT8

mod activation;
mod weight;

pub use activation::{
    dequantize_activations, quantize_activations, quantize_ste, QuantizedActivations,
};
pub use weight::{dequantize_weights, quantize_weights, TernaryWeight};
