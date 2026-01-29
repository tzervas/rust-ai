//! BitNet integration module.
//!
//! Re-exports and wraps bitnet-quantize functionality for Python bindings.
//!
//! The main quantization function `quantize_weights_absmean` in `lib.rs`
//! delegates to `bitnet_quantize::quantize_weights` for the core algorithm.

// Re-export key types for library users
#[allow(unused_imports)]
pub use bitnet_quantize::{BitLinear, BitNetConfig, TernaryWeight};
#[allow(unused_imports)]
pub use bitnet_quantize::{dequantize_weights, quantize_weights};
