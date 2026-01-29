//! Neural network layers for BitNet.
//!
//! This module provides:
//! - `BitLinear`: Drop-in replacement for `nn::Linear` with ternary weights
//! - Straight-Through Estimator for training

mod bitlinear;
mod ste;

pub use bitlinear::BitLinear;
pub use ste::{int8_ste, ste_backward, ste_forward, ternary_ste};
