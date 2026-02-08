//! CubeCL GPU kernel implementations.
//!
//! This module contains production GPU kernels extracted and adapted from unsloth-rs.

#[cfg(feature = "cuda")]
pub mod attention;

#[cfg(feature = "cuda")]
pub mod ternary;

#[cfg(feature = "cuda")]
pub use attention::*;

#[cfg(feature = "cuda")]
pub use ternary::*;
