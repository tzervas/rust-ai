// SPDX-License-Identifier: MIT
// Copyright 2026 Tyler Zervas

//! GPU-accelerated operations for trit-vsa using CubeCL.
//!
//! This module provides CUDA kernel implementations for all core VSA operations:
//!
//! - **Bind/Unbind**: Ternary composition via modular arithmetic
//! - **Bundle**: Majority voting across multiple vectors
//! - **Similarity**: Dot product and Hamming distance with parallel reductions
//! - **Random**: GPU-parallel xorshift PRNG for random vector generation
//!
//! ## Architecture
//!
//! The GPU implementation uses a two-layer design:
//!
//! 1. **Kernels** ([`kernels`]) - Low-level CubeCL kernel definitions
//! 2. **Ops** ([`ops`]) - High-level wrappers implementing `GpuDispatchable`
//!
//! ## Feature Gate
//!
//! This module requires the `cuda` feature:
//!
//! ```toml
//! [dependencies]
//! trit-vsa = { version = "0.1", features = ["cuda"] }
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! use trit_vsa::gpu::{GpuDispatchable, GpuBind};
//! use trit_vsa::PackedTritVec;
//! use candle_core::Device;
//!
//! let device = Device::cuda_if_available(0)?;
//! let a = PackedTritVec::random(10000);
//! let b = PackedTritVec::random(10000);
//!
//! let result = GpuBind.dispatch(&(a, b), &device)?;
//! ```

pub mod kernels;
pub mod ops;
pub mod traits;

pub use ops::{
    GpuBind, GpuBundle, GpuCosineSimilarity, GpuDotSimilarity, GpuHammingDistance, GpuRandom,
    GpuUnbind, RandomInput,
};
pub use traits::{warn_if_cpu, GpuDispatchable, GpuError, GpuResult};

use crate::{PackedTritVec, Result};
use candle_core::Device;

// =============================================================================
// CONVENIENCE WRAPPER FUNCTIONS FOR DISPATCH.RS
// =============================================================================

/// Convenience wrapper for GPU dot product similarity.
///
/// Automatically dispatches to GPU or CPU based on the provided device.
///
/// # Arguments
///
/// * `a` - First ternary vector
/// * `b` - Second ternary vector
/// * `device` - Target device (CPU or CUDA)
///
/// # Returns
///
/// Dot product similarity as i32 (range: -n to +n where n = vector dimension)
///
/// # Errors
///
/// Returns error if:
/// - Vectors have mismatched dimensions
/// - GPU kernel launch fails
/// - Device allocation fails
pub fn gpu_dot(a: &PackedTritVec, b: &PackedTritVec, device: &Device) -> Result<i32> {
    GpuDotSimilarity
        .dispatch(&(a.clone(), b.clone()), device)
        .map_err(Into::into)
}

/// Convenience wrapper for GPU cosine similarity.
///
/// Automatically dispatches to GPU or CPU based on the provided device.
///
/// # Arguments
///
/// * `a` - First ternary vector
/// * `b` - Second ternary vector
/// * `device` - Target device (CPU or CUDA)
///
/// # Returns
///
/// Cosine similarity as f32 (range: -1.0 to +1.0)
///
/// # Errors
///
/// Returns error if:
/// - Vectors have mismatched dimensions
/// - GPU kernel launch fails
/// - Device allocation fails
pub fn gpu_cosine_similarity(a: &PackedTritVec, b: &PackedTritVec, device: &Device) -> Result<f32> {
    GpuCosineSimilarity
        .dispatch(&(a.clone(), b.clone()), device)
        .map_err(Into::into)
}

/// Convenience wrapper for GPU bind operation.
///
/// Automatically dispatches to GPU or CPU based on the provided device.
///
/// # Arguments
///
/// * `a` - First ternary vector
/// * `b` - Second ternary vector
/// * `device` - Target device (CPU or CUDA)
///
/// # Returns
///
/// Bound vector as PackedTritVec
///
/// # Errors
///
/// Returns error if:
/// - Vectors have mismatched dimensions
/// - GPU kernel launch fails
/// - Device allocation fails
pub fn gpu_bind(a: &PackedTritVec, b: &PackedTritVec, device: &Device) -> Result<PackedTritVec> {
    GpuBind
        .dispatch(&(a.clone(), b.clone()), device)
        .map_err(Into::into)
}

/// Convenience wrapper for GPU unbind operation.
///
/// Automatically dispatches to GPU or CPU based on the provided device.
///
/// # Arguments
///
/// * `a` - First ternary vector
/// * `b` - Second ternary vector
/// * `device` - Target device (CPU or CUDA)
///
/// # Returns
///
/// Unbound vector as PackedTritVec
///
/// # Errors
///
/// Returns error if:
/// - Vectors have mismatched dimensions
/// - GPU kernel launch fails
/// - Device allocation fails
pub fn gpu_unbind(a: &PackedTritVec, b: &PackedTritVec, device: &Device) -> Result<PackedTritVec> {
    GpuUnbind
        .dispatch(&(a.clone(), b.clone()), device)
        .map_err(Into::into)
}

/// Convenience wrapper for GPU bundle operation.
///
/// Automatically dispatches to GPU or CPU based on the provided device.
///
/// # Arguments
///
/// * `vectors` - Slice of ternary vectors to bundle
/// * `device` - Target device (CPU or CUDA)
///
/// # Returns
///
/// Bundled vector as PackedTritVec
///
/// # Errors
///
/// Returns error if:
/// - Input vector list is empty
/// - Vectors have mismatched dimensions
/// - GPU kernel launch fails
/// - Device allocation fails
pub fn gpu_bundle(vectors: &[PackedTritVec], device: &Device) -> Result<PackedTritVec> {
    GpuBundle
        .dispatch(&vectors.to_vec(), device)
        .map_err(Into::into)
}

/// Convenience wrapper for GPU Hamming distance.
///
/// Automatically dispatches to GPU or CPU based on the provided device.
///
/// # Arguments
///
/// * `a` - First ternary vector
/// * `b` - Second ternary vector
/// * `device` - Target device (CPU or CUDA)
///
/// # Returns
///
/// Hamming distance as usize (number of differing positions)
///
/// # Errors
///
/// Returns error if:
/// - Vectors have mismatched dimensions
/// - GPU kernel launch fails
/// - Device allocation fails
pub fn gpu_hamming_distance(
    a: &PackedTritVec,
    b: &PackedTritVec,
    device: &Device,
) -> Result<usize> {
    GpuHammingDistance
        .dispatch(&(a.clone(), b.clone()), device)
        .map_err(Into::into)
}
