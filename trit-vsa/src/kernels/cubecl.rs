// SPDX-License-Identifier: MIT
// Copyright 2026 Tyler Zervas

//! CubeCL/CUDA backend implementation for ternary VSA operations.
//!
//! This module provides GPU-accelerated implementations using CubeCL,
//! which compiles to CUDA for NVIDIA GPUs.
//!
//! ## Performance Characteristics
//!
//! - Best for large vectors (> 4096 dimensions)
//! - Higher latency due to GPU transfer overhead
//! - Massive parallelism for element-wise operations
//!
//! ## Feature Gate
//!
//! Requires the `cuda` feature:
//!
//! ```toml
//! [dependencies]
//! trit-vsa = { version = "0.1", features = ["cuda"] }
//! ```

#![cfg(feature = "cuda")]
#![allow(unsafe_code)]

use crate::gpu::kernels::{
    final_reduction_kernel, grid_size, ternary_bind_kernel, ternary_bundle_small_kernel,
    ternary_dot_kernel, ternary_hamming_kernel, ternary_random_kernel, ternary_unbind_kernel,
    BLOCK_SIZE,
};
use crate::gpu::traits::GpuError;
use crate::kernels::{check_dimensions, RandomConfig, TernaryBackend};
use crate::{PackedTritVec, Result, TernaryError, Trit};

use cubecl::bytes::Bytes;
use cubecl::prelude::*;
use cubecl_cuda::CudaRuntime;

/// CubeCL/CUDA backend for ternary operations.
///
/// This backend uses CubeCL to compile kernels to CUDA for GPU execution.
///
/// # Availability
///
/// The backend performs a runtime check for CUDA availability.
/// Use `is_available()` to check before use.
///
/// # Performance
///
/// For best performance:
/// - Use for vectors with > 4096 dimensions
/// - Batch operations when possible to amortize transfer overhead
/// - Keep data on GPU between operations when possible
#[derive(Debug, Clone)]
pub struct CubeclBackend {
    /// Cached availability status.
    available: bool,
}

impl CubeclBackend {
    /// Create a new CubeCL backend.
    ///
    /// Performs runtime detection of CUDA availability.
    #[must_use]
    pub fn new() -> Self {
        let available = Self::check_cuda_available();
        Self { available }
    }

    /// Check if CUDA is available at runtime.
    fn check_cuda_available() -> bool {
        // Try to create a CUDA client
        // This will fail if CUDA is not available
        std::panic::catch_unwind(|| {
            let _ = CudaRuntime::client(&Default::default());
        })
        .is_ok()
    }

    /// Get a CUDA client for kernel execution.
    fn get_client(&self) -> std::result::Result<ComputeClient<CudaRuntime>, GpuError> {
        if !self.available {
            return Err(GpuError::device_not_available("CUDA"));
        }
        Ok(CudaRuntime::client(&Default::default()))
    }
}

impl Default for CubeclBackend {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// CONVERSION HELPERS
// =============================================================================

/// Convert PackedTritVec to encoded i32 array for GPU processing.
///
/// Encoding: -1 -> 0, 0 -> 1, +1 -> 2
fn packed_to_encoded(vec: &PackedTritVec) -> Vec<i32> {
    let mut encoded = Vec::with_capacity(vec.len());
    for i in 0..vec.len() {
        let trit = vec.get(i);
        encoded.push(match trit.value() {
            -1 => 0,
            0 => 1,
            1 => 2,
            _ => unreachable!(),
        });
    }
    encoded
}

/// Convert encoded i32 array back to PackedTritVec.
///
/// Decoding: 0 -> -1, 1 -> 0, 2 -> +1
fn encoded_to_packed(encoded: &[i32], len: usize) -> PackedTritVec {
    let mut result = PackedTritVec::new(len);
    for (i, &val) in encoded.iter().take(len).enumerate() {
        let trit = match val {
            0 => Trit::N,
            1 => Trit::Z,
            2 => Trit::P,
            _ => Trit::Z, // Fallback for invalid values
        };
        result.set(i, trit);
    }
    result
}

// =============================================================================
// TRAIT IMPLEMENTATION
// =============================================================================

impl TernaryBackend for CubeclBackend {
    fn name(&self) -> &'static str {
        "cubecl-cuda"
    }

    fn is_available(&self) -> bool {
        self.available
    }

    fn bind(&self, a: &PackedTritVec, b: &PackedTritVec) -> Result<PackedTritVec> {
        check_dimensions(a, b)?;

        if a.is_empty() {
            return Ok(PackedTritVec::new(0));
        }

        let len = a.len() as u32;
        let client = self.get_client()?;

        let a_encoded = packed_to_encoded(a);
        let b_encoded = packed_to_encoded(b);

        let a_handle = client.create(Bytes::from_bytes_vec(i32::as_bytes(&a_encoded).to_vec()));
        let b_handle = client.create(Bytes::from_bytes_vec(i32::as_bytes(&b_encoded).to_vec()));
        let out_handle = client.empty(a_encoded.len() * std::mem::size_of::<i32>());

        let cube_count = CubeCount::Static(grid_size(len, BLOCK_SIZE), 1, 1);
        let cube_dim = CubeDim::new(&client, BLOCK_SIZE as usize);

        // SAFETY: Handles are valid and properly sized
        unsafe {
            ternary_bind_kernel::launch::<i32, CudaRuntime>(
                &client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts::<i32>(&a_handle, len as usize, 1),
                ArrayArg::from_raw_parts::<i32>(&b_handle, len as usize, 1),
                ArrayArg::from_raw_parts::<i32>(&out_handle, len as usize, 1),
                len,
            )
            .map_err(|e| GpuError::kernel(format!("bind kernel launch failed: {e}")))?;
        }

        let out_bytes = client.read_one(out_handle);
        let out_encoded: Vec<i32> = i32::from_bytes(&out_bytes).to_vec();

        Ok(encoded_to_packed(&out_encoded, a.len()))
    }

    fn unbind(&self, a: &PackedTritVec, b: &PackedTritVec) -> Result<PackedTritVec> {
        check_dimensions(a, b)?;

        if a.is_empty() {
            return Ok(PackedTritVec::new(0));
        }

        let len = a.len() as u32;
        let client = self.get_client()?;

        let a_encoded = packed_to_encoded(a);
        let b_encoded = packed_to_encoded(b);

        let a_handle = client.create(Bytes::from_bytes_vec(i32::as_bytes(&a_encoded).to_vec()));
        let b_handle = client.create(Bytes::from_bytes_vec(i32::as_bytes(&b_encoded).to_vec()));
        let out_handle = client.empty(a_encoded.len() * std::mem::size_of::<i32>());

        let cube_count = CubeCount::Static(grid_size(len, BLOCK_SIZE), 1, 1);
        let cube_dim = CubeDim::new(&client, BLOCK_SIZE as usize);

        // SAFETY: Handles are valid and properly sized
        unsafe {
            ternary_unbind_kernel::launch::<i32, CudaRuntime>(
                &client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts::<i32>(&a_handle, len as usize, 1),
                ArrayArg::from_raw_parts::<i32>(&b_handle, len as usize, 1),
                ArrayArg::from_raw_parts::<i32>(&out_handle, len as usize, 1),
                len,
            )
            .map_err(|e| GpuError::kernel(format!("unbind kernel launch failed: {e}")))?;
        }

        let out_bytes = client.read_one(out_handle);
        let out_encoded: Vec<i32> = i32::from_bytes(&out_bytes).to_vec();

        Ok(encoded_to_packed(&out_encoded, a.len()))
    }

    fn bundle(&self, vectors: &[&PackedTritVec]) -> Result<PackedTritVec> {
        if vectors.is_empty() {
            return Err(TernaryError::EmptyVector);
        }

        let dim = vectors[0].len();
        for v in vectors.iter().skip(1) {
            if v.len() != dim {
                return Err(TernaryError::DimensionMismatch {
                    expected: dim,
                    actual: v.len(),
                });
            }
        }

        if dim == 0 {
            return Ok(PackedTritVec::new(0));
        }

        let num_vectors = vectors.len() as u32;
        let dim_u32 = dim as u32;
        let client = self.get_client()?;

        // Flatten all vectors into one array [num_vectors, dim]
        let mut flattened = Vec::with_capacity(vectors.len() * dim);
        for vec in vectors {
            flattened.extend(packed_to_encoded(vec));
        }

        let vectors_handle =
            client.create(Bytes::from_bytes_vec(i32::as_bytes(&flattened).to_vec()));
        let out_handle = client.empty(dim * std::mem::size_of::<i32>());

        let cube_count = CubeCount::Static(grid_size(dim_u32, BLOCK_SIZE), 1, 1);
        let cube_dim = CubeDim::new(&client, BLOCK_SIZE as usize);

        // SAFETY: Handles are valid and properly sized
        unsafe {
            ternary_bundle_small_kernel::launch::<i32, CudaRuntime>(
                &client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts::<i32>(&vectors_handle, flattened.len(), 1),
                ArrayArg::from_raw_parts::<i32>(&out_handle, dim, 1),
                num_vectors,
                dim_u32,
            )
            .map_err(|e| GpuError::kernel(format!("bundle kernel launch failed: {e}")))?;
        }

        let out_bytes = client.read_one(out_handle);
        let out_encoded: Vec<i32> = i32::from_bytes(&out_bytes).to_vec();

        Ok(encoded_to_packed(&out_encoded, dim))
    }

    fn dot_similarity(&self, a: &PackedTritVec, b: &PackedTritVec) -> Result<i32> {
        check_dimensions(a, b)?;

        if a.is_empty() {
            return Ok(0);
        }

        let len = a.len() as u32;
        let client = self.get_client()?;

        let a_encoded = packed_to_encoded(a);
        let b_encoded = packed_to_encoded(b);

        let a_handle = client.create(Bytes::from_bytes_vec(i32::as_bytes(&a_encoded).to_vec()));
        let b_handle = client.create(Bytes::from_bytes_vec(i32::as_bytes(&b_encoded).to_vec()));

        let num_blocks = grid_size(len, BLOCK_SIZE);
        let partial_sums_handle = client.empty(num_blocks as usize * std::mem::size_of::<u32>());

        let cube_count = CubeCount::Static(num_blocks, 1, 1);
        let cube_dim = CubeDim::new(&client, BLOCK_SIZE as usize);

        // SAFETY: Handles are valid and properly sized
        unsafe {
            ternary_dot_kernel::launch::<CudaRuntime>(
                &client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts::<u32>(&a_handle, len as usize, 1),
                ArrayArg::from_raw_parts::<u32>(&b_handle, len as usize, 1),
                ArrayArg::from_raw_parts::<u32>(&partial_sums_handle, num_blocks as usize, 1),
                len,
            )
            .map_err(|e| GpuError::kernel(format!("dot kernel launch failed: {e}")))?;
        }

        let result_handle = client.empty(std::mem::size_of::<u32>());

        // SAFETY: Handles are valid and properly sized
        unsafe {
            final_reduction_kernel::launch::<CudaRuntime>(
                &client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new(&client, BLOCK_SIZE as usize),
                ArrayArg::from_raw_parts::<u32>(&partial_sums_handle, num_blocks as usize, 1),
                ArrayArg::from_raw_parts::<u32>(&result_handle, 1, 1),
                num_blocks,
            )
            .map_err(|e| GpuError::kernel(format!("final_reduction kernel launch failed: {e}")))?;
        }

        let result_bytes = client.read_one(result_handle);
        let result: Vec<u32> = u32::from_bytes(&result_bytes).to_vec();

        Ok(result.first().copied().unwrap_or(0) as i32)
    }

    fn cosine_similarity(&self, a: &PackedTritVec, b: &PackedTritVec) -> Result<f32> {
        let dot = self.dot_similarity(a, b)?;

        // Norms computed on CPU (cheap for ternary vectors)
        let norm_a = (a.count_nonzero() as f32).sqrt();
        let norm_b = (b.count_nonzero() as f32).sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            return Ok(0.0);
        }

        Ok(dot as f32 / (norm_a * norm_b))
    }

    fn hamming_distance(&self, a: &PackedTritVec, b: &PackedTritVec) -> Result<usize> {
        check_dimensions(a, b)?;

        if a.is_empty() {
            return Ok(0);
        }

        let len = a.len() as u32;
        let client = self.get_client()?;

        let a_encoded = packed_to_encoded(a);
        let b_encoded = packed_to_encoded(b);

        let a_handle = client.create(Bytes::from_bytes_vec(i32::as_bytes(&a_encoded).to_vec()));
        let b_handle = client.create(Bytes::from_bytes_vec(i32::as_bytes(&b_encoded).to_vec()));

        let num_blocks = grid_size(len, BLOCK_SIZE);
        let partial_counts_handle = client.empty(num_blocks as usize * std::mem::size_of::<u32>());

        let cube_count = CubeCount::Static(num_blocks, 1, 1);
        let cube_dim = CubeDim::new(&client, BLOCK_SIZE as usize);

        // SAFETY: Handles are valid and properly sized
        unsafe {
            ternary_hamming_kernel::launch::<CudaRuntime>(
                &client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts::<u32>(&a_handle, len as usize, 1),
                ArrayArg::from_raw_parts::<u32>(&b_handle, len as usize, 1),
                ArrayArg::from_raw_parts::<u32>(&partial_counts_handle, num_blocks as usize, 1),
                len,
            )
            .map_err(|e| GpuError::kernel(format!("hamming kernel launch failed: {e}")))?;
        }

        let result_handle = client.empty(std::mem::size_of::<u32>());

        // SAFETY: Handles are valid and properly sized
        unsafe {
            final_reduction_kernel::launch::<CudaRuntime>(
                &client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new(&client, BLOCK_SIZE as usize),
                ArrayArg::from_raw_parts::<u32>(&partial_counts_handle, num_blocks as usize, 1),
                ArrayArg::from_raw_parts::<u32>(&result_handle, 1, 1),
                num_blocks,
            )
            .map_err(|e| GpuError::kernel(format!("final_reduction kernel launch failed: {e}")))?;
        }

        let result_bytes = client.read_one(result_handle);
        let result: Vec<u32> = u32::from_bytes(&result_bytes).to_vec();

        Ok(result.first().copied().unwrap_or(0) as usize)
    }

    fn random(&self, config: &RandomConfig) -> Result<PackedTritVec> {
        if config.dim == 0 {
            return Ok(PackedTritVec::new(0));
        }

        let len = config.dim as u32;
        let client = self.get_client()?;

        // Initialize seeds on CPU (could also be a kernel)
        let mut seeds = Vec::with_capacity(config.dim);
        for i in 0..config.dim {
            // Golden ratio hash mixing for unique seeds
            let base = config.seed as u32;
            let mixed = base.wrapping_add((i as u32).wrapping_mul(2_654_435_769));
            let mixed2 = mixed ^ (mixed >> 16);
            let mixed3 = mixed2.wrapping_mul(2_246_822_519);
            let mixed4 = mixed3 ^ (mixed3 >> 13);
            let mixed5 = mixed4.wrapping_mul(3_266_489_917);
            seeds.push(mixed5 ^ (mixed5 >> 16));
        }

        let seeds_handle = client.create(Bytes::from_bytes_vec(u32::as_bytes(&seeds).to_vec()));
        let out_handle = client.empty(config.dim * std::mem::size_of::<u32>());

        let cube_count = CubeCount::Static(grid_size(len, BLOCK_SIZE), 1, 1);
        let cube_dim = CubeDim::new(&client, BLOCK_SIZE as usize);

        // SAFETY: Handles are valid and properly sized
        unsafe {
            ternary_random_kernel::launch::<CudaRuntime>(
                &client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts::<u32>(&out_handle, config.dim, 1),
                ArrayArg::from_raw_parts::<u32>(&seeds_handle, config.dim, 1),
                len,
            )
            .map_err(|e| GpuError::kernel(format!("random kernel launch failed: {e}")))?;
        }

        let out_bytes = client.read_one(out_handle);
        let out_u32: Vec<u32> = u32::from_bytes(&out_bytes).to_vec();

        // Convert u32 (0, 1, 2) to i32 for encoding
        let out_encoded: Vec<i32> = out_u32.iter().map(|&v| v as i32).collect();

        Ok(encoded_to_packed(&out_encoded, config.dim))
    }

    fn negate(&self, a: &PackedTritVec) -> Result<PackedTritVec> {
        // Negation is simple enough to do on CPU
        // GPU overhead would be more than the computation
        Ok(a.negated())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_vector(values: &[i8]) -> PackedTritVec {
        let mut vec = PackedTritVec::new(values.len());
        for (i, &v) in values.iter().enumerate() {
            let trit = match v {
                -1 => Trit::N,
                0 => Trit::Z,
                1 => Trit::P,
                _ => panic!("Invalid trit value"),
            };
            vec.set(i, trit);
        }
        vec
    }

    #[test]
    fn test_cubecl_backend_creation() {
        let backend = CubeclBackend::new();
        // Backend creation should always succeed, availability depends on system
        assert_eq!(backend.name(), "cubecl-cuda");
    }

    #[test]
    fn test_encoded_conversion_roundtrip() {
        let original = make_test_vector(&[1, -1, 0, 1, -1, 0, 1]);
        let encoded = packed_to_encoded(&original);
        let recovered = encoded_to_packed(&encoded, original.len());

        for i in 0..original.len() {
            assert_eq!(
                original.get(i),
                recovered.get(i),
                "mismatch at position {i}"
            );
        }
    }

    // GPU tests are only run if CUDA is available
    #[test]
    #[ignore = "requires CUDA"]
    fn test_cubecl_bind_unbind() {
        let backend = CubeclBackend::new();
        if !backend.is_available() {
            return;
        }

        let a = make_test_vector(&[1, -1, 0, 1, -1, 0, 1, -1]);
        let b = make_test_vector(&[-1, 1, 0, -1, 1, 0, -1, 1]);

        let bound = backend.bind(&a, &b).unwrap();
        let recovered = backend.unbind(&bound, &b).unwrap();

        for i in 0..a.len() {
            assert_eq!(recovered.get(i), a.get(i), "mismatch at position {i}");
        }
    }

    #[test]
    #[ignore = "requires CUDA"]
    fn test_cubecl_dot_similarity() {
        let backend = CubeclBackend::new();
        if !backend.is_available() {
            return;
        }

        let a = make_test_vector(&[1, 1, -1, -1]);
        let dot = backend.dot_similarity(&a, &a).unwrap();
        assert_eq!(dot, 4);
    }

    #[test]
    #[ignore = "requires CUDA"]
    fn test_cubecl_large_vectors() {
        let backend = CubeclBackend::new();
        if !backend.is_available() {
            return;
        }

        // Test with large vectors to verify GPU performance
        let r1 = backend.random(&RandomConfig::new(10000, 1)).unwrap();
        let r2 = backend.random(&RandomConfig::new(10000, 2)).unwrap();

        let _ = backend.bind(&r1, &r2).unwrap();
        let _ = backend.dot_similarity(&r1, &r2).unwrap();
        let _ = backend.hamming_distance(&r1, &r2).unwrap();
    }
}
