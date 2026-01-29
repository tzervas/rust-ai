// SPDX-License-Identifier: MIT
// Copyright 2026 Tyler Zervas

#![allow(unsafe_code)]

//! High-level GPU operation wrappers for ternary VSA.
//!
//! This module provides ergonomic interfaces to the CubeCL kernels,
//! implementing the local `GpuDispatchable` trait.
//!
//! ## Available Operations
//!
//! | Operation | Description | Complexity |
//! |-----------|-------------|------------|
//! | [`GpuBind`] | Ternary composition | O(n) |
//! | [`GpuUnbind`] | Inverse composition | O(n) |
//! | [`GpuBundle`] | Majority voting | O(n×k) |
//! | [`GpuDotSimilarity`] | Dot product similarity | O(n) |
//! | [`GpuHammingDistance`] | Hamming distance | O(n) |
//! | [`GpuRandom`] | Random vector generation | O(n) |
//!
//! ## Usage Pattern
//!
//! All operations follow the `GpuDispatchable` pattern:
//!
//! ```rust,ignore
//! use trit_vsa::gpu::{GpuDispatchable, ops::GpuBind};
//! use candle_core::Device;
//!
//! let device = Device::cuda_if_available(0)?;
//! let result = GpuBind.dispatch(&(vec_a, vec_b), &device)?;
//! ```

use crate::vsa;
use crate::PackedTritVec;
use candle_core::Device;
use cubecl::prelude::*;
use cubecl_cuda::CudaRuntime;

use super::traits::{warn_if_cpu, GpuDispatchable, GpuError, GpuResult};

use super::kernels::{
    final_reduction_kernel, grid_size, ternary_bind_kernel, ternary_bundle_small_kernel,
    ternary_dot_kernel, ternary_hamming_kernel, ternary_random_kernel, ternary_unbind_kernel,
    BLOCK_SIZE,
};

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/// Check that two vectors have the same dimensions.
fn check_dims(a: &PackedTritVec, b: &PackedTritVec) -> GpuResult<()> {
    if a.len() != b.len() {
        return Err(GpuError::dim_mismatch(format!(
            "vectors must have same dimensions: {} vs {}",
            a.len(),
            b.len()
        )));
    }
    Ok(())
}

/// Convert PackedTritVec to encoded i32 array for GPU processing.
///
/// Encoding: -1 → 0, 0 → 1, +1 → 2
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
/// Decoding: 0 → -1, 1 → 0, 2 → +1
fn encoded_to_packed(encoded: &[i32], len: usize) -> PackedTritVec {
    use crate::Trit;

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

/// Get CUDA client for kernel execution.
fn get_cuda_client() -> GpuResult<ComputeClient<CudaRuntime>> {
    Ok(cubecl_cuda::CudaRuntime::client(&Default::default()))
}

// =============================================================================
// GPU BIND OPERATION
// =============================================================================

/// GPU-accelerated ternary bind operation.
///
/// Bind is the composition operation in VSA, analogous to XOR in binary systems.
/// It creates associations between vectors.
///
/// # Properties
///
/// - Commutative: `bind(a, b) == bind(b, a)`
/// - Associative: `bind(bind(a, b), c) == bind(a, bind(b, c))`
/// - Self-inverse with unbind: `unbind(bind(a, b), b) == a`
///
/// # Example
///
/// ```rust,ignore
/// use trit_vsa::gpu::{GpuDispatchable, ops::GpuBind};
/// use trit_vsa::PackedTritVec;
/// use candle_core::Device;
///
/// let device = Device::cuda_if_available(0)?;
/// let a = PackedTritVec::random(10000);
/// let b = PackedTritVec::random(10000);
///
/// let bound = GpuBind.dispatch(&(a, b), &device)?;
/// ```
pub struct GpuBind;

impl GpuDispatchable for GpuBind {
    type Input = (PackedTritVec, PackedTritVec);
    type Output = PackedTritVec;

    fn dispatch_gpu(&self, input: &Self::Input, _device: &Device) -> GpuResult<Self::Output> {
        let (a, b) = input;

        if a.len() != b.len() {
            return Err(GpuError::dim_mismatch(format!(
                "bind requires equal dimensions: {} vs {}",
                a.len(),
                b.len()
            )));
        }

        if a.is_empty() {
            return Ok(PackedTritVec::new(0));
        }

        let len = a.len() as u32;
        let client = get_cuda_client()?;

        // Convert to encoded representation
        let a_encoded = packed_to_encoded(a);
        let b_encoded = packed_to_encoded(b);

        // Create GPU buffers
        let a_handle = client.create(cubecl::bytes::Bytes::from_bytes_vec(
            i32::as_bytes(&a_encoded).to_vec(),
        ));
        let b_handle = client.create(cubecl::bytes::Bytes::from_bytes_vec(
            i32::as_bytes(&b_encoded).to_vec(),
        ));
        let out_handle = client.empty(a_encoded.len() * std::mem::size_of::<i32>());

        // Launch kernel
        let cube_count = CubeCount::Static(grid_size(len, BLOCK_SIZE), 1, 1);
        let cube_dim = CubeDim::new(&client, BLOCK_SIZE as usize);

        // SAFETY: Handles are valid and properly sized for the kernel operation
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

        // Read back results
        let out_bytes = client.read_one(out_handle);
        let out_encoded: Vec<i32> = i32::from_bytes(&out_bytes).to_vec();

        Ok(encoded_to_packed(&out_encoded, a.len()))
    }

    fn dispatch_cpu(&self, input: &Self::Input, device: &Device) -> GpuResult<Self::Output> {
        warn_if_cpu(device, "trit-vsa");
        let (a, b) = input;
        check_dims(a, b)?;
        Ok(vsa::bind(a, b))
    }
}

// =============================================================================
// GPU UNBIND OPERATION
// =============================================================================

/// GPU-accelerated ternary unbind operation.
///
/// Unbind is the inverse of bind, used to recover associated vectors.
///
/// # Properties
///
/// - `unbind(bind(a, b), b) == a`
/// - `unbind(a, b) == bind(a, negate(b))`
pub struct GpuUnbind;

impl GpuDispatchable for GpuUnbind {
    type Input = (PackedTritVec, PackedTritVec);
    type Output = PackedTritVec;

    fn dispatch_gpu(&self, input: &Self::Input, _device: &Device) -> GpuResult<Self::Output> {
        let (a, b) = input;

        if a.len() != b.len() {
            return Err(GpuError::dim_mismatch(format!(
                "unbind requires equal dimensions: {} vs {}",
                a.len(),
                b.len()
            )));
        }

        if a.is_empty() {
            return Ok(PackedTritVec::new(0));
        }

        let len = a.len() as u32;
        let client = get_cuda_client()?;

        let a_encoded = packed_to_encoded(a);
        let b_encoded = packed_to_encoded(b);

        let a_handle = client.create(cubecl::bytes::Bytes::from_bytes_vec(
            i32::as_bytes(&a_encoded).to_vec(),
        ));
        let b_handle = client.create(cubecl::bytes::Bytes::from_bytes_vec(
            i32::as_bytes(&b_encoded).to_vec(),
        ));
        let out_handle = client.empty(a_encoded.len() * std::mem::size_of::<i32>());

        let cube_count = CubeCount::Static(grid_size(len, BLOCK_SIZE), 1, 1);
        let cube_dim = CubeDim::new(&client, BLOCK_SIZE as usize);

        // SAFETY: Handles are valid and properly sized for the kernel operation
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

    fn dispatch_cpu(&self, input: &Self::Input, device: &Device) -> GpuResult<Self::Output> {
        warn_if_cpu(device, "trit-vsa");
        let (a, b) = input;
        check_dims(a, b)?;
        Ok(vsa::unbind(a, b))
    }
}

// =============================================================================
// GPU BUNDLE OPERATION
// =============================================================================

/// GPU-accelerated ternary bundle (majority voting) operation.
///
/// Bundle combines multiple vectors into one that is similar to all inputs.
/// This is the "addition" operation in hyperdimensional computing.
///
/// # Algorithm
///
/// For each dimension, counts votes for each trit value and selects majority.
/// Ties are resolved to zero (neutral element).
///
/// # Complexity
///
/// O(n × k) where n is dimension and k is number of vectors.
pub struct GpuBundle;

impl GpuDispatchable for GpuBundle {
    type Input = Vec<PackedTritVec>;
    type Output = PackedTritVec;

    fn dispatch_gpu(&self, input: &Self::Input, _device: &Device) -> GpuResult<Self::Output> {
        if input.is_empty() {
            return Err(GpuError::invalid_config("cannot bundle empty vector list"));
        }

        let dim = input[0].len();
        for v in input.iter().skip(1) {
            if v.len() != dim {
                return Err(GpuError::dim_mismatch(format!(
                    "all vectors must have same dimensions: expected {}, got {}",
                    dim,
                    v.len()
                )));
            }
        }

        if dim == 0 {
            return Ok(PackedTritVec::new(0));
        }

        let num_vectors = input.len() as u32;
        let dim_u32 = dim as u32;
        let client = get_cuda_client()?;

        // Flatten all vectors into one array [num_vectors, dim]
        let mut flattened = Vec::with_capacity(input.len() * dim);
        for vec in input {
            flattened.extend(packed_to_encoded(vec));
        }

        let vectors_handle = client.create(cubecl::bytes::Bytes::from_bytes_vec(
            i32::as_bytes(&flattened).to_vec(),
        ));
        let out_handle = client.empty(dim * std::mem::size_of::<i32>());

        let cube_count = CubeCount::Static(grid_size(dim_u32, BLOCK_SIZE), 1, 1);
        let cube_dim = CubeDim::new(&client, BLOCK_SIZE as usize);

        // Use small bundle kernel for efficient single-pass processing
        // SAFETY: Handles are valid and properly sized for the kernel operation
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
            .map_err(|e| GpuError::kernel(format!("bundle_small kernel launch failed: {e}")))?;
        }

        let out_bytes = client.read_one(out_handle);
        let out_encoded: Vec<i32> = i32::from_bytes(&out_bytes).to_vec();

        Ok(encoded_to_packed(&out_encoded, dim))
    }

    fn dispatch_cpu(&self, input: &Self::Input, device: &Device) -> GpuResult<Self::Output> {
        warn_if_cpu(device, "trit-vsa");

        if input.is_empty() {
            return Err(GpuError::invalid_config("cannot bundle empty vector list"));
        }

        let refs: Vec<&PackedTritVec> = input.iter().collect();
        Ok(vsa::bundle_many(&refs))
    }
}

// =============================================================================
// GPU DOT SIMILARITY OPERATION
// =============================================================================

/// GPU-accelerated dot product similarity.
///
/// Computes the dot product of two ternary vectors, which measures similarity.
/// The result ranges from -n to +n where n is the vector dimension.
///
/// # Formula
///
/// For balanced ternary vectors with values in {-1, 0, +1}:
/// `dot(a, b) = Σ(a[i] × b[i])`
///
/// # Interpretation
///
/// - Positive: vectors are similar
/// - Zero: vectors are orthogonal
/// - Negative: vectors are anti-similar
pub struct GpuDotSimilarity;

impl GpuDispatchable for GpuDotSimilarity {
    type Input = (PackedTritVec, PackedTritVec);
    type Output = i32;

    fn dispatch_gpu(&self, input: &Self::Input, _device: &Device) -> GpuResult<Self::Output> {
        let (a, b) = input;

        if a.len() != b.len() {
            return Err(GpuError::dim_mismatch(format!(
                "dot requires equal dimensions: {} vs {}",
                a.len(),
                b.len()
            )));
        }

        if a.is_empty() {
            return Ok(0);
        }

        let len = a.len() as u32;
        let client = get_cuda_client()?;

        let a_encoded = packed_to_encoded(a);
        let b_encoded = packed_to_encoded(b);

        let a_handle = client.create(cubecl::bytes::Bytes::from_bytes_vec(
            i32::as_bytes(&a_encoded).to_vec(),
        ));
        let b_handle = client.create(cubecl::bytes::Bytes::from_bytes_vec(
            i32::as_bytes(&b_encoded).to_vec(),
        ));

        // Calculate number of blocks for reduction
        let num_blocks = grid_size(len, BLOCK_SIZE);
        let partial_sums_handle = client.empty(num_blocks as usize * std::mem::size_of::<i32>());

        let cube_count = CubeCount::Static(num_blocks, 1, 1);
        let cube_dim = CubeDim::new(&client, BLOCK_SIZE as usize);

        // First pass: compute partial sums per block
        // SAFETY: Handles are valid and properly sized for the kernel operation
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

        // Final reduction: sum all partial results
        let result_handle = client.empty(std::mem::size_of::<u32>());

        // SAFETY: Handles are valid and properly sized for the kernel operation
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

    fn dispatch_cpu(&self, input: &Self::Input, device: &Device) -> GpuResult<Self::Output> {
        warn_if_cpu(device, "trit-vsa");
        let (a, b) = input;
        check_dims(a, b)?;
        Ok(a.dot(b))
    }
}

// =============================================================================
// GPU HAMMING DISTANCE OPERATION
// =============================================================================

/// GPU-accelerated Hamming distance computation.
///
/// Counts the number of positions where two vectors differ.
///
/// # Range
///
/// Returns a value in [0, n] where n is the vector dimension.
/// - 0: vectors are identical
/// - n: vectors differ at every position
pub struct GpuHammingDistance;

impl GpuDispatchable for GpuHammingDistance {
    type Input = (PackedTritVec, PackedTritVec);
    type Output = usize;

    fn dispatch_gpu(&self, input: &Self::Input, _device: &Device) -> GpuResult<Self::Output> {
        let (a, b) = input;

        if a.len() != b.len() {
            return Err(GpuError::dim_mismatch(format!(
                "hamming requires equal dimensions: {} vs {}",
                a.len(),
                b.len()
            )));
        }

        if a.is_empty() {
            return Ok(0);
        }

        let len = a.len() as u32;
        let client = get_cuda_client()?;

        let a_encoded = packed_to_encoded(a);
        let b_encoded = packed_to_encoded(b);

        let a_handle = client.create(cubecl::bytes::Bytes::from_bytes_vec(
            i32::as_bytes(&a_encoded).to_vec(),
        ));
        let b_handle = client.create(cubecl::bytes::Bytes::from_bytes_vec(
            i32::as_bytes(&b_encoded).to_vec(),
        ));

        let num_blocks = grid_size(len, BLOCK_SIZE);
        let partial_counts_handle = client.empty(num_blocks as usize * std::mem::size_of::<i32>());

        let cube_count = CubeCount::Static(num_blocks, 1, 1);
        let cube_dim = CubeDim::new(&client, BLOCK_SIZE as usize);

        // SAFETY: Handles are valid and properly sized for the kernel operation
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

        // SAFETY: Handles are valid and properly sized for the kernel operation
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

    fn dispatch_cpu(&self, input: &Self::Input, device: &Device) -> GpuResult<Self::Output> {
        warn_if_cpu(device, "trit-vsa");
        let (a, b) = input;
        check_dims(a, b)?;
        Ok(vsa::hamming_distance(a, b))
    }
}

// =============================================================================
// GPU RANDOM GENERATION
// =============================================================================

/// GPU-accelerated random ternary vector generation.
///
/// Generates random vectors using parallel xorshift32 PRNG on the GPU.
/// Each element gets a unique seed derived from the base seed and position.
///
/// # Algorithm
///
/// Uses xorshift32 with golden ratio mixing for seed initialization.
/// The distribution is uniform across {-1, 0, +1}.
pub struct GpuRandom;

/// Input for random vector generation.
#[derive(Clone, Debug)]
pub struct RandomInput {
    /// Vector dimension.
    pub dim: usize,
    /// Random seed.
    pub seed: u32,
}

impl RandomInput {
    /// Create a new random input configuration.
    #[must_use]
    pub fn new(dim: usize, seed: u32) -> Self {
        Self { dim, seed }
    }
}

impl GpuDispatchable for GpuRandom {
    type Input = RandomInput;
    type Output = PackedTritVec;

    fn dispatch_gpu(&self, input: &Self::Input, _device: &Device) -> GpuResult<Self::Output> {
        if input.dim == 0 {
            return Ok(PackedTritVec::new(0));
        }

        let len = input.dim as u32;
        let client = get_cuda_client()?;

        // Initialize seeds on CPU (could also be a kernel)
        let mut seeds = Vec::with_capacity(input.dim);
        for i in 0..input.dim {
            // Golden ratio hash mixing for unique seeds
            let mixed = input
                .seed
                .wrapping_add((i as u32).wrapping_mul(2_654_435_769));
            let mixed2 = mixed ^ (mixed >> 16);
            let mixed3 = mixed2.wrapping_mul(2_246_822_519);
            let mixed4 = mixed3 ^ (mixed3 >> 13);
            let mixed5 = mixed4.wrapping_mul(3_266_489_917);
            seeds.push(mixed5 ^ (mixed5 >> 16));
        }

        let seeds_handle = client.create(cubecl::bytes::Bytes::from_bytes_vec(
            u32::as_bytes(&seeds).to_vec(),
        ));
        let out_handle = client.empty(input.dim * std::mem::size_of::<u32>());

        let cube_count = CubeCount::Static(grid_size(len, BLOCK_SIZE), 1, 1);
        let cube_dim = CubeDim::new(&client, BLOCK_SIZE as usize);

        // SAFETY: Handles are valid and properly sized for the kernel operation
        unsafe {
            ternary_random_kernel::launch::<CudaRuntime>(
                &client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts::<u32>(&out_handle, input.dim, 1),
                ArrayArg::from_raw_parts::<u32>(&seeds_handle, input.dim, 1),
                len,
            )
            .map_err(|e| GpuError::kernel(format!("random kernel launch failed: {e}")))?;
        }

        let out_bytes = client.read_one(out_handle);
        let out_u32: Vec<u32> = u32::from_bytes(&out_bytes).to_vec();

        // Convert u32 (0, 1, 2) to i32 for encoding
        let out_encoded: Vec<i32> = out_u32.iter().map(|&v| v as i32).collect();

        Ok(encoded_to_packed(&out_encoded, input.dim))
    }

    fn dispatch_cpu(&self, input: &Self::Input, device: &Device) -> GpuResult<Self::Output> {
        use crate::Trit;

        warn_if_cpu(device, "trit-vsa");

        // Simple CPU random generation
        let mut result = PackedTritVec::new(input.dim);
        let mut state = input.seed;

        for i in 0..input.dim {
            // Mix seed with position
            let mut s = state.wrapping_add((i as u32).wrapping_mul(2_654_435_769));

            // Xorshift32
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;

            let trit = match s % 3 {
                0 => Trit::N,
                1 => Trit::Z,
                _ => Trit::P,
            };
            result.set(i, trit);

            // Update state
            state = state.wrapping_add(s);
        }

        Ok(result)
    }
}

// =============================================================================
// COSINE SIMILARITY (DERIVED)
// =============================================================================

/// GPU-accelerated cosine similarity computation.
///
/// Cosine similarity is derived from dot product:
/// `cos(a, b) = dot(a, b) / (||a|| × ||b||)`
///
/// For ternary vectors, norms are `sqrt(count_nonzero)`.
pub struct GpuCosineSimilarity;

impl GpuDispatchable for GpuCosineSimilarity {
    type Input = (PackedTritVec, PackedTritVec);
    type Output = f32;

    fn dispatch_gpu(&self, input: &Self::Input, device: &Device) -> GpuResult<Self::Output> {
        let (a, b) = input;

        // Compute dot product on GPU
        let dot = GpuDotSimilarity.dispatch_gpu(input, device)?;

        // Norms computed on CPU (cheap for ternary vectors)
        let norm_a = (a.count_nonzero() as f32).sqrt();
        let norm_b = (b.count_nonzero() as f32).sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            return Ok(0.0);
        }

        Ok(dot as f32 / (norm_a * norm_b))
    }

    fn dispatch_cpu(&self, input: &Self::Input, device: &Device) -> GpuResult<Self::Output> {
        warn_if_cpu(device, "trit-vsa");
        let (a, b) = input;
        check_dims(a, b)?;
        Ok(vsa::cosine_similarity(a, b))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Trit;

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
    fn test_packed_to_encoded() {
        let vec = make_test_vector(&[-1, 0, 1, -1, 1]);
        let encoded = packed_to_encoded(&vec);
        assert_eq!(encoded, vec![0, 1, 2, 0, 2]);
    }

    #[test]
    fn test_encoded_to_packed() {
        let encoded = vec![0, 1, 2, 0, 2];
        let vec = encoded_to_packed(&encoded, 5);

        assert_eq!(vec.get(0), Trit::N);
        assert_eq!(vec.get(1), Trit::Z);
        assert_eq!(vec.get(2), Trit::P);
        assert_eq!(vec.get(3), Trit::N);
        assert_eq!(vec.get(4), Trit::P);
    }

    #[test]
    fn test_cpu_bind_consistency() {
        let a = make_test_vector(&[1, 0, -1, 1]);
        let b = make_test_vector(&[1, -1, 0, -1]);

        let cpu_result = GpuBind.dispatch_cpu(&(a.clone(), b.clone()), &Device::Cpu);
        assert!(cpu_result.is_ok());

        let expected = vsa::bind(&a, &b);
        let result = cpu_result.unwrap();

        for i in 0..4 {
            assert_eq!(result.get(i), expected.get(i));
        }
    }

    #[test]
    fn test_cpu_unbind_inverse() {
        let a = make_test_vector(&[1, 0, -1, 1]);
        let b = make_test_vector(&[1, -1, 0, -1]);

        // bind then unbind should recover original
        let bound = GpuBind
            .dispatch_cpu(&(a.clone(), b.clone()), &Device::Cpu)
            .unwrap();
        let recovered = GpuUnbind.dispatch_cpu(&(bound, b), &Device::Cpu).unwrap();

        for i in 0..4 {
            assert_eq!(recovered.get(i), a.get(i));
        }
    }

    #[test]
    fn test_cpu_bundle() {
        let a = make_test_vector(&[1, 1, -1, 0]);
        let b = make_test_vector(&[1, -1, -1, 1]);
        let c = make_test_vector(&[1, 0, 1, -1]);

        let result = GpuBundle
            .dispatch_cpu(&vec![a, b, c], &Device::Cpu)
            .unwrap();

        // Position 0: 1, 1, 1 → majority is 1
        assert_eq!(result.get(0), Trit::P);
        // Position 2: -1, -1, 1 → majority is -1
        assert_eq!(result.get(2), Trit::N);
    }

    #[test]
    fn test_cpu_dot() {
        let a = make_test_vector(&[1, 0, -1, 1]);
        let b = make_test_vector(&[1, -1, -1, 0]);

        let dot = GpuDotSimilarity
            .dispatch_cpu(&(a.clone(), b.clone()), &Device::Cpu)
            .unwrap();

        // Expected: 1*1 + 0*(-1) + (-1)*(-1) + 1*0 = 1 + 0 + 1 + 0 = 2
        assert_eq!(dot, 2);
    }

    #[test]
    fn test_cpu_hamming() {
        let a = make_test_vector(&[1, 0, -1, 1]);
        let b = make_test_vector(&[1, -1, -1, 0]);

        let dist = GpuHammingDistance
            .dispatch_cpu(&(a, b), &Device::Cpu)
            .unwrap();

        // Positions 1 and 3 differ
        assert_eq!(dist, 2);
    }

    #[test]
    fn test_cpu_random() {
        let input = RandomInput::new(100, 42);
        let result = GpuRandom.dispatch_cpu(&input, &Device::Cpu).unwrap();

        assert_eq!(result.len(), 100);

        // Check that we get a mix of values (statistical test)
        let pos = result.count_positive();
        let neg = result.count_negative();
        let zero = result.len() - pos - neg;

        // With uniform distribution, each should be roughly 33%
        // Allow wide margin for small sample
        assert!(pos > 10, "too few positive: {pos}");
        assert!(neg > 10, "too few negative: {neg}");
        assert!(zero > 10, "too few zero: {zero}");
    }

    #[test]
    fn test_empty_vectors() {
        let a = PackedTritVec::new(0);
        let b = PackedTritVec::new(0);

        assert!(GpuBind
            .dispatch_cpu(&(a.clone(), b.clone()), &Device::Cpu)
            .is_ok());
        assert!(GpuUnbind
            .dispatch_cpu(&(a.clone(), b.clone()), &Device::Cpu)
            .is_ok());
        assert_eq!(
            GpuDotSimilarity
                .dispatch_cpu(&(a.clone(), b.clone()), &Device::Cpu)
                .unwrap(),
            0
        );
        assert_eq!(
            GpuHammingDistance
                .dispatch_cpu(&(a, b), &Device::Cpu)
                .unwrap(),
            0
        );
    }

    #[test]
    fn test_dimension_mismatch() {
        let a = make_test_vector(&[1, 0, -1]);
        let b = make_test_vector(&[1, -1]);

        assert!(GpuBind
            .dispatch_cpu(&(a.clone(), b.clone()), &Device::Cpu)
            .is_err());
        assert!(GpuUnbind
            .dispatch_cpu(&(a.clone(), b.clone()), &Device::Cpu)
            .is_err());
        assert!(GpuDotSimilarity
            .dispatch_cpu(&(a.clone(), b.clone()), &Device::Cpu)
            .is_err());
        assert!(GpuHammingDistance
            .dispatch_cpu(&(a, b), &Device::Cpu)
            .is_err());
    }
}
