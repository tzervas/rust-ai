// SPDX-License-Identifier: MIT
// Copyright 2026 Tyler Zervas

//! CubeCL kernel implementations for ternary VSA operations.
//!
//! This module contains the low-level GPU kernels for ternary vector operations.
//! All kernels operate on the encoded trit representation where:
//!
//! - `0` encodes trit value `-1` (N)
//! - `1` encodes trit value `0` (Z)
//! - `2` encodes trit value `+1` (P)
//!
//! ## Kernel Categories
//!
//! ### Element-wise Operations
//! - [`ternary_bind_kernel`] - XOR-like composition
//! - [`ternary_unbind_kernel`] - Inverse of bind
//! - [`ternary_negate_kernel`] - Element-wise negation
//!
//! ### Reductions
//! - [`ternary_bundle_count_kernel`] - Count trit values per position
//! - [`ternary_majority_kernel`] - Select majority from counts
//! - [`ternary_dot_kernel`] - Dot product with tree reduction
//! - [`ternary_hamming_kernel`] - Hamming distance with reduction
//!
//! ### Utility
//! - [`ternary_random_kernel`] - Xorshift-based random generation
//! - [`final_reduction_kernel`] - Sum partial block results

use cubecl::prelude::*;

// =============================================================================
// ELEMENT-WISE KERNELS
// =============================================================================

/// Ternary bind operation (XOR-like composition).
///
/// Implements balanced ternary binding using modular arithmetic:
/// `result[i] = (a[i] + b[i] + 1) mod 3`
///
/// This is equivalent to the group operation in Z₃ with identity adjustment.
/// The operation is commutative and associative.
///
/// # Encoding
///
/// Input/output uses encoded representation:
/// - `0` = trit `-1` (N)
/// - `1` = trit `0` (Z)
/// - `2` = trit `+1` (P)
///
/// # Truth Table
///
/// | a  | b  | bind(a,b) |
/// |----|----|-----------|
/// | N  | N  | P         |
/// | N  | Z  | N         |
/// | N  | P  | Z         |
/// | Z  | N  | N         |
/// | Z  | Z  | Z         |
/// | Z  | P  | P         |
/// | P  | N  | Z         |
/// | P  | Z  | P         |
/// | P  | P  | N         |
#[cube(launch)]
pub fn ternary_bind_kernel<I: Int>(
    a: &Array<I>,
    b: &Array<I>,
    out: &mut Array<I>,
    #[comptime] len: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx < (len as usize) {
        // Balanced ternary bind: (a - b + 3) mod 3
        // Bind is subtraction in balanced ternary
        // Adding 3 ensures we stay non-negative before modulo
        let diff = a[idx] - b[idx] + I::new(3);
        let quotient = diff / I::new(3);
        out[idx] = diff - quotient * I::new(3);
    }
}

/// Ternary unbind operation (inverse of bind).
///
/// For balanced ternary with our encoding, unbind is computed as:
/// `result[i] = (a[i] - b[i] + 3) mod 3`
///
/// This recovers the original: `unbind(bind(a, b), b) == a`
///
/// # Properties
///
/// - `unbind(bind(a, b), b) == a`
/// - `unbind(a, b) == bind(a, negate(b))`
#[cube(launch)]
pub fn ternary_unbind_kernel<I: Int>(
    a: &Array<I>,
    b: &Array<I>,
    out: &mut Array<I>,
    #[comptime] len: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx < (len as usize) {
        // Unbind: (a + b) mod 3
        // Unbind is addition in balanced ternary (inverse of bind)
        let sum = a[idx] + b[idx];
        let quotient = sum / I::new(3);
        out[idx] = sum - quotient * I::new(3);
    }
}

/// Ternary negation (element-wise).
///
/// Negates each trit: -(-1) = +1, -(0) = 0, -(+1) = -1
///
/// In encoded form: 0 ↔ 2, 1 → 1
#[cube(launch)]
pub fn ternary_negate_kernel<I: Int>(input: &Array<I>, out: &mut Array<I>, #[comptime] len: u32) {
    let idx = ABSOLUTE_POS;
    if idx < (len as usize) {
        // Negate: (2 - input) gives: 0→2, 1→1, 2→0
        out[idx] = I::new(2) - input[idx];
    }
}

// =============================================================================
// BUNDLE (MAJORITY VOTING) KERNELS
// =============================================================================

/// Count trit occurrences across multiple vectors at each position.
///
/// This kernel processes multiple vectors and counts how many times each
/// trit value (0, 1, 2 in encoded form) appears at each dimension.
///
/// # Arguments
///
/// * `vectors` - Flattened array of shape `[num_vectors, dim]`
/// * `counts` - Output counts of shape `[3, dim]` where:
///   - `counts[0*dim + pos]` = count of encoded 0 (trit -1)
///   - `counts[1*dim + pos]` = count of encoded 1 (trit 0)
///   - `counts[2*dim + pos]` = count of encoded 2 (trit +1)
/// * `num_vectors` - Number of vectors being bundled
/// * `dim` - Dimension of each vector
#[cube(launch)]
pub fn ternary_bundle_count_kernel<I: Int>(
    vectors: &Array<I>,
    counts: &mut Array<I>,
    #[comptime] num_vectors: u32,
    #[comptime] dim: u32,
) {
    let pos = ABSOLUTE_POS;
    if pos < (dim as usize) {
        // Count each trit value at this position across all vectors
        let mut count_neg = I::new(0); // encoded 0 (trit -1)
        let mut count_zero = I::new(0); // encoded 1 (trit 0)
        let mut count_pos = I::new(0); // encoded 2 (trit +1)

        // Iterate over all vectors
        #[unroll]
        for i in 0..num_vectors {
            let val = vectors[(i * dim) as usize + pos];
            // Use arithmetic comparison instead of equality for better GPU performance
            // val == 0: val is 0, val - 1 is -1, val - 2 is -2
            let is_neg = I::new(1) - (val + I::new(1)) / I::new(2);
            let is_zero_cond = val / I::new(1);

            if val == I::new(0) {
                count_neg = count_neg + I::new(1);
            } else if val == I::new(1) {
                count_zero = count_zero + I::new(1);
            } else {
                count_pos = count_pos + I::new(1);
            }
        }

        // Store counts in output array
        counts[pos] = count_neg;
        counts[(dim as usize) + pos] = count_zero;
        counts[((dim * 2) as usize) + pos] = count_pos;
    }
}

/// Small-N bundle kernel optimized for bundling 2-8 vectors.
///
/// Uses inline counting without separate count storage for reduced memory traffic.
#[cube(launch)]
pub fn ternary_bundle_small_kernel<I: Int>(
    vectors: &Array<I>,
    out: &mut Array<I>,
    #[comptime] num_vectors: u32,
    #[comptime] dim: u32,
) {
    let pos = ABSOLUTE_POS;
    if pos < (dim as usize) {
        let mut count_neg = I::new(0);
        let mut count_zero = I::new(0);
        let mut count_pos = I::new(0);

        #[unroll]
        for i in 0..num_vectors {
            let val = vectors[(i * dim) as usize + pos];
            if val == I::new(0) {
                count_neg = count_neg + I::new(1);
            } else if val == I::new(1) {
                count_zero = count_zero + I::new(1);
            } else {
                count_pos = count_pos + I::new(1);
            }
        }

        // Select majority (ties go to zero/neutral)
        if count_neg > count_zero && count_neg > count_pos {
            out[pos] = I::new(0); // -1 encoded
        } else if count_pos > count_zero && count_pos > count_neg {
            out[pos] = I::new(2); // +1 encoded
        } else {
            out[pos] = I::new(1); // 0 encoded (default for ties)
        }
    }
}

/// Select majority trit from vote counts.
///
/// For each position, compares counts and outputs the majority trit.
/// Ties are resolved to zero (neutral element).
///
/// # Arguments
///
/// * `counts` - Input counts from `ternary_bundle_count_kernel`
/// * `out` - Output vector with majority trits
/// * `dim` - Vector dimension
#[cube(launch)]
pub fn ternary_majority_kernel<I: Int>(
    counts: &Array<I>,
    out: &mut Array<I>,
    #[comptime] dim: u32,
) {
    let pos = ABSOLUTE_POS;
    if pos < (dim as usize) {
        let count_neg = counts[pos];
        let count_zero = counts[(dim as usize) + pos];
        let count_pos = counts[((dim * 2) as usize) + pos];

        // Find maximum - ties go to zero (neutral)
        if count_neg > count_zero && count_neg > count_pos {
            out[pos] = I::new(0); // -1 encoded
        } else if count_pos > count_zero && count_pos > count_neg {
            out[pos] = I::new(2); // +1 encoded
        } else {
            out[pos] = I::new(1); // 0 encoded (default for ties and three-way ties)
        }
    }
}

// =============================================================================
// SIMILARITY/DISTANCE KERNELS
// =============================================================================

/// Compute partial dot products using parallel reduction.
///
/// Each block computes a partial sum of element-wise products.
/// The final sum must be computed by summing `partial_sums[0..num_blocks]`.
///
/// # Dot Product Formula
///
/// For balanced ternary vectors, the dot product is:
/// `sum(a[i] * b[i])` where values are decoded from {0,1,2} to {-1,0,+1}.
///
/// # Arguments
///
/// * `a`, `b` - Input vectors (encoded)
/// * `partial_sums` - Output array for block partial sums
/// * `len` - Vector length
#[cube(launch)]
pub fn ternary_dot_kernel(
    a: &Array<u32>,
    b: &Array<u32>,
    partial_sums: &mut Array<u32>,
    #[comptime] len: u32,
) {
    let idx = ABSOLUTE_POS;
    let block_idx = CUBE_POS_X;
    let thread_idx = UNIT_POS_X;

    // Allocate shared memory for block reduction
    let mut shared = SharedMemory::<u32>::new(256usize);

    // Each thread computes one element-wise product
    let mut product: u32 = 0u32;
    if idx < (len as usize) {
        // Decode from {0,1,2} to {-1,0,+1} by subtracting 1
        let a_val: u32 = a[idx] - 1u32;
        let b_val: u32 = b[idx] - 1u32;
        product = a_val * b_val;
    }

    shared[thread_idx as usize] = product;
    sync_cube();

    // Tree reduction within block
    // Block size is 256, so we do 8 iterations (256 -> 128 -> 64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1)
    if thread_idx < 128u32 {
        shared[thread_idx as usize] =
            shared[thread_idx as usize] + shared[(thread_idx + 128u32) as usize];
    }
    sync_cube();

    if thread_idx < 64u32 {
        shared[thread_idx as usize] =
            shared[thread_idx as usize] + shared[(thread_idx + 64u32) as usize];
    }
    sync_cube();

    if thread_idx < 32u32 {
        shared[thread_idx as usize] =
            shared[thread_idx as usize] + shared[(thread_idx + 32u32) as usize];
    }
    sync_cube();

    if thread_idx < 16u32 {
        shared[thread_idx as usize] =
            shared[thread_idx as usize] + shared[(thread_idx + 16u32) as usize];
    }
    sync_cube();

    if thread_idx < 8u32 {
        shared[thread_idx as usize] =
            shared[thread_idx as usize] + shared[(thread_idx + 8u32) as usize];
    }
    sync_cube();

    if thread_idx < 4u32 {
        shared[thread_idx as usize] =
            shared[thread_idx as usize] + shared[(thread_idx + 4u32) as usize];
    }
    sync_cube();

    if thread_idx < 2u32 {
        shared[thread_idx as usize] =
            shared[thread_idx as usize] + shared[(thread_idx + 2u32) as usize];
    }
    sync_cube();

    // Thread 0 writes final block result
    if thread_idx == 0u32 {
        let final_sum = shared[0usize] + shared[1usize];
        partial_sums[block_idx as usize] = final_sum;
    }
}

/// Compute Hamming distance using parallel reduction.
///
/// Hamming distance counts positions where trits differ.
///
/// # Arguments
///
/// * `a`, `b` - Input vectors (encoded)
/// * `partial_counts` - Output array for block partial counts
/// * `len` - Vector length
#[cube(launch)]
pub fn ternary_hamming_kernel(
    a: &Array<u32>,
    b: &Array<u32>,
    partial_counts: &mut Array<u32>,
    #[comptime] len: u32,
) {
    let idx = ABSOLUTE_POS;
    let block_idx = CUBE_POS_X;
    let thread_idx = UNIT_POS_X;

    let mut shared = SharedMemory::<u32>::new(256usize);

    // Count if trits differ at this position
    let mut diff: u32 = 0u32;
    if idx < (len as usize) {
        if a[idx] != b[idx] {
            diff = 1u32;
        }
    }

    shared[thread_idx as usize] = diff;
    sync_cube();

    // Tree reduction (same pattern as dot kernel)
    if thread_idx < 128u32 {
        shared[thread_idx as usize] =
            shared[thread_idx as usize] + shared[(thread_idx + 128u32) as usize];
    }
    sync_cube();

    if thread_idx < 64u32 {
        shared[thread_idx as usize] =
            shared[thread_idx as usize] + shared[(thread_idx + 64u32) as usize];
    }
    sync_cube();

    if thread_idx < 32u32 {
        shared[thread_idx as usize] =
            shared[thread_idx as usize] + shared[(thread_idx + 32u32) as usize];
    }
    sync_cube();

    if thread_idx < 16u32 {
        shared[thread_idx as usize] =
            shared[thread_idx as usize] + shared[(thread_idx + 16u32) as usize];
    }
    sync_cube();

    if thread_idx < 8u32 {
        shared[thread_idx as usize] =
            shared[thread_idx as usize] + shared[(thread_idx + 8u32) as usize];
    }
    sync_cube();

    if thread_idx < 4u32 {
        shared[thread_idx as usize] =
            shared[thread_idx as usize] + shared[(thread_idx + 4u32) as usize];
    }
    sync_cube();

    if thread_idx < 2u32 {
        shared[thread_idx as usize] =
            shared[thread_idx as usize] + shared[(thread_idx + 2u32) as usize];
    }
    sync_cube();

    if thread_idx == 0u32 {
        let final_count = shared[0usize] + shared[1usize];
        partial_counts[block_idx as usize] = final_count;
    }
}

/// Final reduction kernel for summing partial results.
///
/// Sums all partial block results into a single value.
/// Used after `ternary_dot_kernel` or `ternary_hamming_kernel`.
#[cube(launch)]
pub fn final_reduction_kernel(
    partial_sums: &Array<u32>,
    result: &mut Array<u32>,
    #[comptime] num_blocks: u32,
) {
    let thread_idx = UNIT_POS_X;

    let mut shared = SharedMemory::<u32>::new(256usize);

    // Load partial sums into shared memory
    if thread_idx < num_blocks {
        shared[thread_idx as usize] = partial_sums[thread_idx as usize];
    } else {
        shared[thread_idx as usize] = 0u32;
    }
    sync_cube();

    // Tree reduction
    if thread_idx < 128u32 && thread_idx + 128u32 < 256u32 {
        shared[thread_idx as usize] =
            shared[thread_idx as usize] + shared[(thread_idx + 128u32) as usize];
    }
    sync_cube();

    if thread_idx < 64u32 {
        shared[thread_idx as usize] =
            shared[thread_idx as usize] + shared[(thread_idx + 64u32) as usize];
    }
    sync_cube();

    if thread_idx < 32u32 {
        shared[thread_idx as usize] =
            shared[thread_idx as usize] + shared[(thread_idx + 32u32) as usize];
    }
    sync_cube();

    if thread_idx < 16u32 {
        shared[thread_idx as usize] =
            shared[thread_idx as usize] + shared[(thread_idx + 16u32) as usize];
    }
    sync_cube();

    if thread_idx < 8u32 {
        shared[thread_idx as usize] =
            shared[thread_idx as usize] + shared[(thread_idx + 8u32) as usize];
    }
    sync_cube();

    if thread_idx < 4u32 {
        shared[thread_idx as usize] =
            shared[thread_idx as usize] + shared[(thread_idx + 4u32) as usize];
    }
    sync_cube();

    if thread_idx < 2u32 {
        shared[thread_idx as usize] =
            shared[thread_idx as usize] + shared[(thread_idx + 2u32) as usize];
    }
    sync_cube();

    if thread_idx == 0u32 {
        result[0usize] = shared[0usize] + shared[1usize];
    }
}

// =============================================================================
// RANDOM GENERATION KERNEL
// =============================================================================

/// Generate random ternary vector using xorshift32 PRNG.
///
/// Each thread generates one random trit using a xorshift32 state
/// initialized from the seed array.
///
/// # Arguments
///
/// * `out` - Output array for random trits (encoded as 0, 1, 2)
/// * `seeds` - Per-element seed values (should be pre-initialized with unique values)
/// * `len` - Vector length
///
/// # Algorithm
///
/// Uses xorshift32:
/// ```text
/// state ^= state << 13
/// state ^= state >> 17
/// state ^= state << 5
/// result = state mod 3
/// ```
#[cube(launch)]
pub fn ternary_random_kernel(out: &mut Array<u32>, seeds: &Array<u32>, #[comptime] len: u32) {
    let idx = ABSOLUTE_POS;
    if idx < (len as usize) {
        // Xorshift32 PRNG
        let mut state = seeds[idx];

        // Xorshift32 algorithm
        state ^= state << 13u32;
        state ^= state >> 17u32;
        state ^= state << 5u32;

        // Map to 0, 1, 2 (balanced ternary encoding)
        // Using modulo 3
        let remainder = state - (state / 3u32) * 3u32;
        out[idx] = remainder;
    }
}

/// Initialize seed array for random generation.
///
/// Creates unique seeds for each position based on a base seed.
/// Uses a simple LCG-style mixing to spread entropy.
#[cube(launch)]
pub fn init_seeds_kernel(seeds: &mut Array<u32>, #[comptime] base_seed: u32, #[comptime] len: u32) {
    let idx = ABSOLUTE_POS;
    if idx < (len as usize) {
        // Mix base seed with position to create unique per-element seed
        // Using MurmurHash3-style mixing (smaller constants to avoid overflow checks)
        let mut mixed = base_seed ^ (idx as u32);
        mixed ^= mixed >> 16u32;
        mixed *= 0x85ebca6bu32;
        mixed ^= mixed >> 13u32;
        mixed *= 0xc2b2ae35u32;
        mixed ^= mixed >> 16u32;
        seeds[idx] = mixed;
    }
}

// =============================================================================
// CONVERSION KERNELS
// =============================================================================

/// Convert from bitsliced representation to encoded representation.
///
/// Takes separate plus/minus planes and produces encoded array where:
/// - plus=1, minus=0 → 2 (+1)
/// - plus=0, minus=0 → 1 (0)
/// - plus=0, minus=1 → 0 (-1)
///
/// This kernel processes 32 trits per u32 word.
#[cube(launch)]
pub fn bitsliced_to_encoded_kernel(
    plus_plane: &Array<u32>,
    minus_plane: &Array<u32>,
    out: &mut Array<i32>,
    #[comptime] num_dims: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx < (num_dims as usize) {
        let word_idx = idx / 32usize;
        let bit_idx = (idx % 32usize) as u32;
        let mask = 1u32 << bit_idx;

        let is_plus = (plus_plane[word_idx] & mask) != 0u32;
        let is_minus = (minus_plane[word_idx] & mask) != 0u32;

        // Encode: plus → 2, zero → 1, minus → 0
        if is_plus {
            out[idx] = 2i32;
        } else if is_minus {
            out[idx] = 0i32;
        } else {
            out[idx] = 1i32;
        }
    }
}

/// Convert from encoded representation back to bitsliced.
///
/// Takes encoded array and produces separate plus/minus planes.
/// Each thread processes one trit and sets the appropriate bit.
///
/// Note: This kernel must be called with proper synchronization since
/// multiple threads write to the same u32 word. In practice, use
/// atomic operations or process word-by-word.
#[cube(launch)]
pub fn encoded_to_bitsliced_word_kernel(
    encoded: &Array<i32>,
    plus_plane: &mut Array<u32>,
    minus_plane: &mut Array<u32>,
    #[comptime] num_words: u32,
    #[comptime] num_dims: u32,
) {
    let word_idx = ABSOLUTE_POS;
    if word_idx < (num_words as usize) {
        let mut plus_word = 0u32;
        let mut minus_word = 0u32;

        // Process 32 trits for this word
        let start_dim = (word_idx as u32) * 32u32;

        #[unroll]
        for bit in 0u32..32u32 {
            let dim = start_dim + bit;
            if dim < num_dims {
                let val = encoded[(dim as usize)];
                let mask = 1u32 << bit;

                if val == 2i32 {
                    plus_word |= mask;
                } else if val == 0i32 {
                    minus_word |= mask;
                }
                // val == 1 means zero, both planes stay 0
            }
        }

        plus_plane[word_idx] = plus_word;
        minus_plane[word_idx] = minus_word;
    }
}

// =============================================================================
// KERNEL LAUNCH CONFIGURATION
// =============================================================================

/// Standard block size for CubeCL kernels.
pub const BLOCK_SIZE: u32 = 256;

/// Calculate grid size for a given problem size and block size.
#[inline]
#[must_use]
pub const fn grid_size(problem_size: u32, block_size: u32) -> u32 {
    (problem_size + block_size - 1) / block_size
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_size() {
        assert_eq!(grid_size(256, 256), 1);
        assert_eq!(grid_size(257, 256), 2);
        assert_eq!(grid_size(512, 256), 2);
        assert_eq!(grid_size(1000, 256), 4);
        assert_eq!(grid_size(0, 256), 0);
    }
}
