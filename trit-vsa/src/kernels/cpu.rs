// SPDX-License-Identifier: MIT
// Copyright 2026 Tyler Zervas

//! CPU backend implementation for ternary VSA operations.
//!
//! This module provides a pure-Rust CPU implementation with optional SIMD
//! acceleration. It serves as the fallback when GPU backends are unavailable.
//!
//! ## SIMD Support
//!
//! When the `simd` feature is enabled and the CPU supports it:
//! - x86_64: AVX2 (256-bit SIMD)
//! - aarch64: NEON (128-bit SIMD)
//!
//! ## Performance Characteristics
//!
//! - Best for small to medium vectors (< 4096 dimensions)
//! - Low latency due to no GPU transfer overhead
//! - Scales linearly with problem size

use crate::kernels::{check_dimensions, RandomConfig, TernaryBackend};
use crate::vsa;
use crate::{PackedTritVec, Result, TernaryError, Trit};

/// CPU backend for ternary operations.
///
/// This backend implements all ternary operations using scalar code
/// with optional SIMD acceleration.
///
/// # SIMD Acceleration
///
/// When `use_simd` is true and the `simd` feature is enabled:
/// - Dot product uses AVX2/NEON when available
/// - Other operations use optimized scalar code
///
/// # Thread Safety
///
/// This backend is `Send + Sync` and can be safely shared across threads.
#[derive(Debug, Clone)]
pub struct CpuBackend {
    /// Whether to use SIMD when available.
    use_simd: bool,
}

impl CpuBackend {
    /// Create a new CPU backend.
    ///
    /// # Arguments
    ///
    /// * `use_simd` - Whether to use SIMD acceleration when available
    #[must_use]
    pub fn new(use_simd: bool) -> Self {
        Self { use_simd }
    }

    /// Create a CPU backend with SIMD enabled (if available).
    #[must_use]
    pub fn with_simd() -> Self {
        Self::new(true)
    }

    /// Create a CPU backend with SIMD disabled (pure scalar).
    #[must_use]
    pub fn scalar_only() -> Self {
        Self::new(false)
    }

    /// Check if SIMD is enabled and available.
    #[must_use]
    pub fn simd_enabled(&self) -> bool {
        if !self.use_simd {
            return false;
        }

        #[cfg(feature = "simd")]
        {
            crate::simd::simd_available()
        }

        #[cfg(not(feature = "simd"))]
        {
            false
        }
    }

    /// Get the SIMD implementation name.
    #[must_use]
    pub fn simd_impl_name(&self) -> &'static str {
        if !self.use_simd {
            return "disabled";
        }

        #[cfg(feature = "simd")]
        {
            crate::simd::simd_impl_name()
        }

        #[cfg(not(feature = "simd"))]
        {
            "unavailable"
        }
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::with_simd()
    }
}

impl TernaryBackend for CpuBackend {
    fn name(&self) -> &'static str {
        if self.simd_enabled() {
            match self.simd_impl_name() {
                "AVX2" => "cpu-avx2",
                "NEON" => "cpu-neon",
                _ => "cpu",
            }
        } else {
            "cpu"
        }
    }

    fn is_available(&self) -> bool {
        true // CPU is always available
    }

    fn bind(&self, a: &PackedTritVec, b: &PackedTritVec) -> Result<PackedTritVec> {
        check_dimensions(a, b)?;
        Ok(vsa::bind(a, b))
    }

    fn unbind(&self, a: &PackedTritVec, b: &PackedTritVec) -> Result<PackedTritVec> {
        check_dimensions(a, b)?;
        Ok(vsa::unbind(a, b))
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

        Ok(vsa::bundle_many(vectors))
    }

    fn dot_similarity(&self, a: &PackedTritVec, b: &PackedTritVec) -> Result<i32> {
        check_dimensions(a, b)?;

        // Use SIMD if enabled and available
        #[cfg(feature = "simd")]
        if self.use_simd {
            return Ok(crate::simd::simd_dot(a, b));
        }

        // Scalar fallback
        Ok(a.dot(b))
    }

    fn cosine_similarity(&self, a: &PackedTritVec, b: &PackedTritVec) -> Result<f32> {
        check_dimensions(a, b)?;
        Ok(vsa::cosine_similarity(a, b))
    }

    fn hamming_distance(&self, a: &PackedTritVec, b: &PackedTritVec) -> Result<usize> {
        check_dimensions(a, b)?;
        Ok(vsa::hamming_distance(a, b))
    }

    fn random(&self, config: &RandomConfig) -> Result<PackedTritVec> {
        if config.dim == 0 {
            return Ok(PackedTritVec::new(0));
        }

        let mut result = PackedTritVec::new(config.dim);
        let mut state = config.seed;

        for i in 0..config.dim {
            // Mix seed with position using golden ratio hash
            let mut s = state.wrapping_add((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));

            // Xorshift64 PRNG
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;

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

    fn negate(&self, a: &PackedTritVec) -> Result<PackedTritVec> {
        Ok(a.negated())
    }
}

// =============================================================================
// SIMD-OPTIMIZED OPERATIONS (for future expansion)
// =============================================================================

/// SIMD-accelerated bind operation for large vectors.
///
/// This is a placeholder for future SIMD optimization of bind.
/// Currently falls back to scalar.
#[allow(dead_code)]
fn bind_simd(a: &PackedTritVec, b: &PackedTritVec) -> PackedTritVec {
    // For now, just use the scalar implementation
    // Future: implement AVX2/NEON optimized version
    vsa::bind(a, b)
}

/// SIMD-accelerated bundle operation.
///
/// This is a placeholder for future SIMD optimization of bundle.
#[allow(dead_code)]
fn bundle_simd(vectors: &[&PackedTritVec]) -> PackedTritVec {
    // For now, just use the scalar implementation
    vsa::bundle_many(vectors)
}

/// SIMD-accelerated Hamming distance.
///
/// Uses popcount instructions which are well-optimized by LLVM.
#[allow(dead_code)]
fn hamming_simd(a: &PackedTritVec, b: &PackedTritVec) -> usize {
    // The scalar implementation already uses popcount which LLVM
    // vectorizes well. For now, just use that.
    vsa::hamming_distance(a, b)
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
    fn test_cpu_backend_creation() {
        let backend = CpuBackend::with_simd();
        assert!(backend.is_available());

        let scalar_backend = CpuBackend::scalar_only();
        assert!(scalar_backend.is_available());
        assert!(!scalar_backend.simd_enabled());
    }

    #[test]
    fn test_bind_unbind_roundtrip() {
        let backend = CpuBackend::default();

        let a = make_test_vector(&[1, -1, 0, 1, -1, 0, 1, -1]);
        let b = make_test_vector(&[-1, 1, 0, -1, 1, 0, -1, 1]);

        let bound = backend.bind(&a, &b).unwrap();
        let recovered = backend.unbind(&bound, &b).unwrap();

        for i in 0..a.len() {
            assert_eq!(recovered.get(i), a.get(i), "mismatch at position {i}");
        }
    }

    #[test]
    fn test_bundle_majority() {
        let backend = CpuBackend::default();

        let a = make_test_vector(&[1, 1, 1, 0, 0]);
        let b = make_test_vector(&[1, 1, -1, 1, -1]);
        let c = make_test_vector(&[1, -1, -1, -1, 0]);

        let result = backend.bundle(&[&a, &b, &c]).unwrap();

        // Position 0: 1, 1, 1 -> 1
        assert_eq!(result.get(0), Trit::P);
        // Position 1: 1, 1, -1 -> 1
        assert_eq!(result.get(1), Trit::P);
        // Position 2: 1, -1, -1 -> -1
        assert_eq!(result.get(2), Trit::N);
        // Position 3: 0, 1, -1 -> 0 (tie)
        assert_eq!(result.get(3), Trit::Z);
        // Position 4: 0, -1, 0 -> -1
        assert_eq!(result.get(4), Trit::N);
    }

    #[test]
    fn test_dot_similarity_values() {
        let backend = CpuBackend::default();

        // Identical vectors
        let a = make_test_vector(&[1, 1, -1, -1]);
        let dot = backend.dot_similarity(&a, &a).unwrap();
        assert_eq!(dot, 4); // 1+1+1+1

        // Opposite vectors
        let neg_a = make_test_vector(&[-1, -1, 1, 1]);
        let dot_opposite = backend.dot_similarity(&a, &neg_a).unwrap();
        assert_eq!(dot_opposite, -4);

        // Orthogonal vectors
        let b = make_test_vector(&[1, -1, 1, -1]);
        let dot_orth = backend.dot_similarity(&a, &b).unwrap();
        assert_eq!(dot_orth, 0);
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let backend = CpuBackend::default();

        let a = make_test_vector(&[1, 1, -1, -1, 0, 0]);
        let sim = backend.cosine_similarity(&a, &a).unwrap();
        assert!((sim - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_hamming_distance_values() {
        let backend = CpuBackend::default();

        // Identical
        let a = make_test_vector(&[1, 0, -1]);
        assert_eq!(backend.hamming_distance(&a, &a).unwrap(), 0);

        // All different
        let b = make_test_vector(&[-1, 1, 0]);
        assert_eq!(backend.hamming_distance(&a, &b).unwrap(), 3);

        // One different
        let c = make_test_vector(&[1, 0, 0]);
        assert_eq!(backend.hamming_distance(&a, &c).unwrap(), 1);
    }

    #[test]
    fn test_random_deterministic() {
        let backend = CpuBackend::default();

        let config = RandomConfig::new(1000, 12345);

        let r1 = backend.random(&config).unwrap();
        let r2 = backend.random(&config).unwrap();

        // Same seed should produce same result
        for i in 0..r1.len() {
            assert_eq!(r1.get(i), r2.get(i), "mismatch at position {i}");
        }
    }

    #[test]
    fn test_random_different_seeds() {
        let backend = CpuBackend::default();

        let r1 = backend.random(&RandomConfig::new(100, 1)).unwrap();
        let r2 = backend.random(&RandomConfig::new(100, 2)).unwrap();

        // Different seeds should (almost certainly) produce different results
        let mut different = false;
        for i in 0..r1.len() {
            if r1.get(i) != r2.get(i) {
                different = true;
                break;
            }
        }
        assert!(
            different,
            "different seeds should produce different vectors"
        );
    }

    #[test]
    fn test_negate() {
        let backend = CpuBackend::default();

        let a = make_test_vector(&[1, -1, 0, 1]);
        let neg = backend.negate(&a).unwrap();

        assert_eq!(neg.get(0), Trit::N);
        assert_eq!(neg.get(1), Trit::P);
        assert_eq!(neg.get(2), Trit::Z);
        assert_eq!(neg.get(3), Trit::N);
    }

    #[test]
    fn test_empty_vectors() {
        let backend = CpuBackend::default();

        let a = PackedTritVec::new(0);
        let b = PackedTritVec::new(0);

        assert!(backend.bind(&a, &b).is_ok());
        assert!(backend.unbind(&a, &b).is_ok());
        assert_eq!(backend.dot_similarity(&a, &b).unwrap(), 0);
        assert_eq!(backend.hamming_distance(&a, &b).unwrap(), 0);
    }

    #[test]
    fn test_bundle_empty() {
        let backend = CpuBackend::default();

        let result = backend.bundle(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_large_vector_performance() {
        let backend = CpuBackend::default();

        // Create large random vectors
        let r1 = backend.random(&RandomConfig::new(10000, 1)).unwrap();
        let r2 = backend.random(&RandomConfig::new(10000, 2)).unwrap();

        // These should complete quickly
        let _ = backend.bind(&r1, &r2).unwrap();
        let _ = backend.dot_similarity(&r1, &r2).unwrap();
        let _ = backend.hamming_distance(&r1, &r2).unwrap();
    }
}
