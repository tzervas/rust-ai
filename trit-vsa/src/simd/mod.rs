//! SIMD-optimized operations for ternary vectors.
//!
//! This module provides hardware-accelerated implementations when the `simd`
//! feature is enabled. Falls back to scalar implementations otherwise.
//!
//! ## Supported Platforms
//!
//! - **`x86_64`**: AVX2 (256-bit SIMD)
//! - **`aarch64`**: NEON (128-bit SIMD)

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
mod avx2;

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
mod neon;

use crate::packed::PackedTritVec;

/// SIMD-accelerated dot product.
///
/// Uses AVX2/NEON when available, falls back to scalar otherwise.
#[must_use]
pub fn simd_dot(a: &PackedTritVec, b: &PackedTritVec) -> i32 {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: We've verified AVX2 is available
            return unsafe { avx2::dot_avx2(a, b) };
        }
    }

    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    {
        // NEON is always available on aarch64
        return unsafe { neon::dot_neon(a, b) };
    }

    // Fallback to scalar implementation
    a.dot(b)
}

/// SIMD-accelerated popcount sum.
///
/// Counts total set bits across both planes.
#[must_use]
pub fn simd_popcount_total(vec: &PackedTritVec) -> usize {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("popcnt") {
            return unsafe { avx2::popcount_total_avx2(vec) };
        }
    }

    // Fallback
    vec.count_nonzero()
}

/// Check if SIMD optimizations are available on this platform.
#[must_use]
pub fn simd_available() -> bool {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        return is_x86_feature_detected!("avx2");
    }

    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    {
        return true; // NEON is always available
    }

    #[allow(unreachable_code)]
    false
}

/// Get the name of the SIMD implementation being used.
#[must_use]
pub fn simd_impl_name() -> &'static str {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            return "AVX2";
        }
    }

    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    {
        return "NEON";
    }

    #[allow(unreachable_code)]
    "scalar"
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trit::Trit;

    #[test]
    fn test_simd_dot_matches_scalar() {
        let mut a = PackedTritVec::new(256);
        let mut b = PackedTritVec::new(256);

        // Set up test vectors
        for i in 0..128 {
            a.set(i, Trit::P);
            b.set(i, if i % 2 == 0 { Trit::P } else { Trit::N });
        }

        let scalar_result = a.dot(&b);
        let simd_result = simd_dot(&a, &b);

        assert_eq!(scalar_result, simd_result);
    }

    #[test]
    fn test_simd_available() {
        // This just verifies the function doesn't panic
        let _available = simd_available();
        let _name = simd_impl_name();
    }
}
