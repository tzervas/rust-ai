// SPDX-License-Identifier: MIT
// Copyright 2026 Tyler Zervas

//! Burn framework backend stub for ternary VSA operations.
//!
//! This module provides a placeholder backend for future integration with
//! the [Burn](https://github.com/tracel-ai/burn) deep learning framework.
//!
//! ## Status
//!
//! **This is a stub implementation.** All operations currently fall back to
//! the CPU backend. Full implementation is planned for a future release.
//!
//! ## Planned Features
//!
//! When fully implemented, this backend will provide:
//! - Automatic backend selection (WebGPU, CUDA, CPU)
//! - Tensor graph optimization
//! - Automatic differentiation support
//! - Cross-platform execution
//!
//! ## Usage
//!
//! ```rust,ignore
//! use trit_vsa::kernels::{BurnBackend, TernaryBackend, BackendConfig, BackendPreference};
//!
//! // When fully implemented:
//! let config = BackendConfig {
//!     preferred: BackendPreference::Burn,
//!     ..Default::default()
//! };
//! let backend = trit_vsa::kernels::get_backend(&config);
//! ```

use crate::kernels::{CpuBackend, RandomConfig, TernaryBackend};
use crate::{PackedTritVec, Result};

/// Burn framework backend for ternary operations.
///
/// **Status: Stub implementation - falls back to CPU.**
///
/// This backend is a placeholder for future integration with the Burn
/// deep learning framework. Currently, all operations delegate to the
/// CPU backend.
///
/// # Future Plans
///
/// When fully implemented, this backend will:
/// - Support multiple Burn backends (WebGPU, CUDA, Vulkan, CPU)
/// - Enable automatic differentiation for gradient-based optimization
/// - Allow integration with Burn's tensor graph for fusion and optimization
/// - Provide cross-platform GPU support beyond CUDA
///
/// # Example (Future API)
///
/// ```rust,ignore
/// use trit_vsa::kernels::BurnBackend;
/// use burn::backend::Autodiff;
/// use burn::backend::wgpu::Wgpu;
///
/// // Future: Configure Burn backend
/// let backend = BurnBackend::with_backend::<Autodiff<Wgpu>>();
/// ```
#[derive(Debug, Clone)]
pub struct BurnBackend {
    /// Internal CPU fallback.
    cpu_fallback: CpuBackend,
    /// Whether Burn is available (stub: always false for now).
    #[allow(dead_code)]
    burn_available: bool,
}

impl BurnBackend {
    /// Create a new Burn backend.
    ///
    /// Note: Currently falls back to CPU implementation.
    #[must_use]
    pub fn new() -> Self {
        Self {
            cpu_fallback: CpuBackend::with_simd(),
            burn_available: false, // Stub: not implemented yet
        }
    }

    /// Check if Burn backend is fully implemented.
    ///
    /// Currently always returns `false` as this is a stub.
    #[must_use]
    pub fn is_burn_implemented() -> bool {
        false
    }
}

impl Default for BurnBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl TernaryBackend for BurnBackend {
    fn name(&self) -> &'static str {
        // Return actual backend name once implemented
        // For now, indicate it's falling back to CPU
        "burn-stub-cpu"
    }

    fn is_available(&self) -> bool {
        // Burn stub is "available" in that it falls back to CPU
        true
    }

    fn bind(&self, a: &PackedTritVec, b: &PackedTritVec) -> Result<PackedTritVec> {
        // TODO: Implement with Burn tensors
        self.cpu_fallback.bind(a, b)
    }

    fn unbind(&self, a: &PackedTritVec, b: &PackedTritVec) -> Result<PackedTritVec> {
        // TODO: Implement with Burn tensors
        self.cpu_fallback.unbind(a, b)
    }

    fn bundle(&self, vectors: &[&PackedTritVec]) -> Result<PackedTritVec> {
        // TODO: Implement with Burn tensors
        self.cpu_fallback.bundle(vectors)
    }

    fn dot_similarity(&self, a: &PackedTritVec, b: &PackedTritVec) -> Result<i32> {
        // TODO: Implement with Burn tensors
        self.cpu_fallback.dot_similarity(a, b)
    }

    fn cosine_similarity(&self, a: &PackedTritVec, b: &PackedTritVec) -> Result<f32> {
        // TODO: Implement with Burn tensors
        self.cpu_fallback.cosine_similarity(a, b)
    }

    fn hamming_distance(&self, a: &PackedTritVec, b: &PackedTritVec) -> Result<usize> {
        // TODO: Implement with Burn tensors
        self.cpu_fallback.hamming_distance(a, b)
    }

    fn random(&self, config: &RandomConfig) -> Result<PackedTritVec> {
        // TODO: Implement with Burn random generation
        self.cpu_fallback.random(config)
    }

    fn negate(&self, a: &PackedTritVec) -> Result<PackedTritVec> {
        // TODO: Implement with Burn tensors
        self.cpu_fallback.negate(a)
    }
}

// =============================================================================
// BURN INTEGRATION ROADMAP
// =============================================================================

/// Planned Burn tensor representation for ternary vectors.
///
/// This is a placeholder showing the planned API for Burn integration.
#[allow(dead_code)]
mod planned_api {
    /// Burn-based ternary tensor representation.
    ///
    /// When implemented, this will wrap Burn tensors for ternary operations.
    ///
    /// # Planned Features
    ///
    /// - Automatic backend selection (WebGPU, CUDA, CPU)
    /// - Gradient tracking for training
    /// - Tensor fusion optimization
    pub struct BurnTritTensor<B> {
        /// Plus plane as Burn tensor (u32 packed bits)
        #[allow(dead_code)]
        plus_plane: std::marker::PhantomData<B>,
        /// Minus plane as Burn tensor (u32 packed bits)
        #[allow(dead_code)]
        minus_plane: std::marker::PhantomData<B>,
        /// Logical dimension count
        #[allow(dead_code)]
        num_dims: usize,
    }

    /// Planned conversion trait for Burn tensors.
    pub trait ToBurnTensor<B> {
        /// Convert to Burn tensor representation.
        fn to_burn_tensor(&self) -> BurnTritTensor<B>;
    }

    /// Planned operations using Burn primitives.
    ///
    /// These operations would be implemented using Burn's tensor operations:
    /// - Element-wise: `tensor.add()`, `tensor.sub()`, `tensor.mul()`
    /// - Reduction: `tensor.sum()`, `tensor.mean()`
    /// - Custom kernels: `burn_kernel!` macro for specialized operations
    pub trait BurnTritOps<B> {
        /// Bind using Burn tensors.
        fn bind_burn(&self, other: &Self) -> Self;

        /// Dot product using Burn reduction.
        fn dot_burn(&self, other: &Self) -> i32;

        /// Bundle using Burn aggregation.
        fn bundle_burn(vectors: &[&Self]) -> Self;
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
    fn test_burn_backend_creation() {
        let backend = BurnBackend::new();
        assert!(backend.is_available());
        assert_eq!(backend.name(), "burn-stub-cpu");
    }

    #[test]
    fn test_burn_stub_not_implemented() {
        assert!(!BurnBackend::is_burn_implemented());
    }

    #[test]
    fn test_burn_backend_fallback() {
        let backend = BurnBackend::new();

        let a = make_test_vector(&[1, -1, 0, 1]);
        let b = make_test_vector(&[-1, 1, 0, -1]);

        // All operations should work via CPU fallback
        let bound = backend.bind(&a, &b).unwrap();
        let recovered = backend.unbind(&bound, &b).unwrap();

        for i in 0..a.len() {
            assert_eq!(recovered.get(i), a.get(i), "mismatch at position {i}");
        }
    }

    #[test]
    fn test_burn_backend_dot() {
        let backend = BurnBackend::new();

        let a = make_test_vector(&[1, 1, -1, -1]);
        let dot = backend.dot_similarity(&a, &a).unwrap();
        assert_eq!(dot, 4);
    }

    #[test]
    fn test_burn_backend_random() {
        let backend = BurnBackend::new();

        let config = RandomConfig::new(100, 42);
        let result = backend.random(&config).unwrap();

        assert_eq!(result.len(), 100);
        assert!(result.count_nonzero() > 0);
    }
}
