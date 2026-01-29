// SPDX-License-Identifier: MIT
// Copyright 2026 Tyler Zervas

//! Modular kernel architecture for ternary VSA operations.
//!
//! This module provides a backend-agnostic interface for ternary vector operations,
//! allowing easy swapping between CPU (scalar/SIMD), CUDA (CubeCL), and future
//! backends (e.g., Burn) at runtime.
//!
//! # Architecture
//!
//! ```text
//! +-------------------+
//! |  TernaryBackend   |  <- Trait defining all operations
//! +-------------------+
//!          |
//!    +-----+-----+-----+
//!    |           |     |
//!    v           v     v
//! +------+  +-------+  +------+
//! |  CPU |  | CubeCL|  | Burn |
//! +------+  +-------+  +------+
//! ```
//!
//! # Backend Selection
//!
//! Backends can be selected based on:
//! - Feature flags (`cuda`, `burn`)
//! - Runtime detection (GPU availability)
//! - User configuration (force CPU/GPU)
//! - Problem size thresholds
//!
//! # Usage
//!
//! ```rust,ignore
//! use trit_vsa::kernels::{TernaryBackend, get_backend, BackendConfig};
//!
//! let config = BackendConfig::auto();
//! let backend = get_backend(&config);
//!
//! let result = backend.bind(&vec_a, &vec_b)?;
//! ```

pub mod cpu;

#[cfg(feature = "cuda")]
pub mod cubecl;

// Burn backend stub for future integration
pub mod burn;

use crate::{PackedTritVec, Result, TernaryError};

// Re-export key types
pub use cpu::CpuBackend;

#[cfg(feature = "cuda")]
pub use cubecl::CubeclBackend;

pub use burn::BurnBackend;

/// Configuration for backend selection.
#[derive(Debug, Clone)]
pub struct BackendConfig {
    /// Preferred backend type.
    pub preferred: BackendPreference,
    /// Minimum dimensions for GPU dispatch (default: 4096).
    pub gpu_threshold: usize,
    /// Whether to use SIMD on CPU (default: true).
    pub use_simd: bool,
}

impl Default for BackendConfig {
    fn default() -> Self {
        Self::auto()
    }
}

impl BackendConfig {
    /// Create configuration with automatic backend selection.
    #[must_use]
    pub fn auto() -> Self {
        Self {
            preferred: BackendPreference::Auto,
            gpu_threshold: 4096,
            use_simd: true,
        }
    }

    /// Force CPU backend.
    #[must_use]
    pub fn cpu_only() -> Self {
        Self {
            preferred: BackendPreference::Cpu,
            gpu_threshold: usize::MAX,
            use_simd: true,
        }
    }

    /// Force GPU backend (requires `cuda` feature).
    #[must_use]
    pub fn gpu_only() -> Self {
        Self {
            preferred: BackendPreference::Gpu,
            gpu_threshold: 0,
            use_simd: false,
        }
    }

    /// Set GPU threshold for automatic selection.
    #[must_use]
    pub fn with_gpu_threshold(mut self, threshold: usize) -> Self {
        self.gpu_threshold = threshold;
        self
    }

    /// Enable or disable SIMD on CPU.
    #[must_use]
    pub fn with_simd(mut self, enabled: bool) -> Self {
        self.use_simd = enabled;
        self
    }
}

/// Backend preference for kernel execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BackendPreference {
    /// Automatically select based on problem size and availability.
    #[default]
    Auto,
    /// Force CPU execution.
    Cpu,
    /// Force GPU execution (requires cuda feature).
    Gpu,
    /// Use Burn backend (for future integration).
    Burn,
}

/// Input for random vector generation.
#[derive(Debug, Clone)]
pub struct RandomConfig {
    /// Vector dimension.
    pub dim: usize,
    /// Random seed.
    pub seed: u64,
}

impl RandomConfig {
    /// Create a new random configuration.
    #[must_use]
    pub fn new(dim: usize, seed: u64) -> Self {
        Self { dim, seed }
    }
}

/// Backend-agnostic trait for ternary VSA operations.
///
/// This trait defines all core operations that can be implemented by different
/// backends (CPU, CubeCL/CUDA, Burn, etc.).
///
/// # Implementors
///
/// - [`CpuBackend`]: CPU implementation with optional SIMD acceleration
/// - [`CubeclBackend`]: CUDA implementation via CubeCL (requires `cuda` feature)
/// - [`BurnBackend`]: Burn framework integration (stub for future)
///
/// # Thread Safety
///
/// All implementations must be `Send + Sync` to allow use across threads.
pub trait TernaryBackend: Send + Sync {
    /// Returns the backend name for debugging/logging.
    fn name(&self) -> &'static str;

    /// Returns true if this backend is available on the current system.
    fn is_available(&self) -> bool;

    /// Bind two vectors (composition operation).
    ///
    /// Implements balanced ternary binding: `result[i] = (a[i] - b[i]) mod 3`
    ///
    /// # Properties
    /// - `unbind(bind(a, b), b) == a`
    ///
    /// # Errors
    /// Returns error if vectors have mismatched dimensions.
    fn bind(&self, a: &PackedTritVec, b: &PackedTritVec) -> Result<PackedTritVec>;

    /// Unbind a vector (inverse of bind).
    ///
    /// Implements: `result[i] = (a[i] + b[i]) mod 3`
    ///
    /// # Errors
    /// Returns error if vectors have mismatched dimensions.
    fn unbind(&self, a: &PackedTritVec, b: &PackedTritVec) -> Result<PackedTritVec>;

    /// Bundle multiple vectors using majority voting.
    ///
    /// For each dimension, selects the majority trit value.
    /// Ties resolve to zero.
    ///
    /// # Errors
    /// Returns error if vectors have mismatched dimensions or input is empty.
    fn bundle(&self, vectors: &[&PackedTritVec]) -> Result<PackedTritVec>;

    /// Compute dot product similarity.
    ///
    /// Returns the sum of element-wise products.
    /// Range: [-n, +n] where n = dimension.
    ///
    /// # Errors
    /// Returns error if vectors have mismatched dimensions.
    fn dot_similarity(&self, a: &PackedTritVec, b: &PackedTritVec) -> Result<i32>;

    /// Compute cosine similarity.
    ///
    /// Returns: `dot(a, b) / (||a|| * ||b||)`
    /// Range: [-1.0, +1.0]
    ///
    /// # Errors
    /// Returns error if vectors have mismatched dimensions.
    fn cosine_similarity(&self, a: &PackedTritVec, b: &PackedTritVec) -> Result<f32>;

    /// Compute Hamming distance.
    ///
    /// Counts positions where vectors differ.
    /// Range: [0, n] where n = dimension.
    ///
    /// # Errors
    /// Returns error if vectors have mismatched dimensions.
    fn hamming_distance(&self, a: &PackedTritVec, b: &PackedTritVec) -> Result<usize>;

    /// Generate a random ternary vector.
    ///
    /// Uses the provided seed for reproducibility.
    fn random(&self, config: &RandomConfig) -> Result<PackedTritVec>;

    /// Negate a vector element-wise.
    ///
    /// Returns a new vector where all values are negated.
    fn negate(&self, a: &PackedTritVec) -> Result<PackedTritVec>;
}

/// Dynamic backend dispatcher.
///
/// Wraps any `TernaryBackend` implementation for dynamic dispatch.
pub struct DynamicBackend {
    inner: Box<dyn TernaryBackend>,
}

impl DynamicBackend {
    /// Create a new dynamic backend from a concrete implementation.
    pub fn new<B: TernaryBackend + 'static>(backend: B) -> Self {
        Self {
            inner: Box::new(backend),
        }
    }

    /// Get the underlying backend reference.
    #[must_use]
    pub fn inner(&self) -> &dyn TernaryBackend {
        &*self.inner
    }
}

impl TernaryBackend for DynamicBackend {
    fn name(&self) -> &'static str {
        self.inner.name()
    }

    fn is_available(&self) -> bool {
        self.inner.is_available()
    }

    fn bind(&self, a: &PackedTritVec, b: &PackedTritVec) -> Result<PackedTritVec> {
        self.inner.bind(a, b)
    }

    fn unbind(&self, a: &PackedTritVec, b: &PackedTritVec) -> Result<PackedTritVec> {
        self.inner.unbind(a, b)
    }

    fn bundle(&self, vectors: &[&PackedTritVec]) -> Result<PackedTritVec> {
        self.inner.bundle(vectors)
    }

    fn dot_similarity(&self, a: &PackedTritVec, b: &PackedTritVec) -> Result<i32> {
        self.inner.dot_similarity(a, b)
    }

    fn cosine_similarity(&self, a: &PackedTritVec, b: &PackedTritVec) -> Result<f32> {
        self.inner.cosine_similarity(a, b)
    }

    fn hamming_distance(&self, a: &PackedTritVec, b: &PackedTritVec) -> Result<usize> {
        self.inner.hamming_distance(a, b)
    }

    fn random(&self, config: &RandomConfig) -> Result<PackedTritVec> {
        self.inner.random(config)
    }

    fn negate(&self, a: &PackedTritVec) -> Result<PackedTritVec> {
        self.inner.negate(a)
    }
}

/// Get the appropriate backend based on configuration.
///
/// This function selects the best available backend based on:
/// 1. User preference (if specified)
/// 2. Feature availability (cuda, burn)
/// 3. Hardware detection (GPU presence)
///
/// # Arguments
///
/// * `config` - Backend configuration
///
/// # Returns
///
/// A boxed backend implementation ready for use.
#[must_use]
pub fn get_backend(config: &BackendConfig) -> DynamicBackend {
    match config.preferred {
        BackendPreference::Cpu => DynamicBackend::new(CpuBackend::new(config.use_simd)),

        #[cfg(feature = "cuda")]
        BackendPreference::Gpu => {
            let cubecl = CubeclBackend::new();
            if cubecl.is_available() {
                DynamicBackend::new(cubecl)
            } else {
                // Fall back to CPU if GPU not available
                DynamicBackend::new(CpuBackend::new(config.use_simd))
            }
        }

        #[cfg(not(feature = "cuda"))]
        BackendPreference::Gpu => {
            // No CUDA support compiled in, fall back to CPU
            DynamicBackend::new(CpuBackend::new(config.use_simd))
        }

        BackendPreference::Burn => {
            // Burn backend is a stub - fall back to CPU for now
            DynamicBackend::new(CpuBackend::new(config.use_simd))
        }

        BackendPreference::Auto => {
            // Auto-selection: try GPU first if available and configured
            #[cfg(feature = "cuda")]
            {
                let cubecl = CubeclBackend::new();
                if cubecl.is_available() {
                    return DynamicBackend::new(cubecl);
                }
            }
            // Fall back to CPU
            DynamicBackend::new(CpuBackend::new(config.use_simd))
        }
    }
}

/// Get a backend appropriate for the given problem size.
///
/// This is a convenience function that considers both configuration
/// and problem size when selecting a backend.
///
/// # Arguments
///
/// * `config` - Backend configuration
/// * `problem_size` - Size of the operation (typically vector dimension)
///
/// # Returns
///
/// A backend optimized for the given problem size.
#[must_use]
pub fn get_backend_for_size(config: &BackendConfig, problem_size: usize) -> DynamicBackend {
    // For auto selection, only use GPU if problem size exceeds threshold
    if config.preferred == BackendPreference::Auto && problem_size < config.gpu_threshold {
        return DynamicBackend::new(CpuBackend::new(config.use_simd));
    }

    get_backend(config)
}

/// Check dimension compatibility and return error if mismatched.
pub(crate) fn check_dimensions(a: &PackedTritVec, b: &PackedTritVec) -> Result<()> {
    if a.len() != b.len() {
        return Err(TernaryError::DimensionMismatch {
            expected: a.len(),
            actual: b.len(),
        });
    }
    Ok(())
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
    fn test_backend_config_default() {
        let config = BackendConfig::default();
        assert_eq!(config.preferred, BackendPreference::Auto);
        assert_eq!(config.gpu_threshold, 4096);
        assert!(config.use_simd);
    }

    #[test]
    fn test_backend_config_cpu_only() {
        let config = BackendConfig::cpu_only();
        assert_eq!(config.preferred, BackendPreference::Cpu);
    }

    #[test]
    fn test_get_backend_cpu() {
        let config = BackendConfig::cpu_only();
        let backend = get_backend(&config);
        assert_eq!(backend.name(), "cpu");
        assert!(backend.is_available());
    }

    #[test]
    fn test_cpu_backend_bind() {
        let config = BackendConfig::cpu_only();
        let backend = get_backend(&config);

        let a = make_test_vector(&[1, 0, -1, 1]);
        let b = make_test_vector(&[1, -1, 0, -1]);

        let result = backend.bind(&a, &b).unwrap();
        assert_eq!(result.len(), 4);

        // Verify bind/unbind inverse property
        let recovered = backend.unbind(&result, &b).unwrap();
        for i in 0..4 {
            assert_eq!(recovered.get(i), a.get(i), "mismatch at position {i}");
        }
    }

    #[test]
    fn test_cpu_backend_bundle() {
        let config = BackendConfig::cpu_only();
        let backend = get_backend(&config);

        let a = make_test_vector(&[1, 1, -1, 0]);
        let b = make_test_vector(&[1, -1, -1, 1]);
        let c = make_test_vector(&[1, 0, 1, -1]);

        let result = backend.bundle(&[&a, &b, &c]).unwrap();

        // Position 0: 1, 1, 1 -> majority is 1
        assert_eq!(result.get(0), Trit::P);
        // Position 2: -1, -1, 1 -> majority is -1
        assert_eq!(result.get(2), Trit::N);
    }

    #[test]
    fn test_cpu_backend_dot_similarity() {
        let config = BackendConfig::cpu_only();
        let backend = get_backend(&config);

        let a = make_test_vector(&[1, 0, -1, 1]);
        let b = make_test_vector(&[1, -1, -1, 0]);

        let dot = backend.dot_similarity(&a, &b).unwrap();
        // Expected: 1*1 + 0*(-1) + (-1)*(-1) + 1*0 = 1 + 0 + 1 + 0 = 2
        assert_eq!(dot, 2);
    }

    #[test]
    fn test_cpu_backend_hamming_distance() {
        let config = BackendConfig::cpu_only();
        let backend = get_backend(&config);

        let a = make_test_vector(&[1, 0, -1, 1]);
        let b = make_test_vector(&[1, -1, -1, 0]);

        let dist = backend.hamming_distance(&a, &b).unwrap();
        // Positions 1 and 3 differ
        assert_eq!(dist, 2);
    }

    #[test]
    fn test_cpu_backend_random() {
        let config = BackendConfig::cpu_only();
        let backend = get_backend(&config);

        let random_config = RandomConfig::new(100, 42);
        let result = backend.random(&random_config).unwrap();

        assert_eq!(result.len(), 100);

        // Check distribution (statistical test)
        let pos = result.count_positive();
        let neg = result.count_negative();
        let zero = result.len() - pos - neg;

        assert!(pos > 10, "too few positive: {pos}");
        assert!(neg > 10, "too few negative: {neg}");
        assert!(zero > 10, "too few zero: {zero}");
    }

    #[test]
    fn test_dimension_mismatch() {
        let config = BackendConfig::cpu_only();
        let backend = get_backend(&config);

        let a = make_test_vector(&[1, 0, -1]);
        let b = make_test_vector(&[1, -1]);

        assert!(backend.bind(&a, &b).is_err());
        assert!(backend.unbind(&a, &b).is_err());
        assert!(backend.dot_similarity(&a, &b).is_err());
        assert!(backend.hamming_distance(&a, &b).is_err());
    }

    #[test]
    fn test_get_backend_for_size() {
        let config = BackendConfig::auto().with_gpu_threshold(1000);

        // Small problem should use CPU
        let backend_small = get_backend_for_size(&config, 500);
        assert_eq!(backend_small.name(), "cpu");

        // Large problem would use GPU if available, but falls back to CPU
        let backend_large = get_backend_for_size(&config, 2000);
        // On systems without CUDA, this will still be CPU
        assert!(backend_large.is_available());
    }
}
