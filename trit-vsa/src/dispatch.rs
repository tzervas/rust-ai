//! Smart kernel dispatch for optimal ternary operations.
//!
//! This module provides intelligent routing between different ternary representations
//! and kernel implementations based on operation type, data characteristics, and hardware.
//!
//! # Architecture
//!
//! This module integrates with the modular [`kernels`](crate::kernels) backend system:
//!
//! ```text
//! TritVector                   DispatchConfig
//!     |                              |
//!     v                              v
//! +----------+                +--------------+
//! | dispatch |  ----------->  | BackendConfig|
//! +----------+                +--------------+
//!     |                              |
//!     v                              v
//! +-------------------------------------------+
//! |           TernaryBackend (trait)          |
//! +-------------------------------------------+
//!     |           |              |
//!     v           v              v
//! +------+   +--------+     +------+
//! |  CPU |   | CubeCL |     | Burn |
//! +------+   +--------+     +------+
//! ```
//!
//! # Ternary Representations
//!
//! ## Tritsliced (Implied Zero with Positive/Negative Planes)
//!
//! Two parallel bit planes where:
//! - `+plane[i] = 1` indicates trit `+1`
//! - `-plane[i] = 1` indicates trit `-1`
//! - Both `0` indicates trit `0` (implied zero)
//!
//! **Optimal for:**
//! - Dot products (popcount-based)
//! - Element-wise bind/unbind
//! - Bundle (majority voting)
//! - Dense vectors (< 90% zeros)
//!
//! ## Tritpacked (2-bit per trit)
//!
//! Each trit encoded as 2 bits: `00` = -1, `01` = 0, `10` = +1
//!
//! **Optimal for:**
//! - Sequential access patterns
//! - Serialization/deserialization
//! - Mixed arithmetic operations
//! - Memory-constrained scenarios
//!
//! ## Sparse (COO Format)
//!
//! Separate index lists for positive and negative values.
//!
//! **Optimal for:**
//! - Very sparse vectors (> 90% zeros)
//! - Similarity between sparse vectors
//! - Memory efficiency for high-dimensional sparse data
//!
//! # Dispatch Strategy
//!
//! The dispatcher selects the optimal kernel based on:
//! 1. **Sparsity**: Sparse format for > 90% zeros
//! 2. **Operation type**: Popcount ops -> tritsliced, arithmetic -> tritpacked
//! 3. **Vector size**: GPU for large (> 4096 dims), CPU for small
//! 4. **Hardware**: SIMD availability, GPU presence
//!
//! # Example
//!
//! ```rust,ignore
//! use trit_vsa::dispatch::{TritVector, DispatchConfig, Operation};
//!
//! // Automatic format selection
//! let a = TritVector::from_packed(packed_vec);
//! let b = TritVector::from_packed(other_vec);
//!
//! // Dispatcher chooses optimal kernel
//! let similarity = a.cosine_similarity(&b, &DispatchConfig::auto());
//!
//! // Force specific format
//! let config = DispatchConfig::new()
//!     .prefer_format(Format::Tritsliced)
//!     .gpu_threshold(8192);
//! let result = a.bind(&b, &config);
//! ```
//!
//! # Using the Modular Backend System
//!
//! For more control, use the [`kernels`](crate::kernels) module directly:
//!
//! ```rust,ignore
//! use trit_vsa::kernels::{get_backend, BackendConfig, TernaryBackend};
//!
//! let config = BackendConfig::auto();
//! let backend = get_backend(&config);
//!
//! let result = backend.bind(&a, &b)?;
//! let similarity = backend.dot_similarity(&a, &b)?;
//! ```

use crate::kernels::{self, BackendConfig, BackendPreference, TernaryBackend};
use crate::{PackedTritVec, Result, SparseVec, TernaryError, Trit};

/// Preferred kernel format for operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Format {
    /// Tritsliced: two bit planes (optimal for popcount operations)
    #[default]
    Tritsliced,
    /// Tritpacked: 2 bits per trit (optimal for sequential access)
    Tritpacked,
    /// Sparse: COO format (optimal for > 90% zeros)
    Sparse,
    /// Automatic selection based on data characteristics
    Auto,
}

/// Device preference for computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DevicePreference {
    /// Automatic GPU/CPU selection based on size
    #[default]
    Auto,
    /// Force CPU execution
    Cpu,
    /// Force GPU execution (requires `cuda` feature)
    Gpu,
}

/// Configuration for kernel dispatch.
#[derive(Debug, Clone)]
pub struct DispatchConfig {
    /// Preferred format for operations
    pub format: Format,
    /// Device preference
    pub device: DevicePreference,
    /// Sparsity threshold for automatic sparse selection (default: 0.90)
    pub sparse_threshold: f32,
    /// Minimum dimensions for GPU dispatch (default: 4096)
    pub gpu_threshold: usize,
    /// Enable format caching for repeated operations
    pub cache_conversions: bool,
}

impl Default for DispatchConfig {
    fn default() -> Self {
        Self::auto()
    }
}

impl DispatchConfig {
    /// Create a new configuration with automatic settings.
    #[must_use]
    pub fn auto() -> Self {
        Self {
            format: Format::Auto,
            device: DevicePreference::Auto,
            sparse_threshold: 0.90,
            gpu_threshold: 4096,
            cache_conversions: true,
        }
    }

    /// Create a CPU-only configuration.
    #[must_use]
    pub fn cpu_only() -> Self {
        Self {
            device: DevicePreference::Cpu,
            ..Self::auto()
        }
    }

    /// Set preferred format.
    #[must_use]
    pub fn with_format(mut self, format: Format) -> Self {
        self.format = format;
        self
    }

    /// Set device preference.
    #[must_use]
    pub fn with_device(mut self, device: DevicePreference) -> Self {
        self.device = device;
        self
    }

    /// Set sparsity threshold for automatic sparse format selection.
    #[must_use]
    pub fn with_sparse_threshold(mut self, threshold: f32) -> Self {
        self.sparse_threshold = threshold;
        self
    }

    /// Set minimum dimensions for GPU dispatch.
    #[must_use]
    pub fn with_gpu_threshold(mut self, threshold: usize) -> Self {
        self.gpu_threshold = threshold;
        self
    }
}

/// Operation types for dispatch decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Operation {
    /// Dot product (popcount-optimal)
    Dot,
    /// Cosine similarity
    Similarity,
    /// Bind operation (XOR-like composition)
    Bind,
    /// Unbind operation (inverse of bind)
    Unbind,
    /// Bundle (majority voting)
    Bundle,
    /// Element-wise negation
    Negate,
    /// Hamming distance
    Hamming,
}

impl Operation {
    /// Returns the preferred format for this operation.
    #[must_use]
    pub fn preferred_format(self) -> Format {
        match self {
            // Popcount-based operations prefer tritsliced
            Operation::Dot | Operation::Similarity | Operation::Hamming => Format::Tritsliced,
            // Element-wise operations work well with tritsliced
            Operation::Bind | Operation::Unbind | Operation::Negate => Format::Tritsliced,
            // Bundle needs to track counts, tritsliced is fine
            Operation::Bundle => Format::Tritsliced,
        }
    }

    /// Returns true if this operation can benefit from sparse representation.
    #[must_use]
    pub fn benefits_from_sparse(self) -> bool {
        matches!(self, Operation::Dot | Operation::Similarity)
    }
}

/// Unified ternary vector type with smart dispatch.
#[derive(Debug, Clone)]
pub enum TritVector {
    /// Tritsliced format ([`PackedTritVec`])
    Sliced(PackedTritVec),
    /// Sparse format ([`SparseVec`])
    Sparse(SparseVec),
}

impl TritVector {
    /// Create a new zero vector in tritsliced format.
    #[must_use]
    pub fn new(dims: usize) -> Self {
        Self::Sliced(PackedTritVec::new(dims))
    }

    /// Create from a `PackedTritVec`.
    #[must_use]
    pub fn from_packed(packed: PackedTritVec) -> Self {
        Self::Sliced(packed)
    }

    /// Create from a `SparseVec`.
    #[must_use]
    pub fn from_sparse(sparse: SparseVec) -> Self {
        Self::Sparse(sparse)
    }

    /// Get the number of dimensions.
    #[must_use]
    pub fn dims(&self) -> usize {
        match self {
            Self::Sliced(p) => p.len(),
            Self::Sparse(s) => s.num_dims(),
        }
    }

    /// Compute sparsity (fraction of zeros).
    #[must_use]
    pub fn sparsity(&self) -> f32 {
        match self {
            Self::Sliced(p) => p.sparsity(),
            Self::Sparse(s) => s.sparsity(),
        }
    }

    /// Get value at index.
    #[must_use]
    pub fn get(&self, idx: usize) -> Trit {
        match self {
            Self::Sliced(p) => p.get(idx),
            Self::Sparse(s) => s.get(idx),
        }
    }

    /// Set value at index (may require format conversion).
    pub fn set(&mut self, idx: usize, value: Trit) {
        match self {
            Self::Sliced(p) => p.set(idx, value),
            Self::Sparse(s) => s.set(idx, value),
        }
    }

    /// Convert to `PackedTritVec`.
    #[must_use]
    pub fn to_packed(&self) -> PackedTritVec {
        match self {
            Self::Sliced(p) => p.clone(),
            Self::Sparse(s) => s.to_packed(),
        }
    }

    /// Convert to `SparseVec`.
    #[must_use]
    pub fn to_sparse(&self) -> SparseVec {
        match self {
            Self::Sliced(p) => SparseVec::from_packed(p),
            Self::Sparse(s) => s.clone(),
        }
    }

    /// Select optimal format based on operation and data characteristics.
    fn select_format(
        &self,
        other: Option<&Self>,
        op: Operation,
        config: &DispatchConfig,
    ) -> Format {
        // Explicit format preference overrides auto-selection
        if config.format != Format::Auto {
            return config.format;
        }

        // Check if sparse format would be beneficial
        let self_sparse = self.sparsity() > config.sparse_threshold;
        let other_sparse = other.is_some_and(|o| o.sparsity() > config.sparse_threshold);

        if op.benefits_from_sparse() && self_sparse && other_sparse {
            return Format::Sparse;
        }

        // Default to operation's preferred format
        op.preferred_format()
    }

    /// Convert [`DispatchConfig`] to [`BackendConfig`] for the kernels module.
    fn to_backend_config(config: &DispatchConfig) -> BackendConfig {
        let preferred = match config.device {
            DevicePreference::Cpu => BackendPreference::Cpu,
            DevicePreference::Gpu => BackendPreference::Gpu,
            DevicePreference::Auto => BackendPreference::Auto,
        };

        BackendConfig {
            preferred,
            gpu_threshold: config.gpu_threshold,
            use_simd: true,
        }
    }

    /// Get the appropriate backend based on config and problem size.
    fn get_backend_for_config(&self, config: &DispatchConfig) -> kernels::DynamicBackend {
        let backend_config = Self::to_backend_config(config);
        kernels::get_backend_for_size(&backend_config, self.dims())
    }

    /// Compute dot product with smart dispatch.
    ///
    /// Uses the modular backend system for automatic CPU/GPU selection.
    ///
    /// # Errors
    ///
    /// Returns error if vectors have mismatched dimensions.
    pub fn dot(&self, other: &Self, config: &DispatchConfig) -> Result<i32> {
        if self.dims() != other.dims() {
            return Err(TernaryError::DimensionMismatch {
                expected: self.dims(),
                actual: other.dims(),
            });
        }

        let format = self.select_format(Some(other), Operation::Dot, config);

        match format {
            Format::Sparse => {
                // Sparse path: use direct sparse dot product
                let a = self.to_sparse();
                let b = other.to_sparse();
                Ok(a.dot(&b))
            }
            Format::Tritsliced | Format::Tritpacked | Format::Auto => {
                // Use modular backend system
                let a = self.to_packed();
                let b = other.to_packed();
                let backend = self.get_backend_for_config(config);
                backend.dot_similarity(&a, &b)
            }
        }
    }

    /// Compute cosine similarity with smart dispatch.
    ///
    /// Uses the modular backend system for automatic CPU/GPU selection.
    ///
    /// # Errors
    ///
    /// Returns error if vectors have mismatched dimensions.
    pub fn cosine_similarity(&self, other: &Self, config: &DispatchConfig) -> Result<f32> {
        if self.dims() != other.dims() {
            return Err(TernaryError::DimensionMismatch {
                expected: self.dims(),
                actual: other.dims(),
            });
        }

        let format = self.select_format(Some(other), Operation::Similarity, config);

        match format {
            Format::Sparse => {
                // Sparse path: use direct sparse cosine similarity
                let a = self.to_sparse();
                let b = other.to_sparse();
                Ok(crate::vsa::cosine_similarity_sparse(&a, &b))
            }
            Format::Tritsliced | Format::Tritpacked | Format::Auto => {
                // Use modular backend system
                let a = self.to_packed();
                let b = other.to_packed();
                let backend = self.get_backend_for_config(config);
                backend.cosine_similarity(&a, &b)
            }
        }
    }

    /// Bind operation with smart dispatch.
    ///
    /// Uses the modular backend system for automatic CPU/GPU selection.
    ///
    /// # Errors
    ///
    /// Returns error if vectors have mismatched dimensions.
    pub fn bind(&self, other: &Self, config: &DispatchConfig) -> Result<Self> {
        if self.dims() != other.dims() {
            return Err(TernaryError::DimensionMismatch {
                expected: self.dims(),
                actual: other.dims(),
            });
        }

        let a = self.to_packed();
        let b = other.to_packed();

        // Use modular backend system
        let backend = self.get_backend_for_config(config);
        let result = backend.bind(&a, &b)?;
        Ok(Self::Sliced(result))
    }

    /// Unbind operation with smart dispatch.
    ///
    /// Uses the modular backend system for automatic CPU/GPU selection.
    ///
    /// # Errors
    ///
    /// Returns error if vectors have mismatched dimensions.
    pub fn unbind(&self, other: &Self, config: &DispatchConfig) -> Result<Self> {
        if self.dims() != other.dims() {
            return Err(TernaryError::DimensionMismatch {
                expected: self.dims(),
                actual: other.dims(),
            });
        }

        let a = self.to_packed();
        let b = other.to_packed();

        // Use modular backend system
        let backend = self.get_backend_for_config(config);
        let result = backend.unbind(&a, &b)?;
        Ok(Self::Sliced(result))
    }

    /// Bundle (majority voting) with smart dispatch.
    ///
    /// Uses the modular backend system for automatic CPU/GPU selection.
    ///
    /// # Errors
    ///
    /// Returns error if vectors have mismatched dimensions.
    pub fn bundle(&self, other: &Self, config: &DispatchConfig) -> Result<Self> {
        if self.dims() != other.dims() {
            return Err(TernaryError::DimensionMismatch {
                expected: self.dims(),
                actual: other.dims(),
            });
        }

        let a = self.to_packed();
        let b = other.to_packed();

        // Use modular backend system
        let backend = self.get_backend_for_config(config);
        let result = backend.bundle(&[&a, &b])?;
        Ok(Self::Sliced(result))
    }

    /// Compute Hamming distance with smart dispatch.
    ///
    /// Uses the modular backend system for automatic CPU/GPU selection.
    ///
    /// # Errors
    ///
    /// Returns error if vectors have mismatched dimensions.
    pub fn hamming_distance(&self, other: &Self, config: &DispatchConfig) -> Result<usize> {
        if self.dims() != other.dims() {
            return Err(TernaryError::DimensionMismatch {
                expected: self.dims(),
                actual: other.dims(),
            });
        }

        let a = self.to_packed();
        let b = other.to_packed();

        // Use modular backend system
        let backend = self.get_backend_for_config(config);
        backend.hamming_distance(&a, &b)
    }

    /// Negate all elements.
    #[must_use]
    pub fn negate(&self) -> Self {
        match self {
            Self::Sliced(p) => Self::Sliced(p.negated()),
            Self::Sparse(s) => Self::Sparse(s.negated()),
        }
    }
}

impl From<PackedTritVec> for TritVector {
    fn from(packed: PackedTritVec) -> Self {
        Self::Sliced(packed)
    }
}

impl From<SparseVec> for TritVector {
    fn from(sparse: SparseVec) -> Self {
        Self::Sparse(sparse)
    }
}

/// Statistics about dispatch decisions for profiling.
#[derive(Debug, Default, Clone)]
pub struct DispatchStats {
    /// Number of times tritsliced format was used
    pub tritsliced_count: usize,
    /// Number of times sparse format was used
    pub sparse_count: usize,
    /// Number of GPU dispatches
    pub gpu_count: usize,
    /// Number of CPU dispatches
    pub cpu_count: usize,
    /// Number of format conversions
    pub conversion_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_vector(values: &[i8]) -> TritVector {
        let mut packed = PackedTritVec::new(values.len());
        for (i, &v) in values.iter().enumerate() {
            let trit = match v {
                -1 => Trit::N,
                0 => Trit::Z,
                1 => Trit::P,
                _ => panic!("Invalid trit value"),
            };
            packed.set(i, trit);
        }
        TritVector::Sliced(packed)
    }

    #[test]
    fn test_dispatch_config_default() {
        let config = DispatchConfig::auto();
        assert_eq!(config.format, Format::Auto);
        assert_eq!(config.device, DevicePreference::Auto);
        assert!((config.sparse_threshold - 0.90).abs() < f32::EPSILON);
        assert_eq!(config.gpu_threshold, 4096);
    }

    #[test]
    fn test_trit_vector_from_packed() {
        let packed = PackedTritVec::new(100);
        let tv = TritVector::from_packed(packed.clone());
        assert_eq!(tv.dims(), 100);
        assert!(matches!(tv, TritVector::Sliced(_)));
    }

    #[test]
    fn test_operation_preferred_format() {
        assert_eq!(Operation::Dot.preferred_format(), Format::Tritsliced);
        assert_eq!(Operation::Similarity.preferred_format(), Format::Tritsliced);
        assert_eq!(Operation::Bind.preferred_format(), Format::Tritsliced);
    }

    #[test]
    fn test_dot_product_dispatch() {
        let a = TritVector::new(100);
        let b = TritVector::new(100);
        let config = DispatchConfig::cpu_only();

        let result = a.dot(&b, &config);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 0);
    }

    #[test]
    fn test_dimension_mismatch() {
        let a = TritVector::new(100);
        let b = TritVector::new(200);
        let config = DispatchConfig::cpu_only();

        let result = a.dot(&b, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_bind_unbind_with_backend() {
        let a = make_test_vector(&[1, -1, 0, 1, -1, 0, 1]);
        let b = make_test_vector(&[-1, 1, 0, -1, 1, 0, -1]);
        let config = DispatchConfig::cpu_only();

        let bound = a.bind(&b, &config).unwrap();
        let recovered = bound.unbind(&b, &config).unwrap();

        // Verify bind/unbind inverse property
        for i in 0..a.dims() {
            assert_eq!(recovered.get(i), a.get(i), "mismatch at position {i}");
        }
    }

    #[test]
    fn test_bundle_with_backend() {
        let a = make_test_vector(&[1, 1, -1, 0, 0]);
        let b = make_test_vector(&[1, -1, -1, 1, -1]);
        let config = DispatchConfig::cpu_only();

        let bundled = a.bundle(&b, &config).unwrap();

        // Position 0: 1, 1 -> 1
        assert_eq!(bundled.get(0), Trit::P);
        // Position 1: 1, -1 -> 0 (tie)
        assert_eq!(bundled.get(1), Trit::Z);
        // Position 2: -1, -1 -> -1
        assert_eq!(bundled.get(2), Trit::N);
    }

    #[test]
    fn test_cosine_similarity_with_backend() {
        let a = make_test_vector(&[1, 1, -1, -1]);
        let config = DispatchConfig::cpu_only();

        let sim = a.cosine_similarity(&a, &config).unwrap();
        assert!((sim - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_hamming_distance_with_backend() {
        let a = make_test_vector(&[1, 0, -1, 1]);
        let b = make_test_vector(&[1, -1, -1, 0]);
        let config = DispatchConfig::cpu_only();

        let dist = a.hamming_distance(&b, &config).unwrap();
        // Positions 1 and 3 differ
        assert_eq!(dist, 2);
    }

    #[test]
    fn test_backend_config_conversion() {
        let config = DispatchConfig::cpu_only();
        let backend_config = TritVector::to_backend_config(&config);
        assert_eq!(backend_config.preferred, BackendPreference::Cpu);

        let config = DispatchConfig::auto().with_device(DevicePreference::Gpu);
        let backend_config = TritVector::to_backend_config(&config);
        assert_eq!(backend_config.preferred, BackendPreference::Gpu);
    }

    #[test]
    fn test_auto_backend_selection() {
        let small_vec = TritVector::new(100);
        let large_vec = TritVector::new(10000);

        let config = DispatchConfig::auto().with_gpu_threshold(5000);

        // Small vector should use CPU
        let backend = small_vec.get_backend_for_config(&config);
        assert!(backend.name().starts_with("cpu"));

        // Large vector would use GPU if available, otherwise CPU
        let backend = large_vec.get_backend_for_config(&config);
        assert!(backend.is_available());
    }
}
