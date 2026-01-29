//! Vector Symbolic Architecture (VSA) operations.
//!
//! Provides hyperdimensional computing primitives for gradient compression,
//! associative memory, and symbolic reasoning.
//!
//! # Operations
//!
//! - **Bind**: Associative composition (XOR-like for ternary)
//! - **Bundle**: Superposition via majority voting
//! - **Similarity**: Cosine, dot product, Hamming distance
//!
//! # Example
//!
//! ```rust,ignore
//! use tritter_accel::core::vsa::{VsaOps, VsaConfig};
//! use candle_core::Device;
//!
//! let config = VsaConfig::default();
//! let ops = VsaOps::new(config)?;
//!
//! // Create random ternary vectors
//! let a = ops.random(10000, 42)?;
//! let b = ops.random(10000, 43)?;
//!
//! // Bind creates association
//! let bound = ops.bind(&a, &b)?;
//!
//! // Unbind recovers original
//! let recovered = ops.unbind(&bound, &b)?;
//! assert!(ops.cosine_similarity(&a, &recovered)? > 0.9);
//! ```

use candle_core::Device;
use thiserror::Error;
use trit_vsa::{PackedTritVec, Trit};

#[cfg(feature = "cuda")]
use trit_vsa::gpu::{
    GpuBind, GpuBundle, GpuCosineSimilarity, GpuDispatchable, GpuDotSimilarity, GpuHammingDistance,
    GpuRandom, GpuUnbind, RandomInput,
};

/// Errors from VSA operations.
#[derive(Debug, Error)]
pub enum VsaError {
    /// Vectors have mismatched dimensions.
    #[error("dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    /// Invalid ternary value.
    #[error("invalid value {value} at index {index}")]
    InvalidValue { value: i8, index: usize },

    /// GPU operation failed.
    #[error("GPU error: {0}")]
    Gpu(String),

    /// Empty input.
    #[error("empty input")]
    EmptyInput,
}

/// Configuration for VSA operations.
#[derive(Debug, Clone)]
pub struct VsaConfig {
    /// Preferred device for computation.
    pub device: DevicePreference,
}

/// Device preference for VSA operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DevicePreference {
    /// Use GPU if available, fall back to CPU.
    Auto,
    /// Force GPU (error if unavailable).
    Gpu,
    /// Force CPU.
    Cpu,
}

impl Default for VsaConfig {
    fn default() -> Self {
        Self {
            device: DevicePreference::Auto,
        }
    }
}

impl VsaConfig {
    /// Set device preference.
    pub fn with_device(mut self, device: DevicePreference) -> Self {
        self.device = device;
        self
    }
}

/// VSA operations handler.
///
/// Provides ternary VSA operations with automatic CPU/GPU dispatch.
#[derive(Debug, Clone)]
pub struct VsaOps {
    config: VsaConfig,
}

impl VsaOps {
    /// Create new VSA operations handler.
    pub fn new(config: VsaConfig) -> Self {
        Self { config }
    }

    /// Get the effective device for computation.
    fn get_device(&self) -> Result<Device, VsaError> {
        match self.config.device {
            DevicePreference::Cpu => Ok(Device::Cpu),
            DevicePreference::Gpu => {
                #[cfg(feature = "cuda")]
                {
                    Device::cuda_if_available(0).map_err(|e| VsaError::Gpu(e.to_string()))
                }
                #[cfg(not(feature = "cuda"))]
                {
                    Err(VsaError::Gpu(
                        "CUDA not compiled. Rebuild with --features cuda".to_string(),
                    ))
                }
            }
            DevicePreference::Auto => {
                #[cfg(feature = "cuda")]
                {
                    Ok(Device::cuda_if_available(0).unwrap_or(Device::Cpu))
                }
                #[cfg(not(feature = "cuda"))]
                {
                    Ok(Device::Cpu)
                }
            }
        }
    }

    /// Generate a random ternary vector.
    ///
    /// # Arguments
    ///
    /// * `dim` - Vector dimension
    /// * `seed` - Random seed for reproducibility
    pub fn random(&self, dim: usize, seed: u32) -> Result<PackedTritVec, VsaError> {
        if dim == 0 {
            return Err(VsaError::EmptyInput);
        }

        let device = self.get_device()?;

        #[cfg(feature = "cuda")]
        {
            if matches!(device, Device::Cuda(_)) {
                let input = RandomInput::new(dim, seed);
                return GpuRandom
                    .dispatch(&input, &device)
                    .map_err(|e| VsaError::Gpu(e.to_string()));
            }
        }

        // CPU fallback
        let _ = device; // silence unused warning
        Ok(cpu_random(dim, seed))
    }

    /// Bind two ternary vectors (association).
    ///
    /// Bind is the composition operation, creating associations between vectors.
    /// It is commutative and associative.
    pub fn bind(&self, a: &PackedTritVec, b: &PackedTritVec) -> Result<PackedTritVec, VsaError> {
        if a.len() != b.len() {
            return Err(VsaError::DimensionMismatch {
                expected: a.len(),
                actual: b.len(),
            });
        }

        let device = self.get_device()?;

        #[cfg(feature = "cuda")]
        {
            if matches!(device, Device::Cuda(_)) {
                return GpuBind
                    .dispatch(&(a.clone(), b.clone()), &device)
                    .map_err(|e| VsaError::Gpu(e.to_string()));
            }
        }

        let _ = device;
        Ok(cpu_bind(a, b))
    }

    /// Unbind two ternary vectors (inverse association).
    ///
    /// If bound = bind(a, b), then unbind(bound, b) recovers a.
    pub fn unbind(&self, bound: &PackedTritVec, key: &PackedTritVec) -> Result<PackedTritVec, VsaError> {
        if bound.len() != key.len() {
            return Err(VsaError::DimensionMismatch {
                expected: bound.len(),
                actual: key.len(),
            });
        }

        let device = self.get_device()?;

        #[cfg(feature = "cuda")]
        {
            if matches!(device, Device::Cuda(_)) {
                return GpuUnbind
                    .dispatch(&(bound.clone(), key.clone()), &device)
                    .map_err(|e| VsaError::Gpu(e.to_string()));
            }
        }

        let _ = device;
        // For ternary VSA, unbind is the same as bind (self-inverse)
        Ok(cpu_bind(bound, key))
    }

    /// Bundle multiple vectors (superposition).
    ///
    /// Combines vectors via majority voting at each dimension.
    /// The result is similar to all input vectors.
    pub fn bundle(&self, vectors: &[PackedTritVec]) -> Result<PackedTritVec, VsaError> {
        if vectors.is_empty() {
            return Err(VsaError::EmptyInput);
        }

        let dim = vectors[0].len();
        for (i, v) in vectors.iter().enumerate() {
            if v.len() != dim {
                return Err(VsaError::DimensionMismatch {
                    expected: dim,
                    actual: v.len(),
                });
            }
            let _ = i; // used for error reporting if needed
        }

        let device = self.get_device()?;

        #[cfg(feature = "cuda")]
        {
            if matches!(device, Device::Cuda(_)) {
                return GpuBundle
                    .dispatch(&vectors.to_vec(), &device)
                    .map_err(|e| VsaError::Gpu(e.to_string()));
            }
        }

        let _ = device;
        Ok(cpu_bundle(vectors))
    }

    /// Compute cosine similarity between two vectors.
    ///
    /// Returns a value in [-1, 1].
    pub fn cosine_similarity(&self, a: &PackedTritVec, b: &PackedTritVec) -> Result<f32, VsaError> {
        if a.len() != b.len() {
            return Err(VsaError::DimensionMismatch {
                expected: a.len(),
                actual: b.len(),
            });
        }

        let device = self.get_device()?;

        #[cfg(feature = "cuda")]
        {
            if matches!(device, Device::Cuda(_)) {
                return GpuCosineSimilarity
                    .dispatch(&(a.clone(), b.clone()), &device)
                    .map_err(|e| VsaError::Gpu(e.to_string()));
            }
        }

        let _ = device;
        Ok(cpu_cosine_similarity(a, b))
    }

    /// Compute dot product between two vectors.
    pub fn dot(&self, a: &PackedTritVec, b: &PackedTritVec) -> Result<i32, VsaError> {
        if a.len() != b.len() {
            return Err(VsaError::DimensionMismatch {
                expected: a.len(),
                actual: b.len(),
            });
        }

        let device = self.get_device()?;

        #[cfg(feature = "cuda")]
        {
            if matches!(device, Device::Cuda(_)) {
                return GpuDotSimilarity
                    .dispatch(&(a.clone(), b.clone()), &device)
                    .map_err(|e| VsaError::Gpu(e.to_string()));
            }
        }

        let _ = device;
        Ok(a.dot(b))
    }

    /// Compute Hamming distance between two vectors.
    ///
    /// Returns the number of positions where the vectors differ.
    pub fn hamming_distance(&self, a: &PackedTritVec, b: &PackedTritVec) -> Result<usize, VsaError> {
        if a.len() != b.len() {
            return Err(VsaError::DimensionMismatch {
                expected: a.len(),
                actual: b.len(),
            });
        }

        let device = self.get_device()?;

        #[cfg(feature = "cuda")]
        {
            if matches!(device, Device::Cuda(_)) {
                return GpuHammingDistance
                    .dispatch(&(a.clone(), b.clone()), &device)
                    .map_err(|e| VsaError::Gpu(e.to_string()));
            }
        }

        let _ = device;
        Ok(cpu_hamming_distance(a, b))
    }

    /// Convert i8 slice to PackedTritVec.
    pub fn from_i8(&self, values: &[i8]) -> Result<PackedTritVec, VsaError> {
        let mut packed = PackedTritVec::new(values.len());
        for (i, &v) in values.iter().enumerate() {
            let trit = match v {
                1 => Trit::P,
                0 => Trit::Z,
                -1 => Trit::N,
                _ => return Err(VsaError::InvalidValue { value: v, index: i }),
            };
            packed.set(i, trit);
        }
        Ok(packed)
    }

    /// Convert PackedTritVec to i8 Vec.
    pub fn to_i8(&self, packed: &PackedTritVec) -> Vec<i8> {
        let mut result = Vec::with_capacity(packed.len());
        for i in 0..packed.len() {
            result.push(packed.get(i).value());
        }
        result
    }
}

// CPU implementations

fn cpu_random(dim: usize, seed: u32) -> PackedTritVec {
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    let mut rng = ChaCha8Rng::seed_from_u64(u64::from(seed));
    let mut packed = PackedTritVec::new(dim);

    for i in 0..dim {
        let r: f32 = rng.gen();
        let trit = if r < 0.333 {
            Trit::N
        } else if r < 0.666 {
            Trit::Z
        } else {
            Trit::P
        };
        packed.set(i, trit);
    }

    packed
}

fn cpu_bind(a: &PackedTritVec, b: &PackedTritVec) -> PackedTritVec {
    // Ternary multiplication table:
    // P * P = P, P * Z = Z, P * N = N
    // Z * _ = Z
    // N * P = N, N * Z = Z, N * N = P
    let mut result = PackedTritVec::new(a.len());
    for i in 0..a.len() {
        let va = a.get(i).value();
        let vb = b.get(i).value();
        let prod = va * vb;
        let trit = match prod {
            1 => Trit::P,
            -1 => Trit::N,
            _ => Trit::Z,
        };
        result.set(i, trit);
    }
    result
}

fn cpu_bundle(vectors: &[PackedTritVec]) -> PackedTritVec {
    let dim = vectors[0].len();
    let mut result = PackedTritVec::new(dim);

    for i in 0..dim {
        let mut pos_count = 0i32;
        let mut neg_count = 0i32;

        for v in vectors {
            match v.get(i) {
                Trit::P => pos_count += 1,
                Trit::N => neg_count += 1,
                Trit::Z => {}
            }
        }

        let trit = if pos_count > neg_count {
            Trit::P
        } else if neg_count > pos_count {
            Trit::N
        } else {
            Trit::Z
        };
        result.set(i, trit);
    }

    result
}

fn cpu_cosine_similarity(a: &PackedTritVec, b: &PackedTritVec) -> f32 {
    let dot = a.dot(b) as f32;

    // Count non-zero elements for normalization
    let mut norm_a_sq = 0i32;
    let mut norm_b_sq = 0i32;

    for i in 0..a.len() {
        let va = a.get(i).value() as i32;
        let vb = b.get(i).value() as i32;
        norm_a_sq += va * va;
        norm_b_sq += vb * vb;
    }

    if norm_a_sq == 0 || norm_b_sq == 0 {
        return 0.0;
    }

    dot / ((norm_a_sq as f32).sqrt() * (norm_b_sq as f32).sqrt())
}

fn cpu_hamming_distance(a: &PackedTritVec, b: &PackedTritVec) -> usize {
    let mut distance = 0;
    for i in 0..a.len() {
        if a.get(i) != b.get(i) {
            distance += 1;
        }
    }
    distance
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bind_unbind_roundtrip() {
        let ops = VsaOps::new(VsaConfig::default().with_device(DevicePreference::Cpu));

        // Use non-zero vectors for perfect recovery
        // (zeros in either vector cause information loss)
        let a = ops.from_i8(&[1, -1, 1, -1, 1, -1, 1, -1]).unwrap();
        let b = ops.from_i8(&[1, 1, -1, -1, 1, 1, -1, -1]).unwrap();

        let bound = ops.bind(&a, &b).unwrap();
        let recovered = ops.unbind(&bound, &b).unwrap();

        // Should recover a exactly when no zeros involved
        for i in 0..a.len() {
            assert_eq!(a.get(i), recovered.get(i));
        }
    }

    #[test]
    fn test_bundle_majority() {
        let ops = VsaOps::new(VsaConfig::default().with_device(DevicePreference::Cpu));

        // Create 3 vectors with known values
        let v1 = ops.from_i8(&[1, 1, -1, 0]).unwrap();
        let v2 = ops.from_i8(&[1, -1, -1, 1]).unwrap();
        let v3 = ops.from_i8(&[1, 0, 1, -1]).unwrap();

        let bundled = ops.bundle(&[v1, v2, v3]).unwrap();
        let result = ops.to_i8(&bundled);

        // Position 0: [1, 1, 1] -> majority 1
        assert_eq!(result[0], 1);
        // Position 2: [-1, -1, 1] -> majority -1
        assert_eq!(result[2], -1);
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let ops = VsaOps::new(VsaConfig::default().with_device(DevicePreference::Cpu));

        let a = ops.random(1000, 42).unwrap();
        let sim = ops.cosine_similarity(&a, &a).unwrap();

        // Identical vectors should have similarity 1.0
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_hamming_distance() {
        let ops = VsaOps::new(VsaConfig::default().with_device(DevicePreference::Cpu));

        let a = ops.from_i8(&[1, 0, -1, 1]).unwrap();
        let b = ops.from_i8(&[1, -1, -1, 0]).unwrap();

        // Differences at positions 1, 3
        let dist = ops.hamming_distance(&a, &b).unwrap();
        assert_eq!(dist, 2);
    }
}
