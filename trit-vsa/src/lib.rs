//! Balanced ternary arithmetic library with bitsliced storage and VSA operations.
//!
//! This crate provides efficient representations and operations for balanced ternary
//! values {-1, 0, +1}. It supports both dense (bitsliced) and sparse storage formats,
//! along with Vector Symbolic Architecture (VSA) operations for hyperdimensional computing.
//!
//! # Features
//!
//! - **Core Types**: `Trit`, `Tryte3` (3 trits), `Word6` (6 trits)
//! - **Vector Storage**: `PackedTritVec` (bitsliced), `SparseVec` (COO format)
//! - **VSA Operations**: Bundle (majority), Bind (XOR-like), Similarity
//! - **SIMD**: Optional AVX2/NEON acceleration with the `simd` feature
//!
//! # Quick Start
//!
//! ```rust
//! use trit_vsa::{Trit, PackedTritVec, vsa};
//!
//! // Create a ternary vector
//! let mut vec = PackedTritVec::new(1000);
//! vec.set(0, Trit::P);   // +1
//! vec.set(1, Trit::N);   // -1
//! vec.set(2, Trit::Z);   // 0
//!
//! // Compute dot product
//! let other = PackedTritVec::new(1000);
//! let dot = vec.dot(&other);
//!
//! // VSA operations
//! let bundled = vsa::bundle(&vec, &other);
//! let similarity = vsa::cosine_similarity(&vec, &other);
//! ```
//!
//! # Representation
//!
//! Ternary values are stored using a bitsliced representation with two planes:
//!
//! ```text
//! Value | +plane | -plane
//! ------+--------+-------
//!   +1  |   1    |   0
//!    0  |   0    |   0
//!   -1  |   0    |   1
//! ```
//!
//! This enables efficient popcount-based operations like dot products.
//!
//! # Feature Flags
//!
//! - `default`: No additional features
//! - `simd`: Enable AVX2/NEON SIMD optimizations
//! - `cuda`: Enable GPU acceleration via CubeCL

#![warn(missing_docs)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_lossless)] // We use explicit casts for clarity

pub mod arithmetic;
pub mod dispatch;
mod error;
pub mod kernels;
mod packed;
pub mod simd;
mod sparse;
mod trit;
mod tryte;
pub mod vsa;
mod word;

#[cfg(feature = "cuda")]
pub mod gpu;

pub use dispatch::{DevicePreference, DispatchConfig, Format, Operation, TritVector};
pub use error::{Result, TernaryError};
pub use kernels::{
    get_backend, get_backend_for_size, BackendConfig, BackendPreference, CpuBackend,
    DynamicBackend, RandomConfig, TernaryBackend,
};
pub use packed::PackedTritVec;
pub use sparse::SparseVec;
pub use trit::Trit;
pub use tryte::{Tryte3, TRYTE3_MAX, TRYTE3_MIN};
pub use word::{Word6, WORD6_MAX, WORD6_MIN};

#[cfg(feature = "cuda")]
pub use gpu::{
    warn_if_cpu, GpuBind, GpuBundle, GpuCosineSimilarity, GpuDispatchable, GpuDotSimilarity,
    GpuError, GpuHammingDistance, GpuRandom, GpuResult, GpuUnbind, RandomInput,
};

#[cfg(feature = "cuda")]
pub use kernels::CubeclBackend;

/// Prelude module for convenient imports.
///
/// # Example
///
/// ```rust
/// use trit_vsa::prelude::*;
/// ```
pub mod prelude {
    pub use crate::arithmetic::{from_balanced_ternary, to_balanced_ternary};
    pub use crate::kernels::{
        get_backend, BackendConfig, BackendPreference, CpuBackend, TernaryBackend,
    };
    pub use crate::packed::PackedTritVec;
    pub use crate::sparse::SparseVec;
    pub use crate::trit::Trit;
    pub use crate::tryte::Tryte3;
    pub use crate::vsa::{bind, bundle, cosine_similarity, hamming_distance};
    pub use crate::word::Word6;
    pub use crate::{Result, TernaryError};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_workflow() {
        // Create a ternary vector
        let mut vec = PackedTritVec::new(100);

        // Set some values
        vec.set(0, Trit::P);
        vec.set(1, Trit::N);
        vec.set(50, Trit::P);

        // Verify
        assert_eq!(vec.get(0), Trit::P);
        assert_eq!(vec.get(1), Trit::N);
        assert_eq!(vec.get(2), Trit::Z);
        assert_eq!(vec.count_nonzero(), 3);
    }

    #[test]
    fn test_vsa_workflow() {
        let mut a = PackedTritVec::new(64);
        let mut b = PackedTritVec::new(64);

        // Set up vectors
        for i in 0..32 {
            a.set(i, Trit::P);
            b.set(i, Trit::P);
        }

        // Bundle should produce a similar vector
        let bundled = vsa::bundle(&a, &b);
        let sim = vsa::cosine_similarity(&a, &bundled);
        assert!(sim > 0.9, "bundled vector should be similar to inputs");

        // Bind should produce orthogonal vector
        let bound = vsa::bind(&a, &b);
        let recovered = vsa::unbind(&bound, &b);

        // Recovered should match original
        for i in 0..64 {
            assert_eq!(recovered.get(i), a.get(i));
        }
    }

    #[test]
    fn test_sparse_workflow() {
        let mut sparse = SparseVec::new(10000);
        sparse.set(100, Trit::P);
        sparse.set(5000, Trit::N);

        assert_eq!(sparse.count_nonzero(), 2);
        assert!(sparse.sparsity() > 0.99);

        // Convert to packed for operations
        let packed = sparse.to_packed();
        assert_eq!(packed.get(100), Trit::P);
        assert_eq!(packed.get(5000), Trit::N);
    }

    #[test]
    fn test_tryte_arithmetic() {
        let a = Tryte3::from_value(7).unwrap();
        let b = Tryte3::from_value(5).unwrap();

        let (sum, carry) = a + b;
        let total = sum.value() + carry.value() as i32 * 27;
        assert_eq!(total, 12);
    }

    #[test]
    fn test_word_arithmetic() {
        let a = Word6::from_value(100).unwrap();
        let b = Word6::from_value(50).unwrap();

        let (sum, carry) = a + b;
        let total = sum.value() + carry.value() as i32 * 729;
        assert_eq!(total, 150);
    }
}
