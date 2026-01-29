//! Vector Symbolic Architecture (VSA) operations for ternary vectors.
//!
//! This module provides hyperdimensional computing primitives:
//!
//! - **Bundle**: Superposition (element-wise majority voting)
//! - **Bind**: Composition (XOR-like binding)
//! - **Similarity**: Cosine similarity for retrieval
//!
//! ## References
//!
//! - Kanerva, P. "Hyperdimensional Computing: An Introduction"
//! - Gayler, R.W. "Vector Symbolic Architectures"

mod bind;
mod bundle;
mod similarity;

pub use bind::{bind, bind_many, unbind};
pub use bundle::{bundle, bundle_many, majority_trit};
pub use similarity::{
    cosine_similarity, cosine_similarity_sparse, hamming_distance, hamming_similarity,
};
