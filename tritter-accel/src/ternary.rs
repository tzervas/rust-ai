//! Ternary operations module.
//!
//! Re-exports and wraps trit-vsa functionality for Python bindings.
//!
//! The packing/unpacking functions in `lib.rs` use `trit_vsa::PackedTritVec`
//! for efficient bitsliced ternary storage.

// Re-export key types for library users
#[allow(unused_imports)]
pub use trit_vsa::{PackedTritVec, SparseVec, Trit, Tryte3, Word6};
#[allow(unused_imports)]
pub use trit_vsa::vsa::{bind, bundle, cosine_similarity, unbind};
