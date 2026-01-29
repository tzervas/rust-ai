//! VSA gradient compression module.
//!
//! Re-exports and wraps vsa-optim-rs functionality for Python bindings.
//!
//! The `compress_gradients_vsa` and `decompress_gradients_vsa` functions in
//! `lib.rs` use a simplified random projection approach for API compatibility.
//!
//! For full VSA operations with bind/bundle/unbind, use `VSAGradientCompressor`
//! from vsa-optim-rs directly.

// Re-export key types for library users
#[allow(unused_imports)]
pub use vsa_optim_rs::vsa::{CompressionStats, GradientMetadata, VSAGradientCompressor};
#[allow(unused_imports)]
pub use vsa_optim_rs::VSAConfig;
