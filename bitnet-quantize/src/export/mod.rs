//! Export functionality for BitNet models.
//!
//! This module provides export to various formats:
//! - GGUF (requires `gguf-export` feature)

#[cfg(feature = "gguf-export")]
mod gguf;

#[cfg(feature = "gguf-export")]
pub use gguf::export_gguf;
