//! Error types for bitnet-rs.

use thiserror::Error;

/// Result type alias for bitnet-rs operations.
pub type Result<T> = std::result::Result<T, BitNetError>;

/// Errors that can occur during BitNet operations.
#[derive(Debug, Error)]
pub enum BitNetError {
    /// Invalid configuration parameter.
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),

    /// Shape mismatch in tensor operations.
    #[error("shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        /// Expected shape.
        expected: Vec<usize>,
        /// Actual shape.
        actual: Vec<usize>,
    },

    /// Dimension mismatch.
    #[error("dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch {
        /// Expected dimension.
        expected: usize,
        /// Actual dimension.
        actual: usize,
    },

    /// Quantization error.
    #[error("quantization error: {0}")]
    Quantization(String),

    /// Candle tensor operation error.
    #[error("tensor error: {0}")]
    Tensor(#[from] candle_core::Error),

    /// Ternary operation error.
    #[error("ternary error: {0}")]
    Ternary(#[from] trit_vsa::TernaryError),

    /// I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization error.
    #[error("serialization error: {0}")]
    Serialization(String),

    /// Feature not available.
    #[error("feature not available: {0}")]
    FeatureNotAvailable(String),
}
