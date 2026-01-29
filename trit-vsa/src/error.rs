//! Error types for ternary-rs.

use thiserror::Error;

/// Result type alias for ternary-rs operations.
pub type Result<T> = std::result::Result<T, TernaryError>;

/// Errors that can occur during ternary operations.
#[derive(Debug, Error)]
pub enum TernaryError {
    /// Invalid value for ternary conversion.
    #[error("invalid ternary value: {0} (expected -1, 0, or +1)")]
    InvalidValue(i32),

    /// Dimension mismatch in vector operations.
    #[error("dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch {
        /// Expected dimension.
        expected: usize,
        /// Actual dimension.
        actual: usize,
    },

    /// Index out of bounds.
    #[error("index {index} out of bounds for size {size}")]
    IndexOutOfBounds {
        /// The index that was accessed.
        index: usize,
        /// The size of the container.
        size: usize,
    },

    /// Overflow in ternary arithmetic.
    #[error("arithmetic overflow: value {0} exceeds ternary range")]
    Overflow(i64),

    /// Invalid tryte value (must be in range -13 to +13).
    #[error("invalid tryte value: {0} (expected -13 to +13)")]
    InvalidTryteValue(i32),

    /// Invalid word value (must be in range -364 to +364).
    #[error("invalid word value: {0} (expected -364 to +364)")]
    InvalidWordValue(i32),

    /// Empty vector operation.
    #[error("operation not supported on empty vector")]
    EmptyVector,

    /// GPU computation error (only available with cuda feature).
    #[cfg(feature = "cuda")]
    #[error("GPU computation error: {0}")]
    GpuError(#[from] crate::gpu::traits::GpuError),
}
