//! Error types for the Tritter model.

use thiserror::Error;

/// Result type for Tritter operations.
pub type TritterResult<T> = Result<T, TritterError>;

/// Errors that can occur during Tritter model operations.
#[derive(Debug, Error)]
pub enum TritterError {
    /// Tensor operation failed
    #[error("Candle error: {0}")]
    Candle(#[from] candle_core::Error),

    /// Shape mismatch
    #[error("Shape mismatch: expected {expected}, got {got}")]
    ShapeMismatch { expected: String, got: String },

    /// Invalid configuration
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// Quantization error
    #[error("Quantization error: {0}")]
    Quantization(String),

    /// Training error
    #[error("Training error: {0}")]
    Training(String),

    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization error
    #[error("Serialization error: {0}")]
    Serde(#[from] serde_json::Error),

    /// Gradient not computed
    #[error("Gradients not computed - call backward() first")]
    NoGradients,

    /// Device mismatch
    #[error("Device mismatch: expected {expected:?}, got {got:?}")]
    DeviceMismatch {
        expected: candle_core::Device,
        got: candle_core::Device,
    },

    /// Data loading error
    #[error("Data error: {0}")]
    Data(String),

    /// Tokenizer error
    #[error("Tokenizer error: {0}")]
    Tokenizer(String),
}

impl TritterError {
    /// Create a shape mismatch error
    pub fn shape_mismatch(expected: impl Into<String>, got: impl Into<String>) -> Self {
        Self::ShapeMismatch {
            expected: expected.into(),
            got: got.into(),
        }
    }

    /// Create an invalid config error
    pub fn invalid_config(msg: impl Into<String>) -> Self {
        Self::InvalidConfig(msg.into())
    }

    /// Create a quantization error
    pub fn quantization(msg: impl Into<String>) -> Self {
        Self::Quantization(msg.into())
    }

    /// Create a training error
    pub fn training(msg: impl Into<String>) -> Self {
        Self::Training(msg.into())
    }

    /// Create a data loading error
    pub fn data(msg: impl Into<String>) -> Self {
        Self::Data(msg.into())
    }

    /// Create a tokenizer error
    pub fn tokenizer(msg: impl Into<String>) -> Self {
        Self::Tokenizer(msg.into())
    }
}

/// Convert hybrid trainer errors
impl From<hybrid_predict_trainer_rs::HybridTrainingError> for TritterError {
    fn from(e: hybrid_predict_trainer_rs::HybridTrainingError) -> Self {
        Self::Training(e.to_string())
    }
}
