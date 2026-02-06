//! Real-time Training Metrics Streaming with Rerun Integration
//!
//! This module provides real-time visualization of ML training metrics using
//! [Rerun](https://rerun.io), an open-source visualization tool for multimodal data.
//!
//! # Features
//!
//! - **Training Metrics**: Stream loss, learning rate, gradient norms, and training phase
//! - **Embedding Visualization**: 3D point clouds of embedding spaces with labels
//! - **Attention Patterns**: Heatmap visualization of attention weights
//! - **Network Architecture**: Visual representation of model structure
//! - **Loss Landscape**: 3D surface plots with optimization trajectory
//!
//! # Example
//!
//! ```rust,ignore
//! use training_tools::rerun_viz::{RerunLogger, LayerInfo, LayerType};
//!
//! // Create a logger that connects to a running Rerun viewer
//! let logger = RerunLogger::new("my_training_run")?;
//!
//! // Log training metrics each step
//! for step in 0..10000 {
//!     let loss = train_step();
//!     logger.log_step(step, loss, current_lr, grad_norm);
//! }
//!
//! // Log embedding projections periodically
//! if step % 100 == 0 {
//!     let embeddings = model.get_embeddings();
//!     logger.log_embeddings(&embeddings, &labels);
//! }
//! ```
//!
//! # Rerun Viewer
//!
//! To view the visualization, install and run the Rerun viewer:
//! ```bash
//! pip install rerun-sdk  # or cargo install rerun-cli
//! rerun
//! ```
//!
//! The logger will automatically connect to the viewer on `localhost:9876`.

#[cfg(feature = "rerun")]
mod attention;
#[cfg(feature = "rerun")]
mod embeddings;
#[cfg(feature = "rerun")]
mod landscape;
#[cfg(feature = "rerun")]
mod logger;
#[cfg(feature = "rerun")]
mod metrics;

#[cfg(feature = "rerun")]
pub use attention::AttentionLogger;
#[cfg(feature = "rerun")]
pub use embeddings::EmbeddingLogger;
#[cfg(feature = "rerun")]
pub use landscape::LandscapeLogger;
#[cfg(feature = "rerun")]
pub use logger::{LayerInfo, LayerType, RerunLogger};
#[cfg(feature = "rerun")]
pub use metrics::{MetricsLogger, TrainingPhaseLog};

use thiserror::Error;

/// Errors that can occur during Rerun visualization operations.
#[derive(Debug, Error)]
pub enum RerunError {
    #[error("Failed to create recording stream: {0}")]
    StreamCreationError(String),

    #[error("Failed to log data: {0}")]
    LoggingError(String),

    #[error("Invalid data shape: expected {expected}, got {got}")]
    ShapeMismatch { expected: String, got: String },

    #[error("Empty input: {0}")]
    EmptyInput(String),

    #[error("Rerun feature not enabled. Add `rerun` feature to Cargo.toml")]
    FeatureNotEnabled,

    #[error("Connection failed: {0}")]
    ConnectionError(String),
}

/// Result type for Rerun visualization operations.
pub type RerunResult<T> = Result<T, RerunError>;

/// Stub implementation when rerun feature is disabled
#[cfg(not(feature = "rerun"))]
pub struct RerunLogger;

#[cfg(not(feature = "rerun"))]
impl RerunLogger {
    pub fn new(_app_id: &str) -> RerunResult<Self> {
        Err(RerunError::FeatureNotEnabled)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = RerunError::ShapeMismatch {
            expected: "[100, 768]".to_string(),
            got: "[100, 512]".to_string(),
        };
        assert!(err.to_string().contains("expected [100, 768]"));
    }

    #[test]
    #[cfg(not(feature = "rerun"))]
    fn test_stub_returns_feature_error() {
        let result = RerunLogger::new("test");
        assert!(matches!(result, Err(RerunError::FeatureNotEnabled)));
    }
}
