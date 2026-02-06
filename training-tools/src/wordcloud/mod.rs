//! 3D Token and Concept Cloud Visualization
//!
//! This module provides 3D visualization of tokens and their semantic relationships.
//!
//! # Features
//!
//! - **TokenCloud3D**: Main structure for 3D token visualization
//! - **ConceptCluster**: Group related tokens into semantic clusters
//! - **WordRelationGraph**: Show relationships between tokens in 3D space
//!
//! # Example
//!
//! ```no_run
//! use training_tools::wordcloud::{TokenCloud3D, ClusterConfig};
//!
//! let vocab = vec!["the".to_string(), "cat".to_string(), "sat".to_string()];
//! let frequencies = vec![1000, 50, 30];
//!
//! let mut cloud = TokenCloud3D::from_vocabulary(&vocab, &frequencies);
//! cloud.cluster(3);
//!
//! let render_data = cloud.to_render_data();
//! ```

mod cluster;
mod layout;
mod render;
mod token;

pub use cluster::{ClusterConfig, ConceptCluster, KMeansClusterer};
pub use layout::{ForceDirectedLayout, Layout3D, LayoutConfig, SphericalLayout};
pub use render::{RenderConfig, RenderToken, WordRelationGraph};
pub use token::{Token3D, TokenCloud3D};

use thiserror::Error;

/// Errors that can occur during wordcloud operations
#[derive(Debug, Error)]
pub enum WordCloudError {
    /// Vocabulary and frequencies have mismatched lengths
    #[error("vocabulary length ({vocab_len}) does not match frequencies length ({freq_len})")]
    LengthMismatch { vocab_len: usize, freq_len: usize },

    /// Embeddings dimension mismatch
    #[error("embedding dimension ({got}) does not match expected ({expected})")]
    EmbeddingDimensionMismatch { expected: usize, got: usize },

    /// Embeddings count mismatch
    #[error("embedding count ({got}) does not match token count ({expected})")]
    EmbeddingCountMismatch { expected: usize, got: usize },

    /// Invalid cluster count
    #[error("cluster count ({count}) must be between 1 and token count ({max})")]
    InvalidClusterCount { count: usize, max: usize },

    /// Layout computation failed
    #[error("layout computation failed: {0}")]
    LayoutFailed(String),

    /// No tokens available
    #[error("no tokens available for operation")]
    NoTokens,
}

/// Result type for wordcloud operations
pub type Result<T> = std::result::Result<T, WordCloudError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_workflow() {
        let vocab: Vec<String> = vec![
            "the".into(),
            "cat".into(),
            "sat".into(),
            "on".into(),
            "mat".into(),
        ];
        let frequencies = vec![1000, 50, 30, 500, 20];

        let mut cloud = TokenCloud3D::from_vocabulary(&vocab, &frequencies);
        assert_eq!(cloud.token_count(), 5);

        // Add mock embeddings (3D for simplicity)
        let embeddings: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.5, 0.5, 0.0],
            vec![0.6, 0.4, 0.1],
            vec![0.9, 0.1, 0.0],
            vec![0.55, 0.45, 0.05],
        ];
        cloud.add_embeddings(&embeddings).unwrap();

        // Cluster into 2 groups
        cloud.cluster(2);
        assert_eq!(cloud.cluster_count(), 2);

        // Get render data
        let render_data = cloud.to_render_data();
        assert_eq!(render_data.len(), 5);
    }

    #[test]
    fn test_length_mismatch_error() {
        let vocab: Vec<String> = vec!["a".into(), "b".into()];
        let frequencies = vec![1, 2, 3]; // Wrong length

        let result = TokenCloud3D::try_from_vocabulary(&vocab, &frequencies);
        assert!(matches!(result, Err(WordCloudError::LengthMismatch { .. })));
    }
}
