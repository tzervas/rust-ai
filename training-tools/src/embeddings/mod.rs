//! Dimensionality reduction and embedding visualization module.
//!
//! This module provides tools for projecting high-dimensional embeddings
//! to lower dimensions for visualization purposes.
//!
//! # Supported Methods
//!
//! - **PCA**: Principal Component Analysis - fast linear projection
//! - **t-SNE**: t-Distributed Stochastic Neighbor Embedding - preserves local structure
//! - **UMAP**: Uniform Manifold Approximation and Projection - preserves global + local structure
//!
//! # Example
//!
//! ```rust,ignore
//! use training_tools::embeddings::{EmbeddingProjector, ProjectionMethod};
//!
//! // Create a PCA projector
//! let projector = EmbeddingProjector::new(ProjectionMethod::PCA);
//!
//! // Project high-dimensional embeddings to 3D
//! let embeddings: Vec<Vec<f32>> = /* your embeddings */;
//! let projected = projector.project(&embeddings);
//!
//! // Each point is now [x, y, z]
//! for point in projected {
//!     println!("3D point: {:?}", point);
//! }
//! ```

mod pca;
mod projector;
mod tsne;
mod umap;

pub use pca::{PCABuilder, PCA};
pub use projector::{EmbeddingProjector, ProjectionMethod, ProjectionStats};
pub use tsne::{TSNEBuilder, TSNE};
pub use umap::{UMAPBuilder, UMAP};

/// Error types for embedding operations
#[derive(Debug, thiserror::Error)]
pub enum EmbeddingError {
    #[error("Empty input: no embeddings provided")]
    EmptyInput,

    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("Insufficient data: need at least {min} samples, got {got}")]
    InsufficientData { min: usize, got: usize },

    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    #[error("Numerical error: {0}")]
    NumericalError(String),

    #[error("Not fitted: call fit() before transform()")]
    NotFitted,
}

pub type Result<T> = std::result::Result<T, EmbeddingError>;

/// Utility functions for embedding operations
pub(crate) mod utils {
    use super::Result;

    /// Compute Euclidean distance between two vectors
    #[inline]
    pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    /// Compute squared Euclidean distance (faster, no sqrt)
    #[inline]
    pub fn squared_euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
    }

    /// Compute pairwise distance matrix
    pub fn pairwise_distances(data: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let n = data.len();
        let mut distances = vec![vec![0.0f32; n]; n];

        for i in 0..n {
            for j in (i + 1)..n {
                let d = euclidean_distance(&data[i], &data[j]);
                distances[i][j] = d;
                distances[j][i] = d;
            }
        }

        distances
    }

    /// Compute mean of vectors
    pub fn compute_mean(data: &[Vec<f32>]) -> Result<Vec<f32>> {
        if data.is_empty() {
            return Err(super::EmbeddingError::EmptyInput);
        }

        let dim = data[0].len();
        let n = data.len() as f32;
        let mut mean = vec![0.0f32; dim];

        for row in data {
            for (i, &val) in row.iter().enumerate() {
                mean[i] += val;
            }
        }

        for val in &mut mean {
            *val /= n;
        }

        Ok(mean)
    }

    /// Center data by subtracting mean
    pub fn center_data(data: &[Vec<f32>], mean: &[f32]) -> Vec<Vec<f32>> {
        data.iter()
            .map(|row| row.iter().zip(mean.iter()).map(|(x, m)| x - m).collect())
            .collect()
    }

    /// Normalize vector to unit length
    pub fn normalize(v: &mut [f32]) {
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-10 {
            for x in v.iter_mut() {
                *x /= norm;
            }
        }
    }

    /// Compute dot product
    #[inline]
    pub fn dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_euclidean_distance() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((utils::euclidean_distance(&a, &b) - 1.0).abs() < 1e-6);

        let c = vec![1.0, 1.0, 1.0];
        let expected = 3.0f32.sqrt();
        assert!((utils::euclidean_distance(&a, &c) - expected).abs() < 1e-6);
    }

    #[test]
    fn test_compute_mean() {
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];

        let mean = utils::compute_mean(&data).unwrap();
        assert!((mean[0] - 3.0).abs() < 1e-6);
        assert!((mean[1] - 4.0).abs() < 1e-6);
    }
}
