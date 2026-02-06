//! Unified embedding projector that supports multiple methods.

use super::{pca::PCA, tsne::TSNE, umap::UMAP, EmbeddingError, Result};

/// Projection method for dimensionality reduction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProjectionMethod {
    /// Principal Component Analysis - fast, linear
    PCA,
    /// t-SNE - preserves local structure
    TSNE,
    /// UMAP - preserves local and global structure
    UMAP,
}

impl Default for ProjectionMethod {
    fn default() -> Self {
        ProjectionMethod::PCA
    }
}

/// Statistics about the projection.
#[derive(Debug, Clone)]
pub struct ProjectionStats {
    /// Number of input samples
    pub n_samples: usize,
    /// Input dimensionality
    pub input_dim: usize,
    /// Output dimensionality
    pub output_dim: usize,
    /// Method used
    pub method: ProjectionMethod,
    /// For PCA: explained variance ratio
    pub explained_variance: Option<Vec<f32>>,
    /// For t-SNE: perplexity used
    pub perplexity: Option<f32>,
    /// For UMAP: n_neighbors used
    pub n_neighbors: Option<usize>,
}

/// Unified projector for dimensionality reduction and embedding visualization.
///
/// Supports PCA, t-SNE, and UMAP projection methods.
///
/// # Example
///
/// ```rust,ignore
/// use training_tools::embeddings::{EmbeddingProjector, ProjectionMethod};
///
/// let projector = EmbeddingProjector::new(ProjectionMethod::PCA);
///
/// let embeddings: Vec<Vec<f32>> = /* high-dimensional embeddings */;
/// let projected = projector.project(&embeddings)?;
///
/// for [x, y, z] in projected {
///     println!("Point: ({}, {}, {})", x, y, z);
/// }
/// ```
#[derive(Debug, Clone)]
pub struct EmbeddingProjector {
    method: ProjectionMethod,
    /// Perplexity for t-SNE (default: 30)
    pub perplexity: f32,
    /// Number of neighbors for UMAP (default: 15)
    pub n_neighbors: usize,
    /// Number of iterations for t-SNE/UMAP
    pub n_iter: usize,
    /// Minimum distance for UMAP
    pub min_dist: f32,
    /// Learning rate
    pub learning_rate: f32,
    /// Random seed
    pub seed: u64,

    // Internal state for incremental projection
    pca: Option<PCA>,
    last_original_data: Option<Vec<Vec<f32>>>,
    umap: Option<UMAP>,
}

impl Default for EmbeddingProjector {
    fn default() -> Self {
        Self::new(ProjectionMethod::PCA)
    }
}

impl EmbeddingProjector {
    /// Create a new projector with the specified method.
    pub fn new(method: ProjectionMethod) -> Self {
        Self {
            method,
            perplexity: 30.0,
            n_neighbors: 15,
            n_iter: 500,
            min_dist: 0.1,
            learning_rate: 200.0,
            seed: 42,
            pca: None,
            last_original_data: None,
            umap: None,
        }
    }

    /// Create a PCA projector.
    pub fn pca() -> Self {
        Self::new(ProjectionMethod::PCA)
    }

    /// Create a t-SNE projector with specified perplexity.
    pub fn tsne(perplexity: f32) -> Self {
        let mut p = Self::new(ProjectionMethod::TSNE);
        p.perplexity = perplexity;
        p
    }

    /// Create a UMAP projector with specified n_neighbors.
    pub fn umap(n_neighbors: usize) -> Self {
        let mut p = Self::new(ProjectionMethod::UMAP);
        p.n_neighbors = n_neighbors;
        p
    }

    /// Set the projection method.
    pub fn with_method(mut self, method: ProjectionMethod) -> Self {
        self.method = method;
        self
    }

    /// Set perplexity for t-SNE.
    pub fn with_perplexity(mut self, perplexity: f32) -> Self {
        self.perplexity = perplexity;
        self
    }

    /// Set n_neighbors for UMAP.
    pub fn with_n_neighbors(mut self, n: usize) -> Self {
        self.n_neighbors = n;
        self
    }

    /// Set number of iterations.
    pub fn with_n_iter(mut self, n: usize) -> Self {
        self.n_iter = n;
        self
    }

    /// Set minimum distance for UMAP.
    pub fn with_min_dist(mut self, d: f32) -> Self {
        self.min_dist = d;
        self
    }

    /// Set learning rate.
    pub fn with_learning_rate(mut self, lr: f32) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set random seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Project embeddings to 3D.
    ///
    /// # Arguments
    /// * `embeddings` - High-dimensional embedding vectors
    ///
    /// # Returns
    /// Vector of 3D points [x, y, z] for each input embedding
    pub fn project(&mut self, embeddings: &[Vec<f32>]) -> Result<Vec<[f32; 3]>> {
        if embeddings.is_empty() {
            return Err(EmbeddingError::EmptyInput);
        }

        match self.method {
            ProjectionMethod::PCA => self.project_pca(embeddings),
            ProjectionMethod::TSNE => self.project_tsne(embeddings),
            ProjectionMethod::UMAP => self.project_umap(embeddings),
        }
    }

    /// Project a single new point incrementally.
    ///
    /// For PCA, uses the fitted components.
    /// For t-SNE, approximates position based on nearest neighbors.
    /// For UMAP, uses the fitted model.
    ///
    /// Must call `project()` first to fit the model.
    pub fn project_incremental(&mut self, new_point: &[f32]) -> Result<[f32; 3]> {
        match self.method {
            ProjectionMethod::PCA => self.project_pca_incremental(new_point),
            ProjectionMethod::TSNE => self.project_tsne_incremental(new_point),
            ProjectionMethod::UMAP => self.project_umap_incremental(new_point),
        }
    }

    /// Get statistics about the last projection.
    pub fn stats(&self) -> Option<ProjectionStats> {
        let original_data = self.last_original_data.as_ref()?;

        if original_data.is_empty() {
            return None;
        }

        Some(ProjectionStats {
            n_samples: original_data.len(),
            input_dim: original_data[0].len(),
            output_dim: 3,
            method: self.method,
            explained_variance: self.pca.as_ref().and_then(|p| p.explained_variance_ratio()),
            perplexity: if self.method == ProjectionMethod::TSNE {
                Some(self.perplexity)
            } else {
                None
            },
            n_neighbors: if self.method == ProjectionMethod::UMAP {
                Some(self.n_neighbors)
            } else {
                None
            },
        })
    }

    /// Get the current projection method.
    pub fn method(&self) -> ProjectionMethod {
        self.method
    }

    // ---- Internal methods ----

    fn project_pca(&mut self, embeddings: &[Vec<f32>]) -> Result<Vec<[f32; 3]>> {
        let mut pca = PCA::new(3);
        pca.fit(embeddings)?;
        let result = pca.transform_3d(embeddings)?;

        self.pca = Some(pca);
        self.last_original_data = Some(embeddings.to_vec());

        Ok(result)
    }

    fn project_tsne(&mut self, embeddings: &[Vec<f32>]) -> Result<Vec<[f32; 3]>> {
        let mut tsne = TSNE::builder()
            .n_components(3)
            .perplexity(self.perplexity)
            .n_iter(self.n_iter)
            .learning_rate(self.learning_rate)
            .seed(self.seed)
            .build();

        let result = tsne.transform_3d(embeddings)?;

        self.last_original_data = Some(embeddings.to_vec());

        Ok(result)
    }

    fn project_umap(&mut self, embeddings: &[Vec<f32>]) -> Result<Vec<[f32; 3]>> {
        let mut umap = UMAP::builder()
            .n_components(3)
            .n_neighbors(self.n_neighbors)
            .min_dist(self.min_dist)
            .n_epochs(self.n_iter)
            .learning_rate(self.learning_rate / 200.0) // UMAP uses smaller LR
            .seed(self.seed)
            .build();

        let result = umap.transform_3d(embeddings)?;

        self.umap = Some(umap);
        self.last_original_data = Some(embeddings.to_vec());

        Ok(result)
    }

    fn project_pca_incremental(&self, new_point: &[f32]) -> Result<[f32; 3]> {
        let pca = self.pca.as_ref().ok_or(EmbeddingError::NotFitted)?;

        let transformed = pca.transform(&[new_point.to_vec()])?;
        if transformed.is_empty() || transformed[0].len() < 3 {
            return Err(EmbeddingError::NumericalError(
                "Insufficient output dimensions".to_string(),
            ));
        }

        Ok([transformed[0][0], transformed[0][1], transformed[0][2]])
    }

    fn project_tsne_incremental(&self, new_point: &[f32]) -> Result<[f32; 3]> {
        // t-SNE doesn't have a natural way to add points incrementally
        // We approximate by using weighted average of k-nearest neighbors

        let original_data = self
            .last_original_data
            .as_ref()
            .ok_or(EmbeddingError::NotFitted)?;

        // Simple approximation using PCA fallback
        // In production, you'd want to implement proper out-of-sample extension

        // For now, we'll use a weighted average approach based on distances
        // This is a simplified approximation

        // Find k nearest neighbors
        let _k = 10.min(original_data.len());
        let mut distances: Vec<(usize, f32)> = original_data
            .iter()
            .enumerate()
            .map(|(i, p)| (i, super::utils::euclidean_distance(new_point, p)))
            .collect();

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // We'd need the t-SNE embedding to compute the weighted average
        // For now, return an error indicating this limitation
        Err(EmbeddingError::InvalidParameter(
            "Incremental t-SNE projection not fully implemented. \
             Consider using PCA or UMAP for incremental projection."
                .to_string(),
        ))
    }

    fn project_umap_incremental(&self, new_point: &[f32]) -> Result<[f32; 3]> {
        let umap = self.umap.as_ref().ok_or(EmbeddingError::NotFitted)?;
        let original_data = self
            .last_original_data
            .as_ref()
            .ok_or(EmbeddingError::NotFitted)?;

        let projected = umap.transform_point(new_point, original_data)?;

        if projected.len() < 3 {
            return Err(EmbeddingError::NumericalError(
                "Insufficient output dimensions".to_string(),
            ));
        }

        Ok([projected[0], projected[1], projected[2]])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_test_data(n: usize, dim: usize) -> Vec<Vec<f32>> {
        let mut data = Vec::with_capacity(n);
        let mut seed = 42u64;

        for _ in 0..n {
            let mut point = Vec::with_capacity(dim);
            for _ in 0..dim {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                let val = ((seed >> 33) as f32) / (u32::MAX as f32);
                point.push(val);
            }
            data.push(point);
        }

        data
    }

    #[test]
    fn test_projector_pca() {
        let data = generate_test_data(50, 20);

        let mut projector = EmbeddingProjector::pca();
        let result = projector.project(&data).unwrap();

        assert_eq!(result.len(), 50);
        for point in &result {
            assert!(point[0].is_finite());
            assert!(point[1].is_finite());
            assert!(point[2].is_finite());
        }

        // Test incremental
        let new_point = generate_test_data(1, 20).remove(0);
        let incremental = projector.project_incremental(&new_point).unwrap();
        assert!(incremental[0].is_finite());
    }

    #[test]
    fn test_projector_tsne() {
        let data = generate_test_data(100, 15);

        let mut projector = EmbeddingProjector::tsne(15.0).with_n_iter(100);

        let result = projector.project(&data).unwrap();

        assert_eq!(result.len(), 100);
        for point in &result {
            assert!(point[0].is_finite());
            assert!(point[1].is_finite());
            assert!(point[2].is_finite());
        }
    }

    #[test]
    fn test_projector_umap() {
        let data = generate_test_data(80, 12);

        let mut projector = EmbeddingProjector::umap(10).with_n_iter(50);

        let result = projector.project(&data).unwrap();

        assert_eq!(result.len(), 80);
        for point in &result {
            assert!(point[0].is_finite());
            assert!(point[1].is_finite());
            assert!(point[2].is_finite());
        }

        // Test incremental UMAP
        let new_point = generate_test_data(1, 12).remove(0);
        let incremental = projector.project_incremental(&new_point).unwrap();
        assert!(incremental[0].is_finite());
    }

    #[test]
    fn test_projector_stats() {
        let data = generate_test_data(30, 10);

        let mut projector = EmbeddingProjector::pca();
        projector.project(&data).unwrap();

        let stats = projector.stats().unwrap();
        assert_eq!(stats.n_samples, 30);
        assert_eq!(stats.input_dim, 10);
        assert_eq!(stats.output_dim, 3);
        assert_eq!(stats.method, ProjectionMethod::PCA);
        assert!(stats.explained_variance.is_some());
    }

    #[test]
    fn test_projector_builder_pattern() {
        let projector = EmbeddingProjector::new(ProjectionMethod::TSNE)
            .with_perplexity(25.0)
            .with_n_iter(300)
            .with_learning_rate(150.0)
            .with_seed(123);

        assert_eq!(projector.method(), ProjectionMethod::TSNE);
        assert!((projector.perplexity - 25.0).abs() < 1e-6);
        assert_eq!(projector.n_iter, 300);
    }
}
