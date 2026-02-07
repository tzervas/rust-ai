//! Principal Component Analysis (PCA) implementation.
//!
//! Pure Rust implementation using power iteration for eigendecomposition.

use super::{utils, EmbeddingError, Result};

/// Principal Component Analysis for dimensionality reduction.
///
/// Uses power iteration to compute principal components without
/// external linear algebra dependencies.
#[derive(Debug, Clone)]
pub struct PCA {
    /// Number of components to keep
    n_components: usize,
    /// Computed principal components (eigenvectors)
    components: Option<Vec<Vec<f32>>>,
    /// Mean of training data
    mean: Option<Vec<f32>>,
    /// Explained variance for each component
    explained_variance: Option<Vec<f32>>,
    /// Total variance in original data
    total_variance: Option<f32>,
    /// Maximum iterations for power iteration
    max_iter: usize,
    /// Convergence tolerance
    tolerance: f32,
}

impl Default for PCA {
    fn default() -> Self {
        Self::new(3)
    }
}

impl PCA {
    /// Create a new PCA instance with specified number of components.
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            components: None,
            mean: None,
            explained_variance: None,
            total_variance: None,
            max_iter: 100,
            tolerance: 1e-6,
        }
    }

    /// Create a builder for more configuration options.
    pub fn builder() -> PCABuilder {
        PCABuilder::default()
    }

    /// Fit the PCA model to data.
    ///
    /// # Arguments
    /// * `data` - Training data as Vec of Vec (n_samples x n_features)
    pub fn fit(&mut self, data: &[Vec<f32>]) -> Result<&mut Self> {
        if data.is_empty() {
            return Err(EmbeddingError::EmptyInput);
        }

        let n_samples = data.len();
        let n_features = data[0].len();

        // Validate dimensions
        for (i, row) in data.iter().enumerate() {
            if row.len() != n_features {
                return Err(EmbeddingError::DimensionMismatch {
                    expected: n_features,
                    got: row.len(),
                });
            }
            // Check for NaN/Inf
            for &val in row {
                if !val.is_finite() {
                    return Err(EmbeddingError::NumericalError(format!(
                        "Non-finite value at sample {}",
                        i
                    )));
                }
            }
        }

        // Compute mean
        let mean = utils::compute_mean(data)?;
        self.mean = Some(mean.clone());

        // Center data
        let centered = utils::center_data(data, &mean);

        // Compute covariance matrix: (1/n) * X^T * X
        let cov = self.compute_covariance(&centered, n_samples, n_features);

        // Compute total variance (trace of covariance matrix)
        let total_var: f32 = (0..n_features).map(|i| cov[i][i]).sum();
        self.total_variance = Some(total_var);

        // Extract principal components using power iteration
        let n_components = self.n_components.min(n_features).min(n_samples);
        let (components, eigenvalues) = self.power_iteration(&cov, n_components)?;

        self.components = Some(components);
        self.explained_variance = Some(eigenvalues);

        Ok(self)
    }

    /// Transform data using fitted PCA model.
    ///
    /// Projects data onto principal components.
    pub fn transform(&self, data: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        let components = self.components.as_ref().ok_or(EmbeddingError::NotFitted)?;
        let mean = self.mean.as_ref().ok_or(EmbeddingError::NotFitted)?;

        if data.is_empty() {
            return Ok(Vec::new());
        }

        let n_features = mean.len();

        // Validate and transform
        let mut result = Vec::with_capacity(data.len());

        for row in data {
            if row.len() != n_features {
                return Err(EmbeddingError::DimensionMismatch {
                    expected: n_features,
                    got: row.len(),
                });
            }

            // Center the data point
            let centered: Vec<f32> = row.iter().zip(mean.iter()).map(|(x, m)| x - m).collect();

            // Project onto each component
            let projected: Vec<f32> = components
                .iter()
                .map(|pc| utils::dot(&centered, pc))
                .collect();

            result.push(projected);
        }

        Ok(result)
    }

    /// Fit and transform in one step.
    pub fn fit_transform(&mut self, data: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        self.fit(data)?;
        self.transform(data)
    }

    /// Transform to exactly 3 dimensions for visualization.
    pub fn transform_3d(&self, data: &[Vec<f32>]) -> Result<Vec<[f32; 3]>> {
        if self.n_components < 3 {
            return Err(EmbeddingError::InvalidParameter(
                "Need at least 3 components for 3D projection".to_string(),
            ));
        }

        let transformed = self.transform(data)?;
        let result: Vec<[f32; 3]> = transformed
            .into_iter()
            .map(|v| [v[0], v[1], v[2]])
            .collect();

        Ok(result)
    }

    /// Get the explained variance ratio for each component.
    pub fn explained_variance_ratio(&self) -> Option<Vec<f32>> {
        let explained = self.explained_variance.as_ref()?;
        let total = self.total_variance?;

        if total < 1e-10 {
            return None;
        }

        Some(explained.iter().map(|&v| v / total).collect())
    }

    /// Get the cumulative explained variance ratio.
    pub fn cumulative_variance_ratio(&self) -> Option<Vec<f32>> {
        let ratios = self.explained_variance_ratio()?;
        let mut cumulative = Vec::with_capacity(ratios.len());
        let mut sum = 0.0;

        for ratio in ratios {
            sum += ratio;
            cumulative.push(sum);
        }

        Some(cumulative)
    }

    /// Compute covariance matrix from centered data.
    fn compute_covariance(
        &self,
        centered: &[Vec<f32>],
        n_samples: usize,
        n_features: usize,
    ) -> Vec<Vec<f32>> {
        let mut cov = vec![vec![0.0f32; n_features]; n_features];
        let scale = 1.0 / (n_samples as f32 - 1.0).max(1.0);

        for i in 0..n_features {
            for j in i..n_features {
                let mut sum = 0.0f32;
                for row in centered {
                    sum += row[i] * row[j];
                }
                let val = sum * scale;
                cov[i][j] = val;
                cov[j][i] = val;
            }
        }

        cov
    }

    /// Power iteration method to extract principal components.
    ///
    /// Computes eigenvalues and eigenvectors using deflation.
    fn power_iteration(
        &self,
        matrix: &[Vec<f32>],
        n_components: usize,
    ) -> Result<(Vec<Vec<f32>>, Vec<f32>)> {
        let n = matrix.len();
        let mut components = Vec::with_capacity(n_components);
        let mut eigenvalues = Vec::with_capacity(n_components);

        // Work with a mutable copy for deflation
        let mut mat: Vec<Vec<f32>> = matrix.to_vec();

        for _ in 0..n_components {
            // Initialize random vector
            let mut v: Vec<f32> = (0..n)
                .map(|i| ((i * 7 + 13) % 100) as f32 / 100.0)
                .collect();
            utils::normalize(&mut v);

            let mut eigenvalue = 0.0f32;

            // Power iteration
            for _ in 0..self.max_iter {
                // Multiply: v_new = A * v
                let mut v_new = vec![0.0f32; n];
                for i in 0..n {
                    for j in 0..n {
                        v_new[i] += mat[i][j] * v[j];
                    }
                }

                // Compute eigenvalue (Rayleigh quotient)
                let new_eigenvalue = utils::dot(&v_new, &v);

                // Normalize
                utils::normalize(&mut v_new);

                // Check convergence
                let diff: f32 = v.iter().zip(v_new.iter()).map(|(a, b)| (a - b).abs()).sum();

                v = v_new;
                eigenvalue = new_eigenvalue;

                if diff < self.tolerance {
                    break;
                }
            }

            components.push(v.clone());
            eigenvalues.push(eigenvalue.max(0.0)); // Ensure non-negative

            // Deflate matrix: A = A - λ * v * v^T
            for i in 0..n {
                for j in 0..n {
                    mat[i][j] -= eigenvalue * v[i] * v[j];
                }
            }
        }

        Ok((components, eigenvalues))
    }
}

/// Builder for configuring PCA.
#[derive(Debug, Clone)]
pub struct PCABuilder {
    n_components: usize,
    max_iter: usize,
    tolerance: f32,
}

impl Default for PCABuilder {
    fn default() -> Self {
        Self {
            n_components: 3,
            max_iter: 100,
            tolerance: 1e-6,
        }
    }
}

impl PCABuilder {
    /// Set number of components.
    pub fn n_components(mut self, n: usize) -> Self {
        self.n_components = n;
        self
    }

    /// Set maximum iterations for power iteration.
    pub fn max_iter(mut self, n: usize) -> Self {
        self.max_iter = n;
        self
    }

    /// Set convergence tolerance.
    pub fn tolerance(mut self, tol: f32) -> Self {
        self.tolerance = tol;
        self
    }

    /// Build the PCA instance.
    pub fn build(self) -> PCA {
        PCA {
            n_components: self.n_components,
            components: None,
            mean: None,
            explained_variance: None,
            total_variance: None,
            max_iter: self.max_iter,
            tolerance: self.tolerance,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pca_basic() {
        // Simple 2D data that clearly has one dominant direction
        let data = vec![
            vec![1.0, 2.0],
            vec![2.0, 4.0],
            vec![3.0, 6.0],
            vec![4.0, 8.0],
            vec![5.0, 10.0],
        ];

        let mut pca = PCA::new(2);
        let result = pca.fit_transform(&data).unwrap();

        // Should have 5 samples with 2 components
        assert_eq!(result.len(), 5);
        assert_eq!(result[0].len(), 2);

        // First component should capture most variance
        let ratios = pca.explained_variance_ratio().unwrap();
        assert!(ratios[0] > 0.99); // Nearly all variance in first component
    }

    #[test]
    fn test_pca_3d_projection() {
        let data: Vec<Vec<f32>> = (0..20)
            .map(|i| vec![i as f32, (i * 2) as f32, (i * 3) as f32, (i * 4) as f32])
            .collect();

        let mut pca = PCA::new(3);
        pca.fit(&data).unwrap();

        let projected = pca.transform_3d(&data).unwrap();
        assert_eq!(projected.len(), 20);
        assert_eq!(projected[0].len(), 3);
    }

    #[test]
    fn test_pca_empty_input() {
        let mut pca = PCA::new(3);
        let result = pca.fit(&[]);
        assert!(matches!(result, Err(EmbeddingError::EmptyInput)));
    }

    #[test]
    fn test_pca_builder() {
        let pca = PCA::builder()
            .n_components(5)
            .max_iter(200)
            .tolerance(1e-8)
            .build();

        assert_eq!(pca.n_components, 5);
        assert_eq!(pca.max_iter, 200);
        assert!((pca.tolerance - 1e-8).abs() < 1e-10);
    }
}
