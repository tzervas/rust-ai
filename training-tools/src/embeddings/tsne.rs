//! t-Distributed Stochastic Neighbor Embedding (t-SNE) implementation.
//!
//! Simplified implementation for visualization of high-dimensional data.

use super::{utils, EmbeddingError, Result};

/// t-SNE for non-linear dimensionality reduction.
///
/// Preserves local neighborhood structure while mapping to low dimensions.
#[derive(Debug, Clone)]
pub struct TSNE {
    /// Target dimensionality (usually 2 or 3)
    n_components: usize,
    /// Perplexity parameter (effective number of neighbors)
    perplexity: f32,
    /// Learning rate for gradient descent
    learning_rate: f32,
    /// Number of iterations
    n_iter: usize,
    /// Early exaggeration factor
    early_exaggeration: f32,
    /// Number of early exaggeration iterations
    early_exaggeration_iter: usize,
    /// Momentum for optimization
    momentum: f32,
    /// Final momentum (used after early exaggeration)
    final_momentum: f32,
    /// Minimum gain for adaptive learning
    min_gain: f32,
    /// Random seed for initialization
    seed: u64,
    /// Current embedding (set after fit)
    embedding: Option<Vec<Vec<f32>>>,
}

impl Default for TSNE {
    fn default() -> Self {
        Self::new(3)
    }
}

impl TSNE {
    /// Create a new t-SNE instance.
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            perplexity: 30.0,
            learning_rate: 200.0,
            n_iter: 1000,
            early_exaggeration: 12.0,
            early_exaggeration_iter: 250,
            momentum: 0.5,
            final_momentum: 0.8,
            min_gain: 0.01,
            seed: 42,
            embedding: None,
        }
    }

    /// Create a builder for configuration.
    pub fn builder() -> TSNEBuilder {
        TSNEBuilder::default()
    }

    /// Get perplexity value.
    pub fn perplexity(&self) -> f32 {
        self.perplexity
    }

    /// Fit and transform data.
    pub fn fit_transform(&mut self, data: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        if data.is_empty() {
            return Err(EmbeddingError::EmptyInput);
        }

        let n = data.len();
        let min_samples = (3.0 * self.perplexity).ceil() as usize + 1;

        if n < min_samples {
            return Err(EmbeddingError::InsufficientData {
                min: min_samples,
                got: n,
            });
        }

        // Validate input dimensions
        let dim = data[0].len();
        for row in data.iter() {
            if row.len() != dim {
                return Err(EmbeddingError::DimensionMismatch {
                    expected: dim,
                    got: row.len(),
                });
            }
        }

        // Compute pairwise distances
        let distances = utils::pairwise_distances(data);

        // Compute joint probabilities P
        let p_matrix = self.compute_joint_probabilities(&distances)?;

        // Initialize low-dimensional embedding randomly
        let mut embedding = self.initialize_embedding(n);

        // Optimization arrays
        let mut gains = vec![vec![1.0f32; self.n_components]; n];
        let mut velocities = vec![vec![0.0f32; self.n_components]; n];

        // Gradient descent optimization
        for iter in 0..self.n_iter {
            // Compute Q matrix (student t-distribution in low-dim)
            let (q_matrix, sum_q) = self.compute_q_distribution(&embedding);

            // Compute gradients
            let gradients = self.compute_gradients(&p_matrix, &q_matrix, sum_q, &embedding, iter);

            // Update embedding with momentum
            let current_momentum = if iter < self.early_exaggeration_iter {
                self.momentum
            } else {
                self.final_momentum
            };

            for i in 0..n {
                for j in 0..self.n_components {
                    // Adaptive learning rate (gains)
                    let sign_match = (gradients[i][j] > 0.0) == (velocities[i][j] > 0.0);
                    if sign_match {
                        gains[i][j] = (gains[i][j] * 0.8).max(self.min_gain);
                    } else {
                        gains[i][j] += 0.2;
                    }

                    // Update velocity with momentum
                    velocities[i][j] = current_momentum * velocities[i][j]
                        - self.learning_rate * gains[i][j] * gradients[i][j];

                    // Update embedding
                    embedding[i][j] += velocities[i][j];
                }
            }

            // Re-center embedding
            self.center_embedding(&mut embedding);
        }

        self.embedding = Some(embedding.clone());
        Ok(embedding)
    }

    /// Transform to exactly 3 dimensions.
    pub fn transform_3d(&mut self, data: &[Vec<f32>]) -> Result<Vec<[f32; 3]>> {
        if self.n_components != 3 {
            return Err(EmbeddingError::InvalidParameter(
                "n_components must be 3 for 3D transform".to_string(),
            ));
        }

        let embedding = self.fit_transform(data)?;
        let result: Vec<[f32; 3]> = embedding.into_iter().map(|v| [v[0], v[1], v[2]]).collect();

        Ok(result)
    }

    /// Compute joint probability matrix P using binary search for sigma.
    fn compute_joint_probabilities(&self, distances: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        let n = distances.len();
        let target_entropy = self.perplexity.ln();

        let mut p = vec![vec![0.0f32; n]; n];

        // Compute conditional probabilities for each point
        for i in 0..n {
            // Binary search for sigma that achieves target perplexity
            let mut sigma_min = 1e-10f32;
            let mut sigma_max = 1e10f32;
            let mut sigma = 1.0f32;

            for _ in 0..50 {
                // Maximum binary search iterations
                // Compute conditional probabilities with current sigma
                let (probs, entropy) = self.compute_conditional_probs(&distances[i], i, sigma);

                if (entropy - target_entropy).abs() < 1e-5 {
                    // Found good sigma
                    for (j, &prob) in probs.iter().enumerate() {
                        p[i][j] = prob;
                    }
                    break;
                }

                if entropy > target_entropy {
                    sigma_max = sigma;
                } else {
                    sigma_min = sigma;
                }

                sigma = (sigma_min + sigma_max) / 2.0;
            }

            // Store probabilities
            let (probs, _) = self.compute_conditional_probs(&distances[i], i, sigma);
            for (j, &prob) in probs.iter().enumerate() {
                p[i][j] = prob;
            }
        }

        // Symmetrize: P_ij = (P_i|j + P_j|i) / 2n
        let mut p_symmetric = vec![vec![0.0f32; n]; n];
        for i in 0..n {
            for j in 0..n {
                p_symmetric[i][j] = (p[i][j] + p[j][i]) / (2.0 * n as f32);
            }
        }

        // Apply early exaggeration
        for row in &mut p_symmetric {
            for val in row {
                *val *= self.early_exaggeration;
            }
        }

        Ok(p_symmetric)
    }

    /// Compute conditional probabilities and entropy for a single point.
    fn compute_conditional_probs(
        &self,
        distances: &[f32],
        i: usize,
        sigma: f32,
    ) -> (Vec<f32>, f32) {
        let n = distances.len();
        let beta = 1.0 / (2.0 * sigma * sigma);

        let mut probs = vec![0.0f32; n];
        let mut sum_p = 0.0f32;

        for j in 0..n {
            if i == j {
                probs[j] = 0.0;
            } else {
                probs[j] = (-beta * distances[j] * distances[j]).exp();
                sum_p += probs[j];
            }
        }

        // Normalize and compute entropy
        let mut entropy = 0.0f32;
        if sum_p > 1e-10 {
            for j in 0..n {
                probs[j] /= sum_p;
                if probs[j] > 1e-10 {
                    entropy -= probs[j] * probs[j].ln();
                }
            }
        }

        (probs, entropy)
    }

    /// Compute Q distribution (Student's t with df=1 in low-dim space).
    fn compute_q_distribution(&self, embedding: &[Vec<f32>]) -> (Vec<Vec<f32>>, f32) {
        let n = embedding.len();
        let mut q = vec![vec![0.0f32; n]; n];
        let mut sum_q = 0.0f32;

        for i in 0..n {
            for j in (i + 1)..n {
                let dist_sq = utils::squared_euclidean_distance(&embedding[i], &embedding[j]);
                let val = 1.0 / (1.0 + dist_sq);
                q[i][j] = val;
                q[j][i] = val;
                sum_q += 2.0 * val;
            }
        }

        (q, sum_q)
    }

    /// Compute gradients for optimization.
    fn compute_gradients(
        &self,
        p: &[Vec<f32>],
        q: &[Vec<f32>],
        sum_q: f32,
        embedding: &[Vec<f32>],
        iter: usize,
    ) -> Vec<Vec<f32>> {
        let n = embedding.len();
        let mut gradients = vec![vec![0.0f32; self.n_components]; n];

        // Adjust P after early exaggeration phase
        let p_scale = if iter >= self.early_exaggeration_iter {
            1.0 / self.early_exaggeration
        } else {
            1.0
        };

        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }

                let p_ij = p[i][j] * p_scale;
                let q_ij = q[i][j] / sum_q.max(1e-10);

                // Compute gradient contribution
                let mult = 4.0 * (p_ij - q_ij) * q[i][j];

                for d in 0..self.n_components {
                    gradients[i][d] += mult * (embedding[i][d] - embedding[j][d]);
                }
            }
        }

        gradients
    }

    /// Initialize embedding randomly.
    fn initialize_embedding(&self, n: usize) -> Vec<Vec<f32>> {
        let mut embedding = Vec::with_capacity(n);
        let mut rng_state = self.seed;

        for _ in 0..n {
            let mut point = Vec::with_capacity(self.n_components);
            for _ in 0..self.n_components {
                // Simple LCG random number generator
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let rand_val = ((rng_state >> 33) as f32) / (u32::MAX as f32) - 0.5;
                point.push(rand_val * 0.0001); // Small initial values
            }
            embedding.push(point);
        }

        embedding
    }

    /// Center embedding to have zero mean.
    fn center_embedding(&self, embedding: &mut [Vec<f32>]) {
        if embedding.is_empty() {
            return;
        }

        let n = embedding.len();
        let dim = embedding[0].len();

        // Compute mean
        let mut mean = vec![0.0f32; dim];
        for point in embedding.iter() {
            for (i, &val) in point.iter().enumerate() {
                mean[i] += val;
            }
        }
        for val in &mut mean {
            *val /= n as f32;
        }

        // Subtract mean
        for point in embedding.iter_mut() {
            for (i, val) in point.iter_mut().enumerate() {
                *val -= mean[i];
            }
        }
    }
}

/// Builder for configuring t-SNE.
#[derive(Debug, Clone)]
pub struct TSNEBuilder {
    n_components: usize,
    perplexity: f32,
    learning_rate: f32,
    n_iter: usize,
    early_exaggeration: f32,
    seed: u64,
}

impl Default for TSNEBuilder {
    fn default() -> Self {
        Self {
            n_components: 3,
            perplexity: 30.0,
            learning_rate: 200.0,
            n_iter: 1000,
            early_exaggeration: 12.0,
            seed: 42,
        }
    }
}

impl TSNEBuilder {
    /// Set number of output dimensions.
    pub fn n_components(mut self, n: usize) -> Self {
        self.n_components = n;
        self
    }

    /// Set perplexity (effective number of neighbors).
    ///
    /// Typical values: 5-50. Higher values consider more neighbors.
    pub fn perplexity(mut self, p: f32) -> Self {
        self.perplexity = p;
        self
    }

    /// Set learning rate.
    ///
    /// Typical values: 10-1000. Default is 200.
    pub fn learning_rate(mut self, lr: f32) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set number of iterations.
    pub fn n_iter(mut self, n: usize) -> Self {
        self.n_iter = n;
        self
    }

    /// Set early exaggeration factor.
    pub fn early_exaggeration(mut self, e: f32) -> Self {
        self.early_exaggeration = e;
        self
    }

    /// Set random seed.
    pub fn seed(mut self, s: u64) -> Self {
        self.seed = s;
        self
    }

    /// Build the t-SNE instance.
    pub fn build(self) -> TSNE {
        TSNE {
            n_components: self.n_components,
            perplexity: self.perplexity,
            learning_rate: self.learning_rate,
            n_iter: self.n_iter,
            early_exaggeration: self.early_exaggeration,
            early_exaggeration_iter: 250,
            momentum: 0.5,
            final_momentum: 0.8,
            min_gain: 0.01,
            seed: self.seed,
            embedding: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_clustered_data(
        n_clusters: usize,
        points_per_cluster: usize,
        dim: usize,
    ) -> Vec<Vec<f32>> {
        let mut data = Vec::new();
        let mut seed = 12345u64;

        for c in 0..n_clusters {
            let center: Vec<f32> = (0..dim).map(|d| (c * 10 + d) as f32).collect();

            for _ in 0..points_per_cluster {
                let mut point = Vec::with_capacity(dim);
                for &center_val in &center {
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let noise = ((seed >> 33) as f32) / (u32::MAX as f32) - 0.5;
                    point.push(center_val + noise);
                }
                data.push(point);
            }
        }

        data
    }

    #[test]
    fn test_tsne_basic() {
        // Generate simple clustered data
        let data = generate_clustered_data(3, 50, 10);

        let mut tsne = TSNE::builder()
            .n_components(2)
            .perplexity(15.0)
            .n_iter(250) // Fewer iterations for test
            .build();

        let result = tsne.fit_transform(&data).unwrap();

        assert_eq!(result.len(), 150);
        assert_eq!(result[0].len(), 2);

        // Check that output values are reasonable (not NaN or huge)
        for point in &result {
            for &val in point {
                assert!(val.is_finite());
                assert!(val.abs() < 1000.0);
            }
        }
    }

    #[test]
    fn test_tsne_3d() {
        let data = generate_clustered_data(2, 50, 8);

        let mut tsne = TSNE::builder()
            .n_components(3)
            .perplexity(10.0)
            .n_iter(100)
            .build();

        let result = tsne.transform_3d(&data).unwrap();

        assert_eq!(result.len(), 100);
        for point in &result {
            assert!(point[0].is_finite());
            assert!(point[1].is_finite());
            assert!(point[2].is_finite());
        }
    }

    #[test]
    fn test_tsne_insufficient_data() {
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0]]; // Only 2 points

        let mut tsne = TSNE::new(2);
        let result = tsne.fit_transform(&data);

        assert!(matches!(
            result,
            Err(EmbeddingError::InsufficientData { .. })
        ));
    }
}
