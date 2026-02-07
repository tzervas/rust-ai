//! Uniform Manifold Approximation and Projection (UMAP) implementation.
//!
//! Basic implementation for dimensionality reduction that preserves
//! both local and global structure.

use super::{utils, EmbeddingError, Result};

/// UMAP for dimensionality reduction.
///
/// Preserves both local neighborhood structure and global topology
/// through fuzzy topological representation.
#[derive(Debug, Clone)]
pub struct UMAP {
    /// Number of output dimensions
    n_components: usize,
    /// Number of neighbors for local structure
    n_neighbors: usize,
    /// Minimum distance between embedded points
    min_dist: f32,
    /// Spread of embedded points
    spread: f32,
    /// Number of training epochs
    n_epochs: usize,
    /// Learning rate
    learning_rate: f32,
    /// Negative sample rate
    negative_sample_rate: usize,
    /// Random seed
    seed: u64,
    /// Current embedding
    embedding: Option<Vec<Vec<f32>>>,
    /// Fitted graph (for incremental projection)
    fitted_graph: Option<FuzzyGraph>,
}

/// Represents the fuzzy simplicial set (weighted k-NN graph).
#[derive(Debug, Clone)]
struct FuzzyGraph {
    /// Number of points
    n_points: usize,
    /// Sparse adjacency: (row, col, weight)
    edges: Vec<(usize, usize, f32)>,
    /// For each point, indices of its neighbors
    neighbors: Vec<Vec<usize>>,
}

impl Default for UMAP {
    fn default() -> Self {
        Self::new(3)
    }
}

impl UMAP {
    /// Create a new UMAP instance.
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            n_neighbors: 15,
            min_dist: 0.1,
            spread: 1.0,
            n_epochs: 200,
            learning_rate: 1.0,
            negative_sample_rate: 5,
            seed: 42,
            embedding: None,
            fitted_graph: None,
        }
    }

    /// Create a builder for configuration.
    pub fn builder() -> UMAPBuilder {
        UMAPBuilder::default()
    }

    /// Get number of neighbors.
    pub fn n_neighbors(&self) -> usize {
        self.n_neighbors
    }

    /// Fit and transform data.
    pub fn fit_transform(&mut self, data: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        if data.is_empty() {
            return Err(EmbeddingError::EmptyInput);
        }

        let n = data.len();

        if n < self.n_neighbors + 1 {
            return Err(EmbeddingError::InsufficientData {
                min: self.n_neighbors + 1,
                got: n,
            });
        }

        // Validate dimensions
        let dim = data[0].len();
        for row in data {
            if row.len() != dim {
                return Err(EmbeddingError::DimensionMismatch {
                    expected: dim,
                    got: row.len(),
                });
            }
        }

        // Build fuzzy simplicial set (k-NN graph with fuzzy weights)
        let graph = self.build_fuzzy_graph(data)?;

        // Initialize embedding
        let mut embedding = self.spectral_init(&graph);

        // Compute a/b parameters for the smooth approximation
        let (a, b) = self.find_ab_params();

        // Optimize embedding using SGD
        self.optimize_embedding(&mut embedding, &graph, a, b);

        self.fitted_graph = Some(graph);
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

    /// Project a single new point using the fitted model.
    ///
    /// Requires calling fit_transform first.
    pub fn transform_point(&self, point: &[f32], original_data: &[Vec<f32>]) -> Result<Vec<f32>> {
        let graph = self
            .fitted_graph
            .as_ref()
            .ok_or(EmbeddingError::NotFitted)?;
        let embedding = self.embedding.as_ref().ok_or(EmbeddingError::NotFitted)?;

        if original_data.is_empty() {
            return Err(EmbeddingError::EmptyInput);
        }

        // Find k nearest neighbors in original space
        let mut distances: Vec<(usize, f32)> = original_data
            .iter()
            .enumerate()
            .map(|(i, p)| (i, utils::euclidean_distance(point, p)))
            .collect();

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Compute weighted average of neighbor embeddings
        let k = self.n_neighbors.min(graph.n_points);
        let mut result = vec![0.0f32; self.n_components];
        let mut weight_sum = 0.0f32;

        for &(idx, dist) in distances.iter().take(k) {
            let weight = 1.0 / (dist + 1e-10);
            weight_sum += weight;

            for (d, val) in result.iter_mut().enumerate() {
                *val += weight * embedding[idx][d];
            }
        }

        for val in &mut result {
            *val /= weight_sum;
        }

        Ok(result)
    }

    /// Build the fuzzy simplicial set (weighted k-NN graph).
    fn build_fuzzy_graph(&self, data: &[Vec<f32>]) -> Result<FuzzyGraph> {
        let n = data.len();
        let k = self.n_neighbors;

        // Compute k-nearest neighbors for each point
        let mut neighbors = vec![Vec::with_capacity(k); n];
        let mut knn_distances = vec![Vec::with_capacity(k); n];

        for i in 0..n {
            // Compute distances to all other points
            let mut dists: Vec<(usize, f32)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| (j, utils::euclidean_distance(&data[i], &data[j])))
                .collect();

            // Sort by distance
            dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            // Keep k nearest
            for &(j, d) in dists.iter().take(k) {
                neighbors[i].push(j);
                knn_distances[i].push(d);
            }
        }

        // Compute local connectivity (rho) - distance to nearest neighbor
        let rho: Vec<f32> = knn_distances
            .iter()
            .map(|dists| dists.first().copied().unwrap_or(0.0))
            .collect();

        // Compute sigma using binary search to match log2(k) perplexity
        let target = (k as f32).log2();
        let sigma: Vec<f32> = (0..n)
            .map(|i| self.find_sigma(&knn_distances[i], rho[i], target))
            .collect();

        // Build fuzzy set memberships
        let mut edges = Vec::new();

        for i in 0..n {
            for (idx, &j) in neighbors[i].iter().enumerate() {
                let d = knn_distances[i][idx];
                let rho_i = rho[i];
                let sigma_i = sigma[i];

                // Compute membership strength
                let membership = if d <= rho_i {
                    1.0
                } else {
                    (-(d - rho_i) / sigma_i).exp()
                };

                if membership > 1e-10 {
                    edges.push((i, j, membership));
                }
            }
        }

        // Symmetrize: combine(a, b) = a + b - a*b
        let mut edge_map = std::collections::HashMap::new();

        for &(i, j, w) in &edges {
            let key = if i < j { (i, j) } else { (j, i) };
            let entry = edge_map.entry(key).or_insert((0.0f32, 0.0f32));

            if i < j {
                entry.0 = w;
            } else {
                entry.1 = w;
            }
        }

        let symmetric_edges: Vec<(usize, usize, f32)> = edge_map
            .into_iter()
            .flat_map(|((i, j), (w1, w2))| {
                let combined = w1 + w2 - w1 * w2;
                if combined > 1e-10 {
                    vec![(i, j, combined), (j, i, combined)]
                } else {
                    vec![]
                }
            })
            .collect();

        Ok(FuzzyGraph {
            n_points: n,
            edges: symmetric_edges,
            neighbors,
        })
    }

    /// Binary search for sigma to achieve target sum of memberships.
    fn find_sigma(&self, distances: &[f32], rho: f32, target: f32) -> f32 {
        let mut sigma_lo = 1e-10f32;
        let mut sigma_hi = 1e10f32;
        let mut sigma = 1.0f32;

        for _ in 0..64 {
            let mut sum = 0.0f32;

            for &d in distances {
                if d > rho {
                    sum += (-(d - rho) / sigma).exp();
                } else {
                    sum += 1.0;
                }
            }

            if (sum - target).abs() < 1e-5 {
                break;
            }

            if sum > target {
                sigma_hi = sigma;
            } else {
                sigma_lo = sigma;
            }

            sigma = (sigma_lo + sigma_hi) / 2.0;
        }

        sigma
    }

    /// Find a, b parameters for the smooth approximation curve.
    fn find_ab_params(&self) -> (f32, f32) {
        // These approximate the curve: 1 / (1 + a * d^(2b))
        // where d is the distance and we want:
        // - f(min_dist) ~= 1
        // - f(spread) ~= 0.5

        let min_dist = self.min_dist;
        let spread = self.spread;

        // Use approximate closed-form solution
        let b = 1.0f32;
        let a = ((spread / min_dist).powf(b) - 1.0) / spread.powf(2.0 * b);

        (a.max(0.001), b)
    }

    /// Initialize embedding using spectral layout (simplified).
    fn spectral_init(&self, graph: &FuzzyGraph) -> Vec<Vec<f32>> {
        let n = graph.n_points;

        // Simplified spectral initialization using random walk
        let mut embedding = Vec::with_capacity(n);
        let mut rng_state = self.seed;

        // Initialize with random values scaled by graph structure
        for i in 0..n {
            let mut point = Vec::with_capacity(self.n_components);

            // Use neighbor indices to seed position
            let neighbor_sum: f32 = graph.neighbors[i].iter().take(5).map(|&j| j as f32).sum();

            for d in 0..self.n_components {
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let rand = ((rng_state >> 33) as f32) / (u32::MAX as f32) - 0.5;

                // Combine random with structure-based seeding
                let val = rand * 10.0 + (neighbor_sum / n as f32) * (d as f32 + 1.0);
                point.push(val);
            }

            embedding.push(point);
        }

        // Run a few iterations of spectral-like smoothing
        for _ in 0..10 {
            let old_embedding = embedding.clone();

            for i in 0..n {
                let neighbors = &graph.neighbors[i];
                if neighbors.is_empty() {
                    continue;
                }

                for d in 0..self.n_components {
                    let neighbor_avg: f32 =
                        neighbors.iter().map(|&j| old_embedding[j][d]).sum::<f32>()
                            / neighbors.len() as f32;

                    embedding[i][d] = 0.5 * old_embedding[i][d] + 0.5 * neighbor_avg;
                }
            }
        }

        embedding
    }

    /// Optimize embedding using negative sampling SGD.
    fn optimize_embedding(&self, embedding: &mut [Vec<f32>], graph: &FuzzyGraph, a: f32, b: f32) {
        let n = graph.n_points;
        let n_edges = graph.edges.len();

        if n_edges == 0 {
            return;
        }

        let mut rng_state = self.seed;

        // Compute epoch schedule (which edges to sample when)
        let epochs_per_sample: Vec<f32> = graph
            .edges
            .iter()
            .map(|&(_, _, w)| {
                let max_w = graph.edges.iter().map(|e| e.2).fold(0.0f32, f32::max);
                if w > 1e-10 {
                    max_w / w
                } else {
                    f32::MAX
                }
            })
            .collect();

        let epochs_per_negative_sample: Vec<f32> = epochs_per_sample
            .iter()
            .map(|&e| e / self.negative_sample_rate as f32)
            .collect();

        let mut epoch_of_next_sample: Vec<f32> = epochs_per_sample.clone();
        let mut epoch_of_next_negative_sample: Vec<f32> = epochs_per_negative_sample.clone();

        // Training loop
        for epoch in 0..self.n_epochs {
            let alpha = self.learning_rate * (1.0 - epoch as f32 / self.n_epochs as f32);

            for (edge_idx, &(i, j, _)) in graph.edges.iter().enumerate() {
                if epoch_of_next_sample[edge_idx] > epoch as f32 {
                    continue;
                }

                // Positive sample: attract
                let dist_sq = utils::squared_euclidean_distance(&embedding[i], &embedding[j]);
                let dist = dist_sq.sqrt().max(0.001);

                // Gradient for attractive force
                let grad_coeff =
                    -2.0 * a * b * dist.powf(2.0 * b - 2.0) / (1.0 + a * dist.powf(2.0 * b));

                for d in 0..self.n_components {
                    let grad = grad_coeff * (embedding[i][d] - embedding[j][d]);
                    let clipped = grad.clamp(-4.0, 4.0);

                    embedding[i][d] += alpha * clipped;
                    embedding[j][d] -= alpha * clipped;
                }

                epoch_of_next_sample[edge_idx] += epochs_per_sample[edge_idx];

                // Negative samples: repel
                let n_neg = self.negative_sample_rate;
                for _ in 0..n_neg {
                    if epoch_of_next_negative_sample[edge_idx] > epoch as f32 {
                        break;
                    }

                    // Random negative sample
                    rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let k = (rng_state as usize) % n;

                    if k == i {
                        continue;
                    }

                    let dist_sq = utils::squared_euclidean_distance(&embedding[i], &embedding[k]);
                    let dist = dist_sq.sqrt().max(0.001);

                    // Gradient for repulsive force
                    let grad_coeff = 2.0 * b / ((0.001 + dist_sq) * (1.0 + a * dist.powf(2.0 * b)));

                    for d in 0..self.n_components {
                        let grad = grad_coeff * (embedding[i][d] - embedding[k][d]);
                        let clipped = grad.clamp(-4.0, 4.0);

                        embedding[i][d] += alpha * clipped;
                    }

                    epoch_of_next_negative_sample[edge_idx] += epochs_per_negative_sample[edge_idx];
                }
            }
        }
    }
}

/// Builder for configuring UMAP.
#[derive(Debug, Clone)]
pub struct UMAPBuilder {
    n_components: usize,
    n_neighbors: usize,
    min_dist: f32,
    spread: f32,
    n_epochs: usize,
    learning_rate: f32,
    negative_sample_rate: usize,
    seed: u64,
}

impl Default for UMAPBuilder {
    fn default() -> Self {
        Self {
            n_components: 3,
            n_neighbors: 15,
            min_dist: 0.1,
            spread: 1.0,
            n_epochs: 200,
            learning_rate: 1.0,
            negative_sample_rate: 5,
            seed: 42,
        }
    }
}

impl UMAPBuilder {
    /// Set number of output dimensions.
    pub fn n_components(mut self, n: usize) -> Self {
        self.n_components = n;
        self
    }

    /// Set number of neighbors for local structure.
    pub fn n_neighbors(mut self, n: usize) -> Self {
        self.n_neighbors = n;
        self
    }

    /// Set minimum distance between embedded points.
    pub fn min_dist(mut self, d: f32) -> Self {
        self.min_dist = d;
        self
    }

    /// Set spread of embedded points.
    pub fn spread(mut self, s: f32) -> Self {
        self.spread = s;
        self
    }

    /// Set number of optimization epochs.
    pub fn n_epochs(mut self, n: usize) -> Self {
        self.n_epochs = n;
        self
    }

    /// Set learning rate.
    pub fn learning_rate(mut self, lr: f32) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set negative sample rate.
    pub fn negative_sample_rate(mut self, rate: usize) -> Self {
        self.negative_sample_rate = rate;
        self
    }

    /// Set random seed.
    pub fn seed(mut self, s: u64) -> Self {
        self.seed = s;
        self
    }

    /// Build the UMAP instance.
    pub fn build(self) -> UMAP {
        UMAP {
            n_components: self.n_components,
            n_neighbors: self.n_neighbors,
            min_dist: self.min_dist,
            spread: self.spread,
            n_epochs: self.n_epochs,
            learning_rate: self.learning_rate,
            negative_sample_rate: self.negative_sample_rate,
            seed: self.seed,
            embedding: None,
            fitted_graph: None,
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
    fn test_umap_basic() {
        let data = generate_clustered_data(3, 30, 10);

        let mut umap = UMAP::builder()
            .n_components(2)
            .n_neighbors(10)
            .n_epochs(50)
            .build();

        let result = umap.fit_transform(&data).unwrap();

        assert_eq!(result.len(), 90);
        assert_eq!(result[0].len(), 2);

        // Check values are finite
        for point in &result {
            for &val in point {
                assert!(val.is_finite());
            }
        }
    }

    #[test]
    fn test_umap_3d() {
        let data = generate_clustered_data(2, 30, 8);

        let mut umap = UMAP::builder()
            .n_components(3)
            .n_neighbors(10)
            .n_epochs(50)
            .build();

        let result = umap.transform_3d(&data).unwrap();

        assert_eq!(result.len(), 60);
        for point in &result {
            assert!(point[0].is_finite());
            assert!(point[1].is_finite());
            assert!(point[2].is_finite());
        }
    }

    #[test]
    fn test_umap_transform_point() {
        let data = generate_clustered_data(2, 30, 5);

        let mut umap = UMAP::builder()
            .n_components(2)
            .n_neighbors(10)
            .n_epochs(50)
            .build();

        umap.fit_transform(&data).unwrap();

        // Transform a new point similar to cluster 0
        let new_point = vec![0.5, 1.5, 2.5, 3.5, 4.5];
        let projected = umap.transform_point(&new_point, &data).unwrap();

        assert_eq!(projected.len(), 2);
        assert!(projected[0].is_finite());
        assert!(projected[1].is_finite());
    }

    #[test]
    fn test_umap_insufficient_data() {
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0]]; // Only 2 points

        let mut umap = UMAP::new(2);
        let result = umap.fit_transform(&data);

        assert!(matches!(
            result,
            Err(EmbeddingError::InsufficientData { .. })
        ));
    }
}
