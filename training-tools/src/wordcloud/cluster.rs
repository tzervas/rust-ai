//! Clustering algorithms for semantic grouping of tokens

use std::collections::HashMap;

/// Configuration for clustering algorithms
#[derive(Debug, Clone)]
pub struct ClusterConfig {
    /// Maximum iterations for k-means
    pub max_iterations: usize,
    /// Convergence threshold
    pub convergence_threshold: f32,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
    /// Minimum cluster size (clusters smaller than this are merged)
    pub min_cluster_size: usize,
}

impl Default for ClusterConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            convergence_threshold: 1e-4,
            seed: None,
            min_cluster_size: 1,
        }
    }
}

/// A semantic cluster of related tokens
#[derive(Debug, Clone)]
pub struct ConceptCluster {
    /// Center point of the cluster in 3D space
    pub center: [f32; 3],
    /// Radius encompassing all tokens in the cluster
    pub radius: f32,
    /// Indices of tokens belonging to this cluster
    pub tokens: Vec<usize>,
    /// Human-readable label for the cluster
    pub label: String,
}

impl ConceptCluster {
    /// Create a new empty cluster
    pub fn new(center: [f32; 3]) -> Self {
        Self {
            center,
            radius: 0.0,
            tokens: Vec::new(),
            label: String::new(),
        }
    }

    /// Check if a point is inside the cluster sphere
    pub fn contains(&self, point: [f32; 3]) -> bool {
        let dx = point[0] - self.center[0];
        let dy = point[1] - self.center[1];
        let dz = point[2] - self.center[2];
        (dx * dx + dy * dy + dz * dz).sqrt() <= self.radius
    }

    /// Get the number of tokens in this cluster
    pub fn size(&self) -> usize {
        self.tokens.len()
    }

    /// Check if the cluster is empty
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Calculate distance from cluster center to a point
    pub fn distance_to(&self, point: [f32; 3]) -> f32 {
        let dx = point[0] - self.center[0];
        let dy = point[1] - self.center[1];
        let dz = point[2] - self.center[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}

/// K-means clustering implementation
#[derive(Debug, Clone)]
pub struct KMeansClusterer {
    config: ClusterConfig,
}

impl KMeansClusterer {
    /// Create a new k-means clusterer with the given configuration
    pub fn new(config: ClusterConfig) -> Self {
        Self { config }
    }

    /// Perform k-means clustering on the given points
    ///
    /// Returns (assignments, centroids) where:
    /// - assignments[i] is the cluster id for point i
    /// - centroids[j] is the centroid vector for cluster j
    pub fn cluster(&self, points: &[Vec<f32>], k: usize) -> (Vec<usize>, Vec<Vec<f32>>) {
        if points.is_empty() || k == 0 {
            return (Vec::new(), Vec::new());
        }

        let n = points.len();
        let dim = points[0].len();

        // Handle edge case where k >= n
        if k >= n {
            let assignments: Vec<usize> = (0..n).collect();
            let centroids: Vec<Vec<f32>> = points.to_vec();
            return (assignments, centroids);
        }

        // Initialize centroids using k-means++ algorithm
        let mut centroids = self.kmeans_plus_plus_init(points, k);
        let mut assignments = vec![0usize; n];
        let mut prev_assignments = vec![usize::MAX; n];

        for iteration in 0..self.config.max_iterations {
            // Assignment step: assign each point to nearest centroid
            for (i, point) in points.iter().enumerate() {
                let mut min_dist = f32::MAX;
                let mut best_cluster = 0;

                for (j, centroid) in centroids.iter().enumerate() {
                    let dist = euclidean_distance(point, centroid);
                    if dist < min_dist {
                        min_dist = dist;
                        best_cluster = j;
                    }
                }
                assignments[i] = best_cluster;
            }

            // Check for convergence
            if assignments == prev_assignments {
                break;
            }
            prev_assignments.clone_from(&assignments);

            // Update step: compute new centroids
            let mut new_centroids = vec![vec![0.0f32; dim]; k];
            let mut counts = vec![0usize; k];

            for (i, &cluster) in assignments.iter().enumerate() {
                counts[cluster] += 1;
                for (j, &val) in points[i].iter().enumerate() {
                    new_centroids[cluster][j] += val;
                }
            }

            // Average the centroids
            for (j, centroid) in new_centroids.iter_mut().enumerate() {
                if counts[j] > 0 {
                    for val in centroid.iter_mut() {
                        *val /= counts[j] as f32;
                    }
                } else {
                    // Empty cluster: keep old centroid or reinitialize
                    centroid.clone_from(&centroids[j]);
                }
            }

            // Check for convergence using centroid movement
            let max_movement: f32 = centroids
                .iter()
                .zip(new_centroids.iter())
                .map(|(old, new)| euclidean_distance(old, new))
                .fold(0.0f32, f32::max);

            centroids = new_centroids;

            if max_movement < self.config.convergence_threshold {
                tracing::debug!("K-means converged at iteration {}", iteration);
                break;
            }
        }

        (assignments, centroids)
    }

    /// Initialize centroids using k-means++ algorithm
    fn kmeans_plus_plus_init(&self, points: &[Vec<f32>], k: usize) -> Vec<Vec<f32>> {
        let n = points.len();
        let mut centroids = Vec::with_capacity(k);

        // Use deterministic initialization based on seed or data
        let first_idx = if let Some(seed) = self.config.seed {
            (seed as usize) % n
        } else {
            // Pick point with median L2 norm
            let mut norms: Vec<(usize, f32)> = points
                .iter()
                .enumerate()
                .map(|(i, p)| (i, p.iter().map(|x| x * x).sum::<f32>().sqrt()))
                .collect();
            norms.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            norms[n / 2].0
        };

        centroids.push(points[first_idx].clone());

        // Add remaining centroids
        for _ in 1..k {
            // Calculate distances to nearest centroid for each point
            let distances: Vec<f32> = points
                .iter()
                .map(|point| {
                    centroids
                        .iter()
                        .map(|c| euclidean_distance(point, c))
                        .fold(f32::MAX, f32::min)
                })
                .collect();

            // Pick point with maximum distance (deterministic k-means++)
            let max_idx = distances
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);

            centroids.push(points[max_idx].clone());
        }

        centroids
    }
}

/// Hierarchical clustering for building concept trees
#[derive(Debug, Clone)]
pub struct HierarchicalClusterer {
    /// Linkage method
    pub linkage: Linkage,
}

/// Linkage methods for hierarchical clustering
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Linkage {
    /// Single linkage (minimum distance)
    Single,
    /// Complete linkage (maximum distance)
    Complete,
    /// Average linkage (UPGMA)
    Average,
}

impl HierarchicalClusterer {
    /// Create a new hierarchical clusterer
    pub fn new(linkage: Linkage) -> Self {
        Self { linkage }
    }

    /// Perform agglomerative clustering
    ///
    /// Returns a dendrogram represented as merge steps:
    /// Each entry (i, j, distance) means clusters i and j were merged at the given distance
    pub fn cluster(&self, points: &[Vec<f32>]) -> Vec<(usize, usize, f32)> {
        let n = points.len();
        if n <= 1 {
            return Vec::new();
        }

        // Initialize distance matrix
        let mut distances: HashMap<(usize, usize), f32> = HashMap::new();
        for i in 0..n {
            for j in (i + 1)..n {
                let dist = euclidean_distance(&points[i], &points[j]);
                distances.insert((i, j), dist);
            }
        }

        // Track which clusters are active
        let mut active: Vec<bool> = vec![true; n];
        let mut cluster_sizes: Vec<usize> = vec![1; n];
        let mut merges = Vec::with_capacity(n - 1);

        // Cluster indices start at n for merged clusters
        let mut next_cluster = n;

        for _ in 0..(n - 1) {
            // Find closest pair of active clusters
            let mut min_dist = f32::MAX;
            let mut best_pair = (0, 0);

            for (&(i, j), &dist) in &distances {
                if active[i] && active[j] && dist < min_dist {
                    min_dist = dist;
                    best_pair = (i, j);
                }
            }

            let (ci, cj) = best_pair;
            merges.push((ci, cj, min_dist));

            // Merge clusters ci and cj into a new cluster
            active[ci] = false;
            active[cj] = false;

            // Update distances for the new merged cluster
            let new_size = cluster_sizes[ci] + cluster_sizes[cj];

            for k in 0..active.len() {
                if !active[k] || k == ci || k == cj {
                    continue;
                }

                let dist_ci = distances
                    .get(&(ci.min(k), ci.max(k)))
                    .copied()
                    .unwrap_or(f32::MAX);
                let dist_cj = distances
                    .get(&(cj.min(k), cj.max(k)))
                    .copied()
                    .unwrap_or(f32::MAX);

                let new_dist = match self.linkage {
                    Linkage::Single => dist_ci.min(dist_cj),
                    Linkage::Complete => dist_ci.max(dist_cj),
                    Linkage::Average => {
                        (dist_ci * cluster_sizes[ci] as f32 + dist_cj * cluster_sizes[cj] as f32)
                            / new_size as f32
                    }
                };

                // Store distance with new cluster index
                distances.insert((k.min(next_cluster), k.max(next_cluster)), new_dist);
            }

            // Mark new cluster as active
            if next_cluster < active.len() {
                active[next_cluster] = true;
                cluster_sizes[next_cluster] = new_size;
            } else {
                active.push(true);
                cluster_sizes.push(new_size);
            }

            next_cluster += 1;
        }

        merges
    }

    /// Cut the dendrogram at a specific number of clusters
    pub fn cut_tree(
        &self,
        merges: &[(usize, usize, f32)],
        n_points: usize,
        n_clusters: usize,
    ) -> Vec<usize> {
        if n_clusters >= n_points {
            return (0..n_points).collect();
        }

        // Start with each point in its own cluster
        let mut assignments: Vec<usize> = (0..n_points).collect();

        // Apply merges until we have n_clusters
        let merges_to_apply = n_points.saturating_sub(n_clusters);

        for (i, j, _) in merges.iter().take(merges_to_apply) {
            // Merge cluster j into cluster i
            let target = assignments[*i];
            let source = assignments[*j];

            for a in &mut assignments {
                if *a == source {
                    *a = target;
                }
            }
        }

        // Renumber clusters to be consecutive
        let mut cluster_map: HashMap<usize, usize> = HashMap::new();
        let mut next_id = 0;

        for a in &mut assignments {
            if let Some(&new_id) = cluster_map.get(a) {
                *a = new_id;
            } else {
                cluster_map.insert(*a, next_id);
                *a = next_id;
                next_id += 1;
            }
        }

        assignments
    }
}

/// Calculate Euclidean distance between two vectors
fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}

/// Silhouette score for evaluating cluster quality
pub fn silhouette_score(points: &[Vec<f32>], assignments: &[usize]) -> f32 {
    if points.len() <= 1 {
        return 0.0;
    }

    let n = points.len();
    let mut total_score = 0.0;

    for i in 0..n {
        let cluster_i = assignments[i];

        // Calculate a(i): mean distance to points in same cluster
        let same_cluster: Vec<_> = (0..n)
            .filter(|&j| j != i && assignments[j] == cluster_i)
            .collect();

        let a_i = if same_cluster.is_empty() {
            0.0
        } else {
            same_cluster
                .iter()
                .map(|&j| euclidean_distance(&points[i], &points[j]))
                .sum::<f32>()
                / same_cluster.len() as f32
        };

        // Calculate b(i): minimum mean distance to points in other clusters
        let other_clusters: Vec<usize> = assignments
            .iter()
            .copied()
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .filter(|&c| c != cluster_i)
            .collect();

        let b_i = other_clusters
            .iter()
            .map(|&c| {
                let other_points: Vec<_> = (0..n).filter(|&j| assignments[j] == c).collect();
                if other_points.is_empty() {
                    f32::MAX
                } else {
                    other_points
                        .iter()
                        .map(|&j| euclidean_distance(&points[i], &points[j]))
                        .sum::<f32>()
                        / other_points.len() as f32
                }
            })
            .fold(f32::MAX, f32::min);

        // Silhouette coefficient for point i
        let s_i = if a_i.max(b_i) == 0.0 {
            0.0
        } else {
            (b_i - a_i) / a_i.max(b_i)
        };

        total_score += s_i;
    }

    total_score / n as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kmeans_basic() {
        let points = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.1],
            vec![10.0, 10.0],
            vec![10.1, 10.1],
        ];

        let clusterer = KMeansClusterer::new(ClusterConfig::default());
        let (assignments, _centroids) = clusterer.cluster(&points, 2);

        // Points 0,1 should be in one cluster, 2,3 in another
        assert_eq!(assignments[0], assignments[1]);
        assert_eq!(assignments[2], assignments[3]);
        assert_ne!(assignments[0], assignments[2]);
    }

    #[test]
    fn test_kmeans_single_cluster() {
        let points = vec![vec![1.0, 1.0], vec![2.0, 2.0], vec![3.0, 3.0]];

        let clusterer = KMeansClusterer::new(ClusterConfig::default());
        let (assignments, centroids) = clusterer.cluster(&points, 1);

        assert!(assignments.iter().all(|&a| a == 0));
        assert_eq!(centroids.len(), 1);
    }

    #[test]
    fn test_concept_cluster() {
        let mut cluster = ConceptCluster::new([0.0, 0.0, 0.0]);
        cluster.radius = 5.0;

        assert!(cluster.contains([1.0, 1.0, 1.0]));
        assert!(!cluster.contains([10.0, 10.0, 10.0]));
    }

    #[test]
    fn test_hierarchical_clustering() {
        let points = vec![
            vec![0.0, 0.0],
            vec![0.5, 0.5],
            vec![10.0, 10.0],
            vec![10.5, 10.5],
        ];

        let clusterer = HierarchicalClusterer::new(Linkage::Single);
        let merges = clusterer.cluster(&points);

        assert_eq!(merges.len(), 3); // n-1 merges

        // Cut into 2 clusters
        let assignments = clusterer.cut_tree(&merges, 4, 2);
        assert_eq!(assignments[0], assignments[1]);
        assert_eq!(assignments[2], assignments[3]);
    }

    #[test]
    fn test_silhouette_score() {
        let points = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.1],
            vec![10.0, 10.0],
            vec![10.1, 10.1],
        ];
        let assignments = vec![0, 0, 1, 1];

        let score = silhouette_score(&points, &assignments);
        // Good clustering should have high silhouette score
        assert!(score > 0.9);
    }
}
