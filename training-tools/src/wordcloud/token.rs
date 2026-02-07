//! Token representation and TokenCloud3D implementation

use super::cluster::{ClusterConfig, ConceptCluster, KMeansClusterer};
use super::layout::{ForceDirectedLayout, Layout3D, LayoutConfig};
use super::render::RenderToken;
use super::{Result, WordCloudError};

/// A single token in 3D space
#[derive(Debug, Clone)]
pub struct Token3D {
    /// The text content of the token
    pub text: String,
    /// 3D position [x, y, z]
    pub position: [f32; 3],
    /// Size based on frequency (normalized 0.0-1.0)
    pub size: f32,
    /// RGBA color based on category/cluster
    pub color: [f32; 4],
    /// Importance score (0.0-1.0)
    pub importance: f32,
    /// Original frequency count
    pub frequency: u32,
    /// Cluster assignment (None if not clustered)
    pub cluster_id: Option<usize>,
    /// Original embedding vector (if provided)
    pub embedding: Option<Vec<f32>>,
}

impl Token3D {
    /// Create a new token with default position and color
    pub fn new(text: String, frequency: u32) -> Self {
        Self {
            text,
            position: [0.0, 0.0, 0.0],
            size: 0.0,
            color: [1.0, 1.0, 1.0, 1.0], // White default
            importance: 0.0,
            frequency,
            cluster_id: None,
            embedding: None,
        }
    }

    /// Set the embedding vector for this token
    pub fn set_embedding(&mut self, embedding: Vec<f32>) {
        self.embedding = Some(embedding);
    }

    /// Get the distance to another token in 3D space
    pub fn distance_to(&self, other: &Token3D) -> f32 {
        let dx = self.position[0] - other.position[0];
        let dy = self.position[1] - other.position[1];
        let dz = self.position[2] - other.position[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}

/// Main 3D token cloud structure
#[derive(Debug, Clone)]
pub struct TokenCloud3D {
    /// All tokens in the cloud
    pub(crate) tokens: Vec<Token3D>,
    /// Computed clusters
    pub(crate) clusters: Vec<ConceptCluster>,
    /// Whether positions have been computed
    positions_computed: bool,
    /// Layout configuration
    layout_config: LayoutConfig,
    /// Cluster configuration
    cluster_config: ClusterConfig,
}

impl TokenCloud3D {
    /// Create a new token cloud from vocabulary and frequencies
    ///
    /// # Panics
    ///
    /// Panics if vocabulary and frequencies have different lengths.
    /// Use `try_from_vocabulary` for a fallible version.
    pub fn from_vocabulary(vocab: &[String], frequencies: &[u32]) -> Self {
        Self::try_from_vocabulary(vocab, frequencies)
            .expect("vocabulary and frequencies must have same length")
    }

    /// Try to create a new token cloud from vocabulary and frequencies
    pub fn try_from_vocabulary(vocab: &[String], frequencies: &[u32]) -> Result<Self> {
        if vocab.len() != frequencies.len() {
            return Err(WordCloudError::LengthMismatch {
                vocab_len: vocab.len(),
                freq_len: frequencies.len(),
            });
        }

        if vocab.is_empty() {
            return Err(WordCloudError::NoTokens);
        }

        // Find max frequency for normalization
        let max_freq = *frequencies.iter().max().unwrap_or(&1) as f32;

        let tokens: Vec<Token3D> = vocab
            .iter()
            .zip(frequencies.iter())
            .map(|(text, &freq)| {
                let mut token = Token3D::new(text.clone(), freq);
                // Normalize size and importance based on frequency
                let normalized = freq as f32 / max_freq;
                token.size = 0.1 + normalized * 0.9; // Range: 0.1 to 1.0
                token.importance = normalized;
                token
            })
            .collect();

        Ok(Self {
            tokens,
            clusters: Vec::new(),
            positions_computed: false,
            layout_config: LayoutConfig::default(),
            cluster_config: ClusterConfig::default(),
        })
    }

    /// Set layout configuration
    pub fn set_layout_config(&mut self, config: LayoutConfig) {
        self.layout_config = config;
        self.positions_computed = false;
    }

    /// Set cluster configuration
    pub fn set_cluster_config(&mut self, config: ClusterConfig) {
        self.cluster_config = config;
    }

    /// Add embeddings to tokens for semantic positioning
    pub fn add_embeddings(&mut self, embeddings: &[Vec<f32>]) -> Result<()> {
        if embeddings.len() != self.tokens.len() {
            return Err(WordCloudError::EmbeddingCountMismatch {
                expected: self.tokens.len(),
                got: embeddings.len(),
            });
        }

        // Check dimension consistency
        if !embeddings.is_empty() {
            let expected_dim = embeddings[0].len();
            for (i, emb) in embeddings.iter().enumerate() {
                if emb.len() != expected_dim {
                    return Err(WordCloudError::EmbeddingDimensionMismatch {
                        expected: expected_dim,
                        got: emb.len(),
                    });
                }
                self.tokens[i].set_embedding(emb.clone());
            }
        }

        // Embeddings changed, need to recompute positions
        self.positions_computed = false;
        Ok(())
    }

    /// Cluster tokens into semantic groups
    pub fn cluster(&mut self, n_clusters: usize) {
        self.try_cluster(n_clusters).expect("clustering failed");
    }

    /// Try to cluster tokens into semantic groups
    pub fn try_cluster(&mut self, n_clusters: usize) -> Result<()> {
        if n_clusters == 0 || n_clusters > self.tokens.len() {
            return Err(WordCloudError::InvalidClusterCount {
                count: n_clusters,
                max: self.tokens.len(),
            });
        }

        // Collect embeddings or use positions for clustering
        let points: Vec<Vec<f32>> = self
            .tokens
            .iter()
            .map(|t| t.embedding.clone().unwrap_or_else(|| t.position.to_vec()))
            .collect();

        // Perform k-means clustering
        let clusterer = KMeansClusterer::new(self.cluster_config.clone());
        let (assignments, centroids) = clusterer.cluster(&points, n_clusters);

        // Assign clusters to tokens
        for (i, &cluster_id) in assignments.iter().enumerate() {
            self.tokens[i].cluster_id = Some(cluster_id);
        }

        // Create ConceptCluster structures
        self.clusters = (0..n_clusters)
            .map(|cluster_id| {
                let token_indices: Vec<usize> = assignments
                    .iter()
                    .enumerate()
                    .filter(|(_, &c)| c == cluster_id)
                    .map(|(i, _)| i)
                    .collect();

                // Project centroid to 3D if needed
                let center = if centroids[cluster_id].len() >= 3 {
                    [
                        centroids[cluster_id][0],
                        centroids[cluster_id][1],
                        centroids[cluster_id][2],
                    ]
                } else {
                    // Use PCA projection or simple padding
                    let mut center = [0.0f32; 3];
                    for (i, &v) in centroids[cluster_id].iter().take(3).enumerate() {
                        center[i] = v;
                    }
                    center
                };

                // Calculate radius as max distance from center
                let radius = token_indices
                    .iter()
                    .map(|&i| {
                        let pos = &self.tokens[i].position;
                        let dx = pos[0] - center[0];
                        let dy = pos[1] - center[1];
                        let dz = pos[2] - center[2];
                        (dx * dx + dy * dy + dz * dz).sqrt()
                    })
                    .fold(0.0f32, f32::max);

                // Generate cluster label from most frequent token
                let label = token_indices
                    .iter()
                    .max_by_key(|&&i| self.tokens[i].frequency)
                    .map(|&i| format!("Cluster: {}", self.tokens[i].text))
                    .unwrap_or_else(|| format!("Cluster {}", cluster_id));

                ConceptCluster {
                    center,
                    radius: radius.max(0.1), // Minimum radius
                    tokens: token_indices,
                    label,
                }
            })
            .collect();

        // Assign colors based on cluster
        self.assign_cluster_colors();

        Ok(())
    }

    /// Assign distinct colors to each cluster
    fn assign_cluster_colors(&mut self) {
        let n_clusters = self.clusters.len();
        if n_clusters == 0 {
            return;
        }

        // Generate distinct colors using HSV
        let cluster_colors: Vec<[f32; 4]> = (0..n_clusters)
            .map(|i| {
                let hue = (i as f32 / n_clusters as f32) * 360.0;
                let (r, g, b) = hsv_to_rgb(hue, 0.7, 0.9);
                [r, g, b, 1.0]
            })
            .collect();

        for token in &mut self.tokens {
            if let Some(cluster_id) = token.cluster_id {
                token.color = cluster_colors[cluster_id];
            }
        }
    }

    /// Compute 3D positions for all tokens
    pub fn compute_layout(&mut self) {
        let layout = ForceDirectedLayout::new(self.layout_config.clone());

        // Collect embeddings for layout computation
        let embeddings: Vec<Option<&Vec<f32>>> =
            self.tokens.iter().map(|t| t.embedding.as_ref()).collect();

        let frequencies: Vec<f32> = self.tokens.iter().map(|t| t.frequency as f32).collect();

        let positions = layout.compute(&embeddings, &frequencies);

        for (token, pos) in self.tokens.iter_mut().zip(positions.into_iter()) {
            token.position = pos;
        }

        self.positions_computed = true;
    }

    /// Convert to render-ready data
    pub fn to_render_data(&self) -> Vec<RenderToken> {
        self.tokens
            .iter()
            .enumerate()
            .map(|(id, token)| RenderToken {
                id,
                text: token.text.clone(),
                position: token.position,
                size: token.size,
                color: token.color,
                importance: token.importance,
                cluster_id: token.cluster_id,
            })
            .collect()
    }

    /// Get token count
    pub fn token_count(&self) -> usize {
        self.tokens.len()
    }

    /// Get cluster count
    pub fn cluster_count(&self) -> usize {
        self.clusters.len()
    }

    /// Get tokens in a specific cluster
    pub fn tokens_in_cluster(&self, cluster_id: usize) -> Vec<&Token3D> {
        if cluster_id >= self.clusters.len() {
            return Vec::new();
        }
        self.clusters[cluster_id]
            .tokens
            .iter()
            .map(|&i| &self.tokens[i])
            .collect()
    }

    /// Get all clusters
    pub fn clusters(&self) -> &[ConceptCluster] {
        &self.clusters
    }

    /// Get a token by index
    pub fn get_token(&self, index: usize) -> Option<&Token3D> {
        self.tokens.get(index)
    }

    /// Get mutable access to a token by index
    pub fn get_token_mut(&mut self, index: usize) -> Option<&mut Token3D> {
        self.tokens.get_mut(index)
    }

    /// Find tokens near a 3D point
    pub fn find_near(&self, point: [f32; 3], radius: f32) -> Vec<(usize, &Token3D)> {
        self.tokens
            .iter()
            .enumerate()
            .filter(|(_, token)| {
                let dx = token.position[0] - point[0];
                let dy = token.position[1] - point[1];
                let dz = token.position[2] - point[2];
                (dx * dx + dy * dy + dz * dz).sqrt() <= radius
            })
            .collect()
    }

    /// Get the bounding box of all tokens
    pub fn bounding_box(&self) -> ([f32; 3], [f32; 3]) {
        if self.tokens.is_empty() {
            return ([0.0; 3], [0.0; 3]);
        }

        let mut min = [f32::MAX; 3];
        let mut max = [f32::MIN; 3];

        for token in &self.tokens {
            for i in 0..3 {
                min[i] = min[i].min(token.position[i]);
                max[i] = max[i].max(token.position[i]);
            }
        }

        (min, max)
    }
}

/// Convert HSV to RGB
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (f32, f32, f32) {
    let c = v * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = v - c;

    let (r, g, b) = match (h / 60.0) as i32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };

    (r + m, g + m, b + m)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token3d_creation() {
        let token = Token3D::new("hello".to_string(), 100);
        assert_eq!(token.text, "hello");
        assert_eq!(token.frequency, 100);
        assert_eq!(token.position, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_token3d_distance() {
        let mut t1 = Token3D::new("a".to_string(), 1);
        let mut t2 = Token3D::new("b".to_string(), 1);
        t1.position = [0.0, 0.0, 0.0];
        t2.position = [3.0, 4.0, 0.0];
        assert!((t1.distance_to(&t2) - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_cloud_from_vocabulary() {
        let vocab: Vec<String> = vec!["a".into(), "b".into(), "c".into()];
        let freqs = vec![100, 50, 25];
        let cloud = TokenCloud3D::from_vocabulary(&vocab, &freqs);

        assert_eq!(cloud.token_count(), 3);
        assert_eq!(cloud.tokens[0].text, "a");
        assert_eq!(cloud.tokens[0].size, 1.0); // Highest frequency = size 1.0
    }

    #[test]
    fn test_bounding_box() {
        let vocab: Vec<String> = vec!["a".into(), "b".into()];
        let freqs = vec![1, 1];
        let mut cloud = TokenCloud3D::from_vocabulary(&vocab, &freqs);

        cloud.tokens[0].position = [-1.0, -2.0, -3.0];
        cloud.tokens[1].position = [4.0, 5.0, 6.0];

        let (min, max) = cloud.bounding_box();
        assert_eq!(min, [-1.0, -2.0, -3.0]);
        assert_eq!(max, [4.0, 5.0, 6.0]);
    }
}
