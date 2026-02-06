//! Render data generation for 3D visualization

use super::cluster::ConceptCluster;
use super::token::Token3D;
use std::collections::HashMap;

/// Configuration for rendering
#[derive(Debug, Clone)]
pub struct RenderConfig {
    /// Minimum font size for tokens
    pub min_font_size: f32,
    /// Maximum font size for tokens
    pub max_font_size: f32,
    /// Opacity for less important tokens
    pub min_opacity: f32,
    /// Whether to show cluster boundaries
    pub show_clusters: bool,
    /// Whether to show relationship lines
    pub show_relationships: bool,
    /// Minimum relationship strength to display
    pub relationship_threshold: f32,
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self {
            min_font_size: 8.0,
            max_font_size: 48.0,
            min_opacity: 0.3,
            show_clusters: true,
            show_relationships: true,
            relationship_threshold: 0.5,
        }
    }
}

/// Render-ready token data
#[derive(Debug, Clone)]
pub struct RenderToken {
    /// Unique identifier
    pub id: usize,
    /// The text to display
    pub text: String,
    /// 3D position [x, y, z]
    pub position: [f32; 3],
    /// Size factor (0.0-1.0)
    pub size: f32,
    /// RGBA color
    pub color: [f32; 4],
    /// Importance score (0.0-1.0)
    pub importance: f32,
    /// Cluster assignment
    pub cluster_id: Option<usize>,
}

impl RenderToken {
    /// Get screen-space size based on render config
    pub fn font_size(&self, config: &RenderConfig) -> f32 {
        config.min_font_size + self.size * (config.max_font_size - config.min_font_size)
    }

    /// Get adjusted opacity based on importance
    pub fn opacity(&self, config: &RenderConfig) -> f32 {
        config.min_opacity + self.importance * (1.0 - config.min_opacity)
    }

    /// Get the color with adjusted opacity
    pub fn color_with_opacity(&self, config: &RenderConfig) -> [f32; 4] {
        let mut color = self.color;
        color[3] *= self.opacity(config);
        color
    }
}

/// A relationship edge between two tokens
#[derive(Debug, Clone)]
pub struct TokenRelation {
    /// Source token index
    pub source: usize,
    /// Target token index
    pub target: usize,
    /// Relationship strength (0.0-1.0)
    pub strength: f32,
    /// Optional label for the relationship
    pub label: Option<String>,
    /// Line color RGBA
    pub color: [f32; 4],
}

/// 3D graph showing relationships between tokens
#[derive(Debug, Clone)]
pub struct WordRelationGraph {
    /// All tokens (nodes)
    pub tokens: Vec<RenderToken>,
    /// Relationships (edges)
    pub relations: Vec<TokenRelation>,
    /// Cluster information
    pub clusters: Vec<RenderCluster>,
    /// Render configuration
    pub config: RenderConfig,
}

/// Cluster data for rendering
#[derive(Debug, Clone)]
pub struct RenderCluster {
    /// Cluster ID
    pub id: usize,
    /// Center position
    pub center: [f32; 3],
    /// Bounding sphere radius
    pub radius: f32,
    /// Cluster label
    pub label: String,
    /// Cluster color (derived from member tokens)
    pub color: [f32; 4],
    /// Number of tokens in this cluster
    pub token_count: usize,
}

impl WordRelationGraph {
    /// Create a new word relation graph from tokens
    pub fn new(tokens: Vec<RenderToken>, config: RenderConfig) -> Self {
        Self {
            tokens,
            relations: Vec::new(),
            clusters: Vec::new(),
            config,
        }
    }

    /// Create from Token3D and ConceptCluster data
    pub fn from_cloud(
        tokens: &[Token3D],
        clusters: &[ConceptCluster],
        config: RenderConfig,
    ) -> Self {
        let render_tokens: Vec<RenderToken> = tokens
            .iter()
            .enumerate()
            .map(|(id, t)| RenderToken {
                id,
                text: t.text.clone(),
                position: t.position,
                size: t.size,
                color: t.color,
                importance: t.importance,
                cluster_id: t.cluster_id,
            })
            .collect();

        let render_clusters: Vec<RenderCluster> = clusters
            .iter()
            .enumerate()
            .map(|(id, c)| {
                // Average color from member tokens
                let avg_color = if c.tokens.is_empty() {
                    [0.5, 0.5, 0.5, 0.5]
                } else {
                    let mut sum = [0.0f32; 4];
                    for &idx in &c.tokens {
                        for i in 0..4 {
                            sum[i] += tokens[idx].color[i];
                        }
                    }
                    let n = c.tokens.len() as f32;
                    [sum[0] / n, sum[1] / n, sum[2] / n, sum[3] / n]
                };

                RenderCluster {
                    id,
                    center: c.center,
                    radius: c.radius,
                    label: c.label.clone(),
                    color: avg_color,
                    token_count: c.tokens.len(),
                }
            })
            .collect();

        Self {
            tokens: render_tokens,
            relations: Vec::new(),
            clusters: render_clusters,
            config,
        }
    }

    /// Add relationships based on embedding similarity
    pub fn add_similarity_relations(&mut self, embeddings: &[Vec<f32>], threshold: f32) {
        for i in 0..self.tokens.len() {
            for j in (i + 1)..self.tokens.len() {
                if i < embeddings.len() && j < embeddings.len() {
                    let similarity = cosine_similarity(&embeddings[i], &embeddings[j]);

                    if similarity >= threshold {
                        // Interpolate color between the two tokens
                        let color = [
                            (self.tokens[i].color[0] + self.tokens[j].color[0]) / 2.0,
                            (self.tokens[i].color[1] + self.tokens[j].color[1]) / 2.0,
                            (self.tokens[i].color[2] + self.tokens[j].color[2]) / 2.0,
                            similarity * 0.5, // Opacity based on strength
                        ];

                        self.relations.push(TokenRelation {
                            source: i,
                            target: j,
                            strength: similarity,
                            label: None,
                            color,
                        });
                    }
                }
            }
        }
    }

    /// Add explicit relationships from a map
    pub fn add_relations(&mut self, relations: &[(usize, usize, f32)]) {
        for &(source, target, strength) in relations {
            if source < self.tokens.len() && target < self.tokens.len() {
                let color = [
                    (self.tokens[source].color[0] + self.tokens[target].color[0]) / 2.0,
                    (self.tokens[source].color[1] + self.tokens[target].color[1]) / 2.0,
                    (self.tokens[source].color[2] + self.tokens[target].color[2]) / 2.0,
                    strength * 0.7,
                ];

                self.relations.push(TokenRelation {
                    source,
                    target,
                    strength,
                    label: None,
                    color,
                });
            }
        }
    }

    /// Get all relationships for a specific token
    pub fn relations_for(&self, token_id: usize) -> Vec<&TokenRelation> {
        self.relations
            .iter()
            .filter(|r| r.source == token_id || r.target == token_id)
            .collect()
    }

    /// Get the most connected tokens
    pub fn most_connected(&self, n: usize) -> Vec<(usize, usize)> {
        let mut connection_counts: HashMap<usize, usize> = HashMap::new();

        for relation in &self.relations {
            *connection_counts.entry(relation.source).or_insert(0) += 1;
            *connection_counts.entry(relation.target).or_insert(0) += 1;
        }

        let mut counts: Vec<_> = connection_counts.into_iter().collect();
        counts.sort_by(|a, b| b.1.cmp(&a.1));
        counts.truncate(n);
        counts
    }

    /// Filter relations by minimum strength
    pub fn filter_relations(&mut self, min_strength: f32) {
        self.relations
            .retain(|r| r.strength >= min_strength);
    }

    /// Get render data for all visible elements
    pub fn to_scene_data(&self) -> SceneData<'_> {
        let visible_tokens: Vec<&RenderToken> = self
            .tokens
            .iter()
            .filter(|t| t.opacity(&self.config) > 0.1)
            .collect();

        let visible_relations: Vec<&TokenRelation> = if self.config.show_relationships {
            self.relations
                .iter()
                .filter(|r| r.strength >= self.config.relationship_threshold)
                .collect()
        } else {
            Vec::new()
        };

        let visible_clusters: Vec<&RenderCluster> = if self.config.show_clusters {
            self.clusters.iter().collect()
        } else {
            Vec::new()
        };

        SceneData {
            tokens: visible_tokens,
            relations: visible_relations,
            clusters: visible_clusters,
        }
    }

    /// Calculate the center of mass
    pub fn center_of_mass(&self) -> [f32; 3] {
        if self.tokens.is_empty() {
            return [0.0; 3];
        }

        let mut sum = [0.0f32; 3];
        let mut total_weight = 0.0f32;

        for token in &self.tokens {
            let weight = token.importance;
            sum[0] += token.position[0] * weight;
            sum[1] += token.position[1] * weight;
            sum[2] += token.position[2] * weight;
            total_weight += weight;
        }

        if total_weight > 0.0 {
            [
                sum[0] / total_weight,
                sum[1] / total_weight,
                sum[2] / total_weight,
            ]
        } else {
            [0.0; 3]
        }
    }

    /// Get bounding box
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

/// Scene data ready for rendering
#[derive(Debug)]
pub struct SceneData<'a> {
    /// Visible tokens
    pub tokens: Vec<&'a RenderToken>,
    /// Visible relationships
    pub relations: Vec<&'a TokenRelation>,
    /// Visible clusters
    pub clusters: Vec<&'a RenderCluster>,
}

impl<'a> SceneData<'a> {
    /// Count total elements
    pub fn element_count(&self) -> usize {
        self.tokens.len() + self.relations.len() + self.clusters.len()
    }

    /// Check if scene is empty
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty() && self.relations.is_empty() && self.clusters.is_empty()
    }
}

/// Export format for 3D visualization
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CloudExport {
    /// Token data
    pub tokens: Vec<TokenExport>,
    /// Relationship data
    pub relations: Vec<RelationExport>,
    /// Cluster data
    pub clusters: Vec<ClusterExport>,
}

/// Serializable token for export
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TokenExport {
    pub id: usize,
    pub text: String,
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub size: f32,
    pub color: String, // Hex color
    pub importance: f32,
    pub cluster_id: Option<usize>,
}

/// Serializable relation for export
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RelationExport {
    pub source: usize,
    pub target: usize,
    pub strength: f32,
    pub label: Option<String>,
}

/// Serializable cluster for export
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ClusterExport {
    pub id: usize,
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub radius: f32,
    pub label: String,
    pub color: String,
}

impl WordRelationGraph {
    /// Export to serializable format
    pub fn export(&self) -> CloudExport {
        let tokens: Vec<TokenExport> = self
            .tokens
            .iter()
            .map(|t| TokenExport {
                id: t.id,
                text: t.text.clone(),
                x: t.position[0],
                y: t.position[1],
                z: t.position[2],
                size: t.size,
                color: rgba_to_hex(&t.color),
                importance: t.importance,
                cluster_id: t.cluster_id,
            })
            .collect();

        let relations: Vec<RelationExport> = self
            .relations
            .iter()
            .map(|r| RelationExport {
                source: r.source,
                target: r.target,
                strength: r.strength,
                label: r.label.clone(),
            })
            .collect();

        let clusters: Vec<ClusterExport> = self
            .clusters
            .iter()
            .map(|c| ClusterExport {
                id: c.id,
                x: c.center[0],
                y: c.center[1],
                z: c.center[2],
                radius: c.radius,
                label: c.label.clone(),
                color: rgba_to_hex(&c.color),
            })
            .collect();

        CloudExport {
            tokens,
            relations,
            clusters,
        }
    }

    /// Export to JSON string
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(&self.export())
    }
}

/// Convert RGBA float array to hex color string
fn rgba_to_hex(rgba: &[f32; 4]) -> String {
    let r = (rgba[0] * 255.0).clamp(0.0, 255.0) as u8;
    let g = (rgba[1] * 255.0).clamp(0.0, 255.0) as u8;
    let b = (rgba[2] * 255.0).clamp(0.0, 255.0) as u8;
    let a = (rgba[3] * 255.0).clamp(0.0, 255.0) as u8;
    format!("#{:02x}{:02x}{:02x}{:02x}", r, g, b, a)
}

/// Calculate cosine similarity
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_render_token() {
        let token = RenderToken {
            id: 0,
            text: "test".to_string(),
            position: [0.0, 0.0, 0.0],
            size: 0.5,
            color: [1.0, 0.0, 0.0, 1.0],
            importance: 0.8,
            cluster_id: None,
        };

        let config = RenderConfig::default();
        let font_size = token.font_size(&config);
        assert!(font_size > config.min_font_size);
        assert!(font_size < config.max_font_size);
    }

    #[test]
    fn test_word_relation_graph() {
        let tokens = vec![
            RenderToken {
                id: 0,
                text: "a".to_string(),
                position: [0.0, 0.0, 0.0],
                size: 1.0,
                color: [1.0, 0.0, 0.0, 1.0],
                importance: 1.0,
                cluster_id: Some(0),
            },
            RenderToken {
                id: 1,
                text: "b".to_string(),
                position: [10.0, 0.0, 0.0],
                size: 0.5,
                color: [0.0, 1.0, 0.0, 1.0],
                importance: 0.5,
                cluster_id: Some(0),
            },
        ];

        let mut graph = WordRelationGraph::new(tokens, RenderConfig::default());
        graph.add_relations(&[(0, 1, 0.8)]);

        assert_eq!(graph.tokens.len(), 2);
        assert_eq!(graph.relations.len(), 1);
    }

    #[test]
    fn test_rgba_to_hex() {
        assert_eq!(rgba_to_hex(&[1.0, 0.0, 0.0, 1.0]), "#ff0000ff");
        assert_eq!(rgba_to_hex(&[0.0, 1.0, 0.0, 0.5]), "#00ff007f");
    }

    #[test]
    fn test_export() {
        let tokens = vec![RenderToken {
            id: 0,
            text: "test".to_string(),
            position: [1.0, 2.0, 3.0],
            size: 0.5,
            color: [1.0, 0.0, 0.0, 1.0],
            importance: 0.8,
            cluster_id: None,
        }];

        let graph = WordRelationGraph::new(tokens, RenderConfig::default());
        let export = graph.export();

        assert_eq!(export.tokens.len(), 1);
        assert_eq!(export.tokens[0].text, "test");
        assert_eq!(export.tokens[0].x, 1.0);
    }

    #[test]
    fn test_center_of_mass() {
        let tokens = vec![
            RenderToken {
                id: 0,
                text: "a".to_string(),
                position: [0.0, 0.0, 0.0],
                size: 1.0,
                color: [1.0, 1.0, 1.0, 1.0],
                importance: 1.0,
                cluster_id: None,
            },
            RenderToken {
                id: 1,
                text: "b".to_string(),
                position: [10.0, 10.0, 10.0],
                size: 1.0,
                color: [1.0, 1.0, 1.0, 1.0],
                importance: 1.0,
                cluster_id: None,
            },
        ];

        let graph = WordRelationGraph::new(tokens, RenderConfig::default());
        let center = graph.center_of_mass();

        assert!((center[0] - 5.0).abs() < 0.01);
        assert!((center[1] - 5.0).abs() < 0.01);
        assert!((center[2] - 5.0).abs() < 0.01);
    }
}
