//! Attention pattern visualization
//!
//! Provides 3D visualization of transformer attention patterns,
//! showing how tokens attend to each other across layers and heads.

use nalgebra::{Point3, Vector3};

use super::colors::{tab10, Color, Colormap, ColormapPreset};
use super::engine::{Mesh3D, ObjectId, Vertex3D, Viz3DEngine};

/// Configuration for attention visualization
#[derive(Debug, Clone)]
pub struct AttentionFlowConfig {
    /// Spacing between tokens along X axis
    pub token_spacing: f32,
    /// Spacing between layers along Z axis
    pub layer_spacing: f32,
    /// Spacing between heads along Y axis
    pub head_spacing: f32,
    /// Token node size
    pub token_size: f32,
    /// Minimum attention weight to visualize
    pub attention_threshold: f32,
    /// Maximum line width for attention connections
    pub max_line_width: f32,
    /// Colormap for attention weights
    pub attention_colormap: ColormapPreset,
    /// Color attention by head
    pub color_by_head: bool,
    /// Show token labels
    pub show_labels: bool,
    /// Attention line opacity multiplier
    pub line_opacity: f32,
    /// Aggregate heads (average attention)
    pub aggregate_heads: bool,
}

impl Default for AttentionFlowConfig {
    fn default() -> Self {
        Self {
            token_spacing: 1.0,
            layer_spacing: 3.0,
            head_spacing: 0.5,
            token_size: 0.15,
            attention_threshold: 0.05,
            max_line_width: 0.1,
            attention_colormap: ColormapPreset::Plasma,
            color_by_head: true,
            show_labels: true,
            line_opacity: 0.6,
            aggregate_heads: false,
        }
    }
}

/// A single token in the sequence
#[derive(Debug, Clone)]
pub struct Token {
    /// Token index in sequence
    pub index: usize,
    /// Token text representation
    pub text: String,
    /// Token embedding (optional, for coloring)
    pub embedding_norm: Option<f32>,
    /// Is this a special token (BOS, EOS, PAD)
    pub is_special: bool,
}

impl Token {
    /// Create a new token
    pub fn new(index: usize, text: impl Into<String>) -> Self {
        Self {
            index,
            text: text.into(),
            embedding_norm: None,
            is_special: false,
        }
    }

    /// Mark as special token
    pub fn special(mut self) -> Self {
        self.is_special = true;
        self
    }

    /// Set embedding norm
    pub fn with_embedding_norm(mut self, norm: f32) -> Self {
        self.embedding_norm = Some(norm);
        self
    }
}

/// Attention weights for a single head
#[derive(Debug, Clone)]
pub struct AttentionHead {
    /// Head index
    pub index: usize,
    /// Head name (optional)
    pub name: Option<String>,
    /// Attention matrix [seq_len x seq_len]
    /// Row i contains attention weights from token i to all other tokens
    pub weights: Vec<Vec<f32>>,
}

impl AttentionHead {
    /// Create a new attention head
    pub fn new(index: usize, weights: Vec<Vec<f32>>) -> Self {
        Self {
            index,
            name: None,
            weights,
        }
    }

    /// Set head name
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Get attention weight from token i to token j
    pub fn get_weight(&self, from: usize, to: usize) -> f32 {
        self.weights
            .get(from)
            .and_then(|row| row.get(to))
            .copied()
            .unwrap_or(0.0)
    }

    /// Get sequence length
    pub fn seq_len(&self) -> usize {
        self.weights.len()
    }
}

/// Attention patterns for a single layer
#[derive(Debug, Clone)]
pub struct AttentionLayer {
    /// Layer index
    pub index: usize,
    /// Layer name
    pub name: String,
    /// Attention heads in this layer
    pub heads: Vec<AttentionHead>,
}

impl AttentionLayer {
    /// Create a new attention layer
    pub fn new(index: usize, heads: Vec<AttentionHead>) -> Self {
        Self {
            index,
            name: format!("Layer {}", index),
            heads,
        }
    }

    /// Set layer name
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Number of heads
    pub fn num_heads(&self) -> usize {
        self.heads.len()
    }

    /// Get averaged attention across all heads
    pub fn average_attention(&self) -> Vec<Vec<f32>> {
        if self.heads.is_empty() {
            return Vec::new();
        }

        let seq_len = self.heads[0].seq_len();
        let num_heads = self.heads.len() as f32;

        let mut avg = vec![vec![0.0; seq_len]; seq_len];

        for head in &self.heads {
            for (i, row) in head.weights.iter().enumerate() {
                for (j, &w) in row.iter().enumerate() {
                    avg[i][j] += w / num_heads;
                }
            }
        }

        avg
    }
}

/// 3D attention flow visualization
#[derive(Debug)]
pub struct AttentionFlow3D {
    /// Configuration
    pub config: AttentionFlowConfig,
    /// Input tokens
    tokens: Vec<Token>,
    /// Attention layers
    layers: Vec<AttentionLayer>,
    /// Selected layer index (None = all layers)
    selected_layer: Option<usize>,
    /// Selected head index (None = all heads)
    selected_head: Option<usize>,
    /// Generated mesh IDs for tokens
    token_mesh_ids: Vec<ObjectId>,
    /// Generated mesh IDs for attention lines
    attention_mesh_ids: Vec<ObjectId>,
}

impl AttentionFlow3D {
    /// Create a new attention flow visualization
    pub fn new() -> Self {
        Self::with_config(AttentionFlowConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: AttentionFlowConfig) -> Self {
        Self {
            config,
            tokens: Vec::new(),
            layers: Vec::new(),
            selected_layer: None,
            selected_head: None,
            token_mesh_ids: Vec::new(),
            attention_mesh_ids: Vec::new(),
        }
    }

    /// Set the input tokens
    pub fn set_tokens(&mut self, tokens: Vec<Token>) {
        self.tokens = tokens;
    }

    /// Add tokens from string slice
    pub fn add_tokens_from_strings(&mut self, texts: &[&str]) {
        for (i, &text) in texts.iter().enumerate() {
            self.tokens.push(Token::new(i, text));
        }
    }

    /// Add an attention layer
    pub fn add_layer(&mut self, layer: AttentionLayer) {
        self.layers.push(layer);
    }

    /// Set all layers at once
    pub fn set_layers(&mut self, layers: Vec<AttentionLayer>) {
        self.layers = layers;
    }

    /// Get number of layers
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Get sequence length
    pub fn seq_len(&self) -> usize {
        self.tokens.len()
    }

    /// Select a specific layer to visualize
    pub fn select_layer(&mut self, layer: Option<usize>) {
        self.selected_layer = layer;
    }

    /// Select a specific head to visualize
    pub fn select_head(&mut self, head: Option<usize>) {
        self.selected_head = head;
    }

    /// Calculate 3D position for a token at a given layer
    fn token_position(
        &self,
        token_idx: usize,
        layer_idx: usize,
        head_idx: Option<usize>,
    ) -> Point3<f32> {
        let x = (token_idx as f32 - self.tokens.len() as f32 / 2.0) * self.config.token_spacing;
        let z = layer_idx as f32 * self.config.layer_spacing;
        let y = head_idx
            .map(|h| (h as f32 - 2.0) * self.config.head_spacing)
            .unwrap_or(0.0);

        Point3::new(x, y, z)
    }

    /// Build 3D meshes and add to engine
    pub fn build_meshes(&mut self, engine: &mut Viz3DEngine) {
        // Clear existing meshes
        for &id in &self.token_mesh_ids {
            engine.remove_mesh(id);
        }
        for &id in &self.attention_mesh_ids {
            engine.remove_mesh(id);
        }
        self.token_mesh_ids.clear();
        self.attention_mesh_ids.clear();

        if self.tokens.is_empty() || self.layers.is_empty() {
            return;
        }

        let palette = tab10();
        let colormap = self.config.attention_colormap.colormap();

        // Determine which layers to visualize
        let layer_range: Vec<usize> = match self.selected_layer {
            Some(idx) => vec![idx],
            None => (0..self.layers.len()).collect(),
        };

        // Build token meshes at each layer
        for &layer_idx in &layer_range {
            for (token_idx, token) in self.tokens.iter().enumerate() {
                let pos = self.token_position(token_idx, layer_idx, None);

                let color = if token.is_special {
                    Color::rgba(0.5, 0.5, 0.5, 0.8)
                } else if let Some(norm) = token.embedding_norm {
                    colormap.map(norm.clamp(0.0, 1.0))
                } else {
                    Color::WHITE
                };

                let mut mesh = Mesh3D::sphere(
                    format!("token_{}_{}", layer_idx, token_idx),
                    pos,
                    self.config.token_size,
                    12,
                );
                mesh.set_color(color);
                self.token_mesh_ids.push(engine.add_mesh(mesh));
            }
        }

        // Build attention connections
        for &layer_idx in &layer_range {
            if layer_idx >= self.layers.len() {
                continue;
            }

            // Determine which heads to visualize
            let num_heads = self.layers[layer_idx].num_heads();
            let head_range: Vec<usize> = match self.selected_head {
                Some(idx) => vec![idx],
                None => (0..num_heads).collect(),
            };

            if self.config.aggregate_heads {
                // Visualize averaged attention
                let avg_weights = self.layers[layer_idx].average_attention();
                self.build_attention_lines(engine, layer_idx, &avg_weights, None, &colormap);
            } else {
                // Visualize each head
                for &head_idx in &head_range {
                    if head_idx >= num_heads {
                        continue;
                    }

                    let weights = self.layers[layer_idx].heads[head_idx].weights.clone();
                    let color = if self.config.color_by_head {
                        Some(palette.get(head_idx))
                    } else {
                        None
                    };

                    self.build_attention_lines(engine, layer_idx, &weights, color, &colormap);
                }
            }
        }
    }

    /// Build attention lines for a single attention matrix
    fn build_attention_lines(
        &mut self,
        engine: &mut Viz3DEngine,
        layer_idx: usize,
        weights: &[Vec<f32>],
        head_color: Option<Color>,
        colormap: &Colormap,
    ) {
        let seq_len = self.tokens.len();

        for from in 0..seq_len {
            let from_pos = self.token_position(from, layer_idx, None);

            for to in 0..seq_len {
                if from >= weights.len() || to >= weights[from].len() {
                    continue;
                }

                let weight = weights[from][to];
                if weight < self.config.attention_threshold {
                    continue;
                }

                let to_pos = self.token_position(to, layer_idx, None);

                // Offset the line slightly above the tokens
                let offset = Vector3::new(0.0, 0.1 + weight * 0.2, 0.0);
                let from_pos = from_pos + offset;
                let to_pos = to_pos + offset;

                let color = head_color.unwrap_or_else(|| colormap.map(weight));
                let alpha = weight * self.config.line_opacity;

                let mesh = Mesh3D::line(
                    format!("attn_{}_{}_{}", layer_idx, from, to),
                    from_pos,
                    to_pos,
                    Color::rgba(color.r, color.g, color.b, alpha),
                );

                self.attention_mesh_ids.push(engine.add_mesh(mesh));
            }
        }
    }

    /// Create attention connections between consecutive layers
    pub fn build_layer_connections(&mut self, engine: &mut Viz3DEngine) {
        if self.layers.len() < 2 {
            return;
        }

        let seq_len = self.tokens.len();
        let color = Color::rgba(0.3, 0.3, 0.3, 0.2);

        for layer_idx in 0..self.layers.len() - 1 {
            for token_idx in 0..seq_len {
                let from_pos = self.token_position(token_idx, layer_idx, None);
                let to_pos = self.token_position(token_idx, layer_idx + 1, None);

                let mesh = Mesh3D::line(
                    format!("layer_conn_{}_{}", layer_idx, token_idx),
                    from_pos,
                    to_pos,
                    color,
                );
                self.attention_mesh_ids.push(engine.add_mesh(mesh));
            }
        }
    }

    /// Get attention statistics
    pub fn stats(&self) -> AttentionStats {
        let total_heads = self.layers.iter().map(|l| l.num_heads()).sum();

        // Find max attention weight
        let max_attention = self
            .layers
            .iter()
            .flat_map(|l| l.heads.iter())
            .flat_map(|h| h.weights.iter())
            .flat_map(|row| row.iter())
            .cloned()
            .fold(0.0f32, f32::max);

        // Count significant attention connections
        let significant_connections: usize = self
            .layers
            .iter()
            .flat_map(|l| l.heads.iter())
            .flat_map(|h| h.weights.iter())
            .flat_map(|row| row.iter())
            .filter(|&&w| w >= self.config.attention_threshold)
            .count();

        AttentionStats {
            num_layers: self.layers.len(),
            num_heads: total_heads,
            seq_len: self.tokens.len(),
            max_attention,
            significant_connections,
        }
    }

    /// Generate a sample causal attention pattern
    pub fn sample_causal(seq_len: usize, num_layers: usize, num_heads: usize) -> Self {
        let mut flow = Self::new();

        // Create tokens
        let tokens: Vec<Token> = (0..seq_len)
            .map(|i| Token::new(i, format!("tok{}", i)))
            .collect();
        flow.set_tokens(tokens);

        // Create layers with causal attention
        for layer_idx in 0..num_layers {
            let mut heads = Vec::new();

            for head_idx in 0..num_heads {
                // Generate causal attention pattern with some variation
                let mut weights = vec![vec![0.0; seq_len]; seq_len];

                for i in 0..seq_len {
                    let mut row_sum = 0.0;
                    for j in 0..=i {
                        // Causal: can only attend to past tokens
                        let base_weight = if j == i {
                            0.3 // Self-attention
                        } else {
                            0.7 / (i - j + 1) as f32 // Decay with distance
                        };
                        // Add some head-specific variation
                        let variation =
                            ((head_idx * 7 + layer_idx * 13 + i * 3 + j * 5) % 10) as f32 / 20.0;
                        weights[i][j] = (base_weight + variation).max(0.0);
                        row_sum += weights[i][j];
                    }
                    // Normalize (softmax-like)
                    if row_sum > 0.0 {
                        for j in 0..=i {
                            weights[i][j] /= row_sum;
                        }
                    }
                }

                heads.push(AttentionHead::new(head_idx, weights));
            }

            flow.add_layer(AttentionLayer::new(layer_idx, heads));
        }

        flow
    }
}

impl Default for AttentionFlow3D {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about attention patterns
#[derive(Debug, Clone)]
pub struct AttentionStats {
    /// Number of layers
    pub num_layers: usize,
    /// Total number of heads
    pub num_heads: usize,
    /// Sequence length
    pub seq_len: usize,
    /// Maximum attention weight
    pub max_attention: f32,
    /// Number of attention connections above threshold
    pub significant_connections: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attention_creation() {
        let mut flow = AttentionFlow3D::new();
        flow.add_tokens_from_strings(&["Hello", "world", "!"]);

        assert_eq!(flow.seq_len(), 3);
    }

    #[test]
    fn test_causal_sample() {
        let flow = AttentionFlow3D::sample_causal(8, 2, 4);

        assert_eq!(flow.seq_len(), 8);
        assert_eq!(flow.num_layers(), 2);

        // Check causal mask: attention from token 0 should only attend to token 0
        let layer = &flow.layers[0];
        let head = &layer.heads[0];
        assert!(head.get_weight(0, 1) < 0.01); // Token 0 shouldn't attend to token 1
        assert!(head.get_weight(0, 0) > 0.0); // Token 0 should attend to itself
    }

    #[test]
    fn test_average_attention() {
        let weights1 = vec![vec![0.5, 0.5], vec![0.3, 0.7]];
        let weights2 = vec![vec![0.3, 0.7], vec![0.5, 0.5]];

        let layer = AttentionLayer::new(
            0,
            vec![
                AttentionHead::new(0, weights1),
                AttentionHead::new(1, weights2),
            ],
        );

        let avg = layer.average_attention();
        assert!((avg[0][0] - 0.4).abs() < 0.01);
        assert!((avg[0][1] - 0.6).abs() < 0.01);
    }
}
