//! 3D Flow Rendering Module
//!
//! Generates 3D representations of information flow through the model.
//! Produces data suitable for visualization in 3D rendering engines.

use super::{InferenceViz, InferenceVizError, LayerType, Result};
use serde::{Deserialize, Serialize};

/// A node in the 3D flow visualization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlowNode3D {
    /// Position in 3D space [x, y, z]
    pub position: [f32; 3],
    /// RGBA color [r, g, b, a]
    pub color: [f32; 4],
    /// Node size (radius)
    pub size: f32,
    /// Indices of connected nodes
    pub connections: Vec<usize>,
    /// Connection weights (strength of each connection)
    pub connection_weights: Vec<f32>,
    /// Label for the node
    pub label: String,
    /// Node type for rendering hints
    pub node_type: FlowNodeType,
    /// Metadata for additional information
    pub metadata: NodeMetadata,
}

/// Type of node in the flow visualization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FlowNodeType {
    /// Input token
    Token,
    /// Layer processing node
    Layer,
    /// Attention head
    AttentionHead,
    /// Output prediction
    Output,
    /// Intermediate representation
    Hidden,
}

/// Additional metadata for a node.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NodeMetadata {
    /// Layer index (if applicable)
    pub layer_idx: Option<usize>,
    /// Head index (if applicable)
    pub head_idx: Option<usize>,
    /// Token position (if applicable)
    pub token_position: Option<usize>,
    /// Activation magnitude
    pub activation_magnitude: Option<f32>,
    /// Attention score
    pub attention_score: Option<f32>,
    /// Custom properties
    pub custom: std::collections::HashMap<String, String>,
}

/// Configuration for 3D flow rendering.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RenderConfig {
    /// Scale factor for the visualization
    pub scale: f32,
    /// Spacing between layers (z-axis)
    pub layer_spacing: f32,
    /// Spacing between tokens (x-axis)
    pub token_spacing: f32,
    /// Minimum node size
    pub min_node_size: f32,
    /// Maximum node size
    pub max_node_size: f32,
    /// Minimum connection weight to show
    pub min_connection_weight: f32,
    /// Maximum connections per node
    pub max_connections_per_node: usize,
    /// Color scheme for nodes
    pub color_scheme: ColorScheme,
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self {
            scale: 1.0,
            layer_spacing: 2.0,
            token_spacing: 1.5,
            min_node_size: 0.1,
            max_node_size: 0.5,
            min_connection_weight: 0.1,
            max_connections_per_node: 10,
            color_scheme: ColorScheme::default(),
        }
    }
}

/// Color scheme for the visualization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColorScheme {
    /// Color for token nodes
    pub token_color: [f32; 4],
    /// Color for layer nodes
    pub layer_color: [f32; 4],
    /// Color for attention nodes
    pub attention_color: [f32; 4],
    /// Color for output nodes
    pub output_color: [f32; 4],
    /// Color for hidden nodes
    pub hidden_color: [f32; 4],
    /// High activation color (for gradient)
    pub high_activation: [f32; 4],
    /// Low activation color (for gradient)
    pub low_activation: [f32; 4],
}

impl Default for ColorScheme {
    fn default() -> Self {
        Self {
            token_color: [0.2, 0.6, 0.9, 1.0],     // Blue
            layer_color: [0.3, 0.8, 0.3, 1.0],     // Green
            attention_color: [0.9, 0.5, 0.2, 1.0], // Orange
            output_color: [0.8, 0.2, 0.2, 1.0],    // Red
            hidden_color: [0.5, 0.5, 0.5, 0.8],    // Gray
            high_activation: [1.0, 0.8, 0.0, 1.0], // Yellow
            low_activation: [0.2, 0.2, 0.4, 0.6],  // Dark blue
        }
    }
}

/// 3D flow renderer.
pub struct FlowRenderer3D {
    /// Configuration
    pub config: RenderConfig,
}

impl Default for FlowRenderer3D {
    fn default() -> Self {
        Self::new()
    }
}

impl FlowRenderer3D {
    /// Create a new renderer with default configuration.
    pub fn new() -> Self {
        Self {
            config: RenderConfig::default(),
        }
    }

    /// Create a renderer with custom configuration.
    pub fn with_config(config: RenderConfig) -> Self {
        Self { config }
    }

    /// Generate 3D flow nodes from inference visualization data.
    pub fn generate(&self, viz: &InferenceViz) -> Result<Vec<FlowNode3D>> {
        generate_3d_flow_with_config(viz, &self.config)
    }

    /// Generate nodes for a single layer.
    pub fn generate_layer_nodes(
        &self,
        viz: &InferenceViz,
        layer_idx: usize,
    ) -> Result<Vec<FlowNode3D>> {
        if layer_idx >= viz.model_config.num_layers {
            return Err(InferenceVizError::InvalidLayerIndex {
                index: layer_idx,
                num_layers: viz.model_config.num_layers,
            });
        }

        let mut nodes = Vec::new();
        let z = layer_idx as f32 * self.config.layer_spacing * self.config.scale;

        // Layer processing node
        let layer_activation = viz
            .layer_activations
            .iter()
            .find(|a| a.layer_idx == layer_idx);

        let activation_magnitude = layer_activation
            .map(|a| a.output_stats.mean.abs())
            .unwrap_or(0.5);

        let layer_type = layer_activation
            .map(|a| a.layer_type)
            .unwrap_or(LayerType::Attention);

        let color = self.layer_type_color(layer_type, activation_magnitude);
        let size = self.activation_to_size(activation_magnitude);

        nodes.push(FlowNode3D {
            position: [0.0, 0.0, z],
            color,
            size,
            connections: Vec::new(),
            connection_weights: Vec::new(),
            label: format!("Layer {} ({:?})", layer_idx, layer_type),
            node_type: FlowNodeType::Layer,
            metadata: NodeMetadata {
                layer_idx: Some(layer_idx),
                activation_magnitude: Some(activation_magnitude),
                ..Default::default()
            },
        });

        // Attention head nodes (if available)
        let attention_patterns: Vec<_> = viz
            .attention_patterns
            .iter()
            .filter(|p| p.layer_idx == layer_idx)
            .collect();

        for (head_offset, pattern) in attention_patterns.iter().enumerate() {
            let x = (head_offset as f32 - attention_patterns.len() as f32 / 2.0)
                * self.config.token_spacing
                * 0.5
                * self.config.scale;
            let y = 0.5 * self.config.scale;

            // Compute average attention for coloring
            let avg_attention: f32 = pattern
                .weights
                .iter()
                .flat_map(|row| row.iter())
                .sum::<f32>()
                / (pattern.seq_len * pattern.seq_len) as f32;

            let color = self.blend_colors(
                &self.config.color_scheme.low_activation,
                &self.config.color_scheme.attention_color,
                avg_attention.min(1.0),
            );

            nodes.push(FlowNode3D {
                position: [x, y, z],
                color,
                size: self.config.min_node_size * self.config.scale,
                connections: Vec::new(),
                connection_weights: Vec::new(),
                label: format!("Head {}", pattern.head_idx),
                node_type: FlowNodeType::AttentionHead,
                metadata: NodeMetadata {
                    layer_idx: Some(layer_idx),
                    head_idx: Some(pattern.head_idx),
                    attention_score: Some(avg_attention),
                    ..Default::default()
                },
            });
        }

        Ok(nodes)
    }

    /// Get color for a layer type.
    fn layer_type_color(&self, layer_type: LayerType, activation: f32) -> [f32; 4] {
        let base_color = match layer_type {
            LayerType::Embedding => self.config.color_scheme.token_color,
            LayerType::Attention => self.config.color_scheme.attention_color,
            LayerType::LayerNorm => self.config.color_scheme.hidden_color,
            LayerType::FeedForward => self.config.color_scheme.layer_color,
            LayerType::Residual => self.config.color_scheme.hidden_color,
            LayerType::Output => self.config.color_scheme.output_color,
        };

        self.blend_colors(
            &self.config.color_scheme.low_activation,
            &base_color,
            activation.min(1.0),
        )
    }

    /// Convert activation magnitude to node size.
    fn activation_to_size(&self, activation: f32) -> f32 {
        let normalized = activation.abs().min(1.0);
        let range = self.config.max_node_size - self.config.min_node_size;
        (self.config.min_node_size + normalized * range) * self.config.scale
    }

    /// Blend two colors.
    fn blend_colors(&self, a: &[f32; 4], b: &[f32; 4], t: f32) -> [f32; 4] {
        let t = t.clamp(0.0, 1.0);
        [
            a[0] + (b[0] - a[0]) * t,
            a[1] + (b[1] - a[1]) * t,
            a[2] + (b[2] - a[2]) * t,
            a[3] + (b[3] - a[3]) * t,
        ]
    }
}

/// Generate 3D flow visualization with default configuration.
pub fn generate_3d_flow(viz: &InferenceViz) -> Result<Vec<FlowNode3D>> {
    generate_3d_flow_with_config(viz, &RenderConfig::default())
}

/// Generate 3D flow visualization with custom configuration.
pub fn generate_3d_flow_with_config(
    viz: &InferenceViz,
    config: &RenderConfig,
) -> Result<Vec<FlowNode3D>> {
    let mut nodes = Vec::new();
    let mut node_index = 0;

    let seq_len = viz.tokens.len().max(1);
    let num_layers = viz.model_config.num_layers;

    // Create token input nodes (bottom layer, z = 0)
    let token_node_start = node_index;
    for (token_idx, token) in viz.tokens.iter().enumerate() {
        let x = (token_idx as f32 - seq_len as f32 / 2.0) * config.token_spacing * config.scale;
        let y = 0.0;
        let z = 0.0;

        nodes.push(FlowNode3D {
            position: [x, y, z],
            color: config.color_scheme.token_color,
            size: config.min_node_size * config.scale,
            connections: Vec::new(),
            connection_weights: Vec::new(),
            label: token.clone(),
            node_type: FlowNodeType::Token,
            metadata: NodeMetadata {
                token_position: Some(token_idx),
                ..Default::default()
            },
        });
        node_index += 1;
    }

    // Create layer nodes
    let mut prev_layer_start = token_node_start;
    let mut prev_layer_count = seq_len;

    for layer_idx in 0..num_layers {
        let layer_node_start = node_index;
        let z = (layer_idx + 1) as f32 * config.layer_spacing * config.scale;

        // Find activation data for this layer
        let layer_activation = viz
            .layer_activations
            .iter()
            .find(|a| a.layer_idx == layer_idx);

        let activation_magnitude = layer_activation
            .map(|a| a.output_stats.mean.abs())
            .unwrap_or(0.5);

        let layer_type = layer_activation
            .map(|a| a.layer_type)
            .unwrap_or(if layer_idx % 2 == 0 {
                LayerType::Attention
            } else {
                LayerType::FeedForward
            });

        // Compute color based on layer type and activation
        let base_color = match layer_type {
            LayerType::Attention => config.color_scheme.attention_color,
            LayerType::FeedForward => config.color_scheme.layer_color,
            _ => config.color_scheme.hidden_color,
        };

        // One node per token position in this layer
        for token_idx in 0..seq_len {
            let x = (token_idx as f32 - seq_len as f32 / 2.0) * config.token_spacing * config.scale;
            let y = 0.0;

            // Blend color based on activation
            let color = blend_colors(
                &config.color_scheme.low_activation,
                &base_color,
                activation_magnitude.min(1.0),
            );

            let size = config.min_node_size
                + (config.max_node_size - config.min_node_size)
                    * activation_magnitude.min(1.0)
                    * config.scale;

            // Connect to previous layer
            let mut connections = Vec::new();
            let mut connection_weights = Vec::new();

            // Direct connection from same position in previous layer
            if prev_layer_count > token_idx {
                connections.push(prev_layer_start + token_idx);
                connection_weights.push(1.0);
            }

            // Attention connections (if available)
            if let Some(patterns) = viz
                .attention_patterns
                .iter()
                .filter(|p| p.layer_idx == layer_idx)
                .next()
            {
                if let Some(row) = patterns.weights.get(token_idx) {
                    for (src_idx, &weight) in row.iter().enumerate() {
                        if weight >= config.min_connection_weight
                            && src_idx != token_idx
                            && connections.len() < config.max_connections_per_node
                        {
                            if prev_layer_count > src_idx {
                                connections.push(prev_layer_start + src_idx);
                                connection_weights.push(weight);
                            }
                        }
                    }
                }
            }

            nodes.push(FlowNode3D {
                position: [x, y, z],
                color,
                size,
                connections,
                connection_weights,
                label: format!("L{}:{}", layer_idx, token_idx),
                node_type: FlowNodeType::Hidden,
                metadata: NodeMetadata {
                    layer_idx: Some(layer_idx),
                    token_position: Some(token_idx),
                    activation_magnitude: Some(activation_magnitude),
                    ..Default::default()
                },
            });
            node_index += 1;
        }

        prev_layer_start = layer_node_start;
        prev_layer_count = seq_len;
    }

    // Create output nodes (if logits are available)
    if !viz.output_logits.is_empty() {
        let z = (num_layers + 1) as f32 * config.layer_spacing * config.scale;

        // Show top predictions as output nodes
        let probs = super::predictions::softmax(&viz.output_logits);
        let mut indexed: Vec<(usize, f32)> =
            probs.iter().enumerate().map(|(i, &p)| (i, p)).collect();
        indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        let top_k = 5.min(indexed.len());
        for (rank, (token_id, prob)) in indexed.iter().take(top_k).enumerate() {
            let x = (rank as f32 - top_k as f32 / 2.0) * config.token_spacing * config.scale;
            let y = 0.5 * config.scale;

            // Color intensity based on probability
            let color = blend_colors(
                &config.color_scheme.low_activation,
                &config.color_scheme.output_color,
                *prob,
            );

            let size = config.min_node_size
                + (config.max_node_size - config.min_node_size) * prob * config.scale;

            // Connect to last hidden layer
            let mut connections = Vec::new();
            let mut connection_weights = Vec::new();
            if prev_layer_count > 0 {
                // Connect to last token position
                connections.push(prev_layer_start + prev_layer_count - 1);
                connection_weights.push(*prob);
            }

            nodes.push(FlowNode3D {
                position: [x, y, z],
                color,
                size,
                connections,
                connection_weights,
                label: format!("P{}: {:.2}%", token_id, prob * 100.0),
                node_type: FlowNodeType::Output,
                metadata: NodeMetadata {
                    attention_score: Some(*prob),
                    ..Default::default()
                },
            });
        }
    }

    Ok(nodes)
}

/// Blend two colors.
fn blend_colors(a: &[f32; 4], b: &[f32; 4], t: f32) -> [f32; 4] {
    let t = t.clamp(0.0, 1.0);
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
        a[3] + (b[3] - a[3]) * t,
    ]
}

/// Export 3D flow to glTF format header (for use with external tools).
pub fn export_gltf_header(nodes: &[FlowNode3D]) -> String {
    let mut json = String::from("{\n");
    json.push_str("  \"asset\": { \"version\": \"2.0\", \"generator\": \"training-tools\" },\n");
    json.push_str("  \"nodes\": [\n");

    for (i, node) in nodes.iter().enumerate() {
        json.push_str(&format!(
            "    {{ \"name\": \"{}\", \"translation\": [{:.3}, {:.3}, {:.3}] }}{}",
            node.label,
            node.position[0],
            node.position[1],
            node.position[2],
            if i < nodes.len() - 1 { ",\n" } else { "\n" }
        ));
    }

    json.push_str("  ],\n");
    json.push_str(&format!("  \"nodeCount\": {}\n", nodes.len()));
    json.push_str("}\n");

    json
}

/// Export 3D flow to OBJ format (simple mesh representation).
pub fn export_obj(nodes: &[FlowNode3D]) -> String {
    let mut obj = String::from("# Training Flow Visualization\n");
    obj.push_str("# Generated by training-tools\n\n");

    // Vertices
    for (i, node) in nodes.iter().enumerate() {
        obj.push_str(&format!(
            "v {:.4} {:.4} {:.4}\n",
            node.position[0], node.position[1], node.position[2]
        ));
        // Add node name as comment
        obj.push_str(&format!("# v{}: {}\n", i + 1, node.label));
    }

    obj.push('\n');

    // Edges as line elements
    for (i, node) in nodes.iter().enumerate() {
        for &conn in &node.connections {
            if conn < nodes.len() {
                // OBJ indices are 1-based
                obj.push_str(&format!("l {} {}\n", i + 1, conn + 1));
            }
        }
    }

    obj
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference_viz::{LayerActivation, ModelConfig};

    fn create_test_viz() -> InferenceViz {
        let config = ModelConfig::new(2, 64, 2, 100);
        let mut viz = InferenceViz::new(config);

        viz.set_tokens(vec!["hello".to_string(), "world".to_string()], vec![1, 2]);

        viz.layer_activations.push(LayerActivation::new(
            0,
            LayerType::Attention,
            vec![0.1, 0.2],
            vec![0.3, 0.4],
        ));

        viz.output_logits = vec![0.1, 0.5, 0.2, 0.15, 0.05];

        viz
    }

    #[test]
    fn test_generate_3d_flow() {
        let viz = create_test_viz();
        let nodes = generate_3d_flow(&viz).unwrap();

        assert!(!nodes.is_empty());

        // Should have token nodes
        let token_nodes: Vec<_> = nodes
            .iter()
            .filter(|n| n.node_type == FlowNodeType::Token)
            .collect();
        assert_eq!(token_nodes.len(), 2);

        // Should have output nodes
        let output_nodes: Vec<_> = nodes
            .iter()
            .filter(|n| n.node_type == FlowNodeType::Output)
            .collect();
        assert!(!output_nodes.is_empty());
    }

    #[test]
    fn test_flow_renderer() {
        let viz = create_test_viz();
        let renderer = FlowRenderer3D::new();
        let nodes = renderer.generate(&viz).unwrap();

        assert!(!nodes.is_empty());
    }

    #[test]
    fn test_export_gltf_header() {
        let nodes = vec![FlowNode3D {
            position: [1.0, 2.0, 3.0],
            color: [1.0, 0.0, 0.0, 1.0],
            size: 0.5,
            connections: vec![],
            connection_weights: vec![],
            label: "test".to_string(),
            node_type: FlowNodeType::Token,
            metadata: NodeMetadata::default(),
        }];

        let gltf = export_gltf_header(&nodes);
        assert!(gltf.contains("\"version\": \"2.0\""));
        assert!(gltf.contains("\"name\": \"test\""));
    }

    #[test]
    fn test_export_obj() {
        let nodes = vec![
            FlowNode3D {
                position: [0.0, 0.0, 0.0],
                color: [1.0, 0.0, 0.0, 1.0],
                size: 0.5,
                connections: vec![1],
                connection_weights: vec![1.0],
                label: "a".to_string(),
                node_type: FlowNodeType::Token,
                metadata: NodeMetadata::default(),
            },
            FlowNode3D {
                position: [1.0, 0.0, 0.0],
                color: [0.0, 1.0, 0.0, 1.0],
                size: 0.5,
                connections: vec![],
                connection_weights: vec![],
                label: "b".to_string(),
                node_type: FlowNodeType::Token,
                metadata: NodeMetadata::default(),
            },
        ];

        let obj = export_obj(&nodes);
        assert!(obj.contains("v 0.0000 0.0000 0.0000"));
        assert!(obj.contains("l 1 2"));
    }

    #[test]
    fn test_blend_colors() {
        let a = [0.0, 0.0, 0.0, 1.0];
        let b = [1.0, 1.0, 1.0, 1.0];

        let mid = blend_colors(&a, &b, 0.5);
        assert!((mid[0] - 0.5).abs() < 1e-6);
        assert!((mid[1] - 0.5).abs() < 1e-6);
        assert!((mid[2] - 0.5).abs() < 1e-6);
    }
}
