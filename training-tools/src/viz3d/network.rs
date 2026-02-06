//! Neural network architecture visualization
//!
//! Provides 3D visualization of neural network layers, connections, and activations.

use std::collections::HashMap;

use nalgebra::{Point3, Vector3};

use super::colors::{tab10, Color, Colormap, ColormapPreset};
use super::engine::{Mesh3D, ObjectId, Vertex3D, Viz3DEngine};

/// Configuration for network visualization
#[derive(Debug, Clone)]
pub struct NetworkGraphConfig {
    /// Spacing between layers along X axis
    pub layer_spacing: f32,
    /// Spacing between nodes within a layer
    pub node_spacing: f32,
    /// Node sphere radius
    pub node_radius: f32,
    /// Connection line thickness
    pub connection_thickness: f32,
    /// Maximum nodes to show per layer (sample if exceeded)
    pub max_nodes_per_layer: usize,
    /// Show connection weights
    pub show_weights: bool,
    /// Color nodes by activation value
    pub color_by_activation: bool,
    /// Colormap for activation values
    pub activation_colormap: ColormapPreset,
    /// Opacity for connections
    pub connection_opacity: f32,
}

impl Default for NetworkGraphConfig {
    fn default() -> Self {
        Self {
            layer_spacing: 3.0,
            node_spacing: 0.5,
            node_radius: 0.15,
            connection_thickness: 0.02,
            max_nodes_per_layer: 64,
            show_weights: true,
            color_by_activation: true,
            activation_colormap: ColormapPreset::Viridis,
            connection_opacity: 0.3,
        }
    }
}

/// Types of neural network layers
#[derive(Debug, Clone, PartialEq)]
pub enum LayerType {
    /// Input layer
    Input,
    /// Dense/fully connected layer
    Dense,
    /// Convolutional layer
    Conv2D {
        kernel_size: (usize, usize),
        filters: usize,
    },
    /// Attention layer
    Attention { heads: usize },
    /// Layer normalization
    LayerNorm,
    /// Embedding layer
    Embedding { vocab_size: usize },
    /// Output layer
    Output,
    /// Custom layer type
    Custom(String),
}

impl LayerType {
    /// Get display name for the layer type
    pub fn display_name(&self) -> &str {
        match self {
            LayerType::Input => "Input",
            LayerType::Dense => "Dense",
            LayerType::Conv2D { .. } => "Conv2D",
            LayerType::Attention { .. } => "Attention",
            LayerType::LayerNorm => "LayerNorm",
            LayerType::Embedding { .. } => "Embedding",
            LayerType::Output => "Output",
            LayerType::Custom(name) => name,
        }
    }

    /// Get color for this layer type
    pub fn color(&self) -> Color {
        let palette = tab10();
        match self {
            LayerType::Input => palette.get(0),
            LayerType::Dense => palette.get(1),
            LayerType::Conv2D { .. } => palette.get(2),
            LayerType::Attention { .. } => palette.get(3),
            LayerType::LayerNorm => palette.get(4),
            LayerType::Embedding { .. } => palette.get(5),
            LayerType::Output => palette.get(6),
            LayerType::Custom(_) => palette.get(7),
        }
    }
}

/// Represents a layer in the network
#[derive(Debug, Clone)]
pub struct NetworkLayer {
    /// Layer name
    pub name: String,
    /// Layer type
    pub layer_type: LayerType,
    /// Number of units/neurons in this layer
    pub units: usize,
    /// Input shape (if applicable)
    pub input_shape: Option<Vec<usize>>,
    /// Output shape
    pub output_shape: Vec<usize>,
    /// Number of parameters
    pub parameters: usize,
    /// Activation values (one per unit, for visualization)
    pub activations: Option<Vec<f32>>,
    /// Gradient magnitudes (one per unit)
    pub gradients: Option<Vec<f32>>,
}

impl NetworkLayer {
    /// Create a new layer
    pub fn new(name: impl Into<String>, layer_type: LayerType, units: usize) -> Self {
        Self {
            name: name.into(),
            layer_type,
            units,
            input_shape: None,
            output_shape: vec![units],
            parameters: 0,
            activations: None,
            gradients: None,
        }
    }

    /// Set the number of parameters
    pub fn with_parameters(mut self, params: usize) -> Self {
        self.parameters = params;
        self
    }

    /// Set input/output shapes
    pub fn with_shapes(mut self, input: Vec<usize>, output: Vec<usize>) -> Self {
        self.input_shape = Some(input);
        self.output_shape = output;
        self
    }

    /// Set activation values
    pub fn with_activations(mut self, activations: Vec<f32>) -> Self {
        self.activations = Some(activations);
        self
    }
}

/// Represents a connection between layers
#[derive(Debug, Clone)]
pub struct LayerConnection {
    /// Source layer index
    pub from_layer: usize,
    /// Target layer index
    pub to_layer: usize,
    /// Connection weights (sampled for visualization)
    pub weights: Option<Vec<f32>>,
    /// Is this a skip/residual connection
    pub is_residual: bool,
}

/// 3D network graph visualization
#[derive(Debug)]
pub struct NetworkGraph3D {
    /// Graph configuration
    pub config: NetworkGraphConfig,
    /// Network layers
    layers: Vec<NetworkLayer>,
    /// Layer connections
    connections: Vec<LayerConnection>,
    /// Generated mesh IDs for layers
    layer_mesh_ids: HashMap<usize, Vec<ObjectId>>,
    /// Generated mesh IDs for connections
    connection_mesh_ids: Vec<ObjectId>,
}

impl NetworkGraph3D {
    /// Create a new network graph visualization
    pub fn new() -> Self {
        Self::with_config(NetworkGraphConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: NetworkGraphConfig) -> Self {
        Self {
            config,
            layers: Vec::new(),
            connections: Vec::new(),
            layer_mesh_ids: HashMap::new(),
            connection_mesh_ids: Vec::new(),
        }
    }

    /// Add a layer to the network
    pub fn add_layer(&mut self, layer: NetworkLayer) -> usize {
        let idx = self.layers.len();
        self.layers.push(layer);
        idx
    }

    /// Add a connection between layers
    pub fn add_connection(&mut self, from: usize, to: usize) {
        self.connections.push(LayerConnection {
            from_layer: from,
            to_layer: to,
            weights: None,
            is_residual: false,
        });
    }

    /// Add a residual/skip connection
    pub fn add_residual_connection(&mut self, from: usize, to: usize) {
        self.connections.push(LayerConnection {
            from_layer: from,
            to_layer: to,
            weights: None,
            is_residual: true,
        });
    }

    /// Get a layer by index
    pub fn get_layer(&self, idx: usize) -> Option<&NetworkLayer> {
        self.layers.get(idx)
    }

    /// Get a mutable layer by index
    pub fn get_layer_mut(&mut self, idx: usize) -> Option<&mut NetworkLayer> {
        self.layers.get_mut(idx)
    }

    /// Number of layers
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    /// Update activation values for a layer
    pub fn update_activations(&mut self, layer_idx: usize, activations: Vec<f32>) {
        if let Some(layer) = self.layers.get_mut(layer_idx) {
            layer.activations = Some(activations);
        }
    }

    /// Calculate the 3D position for a node
    fn node_position(&self, layer_idx: usize, node_idx: usize, total_nodes: usize) -> Point3<f32> {
        let x = layer_idx as f32 * self.config.layer_spacing;
        let y_offset = (total_nodes as f32 - 1.0) * self.config.node_spacing / 2.0;
        let y = node_idx as f32 * self.config.node_spacing - y_offset;
        Point3::new(x, y, 0.0)
    }

    /// Generate 3D meshes for the network and add to engine
    pub fn build_meshes(&mut self, engine: &mut Viz3DEngine) {
        // Clear existing meshes
        for ids in self.layer_mesh_ids.values() {
            for &id in ids {
                engine.remove_mesh(id);
            }
        }
        for &id in &self.connection_mesh_ids {
            engine.remove_mesh(id);
        }
        self.layer_mesh_ids.clear();
        self.connection_mesh_ids.clear();

        let colormap = self.config.activation_colormap.colormap();

        // Build layer meshes
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let mut mesh_ids = Vec::new();

            // Determine how many nodes to visualize
            let num_nodes = layer.units.min(self.config.max_nodes_per_layer);
            let sample_rate = if layer.units > num_nodes {
                layer.units / num_nodes
            } else {
                1
            };

            for node_idx in 0..num_nodes {
                let actual_node = node_idx * sample_rate;
                let pos = self.node_position(layer_idx, node_idx, num_nodes);

                let mut mesh = Mesh3D::sphere(
                    format!("{}_{}", layer.name, node_idx),
                    pos,
                    self.config.node_radius,
                    12,
                );

                // Color by activation if available
                let color = if self.config.color_by_activation {
                    if let Some(ref activations) = layer.activations {
                        if actual_node < activations.len() {
                            let act = activations[actual_node];
                            colormap.map(act.clamp(0.0, 1.0))
                        } else {
                            layer.layer_type.color()
                        }
                    } else {
                        layer.layer_type.color()
                    }
                } else {
                    layer.layer_type.color()
                };

                mesh.set_color(color);
                mesh_ids.push(engine.add_mesh(mesh));
            }

            self.layer_mesh_ids.insert(layer_idx, mesh_ids);
        }

        // Build connection meshes
        for conn in &self.connections {
            if conn.from_layer >= self.layers.len() || conn.to_layer >= self.layers.len() {
                continue;
            }

            let from_layer = &self.layers[conn.from_layer];
            let to_layer = &self.layers[conn.to_layer];

            let from_nodes = from_layer.units.min(self.config.max_nodes_per_layer);
            let to_nodes = to_layer.units.min(self.config.max_nodes_per_layer);

            // For dense connections, only draw sampled connections to avoid clutter
            let max_connections = 100;
            let from_sample = (from_nodes as f32
                / (max_connections as f32 / to_nodes as f32).max(1.0).sqrt())
            .ceil() as usize;
            let to_sample = (to_nodes as f32
                / (max_connections as f32 / from_nodes as f32).max(1.0).sqrt())
            .ceil() as usize;

            for from_idx in (0..from_nodes).step_by(from_sample.max(1)) {
                let from_pos = self.node_position(conn.from_layer, from_idx, from_nodes);

                for to_idx in (0..to_nodes).step_by(to_sample.max(1)) {
                    let to_pos = self.node_position(conn.to_layer, to_idx, to_nodes);

                    let color = if conn.is_residual {
                        Color::rgba(0.2, 0.8, 0.2, self.config.connection_opacity)
                    } else {
                        Color::rgba(0.5, 0.5, 0.5, self.config.connection_opacity)
                    };

                    let mut mesh = Mesh3D::line(
                        format!(
                            "conn_{}_{}_{}_{}",
                            conn.from_layer, from_idx, conn.to_layer, to_idx
                        ),
                        from_pos,
                        to_pos,
                        color,
                    );
                    mesh.opacity = self.config.connection_opacity;

                    self.connection_mesh_ids.push(engine.add_mesh(mesh));
                }
            }
        }
    }

    /// Create a simple feedforward network visualization
    pub fn feedforward(layer_sizes: &[usize]) -> Self {
        let mut graph = Self::new();

        for (i, &size) in layer_sizes.iter().enumerate() {
            let layer_type = if i == 0 {
                LayerType::Input
            } else if i == layer_sizes.len() - 1 {
                LayerType::Output
            } else {
                LayerType::Dense
            };

            let name = match layer_type {
                LayerType::Input => "Input".to_string(),
                LayerType::Output => "Output".to_string(),
                _ => format!("Hidden_{}", i),
            };

            graph.add_layer(NetworkLayer::new(name, layer_type, size));

            if i > 0 {
                graph.add_connection(i - 1, i);
            }
        }

        graph
    }

    /// Create a transformer-style network visualization
    pub fn transformer(
        vocab_size: usize,
        embed_dim: usize,
        num_layers: usize,
        num_heads: usize,
        ff_dim: usize,
    ) -> Self {
        let mut graph = Self::new();

        // Embedding layer
        let embed_idx = graph.add_layer(
            NetworkLayer::new("Embedding", LayerType::Embedding { vocab_size }, embed_dim)
                .with_parameters(vocab_size * embed_dim),
        );

        let mut prev_idx = embed_idx;

        // Transformer blocks
        for i in 0..num_layers {
            // Self-attention
            let attn_idx = graph.add_layer(
                NetworkLayer::new(
                    format!("Attention_{}", i),
                    LayerType::Attention { heads: num_heads },
                    embed_dim,
                )
                .with_parameters(4 * embed_dim * embed_dim),
            );
            graph.add_connection(prev_idx, attn_idx);
            graph.add_residual_connection(prev_idx, attn_idx);

            // Layer norm
            let ln1_idx = graph.add_layer(
                NetworkLayer::new(
                    format!("LayerNorm_{}_1", i),
                    LayerType::LayerNorm,
                    embed_dim,
                )
                .with_parameters(2 * embed_dim),
            );
            graph.add_connection(attn_idx, ln1_idx);

            // FFN
            let ff_idx = graph.add_layer(
                NetworkLayer::new(format!("FFN_{}", i), LayerType::Dense, ff_dim)
                    .with_parameters(embed_dim * ff_dim + ff_dim * embed_dim),
            );
            graph.add_connection(ln1_idx, ff_idx);
            graph.add_residual_connection(ln1_idx, ff_idx);

            // Layer norm
            let ln2_idx = graph.add_layer(
                NetworkLayer::new(
                    format!("LayerNorm_{}_2", i),
                    LayerType::LayerNorm,
                    embed_dim,
                )
                .with_parameters(2 * embed_dim),
            );
            graph.add_connection(ff_idx, ln2_idx);

            prev_idx = ln2_idx;
        }

        // Output layer
        let output_idx = graph.add_layer(
            NetworkLayer::new("Output", LayerType::Output, vocab_size)
                .with_parameters(embed_dim * vocab_size),
        );
        graph.add_connection(prev_idx, output_idx);

        graph
    }

    /// Get total parameter count
    pub fn total_parameters(&self) -> usize {
        self.layers.iter().map(|l| l.parameters).sum()
    }

    /// Get network summary as string
    pub fn summary(&self) -> String {
        let mut s = String::new();
        s.push_str("Network Architecture:\n");
        s.push_str(&format!(
            "{:<20} {:<15} {:<15} {:<15}\n",
            "Layer", "Type", "Output Shape", "Parameters"
        ));
        s.push_str(&"-".repeat(65));
        s.push('\n');

        for layer in &self.layers {
            let shape_str = format!("{:?}", layer.output_shape);
            s.push_str(&format!(
                "{:<20} {:<15} {:<15} {:<15}\n",
                layer.name,
                layer.layer_type.display_name(),
                shape_str,
                layer.parameters
            ));
        }

        s.push_str(&"-".repeat(65));
        s.push('\n');
        s.push_str(&format!("Total parameters: {}\n", self.total_parameters()));

        s
    }
}

impl Default for NetworkGraph3D {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feedforward_creation() {
        let graph = NetworkGraph3D::feedforward(&[784, 256, 128, 10]);
        assert_eq!(graph.layer_count(), 4);
        assert_eq!(graph.connections.len(), 3);
    }

    #[test]
    fn test_transformer_creation() {
        let graph = NetworkGraph3D::transformer(32000, 512, 6, 8, 2048);
        assert!(graph.layer_count() > 6);
        assert!(graph.total_parameters() > 0);
    }

    #[test]
    fn test_layer_positions() {
        let graph = NetworkGraph3D::new();
        let pos0 = graph.node_position(0, 0, 10);
        let pos1 = graph.node_position(1, 0, 10);

        assert!(pos1.x > pos0.x);
    }
}
