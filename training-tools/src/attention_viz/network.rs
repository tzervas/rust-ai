//! Network architecture visualization.
//!
//! Provides visualization of transformer model architecture including:
//! - Layer stack representation
//! - Parameter counts per layer
//! - Gradient flow visualization
//! - Activation patterns

use super::{OutputFormat, Renderer, VizError, VizResult};
use serde::{Deserialize, Serialize};

/// Types of layers in a neural network.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LayerType {
    /// Token/position embedding layer.
    Embedding,
    /// Multi-head attention layer.
    Attention,
    /// Feed-forward MLP layer.
    MLP,
    /// Layer normalization.
    LayerNorm,
    /// Final output/projection layer.
    Output,
    /// Dropout layer.
    Dropout,
    /// Residual connection.
    Residual,
    /// Custom/other layer type.
    Custom,
}

impl LayerType {
    /// Get a display name for this layer type.
    pub fn display_name(&self) -> &'static str {
        match self {
            LayerType::Embedding => "Embedding",
            LayerType::Attention => "Attention",
            LayerType::MLP => "MLP",
            LayerType::LayerNorm => "LayerNorm",
            LayerType::Output => "Output",
            LayerType::Dropout => "Dropout",
            LayerType::Residual => "Residual",
            LayerType::Custom => "Custom",
        }
    }

    /// Get an ASCII icon for this layer type.
    pub fn ascii_icon(&self) -> &'static str {
        match self {
            LayerType::Embedding => "[E]",
            LayerType::Attention => "[A]",
            LayerType::MLP => "[M]",
            LayerType::LayerNorm => "[N]",
            LayerType::Output => "[O]",
            LayerType::Dropout => "[D]",
            LayerType::Residual => "[R]",
            LayerType::Custom => "[?]",
        }
    }

    /// Get a color for SVG rendering.
    pub fn svg_color(&self) -> &'static str {
        match self {
            LayerType::Embedding => "#4CAF50", // Green
            LayerType::Attention => "#2196F3", // Blue
            LayerType::MLP => "#FF9800",       // Orange
            LayerType::LayerNorm => "#9C27B0", // Purple
            LayerType::Output => "#F44336",    // Red
            LayerType::Dropout => "#607D8B",   // Gray
            LayerType::Residual => "#00BCD4",  // Cyan
            LayerType::Custom => "#795548",    // Brown
        }
    }
}

/// Visualization data for a single layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerViz {
    /// Name/identifier for this layer.
    pub name: String,
    /// Type of layer.
    pub layer_type: LayerType,
    /// Number of parameters.
    pub params: u64,
    /// Input shape.
    pub input_shape: Vec<usize>,
    /// Output shape.
    pub output_shape: Vec<usize>,
    /// Gradient norm (if available during training).
    pub gradient_norm: Option<f32>,
    /// Mean activation value (if available).
    pub activation_mean: Option<f32>,
    /// Activation standard deviation (if available).
    pub activation_std: Option<f32>,
    /// Whether this layer is frozen (no gradients).
    pub frozen: bool,
    /// Additional metadata.
    pub metadata: Option<serde_json::Value>,
}

impl LayerViz {
    /// Create a new layer visualization.
    pub fn new(name: String, layer_type: LayerType) -> Self {
        Self {
            name,
            layer_type,
            params: 0,
            input_shape: Vec::new(),
            output_shape: Vec::new(),
            gradient_norm: None,
            activation_mean: None,
            activation_std: None,
            frozen: false,
            metadata: None,
        }
    }

    /// Set parameter count.
    pub fn with_params(mut self, params: u64) -> Self {
        self.params = params;
        self
    }

    /// Set input shape.
    pub fn with_input_shape(mut self, shape: Vec<usize>) -> Self {
        self.input_shape = shape;
        self
    }

    /// Set output shape.
    pub fn with_output_shape(mut self, shape: Vec<usize>) -> Self {
        self.output_shape = shape;
        self
    }

    /// Set gradient norm.
    pub fn with_gradient_norm(mut self, norm: f32) -> Self {
        self.gradient_norm = Some(norm);
        self
    }

    /// Set activation statistics.
    pub fn with_activations(mut self, mean: f32, std: f32) -> Self {
        self.activation_mean = Some(mean);
        self.activation_std = Some(std);
        self
    }

    /// Mark as frozen.
    pub fn frozen(mut self) -> Self {
        self.frozen = true;
        self
    }

    /// Format parameter count with units.
    pub fn format_params(&self) -> String {
        format_param_count(self.params)
    }

    /// Format shapes as string.
    pub fn format_shapes(&self) -> String {
        let input = if self.input_shape.is_empty() {
            "?".to_string()
        } else {
            format!("{:?}", self.input_shape)
        };
        let output = if self.output_shape.is_empty() {
            "?".to_string()
        } else {
            format!("{:?}", self.output_shape)
        };
        format!("{} -> {}", input, output)
    }
}

/// Type of connection between layers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConnectionType {
    /// Standard forward connection.
    Forward,
    /// Skip/residual connection.
    Skip,
    /// Attention connection (query to key/value).
    Attention,
}

/// A connection between two layers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Connection {
    /// Source layer index.
    pub from: usize,
    /// Target layer index.
    pub to: usize,
    /// Type of connection.
    pub connection_type: ConnectionType,
    /// Label for this connection.
    pub label: Option<String>,
}

impl Connection {
    /// Create a new forward connection.
    pub fn forward(from: usize, to: usize) -> Self {
        Self {
            from,
            to,
            connection_type: ConnectionType::Forward,
            label: None,
        }
    }

    /// Create a skip connection.
    pub fn skip(from: usize, to: usize) -> Self {
        Self {
            from,
            to,
            connection_type: ConnectionType::Skip,
            label: Some("skip".to_string()),
        }
    }

    /// Create an attention connection.
    pub fn attention(from: usize, to: usize) -> Self {
        Self {
            from,
            to,
            connection_type: ConnectionType::Attention,
            label: Some("attn".to_string()),
        }
    }

    /// Set a label.
    pub fn with_label(mut self, label: String) -> Self {
        self.label = Some(label);
        self
    }
}

/// Configuration for network visualization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkVizConfig {
    /// Title for the visualization.
    pub title: Option<String>,
    /// Whether to show parameter counts.
    pub show_params: bool,
    /// Whether to show shapes.
    pub show_shapes: bool,
    /// Whether to show gradient norms.
    pub show_gradients: bool,
    /// Whether to show activation statistics.
    pub show_activations: bool,
    /// Whether to highlight frozen layers.
    pub highlight_frozen: bool,
}

impl Default for NetworkVizConfig {
    fn default() -> Self {
        Self {
            title: None,
            show_params: true,
            show_shapes: true,
            show_gradients: true,
            show_activations: false,
            highlight_frozen: true,
        }
    }
}

/// Complete network architecture visualization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkArchViz {
    /// Layers in the network.
    pub layers: Vec<LayerViz>,
    /// Connections between layers.
    pub connections: Vec<Connection>,
    /// Configuration.
    pub config: NetworkVizConfig,
}

impl NetworkArchViz {
    /// Create a new network visualization.
    pub fn new(layers: Vec<LayerViz>) -> Self {
        // Auto-generate forward connections
        let connections: Vec<Connection> = (0..layers.len().saturating_sub(1))
            .map(|i| Connection::forward(i, i + 1))
            .collect();

        Self {
            layers,
            connections,
            config: NetworkVizConfig::default(),
        }
    }

    /// Set connections explicitly.
    pub fn with_connections(mut self, connections: Vec<Connection>) -> Self {
        self.connections = connections;
        self
    }

    /// Add a connection.
    pub fn add_connection(&mut self, connection: Connection) {
        self.connections.push(connection);
    }

    /// Set configuration.
    pub fn with_config(mut self, config: NetworkVizConfig) -> Self {
        self.config = config;
        self
    }

    /// Get total parameter count.
    pub fn total_params(&self) -> u64 {
        self.layers.iter().map(|l| l.params).sum()
    }

    /// Get parameter count by layer type.
    pub fn params_by_type(&self) -> std::collections::HashMap<LayerType, u64> {
        let mut counts = std::collections::HashMap::new();
        for layer in &self.layers {
            *counts.entry(layer.layer_type).or_insert(0) += layer.params;
        }
        counts
    }

    /// Get layers with gradient information.
    pub fn layers_with_gradients(&self) -> Vec<&LayerViz> {
        self.layers
            .iter()
            .filter(|l| l.gradient_norm.is_some())
            .collect()
    }

    /// Check for potential gradient issues.
    pub fn gradient_health(&self) -> GradientHealth {
        let grads: Vec<f32> = self.layers.iter().filter_map(|l| l.gradient_norm).collect();

        if grads.is_empty() {
            return GradientHealth::Unknown;
        }

        let max = grads.iter().cloned().fold(0.0f32, f32::max);
        let min = grads.iter().cloned().fold(f32::INFINITY, f32::min);
        let mean = grads.iter().sum::<f32>() / grads.len() as f32;

        // Check for vanishing gradients
        if min < 1e-7 {
            return GradientHealth::Vanishing {
                min_norm: min,
                affected_layers: self
                    .layers
                    .iter()
                    .enumerate()
                    .filter(|(_, l)| l.gradient_norm.map(|g| g < 1e-7).unwrap_or(false))
                    .map(|(i, _)| i)
                    .collect(),
            };
        }

        // Check for exploding gradients
        if max > 100.0 || max / mean > 100.0 {
            return GradientHealth::Exploding {
                max_norm: max,
                affected_layers: self
                    .layers
                    .iter()
                    .enumerate()
                    .filter(|(_, l)| l.gradient_norm.map(|g| g > 100.0).unwrap_or(false))
                    .map(|(i, _)| i)
                    .collect(),
            };
        }

        GradientHealth::Healthy {
            mean_norm: mean,
            max_norm: max,
            min_norm: min,
        }
    }

    /// Create a standard transformer block architecture.
    pub fn transformer_block(
        block_idx: usize,
        hidden_size: usize,
        num_heads: usize,
        mlp_ratio: usize,
    ) -> Vec<LayerViz> {
        let head_dim = hidden_size / num_heads;
        let mlp_hidden = hidden_size * mlp_ratio;

        vec![
            LayerViz::new(format!("block_{}.ln1", block_idx), LayerType::LayerNorm)
                .with_params(hidden_size as u64 * 2)
                .with_input_shape(vec![hidden_size])
                .with_output_shape(vec![hidden_size]),
            LayerViz::new(format!("block_{}.attn", block_idx), LayerType::Attention)
                .with_params(
                    (4 * hidden_size * hidden_size + 4 * hidden_size) as u64, // Q, K, V, O projections
                )
                .with_input_shape(vec![hidden_size])
                .with_output_shape(vec![hidden_size]),
            LayerViz::new(format!("block_{}.ln2", block_idx), LayerType::LayerNorm)
                .with_params(hidden_size as u64 * 2)
                .with_input_shape(vec![hidden_size])
                .with_output_shape(vec![hidden_size]),
            LayerViz::new(format!("block_{}.mlp", block_idx), LayerType::MLP)
                .with_params((2 * hidden_size * mlp_hidden + hidden_size + mlp_hidden) as u64)
                .with_input_shape(vec![hidden_size])
                .with_output_shape(vec![hidden_size]),
        ]
    }

    /// Create a standard GPT-style architecture.
    pub fn gpt_architecture(
        vocab_size: usize,
        hidden_size: usize,
        num_layers: usize,
        num_heads: usize,
        max_seq_len: usize,
    ) -> Self {
        let mut layers = Vec::new();

        // Token embedding
        layers.push(
            LayerViz::new("tok_emb".to_string(), LayerType::Embedding)
                .with_params((vocab_size * hidden_size) as u64)
                .with_input_shape(vec![max_seq_len])
                .with_output_shape(vec![max_seq_len, hidden_size]),
        );

        // Position embedding
        layers.push(
            LayerViz::new("pos_emb".to_string(), LayerType::Embedding)
                .with_params((max_seq_len * hidden_size) as u64)
                .with_input_shape(vec![max_seq_len])
                .with_output_shape(vec![max_seq_len, hidden_size]),
        );

        // Transformer blocks
        for i in 0..num_layers {
            layers.extend(Self::transformer_block(i, hidden_size, num_heads, 4));
        }

        // Final layer norm
        layers.push(
            LayerViz::new("ln_f".to_string(), LayerType::LayerNorm)
                .with_params(hidden_size as u64 * 2)
                .with_input_shape(vec![max_seq_len, hidden_size])
                .with_output_shape(vec![max_seq_len, hidden_size]),
        );

        // Output projection (often tied with embedding)
        layers.push(
            LayerViz::new("lm_head".to_string(), LayerType::Output)
                .with_params((hidden_size * vocab_size) as u64)
                .with_input_shape(vec![max_seq_len, hidden_size])
                .with_output_shape(vec![max_seq_len, vocab_size]),
        );

        Self::new(layers)
    }
}

/// Health status of gradients in the network.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GradientHealth {
    /// Gradients are healthy.
    Healthy {
        mean_norm: f32,
        max_norm: f32,
        min_norm: f32,
    },
    /// Vanishing gradient problem detected.
    Vanishing {
        min_norm: f32,
        affected_layers: Vec<usize>,
    },
    /// Exploding gradient problem detected.
    Exploding {
        max_norm: f32,
        affected_layers: Vec<usize>,
    },
    /// No gradient information available.
    Unknown,
}

impl Renderer for NetworkArchViz {
    type Output = String;

    fn render(&self, format: OutputFormat) -> VizResult<Self::Output> {
        match format {
            OutputFormat::Ascii => self.render_ascii(),
            OutputFormat::Svg => self.render_svg(),
            OutputFormat::Html => self.render_html(),
            OutputFormat::Json => self.render_json(),
        }
    }
}

impl NetworkArchViz {
    /// Render as ASCII diagram.
    pub fn render_ascii(&self) -> VizResult<String> {
        let mut output = String::new();

        // Title
        if let Some(ref title) = self.config.title {
            output.push_str(title);
            output.push('\n');
            output.push_str(&"=".repeat(title.len()));
            output.push_str("\n\n");
        }

        // Summary
        output.push_str(&format!(
            "Total Parameters: {}\n\n",
            format_param_count(self.total_params())
        ));

        // Layer diagram
        let max_name_len = self.layers.iter().map(|l| l.name.len()).max().unwrap_or(10);

        for (i, layer) in self.layers.iter().enumerate() {
            // Layer box
            let icon = layer.layer_type.ascii_icon();
            let frozen_marker = if layer.frozen { " (frozen)" } else { "" };

            output.push_str(&format!(
                "{:>3} {} {:<width$}{}",
                i,
                icon,
                layer.name,
                frozen_marker,
                width = max_name_len
            ));

            // Parameter count
            if self.config.show_params && layer.params > 0 {
                output.push_str(&format!("  | params: {}", layer.format_params()));
            }

            // Gradient norm
            if self.config.show_gradients {
                if let Some(grad) = layer.gradient_norm {
                    output.push_str(&format!("  | grad: {:.2e}", grad));
                }
            }

            output.push('\n');

            // Connection arrow (except for last layer)
            if i < self.layers.len() - 1 {
                output.push_str(&format!("    {:>width$}|\n", "", width = max_name_len));
                output.push_str(&format!("    {:>width$}v\n", "", width = max_name_len));
            }
        }

        // Skip connections
        let skip_conns: Vec<_> = self
            .connections
            .iter()
            .filter(|c| c.connection_type == ConnectionType::Skip)
            .collect();

        if !skip_conns.is_empty() {
            output.push_str("\nSkip Connections:\n");
            for conn in skip_conns {
                output.push_str(&format!(
                    "  {} -> {} {}\n",
                    self.layers
                        .get(conn.from)
                        .map(|l| l.name.as_str())
                        .unwrap_or("?"),
                    self.layers
                        .get(conn.to)
                        .map(|l| l.name.as_str())
                        .unwrap_or("?"),
                    conn.label.as_deref().unwrap_or("")
                ));
            }
        }

        // Gradient health
        output.push_str("\nGradient Health: ");
        match self.gradient_health() {
            GradientHealth::Healthy { mean_norm, .. } => {
                output.push_str(&format!("OK (mean: {:.2e})\n", mean_norm));
            }
            GradientHealth::Vanishing {
                min_norm,
                affected_layers,
            } => {
                output.push_str(&format!(
                    "WARNING - Vanishing (min: {:.2e}, layers: {:?})\n",
                    min_norm, affected_layers
                ));
            }
            GradientHealth::Exploding {
                max_norm,
                affected_layers,
            } => {
                output.push_str(&format!(
                    "WARNING - Exploding (max: {:.2e}, layers: {:?})\n",
                    max_norm, affected_layers
                ));
            }
            GradientHealth::Unknown => {
                output.push_str("Unknown (no gradient data)\n");
            }
        }

        Ok(output)
    }

    /// Render as SVG.
    pub fn render_svg(&self) -> VizResult<String> {
        let layer_height = 50;
        let layer_width = 200;
        let spacing = 20;
        let margin = 50;

        let width = layer_width + margin * 2;
        let height = self.layers.len() * (layer_height + spacing) + margin * 2;

        let mut svg = format!(
            r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {} {}">"#,
            width, height
        );

        // Background
        svg.push_str(&format!(
            "<rect width=\"{}\" height=\"{}\" fill=\"#fafafa\"/>",
            width, height
        ));

        // Title
        if let Some(ref title) = self.config.title {
            svg.push_str(&format!(
                r#"<text x="{}" y="25" text-anchor="middle" font-size="16" font-weight="bold">{}</text>"#,
                width / 2,
                title
            ));
        }

        // Draw layers
        for (i, layer) in self.layers.iter().enumerate() {
            let x = margin;
            let y = margin + i * (layer_height + spacing);

            // Layer rectangle
            let fill = if layer.frozen {
                "#e0e0e0"
            } else {
                layer.layer_type.svg_color()
            };

            svg.push_str(&format!(
                "<rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" fill=\"{}\" rx=\"5\" stroke=\"#333\" stroke-width=\"1\"/>",
                x, y, layer_width, layer_height, fill
            ));

            // Layer name
            svg.push_str(&format!(
                r#"<text x="{}" y="{}" text-anchor="middle" fill="white" font-size="12" font-weight="bold">{}</text>"#,
                x + layer_width / 2,
                y + 20,
                layer.name
            ));

            // Layer type and params
            let info = format!(
                "{} | {}",
                layer.layer_type.display_name(),
                layer.format_params()
            );
            svg.push_str(&format!(
                r#"<text x="{}" y="{}" text-anchor="middle" fill="white" font-size="10">{}</text>"#,
                x + layer_width / 2,
                y + 38,
                info
            ));

            // Connection arrow to next layer
            if i < self.layers.len() - 1 {
                let arrow_x = x + layer_width / 2;
                let arrow_y1 = y + layer_height;
                let arrow_y2 = y + layer_height + spacing;

                svg.push_str(&format!(
                    "<line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"#333\" stroke-width=\"2\"/>",
                    arrow_x, arrow_y1, arrow_x, arrow_y2
                ));

                // Arrow head
                svg.push_str(&format!(
                    "<polygon points=\"{},{} {},{} {},{}\" fill=\"#333\"/>",
                    arrow_x,
                    arrow_y2,
                    arrow_x - 5,
                    arrow_y2 - 8,
                    arrow_x + 5,
                    arrow_y2 - 8
                ));
            }
        }

        svg.push_str("</svg>");
        Ok(svg)
    }

    /// Render as HTML.
    pub fn render_html(&self) -> VizResult<String> {
        let svg = self.render_svg()?;
        let title = self
            .config
            .title
            .as_deref()
            .unwrap_or("Network Architecture");
        let total_params = format_param_count(self.total_params());
        let json =
            serde_json::to_string(&self).map_err(|e| VizError::RenderError(e.to_string()))?;

        // Build layer table
        let mut table_rows = String::new();
        for (i, layer) in self.layers.iter().enumerate() {
            let grad_display = layer
                .gradient_norm
                .map(|g| format!("{:.2e}", g))
                .unwrap_or_else(|| "-".to_string());

            table_rows.push_str(&format!(
                r#"<tr>
                    <td>{}</td>
                    <td>{}</td>
                    <td>{}</td>
                    <td>{}</td>
                    <td>{}</td>
                    <td>{}</td>
                </tr>"#,
                i,
                layer.name,
                layer.layer_type.display_name(),
                layer.format_params(),
                layer.format_shapes(),
                grad_display
            ));
        }

        Ok(format!(
            r#"<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{}</title>
    <style>
        body {{ font-family: sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{ margin-bottom: 20px; }}
        .viz-container {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }}
        svg {{ max-width: 100%; height: auto; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #f0f0f0; }}
        .summary {{ display: flex; gap: 20px; margin-bottom: 20px; }}
        .summary-item {{ padding: 15px; background: white; border-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        .summary-value {{ font-size: 24px; font-weight: bold; }}
        .summary-label {{ font-size: 12px; color: #666; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{}</h1>
        </div>
        <div class="summary">
            <div class="summary-item">
                <div class="summary-value">{}</div>
                <div class="summary-label">Total Parameters</div>
            </div>
            <div class="summary-item">
                <div class="summary-value">{}</div>
                <div class="summary-label">Layers</div>
            </div>
        </div>
        <div class="viz-container">
            {}
        </div>
        <div class="viz-container">
            <h3>Layer Details</h3>
            <table>
                <thead>
                    <tr>
                        <th>#</th>
                        <th>Name</th>
                        <th>Type</th>
                        <th>Parameters</th>
                        <th>Shapes</th>
                        <th>Gradient Norm</th>
                    </tr>
                </thead>
                <tbody>
                    {}
                </tbody>
            </table>
        </div>
    </div>
    <script>
        window.networkData = {};
    </script>
</body>
</html>"#,
            title,
            title,
            total_params,
            self.layers.len(),
            svg,
            table_rows,
            json
        ))
    }

    /// Render as JSON.
    pub fn render_json(&self) -> VizResult<String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| VizError::RenderError(format!("JSON serialization failed: {}", e)))
    }
}

/// Utility for rendering network architectures.
pub struct NetworkRenderer;

impl NetworkRenderer {
    /// Create a visualization for a GPT-style model.
    pub fn gpt(
        vocab_size: usize,
        hidden_size: usize,
        num_layers: usize,
        num_heads: usize,
        max_seq_len: usize,
        config: NetworkVizConfig,
    ) -> NetworkArchViz {
        NetworkArchViz::gpt_architecture(
            vocab_size,
            hidden_size,
            num_layers,
            num_heads,
            max_seq_len,
        )
        .with_config(config)
    }

    /// Update gradient norms from a list (in layer order).
    pub fn update_gradients(viz: &mut NetworkArchViz, gradients: &[f32]) {
        for (i, grad) in gradients.iter().enumerate() {
            if let Some(layer) = viz.layers.get_mut(i) {
                layer.gradient_norm = Some(*grad);
            }
        }
    }

    /// Update activation statistics from a list of (mean, std) pairs.
    pub fn update_activations(viz: &mut NetworkArchViz, activations: &[(f32, f32)]) {
        for (i, (mean, std)) in activations.iter().enumerate() {
            if let Some(layer) = viz.layers.get_mut(i) {
                layer.activation_mean = Some(*mean);
                layer.activation_std = Some(*std);
            }
        }
    }
}

/// Format a parameter count with appropriate units.
fn format_param_count(count: u64) -> String {
    if count >= 1_000_000_000 {
        format!("{:.2}B", count as f64 / 1_000_000_000.0)
    } else if count >= 1_000_000 {
        format!("{:.2}M", count as f64 / 1_000_000.0)
    } else if count >= 1_000 {
        format!("{:.2}K", count as f64 / 1_000.0)
    } else {
        count.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_viz_creation() {
        let layer = LayerViz::new("test".to_string(), LayerType::Attention)
            .with_params(1_000_000)
            .with_input_shape(vec![512, 768])
            .with_output_shape(vec![512, 768]);

        assert_eq!(layer.name, "test");
        assert_eq!(layer.layer_type, LayerType::Attention);
        assert_eq!(layer.params, 1_000_000);
    }

    #[test]
    fn test_format_params() {
        let layer = LayerViz::new("test".to_string(), LayerType::MLP).with_params(125_000_000);

        assert!(layer.format_params().contains("M"));
    }

    #[test]
    fn test_network_arch_viz() {
        let layers = vec![
            LayerViz::new("emb".to_string(), LayerType::Embedding).with_params(100_000),
            LayerViz::new("attn".to_string(), LayerType::Attention).with_params(200_000),
            LayerViz::new("out".to_string(), LayerType::Output).with_params(50_000),
        ];

        let viz = NetworkArchViz::new(layers);

        assert_eq!(viz.total_params(), 350_000);
        assert_eq!(viz.connections.len(), 2); // Auto-generated forward connections
    }

    #[test]
    fn test_gpt_architecture() {
        let viz = NetworkArchViz::gpt_architecture(50257, 768, 12, 12, 1024);

        assert!(!viz.layers.is_empty());
        assert!(viz.total_params() > 100_000_000); // Should be ~100M+ params
    }

    #[test]
    fn test_gradient_health_healthy() {
        let layers = vec![
            LayerViz::new("l1".to_string(), LayerType::MLP).with_gradient_norm(0.1),
            LayerViz::new("l2".to_string(), LayerType::MLP).with_gradient_norm(0.2),
            LayerViz::new("l3".to_string(), LayerType::MLP).with_gradient_norm(0.15),
        ];

        let viz = NetworkArchViz::new(layers);
        match viz.gradient_health() {
            GradientHealth::Healthy { .. } => (),
            _ => panic!("Expected healthy gradients"),
        }
    }

    #[test]
    fn test_gradient_health_vanishing() {
        let layers = vec![
            LayerViz::new("l1".to_string(), LayerType::MLP).with_gradient_norm(0.1),
            LayerViz::new("l2".to_string(), LayerType::MLP).with_gradient_norm(1e-8),
            LayerViz::new("l3".to_string(), LayerType::MLP).with_gradient_norm(0.15),
        ];

        let viz = NetworkArchViz::new(layers);
        match viz.gradient_health() {
            GradientHealth::Vanishing {
                affected_layers, ..
            } => {
                assert!(affected_layers.contains(&1));
            }
            _ => panic!("Expected vanishing gradients"),
        }
    }

    #[test]
    fn test_render_ascii() {
        let layers = vec![
            LayerViz::new("embedding".to_string(), LayerType::Embedding).with_params(1000),
            LayerViz::new("output".to_string(), LayerType::Output).with_params(500),
        ];

        let viz = NetworkArchViz::new(layers).with_config(NetworkVizConfig {
            title: Some("Test Network".to_string()),
            ..Default::default()
        });

        let ascii = viz.render_ascii().unwrap();
        assert!(ascii.contains("Test Network"));
        assert!(ascii.contains("[E]"));
        assert!(ascii.contains("[O]"));
    }

    #[test]
    fn test_render_svg() {
        let viz = NetworkArchViz::gpt_architecture(1000, 256, 2, 4, 128);
        let svg = viz.render_svg().unwrap();

        assert!(svg.starts_with("<svg"));
        assert!(svg.contains("rect"));
    }

    #[test]
    fn test_render_html() {
        let viz = NetworkArchViz::gpt_architecture(1000, 256, 2, 4, 128);
        let html = viz.render_html().unwrap();

        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("networkData"));
    }

    #[test]
    fn test_params_by_type() {
        let layers = vec![
            LayerViz::new("e1".to_string(), LayerType::Embedding).with_params(100),
            LayerViz::new("a1".to_string(), LayerType::Attention).with_params(200),
            LayerViz::new("m1".to_string(), LayerType::MLP).with_params(300),
            LayerViz::new("a2".to_string(), LayerType::Attention).with_params(200),
        ];

        let viz = NetworkArchViz::new(layers);
        let by_type = viz.params_by_type();

        assert_eq!(by_type.get(&LayerType::Embedding), Some(&100));
        assert_eq!(by_type.get(&LayerType::Attention), Some(&400));
        assert_eq!(by_type.get(&LayerType::MLP), Some(&300));
    }
}
