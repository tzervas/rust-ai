//! 3D flow diagram generation for attention patterns.
//!
//! Provides visualization of attention as flow connections between tokens,
//! representing the "flow" of information through the transformer.

use super::{
    AttentionHead, AttentionViz, HeadAggregation, OutputFormat, Renderer, VizError, VizResult,
};
use serde::{Deserialize, Serialize};

/// A node in the flow diagram representing a token position.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlowNode {
    /// Position index in the sequence.
    pub position: usize,
    /// Token label (if available).
    pub label: Option<String>,
    /// 3D coordinates for visualization (x, y, z).
    pub coords: (f32, f32, f32),
    /// Total attention received by this node.
    pub attention_received: f32,
    /// Total attention sent from this node.
    pub attention_sent: f32,
}

impl FlowNode {
    /// Create a new flow node.
    pub fn new(position: usize) -> Self {
        Self {
            position,
            label: None,
            coords: (position as f32, 0.0, 0.0),
            attention_received: 0.0,
            attention_sent: 0.0,
        }
    }

    /// Set the label for this node.
    pub fn with_label(mut self, label: String) -> Self {
        self.label = Some(label);
        self
    }

    /// Set the 3D coordinates for this node.
    pub fn with_coords(mut self, x: f32, y: f32, z: f32) -> Self {
        self.coords = (x, y, z);
        self
    }

    /// Get the display label (token or position).
    pub fn display_label(&self) -> String {
        self.label
            .clone()
            .unwrap_or_else(|| format!("[{}]", self.position))
    }

    /// Net attention flow (received - sent).
    pub fn net_attention(&self) -> f32 {
        self.attention_received - self.attention_sent
    }
}

/// A connection between two nodes representing attention flow.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlowConnection {
    /// Source node position (query).
    pub from: usize,
    /// Target node position (key).
    pub to: usize,
    /// Attention weight (strength of connection).
    pub weight: f32,
    /// Head index this connection comes from.
    pub head_idx: Option<usize>,
    /// Layer index this connection comes from.
    pub layer_idx: Option<usize>,
}

impl FlowConnection {
    /// Create a new flow connection.
    pub fn new(from: usize, to: usize, weight: f32) -> Self {
        Self {
            from,
            to,
            weight,
            head_idx: None,
            layer_idx: None,
        }
    }

    /// Set the head index.
    pub fn with_head(mut self, head_idx: usize) -> Self {
        self.head_idx = Some(head_idx);
        self
    }

    /// Set the layer index.
    pub fn with_layer(mut self, layer_idx: usize) -> Self {
        self.layer_idx = Some(layer_idx);
        self
    }

    /// Check if this is a self-attention connection.
    pub fn is_self_attention(&self) -> bool {
        self.from == self.to
    }

    /// Get the "distance" of attention (how far the connection spans).
    pub fn attention_distance(&self) -> usize {
        self.from.abs_diff(self.to)
    }
}

/// Configuration for flow diagram rendering.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlowConfig {
    /// Minimum weight threshold for displaying connections.
    pub weight_threshold: f32,
    /// Maximum number of connections to display.
    pub max_connections: Option<usize>,
    /// Whether to show self-attention connections.
    pub show_self_attention: bool,
    /// Whether to use 3D layout (vs flat 2D).
    pub use_3d: bool,
    /// Node spacing in visualization.
    pub node_spacing: f32,
    /// Title for the diagram.
    pub title: Option<String>,
}

impl Default for FlowConfig {
    fn default() -> Self {
        Self {
            weight_threshold: 0.1,
            max_connections: Some(100),
            show_self_attention: false,
            use_3d: true,
            node_spacing: 50.0,
            title: None,
        }
    }
}

/// A complete flow diagram representing attention patterns.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlowDiagram {
    /// Nodes in the diagram.
    pub nodes: Vec<FlowNode>,
    /// Connections between nodes.
    pub connections: Vec<FlowConnection>,
    /// Configuration.
    pub config: FlowConfig,
}

impl FlowDiagram {
    /// Create a new flow diagram from nodes and connections.
    pub fn new(nodes: Vec<FlowNode>, connections: Vec<FlowConnection>) -> Self {
        Self {
            nodes,
            connections,
            config: FlowConfig::default(),
        }
    }

    /// Set configuration.
    pub fn with_config(mut self, config: FlowConfig) -> Self {
        self.config = config;
        self
    }

    /// Create a flow diagram from attention weights.
    pub fn from_attention(
        weights: &[Vec<f32>],
        tokens: Option<&[String]>,
        config: FlowConfig,
    ) -> VizResult<Self> {
        let seq_len = weights.len();
        if seq_len == 0 {
            return Err(VizError::EmptySequence);
        }

        // Create nodes with 3D positions
        let mut nodes: Vec<FlowNode> = (0..seq_len)
            .map(|i| {
                let angle = 2.0 * std::f32::consts::PI * (i as f32) / (seq_len as f32);
                let radius = config.node_spacing;
                let x = radius * angle.cos();
                let y = radius * angle.sin();
                let z = if config.use_3d {
                    (i as f32 / seq_len as f32) * config.node_spacing
                } else {
                    0.0
                };

                let mut node = FlowNode::new(i).with_coords(x, y, z);
                if let Some(labels) = tokens {
                    if let Some(label) = labels.get(i) {
                        node = node.with_label(label.clone());
                    }
                }
                node
            })
            .collect();

        // Create connections
        let mut connections = Vec::new();
        for (i, row) in weights.iter().enumerate() {
            for (j, &weight) in row.iter().enumerate() {
                if weight >= config.weight_threshold {
                    if config.show_self_attention || i != j {
                        connections.push(FlowConnection::new(i, j, weight));
                    }
                }
            }
        }

        // Sort by weight and limit
        connections.sort_by(|a, b| {
            b.weight
                .partial_cmp(&a.weight)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        if let Some(max) = config.max_connections {
            connections.truncate(max);
        }

        // Update node attention statistics
        for conn in &connections {
            nodes[conn.from].attention_sent += conn.weight;
            nodes[conn.to].attention_received += conn.weight;
        }

        Ok(Self {
            nodes,
            connections,
            config,
        })
    }

    /// Get the number of nodes.
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Get the number of connections.
    pub fn num_connections(&self) -> usize {
        self.connections.len()
    }

    /// Get statistics about the flow diagram.
    pub fn statistics(&self) -> FlowStatistics {
        let total_weight: f32 = self.connections.iter().map(|c| c.weight).sum();
        let max_weight = self
            .connections
            .iter()
            .map(|c| c.weight)
            .fold(0.0f32, f32::max);
        let avg_distance = if self.connections.is_empty() {
            0.0
        } else {
            self.connections
                .iter()
                .map(|c| c.attention_distance() as f32)
                .sum::<f32>()
                / self.connections.len() as f32
        };
        let self_attention_count = self
            .connections
            .iter()
            .filter(|c| c.is_self_attention())
            .count();

        FlowStatistics {
            num_nodes: self.nodes.len(),
            num_connections: self.connections.len(),
            total_weight,
            max_weight,
            avg_distance,
            self_attention_count,
        }
    }

    /// Filter connections to only include those above a threshold.
    pub fn filter_by_weight(&mut self, threshold: f32) {
        self.connections.retain(|c| c.weight >= threshold);
    }

    /// Get the top-k strongest connections.
    pub fn top_k_connections(&self, k: usize) -> Vec<&FlowConnection> {
        let mut sorted: Vec<_> = self.connections.iter().collect();
        sorted.sort_by(|a, b| {
            b.weight
                .partial_cmp(&a.weight)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted.truncate(k);
        sorted
    }
}

/// Statistics about a flow diagram.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlowStatistics {
    /// Number of nodes.
    pub num_nodes: usize,
    /// Number of connections.
    pub num_connections: usize,
    /// Total weight of all connections.
    pub total_weight: f32,
    /// Maximum connection weight.
    pub max_weight: f32,
    /// Average attention distance.
    pub avg_distance: f32,
    /// Count of self-attention connections.
    pub self_attention_count: usize,
}

impl Renderer for FlowDiagram {
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

impl FlowDiagram {
    /// Render as ASCII representation.
    pub fn render_ascii(&self) -> VizResult<String> {
        let mut output = String::new();

        // Title
        if let Some(ref title) = self.config.title {
            output.push_str(title);
            output.push('\n');
            output.push_str(&"=".repeat(title.len()));
            output.push_str("\n\n");
        }

        // Nodes
        output.push_str("Nodes:\n");
        for node in &self.nodes {
            output.push_str(&format!(
                "  [{}] {} - recv: {:.3}, sent: {:.3}, net: {:.3}\n",
                node.position,
                node.display_label(),
                node.attention_received,
                node.attention_sent,
                node.net_attention()
            ));
        }
        output.push('\n');

        // Top connections
        output.push_str("Top Connections:\n");
        let top = self.top_k_connections(20);
        for conn in top {
            let from_label = self.nodes[conn.from].display_label();
            let to_label = self.nodes[conn.to].display_label();
            output.push_str(&format!(
                "  {} -> {} : {:.3}\n",
                from_label, to_label, conn.weight
            ));
        }

        // Statistics
        output.push('\n');
        let stats = self.statistics();
        output.push_str(&format!("Statistics:\n"));
        output.push_str(&format!("  Nodes: {}\n", stats.num_nodes));
        output.push_str(&format!("  Connections: {}\n", stats.num_connections));
        output.push_str(&format!("  Total Weight: {:.3}\n", stats.total_weight));
        output.push_str(&format!("  Max Weight: {:.3}\n", stats.max_weight));
        output.push_str(&format!("  Avg Distance: {:.2}\n", stats.avg_distance));

        Ok(output)
    }

    /// Render as SVG.
    pub fn render_svg(&self) -> VizResult<String> {
        let size = 400;
        let center = size as f32 / 2.0;
        let radius = (size as f32 / 2.0) - 50.0;

        let mut svg = format!(
            r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {} {}">"#,
            size, size
        );

        // Background
        svg.push_str(&format!(
            "<rect width=\"{}\" height=\"{}\" fill=\"#fafafa\"/>",
            size, size
        ));

        // Title
        if let Some(ref title) = self.config.title {
            svg.push_str(&format!(
                r#"<text x="{}" y="20" text-anchor="middle" font-size="14" font-weight="bold">{}</text>"#,
                center, title
            ));
        }

        // Calculate node positions (circular layout)
        let node_positions: Vec<(f32, f32)> = (0..self.nodes.len())
            .map(|i| {
                let angle = 2.0 * std::f32::consts::PI * (i as f32) / (self.nodes.len() as f32)
                    - std::f32::consts::PI / 2.0;
                let x = center + radius * angle.cos();
                let y = center + radius * angle.sin();
                (x, y)
            })
            .collect();

        // Draw connections
        let max_weight = self
            .connections
            .iter()
            .map(|c| c.weight)
            .fold(0.0f32, f32::max);

        for conn in &self.connections {
            let (x1, y1) = node_positions[conn.from];
            let (x2, y2) = node_positions[conn.to];
            let opacity = (conn.weight / max_weight).clamp(0.1, 1.0);
            let stroke_width = 1.0 + (conn.weight / max_weight) * 3.0;

            // Draw as quadratic curve for better visibility
            let mx = (x1 + x2) / 2.0;
            let my = (y1 + y2) / 2.0;
            let dx = x2 - x1;
            let dy = y2 - y1;
            let cx = mx - dy * 0.1;
            let cy = my + dx * 0.1;

            svg.push_str(&format!(
                r#"<path d="M{:.1},{:.1} Q{:.1},{:.1} {:.1},{:.1}" fill="none" stroke="steelblue" stroke-opacity="{:.2}" stroke-width="{:.1}"/>"#,
                x1, y1, cx, cy, x2, y2, opacity, stroke_width
            ));

            // Arrow head
            let arrow_len = 8.0;
            let angle = (y2 - cy).atan2(x2 - cx);
            let ax1 = x2 - arrow_len * (angle + 0.3).cos();
            let ay1 = y2 - arrow_len * (angle + 0.3).sin();
            let ax2 = x2 - arrow_len * (angle - 0.3).cos();
            let ay2 = y2 - arrow_len * (angle - 0.3).sin();

            svg.push_str(&format!(
                r#"<polygon points="{:.1},{:.1} {:.1},{:.1} {:.1},{:.1}" fill="steelblue" fill-opacity="{:.2}"/>"#,
                x2, y2, ax1, ay1, ax2, ay2, opacity
            ));
        }

        // Draw nodes
        for (i, node) in self.nodes.iter().enumerate() {
            let (x, y) = node_positions[i];
            let node_radius = 15.0;

            // Node circle
            svg.push_str(&format!(
                "<circle cx=\"{:.1}\" cy=\"{:.1}\" r=\"{}\" fill=\"white\" stroke=\"#333\" stroke-width=\"2\"/>",
                x, y, node_radius
            ));

            // Node label
            let label = if let Some(ref l) = node.label {
                if l.len() > 4 {
                    format!("{}...", &l[..3])
                } else {
                    l.clone()
                }
            } else {
                node.position.to_string()
            };

            svg.push_str(&format!(
                r#"<text x="{:.1}" y="{:.1}" text-anchor="middle" dominant-baseline="middle" font-size="10">{}</text>"#,
                x, y, label
            ));
        }

        svg.push_str("</svg>");
        Ok(svg)
    }

    /// Render as HTML with interactive features.
    pub fn render_html(&self) -> VizResult<String> {
        let svg = self.render_svg()?;
        let title = self.config.title.as_deref().unwrap_or("Attention Flow");
        let stats = self.statistics();
        let json =
            serde_json::to_string(&self).map_err(|e| VizError::RenderError(e.to_string()))?;

        Ok(format!(
            r#"<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{}</title>
    <style>
        body {{ font-family: sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 900px; margin: 0 auto; }}
        .viz-container {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        svg {{ max-width: 100%; height: auto; }}
        .stats {{ margin-top: 20px; padding: 15px; background: #f0f0f0; border-radius: 4px; }}
        .stats h3 {{ margin-top: 0; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }}
        .stat-item {{ padding: 10px; background: white; border-radius: 4px; }}
        .stat-value {{ font-size: 24px; font-weight: bold; color: #333; }}
        .stat-label {{ font-size: 12px; color: #666; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="viz-container">
            {}
        </div>
        <div class="stats">
            <h3>Flow Statistics</h3>
            <div class="stats-grid">
                <div class="stat-item">
                    <div class="stat-value">{}</div>
                    <div class="stat-label">Nodes</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{}</div>
                    <div class="stat-label">Connections</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{:.2}</div>
                    <div class="stat-label">Avg Distance</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{:.3}</div>
                    <div class="stat-label">Max Weight</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{:.3}</div>
                    <div class="stat-label">Total Weight</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{}</div>
                    <div class="stat-label">Self-Attention</div>
                </div>
            </div>
        </div>
    </div>
    <script>
        // Data for external tools
        window.flowData = {};
    </script>
</body>
</html>"#,
            title,
            svg,
            stats.num_nodes,
            stats.num_connections,
            stats.avg_distance,
            stats.max_weight,
            stats.total_weight,
            stats.self_attention_count,
            json
        ))
    }

    /// Render as JSON.
    pub fn render_json(&self) -> VizResult<String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| VizError::RenderError(format!("JSON serialization failed: {}", e)))
    }
}

/// Renderer for creating flow diagrams from various sources.
pub struct FlowRenderer;

impl FlowRenderer {
    /// Create a flow diagram from an attention head.
    pub fn from_attention_head(head: &AttentionHead, config: FlowConfig) -> VizResult<FlowDiagram> {
        FlowDiagram::from_attention(&head.weights, None, config)
    }

    /// Create a flow diagram from an attention visualization.
    pub fn from_attention_viz(
        viz: &AttentionViz,
        aggregation: HeadAggregation,
        config: FlowConfig,
    ) -> VizResult<FlowDiagram> {
        let weights = viz.aggregate(aggregation);
        FlowDiagram::from_attention(&weights, viz.tokens.as_deref(), config)
    }

    /// Create per-head flow diagrams.
    pub fn head_flows(viz: &AttentionViz, config: FlowConfig) -> VizResult<Vec<FlowDiagram>> {
        viz.heads
            .iter()
            .map(|head| {
                let mut head_config = config.clone();
                head_config.title = Some(format!("Head {}", head.head_idx));
                FlowDiagram::from_attention(&head.weights, viz.tokens.as_deref(), head_config)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_weights(seq_len: usize) -> Vec<Vec<f32>> {
        (0..seq_len)
            .map(|i| {
                let mut row = vec![0.0; seq_len];
                for j in 0..seq_len {
                    let dist = (i as i32 - j as i32).abs() as f32;
                    row[j] = (-dist * 0.5).exp();
                }
                let sum: f32 = row.iter().sum();
                row.iter_mut().for_each(|x| *x /= sum);
                row
            })
            .collect()
    }

    #[test]
    fn test_flow_node_creation() {
        let node = FlowNode::new(0)
            .with_label("test".to_string())
            .with_coords(1.0, 2.0, 3.0);

        assert_eq!(node.position, 0);
        assert_eq!(node.label, Some("test".to_string()));
        assert_eq!(node.coords, (1.0, 2.0, 3.0));
    }

    #[test]
    fn test_flow_connection() {
        let conn = FlowConnection::new(0, 5, 0.8).with_head(2).with_layer(1);

        assert!(!conn.is_self_attention());
        assert_eq!(conn.attention_distance(), 5);
        assert_eq!(conn.head_idx, Some(2));
    }

    #[test]
    fn test_flow_diagram_from_attention() {
        let weights = sample_weights(8);
        let config = FlowConfig {
            weight_threshold: 0.05,
            ..Default::default()
        };

        let diagram = FlowDiagram::from_attention(&weights, None, config).unwrap();

        assert_eq!(diagram.num_nodes(), 8);
        assert!(diagram.num_connections() > 0);
    }

    #[test]
    fn test_flow_statistics() {
        let weights = sample_weights(4);
        let config = FlowConfig {
            weight_threshold: 0.01,
            show_self_attention: true,
            ..Default::default()
        };

        let diagram = FlowDiagram::from_attention(&weights, None, config).unwrap();
        let stats = diagram.statistics();

        assert_eq!(stats.num_nodes, 4);
        assert!(stats.total_weight > 0.0);
    }

    #[test]
    fn test_render_ascii() {
        let weights = sample_weights(4);
        let config = FlowConfig {
            title: Some("Test Flow".to_string()),
            ..Default::default()
        };

        let diagram = FlowDiagram::from_attention(&weights, None, config).unwrap();
        let ascii = diagram.render_ascii().unwrap();

        assert!(ascii.contains("Test Flow"));
        assert!(ascii.contains("Nodes:"));
        assert!(ascii.contains("Statistics:"));
    }

    #[test]
    fn test_render_svg() {
        let weights = sample_weights(4);
        let diagram = FlowDiagram::from_attention(&weights, None, FlowConfig::default()).unwrap();
        let svg = diagram.render_svg().unwrap();

        assert!(svg.starts_with("<svg"));
        assert!(svg.contains("circle"));
        assert!(svg.contains("path"));
    }

    #[test]
    fn test_render_html() {
        let weights = sample_weights(4);
        let diagram = FlowDiagram::from_attention(&weights, None, FlowConfig::default()).unwrap();
        let html = diagram.render_html().unwrap();

        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("flowData"));
    }

    #[test]
    fn test_top_k_connections() {
        let weights = sample_weights(8);
        let config = FlowConfig {
            weight_threshold: 0.01,
            max_connections: None,
            ..Default::default()
        };

        let diagram = FlowDiagram::from_attention(&weights, None, config).unwrap();
        let top_5 = diagram.top_k_connections(5);

        assert!(top_5.len() <= 5);
        // Verify sorted by weight
        for i in 1..top_5.len() {
            assert!(top_5[i - 1].weight >= top_5[i].weight);
        }
    }
}
