//! Architecture Block Diagram Visualization
//!
//! Provides modern CNN/Transformer architecture visualization with:
//! - 3D isometric boxes representing layers
//! - Size proportional to layer dimensions/parameters
//! - Curved skip connection arrows
//! - Layer type color coding
//! - Text labels with layer names and shapes
//!
//! # Example
//!
//! ```rust,ignore
//! use training_tools::viz3d::architecture::*;
//!
//! let mut diagram = ArchitectureDiagram::new()
//!     .add_block(LayerBlock::new("embed", LayerType::Embedding)
//!         .with_shape(vec![512, 768]))
//!     .add_block(LayerBlock::new("attn_1", LayerType::Attention)
//!         .with_shape(vec![512, 768]))
//!     .add_block(LayerBlock::new("mlp_1", LayerType::MLP)
//!         .with_shape(vec![512, 3072]));
//!
//! let svg = diagram.to_svg(1200, 600);
//! std::fs::write("architecture.svg", svg).unwrap();
//! ```

use super::colors::Color;

/// Types of neural network layers with associated default colors
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LayerType {
    /// Input/output embedding layer (purple)
    #[default]
    Embedding,
    /// Self-attention or cross-attention (orange)
    Attention,
    /// Multi-layer perceptron / feed-forward (green)
    MLP,
    /// Layer normalization (cyan)
    LayerNorm,
    /// Convolutional layer (blue)
    Convolution,
    /// Pooling layer (light blue)
    Pooling,
    /// Residual connection block (yellow)
    Residual,
    /// Output projection layer (red)
    Output,
}

impl LayerType {
    /// Get the default color for this layer type
    pub fn default_color(&self) -> [f32; 4] {
        match self {
            LayerType::Embedding => [0.58, 0.40, 0.74, 1.0], // Purple
            LayerType::Attention => [1.0, 0.50, 0.05, 1.0],  // Orange
            LayerType::MLP => [0.17, 0.63, 0.17, 1.0],       // Green
            LayerType::LayerNorm => [0.09, 0.75, 0.81, 1.0], // Cyan
            LayerType::Convolution => [0.12, 0.47, 0.71, 1.0], // Blue
            LayerType::Pooling => [0.39, 0.71, 0.91, 1.0],   // Light blue
            LayerType::Residual => [0.90, 0.86, 0.20, 1.0],  // Yellow
            LayerType::Output => [0.84, 0.15, 0.16, 1.0],    // Red
        }
    }

    /// Get short label for the layer type
    pub fn label(&self) -> &'static str {
        match self {
            LayerType::Embedding => "Embed",
            LayerType::Attention => "Attn",
            LayerType::MLP => "MLP",
            LayerType::LayerNorm => "LN",
            LayerType::Convolution => "Conv",
            LayerType::Pooling => "Pool",
            LayerType::Residual => "Res",
            LayerType::Output => "Out",
        }
    }
}

/// A single layer block in the architecture diagram
#[derive(Debug, Clone)]
pub struct LayerBlock {
    /// Layer name/identifier
    pub name: String,
    /// Type of layer
    pub layer_type: LayerType,
    /// Input tensor shape
    pub input_shape: Vec<usize>,
    /// Output tensor shape
    pub output_shape: Vec<usize>,
    /// Number of trainable parameters
    pub params: u64,
    /// 3D position in the diagram (computed by layout)
    pub position: [f32; 3],
    /// 3D size of the block (computed from shapes)
    pub size: [f32; 3],
    /// RGBA color (defaults to layer type color)
    pub color: [f32; 4],
    /// Optional sublabel (e.g., "12 heads")
    pub sublabel: Option<String>,
}

impl LayerBlock {
    /// Create a new layer block with default settings
    pub fn new(name: impl Into<String>, layer_type: LayerType) -> Self {
        Self {
            name: name.into(),
            layer_type,
            input_shape: Vec::new(),
            output_shape: Vec::new(),
            params: 0,
            position: [0.0, 0.0, 0.0],
            size: [1.0, 1.0, 0.5],
            color: layer_type.default_color(),
            sublabel: None,
        }
    }

    /// Set input and output shapes (output defaults to input if not specified)
    pub fn with_shape(mut self, shape: Vec<usize>) -> Self {
        self.input_shape = shape.clone();
        self.output_shape = shape;
        self
    }

    /// Set input shape
    pub fn with_input_shape(mut self, shape: Vec<usize>) -> Self {
        self.input_shape = shape;
        self
    }

    /// Set output shape
    pub fn with_output_shape(mut self, shape: Vec<usize>) -> Self {
        self.output_shape = shape;
        self
    }

    /// Set number of parameters
    pub fn with_params(mut self, params: u64) -> Self {
        self.params = params;
        self
    }

    /// Set custom color
    pub fn with_color(mut self, color: [f32; 4]) -> Self {
        self.color = color;
        self
    }

    /// Set sublabel (e.g., "12 heads", "4x expansion")
    pub fn with_sublabel(mut self, sublabel: impl Into<String>) -> Self {
        self.sublabel = Some(sublabel.into());
        self
    }

    /// Format shape as string (e.g., "512 x 768")
    pub fn shape_string(&self) -> String {
        if self.output_shape.is_empty() {
            return String::new();
        }
        self.output_shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(" x ")
    }

    /// Format parameter count as human-readable string
    pub fn params_string(&self) -> String {
        if self.params == 0 {
            return String::new();
        }
        if self.params >= 1_000_000_000 {
            format!("{:.1}B", self.params as f64 / 1_000_000_000.0)
        } else if self.params >= 1_000_000 {
            format!("{:.1}M", self.params as f64 / 1_000_000.0)
        } else if self.params >= 1_000 {
            format!("{:.1}K", self.params as f64 / 1_000.0)
        } else {
            self.params.to_string()
        }
    }
}

/// Type of connection between layers
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ConnectionType {
    /// Normal forward flow (straight arrow)
    #[default]
    Sequential,
    /// Residual/skip connection (curved arrow)
    Skip,
    /// Feature concatenation (merging arrow)
    Concat,
    /// Element-wise addition (+ symbol)
    Add,
}

impl ConnectionType {
    /// Get default color for connection type
    pub fn default_color(&self) -> [f32; 4] {
        match self {
            ConnectionType::Sequential => [0.3, 0.3, 0.3, 1.0],
            ConnectionType::Skip => [0.9, 0.4, 0.2, 0.8],
            ConnectionType::Concat => [0.2, 0.6, 0.9, 0.8],
            ConnectionType::Add => [0.2, 0.8, 0.2, 0.8],
        }
    }
}

/// Connection between two layer blocks
#[derive(Debug, Clone)]
pub struct LayerConnection {
    /// Index of source block
    pub from_idx: usize,
    /// Index of destination block
    pub to_idx: usize,
    /// Type of connection
    pub connection_type: ConnectionType,
    /// Optional label
    pub label: Option<String>,
}

impl LayerConnection {
    /// Create a new sequential connection
    pub fn sequential(from: usize, to: usize) -> Self {
        Self {
            from_idx: from,
            to_idx: to,
            connection_type: ConnectionType::Sequential,
            label: None,
        }
    }

    /// Create a skip/residual connection
    pub fn skip(from: usize, to: usize) -> Self {
        Self {
            from_idx: from,
            to_idx: to,
            connection_type: ConnectionType::Skip,
            label: None,
        }
    }

    /// Create an add connection
    pub fn add(from: usize, to: usize) -> Self {
        Self {
            from_idx: from,
            to_idx: to,
            connection_type: ConnectionType::Add,
            label: None,
        }
    }

    /// Create a concat connection
    pub fn concat(from: usize, to: usize) -> Self {
        Self {
            from_idx: from,
            to_idx: to,
            connection_type: ConnectionType::Concat,
            label: None,
        }
    }

    /// Set connection type
    pub fn with_type(mut self, connection_type: ConnectionType) -> Self {
        self.connection_type = connection_type;
        self
    }

    /// Set label
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }
}

/// Style for skip connections
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SkipStyle {
    /// Smooth curved arrow
    #[default]
    CurvedArrow,
    /// Dashed straight line
    DashedLine,
    /// Bezier curve with control points
    BezierCurve,
}

/// Dedicated skip connection with custom styling
#[derive(Debug, Clone)]
pub struct SkipConnection {
    /// Index of source block
    pub from_idx: usize,
    /// Index of destination block
    pub to_idx: usize,
    /// Visual style
    pub style: SkipStyle,
    /// Color (defaults to orange)
    pub color: [f32; 4],
    /// Line width
    pub width: f32,
}

impl SkipConnection {
    /// Create a new skip connection
    pub fn new(from: usize, to: usize) -> Self {
        Self {
            from_idx: from,
            to_idx: to,
            style: SkipStyle::CurvedArrow,
            color: [0.9, 0.4, 0.2, 0.8],
            width: 2.0,
        }
    }

    /// Set style
    pub fn with_style(mut self, style: SkipStyle) -> Self {
        self.style = style;
        self
    }

    /// Set color
    pub fn with_color(mut self, color: [f32; 4]) -> Self {
        self.color = color;
        self
    }

    /// Set line width
    pub fn with_width(mut self, width: f32) -> Self {
        self.width = width;
        self
    }
}

/// Layout style for the architecture diagram
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LayoutStyle {
    /// Left to right horizontal flow
    Horizontal,
    /// Top to bottom vertical flow
    Vertical,
    /// 3D isometric view (default)
    #[default]
    Isometric3D,
}

/// Main architecture diagram container
#[derive(Debug, Clone)]
pub struct ArchitectureDiagram {
    /// All layer blocks
    pub blocks: Vec<LayerBlock>,
    /// Connections between layers (sequential flow)
    pub connections: Vec<LayerConnection>,
    /// Skip/residual connections
    pub skip_connections: Vec<SkipConnection>,
    /// Layout style
    pub layout: LayoutStyle,
    /// Diagram title
    pub title: Option<String>,
    /// Background color
    pub background_color: [f32; 4],
    /// Spacing between blocks
    pub block_spacing: f32,
    /// Base size scale factor
    pub scale_factor: f32,
}

impl Default for ArchitectureDiagram {
    fn default() -> Self {
        Self {
            blocks: Vec::new(),
            connections: Vec::new(),
            skip_connections: Vec::new(),
            layout: LayoutStyle::Isometric3D,
            title: None,
            background_color: [0.95, 0.95, 0.97, 1.0],
            block_spacing: 1.5,
            scale_factor: 1.0,
        }
    }
}

impl ArchitectureDiagram {
    /// Create a new empty architecture diagram
    pub fn new() -> Self {
        Self::default()
    }

    /// Create diagram from a TritterConfig (placeholder for integration)
    pub fn from_tritter_config(_config: &TritterConfig) -> Self {
        // Build diagram from Tritter model configuration
        let mut diagram = Self::new();

        // Embedding layer
        diagram = diagram.add_block(
            LayerBlock::new("embed", LayerType::Embedding)
                .with_shape(vec![_config.seq_len, _config.hidden_dim])
                .with_params(_config.vocab_size as u64 * _config.hidden_dim as u64),
        );

        // Transformer layers
        for i in 0.._config.num_layers {
            let layer_name = format!("layer_{}", i);

            // Layer norm 1
            diagram = diagram.add_block(
                LayerBlock::new(format!("{}_ln1", layer_name), LayerType::LayerNorm)
                    .with_shape(vec![_config.seq_len, _config.hidden_dim])
                    .with_params(_config.hidden_dim as u64 * 2),
            );

            // Attention
            diagram = diagram.add_block(
                LayerBlock::new(format!("{}_attn", layer_name), LayerType::Attention)
                    .with_shape(vec![_config.seq_len, _config.hidden_dim])
                    .with_sublabel(format!("{} heads", _config.num_heads))
                    .with_params(4 * _config.hidden_dim as u64 * _config.hidden_dim as u64),
            );

            // Add skip connection around attention
            let attn_idx = diagram.blocks.len() - 1;
            let ln1_idx = attn_idx - 1;
            diagram
                .skip_connections
                .push(SkipConnection::new(ln1_idx - 1, attn_idx + 1));

            // Layer norm 2
            diagram = diagram.add_block(
                LayerBlock::new(format!("{}_ln2", layer_name), LayerType::LayerNorm)
                    .with_shape(vec![_config.seq_len, _config.hidden_dim])
                    .with_params(_config.hidden_dim as u64 * 2),
            );

            // MLP
            let mlp_dim = _config.hidden_dim * 4;
            diagram = diagram.add_block(
                LayerBlock::new(format!("{}_mlp", layer_name), LayerType::MLP)
                    .with_shape(vec![_config.seq_len, mlp_dim])
                    .with_sublabel("4x expansion")
                    .with_params(2 * _config.hidden_dim as u64 * mlp_dim as u64),
            );

            // Add skip connection around MLP
            let mlp_idx = diagram.blocks.len() - 1;
            let ln2_idx = mlp_idx - 1;
            diagram
                .skip_connections
                .push(SkipConnection::new(ln2_idx - 1, mlp_idx + 1));
        }

        // Final layer norm
        diagram = diagram.add_block(
            LayerBlock::new("final_ln", LayerType::LayerNorm)
                .with_shape(vec![_config.seq_len, _config.hidden_dim])
                .with_params(_config.hidden_dim as u64 * 2),
        );

        // Output projection
        diagram = diagram.add_block(
            LayerBlock::new("output", LayerType::Output)
                .with_shape(vec![_config.seq_len, _config.vocab_size])
                .with_params(_config.hidden_dim as u64 * _config.vocab_size as u64),
        );

        // Add sequential connections
        for i in 0..diagram.blocks.len().saturating_sub(1) {
            diagram
                .connections
                .push(LayerConnection::sequential(i, i + 1));
        }

        diagram.title = Some(format!(
            "Tritter Model ({} layers, {}M params)",
            _config.num_layers,
            diagram.total_params() / 1_000_000
        ));

        diagram.compute_layout();
        diagram
    }

    /// Create diagram from a list of layer descriptions
    pub fn from_layer_list(layers: &[LayerInfo]) -> Self {
        let mut diagram = Self::new();

        for info in layers {
            let block = LayerBlock::new(&info.name, info.layer_type)
                .with_shape(info.shape.clone())
                .with_params(info.params);

            diagram = diagram.add_block(block);
        }

        // Add sequential connections
        for i in 0..diagram.blocks.len().saturating_sub(1) {
            diagram
                .connections
                .push(LayerConnection::sequential(i, i + 1));
        }

        diagram.compute_layout();
        diagram
    }

    /// Add a layer block
    pub fn add_block(mut self, block: LayerBlock) -> Self {
        self.blocks.push(block);
        self
    }

    /// Add a connection
    pub fn add_connection(mut self, connection: LayerConnection) -> Self {
        self.connections.push(connection);
        self
    }

    /// Add a skip connection
    pub fn add_skip(mut self, skip: SkipConnection) -> Self {
        self.skip_connections.push(skip);
        self
    }

    /// Set layout style
    pub fn with_layout(mut self, layout: LayoutStyle) -> Self {
        self.layout = layout;
        self
    }

    /// Set title
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Set background color
    pub fn with_background(mut self, color: [f32; 4]) -> Self {
        self.background_color = color;
        self
    }

    /// Total parameters across all blocks
    pub fn total_params(&self) -> u64 {
        self.blocks.iter().map(|b| b.params).sum()
    }

    /// Compute block sizes based on shapes
    pub fn compute_block_sizes(&mut self) {
        // Find max dimensions for normalization
        let max_dim: usize = self
            .blocks
            .iter()
            .flat_map(|b| b.output_shape.iter())
            .copied()
            .max()
            .unwrap_or(1);

        let max_dim_f = max_dim as f32;

        for block in &mut self.blocks {
            if block.output_shape.is_empty() {
                block.size = [1.0, 1.0, 0.3];
                continue;
            }

            // Size based on dimensions (normalized)
            let width = block.output_shape.get(0).copied().unwrap_or(1) as f32 / max_dim_f;
            let height = block.output_shape.get(1).copied().unwrap_or(1) as f32 / max_dim_f;
            let depth = block.output_shape.get(2).copied().unwrap_or(1) as f32 / max_dim_f;

            // Scale and clamp to reasonable sizes
            let base_scale = 2.0 * self.scale_factor;
            block.size = [
                (width * base_scale).clamp(0.3, 3.0),
                (height * base_scale).clamp(0.3, 3.0),
                (depth * base_scale).clamp(0.2, 1.5),
            ];

            // Make attention blocks wider, MLP blocks taller
            match block.layer_type {
                LayerType::Attention => {
                    block.size[0] *= 1.3;
                    block.size[2] *= 0.7;
                }
                LayerType::MLP => {
                    block.size[1] *= 1.2;
                }
                LayerType::LayerNorm => {
                    block.size = [block.size[0], 0.2, block.size[2] * 0.5];
                }
                _ => {}
            }
        }
    }

    /// Compute positions for all blocks based on layout
    pub fn compute_layout(&mut self) {
        self.compute_block_sizes();

        match self.layout {
            LayoutStyle::Horizontal => self.layout_horizontal(),
            LayoutStyle::Vertical => self.layout_vertical(),
            LayoutStyle::Isometric3D => self.layout_isometric(),
        }
    }

    fn layout_horizontal(&mut self) {
        let mut x = 0.0;
        for block in &mut self.blocks {
            block.position = [x, 0.0, 0.0];
            x += block.size[0] + self.block_spacing;
        }
    }

    fn layout_vertical(&mut self) {
        let mut y = 0.0;
        for block in &mut self.blocks {
            block.position = [0.0, y, 0.0];
            y -= block.size[1] + self.block_spacing;
        }
    }

    fn layout_isometric(&mut self) {
        // Isometric layout with slight depth offset per layer
        let mut x = 0.0;
        let depth_step = 0.1;
        for (i, block) in self.blocks.iter_mut().enumerate() {
            block.position = [x, 0.0, i as f32 * depth_step];
            x += block.size[0] + self.block_spacing;
        }
    }

    /// Project a 3D point to 2D using isometric projection
    pub fn isometric_project(point: [f32; 3]) -> [f32; 2] {
        // Isometric projection angles
        let angle = std::f32::consts::FRAC_PI_6; // 30 degrees

        let x = (point[0] - point[2]) * angle.cos();
        let y = point[1] + (point[0] + point[2]) * angle.sin() * 0.5;

        [x, y]
    }

    /// Render to SVG string
    pub fn to_svg(&self, width: u32, height: u32) -> String {
        let mut svg = SvgBuilder::new(width, height);

        // Background
        svg.rect(
            0.0,
            0.0,
            width as f32,
            height as f32,
            self.background_color,
            None,
        );

        // Title
        if let Some(ref title) = self.title {
            svg.text(
                width as f32 / 2.0,
                30.0,
                title,
                16.0,
                [0.2, 0.2, 0.2, 1.0],
                "middle",
            );
        }

        // Compute bounding box for scaling
        let (min_x, max_x, min_y, max_y) = self.compute_bounds();
        let diagram_width = max_x - min_x;
        let diagram_height = max_y - min_y;

        // Calculate scale and offset to fit in viewport
        let margin = 80.0;
        let available_width = width as f32 - 2.0 * margin;
        let available_height = height as f32 - 2.0 * margin - 40.0; // Extra for title

        let scale_x = available_width / diagram_width.max(1.0);
        let scale_y = available_height / diagram_height.max(1.0);
        let scale = scale_x.min(scale_y);

        let offset_x = margin + (available_width - diagram_width * scale) / 2.0 - min_x * scale;
        let offset_y =
            margin + 40.0 + (available_height - diagram_height * scale) / 2.0 - min_y * scale;

        // Transform function
        let transform =
            |p: [f32; 2]| -> [f32; 2] { [p[0] * scale + offset_x, p[1] * scale + offset_y] };

        // Draw connections first (behind blocks)
        for conn in &self.connections {
            if conn.from_idx >= self.blocks.len() || conn.to_idx >= self.blocks.len() {
                continue;
            }

            let from_block = &self.blocks[conn.from_idx];
            let to_block = &self.blocks[conn.to_idx];

            let from_pos = Self::isometric_project([
                from_block.position[0] + from_block.size[0],
                from_block.position[1] + from_block.size[1] / 2.0,
                from_block.position[2],
            ]);
            let to_pos = Self::isometric_project([
                to_block.position[0],
                to_block.position[1] + to_block.size[1] / 2.0,
                to_block.position[2],
            ]);

            let from_t = transform(from_pos);
            let to_t = transform(to_pos);

            svg.line(
                from_t[0],
                from_t[1],
                to_t[0],
                to_t[1],
                conn.connection_type.default_color(),
                1.5,
            );

            // Arrow head
            svg.arrow_head(
                to_t[0],
                to_t[1],
                from_t,
                conn.connection_type.default_color(),
            );
        }

        // Draw skip connections (curved)
        for skip in &self.skip_connections {
            if skip.from_idx >= self.blocks.len() || skip.to_idx >= self.blocks.len() {
                continue;
            }

            let from_block = &self.blocks[skip.from_idx];
            let to_block = &self.blocks[skip.to_idx];

            let from_pos = Self::isometric_project([
                from_block.position[0] + from_block.size[0] / 2.0,
                from_block.position[1] + from_block.size[1],
                from_block.position[2],
            ]);
            let to_pos = Self::isometric_project([
                to_block.position[0] + to_block.size[0] / 2.0,
                to_block.position[1] + to_block.size[1],
                to_block.position[2],
            ]);

            let from_t = transform(from_pos);
            let to_t = transform(to_pos);

            // Curved path above the blocks
            let mid_y = from_t[1].min(to_t[1]) - 30.0 * scale;
            svg.bezier_curve(
                from_t[0],
                from_t[1],
                from_t[0],
                mid_y,
                to_t[0],
                mid_y,
                to_t[0],
                to_t[1],
                skip.color,
                skip.width,
                matches!(skip.style, SkipStyle::DashedLine),
            );

            // Arrow head
            svg.arrow_head(to_t[0], to_t[1] - 5.0, [to_t[0], mid_y], skip.color);
        }

        // Draw blocks (3D isometric boxes)
        for block in &self.blocks {
            self.draw_isometric_box(&mut svg, block, scale, offset_x, offset_y);
        }

        svg.build()
    }

    /// Draw a single 3D isometric box for a layer block
    fn draw_isometric_box(
        &self,
        svg: &mut SvgBuilder,
        block: &LayerBlock,
        scale: f32,
        offset_x: f32,
        offset_y: f32,
    ) {
        let pos = block.position;
        let size = block.size;

        // 8 corners of the box
        let corners_3d = [
            [pos[0], pos[1], pos[2]],                     // 0: front-bottom-left
            [pos[0] + size[0], pos[1], pos[2]],           // 1: front-bottom-right
            [pos[0] + size[0], pos[1] + size[1], pos[2]], // 2: front-top-right
            [pos[0], pos[1] + size[1], pos[2]],           // 3: front-top-left
            [pos[0], pos[1], pos[2] + size[2]],           // 4: back-bottom-left
            [pos[0] + size[0], pos[1], pos[2] + size[2]], // 5: back-bottom-right
            [pos[0] + size[0], pos[1] + size[1], pos[2] + size[2]], // 6: back-top-right
            [pos[0], pos[1] + size[1], pos[2] + size[2]], // 7: back-top-left
        ];

        // Project to 2D
        let corners_2d: Vec<[f32; 2]> = corners_3d
            .iter()
            .map(|p| {
                let proj = Self::isometric_project(*p);
                [proj[0] * scale + offset_x, proj[1] * scale + offset_y]
            })
            .collect();

        // Draw visible faces (back faces first, then front)
        // Top face (3, 2, 6, 7) - lighter color
        let top_color = lighten_color(block.color, 0.2);
        svg.polygon(
            &[corners_2d[3], corners_2d[2], corners_2d[6], corners_2d[7]],
            top_color,
            Some([0.3, 0.3, 0.3, 1.0]),
        );

        // Right face (1, 5, 6, 2) - slightly darker
        let right_color = darken_color(block.color, 0.1);
        svg.polygon(
            &[corners_2d[1], corners_2d[5], corners_2d[6], corners_2d[2]],
            right_color,
            Some([0.3, 0.3, 0.3, 1.0]),
        );

        // Front face (0, 1, 2, 3) - base color
        svg.polygon(
            &[corners_2d[0], corners_2d[1], corners_2d[2], corners_2d[3]],
            block.color,
            Some([0.2, 0.2, 0.2, 1.0]),
        );

        // Label on front face
        let label_x = (corners_2d[0][0] + corners_2d[1][0]) / 2.0;
        let label_y = (corners_2d[0][1] + corners_2d[3][1]) / 2.0;

        // Layer name
        svg.text(
            label_x,
            label_y - 8.0,
            &block.name,
            11.0 * scale.min(1.0).max(0.5),
            [0.1, 0.1, 0.1, 1.0],
            "middle",
        );

        // Layer type
        svg.text(
            label_x,
            label_y + 4.0,
            block.layer_type.label(),
            9.0 * scale.min(1.0).max(0.5),
            [0.3, 0.3, 0.3, 1.0],
            "middle",
        );

        // Shape info
        let shape_str = block.shape_string();
        if !shape_str.is_empty() {
            svg.text(
                label_x,
                label_y + 16.0,
                &shape_str,
                8.0 * scale.min(1.0).max(0.5),
                [0.4, 0.4, 0.4, 1.0],
                "middle",
            );
        }

        // Parameter count
        let params_str = block.params_string();
        if !params_str.is_empty() {
            svg.text(
                label_x,
                label_y + 28.0,
                &params_str,
                8.0 * scale.min(1.0).max(0.5),
                [0.5, 0.5, 0.5, 1.0],
                "middle",
            );
        }
    }

    /// Compute bounding box of the diagram in 2D projected space
    fn compute_bounds(&self) -> (f32, f32, f32, f32) {
        if self.blocks.is_empty() {
            return (0.0, 1.0, 0.0, 1.0);
        }

        let mut min_x = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        let mut min_y = f32::INFINITY;
        let mut max_y = f32::NEG_INFINITY;

        for block in &self.blocks {
            let corners = [
                [block.position[0], block.position[1], block.position[2]],
                [
                    block.position[0] + block.size[0],
                    block.position[1],
                    block.position[2],
                ],
                [
                    block.position[0] + block.size[0],
                    block.position[1] + block.size[1],
                    block.position[2],
                ],
                [
                    block.position[0],
                    block.position[1] + block.size[1],
                    block.position[2],
                ],
                [
                    block.position[0],
                    block.position[1],
                    block.position[2] + block.size[2],
                ],
                [
                    block.position[0] + block.size[0],
                    block.position[1],
                    block.position[2] + block.size[2],
                ],
                [
                    block.position[0] + block.size[0],
                    block.position[1] + block.size[1],
                    block.position[2] + block.size[2],
                ],
                [
                    block.position[0],
                    block.position[1] + block.size[1],
                    block.position[2] + block.size[2],
                ],
            ];

            for corner in corners {
                let proj = Self::isometric_project(corner);
                min_x = min_x.min(proj[0]);
                max_x = max_x.max(proj[0]);
                min_y = min_y.min(proj[1]);
                max_y = max_y.max(proj[1]);
            }
        }

        (min_x, max_x, min_y, max_y)
    }

    /// Render to a raster image (RGBA buffer)
    pub fn render_3d(&self, width: u32, height: u32) -> Vec<u8> {
        // For now, return a simple placeholder
        // Full 3D rendering would require a GPU library like wgpu
        let svg = self.to_svg(width, height);

        // Simple SVG to RGBA conversion placeholder
        // In production, use resvg or similar
        let pixels = width as usize * height as usize * 4;
        let mut buffer = vec![255u8; pixels];

        // Fill with background color
        for i in 0..width as usize * height as usize {
            buffer[i * 4] = (self.background_color[0] * 255.0) as u8;
            buffer[i * 4 + 1] = (self.background_color[1] * 255.0) as u8;
            buffer[i * 4 + 2] = (self.background_color[2] * 255.0) as u8;
            buffer[i * 4 + 3] = 255;
        }

        buffer
    }
}

/// Simple layer info for building diagrams
#[derive(Debug, Clone)]
pub struct LayerInfo {
    /// Layer name
    pub name: String,
    /// Layer type
    pub layer_type: LayerType,
    /// Tensor shape
    pub shape: Vec<usize>,
    /// Parameter count
    pub params: u64,
}

impl LayerInfo {
    /// Create new layer info
    pub fn new(name: impl Into<String>, layer_type: LayerType) -> Self {
        Self {
            name: name.into(),
            layer_type,
            shape: Vec::new(),
            params: 0,
        }
    }

    /// Set shape
    pub fn with_shape(mut self, shape: Vec<usize>) -> Self {
        self.shape = shape;
        self
    }

    /// Set params
    pub fn with_params(mut self, params: u64) -> Self {
        self.params = params;
        self
    }
}

/// Placeholder Tritter configuration for integration
#[derive(Debug, Clone)]
pub struct TritterConfig {
    /// Number of transformer layers
    pub num_layers: usize,
    /// Hidden dimension
    pub hidden_dim: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Sequence length
    pub seq_len: usize,
    /// Vocabulary size
    pub vocab_size: usize,
}

impl Default for TritterConfig {
    fn default() -> Self {
        Self {
            num_layers: 12,
            hidden_dim: 768,
            num_heads: 12,
            seq_len: 512,
            vocab_size: 50257,
        }
    }
}

// === SVG Builder Helper ===

struct SvgBuilder {
    width: u32,
    height: u32,
    elements: Vec<String>,
}

impl SvgBuilder {
    fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            elements: Vec::new(),
        }
    }

    fn color_to_css(color: [f32; 4]) -> String {
        format!(
            "rgba({},{},{},{})",
            (color[0] * 255.0) as u8,
            (color[1] * 255.0) as u8,
            (color[2] * 255.0) as u8,
            color[3]
        )
    }

    fn rect(&mut self, x: f32, y: f32, w: f32, h: f32, fill: [f32; 4], stroke: Option<[f32; 4]>) {
        let stroke_attr = stroke
            .map(|s| format!(" stroke=\"{}\" stroke-width=\"1\"", Self::color_to_css(s)))
            .unwrap_or_default();

        self.elements.push(format!(
            "<rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" fill=\"{}\"{}/>",
            x,
            y,
            w,
            h,
            Self::color_to_css(fill),
            stroke_attr
        ));
    }

    fn polygon(&mut self, points: &[[f32; 2]], fill: [f32; 4], stroke: Option<[f32; 4]>) {
        let points_str: String = points
            .iter()
            .map(|p| format!("{},{}", p[0], p[1]))
            .collect::<Vec<_>>()
            .join(" ");

        let stroke_attr = stroke
            .map(|s| format!(" stroke=\"{}\" stroke-width=\"1\"", Self::color_to_css(s)))
            .unwrap_or_default();

        self.elements.push(format!(
            "<polygon points=\"{}\" fill=\"{}\"{}/>",
            points_str,
            Self::color_to_css(fill),
            stroke_attr
        ));
    }

    fn line(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, color: [f32; 4], width: f32) {
        self.elements.push(format!(
            "<line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"{}\" stroke-width=\"{}\"/>",
            x1,
            y1,
            x2,
            y2,
            Self::color_to_css(color),
            width
        ));
    }

    fn bezier_curve(
        &mut self,
        x1: f32,
        y1: f32,
        cx1: f32,
        cy1: f32,
        cx2: f32,
        cy2: f32,
        x2: f32,
        y2: f32,
        color: [f32; 4],
        width: f32,
        dashed: bool,
    ) {
        let dash_attr = if dashed {
            " stroke-dasharray=\"5,3\""
        } else {
            ""
        };
        self.elements.push(format!(
            "<path d=\"M{},{} C{},{} {},{} {},{}\" fill=\"none\" stroke=\"{}\" stroke-width=\"{}\"{}/>",
            x1, y1, cx1, cy1, cx2, cy2, x2, y2,
            Self::color_to_css(color),
            width,
            dash_attr
        ));
    }

    fn arrow_head(&mut self, tip_x: f32, tip_y: f32, from: [f32; 2], color: [f32; 4]) {
        let dx = tip_x - from[0];
        let dy = tip_y - from[1];
        let len = (dx * dx + dy * dy).sqrt();
        if len < 0.001 {
            return;
        }

        let ux = dx / len;
        let uy = dy / len;

        let arrow_size = 6.0;
        let base_x = tip_x - ux * arrow_size;
        let base_y = tip_y - uy * arrow_size;

        let perp_x = -uy * arrow_size * 0.5;
        let perp_y = ux * arrow_size * 0.5;

        let points = [
            [tip_x, tip_y],
            [base_x + perp_x, base_y + perp_y],
            [base_x - perp_x, base_y - perp_y],
        ];

        self.polygon(&points, color, None);
    }

    fn text(&mut self, x: f32, y: f32, text: &str, size: f32, color: [f32; 4], anchor: &str) {
        // Escape HTML entities
        let escaped = text
            .replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;");

        self.elements.push(format!(
            "<text x=\"{}\" y=\"{}\" font-family=\"sans-serif\" font-size=\"{}\" fill=\"{}\" text-anchor=\"{}\">{}</text>",
            x, y, size,
            Self::color_to_css(color),
            anchor,
            escaped
        ));
    }

    fn build(self) -> String {
        let mut svg = format!(
            r#"<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}">
"#,
            self.width, self.height, self.width, self.height
        );

        for element in self.elements {
            svg.push_str("  ");
            svg.push_str(&element);
            svg.push('\n');
        }

        svg.push_str("</svg>\n");
        svg
    }
}

// === Color Helper Functions ===

/// Lighten a color by a factor (0.0 = no change, 1.0 = white)
fn lighten_color(color: [f32; 4], factor: f32) -> [f32; 4] {
    [
        color[0] + (1.0 - color[0]) * factor,
        color[1] + (1.0 - color[1]) * factor,
        color[2] + (1.0 - color[2]) * factor,
        color[3],
    ]
}

/// Darken a color by a factor (0.0 = no change, 1.0 = black)
fn darken_color(color: [f32; 4], factor: f32) -> [f32; 4] {
    [
        color[0] * (1.0 - factor),
        color[1] * (1.0 - factor),
        color[2] * (1.0 - factor),
        color[3],
    ]
}

/// Project a 3D point to 2D using isometric projection (standalone function)
pub fn isometric_project(point: [f32; 3]) -> [f32; 2] {
    ArchitectureDiagram::isometric_project(point)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_block_creation() {
        let block = LayerBlock::new("test_layer", LayerType::Attention)
            .with_shape(vec![512, 768])
            .with_params(2_500_000);

        assert_eq!(block.name, "test_layer");
        assert_eq!(block.layer_type, LayerType::Attention);
        assert_eq!(block.output_shape, vec![512, 768]);
        assert_eq!(block.params, 2_500_000);
        assert_eq!(block.params_string(), "2.5M");
        assert_eq!(block.shape_string(), "512 x 768");
    }

    #[test]
    fn test_isometric_projection() {
        let point = [1.0, 0.0, 0.0];
        let proj = isometric_project(point);

        // Should project to positive x
        assert!(proj[0] > 0.0);
    }

    #[test]
    fn test_diagram_creation() {
        let mut diagram = ArchitectureDiagram::new()
            .add_block(LayerBlock::new("embed", LayerType::Embedding).with_shape(vec![512, 768]))
            .add_block(LayerBlock::new("attn", LayerType::Attention).with_shape(vec![512, 768]))
            .add_block(LayerBlock::new("mlp", LayerType::MLP).with_shape(vec![512, 3072]))
            .with_title("Test Model");

        diagram.compute_layout();

        assert_eq!(diagram.blocks.len(), 3);
        assert!(diagram.title.is_some());
    }

    #[test]
    fn test_svg_generation() {
        let mut diagram = ArchitectureDiagram::new()
            .add_block(LayerBlock::new("layer1", LayerType::Embedding))
            .add_block(LayerBlock::new("layer2", LayerType::Attention));

        diagram.compute_layout();

        let svg = diagram.to_svg(800, 600);

        assert!(svg.contains("<svg"));
        assert!(svg.contains("</svg>"));
        assert!(svg.contains("layer1"));
        assert!(svg.contains("layer2"));
    }

    #[test]
    fn test_from_tritter_config() {
        let config = TritterConfig {
            num_layers: 2,
            hidden_dim: 256,
            num_heads: 4,
            seq_len: 128,
            vocab_size: 1000,
        };

        let diagram = ArchitectureDiagram::from_tritter_config(&config);

        // Should have: embed + 2*(ln1 + attn + ln2 + mlp) + final_ln + output
        // = 1 + 2*4 + 1 + 1 = 11 blocks
        assert!(diagram.blocks.len() >= 5);
        assert!(diagram.skip_connections.len() >= 2);
    }

    #[test]
    fn test_layer_type_colors() {
        let attn_color = LayerType::Attention.default_color();
        let mlp_color = LayerType::MLP.default_color();

        // Attention should be orange-ish
        assert!(attn_color[0] > attn_color[2]); // R > B

        // MLP should be green-ish
        assert!(mlp_color[1] > mlp_color[0]); // G > R
    }

    #[test]
    fn test_connection_types() {
        let conn = LayerConnection::skip(0, 5);
        assert_eq!(conn.connection_type, ConnectionType::Skip);

        let conn = LayerConnection::sequential(0, 1);
        assert_eq!(conn.connection_type, ConnectionType::Sequential);
    }

    #[test]
    fn test_params_formatting() {
        let block = LayerBlock::new("test", LayerType::MLP).with_params(1_234_567_890);
        assert_eq!(block.params_string(), "1.2B");

        let block = LayerBlock::new("test", LayerType::MLP).with_params(1_234_567);
        assert_eq!(block.params_string(), "1.2M");

        let block = LayerBlock::new("test", LayerType::MLP).with_params(1_234);
        assert_eq!(block.params_string(), "1.2K");

        let block = LayerBlock::new("test", LayerType::MLP).with_params(123);
        assert_eq!(block.params_string(), "123");
    }
}
