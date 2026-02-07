//! Chord Diagram Visualization for Attention Patterns
//!
//! Provides circular connectivity visualization similar to brain connectivity diagrams,
//! with:
//! - Circular layout with labeled segments (attention heads, layers, tokens)
//! - Curved bezier connections between segments
//! - Color gradient based on attention strength (blue-red diverging colormap)
//! - Segment labels around the circle
//! - Legend with colorbar
//!
//! # Example
//!
//! ```rust,ignore
//! use training_tools::viz3d::chord::{ChordDiagram, ChordSegment, ChordConnection};
//!
//! // Create from attention weights
//! let attention_weights = vec![
//!     vec![0.1, 0.4, 0.3, 0.2],
//!     vec![0.2, 0.1, 0.5, 0.2],
//!     vec![0.3, 0.2, 0.1, 0.4],
//!     vec![0.2, 0.3, 0.3, 0.2],
//! ];
//! let labels = vec!["Token A", "Token B", "Token C", "Token D"]
//!     .into_iter().map(String::from).collect();
//!
//! let diagram = ChordDiagram::from_attention(&attention_weights, &labels);
//! let svg = diagram.to_svg(800, 800);
//! ```

use std::f32::consts::PI;

use super::colors::{coolwarm, plasma, viridis, Color, Colormap, ColormapPreset};

/// A segment (arc) on the chord diagram circle.
#[derive(Debug, Clone)]
pub struct ChordSegment {
    /// Display label for this segment.
    pub label: String,
    /// Start angle in radians (0 = right, PI/2 = top).
    pub start_angle: f32,
    /// End angle in radians.
    pub end_angle: f32,
    /// Fill color for the segment arc.
    pub color: [f32; 4],
    /// Category for grouping (e.g., "Attention", "MLP", "Embedding").
    pub category: String,
}

impl ChordSegment {
    /// Create a new chord segment.
    pub fn new(
        label: impl Into<String>,
        start_angle: f32,
        end_angle: f32,
        category: impl Into<String>,
    ) -> Self {
        Self {
            label: label.into(),
            start_angle,
            end_angle,
            color: [0.5, 0.5, 0.5, 1.0], // Default gray
            category: category.into(),
        }
    }

    /// Set the segment color.
    pub fn with_color(mut self, color: [f32; 4]) -> Self {
        self.color = color;
        self
    }

    /// Get the midpoint angle of the segment.
    pub fn mid_angle(&self) -> f32 {
        (self.start_angle + self.end_angle) / 2.0
    }

    /// Get the angular span of the segment.
    pub fn angular_span(&self) -> f32 {
        self.end_angle - self.start_angle
    }
}

/// A connection (chord) between two segments.
#[derive(Debug, Clone)]
pub struct ChordConnection {
    /// Index of the source segment.
    pub source_idx: usize,
    /// Index of the target segment.
    pub target_idx: usize,
    /// Connection strength (0.0 to 1.0).
    pub strength: f32,
    /// Connection color (typically from colormap based on strength).
    pub color: [f32; 4],
}

impl ChordConnection {
    /// Create a new connection between segments.
    pub fn new(source_idx: usize, target_idx: usize, strength: f32) -> Self {
        Self {
            source_idx,
            target_idx,
            strength: strength.clamp(0.0, 1.0),
            color: [0.5, 0.5, 0.5, 0.5], // Default gray, semi-transparent
        }
    }

    /// Set the connection color.
    pub fn with_color(mut self, color: [f32; 4]) -> Self {
        self.color = color;
        self
    }

    /// Apply a colormap to set color based on strength.
    pub fn with_colormap(mut self, colormap: &Colormap) -> Self {
        let c = colormap.map(self.strength);
        self.color = [c.r, c.g, c.b, 0.6]; // Slightly transparent
        self
    }
}

/// Configuration for chord diagram rendering.
#[derive(Debug, Clone)]
pub struct ChordConfig {
    /// Outer radius of the diagram (normalized to 0-1 space).
    pub outer_radius: f32,
    /// Inner radius where connections attach.
    pub inner_radius: f32,
    /// Gap between segments in radians.
    pub segment_gap: f32,
    /// Colormap for connection strength.
    pub colormap: ColormapPreset,
    /// Minimum connection strength to display.
    pub min_strength_threshold: f32,
    /// Whether to show the legend.
    pub show_legend: bool,
    /// Whether to show segment labels.
    pub show_labels: bool,
    /// Font size for labels.
    pub label_font_size: f32,
    /// Background color.
    pub background_color: [f32; 4],
    /// Curvature factor for bezier connections (0.0 = straight, 1.0 = maximum curve).
    pub curvature: f32,
}

impl Default for ChordConfig {
    fn default() -> Self {
        Self {
            outer_radius: 0.42,
            inner_radius: 0.38,
            segment_gap: 0.02, // ~1 degree
            colormap: ColormapPreset::Coolwarm,
            min_strength_threshold: 0.05,
            show_legend: true,
            show_labels: true,
            label_font_size: 12.0,
            background_color: [1.0, 1.0, 1.0, 1.0],
            curvature: 0.5,
        }
    }
}

/// A complete chord diagram visualization.
#[derive(Debug, Clone)]
pub struct ChordDiagram {
    /// Segments arranged around the circle.
    pub segments: Vec<ChordSegment>,
    /// Connections between segments.
    pub connections: Vec<ChordConnection>,
    /// Diagram title.
    pub title: String,
    /// Rendering configuration.
    pub config: ChordConfig,
}

impl ChordDiagram {
    /// Create a new empty chord diagram.
    pub fn new(title: impl Into<String>) -> Self {
        Self {
            segments: Vec::new(),
            connections: Vec::new(),
            title: title.into(),
            config: ChordConfig::default(),
        }
    }

    /// Set the configuration.
    pub fn with_config(mut self, config: ChordConfig) -> Self {
        self.config = config;
        self
    }

    /// Add a segment to the diagram.
    pub fn add_segment(&mut self, segment: ChordSegment) {
        self.segments.push(segment);
    }

    /// Add a connection to the diagram.
    pub fn add_connection(&mut self, connection: ChordConnection) {
        self.connections.push(connection);
    }

    /// Create a chord diagram from attention weights matrix.
    ///
    /// # Arguments
    /// * `attention_weights` - Square matrix of attention weights [seq_len][seq_len].
    /// * `token_labels` - Labels for each token position.
    ///
    /// # Returns
    /// A chord diagram visualizing the attention pattern.
    pub fn from_attention(attention_weights: &[Vec<f32>], token_labels: &[String]) -> Self {
        let n = attention_weights.len();
        assert!(n > 0, "Attention weights must not be empty");
        assert_eq!(
            token_labels.len(),
            n,
            "Token labels must match attention matrix size"
        );

        let mut diagram = ChordDiagram::new("Attention Pattern");
        let colormap = diagram.config.colormap.colormap();
        let category_colors = category_palette(n);

        // Calculate segment angles with gaps
        let total_gap = diagram.config.segment_gap * n as f32;
        let available_angle = 2.0 * PI - total_gap;
        let angle_per_segment = available_angle / n as f32;

        let mut current_angle = -PI / 2.0; // Start at top

        // Create segments
        for i in 0..n {
            let start = current_angle;
            let end = current_angle + angle_per_segment;

            let segment = ChordSegment::new(token_labels[i].clone(), start, end, "Token")
                .with_color(category_colors[i % category_colors.len()]);

            diagram.add_segment(segment);
            current_angle = end + diagram.config.segment_gap;
        }

        // Create connections from attention weights
        for i in 0..n {
            for j in 0..n {
                let strength = attention_weights[i][j];
                if strength >= diagram.config.min_strength_threshold {
                    let connection = ChordConnection::new(i, j, strength).with_colormap(&colormap);
                    diagram.add_connection(connection);
                }
            }
        }

        diagram
    }

    /// Create a chord diagram from layer gradient connections.
    ///
    /// # Arguments
    /// * `layer_gradients` - Gradient magnitudes for each layer connection.
    ///
    /// # Returns
    /// A chord diagram visualizing gradient flow between layers.
    pub fn from_layer_connections(layer_gradients: &[f32]) -> Self {
        let n = ((1.0 + (1.0 + 8.0 * layer_gradients.len() as f32).sqrt()) / 2.0) as usize;
        let mut diagram = ChordDiagram::new("Layer Gradient Flow");
        let colormap = diagram.config.colormap.colormap();
        let category_colors = layer_category_colors();

        // Calculate segment angles
        let total_gap = diagram.config.segment_gap * n as f32;
        let available_angle = 2.0 * PI - total_gap;
        let angle_per_segment = available_angle / n as f32;

        let mut current_angle = -PI / 2.0;

        // Create layer segments
        for i in 0..n {
            let start = current_angle;
            let end = current_angle + angle_per_segment;

            let category = if i == 0 {
                "Embedding"
            } else if i == n - 1 {
                "Output"
            } else {
                "Hidden"
            };

            let color_idx = match category {
                "Embedding" => 0,
                "Output" => 2,
                _ => 1,
            };

            let segment = ChordSegment::new(format!("Layer {}", i), start, end, category)
                .with_color(category_colors[color_idx]);

            diagram.add_segment(segment);
            current_angle = end + diagram.config.segment_gap;
        }

        // Create connections
        let mut grad_idx = 0;
        for i in 0..n {
            for j in (i + 1)..n {
                if grad_idx < layer_gradients.len() {
                    let strength = layer_gradients[grad_idx].clamp(0.0, 1.0);
                    if strength >= diagram.config.min_strength_threshold {
                        let connection =
                            ChordConnection::new(i, j, strength).with_colormap(&colormap);
                        diagram.add_connection(connection);
                    }
                    grad_idx += 1;
                }
            }
        }

        diagram
    }

    /// Render the diagram to SVG format.
    ///
    /// # Arguments
    /// * `width` - Image width in pixels.
    /// * `height` - Image height in pixels.
    ///
    /// # Returns
    /// SVG string representation.
    pub fn to_svg(&self, width: u32, height: u32) -> String {
        let mut svg = String::new();

        // SVG header
        svg.push_str(&format!(
            r#"<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}">
"#,
            width, height, width, height
        ));

        // Background
        let bg = &self.config.background_color;
        svg.push_str(&format!(
            r#"  <rect width="100%" height="100%" fill="rgba({},{},{},{})"/>
"#,
            (bg[0] * 255.0) as u8,
            (bg[1] * 255.0) as u8,
            (bg[2] * 255.0) as u8,
            bg[3]
        ));

        // Calculate center and scale
        let cx = width as f32 / 2.0;
        let cy = height as f32 / 2.0;
        let scale = width.min(height) as f32;

        // Group for connections (drawn first, behind segments)
        svg.push_str("  <g id=\"connections\">\n");
        for conn in &self.connections {
            if conn.source_idx < self.segments.len() && conn.target_idx < self.segments.len() {
                let svg_conn = self.render_connection_svg(conn, cx, cy, scale);
                svg.push_str(&svg_conn);
            }
        }
        svg.push_str("  </g>\n");

        // Group for segments
        svg.push_str("  <g id=\"segments\">\n");
        for segment in &self.segments {
            let svg_seg = self.render_segment_svg(segment, cx, cy, scale);
            svg.push_str(&svg_seg);
        }
        svg.push_str("  </g>\n");

        // Labels
        if self.config.show_labels {
            svg.push_str("  <g id=\"labels\" font-family=\"sans-serif\">\n");
            for segment in &self.segments {
                let svg_label = self.render_label_svg(segment, cx, cy, scale);
                svg.push_str(&svg_label);
            }
            svg.push_str("  </g>\n");
        }

        // Title
        if !self.title.is_empty() {
            svg.push_str(&format!(
                r#"  <text x="{}" y="30" text-anchor="middle" font-family="sans-serif" font-size="18" font-weight="bold">{}</text>
"#,
                cx, escape_xml(&self.title)
            ));
        }

        // Legend
        if self.config.show_legend {
            let legend = self.render_legend_svg(width, height);
            svg.push_str(&legend);
        }

        svg.push_str("</svg>\n");
        svg
    }

    /// Render the diagram to a pixel buffer (RGBA format).
    ///
    /// # Arguments
    /// * `width` - Image width in pixels.
    /// * `height` - Image height in pixels.
    ///
    /// # Returns
    /// RGBA pixel buffer (width * height * 4 bytes).
    pub fn render(&self, width: u32, height: u32) -> Vec<u8> {
        let mut buffer = vec![0u8; (width * height * 4) as usize];

        // Fill background
        let bg = &self.config.background_color;
        for pixel in buffer.chunks_exact_mut(4) {
            pixel[0] = (bg[0] * 255.0) as u8;
            pixel[1] = (bg[1] * 255.0) as u8;
            pixel[2] = (bg[2] * 255.0) as u8;
            pixel[3] = (bg[3] * 255.0) as u8;
        }

        let cx = width as f32 / 2.0;
        let cy = height as f32 / 2.0;
        let scale = width.min(height) as f32;

        // Draw connections
        for conn in &self.connections {
            if conn.source_idx < self.segments.len() && conn.target_idx < self.segments.len() {
                self.render_connection_pixels(&mut buffer, width, height, conn, cx, cy, scale);
            }
        }

        // Draw segments
        for segment in &self.segments {
            self.render_segment_pixels(&mut buffer, width, height, segment, cx, cy, scale);
        }

        buffer
    }

    /// Calculate bezier control points for a connection curve.
    ///
    /// # Arguments
    /// * `start` - Start point [x, y].
    /// * `end` - End point [x, y].
    /// * `curvature` - Curvature factor (0.0 to 1.0).
    ///
    /// # Returns
    /// Vector of control points for cubic bezier: [start, control1, control2, end].
    pub fn bezier_connection(start: [f32; 2], end: [f32; 2], curvature: f32) -> Vec<[f32; 2]> {
        // Calculate the center point (for chord diagrams, this is typically the circle center)
        let mid_x = (start[0] + end[0]) / 2.0;
        let mid_y = (start[1] + end[1]) / 2.0;

        // Control points curve toward the center
        let curvature = curvature.clamp(0.0, 1.0);

        // For chord diagrams, we want the curve to pass through the center area
        // The control points are positioned between the endpoints and the center
        let ctrl1_x = start[0] * (1.0 - curvature) + mid_x * curvature;
        let ctrl1_y = start[1] * (1.0 - curvature) + mid_y * curvature;

        let ctrl2_x = end[0] * (1.0 - curvature) + mid_x * curvature;
        let ctrl2_y = end[1] * (1.0 - curvature) + mid_y * curvature;

        vec![start, [ctrl1_x, ctrl1_y], [ctrl2_x, ctrl2_y], end]
    }

    /// Get a point on a cubic bezier curve at parameter t.
    fn bezier_point(p0: [f32; 2], p1: [f32; 2], p2: [f32; 2], p3: [f32; 2], t: f32) -> [f32; 2] {
        let t2 = t * t;
        let t3 = t2 * t;
        let mt = 1.0 - t;
        let mt2 = mt * mt;
        let mt3 = mt2 * mt;

        [
            mt3 * p0[0] + 3.0 * mt2 * t * p1[0] + 3.0 * mt * t2 * p2[0] + t3 * p3[0],
            mt3 * p0[1] + 3.0 * mt2 * t * p1[1] + 3.0 * mt * t2 * p2[1] + t3 * p3[1],
        ]
    }

    // --- Private rendering helpers ---

    fn render_segment_svg(&self, segment: &ChordSegment, cx: f32, cy: f32, scale: f32) -> String {
        let outer_r = self.config.outer_radius * scale;
        let inner_r = self.config.inner_radius * scale;

        // Calculate arc points
        let start_outer = polar_to_cartesian(cx, cy, outer_r, segment.start_angle);
        let end_outer = polar_to_cartesian(cx, cy, outer_r, segment.end_angle);
        let start_inner = polar_to_cartesian(cx, cy, inner_r, segment.start_angle);
        let end_inner = polar_to_cartesian(cx, cy, inner_r, segment.end_angle);

        let large_arc = if segment.angular_span() > PI { 1 } else { 0 };

        let color = format!(
            "rgba({},{},{},{})",
            (segment.color[0] * 255.0) as u8,
            (segment.color[1] * 255.0) as u8,
            (segment.color[2] * 255.0) as u8,
            segment.color[3]
        );

        format!(
            r##"    <path d="M {:.1} {:.1} A {:.1} {:.1} 0 {} 1 {:.1} {:.1} L {:.1} {:.1} A {:.1} {:.1} 0 {} 0 {:.1} {:.1} Z" fill="{}" stroke="#333" stroke-width="1"/>
"##,
            start_outer[0],
            start_outer[1],
            outer_r,
            outer_r,
            large_arc,
            end_outer[0],
            end_outer[1],
            end_inner[0],
            end_inner[1],
            inner_r,
            inner_r,
            large_arc,
            start_inner[0],
            start_inner[1],
            color
        )
    }

    fn render_connection_svg(
        &self,
        conn: &ChordConnection,
        cx: f32,
        cy: f32,
        scale: f32,
    ) -> String {
        let source = &self.segments[conn.source_idx];
        let target = &self.segments[conn.target_idx];

        let inner_r = self.config.inner_radius * scale;

        // Connection attaches at segment midpoints on the inner circle
        let source_pt = polar_to_cartesian(cx, cy, inner_r, source.mid_angle());
        let target_pt = polar_to_cartesian(cx, cy, inner_r, target.mid_angle());

        // For self-connections, draw a small loop
        if conn.source_idx == conn.target_idx {
            let offset_r = inner_r * 0.3;
            let loop_ctrl = polar_to_cartesian(cx, cy, offset_r, source.mid_angle());

            let color = format!(
                "rgba({},{},{},{})",
                (conn.color[0] * 255.0) as u8,
                (conn.color[1] * 255.0) as u8,
                (conn.color[2] * 255.0) as u8,
                conn.color[3]
            );

            let stroke_width = 1.0 + conn.strength * 4.0;

            return format!(
                r#"    <path d="M {:.1} {:.1} Q {:.1} {:.1} {:.1} {:.1}" fill="none" stroke="{}" stroke-width="{:.1}" stroke-linecap="round"/>
"#,
                source_pt[0],
                source_pt[1],
                loop_ctrl[0],
                loop_ctrl[1],
                source_pt[0] + 1.0,
                source_pt[1] + 1.0,
                color,
                stroke_width
            );
        }

        // Calculate bezier control points - curve toward center
        let ctrl_points = Self::bezier_connection(source_pt, target_pt, self.config.curvature);

        let color = format!(
            "rgba({},{},{},{})",
            (conn.color[0] * 255.0) as u8,
            (conn.color[1] * 255.0) as u8,
            (conn.color[2] * 255.0) as u8,
            conn.color[3]
        );

        let stroke_width = 1.0 + conn.strength * 4.0;

        format!(
            r#"    <path d="M {:.1} {:.1} C {:.1} {:.1} {:.1} {:.1} {:.1} {:.1}" fill="none" stroke="{}" stroke-width="{:.1}" stroke-linecap="round"/>
"#,
            ctrl_points[0][0],
            ctrl_points[0][1],
            ctrl_points[1][0],
            ctrl_points[1][1],
            ctrl_points[2][0],
            ctrl_points[2][1],
            ctrl_points[3][0],
            ctrl_points[3][1],
            color,
            stroke_width
        )
    }

    fn render_label_svg(&self, segment: &ChordSegment, cx: f32, cy: f32, scale: f32) -> String {
        let label_r = (self.config.outer_radius + 0.05) * scale;
        let angle = segment.mid_angle();
        let pos = polar_to_cartesian(cx, cy, label_r, angle);

        // Determine text anchor based on position
        let (anchor, dx) = if angle.cos() > 0.1 {
            ("start", 5.0)
        } else if angle.cos() < -0.1 {
            ("end", -5.0)
        } else {
            ("middle", 0.0)
        };

        // Rotate text to follow the circle
        let rotation = angle.to_degrees() + 90.0;
        let rotation = if rotation > 90.0 && rotation < 270.0 {
            rotation + 180.0
        } else {
            rotation
        };

        format!(
            r#"    <text x="{:.1}" y="{:.1}" text-anchor="{}" font-size="{:.0}" transform="rotate({:.1} {:.1} {:.1})" dx="{:.0}">{}</text>
"#,
            pos[0],
            pos[1],
            anchor,
            self.config.label_font_size,
            rotation,
            pos[0],
            pos[1],
            dx,
            escape_xml(&segment.label)
        )
    }

    fn render_legend_svg(&self, width: u32, height: u32) -> String {
        let mut svg = String::new();

        // Legend position (bottom right)
        let legend_x = width as f32 - 150.0;
        let legend_y = height as f32 - 100.0;
        let bar_width = 120.0;
        let bar_height = 15.0;

        svg.push_str(&format!(
            r#"  <g id="legend" transform="translate({:.0}, {:.0})">
"#,
            legend_x, legend_y
        ));

        // Legend title
        svg.push_str(r#"    <text x="0" y="-5" font-family="sans-serif" font-size="12" font-weight="bold">Connection Strength</text>
"#);

        // Color bar gradient
        let colormap = self.config.colormap.colormap();
        let num_stops = 10;

        svg.push_str(&format!(
            r#"    <defs>
      <linearGradient id="legend_gradient" x1="0%" y1="0%" x2="100%" y2="0%">
"#
        ));

        for i in 0..=num_stops {
            let t = i as f32 / num_stops as f32;
            let color = colormap.map(t);
            svg.push_str(&format!(
                r#"        <stop offset="{}%" stop-color="rgb({},{},{})"/>
"#,
                (t * 100.0) as u32,
                (color.r * 255.0) as u8,
                (color.g * 255.0) as u8,
                (color.b * 255.0) as u8
            ));
        }

        svg.push_str(
            r#"      </linearGradient>
    </defs>
"#,
        );

        // Color bar
        svg.push_str(&format!(
            r##"    <rect x="0" y="5" width="{:.0}" height="{:.0}" fill="url(#legend_gradient)" stroke="#333" stroke-width="1"/>
"##,
            bar_width, bar_height
        ));

        // Labels
        svg.push_str(&format!(
            r#"    <text x="0" y="{:.0}" font-family="sans-serif" font-size="10" text-anchor="start">0.0</text>
    <text x="{:.0}" y="{:.0}" font-family="sans-serif" font-size="10" text-anchor="middle">0.5</text>
    <text x="{:.0}" y="{:.0}" font-family="sans-serif" font-size="10" text-anchor="end">1.0</text>
"#,
            bar_height + 20.0,
            bar_width / 2.0, bar_height + 20.0,
            bar_width, bar_height + 20.0
        ));

        svg.push_str("  </g>\n");
        svg
    }

    fn render_segment_pixels(
        &self,
        buffer: &mut [u8],
        width: u32,
        height: u32,
        segment: &ChordSegment,
        cx: f32,
        cy: f32,
        scale: f32,
    ) {
        let outer_r = self.config.outer_radius * scale;
        let inner_r = self.config.inner_radius * scale;
        let outer_r2 = outer_r * outer_r;
        let inner_r2 = inner_r * inner_r;

        // Scan relevant bounding box
        let min_x = ((cx - outer_r).max(0.0) as u32).saturating_sub(1);
        let max_x = ((cx + outer_r).min(width as f32 - 1.0) as u32) + 1;
        let min_y = ((cy - outer_r).max(0.0) as u32).saturating_sub(1);
        let max_y = ((cy + outer_r).min(height as f32 - 1.0) as u32) + 1;

        for y in min_y..max_y {
            for x in min_x..max_x {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                let dist2 = dx * dx + dy * dy;

                // Check if within ring
                if dist2 >= inner_r2 && dist2 <= outer_r2 {
                    // Check angle
                    let angle = dy.atan2(dx);
                    let normalized_angle = if angle < segment.start_angle {
                        angle + 2.0 * PI
                    } else {
                        angle
                    };

                    let in_segment = if segment.end_angle > segment.start_angle {
                        normalized_angle >= segment.start_angle
                            && normalized_angle <= segment.end_angle
                    } else {
                        normalized_angle >= segment.start_angle
                            || normalized_angle <= segment.end_angle
                    };

                    if in_segment {
                        let idx = ((y * width + x) * 4) as usize;
                        if idx + 3 < buffer.len() {
                            buffer[idx] = (segment.color[0] * 255.0) as u8;
                            buffer[idx + 1] = (segment.color[1] * 255.0) as u8;
                            buffer[idx + 2] = (segment.color[2] * 255.0) as u8;
                            buffer[idx + 3] = (segment.color[3] * 255.0) as u8;
                        }
                    }
                }
            }
        }
    }

    fn render_connection_pixels(
        &self,
        buffer: &mut [u8],
        width: u32,
        height: u32,
        conn: &ChordConnection,
        cx: f32,
        cy: f32,
        scale: f32,
    ) {
        let source = &self.segments[conn.source_idx];
        let target = &self.segments[conn.target_idx];

        let inner_r = self.config.inner_radius * scale;

        let source_pt = polar_to_cartesian(cx, cy, inner_r, source.mid_angle());
        let target_pt = polar_to_cartesian(cx, cy, inner_r, target.mid_angle());

        // Skip self-connections in pixel mode for simplicity
        if conn.source_idx == conn.target_idx {
            return;
        }

        let ctrl_points = Self::bezier_connection(source_pt, target_pt, self.config.curvature);

        // Sample points along the bezier curve
        let num_samples = 100;
        let line_width = (1.0 + conn.strength * 3.0).ceil() as i32;

        for i in 0..num_samples {
            let t = i as f32 / num_samples as f32;
            let pt = Self::bezier_point(
                ctrl_points[0],
                ctrl_points[1],
                ctrl_points[2],
                ctrl_points[3],
                t,
            );

            // Draw thick line by filling a small area around the point
            let px = pt[0] as i32;
            let py = pt[1] as i32;

            for dy in -line_width..=line_width {
                for dx in -line_width..=line_width {
                    let x = px + dx;
                    let y = py + dy;

                    if x >= 0 && x < width as i32 && y >= 0 && y < height as i32 {
                        let idx = ((y as u32 * width + x as u32) * 4) as usize;
                        if idx + 3 < buffer.len() {
                            // Alpha blending
                            let alpha = conn.color[3];
                            let inv_alpha = 1.0 - alpha;

                            buffer[idx] = ((conn.color[0] * alpha
                                + buffer[idx] as f32 / 255.0 * inv_alpha)
                                * 255.0) as u8;
                            buffer[idx + 1] = ((conn.color[1] * alpha
                                + buffer[idx + 1] as f32 / 255.0 * inv_alpha)
                                * 255.0) as u8;
                            buffer[idx + 2] = ((conn.color[2] * alpha
                                + buffer[idx + 2] as f32 / 255.0 * inv_alpha)
                                * 255.0) as u8;
                            buffer[idx + 3] = 255;
                        }
                    }
                }
            }
        }
    }
}

// --- Scientific colormaps as standalone functions ---

/// Viridis colormap function (blue-green-yellow, perceptually uniform).
///
/// # Arguments
/// * `t` - Value in range [0, 1].
///
/// # Returns
/// RGBA color array.
pub fn colormap_viridis(t: f32) -> [f32; 4] {
    let c = viridis().map(t);
    c.to_array()
}

/// Plasma colormap function (magenta-orange-yellow, perceptually uniform).
///
/// # Arguments
/// * `t` - Value in range [0, 1].
///
/// # Returns
/// RGBA color array.
pub fn colormap_plasma(t: f32) -> [f32; 4] {
    let c = plasma().map(t);
    c.to_array()
}

/// Coolwarm diverging colormap function (blue-white-red).
/// Ideal for attention patterns showing positive/negative correlations.
///
/// # Arguments
/// * `t` - Value in range [0, 1], where 0.5 is neutral.
///
/// # Returns
/// RGBA color array.
pub fn colormap_coolwarm(t: f32) -> [f32; 4] {
    let c = coolwarm().map(t);
    c.to_array()
}

// --- Helper functions ---

/// Convert polar coordinates to Cartesian.
fn polar_to_cartesian(cx: f32, cy: f32, r: f32, angle: f32) -> [f32; 2] {
    [cx + r * angle.cos(), cy + r * angle.sin()]
}

/// Escape XML special characters.
fn escape_xml(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

/// Generate a color palette for categories.
fn category_palette(n: usize) -> Vec<[f32; 4]> {
    let base_colors = vec![
        [0.122, 0.467, 0.706, 1.0], // Blue
        [1.0, 0.498, 0.055, 1.0],   // Orange
        [0.173, 0.627, 0.173, 1.0], // Green
        [0.839, 0.153, 0.157, 1.0], // Red
        [0.580, 0.404, 0.741, 1.0], // Purple
        [0.549, 0.337, 0.294, 1.0], // Brown
        [0.890, 0.467, 0.761, 1.0], // Pink
        [0.498, 0.498, 0.498, 1.0], // Gray
        [0.737, 0.741, 0.133, 1.0], // Olive
        [0.090, 0.745, 0.812, 1.0], // Cyan
    ];

    if n <= base_colors.len() {
        base_colors[..n].to_vec()
    } else {
        // Generate more colors using HSV rainbow
        (0..n)
            .map(|i| {
                let hue = i as f32 / n as f32;
                let (r, g, b) = hsv_to_rgb(hue, 0.7, 0.9);
                [r, g, b, 1.0]
            })
            .collect()
    }
}

/// Layer category colors.
fn layer_category_colors() -> Vec<[f32; 4]> {
    vec![
        [0.173, 0.627, 0.173, 1.0], // Embedding - Green
        [0.122, 0.467, 0.706, 1.0], // Hidden - Blue
        [0.839, 0.153, 0.157, 1.0], // Output - Red
    ]
}

/// Convert HSV to RGB.
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (f32, f32, f32) {
    let h = h * 6.0;
    let i = h.floor() as i32;
    let f = h - i as f32;
    let p = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));

    match i % 6 {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chord_segment_creation() {
        let segment = ChordSegment::new("Token A", 0.0, 0.5, "Token");
        assert_eq!(segment.label, "Token A");
        assert_eq!(segment.category, "Token");
        assert!((segment.mid_angle() - 0.25).abs() < 1e-6);
        assert!((segment.angular_span() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_chord_connection_creation() {
        let conn = ChordConnection::new(0, 1, 0.75);
        assert_eq!(conn.source_idx, 0);
        assert_eq!(conn.target_idx, 1);
        assert!((conn.strength - 0.75).abs() < 1e-6);
    }

    #[test]
    fn test_connection_strength_clamping() {
        let conn_high = ChordConnection::new(0, 1, 1.5);
        assert!((conn_high.strength - 1.0).abs() < 1e-6);

        let conn_low = ChordConnection::new(0, 1, -0.5);
        assert!(conn_low.strength.abs() < 1e-6);
    }

    #[test]
    fn test_from_attention() {
        let weights = vec![
            vec![0.5, 0.3, 0.2],
            vec![0.2, 0.5, 0.3],
            vec![0.3, 0.2, 0.5],
        ];
        let labels = vec!["A".to_string(), "B".to_string(), "C".to_string()];

        let diagram = ChordDiagram::from_attention(&weights, &labels);

        assert_eq!(diagram.segments.len(), 3);
        assert_eq!(diagram.title, "Attention Pattern");
        // With threshold 0.05, all connections should be present
        assert!(diagram.connections.len() > 0);
    }

    #[test]
    fn test_from_layer_connections() {
        // For 4 layers, we need 6 connections (4 choose 2)
        let gradients = vec![0.8, 0.6, 0.4, 0.7, 0.5, 0.3];
        let diagram = ChordDiagram::from_layer_connections(&gradients);

        assert_eq!(diagram.segments.len(), 4);
        assert!(diagram.connections.len() > 0);
    }

    #[test]
    fn test_svg_generation() {
        let weights = vec![vec![0.5, 0.3], vec![0.3, 0.5]];
        let labels = vec!["A".to_string(), "B".to_string()];

        let diagram = ChordDiagram::from_attention(&weights, &labels);
        let svg = diagram.to_svg(400, 400);

        assert!(svg.contains("<?xml"));
        assert!(svg.contains("<svg"));
        assert!(svg.contains("</svg>"));
        assert!(svg.contains("connections"));
        assert!(svg.contains("segments"));
    }

    #[test]
    fn test_pixel_buffer_generation() {
        let weights = vec![vec![0.5, 0.3], vec![0.3, 0.5]];
        let labels = vec!["A".to_string(), "B".to_string()];

        let diagram = ChordDiagram::from_attention(&weights, &labels);
        let buffer = diagram.render(100, 100);

        assert_eq!(buffer.len(), 100 * 100 * 4);
    }

    #[test]
    fn test_bezier_connection_points() {
        let start = [0.0, 0.0];
        let end = [100.0, 100.0];
        let curvature = 0.5;

        let points = ChordDiagram::bezier_connection(start, end, curvature);

        assert_eq!(points.len(), 4);
        assert_eq!(points[0], start);
        assert_eq!(points[3], end);
    }

    #[test]
    fn test_colormap_functions() {
        // Test viridis
        let c0 = colormap_viridis(0.0);
        let c1 = colormap_viridis(1.0);
        assert!(c0[2] > c0[0]); // Start is more blue
        assert!(c1[1] > c1[2]); // End is more yellow-green

        // Test coolwarm
        let cold = colormap_coolwarm(0.0);
        let hot = colormap_coolwarm(1.0);
        assert!(cold[2] > cold[0]); // Cold is blue
        assert!(hot[0] > hot[2]); // Hot is red
    }

    #[test]
    fn test_polar_to_cartesian() {
        let pt = polar_to_cartesian(50.0, 50.0, 10.0, 0.0);
        assert!((pt[0] - 60.0).abs() < 1e-6);
        assert!((pt[1] - 50.0).abs() < 1e-6);

        let pt2 = polar_to_cartesian(50.0, 50.0, 10.0, PI / 2.0);
        assert!((pt2[0] - 50.0).abs() < 1e-6);
        assert!((pt2[1] - 60.0).abs() < 1e-6);
    }

    #[test]
    fn test_xml_escaping() {
        assert_eq!(escape_xml("<test>"), "&lt;test&gt;");
        assert_eq!(escape_xml("a & b"), "a &amp; b");
    }

    #[test]
    fn test_custom_config() {
        let config = ChordConfig {
            outer_radius: 0.45,
            inner_radius: 0.40,
            min_strength_threshold: 0.1,
            colormap: ColormapPreset::Plasma,
            ..Default::default()
        };

        let diagram = ChordDiagram::new("Test").with_config(config);
        assert!((diagram.config.outer_radius - 0.45).abs() < 1e-6);
        assert!(matches!(diagram.config.colormap, ColormapPreset::Plasma));
    }

    #[test]
    fn test_connection_with_colormap() {
        let colormap = coolwarm();
        let conn = ChordConnection::new(0, 1, 0.0).with_colormap(&colormap);
        // Low strength should be blue
        assert!(conn.color[2] > conn.color[0]);

        let conn_high = ChordConnection::new(0, 1, 1.0).with_colormap(&colormap);
        // High strength should be red
        assert!(conn_high.color[0] > conn_high.color[2]);
    }
}
