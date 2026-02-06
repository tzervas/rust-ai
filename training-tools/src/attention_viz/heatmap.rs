//! 2D heat map generation for attention patterns.
//!
//! Provides heat map rendering with configurable color maps,
//! annotations, and output formats.

use super::{
    AttentionHead, AttentionViz, HeadAggregation, OutputFormat, Renderer, VizError, VizResult,
};
use serde::{Deserialize, Serialize};

/// Color map options for heat map rendering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ColorMap {
    /// Blue to white to red (diverging).
    BlueWhiteRed,
    /// White to blue (sequential).
    Blues,
    /// White to red (sequential).
    Reds,
    /// Viridis perceptually uniform color map.
    Viridis,
    /// Grayscale.
    Grayscale,
    /// Purple to orange (diverging).
    PurpleOrange,
}

impl Default for ColorMap {
    fn default() -> Self {
        Self::Viridis
    }
}

impl ColorMap {
    /// Get RGB color for a value in range [0, 1].
    pub fn get_color(&self, value: f32) -> (u8, u8, u8) {
        let v = value.clamp(0.0, 1.0);

        match self {
            ColorMap::BlueWhiteRed => {
                if v < 0.5 {
                    let t = v * 2.0;
                    ((255.0 * t) as u8, (255.0 * t) as u8, 255)
                } else {
                    let t = (v - 0.5) * 2.0;
                    (255, (255.0 * (1.0 - t)) as u8, (255.0 * (1.0 - t)) as u8)
                }
            }
            ColorMap::Blues => (
                (255.0 * (1.0 - v * 0.8)) as u8,
                (255.0 * (1.0 - v * 0.5)) as u8,
                255,
            ),
            ColorMap::Reds => (
                255,
                (255.0 * (1.0 - v * 0.8)) as u8,
                (255.0 * (1.0 - v * 0.8)) as u8,
            ),
            ColorMap::Viridis => {
                // Simplified viridis approximation
                let r = (68.0 + v * (253.0 - 68.0)) as u8;
                let g = (1.0 + v * (231.0 - 1.0)) as u8;
                let b = (84.0 + (1.0 - (v - 0.5).abs() * 2.0) * (150.0 - 84.0)) as u8;
                (r, g, b)
            }
            ColorMap::Grayscale => {
                let c = (255.0 * v) as u8;
                (c, c, c)
            }
            ColorMap::PurpleOrange => {
                if v < 0.5 {
                    let t = v * 2.0;
                    (
                        (128.0 + 127.0 * t) as u8,
                        (0.0 + 128.0 * t) as u8,
                        (128.0 + 127.0 * (1.0 - t)) as u8,
                    )
                } else {
                    let t = (v - 0.5) * 2.0;
                    (
                        255,
                        (128.0 + 127.0 * (1.0 - t)) as u8,
                        (255.0 * (1.0 - t)) as u8,
                    )
                }
            }
        }
    }

    /// Get an ASCII character representing a value in range [0, 1].
    pub fn get_ascii_char(&self, value: f32) -> char {
        let v = value.clamp(0.0, 1.0);
        const CHARS: &[char] = &[' ', '.', ':', '-', '=', '+', '*', '#', '@'];
        let idx = ((v * (CHARS.len() - 1) as f32) as usize).min(CHARS.len() - 1);
        CHARS[idx]
    }
}

/// Configuration for heat map rendering.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeatMapConfig {
    /// Color map to use.
    pub color_map: ColorMap,
    /// Whether to show value annotations.
    pub show_values: bool,
    /// Number of decimal places for value annotations.
    pub value_precision: usize,
    /// Whether to show axis labels.
    pub show_labels: bool,
    /// Title for the heat map.
    pub title: Option<String>,
    /// Minimum value for normalization (None = auto).
    pub vmin: Option<f32>,
    /// Maximum value for normalization (None = auto).
    pub vmax: Option<f32>,
    /// Cell width in characters (for ASCII output).
    pub cell_width: usize,
}

impl Default for HeatMapConfig {
    fn default() -> Self {
        Self {
            color_map: ColorMap::default(),
            show_values: false,
            value_precision: 2,
            show_labels: true,
            title: None,
            vmin: None,
            vmax: None,
            cell_width: 1,
        }
    }
}

/// A 2D heat map representation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeatMap {
    /// The data matrix.
    pub data: Vec<Vec<f32>>,
    /// Row labels.
    pub row_labels: Option<Vec<String>>,
    /// Column labels.
    pub col_labels: Option<Vec<String>>,
    /// Configuration.
    pub config: HeatMapConfig,
}

impl HeatMap {
    /// Create a new heat map from data.
    pub fn new(data: Vec<Vec<f32>>) -> VizResult<Self> {
        if data.is_empty() {
            return Err(VizError::EmptySequence);
        }

        let cols = data[0].len();
        for (i, row) in data.iter().enumerate() {
            if row.len() != cols {
                return Err(VizError::ShapeMismatch {
                    expected: format!("row {} to have {} columns", i, cols),
                    got: format!("{} columns", row.len()),
                });
            }
        }

        Ok(Self {
            data,
            row_labels: None,
            col_labels: None,
            config: HeatMapConfig::default(),
        })
    }

    /// Set row labels.
    pub fn with_row_labels(mut self, labels: Vec<String>) -> Self {
        self.row_labels = Some(labels);
        self
    }

    /// Set column labels.
    pub fn with_col_labels(mut self, labels: Vec<String>) -> Self {
        self.col_labels = Some(labels);
        self
    }

    /// Set configuration.
    pub fn with_config(mut self, config: HeatMapConfig) -> Self {
        self.config = config;
        self
    }

    /// Get the number of rows.
    pub fn num_rows(&self) -> usize {
        self.data.len()
    }

    /// Get the number of columns.
    pub fn num_cols(&self) -> usize {
        self.data.first().map(|r| r.len()).unwrap_or(0)
    }

    /// Get the minimum value in the data.
    pub fn min_value(&self) -> f32 {
        self.data
            .iter()
            .flat_map(|row| row.iter())
            .cloned()
            .fold(f32::INFINITY, f32::min)
    }

    /// Get the maximum value in the data.
    pub fn max_value(&self) -> f32 {
        self.data
            .iter()
            .flat_map(|row| row.iter())
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max)
    }

    /// Normalize a value to [0, 1] range.
    fn normalize(&self, value: f32) -> f32 {
        let vmin = self.config.vmin.unwrap_or_else(|| self.min_value());
        let vmax = self.config.vmax.unwrap_or_else(|| self.max_value());

        if (vmax - vmin).abs() < f32::EPSILON {
            0.5
        } else {
            (value - vmin) / (vmax - vmin)
        }
    }
}

impl Renderer for HeatMap {
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

impl HeatMap {
    /// Render as ASCII art.
    pub fn render_ascii(&self) -> VizResult<String> {
        let mut output = String::new();

        // Title
        if let Some(ref title) = self.config.title {
            output.push_str(title);
            output.push('\n');
            output.push_str(&"-".repeat(title.len()));
            output.push_str("\n\n");
        }

        // Calculate label widths
        let row_label_width = self
            .row_labels
            .as_ref()
            .map(|labels| labels.iter().map(|l| l.len()).max().unwrap_or(0))
            .unwrap_or(0);

        // Column labels
        if self.config.show_labels {
            if let Some(ref col_labels) = self.col_labels {
                output.push_str(&" ".repeat(row_label_width + 1));
                for label in col_labels {
                    let display = if label.len() > self.config.cell_width {
                        &label[..self.config.cell_width]
                    } else {
                        label
                    };
                    output.push_str(&format!(
                        "{:^width$}",
                        display,
                        width = self.config.cell_width
                    ));
                }
                output.push('\n');
            }
        }

        // Data rows
        for (i, row) in self.data.iter().enumerate() {
            // Row label
            if self.config.show_labels {
                if let Some(ref row_labels) = self.row_labels {
                    if let Some(label) = row_labels.get(i) {
                        output.push_str(&format!("{:>width$} ", label, width = row_label_width));
                    }
                } else {
                    output.push_str(&" ".repeat(row_label_width + 1));
                }
            }

            // Data cells
            for &value in row {
                let normalized = self.normalize(value);
                let c = self.config.color_map.get_ascii_char(normalized);
                for _ in 0..self.config.cell_width {
                    output.push(c);
                }
            }
            output.push('\n');
        }

        // Legend
        output.push('\n');
        output.push_str("Legend: ");
        const CHARS: &[char] = &[' ', '.', ':', '-', '=', '+', '*', '#', '@'];
        let vmin = self.config.vmin.unwrap_or_else(|| self.min_value());
        let vmax = self.config.vmax.unwrap_or_else(|| self.max_value());
        output.push_str(&format!("[{:.2} ", vmin));
        for c in CHARS {
            output.push(*c);
        }
        output.push_str(&format!(" {:.2}]", vmax));
        output.push('\n');

        Ok(output)
    }

    /// Render as SVG.
    pub fn render_svg(&self) -> VizResult<String> {
        let cell_size = 20;
        let margin = 50;
        let width = self.num_cols() * cell_size + margin * 2;
        let height = self.num_rows() * cell_size + margin * 2;

        let mut svg = format!(
            r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {} {}">"#,
            width, height
        );

        // Title
        if let Some(ref title) = self.config.title {
            svg.push_str(&format!(
                r#"<text x="{}" y="20" text-anchor="middle" font-size="14" font-weight="bold">{}</text>"#,
                width / 2,
                title
            ));
        }

        // Draw cells
        for (i, row) in self.data.iter().enumerate() {
            for (j, &value) in row.iter().enumerate() {
                let normalized = self.normalize(value);
                let (r, g, b) = self.config.color_map.get_color(normalized);
                let x = margin + j * cell_size;
                let y = margin + i * cell_size;

                svg.push_str(&format!(
                    "<rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" fill=\"rgb({},{},{})\" stroke=\"#ccc\" stroke-width=\"0.5\"/>",
                    x, y, cell_size, cell_size, r, g, b
                ));

                // Value annotation
                if self.config.show_values {
                    svg.push_str(&format!(
                        r#"<text x="{}" y="{}" text-anchor="middle" font-size="8" fill="{}">{:.prec$}</text>"#,
                        x + cell_size / 2,
                        y + cell_size / 2 + 3,
                        if normalized > 0.5 { "white" } else { "black" },
                        value,
                        prec = self.config.value_precision
                    ));
                }
            }
        }

        // Row labels
        if self.config.show_labels {
            if let Some(ref row_labels) = self.row_labels {
                for (i, label) in row_labels.iter().enumerate() {
                    let y = margin + i * cell_size + cell_size / 2 + 4;
                    svg.push_str(&format!(
                        r#"<text x="{}" y="{}" text-anchor="end" font-size="10">{}</text>"#,
                        margin - 5,
                        y,
                        label
                    ));
                }
            }
        }

        // Column labels
        if self.config.show_labels {
            if let Some(ref col_labels) = self.col_labels {
                for (j, label) in col_labels.iter().enumerate() {
                    let x = margin + j * cell_size + cell_size / 2;
                    svg.push_str(&format!(
                        r#"<text x="{}" y="{}" text-anchor="middle" font-size="10" transform="rotate(-45 {} {})">{}</text>"#,
                        x,
                        margin - 5,
                        x,
                        margin - 5,
                        label
                    ));
                }
            }
        }

        svg.push_str("</svg>");
        Ok(svg)
    }

    /// Render as HTML with embedded SVG.
    pub fn render_html(&self) -> VizResult<String> {
        let svg = self.render_svg()?;
        let title = self.config.title.as_deref().unwrap_or("Heat Map");

        Ok(format!(
            r#"<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{}</title>
    <style>
        body {{ font-family: sans-serif; margin: 20px; }}
        .container {{ max-width: 800px; margin: 0 auto; }}
        svg {{ max-width: 100%; height: auto; }}
    </style>
</head>
<body>
    <div class="container">
        {}
    </div>
</body>
</html>"#,
            title, svg
        ))
    }

    /// Render as JSON.
    pub fn render_json(&self) -> VizResult<String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| VizError::RenderError(format!("JSON serialization failed: {}", e)))
    }
}

/// Heat map renderer with additional utilities.
pub struct HeatMapRenderer;

impl HeatMapRenderer {
    /// Create a heat map from an attention head.
    pub fn from_attention_head(head: &AttentionHead, config: HeatMapConfig) -> VizResult<HeatMap> {
        let data = head.weights.clone();
        let seq_len = head.seq_len();
        let labels: Vec<String> = (0..seq_len).map(|i| i.to_string()).collect();

        Ok(HeatMap::new(data)?
            .with_row_labels(labels.clone())
            .with_col_labels(labels)
            .with_config(config))
    }

    /// Create a heat map from aggregated attention visualization.
    pub fn from_attention_viz(
        viz: &AttentionViz,
        aggregation: HeadAggregation,
        config: HeatMapConfig,
    ) -> VizResult<HeatMap> {
        let data = viz.aggregate(aggregation);
        let seq_len = viz.seq_len();

        let labels = viz
            .tokens
            .clone()
            .unwrap_or_else(|| (0..seq_len).map(|i| i.to_string()).collect());

        Ok(HeatMap::new(data)?
            .with_row_labels(labels.clone())
            .with_col_labels(labels)
            .with_config(config))
    }

    /// Create a grid of heat maps, one per attention head.
    pub fn head_grid(viz: &AttentionViz, config: HeatMapConfig) -> VizResult<Vec<HeatMap>> {
        viz.heads
            .iter()
            .map(|head| {
                let mut head_config = config.clone();
                head_config.title = Some(format!("Head {}", head.head_idx));
                Self::from_attention_head(head, head_config)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_data() -> Vec<Vec<f32>> {
        vec![
            vec![0.0, 0.5, 1.0],
            vec![0.3, 0.6, 0.9],
            vec![0.1, 0.4, 0.7],
        ]
    }

    #[test]
    fn test_heat_map_creation() {
        let hm = HeatMap::new(sample_data()).unwrap();
        assert_eq!(hm.num_rows(), 3);
        assert_eq!(hm.num_cols(), 3);
    }

    #[test]
    fn test_heat_map_stats() {
        let hm = HeatMap::new(sample_data()).unwrap();
        assert!((hm.min_value() - 0.0).abs() < 0.01);
        assert!((hm.max_value() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_color_map_get_color() {
        let cm = ColorMap::Grayscale;
        assert_eq!(cm.get_color(0.0), (0, 0, 0));
        assert_eq!(cm.get_color(1.0), (255, 255, 255));
    }

    #[test]
    fn test_ascii_char() {
        let cm = ColorMap::Viridis;
        assert_eq!(cm.get_ascii_char(0.0), ' ');
        assert_eq!(cm.get_ascii_char(1.0), '@');
    }

    #[test]
    fn test_render_ascii() {
        let hm = HeatMap::new(sample_data()).unwrap();
        let ascii = hm.render_ascii().unwrap();
        assert!(!ascii.is_empty());
        assert!(ascii.contains("Legend"));
    }

    #[test]
    fn test_render_svg() {
        let hm = HeatMap::new(sample_data())
            .unwrap()
            .with_config(HeatMapConfig {
                title: Some("Test".to_string()),
                ..Default::default()
            });
        let svg = hm.render_svg().unwrap();
        assert!(svg.starts_with("<svg"));
        assert!(svg.contains("Test"));
    }

    #[test]
    fn test_render_html() {
        let hm = HeatMap::new(sample_data()).unwrap();
        let html = hm.render_html().unwrap();
        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("<svg"));
    }

    #[test]
    fn test_render_json() {
        let hm = HeatMap::new(sample_data()).unwrap();
        let json = hm.render_json().unwrap();
        assert!(json.contains("\"data\""));
    }

    #[test]
    fn test_renderer_trait() {
        let hm = HeatMap::new(sample_data()).unwrap();
        let result = hm.render(OutputFormat::Ascii);
        assert!(result.is_ok());
    }
}
