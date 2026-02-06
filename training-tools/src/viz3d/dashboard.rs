//! Unified Visualization Dashboard
//!
//! A multi-panel dashboard system that combines different visualization types
//! for comprehensive neural network analysis and training monitoring.
//!
//! # Features
//!
//! - **Multi-panel layout**: Flexible grid-based arrangement of panels
//! - **Multiple visualization types**: 3D network, attention heatmaps, metrics charts, etc.
//! - **Synchronized selection**: Click on elements in one panel to highlight in others
//! - **Real-time updates**: Support for streaming training data
//! - **High-resolution export**: PNG and SVG output for publications
//!
//! # Example
//!
//! ```rust,ignore
//! use training_tools::viz3d::dashboard::{
//!     VisualizationDashboard, DashboardLayout, DashboardPanel,
//!     PanelType, PanelPosition, PanelSize,
//! };
//!
//! // Create a dashboard with 3x2 grid
//! let layout = DashboardLayout::new(3, 2);
//! let mut dashboard = VisualizationDashboard::new(layout);
//!
//! // Add a 3D network view in the center
//! dashboard.add_panel(DashboardPanel::new(
//!     "network_3d",
//!     PanelType::Network3D,
//!     PanelPosition::new(0, 0),
//!     PanelSize::new(2, 2),
//! ));
//!
//! // Add attention heatmap on the side
//! dashboard.add_panel(DashboardPanel::new(
//!     "attention",
//!     PanelType::AttentionHeatmap,
//!     PanelPosition::new(2, 0),
//!     PanelSize::new(1, 1),
//! ));
//!
//! // Update and render
//! dashboard.update(0.016); // 60fps delta
//! let pixels = dashboard.render(1920, 1080);
//! ```

use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use super::colors::{Color, Colormap, ColormapPreset};
use super::engine::{Camera3D, Mesh3D, Viz3DEngine};

/// Errors that can occur during dashboard operations.
#[derive(Debug, thiserror::Error)]
pub enum DashboardError {
    #[error("Panel not found: {0}")]
    PanelNotFound(String),

    #[error("Invalid layout: {0}")]
    InvalidLayout(String),

    #[error("Panel overlap at position ({row}, {col})")]
    PanelOverlap { row: usize, col: usize },

    #[error("Panel out of bounds: position ({row}, {col}) exceeds grid ({max_row}, {max_col})")]
    OutOfBounds {
        row: usize,
        col: usize,
        max_row: usize,
        max_col: usize,
    },

    #[error("Render error: {0}")]
    RenderError(String),

    #[error("Export error: {0}")]
    ExportError(String),

    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),
}

/// Result type for dashboard operations.
pub type Result<T> = std::result::Result<T, DashboardError>;

// ============================================================================
// Core Dashboard Types
// ============================================================================

/// Types of visualization panels supported by the dashboard.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PanelType {
    /// 3D neural network visualization with layers and connections.
    Network3D,
    /// 2D attention pattern heatmap.
    AttentionHeatmap,
    /// Chord diagram showing token-to-token attention flow.
    ChordDiagram,
    /// 3D point cloud of embeddings (PCA/t-SNE/UMAP projected).
    EmbeddingCloud,
    /// 3D loss landscape surface with trajectory.
    LossLandscape,
    /// Line/area chart for training metrics (loss, accuracy, etc.).
    MetricsChart,
    /// Architectural diagram showing model structure.
    ArchitectureDiagram,
    /// 3D token/word cloud visualization.
    TokenCloud,
    /// Layer activation histograms and statistics.
    LayerActivations,
    /// Gradient flow visualization through layers.
    GradientFlow,
    /// Custom panel with user-defined content.
    Custom,
}

impl Default for PanelType {
    fn default() -> Self {
        Self::MetricsChart
    }
}

/// Position of a panel in the grid (row, col).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PanelPosition {
    /// Row index (0-based, from top).
    pub row: usize,
    /// Column index (0-based, from left).
    pub col: usize,
}

impl PanelPosition {
    /// Create a new panel position.
    pub fn new(col: usize, row: usize) -> Self {
        Self { row, col }
    }

    /// Position at origin (0, 0).
    pub fn origin() -> Self {
        Self { row: 0, col: 0 }
    }
}

impl Default for PanelPosition {
    fn default() -> Self {
        Self::origin()
    }
}

/// Size of a panel in grid cells (width, height).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PanelSize {
    /// Width in grid columns.
    pub cols: usize,
    /// Height in grid rows.
    pub rows: usize,
}

impl PanelSize {
    /// Create a new panel size.
    pub fn new(cols: usize, rows: usize) -> Self {
        Self { cols, rows }
    }

    /// A single cell (1x1).
    pub fn unit() -> Self {
        Self { cols: 1, rows: 1 }
    }

    /// Total number of cells occupied.
    pub fn cell_count(&self) -> usize {
        self.cols * self.rows
    }
}

impl Default for PanelSize {
    fn default() -> Self {
        Self::unit()
    }
}

// ============================================================================
// Panel Content Types
// ============================================================================

/// Content data for different panel types.
#[derive(Debug, Clone)]
pub enum PanelContent {
    /// 3D neural network scene.
    Neural3D(Neural3DScene),
    /// Attention heatmap data.
    Heatmap(HeatmapView),
    /// Chord diagram data.
    Chord(ChordDiagramData),
    /// Embedding point cloud.
    Embedding(EmbeddingCloudData),
    /// Loss landscape surface.
    Landscape(LossLandscapeData),
    /// Metrics time series chart.
    Chart(ChartView),
    /// Architecture diagram.
    Architecture(ArchitectureDiagram),
    /// Token cloud.
    TokenCloud(TokenCloudData),
    /// Layer activations.
    Activations(LayerActivationData),
    /// Gradient flow.
    GradientFlow(GradientFlowData),
    /// Empty placeholder.
    Empty,
}

impl Default for PanelContent {
    fn default() -> Self {
        Self::Empty
    }
}

/// 3D neural network scene data.
#[derive(Debug, Clone)]
pub struct Neural3DScene {
    /// Layer representations as meshes.
    pub layers: Vec<LayerMeshData>,
    /// Connections between layers.
    pub connections: Vec<ConnectionData>,
    /// Current animation time.
    pub animation_time: f32,
    /// Camera position override.
    pub camera: Option<Camera3D>,
    /// Whether to show gradient heatmap.
    pub show_gradients: bool,
    /// Whether to show activation flow.
    pub show_activations: bool,
}

impl Default for Neural3DScene {
    fn default() -> Self {
        Self {
            layers: Vec::new(),
            connections: Vec::new(),
            animation_time: 0.0,
            camera: None,
            show_gradients: true,
            show_activations: true,
        }
    }
}

/// Data for a single layer mesh.
#[derive(Debug, Clone)]
pub struct LayerMeshData {
    /// Layer name.
    pub name: String,
    /// Layer type identifier.
    pub layer_type: String,
    /// Position in 3D space.
    pub position: [f32; 3],
    /// Size/dimensions of the layer representation.
    pub size: [f32; 3],
    /// Color for the layer.
    pub color: [f32; 4],
    /// Gradient magnitude for this layer.
    pub gradient_magnitude: f32,
    /// Mean activation value.
    pub mean_activation: f32,
    /// Whether this layer is selected.
    pub selected: bool,
}

impl Default for LayerMeshData {
    fn default() -> Self {
        Self {
            name: String::new(),
            layer_type: "dense".to_string(),
            position: [0.0, 0.0, 0.0],
            size: [1.0, 1.0, 1.0],
            color: [0.5, 0.5, 0.8, 1.0],
            gradient_magnitude: 0.0,
            mean_activation: 0.0,
            selected: false,
        }
    }
}

/// Data for a connection between layers.
#[derive(Debug, Clone)]
pub struct ConnectionData {
    /// Source layer name.
    pub from_layer: String,
    /// Target layer name.
    pub to_layer: String,
    /// Connection strength (weight magnitude).
    pub strength: f32,
    /// Flow direction (-1.0 to 1.0, negative = backward pass).
    pub flow_direction: f32,
    /// Whether this connection is highlighted.
    pub highlighted: bool,
}

impl Default for ConnectionData {
    fn default() -> Self {
        Self {
            from_layer: String::new(),
            to_layer: String::new(),
            strength: 1.0,
            flow_direction: 1.0,
            highlighted: false,
        }
    }
}

/// Heatmap visualization data.
#[derive(Debug, Clone)]
pub struct HeatmapView {
    /// 2D data grid [row][col].
    pub data: Vec<Vec<f32>>,
    /// Row labels (e.g., query tokens).
    pub row_labels: Option<Vec<String>>,
    /// Column labels (e.g., key tokens).
    pub col_labels: Option<Vec<String>>,
    /// Title for the heatmap.
    pub title: String,
    /// Colormap to use.
    pub colormap: ColormapPreset,
    /// Value range (min, max). Auto-computed if None.
    pub value_range: Option<(f32, f32)>,
    /// Selected cells (row, col).
    pub selected_cells: Vec<(usize, usize)>,
    /// Whether to show values in cells.
    pub show_values: bool,
}

impl Default for HeatmapView {
    fn default() -> Self {
        Self {
            data: Vec::new(),
            row_labels: None,
            col_labels: None,
            title: "Heatmap".to_string(),
            colormap: ColormapPreset::Viridis,
            value_range: None,
            selected_cells: Vec::new(),
            show_values: false,
        }
    }
}

impl HeatmapView {
    /// Create a heatmap from attention weights.
    pub fn from_attention(weights: &[Vec<f32>], tokens: Option<&[String]>) -> Self {
        Self {
            data: weights.to_vec(),
            row_labels: tokens.map(|t| t.to_vec()),
            col_labels: tokens.map(|t| t.to_vec()),
            title: "Attention Weights".to_string(),
            colormap: ColormapPreset::Plasma,
            value_range: Some((0.0, 1.0)),
            selected_cells: Vec::new(),
            show_values: false,
        }
    }

    /// Get computed value range.
    pub fn computed_range(&self) -> (f32, f32) {
        if let Some(range) = self.value_range {
            return range;
        }

        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;

        for row in &self.data {
            for &val in row {
                min = min.min(val);
                max = max.max(val);
            }
        }

        if min.is_infinite() || max.is_infinite() {
            (0.0, 1.0)
        } else {
            (min, max)
        }
    }

    /// Get dimensions (rows, cols).
    pub fn dimensions(&self) -> (usize, usize) {
        let rows = self.data.len();
        let cols = self.data.first().map(|r| r.len()).unwrap_or(0);
        (rows, cols)
    }
}

/// Chord diagram data for attention flow visualization.
#[derive(Debug, Clone)]
pub struct ChordDiagramData {
    /// Node labels.
    pub nodes: Vec<String>,
    /// Connection matrix [from][to] = weight.
    pub connections: Vec<Vec<f32>>,
    /// Node colors.
    pub node_colors: Vec<[f32; 4]>,
    /// Highlighted connections.
    pub highlighted: Vec<(usize, usize)>,
    /// Whether to show self-connections.
    pub show_self_connections: bool,
    /// Minimum weight to display.
    pub threshold: f32,
}

impl Default for ChordDiagramData {
    fn default() -> Self {
        Self {
            nodes: Vec::new(),
            connections: Vec::new(),
            node_colors: Vec::new(),
            highlighted: Vec::new(),
            show_self_connections: false,
            threshold: 0.01,
        }
    }
}

/// Embedding point cloud data.
#[derive(Debug, Clone)]
pub struct EmbeddingCloudData {
    /// 3D points (after projection).
    pub points: Vec<[f32; 3]>,
    /// Point labels.
    pub labels: Vec<String>,
    /// Point colors.
    pub colors: Vec<[f32; 4]>,
    /// Point sizes (based on frequency/importance).
    pub sizes: Vec<f32>,
    /// Cluster assignments (-1 for unclustered).
    pub clusters: Vec<i32>,
    /// Selected point indices.
    pub selected: Vec<usize>,
    /// Projection method used.
    pub projection_method: String,
}

impl Default for EmbeddingCloudData {
    fn default() -> Self {
        Self {
            points: Vec::new(),
            labels: Vec::new(),
            colors: Vec::new(),
            sizes: Vec::new(),
            clusters: Vec::new(),
            selected: Vec::new(),
            projection_method: "PCA".to_string(),
        }
    }
}

/// Loss landscape surface data.
#[derive(Debug, Clone)]
pub struct LossLandscapeData {
    /// 2D surface heights [y][x].
    pub surface: Vec<Vec<f32>>,
    /// X-axis range (min, max).
    pub x_range: (f32, f32),
    /// Y-axis range (min, max).
    pub y_range: (f32, f32),
    /// Optimization trajectory points [x, y, z].
    pub trajectory: Vec<[f32; 3]>,
    /// Current position marker.
    pub current_position: Option<[f32; 3]>,
    /// Surface colormap.
    pub colormap: ColormapPreset,
    /// Whether to show contour lines.
    pub show_contours: bool,
    /// Whether to show trajectory.
    pub show_trajectory: bool,
}

impl Default for LossLandscapeData {
    fn default() -> Self {
        Self {
            surface: Vec::new(),
            x_range: (-1.0, 1.0),
            y_range: (-1.0, 1.0),
            trajectory: Vec::new(),
            current_position: None,
            colormap: ColormapPreset::Coolwarm,
            show_contours: true,
            show_trajectory: true,
        }
    }
}

/// Chart view for metrics visualization.
#[derive(Debug, Clone)]
pub struct ChartView {
    /// Data series.
    pub series: Vec<ChartSeries>,
    /// X-axis label.
    pub x_label: String,
    /// Y-axis label.
    pub y_label: String,
    /// Chart title.
    pub title: String,
    /// Whether to use logarithmic Y-axis.
    pub log_y: bool,
    /// X-axis range (auto if None).
    pub x_range: Option<(f32, f32)>,
    /// Y-axis range (auto if None).
    pub y_range: Option<(f32, f32)>,
    /// Whether to show legend.
    pub show_legend: bool,
    /// Whether to show grid lines.
    pub show_grid: bool,
    /// Vertical markers (x positions).
    pub markers: Vec<ChartMarker>,
}

impl Default for ChartView {
    fn default() -> Self {
        Self {
            series: Vec::new(),
            x_label: "Step".to_string(),
            y_label: "Value".to_string(),
            title: "Training Metrics".to_string(),
            log_y: false,
            x_range: None,
            y_range: None,
            show_legend: true,
            show_grid: true,
            markers: Vec::new(),
        }
    }
}

impl ChartView {
    /// Add a data series to the chart.
    pub fn add_series(&mut self, series: ChartSeries) {
        self.series.push(series);
    }

    /// Add a marker at a specific x position.
    pub fn add_marker(&mut self, x: f32, label: &str, color: [f32; 4]) {
        self.markers.push(ChartMarker {
            x,
            label: label.to_string(),
            color,
        });
    }

    /// Compute auto ranges from all series.
    pub fn compute_ranges(&self) -> ((f32, f32), (f32, f32)) {
        let mut x_min = f32::INFINITY;
        let mut x_max = f32::NEG_INFINITY;
        let mut y_min = f32::INFINITY;
        let mut y_max = f32::NEG_INFINITY;

        for series in &self.series {
            for &(x, y) in &series.data {
                x_min = x_min.min(x);
                x_max = x_max.max(x);
                y_min = y_min.min(y);
                y_max = y_max.max(y);
            }
        }

        let x_range = if x_min.is_infinite() {
            (0.0, 1.0)
        } else {
            (x_min, x_max)
        };

        let y_range = if y_min.is_infinite() {
            (0.0, 1.0)
        } else {
            // Add some padding
            let padding = (y_max - y_min) * 0.05;
            (y_min - padding, y_max + padding)
        };

        (x_range, y_range)
    }
}

/// A single data series in a chart.
#[derive(Debug, Clone)]
pub struct ChartSeries {
    /// Series name.
    pub name: String,
    /// Data points (x, y).
    pub data: Vec<(f32, f32)>,
    /// Line color.
    pub color: [f32; 4],
    /// Line width.
    pub line_width: f32,
    /// Whether to fill area under the line.
    pub fill: bool,
    /// Whether this series is visible.
    pub visible: bool,
}

impl Default for ChartSeries {
    fn default() -> Self {
        Self {
            name: "Series".to_string(),
            data: Vec::new(),
            color: [0.2, 0.6, 1.0, 1.0],
            line_width: 2.0,
            fill: false,
            visible: true,
        }
    }
}

impl ChartSeries {
    /// Create a new series with the given name and color.
    pub fn new(name: impl Into<String>, color: [f32; 4]) -> Self {
        Self {
            name: name.into(),
            color,
            ..Default::default()
        }
    }

    /// Add a data point.
    pub fn add_point(&mut self, x: f32, y: f32) {
        self.data.push((x, y));
    }

    /// Add multiple data points.
    pub fn add_points(&mut self, points: &[(f32, f32)]) {
        self.data.extend_from_slice(points);
    }
}

/// A marker on a chart (e.g., for phase transitions).
#[derive(Debug, Clone)]
pub struct ChartMarker {
    /// X position.
    pub x: f32,
    /// Marker label.
    pub label: String,
    /// Marker color.
    pub color: [f32; 4],
}

/// Architecture diagram data.
#[derive(Debug, Clone)]
pub struct ArchitectureDiagram {
    /// Blocks/layers in the diagram.
    pub blocks: Vec<ArchBlock>,
    /// Connections between blocks.
    pub connections: Vec<ArchConnection>,
    /// Diagram title.
    pub title: String,
    /// Layout direction (horizontal or vertical).
    pub vertical: bool,
}

impl Default for ArchitectureDiagram {
    fn default() -> Self {
        Self {
            blocks: Vec::new(),
            connections: Vec::new(),
            title: "Model Architecture".to_string(),
            vertical: true,
        }
    }
}

/// A block in the architecture diagram.
#[derive(Debug, Clone)]
pub struct ArchBlock {
    /// Block identifier.
    pub id: String,
    /// Display name.
    pub name: String,
    /// Block type (for styling).
    pub block_type: ArchBlockType,
    /// Position (computed or specified).
    pub position: Option<(f32, f32)>,
    /// Size override.
    pub size: Option<(f32, f32)>,
    /// Additional info (e.g., parameter count).
    pub info: String,
    /// Whether this block is highlighted.
    pub highlighted: bool,
}

/// Types of architecture blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArchBlockType {
    Input,
    Embedding,
    Attention,
    FeedForward,
    Normalization,
    Output,
    Custom,
}

/// A connection in the architecture diagram.
#[derive(Debug, Clone)]
pub struct ArchConnection {
    /// Source block ID.
    pub from: String,
    /// Target block ID.
    pub to: String,
    /// Connection label.
    pub label: Option<String>,
    /// Whether this is a residual/skip connection.
    pub is_residual: bool,
}

/// Token cloud data.
#[derive(Debug, Clone)]
pub struct TokenCloudData {
    /// Token strings.
    pub tokens: Vec<String>,
    /// 3D positions.
    pub positions: Vec<[f32; 3]>,
    /// Token sizes (based on frequency).
    pub sizes: Vec<f32>,
    /// Token colors (based on cluster or category).
    pub colors: Vec<[f32; 4]>,
    /// Selected token indices.
    pub selected: Vec<usize>,
}

impl Default for TokenCloudData {
    fn default() -> Self {
        Self {
            tokens: Vec::new(),
            positions: Vec::new(),
            sizes: Vec::new(),
            colors: Vec::new(),
            selected: Vec::new(),
        }
    }
}

/// Layer activation data.
#[derive(Debug, Clone)]
pub struct LayerActivationData {
    /// Layer names.
    pub layers: Vec<String>,
    /// Histograms for each layer (binned values).
    pub histograms: Vec<Vec<f32>>,
    /// Statistics per layer (mean, std, min, max).
    pub stats: Vec<ActivationStats>,
    /// Selected layer index.
    pub selected_layer: Option<usize>,
}

impl Default for LayerActivationData {
    fn default() -> Self {
        Self {
            layers: Vec::new(),
            histograms: Vec::new(),
            stats: Vec::new(),
            selected_layer: None,
        }
    }
}

/// Statistics for layer activations.
#[derive(Debug, Clone, Copy, Default)]
pub struct ActivationStats {
    pub mean: f32,
    pub std: f32,
    pub min: f32,
    pub max: f32,
    pub dead_neurons_pct: f32,
}

/// Gradient flow data.
#[derive(Debug, Clone)]
pub struct GradientFlowData {
    /// Layer names.
    pub layers: Vec<String>,
    /// Gradient norms per layer.
    pub gradient_norms: Vec<f32>,
    /// Flow directions (positive = forward, negative = backward).
    pub flow_directions: Vec<f32>,
    /// Whether to show as bars or flow lines.
    pub show_as_bars: bool,
}

impl Default for GradientFlowData {
    fn default() -> Self {
        Self {
            layers: Vec::new(),
            gradient_norms: Vec::new(),
            flow_directions: Vec::new(),
            show_as_bars: true,
        }
    }
}

// ============================================================================
// Dashboard Panel
// ============================================================================

/// A single panel in the dashboard.
#[derive(Debug, Clone)]
pub struct DashboardPanel {
    /// Unique identifier for this panel.
    pub id: String,
    /// Type of visualization.
    pub panel_type: PanelType,
    /// Position in the grid.
    pub position: PanelPosition,
    /// Size in grid cells.
    pub size: PanelSize,
    /// Panel content/data.
    pub content: PanelContent,
    /// Panel title (displayed in header).
    pub title: String,
    /// Whether the panel is visible.
    pub visible: bool,
    /// Whether the panel has focus.
    pub focused: bool,
    /// Whether to show the panel border.
    pub show_border: bool,
    /// Whether to show the panel title.
    pub show_title: bool,
    /// Panel-specific settings.
    pub settings: PanelSettings,
}

impl DashboardPanel {
    /// Create a new panel.
    pub fn new(
        id: impl Into<String>,
        panel_type: PanelType,
        position: PanelPosition,
        size: PanelSize,
    ) -> Self {
        let id_str = id.into();
        let title = Self::default_title(panel_type);
        Self {
            id: id_str,
            panel_type,
            position,
            size,
            content: PanelContent::Empty,
            title,
            visible: true,
            focused: false,
            show_border: true,
            show_title: true,
            settings: PanelSettings::default(),
        }
    }

    /// Default title based on panel type.
    fn default_title(panel_type: PanelType) -> String {
        match panel_type {
            PanelType::Network3D => "3D Network View".to_string(),
            PanelType::AttentionHeatmap => "Attention Patterns".to_string(),
            PanelType::ChordDiagram => "Attention Flow".to_string(),
            PanelType::EmbeddingCloud => "Embedding Space".to_string(),
            PanelType::LossLandscape => "Loss Landscape".to_string(),
            PanelType::MetricsChart => "Training Metrics".to_string(),
            PanelType::ArchitectureDiagram => "Model Architecture".to_string(),
            PanelType::TokenCloud => "Token Cloud".to_string(),
            PanelType::LayerActivations => "Layer Activations".to_string(),
            PanelType::GradientFlow => "Gradient Flow".to_string(),
            PanelType::Custom => "Custom Panel".to_string(),
        }
    }

    /// Set the panel title.
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = title.into();
        self
    }

    /// Set the panel content.
    pub fn with_content(mut self, content: PanelContent) -> Self {
        self.content = content;
        self
    }

    /// Get pixel bounds for a given total dashboard size.
    pub fn pixel_bounds(
        &self,
        layout: &DashboardLayout,
        total_width: u32,
        total_height: u32,
    ) -> PanelBounds {
        let cell_width =
            (total_width as f32 - layout.margin * 2.0) / layout.grid_cols as f32 - layout.gap;
        let cell_height =
            (total_height as f32 - layout.margin * 2.0) / layout.grid_rows as f32 - layout.gap;

        let x = layout.margin + self.position.col as f32 * (cell_width + layout.gap);
        let y = layout.margin + self.position.row as f32 * (cell_height + layout.gap);
        let width = self.size.cols as f32 * cell_width + (self.size.cols - 1) as f32 * layout.gap;
        let height = self.size.rows as f32 * cell_height + (self.size.rows - 1) as f32 * layout.gap;

        PanelBounds {
            x: x as u32,
            y: y as u32,
            width: width as u32,
            height: height as u32,
        }
    }
}

/// Pixel bounds for a rendered panel.
#[derive(Debug, Clone, Copy)]
pub struct PanelBounds {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

/// Panel-specific rendering settings.
#[derive(Debug, Clone, Default)]
pub struct PanelSettings {
    /// Override colormap for this panel.
    pub colormap: Option<ColormapPreset>,
    /// Background color override.
    pub background_color: Option<[f32; 4]>,
    /// Font size multiplier.
    pub font_scale: f32,
    /// Animation speed multiplier.
    pub animation_speed: f32,
    /// Custom parameters as key-value pairs.
    pub custom: HashMap<String, String>,
}

impl PanelSettings {
    pub fn new() -> Self {
        Self {
            font_scale: 1.0,
            animation_speed: 1.0,
            ..Default::default()
        }
    }
}

// ============================================================================
// Dashboard Layout
// ============================================================================

/// Layout configuration for the dashboard grid.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardLayout {
    /// Number of columns in the grid.
    pub grid_cols: usize,
    /// Number of rows in the grid.
    pub grid_rows: usize,
    /// Gap between panels in pixels.
    pub gap: f32,
    /// Margin around the entire dashboard in pixels.
    pub margin: f32,
}

impl DashboardLayout {
    /// Create a new layout with specified grid dimensions.
    pub fn new(cols: usize, rows: usize) -> Self {
        Self {
            grid_cols: cols,
            grid_rows: rows,
            gap: 8.0,
            margin: 16.0,
        }
    }

    /// Set gap between panels.
    pub fn with_gap(mut self, gap: f32) -> Self {
        self.gap = gap;
        self
    }

    /// Set margin around dashboard.
    pub fn with_margin(mut self, margin: f32) -> Self {
        self.margin = margin;
        self
    }

    /// Total number of grid cells.
    pub fn cell_count(&self) -> usize {
        self.grid_cols * self.grid_rows
    }

    /// Common layout: single panel.
    pub fn single() -> Self {
        Self::new(1, 1)
    }

    /// Common layout: 2x2 grid.
    pub fn grid_2x2() -> Self {
        Self::new(2, 2)
    }

    /// Common layout: 3x2 grid (wide).
    pub fn grid_3x2() -> Self {
        Self::new(3, 2)
    }

    /// Common layout: 4x3 grid (full dashboard).
    pub fn grid_4x3() -> Self {
        Self::new(4, 3)
    }
}

impl Default for DashboardLayout {
    fn default() -> Self {
        Self::grid_3x2()
    }
}

// ============================================================================
// Selection
// ============================================================================

/// Represents the current selection state across all panels.
#[derive(Debug, Clone, Default)]
pub struct Selection {
    /// Selected layer names.
    pub layers: Vec<String>,
    /// Selected token indices.
    pub tokens: Vec<usize>,
    /// Selected step range (start, end).
    pub step_range: Option<(u64, u64)>,
    /// Selected attention head indices.
    pub attention_heads: Vec<usize>,
    /// Selected embedding point indices.
    pub embedding_points: Vec<usize>,
    /// Primary selection (single item focus).
    pub primary: Option<SelectionItem>,
}

impl Selection {
    /// Create an empty selection.
    pub fn new() -> Self {
        Self::default()
    }

    /// Clear all selections.
    pub fn clear(&mut self) {
        self.layers.clear();
        self.tokens.clear();
        self.step_range = None;
        self.attention_heads.clear();
        self.embedding_points.clear();
        self.primary = None;
    }

    /// Check if anything is selected.
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
            && self.tokens.is_empty()
            && self.step_range.is_none()
            && self.attention_heads.is_empty()
            && self.embedding_points.is_empty()
            && self.primary.is_none()
    }

    /// Select a layer.
    pub fn select_layer(&mut self, name: &str) {
        if !self.layers.contains(&name.to_string()) {
            self.layers.push(name.to_string());
        }
        self.primary = Some(SelectionItem::Layer(name.to_string()));
    }

    /// Select a token.
    pub fn select_token(&mut self, index: usize) {
        if !self.tokens.contains(&index) {
            self.tokens.push(index);
        }
        self.primary = Some(SelectionItem::Token(index));
    }

    /// Select an attention head.
    pub fn select_attention_head(&mut self, layer: usize, head: usize) {
        let idx = layer * 100 + head; // Simple encoding
        if !self.attention_heads.contains(&idx) {
            self.attention_heads.push(idx);
        }
        self.primary = Some(SelectionItem::AttentionHead { layer, head });
    }
}

/// A single selection item.
#[derive(Debug, Clone)]
pub enum SelectionItem {
    Layer(String),
    Token(usize),
    AttentionHead { layer: usize, head: usize },
    EmbeddingPoint(usize),
    TrajectoryStep(u64),
}

// ============================================================================
// Dashboard Theme
// ============================================================================

/// Visual theme for the dashboard.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardTheme {
    /// Main background color.
    pub background: [f32; 4],
    /// Panel background color.
    pub panel_background: [f32; 4],
    /// Panel border color.
    pub border_color: [f32; 4],
    /// Primary text color.
    pub text_color: [f32; 4],
    /// Accent color for highlights.
    pub accent_color: [f32; 4],
    /// Selection highlight color.
    pub selection_color: [f32; 4],
    /// Error/warning color.
    pub error_color: [f32; 4],
    /// Success color.
    pub success_color: [f32; 4],
    /// Base font size.
    pub font_size: f32,
    /// Panel title font size.
    pub title_font_size: f32,
    /// Border width.
    pub border_width: f32,
    /// Corner radius for panels.
    pub corner_radius: f32,
}

impl Default for DashboardTheme {
    fn default() -> Self {
        Self::dark()
    }
}

impl DashboardTheme {
    /// Dark theme (default).
    pub fn dark() -> Self {
        Self {
            background: [0.08, 0.08, 0.12, 1.0],
            panel_background: [0.12, 0.12, 0.18, 1.0],
            border_color: [0.25, 0.25, 0.35, 1.0],
            text_color: [0.9, 0.9, 0.95, 1.0],
            accent_color: [0.3, 0.6, 1.0, 1.0],
            selection_color: [1.0, 0.7, 0.2, 0.8],
            error_color: [1.0, 0.3, 0.3, 1.0],
            success_color: [0.3, 0.9, 0.4, 1.0],
            font_size: 14.0,
            title_font_size: 16.0,
            border_width: 1.0,
            corner_radius: 4.0,
        }
    }

    /// Light theme.
    pub fn light() -> Self {
        Self {
            background: [0.95, 0.95, 0.97, 1.0],
            panel_background: [1.0, 1.0, 1.0, 1.0],
            border_color: [0.8, 0.8, 0.85, 1.0],
            text_color: [0.15, 0.15, 0.2, 1.0],
            accent_color: [0.2, 0.5, 0.9, 1.0],
            selection_color: [1.0, 0.6, 0.1, 0.8],
            error_color: [0.9, 0.2, 0.2, 1.0],
            success_color: [0.2, 0.7, 0.3, 1.0],
            font_size: 14.0,
            title_font_size: 16.0,
            border_width: 1.0,
            corner_radius: 4.0,
        }
    }

    /// High contrast theme for accessibility.
    pub fn high_contrast() -> Self {
        Self {
            background: [0.0, 0.0, 0.0, 1.0],
            panel_background: [0.05, 0.05, 0.05, 1.0],
            border_color: [1.0, 1.0, 1.0, 1.0],
            text_color: [1.0, 1.0, 1.0, 1.0],
            accent_color: [0.0, 1.0, 1.0, 1.0],
            selection_color: [1.0, 1.0, 0.0, 1.0],
            error_color: [1.0, 0.0, 0.0, 1.0],
            success_color: [0.0, 1.0, 0.0, 1.0],
            font_size: 16.0,
            title_font_size: 18.0,
            border_width: 2.0,
            corner_radius: 0.0,
        }
    }
}

// ============================================================================
// Main Dashboard
// ============================================================================

/// The main visualization dashboard combining multiple panel types.
pub struct VisualizationDashboard {
    /// All panels in the dashboard.
    panels: HashMap<String, DashboardPanel>,
    /// Panel rendering order (for z-ordering).
    panel_order: Vec<String>,
    /// Dashboard layout configuration.
    pub layout: DashboardLayout,
    /// Current selection state.
    pub selection: Selection,
    /// Visual theme.
    pub theme: DashboardTheme,
    /// Current time for animations.
    time: f32,
    /// Frame counter.
    frame: u64,
    /// Whether the dashboard needs a full redraw.
    dirty: bool,
    /// 3D rendering engine (shared).
    engine_3d: Viz3DEngine,
    /// Default colormap.
    colormap: Colormap,
    /// Dashboard title.
    pub title: String,
    /// Whether to show the title bar.
    pub show_title_bar: bool,
}

impl VisualizationDashboard {
    /// Create a new dashboard with the specified layout.
    pub fn new(layout: DashboardLayout) -> Self {
        Self {
            panels: HashMap::new(),
            panel_order: Vec::new(),
            layout,
            selection: Selection::new(),
            theme: DashboardTheme::default(),
            time: 0.0,
            frame: 0,
            dirty: true,
            engine_3d: Viz3DEngine::new(),
            colormap: ColormapPreset::Viridis.colormap(),
            title: "Visualization Dashboard".to_string(),
            show_title_bar: true,
        }
    }

    /// Set the dashboard theme.
    pub fn with_theme(mut self, theme: DashboardTheme) -> Self {
        self.theme = theme;
        self.dirty = true;
        self
    }

    /// Set the dashboard title.
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = title.into();
        self
    }

    // ========================================================================
    // Panel Management
    // ========================================================================

    /// Add a panel to the dashboard.
    pub fn add_panel(&mut self, panel: DashboardPanel) -> Result<()> {
        // Validate position is within bounds
        if panel.position.col >= self.layout.grid_cols
            || panel.position.row >= self.layout.grid_rows
        {
            return Err(DashboardError::OutOfBounds {
                row: panel.position.row,
                col: panel.position.col,
                max_row: self.layout.grid_rows,
                max_col: self.layout.grid_cols,
            });
        }

        // Check for overlaps with existing panels
        for existing in self.panels.values() {
            if self.panels_overlap(&panel, existing) {
                return Err(DashboardError::PanelOverlap {
                    row: panel.position.row,
                    col: panel.position.col,
                });
            }
        }

        let id = panel.id.clone();
        self.panels.insert(id.clone(), panel);
        self.panel_order.push(id);
        self.dirty = true;

        Ok(())
    }

    /// Check if two panels overlap.
    fn panels_overlap(&self, a: &DashboardPanel, b: &DashboardPanel) -> bool {
        let a_left = a.position.col;
        let a_right = a.position.col + a.size.cols;
        let a_top = a.position.row;
        let a_bottom = a.position.row + a.size.rows;

        let b_left = b.position.col;
        let b_right = b.position.col + b.size.cols;
        let b_top = b.position.row;
        let b_bottom = b.position.row + b.size.rows;

        !(a_right <= b_left || b_right <= a_left || a_bottom <= b_top || b_bottom <= a_top)
    }

    /// Remove a panel by ID.
    pub fn remove_panel(&mut self, id: &str) -> Option<DashboardPanel> {
        self.panel_order.retain(|i| i != id);
        let removed = self.panels.remove(id);
        if removed.is_some() {
            self.dirty = true;
        }
        removed
    }

    /// Get a reference to a panel by ID.
    pub fn get_panel(&self, id: &str) -> Option<&DashboardPanel> {
        self.panels.get(id)
    }

    /// Get a mutable reference to a panel by ID.
    pub fn get_panel_mut(&mut self, id: &str) -> Option<&mut DashboardPanel> {
        self.dirty = true;
        self.panels.get_mut(id)
    }

    /// Get all panel IDs.
    pub fn panel_ids(&self) -> impl Iterator<Item = &str> {
        self.panel_order.iter().map(|s| s.as_str())
    }

    /// Get the number of panels.
    pub fn panel_count(&self) -> usize {
        self.panels.len()
    }

    /// Clear all panels.
    pub fn clear_panels(&mut self) {
        self.panels.clear();
        self.panel_order.clear();
        self.dirty = true;
    }

    // ========================================================================
    // Preset Dashboards
    // ========================================================================

    /// Create a training monitor preset dashboard.
    ///
    /// Layout:
    /// ```text
    /// +----------------+--------+
    /// |                |  Arch  |
    /// |    Network3D   +--------+
    /// |                | Grads  |
    /// +--------+-------+--------+
    /// |  Loss  | LR/Metrics     |
    /// +--------+----------------+
    /// ```
    pub fn preset_training_monitor() -> Self {
        let layout = DashboardLayout::new(3, 3);
        let mut dashboard = Self::new(layout);
        dashboard.title = "Training Monitor".to_string();

        // Main 3D network view (2x2, top-left)
        let network_panel = DashboardPanel::new(
            "network_3d",
            PanelType::Network3D,
            PanelPosition::new(0, 0),
            PanelSize::new(2, 2),
        );

        // Architecture diagram (1x1, top-right)
        let arch_panel = DashboardPanel::new(
            "architecture",
            PanelType::ArchitectureDiagram,
            PanelPosition::new(2, 0),
            PanelSize::unit(),
        );

        // Gradient flow (1x1, middle-right)
        let gradient_panel = DashboardPanel::new(
            "gradients",
            PanelType::GradientFlow,
            PanelPosition::new(2, 1),
            PanelSize::unit(),
        );

        // Loss chart (1x1, bottom-left)
        let loss_panel = DashboardPanel::new(
            "loss",
            PanelType::MetricsChart,
            PanelPosition::new(0, 2),
            PanelSize::unit(),
        )
        .with_title("Loss");

        // Metrics chart (2x1, bottom-right)
        let metrics_panel = DashboardPanel::new(
            "metrics",
            PanelType::MetricsChart,
            PanelPosition::new(1, 2),
            PanelSize::new(2, 1),
        )
        .with_title("Learning Rate & Metrics");

        let _ = dashboard.add_panel(network_panel);
        let _ = dashboard.add_panel(arch_panel);
        let _ = dashboard.add_panel(gradient_panel);
        let _ = dashboard.add_panel(loss_panel);
        let _ = dashboard.add_panel(metrics_panel);

        dashboard
    }

    /// Create a model inspection preset dashboard.
    ///
    /// Layout:
    /// ```text
    /// +--------+--------+--------+
    /// |  3D    | Embed  | Tokens |
    /// +--------+--------+--------+
    /// | Activations     | Arch   |
    /// +-----------------+--------+
    /// ```
    pub fn preset_model_inspection() -> Self {
        let layout = DashboardLayout::new(3, 2);
        let mut dashboard = Self::new(layout);
        dashboard.title = "Model Inspection".to_string();

        let network_panel = DashboardPanel::new(
            "network",
            PanelType::Network3D,
            PanelPosition::new(0, 0),
            PanelSize::unit(),
        );

        let embedding_panel = DashboardPanel::new(
            "embeddings",
            PanelType::EmbeddingCloud,
            PanelPosition::new(1, 0),
            PanelSize::unit(),
        );

        let token_panel = DashboardPanel::new(
            "tokens",
            PanelType::TokenCloud,
            PanelPosition::new(2, 0),
            PanelSize::unit(),
        );

        let activation_panel = DashboardPanel::new(
            "activations",
            PanelType::LayerActivations,
            PanelPosition::new(0, 1),
            PanelSize::new(2, 1),
        );

        let arch_panel = DashboardPanel::new(
            "architecture",
            PanelType::ArchitectureDiagram,
            PanelPosition::new(2, 1),
            PanelSize::unit(),
        );

        let _ = dashboard.add_panel(network_panel);
        let _ = dashboard.add_panel(embedding_panel);
        let _ = dashboard.add_panel(token_panel);
        let _ = dashboard.add_panel(activation_panel);
        let _ = dashboard.add_panel(arch_panel);

        dashboard
    }

    /// Create an attention analysis preset dashboard.
    ///
    /// Layout:
    /// ```text
    /// +--------+--------+--------+
    /// | Heat 1 | Heat 2 | Chord  |
    /// +--------+--------+--------+
    /// |    Aggregated   | Flow   |
    /// +-----------------+--------+
    /// ```
    pub fn preset_attention_analysis() -> Self {
        let layout = DashboardLayout::new(3, 2);
        let mut dashboard = Self::new(layout);
        dashboard.title = "Attention Analysis".to_string();

        let heatmap1 = DashboardPanel::new(
            "attention_l0",
            PanelType::AttentionHeatmap,
            PanelPosition::new(0, 0),
            PanelSize::unit(),
        )
        .with_title("Layer 0 Attention");

        let heatmap2 = DashboardPanel::new(
            "attention_l1",
            PanelType::AttentionHeatmap,
            PanelPosition::new(1, 0),
            PanelSize::unit(),
        )
        .with_title("Layer 1 Attention");

        let chord = DashboardPanel::new(
            "chord",
            PanelType::ChordDiagram,
            PanelPosition::new(2, 0),
            PanelSize::unit(),
        )
        .with_title("Attention Flow");

        let aggregated = DashboardPanel::new(
            "aggregated",
            PanelType::AttentionHeatmap,
            PanelPosition::new(0, 1),
            PanelSize::new(2, 1),
        )
        .with_title("Aggregated Attention");

        let flow = DashboardPanel::new(
            "flow",
            PanelType::GradientFlow,
            PanelPosition::new(2, 1),
            PanelSize::unit(),
        )
        .with_title("Attention Flow Summary");

        let _ = dashboard.add_panel(heatmap1);
        let _ = dashboard.add_panel(heatmap2);
        let _ = dashboard.add_panel(chord);
        let _ = dashboard.add_panel(aggregated);
        let _ = dashboard.add_panel(flow);

        dashboard
    }

    /// Create a loss landscape analysis preset.
    ///
    /// Layout:
    /// ```text
    /// +----------------+--------+
    /// |                | Traj   |
    /// |   Landscape    +--------+
    /// |                | Grad   |
    /// +----------------+--------+
    /// ```
    pub fn preset_loss_landscape() -> Self {
        let layout = DashboardLayout::new(3, 2);
        let mut dashboard = Self::new(layout);
        dashboard.title = "Loss Landscape Analysis".to_string();

        let landscape = DashboardPanel::new(
            "landscape",
            PanelType::LossLandscape,
            PanelPosition::new(0, 0),
            PanelSize::new(2, 2),
        );

        let trajectory = DashboardPanel::new(
            "trajectory",
            PanelType::MetricsChart,
            PanelPosition::new(2, 0),
            PanelSize::unit(),
        )
        .with_title("Loss Over Time");

        let gradients = DashboardPanel::new(
            "gradients",
            PanelType::GradientFlow,
            PanelPosition::new(2, 1),
            PanelSize::unit(),
        )
        .with_title("Gradient Magnitude");

        let _ = dashboard.add_panel(landscape);
        let _ = dashboard.add_panel(trajectory);
        let _ = dashboard.add_panel(gradients);

        dashboard
    }

    // ========================================================================
    // Update and Animation
    // ========================================================================

    /// Update all panels with a time delta.
    pub fn update(&mut self, dt: f32) {
        self.time += dt;
        self.frame += 1;

        // Update animations in each panel
        for panel in self.panels.values_mut() {
            Self::update_panel_content(&mut panel.content, dt, &panel.settings);
        }

        self.dirty = true;
    }

    /// Update panel content animations.
    fn update_panel_content(content: &mut PanelContent, dt: f32, settings: &PanelSettings) {
        let speed = settings.animation_speed;
        match content {
            PanelContent::Neural3D(scene) => {
                scene.animation_time += dt * speed;
            }
            PanelContent::Landscape(data) => {
                // Could animate trajectory progress
                let _ = (data, dt, speed);
            }
            _ => {}
        }
    }

    /// Get the current animation time.
    pub fn time(&self) -> f32 {
        self.time
    }

    /// Get the current frame number.
    pub fn frame(&self) -> u64 {
        self.frame
    }

    /// Mark the dashboard as needing a redraw.
    pub fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    /// Check if the dashboard needs a redraw.
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    // ========================================================================
    // Selection Synchronization
    // ========================================================================

    /// Synchronize selection across all panels.
    ///
    /// When an element is selected in one panel, this updates related
    /// elements in other panels to show the same selection.
    pub fn sync_selection(&mut self) {
        // Get the current selection state
        let selection = self.selection.clone();

        // Update each panel based on selection
        for panel in self.panels.values_mut() {
            Self::apply_selection_to_panel(panel, &selection);
        }

        self.dirty = true;
    }

    /// Apply selection to a specific panel.
    fn apply_selection_to_panel(panel: &mut DashboardPanel, selection: &Selection) {
        match &mut panel.content {
            PanelContent::Neural3D(scene) => {
                // Highlight selected layers
                for layer in &mut scene.layers {
                    layer.selected = selection.layers.contains(&layer.name);
                }
                // Highlight connections to/from selected layers
                for conn in &mut scene.connections {
                    conn.highlighted = selection.layers.contains(&conn.from_layer)
                        || selection.layers.contains(&conn.to_layer);
                }
            }
            PanelContent::Heatmap(heatmap) => {
                // Highlight selected tokens
                heatmap.selected_cells.clear();
                for &token_idx in &selection.tokens {
                    // Select entire row and column for this token
                    let (rows, cols) = heatmap.dimensions();
                    for i in 0..rows {
                        heatmap.selected_cells.push((token_idx, i));
                        heatmap.selected_cells.push((i, token_idx));
                    }
                    let _ = cols; // Symmetric
                }
            }
            PanelContent::Embedding(data) => {
                data.selected = selection.embedding_points.clone();
            }
            PanelContent::TokenCloud(data) => {
                data.selected = selection.tokens.clone();
            }
            _ => {}
        }
    }

    /// Clear all selections.
    pub fn clear_selection(&mut self) {
        self.selection.clear();
        self.sync_selection();
    }

    // ========================================================================
    // Rendering
    // ========================================================================

    /// Render the entire dashboard to a pixel buffer.
    ///
    /// Returns RGBA pixel data in row-major order.
    pub fn render(&self, width: u32, height: u32) -> Vec<u8> {
        let mut buffer = vec![0u8; (width * height * 4) as usize];

        // Fill background
        self.fill_background(&mut buffer, width, height);

        // Render title bar if enabled
        let content_y_offset = if self.show_title_bar {
            self.render_title_bar(&mut buffer, width);
            32 // Title bar height
        } else {
            0
        };

        // Adjust height for content area
        let content_height = height.saturating_sub(content_y_offset);

        // Render each panel
        for panel_id in &self.panel_order {
            if let Some(panel) = self.panels.get(panel_id) {
                if panel.visible {
                    self.render_panel(panel, &mut buffer, width, content_y_offset, content_height);
                }
            }
        }

        // Mark as clean
        // Note: We can't actually clear dirty flag here since self is immutable
        // The caller should call mark_clean() if needed

        buffer
    }

    /// Fill the background color.
    fn fill_background(&self, buffer: &mut [u8], width: u32, height: u32) {
        let bg = self.theme.background;
        let r = (bg[0] * 255.0) as u8;
        let g = (bg[1] * 255.0) as u8;
        let b = (bg[2] * 255.0) as u8;
        let a = (bg[3] * 255.0) as u8;

        for pixel in buffer.chunks_exact_mut(4) {
            pixel[0] = r;
            pixel[1] = g;
            pixel[2] = b;
            pixel[3] = a;
        }
        let _ = (width, height); // Used in calculation above
    }

    /// Render the title bar.
    fn render_title_bar(&self, buffer: &mut [u8], width: u32) {
        // Simple title bar with background color
        let bg = self.theme.panel_background;
        let r = (bg[0] * 255.0) as u8;
        let g = (bg[1] * 255.0) as u8;
        let b = (bg[2] * 255.0) as u8;
        let a = (bg[3] * 255.0) as u8;

        // Fill title bar area (32 pixels high)
        for y in 0..32 {
            for x in 0..width {
                let idx = ((y * width + x) * 4) as usize;
                if idx + 3 < buffer.len() {
                    buffer[idx] = r;
                    buffer[idx + 1] = g;
                    buffer[idx + 2] = b;
                    buffer[idx + 3] = a;
                }
            }
        }

        // TODO: Actually render title text using a font renderer
        // For now, this is a placeholder
    }

    /// Render a single panel.
    fn render_panel(
        &self,
        panel: &DashboardPanel,
        buffer: &mut [u8],
        width: u32,
        y_offset: u32,
        content_height: u32,
    ) {
        // Calculate pixel bounds
        let bounds = panel.pixel_bounds(&self.layout, width, content_height);
        let bounds = PanelBounds {
            x: bounds.x,
            y: bounds.y + y_offset,
            width: bounds.width,
            height: bounds.height,
        };

        // Draw panel background
        self.draw_panel_background(panel, &bounds, buffer, width);

        // Draw panel border
        if panel.show_border {
            self.draw_panel_border(&bounds, buffer, width, panel.focused);
        }

        // Draw panel title
        if panel.show_title {
            self.draw_panel_title(panel, &bounds, buffer, width);
        }

        // Render panel content
        let content_bounds = self.content_bounds(&bounds, panel.show_title);
        self.render_panel_content(panel, &content_bounds, buffer, width);
    }

    /// Calculate content bounds (excluding title and border).
    fn content_bounds(&self, bounds: &PanelBounds, has_title: bool) -> PanelBounds {
        let title_height = if has_title { 24 } else { 0 };
        let border = self.theme.border_width as u32;

        PanelBounds {
            x: bounds.x + border,
            y: bounds.y + border + title_height,
            width: bounds.width.saturating_sub(border * 2),
            height: bounds.height.saturating_sub(border * 2 + title_height),
        }
    }

    /// Draw panel background.
    fn draw_panel_background(
        &self,
        panel: &DashboardPanel,
        bounds: &PanelBounds,
        buffer: &mut [u8],
        width: u32,
    ) {
        let bg = panel
            .settings
            .background_color
            .unwrap_or(self.theme.panel_background);
        let r = (bg[0] * 255.0) as u8;
        let g = (bg[1] * 255.0) as u8;
        let b = (bg[2] * 255.0) as u8;
        let a = (bg[3] * 255.0) as u8;

        for y in bounds.y..(bounds.y + bounds.height) {
            for x in bounds.x..(bounds.x + bounds.width) {
                let idx = ((y * width + x) * 4) as usize;
                if idx + 3 < buffer.len() {
                    buffer[idx] = r;
                    buffer[idx + 1] = g;
                    buffer[idx + 2] = b;
                    buffer[idx + 3] = a;
                }
            }
        }
    }

    /// Draw panel border.
    fn draw_panel_border(
        &self,
        bounds: &PanelBounds,
        buffer: &mut [u8],
        width: u32,
        focused: bool,
    ) {
        let color = if focused {
            self.theme.accent_color
        } else {
            self.theme.border_color
        };
        let r = (color[0] * 255.0) as u8;
        let g = (color[1] * 255.0) as u8;
        let b = (color[2] * 255.0) as u8;
        let a = (color[3] * 255.0) as u8;

        let border_width = self.theme.border_width as u32;

        // Top and bottom borders
        for y in 0..border_width {
            for x in bounds.x..(bounds.x + bounds.width) {
                // Top
                let idx = (((bounds.y + y) * width + x) * 4) as usize;
                if idx + 3 < buffer.len() {
                    buffer[idx] = r;
                    buffer[idx + 1] = g;
                    buffer[idx + 2] = b;
                    buffer[idx + 3] = a;
                }
                // Bottom
                let idx = (((bounds.y + bounds.height - 1 - y) * width + x) * 4) as usize;
                if idx + 3 < buffer.len() {
                    buffer[idx] = r;
                    buffer[idx + 1] = g;
                    buffer[idx + 2] = b;
                    buffer[idx + 3] = a;
                }
            }
        }

        // Left and right borders
        for y in bounds.y..(bounds.y + bounds.height) {
            for x in 0..border_width {
                // Left
                let idx = ((y * width + bounds.x + x) * 4) as usize;
                if idx + 3 < buffer.len() {
                    buffer[idx] = r;
                    buffer[idx + 1] = g;
                    buffer[idx + 2] = b;
                    buffer[idx + 3] = a;
                }
                // Right
                let idx = ((y * width + bounds.x + bounds.width - 1 - x) * 4) as usize;
                if idx + 3 < buffer.len() {
                    buffer[idx] = r;
                    buffer[idx + 1] = g;
                    buffer[idx + 2] = b;
                    buffer[idx + 3] = a;
                }
            }
        }
    }

    /// Draw panel title.
    fn draw_panel_title(
        &self,
        panel: &DashboardPanel,
        bounds: &PanelBounds,
        buffer: &mut [u8],
        width: u32,
    ) {
        // Draw title background (slightly darker than panel)
        let bg = self.theme.panel_background;
        let title_bg = [
            (bg[0] * 0.8).max(0.0),
            (bg[1] * 0.8).max(0.0),
            (bg[2] * 0.8).max(0.0),
            bg[3],
        ];
        let r = (title_bg[0] * 255.0) as u8;
        let g = (title_bg[1] * 255.0) as u8;
        let b = (title_bg[2] * 255.0) as u8;
        let a = (title_bg[3] * 255.0) as u8;

        let border = self.theme.border_width as u32;
        let title_height = 24;

        for y in (bounds.y + border)..(bounds.y + border + title_height) {
            for x in (bounds.x + border)..(bounds.x + bounds.width - border) {
                let idx = ((y * width + x) * 4) as usize;
                if idx + 3 < buffer.len() {
                    buffer[idx] = r;
                    buffer[idx + 1] = g;
                    buffer[idx + 2] = b;
                    buffer[idx + 3] = a;
                }
            }
        }

        // TODO: Actually render title text
        let _ = &panel.title;
    }

    /// Render panel content based on type.
    fn render_panel_content(
        &self,
        panel: &DashboardPanel,
        bounds: &PanelBounds,
        buffer: &mut [u8],
        width: u32,
    ) {
        match &panel.content {
            PanelContent::Heatmap(heatmap) => {
                self.render_heatmap(heatmap, bounds, buffer, width);
            }
            PanelContent::Chart(chart) => {
                self.render_chart(chart, bounds, buffer, width);
            }
            PanelContent::GradientFlow(flow) => {
                self.render_gradient_flow(flow, bounds, buffer, width);
            }
            PanelContent::Activations(activations) => {
                self.render_activations(activations, bounds, buffer, width);
            }
            // 3D content would use the engine
            PanelContent::Neural3D(_)
            | PanelContent::Embedding(_)
            | PanelContent::Landscape(_)
            | PanelContent::TokenCloud(_) => {
                // Placeholder for 3D rendering
                self.render_3d_placeholder(bounds, buffer, width);
            }
            _ => {
                // Placeholder for unimplemented content types
                self.render_placeholder(bounds, buffer, width);
            }
        }
    }

    /// Render a heatmap.
    fn render_heatmap(
        &self,
        heatmap: &HeatmapView,
        bounds: &PanelBounds,
        buffer: &mut [u8],
        width: u32,
    ) {
        let (rows, cols) = heatmap.dimensions();
        if rows == 0 || cols == 0 {
            return;
        }

        let colormap = heatmap.colormap.colormap();
        let (min_val, max_val) = heatmap.computed_range();

        // Calculate cell size
        let cell_width = bounds.width as f32 / cols as f32;
        let cell_height = bounds.height as f32 / rows as f32;

        for row in 0..rows {
            for col in 0..cols {
                let value = heatmap.data[row][col];
                let normalized = if (max_val - min_val).abs() > 1e-10 {
                    (value - min_val) / (max_val - min_val)
                } else {
                    0.5
                };

                let color = colormap.map(normalized);

                // Check if this cell is selected
                let is_selected = heatmap.selected_cells.contains(&(row, col));
                let final_color = if is_selected {
                    // Blend with selection color
                    Color::rgba(
                        color.r * 0.7 + self.theme.selection_color[0] * 0.3,
                        color.g * 0.7 + self.theme.selection_color[1] * 0.3,
                        color.b * 0.7 + self.theme.selection_color[2] * 0.3,
                        color.a,
                    )
                } else {
                    color
                };

                let r = (final_color.r * 255.0) as u8;
                let g = (final_color.g * 255.0) as u8;
                let b = (final_color.b * 255.0) as u8;
                let a = (final_color.a * 255.0) as u8;

                // Draw cell
                let cell_x = bounds.x + (col as f32 * cell_width) as u32;
                let cell_y = bounds.y + (row as f32 * cell_height) as u32;
                let cell_w = (cell_width as u32).max(1);
                let cell_h = (cell_height as u32).max(1);

                for y in cell_y..(cell_y + cell_h).min(bounds.y + bounds.height) {
                    for x in cell_x..(cell_x + cell_w).min(bounds.x + bounds.width) {
                        let idx = ((y * width + x) * 4) as usize;
                        if idx + 3 < buffer.len() {
                            buffer[idx] = r;
                            buffer[idx + 1] = g;
                            buffer[idx + 2] = b;
                            buffer[idx + 3] = a;
                        }
                    }
                }
            }
        }
    }

    /// Render a chart.
    fn render_chart(&self, chart: &ChartView, bounds: &PanelBounds, buffer: &mut [u8], width: u32) {
        if chart.series.is_empty() {
            return;
        }

        let (x_range, y_range) = chart.compute_ranges();
        let x_range = chart.x_range.unwrap_or(x_range);
        let y_range = chart.y_range.unwrap_or(y_range);

        // Draw grid if enabled
        if chart.show_grid {
            self.draw_chart_grid(bounds, buffer, width, 10, 8);
        }

        // Draw each series
        for series in &chart.series {
            if !series.visible || series.data.is_empty() {
                continue;
            }

            let color = series.color;
            let r = (color[0] * 255.0) as u8;
            let g = (color[1] * 255.0) as u8;
            let b = (color[2] * 255.0) as u8;
            let a = (color[3] * 255.0) as u8;

            // Draw line segments
            let mut prev_pixel: Option<(u32, u32)> = None;
            for &(x, y) in &series.data {
                // Transform to pixel coordinates
                let px = bounds.x
                    + ((x - x_range.0) / (x_range.1 - x_range.0) * bounds.width as f32) as u32;
                let py = bounds.y + bounds.height
                    - ((y - y_range.0) / (y_range.1 - y_range.0) * bounds.height as f32) as u32;

                // Clamp to bounds
                let px = px.clamp(bounds.x, bounds.x + bounds.width - 1);
                let py = py.clamp(bounds.y, bounds.y + bounds.height - 1);

                // Draw line from previous point
                if let Some((prev_x, prev_y)) = prev_pixel {
                    self.draw_line(buffer, width, prev_x, prev_y, px, py, r, g, b, a);
                }

                prev_pixel = Some((px, py));
            }
        }

        // Draw markers
        for marker in &chart.markers {
            let px = bounds.x
                + ((marker.x - x_range.0) / (x_range.1 - x_range.0) * bounds.width as f32) as u32;

            if px >= bounds.x && px < bounds.x + bounds.width {
                let r = (marker.color[0] * 255.0) as u8;
                let g = (marker.color[1] * 255.0) as u8;
                let b = (marker.color[2] * 255.0) as u8;
                let a = (marker.color[3] * 255.0) as u8;

                // Draw vertical line
                for y in bounds.y..(bounds.y + bounds.height) {
                    let idx = ((y * width + px) * 4) as usize;
                    if idx + 3 < buffer.len() {
                        buffer[idx] = r;
                        buffer[idx + 1] = g;
                        buffer[idx + 2] = b;
                        buffer[idx + 3] = a;
                    }
                }
            }
        }
    }

    /// Draw chart grid lines.
    fn draw_chart_grid(
        &self,
        bounds: &PanelBounds,
        buffer: &mut [u8],
        width: u32,
        x_divisions: u32,
        y_divisions: u32,
    ) {
        let grid_color = [0.3, 0.3, 0.4, 0.5];
        let r = (grid_color[0] * 255.0) as u8;
        let g = (grid_color[1] * 255.0) as u8;
        let b = (grid_color[2] * 255.0) as u8;
        let a = (grid_color[3] * 255.0) as u8;

        // Vertical grid lines
        for i in 0..=x_divisions {
            let x = bounds.x + (i * bounds.width / x_divisions);
            for y in bounds.y..(bounds.y + bounds.height) {
                let idx = ((y * width + x) * 4) as usize;
                if idx + 3 < buffer.len() {
                    buffer[idx] = r;
                    buffer[idx + 1] = g;
                    buffer[idx + 2] = b;
                    buffer[idx + 3] = a;
                }
            }
        }

        // Horizontal grid lines
        for i in 0..=y_divisions {
            let y = bounds.y + (i * bounds.height / y_divisions);
            for x in bounds.x..(bounds.x + bounds.width) {
                let idx = ((y * width + x) * 4) as usize;
                if idx + 3 < buffer.len() {
                    buffer[idx] = r;
                    buffer[idx + 1] = g;
                    buffer[idx + 2] = b;
                    buffer[idx + 3] = a;
                }
            }
        }
    }

    /// Draw a line using Bresenham's algorithm.
    #[allow(clippy::too_many_arguments)]
    fn draw_line(
        &self,
        buffer: &mut [u8],
        width: u32,
        x0: u32,
        y0: u32,
        x1: u32,
        y1: u32,
        r: u8,
        g: u8,
        b: u8,
        a: u8,
    ) {
        let dx = (x1 as i32 - x0 as i32).abs();
        let dy = -(y1 as i32 - y0 as i32).abs();
        let sx = if x0 < x1 { 1i32 } else { -1i32 };
        let sy = if y0 < y1 { 1i32 } else { -1i32 };
        let mut err = dx + dy;

        let mut x = x0 as i32;
        let mut y = y0 as i32;

        loop {
            if x >= 0 && y >= 0 {
                let idx = ((y as u32 * width + x as u32) * 4) as usize;
                if idx + 3 < buffer.len() {
                    buffer[idx] = r;
                    buffer[idx + 1] = g;
                    buffer[idx + 2] = b;
                    buffer[idx + 3] = a;
                }
            }

            if x == x1 as i32 && y == y1 as i32 {
                break;
            }

            let e2 = 2 * err;
            if e2 >= dy {
                err += dy;
                x += sx;
            }
            if e2 <= dx {
                err += dx;
                y += sy;
            }
        }
    }

    /// Render gradient flow visualization.
    fn render_gradient_flow(
        &self,
        flow: &GradientFlowData,
        bounds: &PanelBounds,
        buffer: &mut [u8],
        width: u32,
    ) {
        if flow.layers.is_empty() || flow.gradient_norms.is_empty() {
            return;
        }

        let max_norm = flow
            .gradient_norms
            .iter()
            .cloned()
            .fold(0.0f32, f32::max)
            .max(1e-6);

        let bar_width = bounds.width / flow.layers.len() as u32;
        let padding = 2;

        for (i, &norm) in flow.gradient_norms.iter().enumerate() {
            let normalized = norm / max_norm;
            let bar_height = (normalized * bounds.height as f32) as u32;

            // Color based on gradient magnitude
            let color = if normalized > 0.8 {
                self.theme.error_color // Too high
            } else if normalized < 0.1 {
                [0.5, 0.5, 0.5, 1.0] // Too low (vanishing)
            } else {
                self.theme.success_color // Good range
            };

            let r = (color[0] * 255.0) as u8;
            let g = (color[1] * 255.0) as u8;
            let b = (color[2] * 255.0) as u8;
            let a = (color[3] * 255.0) as u8;

            let bar_x = bounds.x + i as u32 * bar_width + padding;
            let bar_y = bounds.y + bounds.height - bar_height;

            for y in bar_y..(bounds.y + bounds.height) {
                for x in bar_x..(bar_x + bar_width - padding * 2) {
                    let idx = ((y * width + x) * 4) as usize;
                    if idx + 3 < buffer.len() {
                        buffer[idx] = r;
                        buffer[idx + 1] = g;
                        buffer[idx + 2] = b;
                        buffer[idx + 3] = a;
                    }
                }
            }
        }
    }

    /// Render layer activations visualization.
    fn render_activations(
        &self,
        activations: &LayerActivationData,
        bounds: &PanelBounds,
        buffer: &mut [u8],
        width: u32,
    ) {
        if activations.histograms.is_empty() {
            return;
        }

        // Render as stacked histograms
        let layer_count = activations.histograms.len();
        let layer_height = bounds.height / layer_count as u32;

        for (layer_idx, histogram) in activations.histograms.iter().enumerate() {
            if histogram.is_empty() {
                continue;
            }

            let max_val = histogram.iter().cloned().fold(0.0f32, f32::max).max(1e-6);
            let bin_width = bounds.width / histogram.len() as u32;
            let layer_y = bounds.y + layer_idx as u32 * layer_height;

            let is_selected = activations.selected_layer == Some(layer_idx);
            let base_color = if is_selected {
                self.theme.accent_color
            } else {
                [0.3, 0.5, 0.8, 0.8]
            };

            for (bin_idx, &value) in histogram.iter().enumerate() {
                let normalized = value / max_val;
                let bar_height = (normalized * (layer_height - 4) as f32) as u32;

                let r = (base_color[0] * 255.0) as u8;
                let g = (base_color[1] * 255.0) as u8;
                let b = (base_color[2] * 255.0) as u8;
                let a = (base_color[3] * 255.0) as u8;

                let bar_x = bounds.x + bin_idx as u32 * bin_width;
                let bar_y = layer_y + layer_height - 2 - bar_height;

                for y in bar_y..(layer_y + layer_height - 2) {
                    for x in bar_x..(bar_x + bin_width.saturating_sub(1)) {
                        let idx = ((y * width + x) * 4) as usize;
                        if idx + 3 < buffer.len() {
                            buffer[idx] = r;
                            buffer[idx + 1] = g;
                            buffer[idx + 2] = b;
                            buffer[idx + 3] = a;
                        }
                    }
                }
            }
        }
    }

    /// Render a placeholder for 3D content.
    fn render_3d_placeholder(&self, bounds: &PanelBounds, buffer: &mut [u8], width: u32) {
        // Draw a simple indicator that this would be 3D content
        let color = self.theme.accent_color;
        let r = (color[0] * 255.0) as u8;
        let g = (color[1] * 255.0) as u8;
        let b = (color[2] * 255.0) as u8;
        let a = (color[3] * 128.0) as u8; // Semi-transparent

        // Draw diagonal lines as a placeholder pattern
        for y in bounds.y..(bounds.y + bounds.height) {
            for x in bounds.x..(bounds.x + bounds.width) {
                let local_x = x - bounds.x;
                let local_y = y - bounds.y;
                if (local_x + local_y) % 20 < 2 {
                    let idx = ((y * width + x) * 4) as usize;
                    if idx + 3 < buffer.len() {
                        buffer[idx] = r;
                        buffer[idx + 1] = g;
                        buffer[idx + 2] = b;
                        buffer[idx + 3] = a;
                    }
                }
            }
        }
    }

    /// Render a placeholder for unimplemented content.
    fn render_placeholder(&self, bounds: &PanelBounds, buffer: &mut [u8], width: u32) {
        // Draw a subtle pattern indicating placeholder content
        let color = [0.3, 0.3, 0.35, 0.3];
        let r = (color[0] * 255.0) as u8;
        let g = (color[1] * 255.0) as u8;
        let b = (color[2] * 255.0) as u8;
        let a = (color[3] * 255.0) as u8;

        for y in bounds.y..(bounds.y + bounds.height) {
            for x in bounds.x..(bounds.x + bounds.width) {
                let local_x = x - bounds.x;
                let local_y = y - bounds.y;
                if (local_x / 10 + local_y / 10) % 2 == 0 {
                    let idx = ((y * width + x) * 4) as usize;
                    if idx + 3 < buffer.len() {
                        buffer[idx] = r;
                        buffer[idx + 1] = g;
                        buffer[idx + 2] = b;
                        buffer[idx + 3] = a;
                    }
                }
            }
        }
    }

    // ========================================================================
    // Export
    // ========================================================================

    /// Export the dashboard to a PNG file.
    pub fn export_png(&self, path: &Path, width: u32, height: u32) -> Result<()> {
        let pixels = self.render(width, height);

        // Write PNG using a simple implementation
        // In production, you'd use the `png` or `image` crate
        Self::write_png(path, &pixels, width, height)
    }

    /// Write pixel data as PNG.
    fn write_png(path: &Path, pixels: &[u8], width: u32, height: u32) -> Result<()> {
        use std::fs::File;
        use std::io::BufWriter;

        // This is a placeholder. In production, use the `png` crate:
        // let file = File::create(path)?;
        // let writer = BufWriter::new(file);
        // let mut encoder = png::Encoder::new(writer, width, height);
        // encoder.set_color(png::ColorType::Rgba);
        // encoder.set_depth(png::BitDepth::Eight);
        // let mut writer = encoder.write_header()?;
        // writer.write_image_data(pixels)?;

        // For now, write raw RGBA data (not a valid PNG)
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        // Write a simple header for identification
        use std::io::Write;
        writeln!(writer, "RGBA {} {}", width, height)?;
        writer.write_all(pixels)?;

        Ok(())
    }

    /// Export the dashboard to an SVG file.
    pub fn export_svg(&self, path: &Path) -> Result<()> {
        use std::fs::File;
        use std::io::{BufWriter, Write};

        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        // Calculate SVG dimensions based on layout
        let svg_width = self.layout.grid_cols as f32 * 300.0 + self.layout.margin * 2.0;
        let svg_height = self.layout.grid_rows as f32 * 200.0 + self.layout.margin * 2.0;

        // Write SVG header
        writeln!(
            writer,
            r#"<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}">
  <defs>
    <style>
      .panel-bg {{ fill: rgba({}, {}, {}, {}); }}
      .panel-border {{ stroke: rgba({}, {}, {}, {}); stroke-width: {}; fill: none; }}
      .title {{ fill: rgba({}, {}, {}, {}); font-family: sans-serif; font-size: {}px; }}
    </style>
  </defs>
  <rect width="100%" height="100%" fill="rgba({}, {}, {}, {})"/>"#,
            svg_width,
            svg_height,
            svg_width,
            svg_height,
            (self.theme.panel_background[0] * 255.0) as u8,
            (self.theme.panel_background[1] * 255.0) as u8,
            (self.theme.panel_background[2] * 255.0) as u8,
            self.theme.panel_background[3],
            (self.theme.border_color[0] * 255.0) as u8,
            (self.theme.border_color[1] * 255.0) as u8,
            (self.theme.border_color[2] * 255.0) as u8,
            self.theme.border_color[3],
            self.theme.border_width,
            (self.theme.text_color[0] * 255.0) as u8,
            (self.theme.text_color[1] * 255.0) as u8,
            (self.theme.text_color[2] * 255.0) as u8,
            self.theme.text_color[3],
            self.theme.title_font_size,
            (self.theme.background[0] * 255.0) as u8,
            (self.theme.background[1] * 255.0) as u8,
            (self.theme.background[2] * 255.0) as u8,
            self.theme.background[3],
        )?;

        // Write each panel as an SVG group
        for panel_id in &self.panel_order {
            if let Some(panel) = self.panels.get(panel_id) {
                if panel.visible {
                    self.write_panel_svg(&mut writer, panel, svg_width as u32, svg_height as u32)?;
                }
            }
        }

        // Close SVG
        writeln!(writer, "</svg>")?;

        Ok(())
    }

    /// Write a panel to SVG.
    fn write_panel_svg<W: std::io::Write>(
        &self,
        writer: &mut W,
        panel: &DashboardPanel,
        width: u32,
        height: u32,
    ) -> Result<()> {
        use std::io::Write;

        let bounds = panel.pixel_bounds(&self.layout, width, height);

        writeln!(writer, r#"  <g id="{}">"#, panel.id)?;

        // Panel background
        writeln!(
            writer,
            r#"    <rect x="{}" y="{}" width="{}" height="{}" class="panel-bg" rx="{}"/>"#,
            bounds.x, bounds.y, bounds.width, bounds.height, self.theme.corner_radius
        )?;

        // Panel border
        if panel.show_border {
            writeln!(
                writer,
                r#"    <rect x="{}" y="{}" width="{}" height="{}" class="panel-border" rx="{}"/>"#,
                bounds.x, bounds.y, bounds.width, bounds.height, self.theme.corner_radius
            )?;
        }

        // Panel title
        if panel.show_title {
            writeln!(
                writer,
                r#"    <text x="{}" y="{}" class="title">{}</text>"#,
                bounds.x + 10,
                bounds.y + 18,
                panel.title
            )?;
        }

        // Content-specific SVG rendering would go here
        // For now, just add a placeholder comment
        writeln!(
            writer,
            "    <!-- {} content would be rendered here -->",
            format!("{:?}", panel.panel_type)
        )?;

        writeln!(writer, "  </g>")?;

        Ok(())
    }
}

impl Default for VisualizationDashboard {
    fn default() -> Self {
        Self::new(DashboardLayout::default())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dashboard_creation() {
        let layout = DashboardLayout::new(3, 2);
        let dashboard = VisualizationDashboard::new(layout);

        assert_eq!(dashboard.panel_count(), 0);
        assert_eq!(dashboard.layout.grid_cols, 3);
        assert_eq!(dashboard.layout.grid_rows, 2);
    }

    #[test]
    fn test_add_panel() {
        let mut dashboard = VisualizationDashboard::new(DashboardLayout::new(3, 2));

        let panel = DashboardPanel::new(
            "test",
            PanelType::MetricsChart,
            PanelPosition::new(0, 0),
            PanelSize::unit(),
        );

        assert!(dashboard.add_panel(panel).is_ok());
        assert_eq!(dashboard.panel_count(), 1);
        assert!(dashboard.get_panel("test").is_some());
    }

    #[test]
    fn test_panel_overlap_detection() {
        let mut dashboard = VisualizationDashboard::new(DashboardLayout::new(3, 2));

        let panel1 = DashboardPanel::new(
            "p1",
            PanelType::Network3D,
            PanelPosition::new(0, 0),
            PanelSize::new(2, 2),
        );

        let panel2 = DashboardPanel::new(
            "p2",
            PanelType::MetricsChart,
            PanelPosition::new(1, 0),
            PanelSize::unit(),
        );

        assert!(dashboard.add_panel(panel1).is_ok());
        // This should fail due to overlap
        assert!(dashboard.add_panel(panel2).is_err());
    }

    #[test]
    fn test_panel_out_of_bounds() {
        let mut dashboard = VisualizationDashboard::new(DashboardLayout::new(2, 2));

        let panel = DashboardPanel::new(
            "oob",
            PanelType::MetricsChart,
            PanelPosition::new(5, 5),
            PanelSize::unit(),
        );

        assert!(matches!(
            dashboard.add_panel(panel),
            Err(DashboardError::OutOfBounds { .. })
        ));
    }

    #[test]
    fn test_preset_dashboards() {
        let training = VisualizationDashboard::preset_training_monitor();
        assert!(training.panel_count() > 0);

        let inspection = VisualizationDashboard::preset_model_inspection();
        assert!(inspection.panel_count() > 0);

        let attention = VisualizationDashboard::preset_attention_analysis();
        assert!(attention.panel_count() > 0);
    }

    #[test]
    fn test_selection() {
        let mut selection = Selection::new();
        assert!(selection.is_empty());

        selection.select_layer("layer1");
        assert!(!selection.is_empty());
        assert!(selection.layers.contains(&"layer1".to_string()));

        selection.select_token(5);
        assert!(selection.tokens.contains(&5));

        selection.clear();
        assert!(selection.is_empty());
    }

    #[test]
    fn test_heatmap_view() {
        let data = vec![
            vec![0.1, 0.5, 0.3],
            vec![0.2, 0.8, 0.4],
            vec![0.6, 0.1, 0.9],
        ];

        let heatmap = HeatmapView {
            data,
            ..Default::default()
        };

        assert_eq!(heatmap.dimensions(), (3, 3));
        let (min, max) = heatmap.computed_range();
        assert!((min - 0.1).abs() < 1e-6);
        assert!((max - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_chart_series() {
        let mut series = ChartSeries::new("test", [1.0, 0.0, 0.0, 1.0]);
        series.add_point(0.0, 1.0);
        series.add_point(1.0, 2.0);
        series.add_point(2.0, 1.5);

        assert_eq!(series.data.len(), 3);
    }

    #[test]
    fn test_chart_ranges() {
        let mut chart = ChartView::default();
        let mut series = ChartSeries::new("loss", [1.0, 0.0, 0.0, 1.0]);
        series.add_points(&[(0.0, 1.0), (1.0, 0.5), (2.0, 0.25)]);
        chart.add_series(series);

        let (x_range, y_range) = chart.compute_ranges();
        assert!((x_range.0 - 0.0).abs() < 1e-6);
        assert!((x_range.1 - 2.0).abs() < 1e-6);
        assert!(y_range.0 < 0.25); // With padding
        assert!(y_range.1 > 1.0); // With padding
    }

    #[test]
    fn test_render_basic() {
        let dashboard = VisualizationDashboard::preset_training_monitor();
        let pixels = dashboard.render(640, 480);

        // Check we got the right amount of data
        assert_eq!(pixels.len(), 640 * 480 * 4);

        // Check background was filled (not all zeros)
        let non_zero = pixels.iter().any(|&p| p != 0);
        assert!(non_zero);
    }

    #[test]
    fn test_theme_presets() {
        let dark = DashboardTheme::dark();
        let light = DashboardTheme::light();
        let high_contrast = DashboardTheme::high_contrast();

        // Dark theme should have dark background
        assert!(dark.background[0] < 0.2);
        // Light theme should have light background
        assert!(light.background[0] > 0.8);
        // High contrast should have black background
        assert!(high_contrast.background[0] < 0.01);
    }

    #[test]
    fn test_panel_pixel_bounds() {
        let layout = DashboardLayout::new(2, 2).with_gap(10.0).with_margin(20.0);

        let panel = DashboardPanel::new(
            "test",
            PanelType::MetricsChart,
            PanelPosition::new(0, 0),
            PanelSize::unit(),
        );

        let bounds = panel.pixel_bounds(&layout, 800, 600);

        // Panel should start after margin
        assert_eq!(bounds.x, 20);
        assert_eq!(bounds.y, 20);

        // Panel width = (800 - 40) / 2 - 10 = 370
        // Panel height = (600 - 40) / 2 - 10 = 270
        assert!(bounds.width > 0);
        assert!(bounds.height > 0);
    }

    #[test]
    fn test_update_animation() {
        let mut dashboard = VisualizationDashboard::new(DashboardLayout::single());

        let mut panel = DashboardPanel::new(
            "3d",
            PanelType::Network3D,
            PanelPosition::origin(),
            PanelSize::unit(),
        );

        panel.content = PanelContent::Neural3D(Neural3DScene::default());
        let _ = dashboard.add_panel(panel);

        dashboard.update(0.016);
        assert!(dashboard.time() > 0.0);
        assert_eq!(dashboard.frame(), 1);

        dashboard.update(0.016);
        assert!(dashboard.time() > 0.016);
        assert_eq!(dashboard.frame(), 2);
    }
}
