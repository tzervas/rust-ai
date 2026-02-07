//! 3D Neural Network Visualization Module
//!
//! This module provides 3D visualization for neural networks:
//! - Layer visualization as 3D meshes (boxes, spheres)
//! - Connection rendering between layers (lines, curves)
//! - Attention pattern flow (particles)
//! - Embedding point clouds
//! - Real-time training data streaming
//! - Chord diagrams for attention pattern connectivity
//!
//! # Bevy-based Interactive Viewer
//!
//! When the `viz3d` feature is enabled, a full Bevy-based 3D viewer is available:
//!
//! ```rust,ignore
//! use training_tools::viz3d::{Viz3dApp, NetworkConfig};
//!
//! let config = NetworkConfig::default();
//! Viz3dApp::run(config);
//! ```
//!
//! # Standalone Engine
//!
//! Without the `viz3d` feature, a standalone engine using nalgebra is available:
//!
//! ```rust,ignore
//! use training_tools::viz3d::{Viz3DEngine, Mesh3D};
//!
//! let mut engine = Viz3DEngine::new();
//! engine.add_mesh(Mesh3D::sphere("sphere1", ...));
//! ```

pub mod architecture;
pub mod attention;
pub mod chord;
pub mod colors;
pub mod dashboard;
pub mod dense_network;
pub mod embeddings;
pub mod engine;
pub mod landscape;
pub mod network;
pub mod neural_3d;

// Bevy-based modules (require viz3d feature)
#[cfg(feature = "viz3d")]
mod bevy_app;
#[cfg(feature = "viz3d")]
mod camera;
#[cfg(feature = "viz3d")]
mod connections;
#[cfg(feature = "viz3d")]
mod layers;
#[cfg(feature = "viz3d")]
mod ui_overlay;

pub use architecture::{
    isometric_project, ArchitectureDiagram, ConnectionType, LayerBlock, LayerConnection, LayerInfo,
    LayerType, LayoutStyle, SkipConnection, SkipStyle, TritterConfig,
};
pub use chord::{
    colormap_coolwarm, colormap_plasma, colormap_viridis, ChordConfig, ChordConnection,
    ChordDiagram, ChordSegment,
};
pub use colors::{
    cividis, coolwarm, inferno, magma, plasma, rainbow, tab10, viridis, CategoricalPalette, Color,
    Colormap, ColormapPreset,
};
pub use dashboard::{
    ActivationStats, ArchBlock, ArchBlockType, ArchConnection,
    ArchitectureDiagram as DashboardArchDiagram, ChartMarker, ChartSeries, ChartView,
    ChordDiagramData, ConnectionData, DashboardError, DashboardLayout, DashboardPanel,
    DashboardTheme, EmbeddingCloudData, GradientFlowData, HeatmapView, LayerActivationData,
    LayerMeshData, LossLandscapeData, Neural3DScene as DashboardNeural3DScene, PanelBounds,
    PanelContent, PanelPosition, PanelSettings, PanelSize, PanelType, Selection, SelectionItem,
    TokenCloudData, VisualizationDashboard,
};
pub use dense_network::{
    ConnectionColorMode, ConnectionInstance, DenseLayer, DenseNetworkViz, DenseVizConfig,
    GridArrangement, NetworkStats, WeightMatrix,
};
pub use engine::{Camera3D, Mesh3D, Vertex3D, Viz3DConfig, Viz3DEngine};
pub use neural_3d::{
    Camera3D as Neural3DCamera, Connection3D, Neural3DScene, Node3D, Particle, ParticleSystem,
    RenderConfig as Neural3DRenderConfig,
};

// New visualization modules
pub use attention::{
    AttentionFlow3D, AttentionFlowConfig, AttentionHead, AttentionLayer, AttentionStats, Token,
};
pub use embeddings::{
    Cluster, DimReductionMethod, EmbeddingCloud3D, EmbeddingCloudConfig, EmbeddingCloudStats,
    EmbeddingPoint, PointLabels,
};
pub use engine::{Light3D, LightKind, ObjectId, SceneStats};
pub use landscape::{
    LandscapeStats, LossLandscape3D, LossLandscapeConfig, SurfaceSource, TrajectoryPoint,
};
pub use network::{
    LayerConnection as NetworkLayerConnection, LayerType as NetworkLayerType, NetworkGraph3D,
    NetworkGraphConfig, NetworkLayer,
};

// Bevy-based exports (require viz3d feature)
#[cfg(feature = "viz3d")]
pub use bevy_app::{NetworkConfig, Viz3dApp, Viz3dPlugin};
#[cfg(feature = "viz3d")]
pub use camera::{CameraController, CameraMode, OrbitCamera};
#[cfg(feature = "viz3d")]
pub use connections::{Connection, ConnectionBundle, ConnectionRenderer, ConnectionStyle};
#[cfg(feature = "viz3d")]
pub use layers::{LayerMesh, LayerMeshBuilder, LayerShape, LayerStyle, NeuralLayer};
#[cfg(feature = "viz3d")]
pub use ui_overlay::{ControlPanel, UiOverlay, UiState};

use thiserror::Error;

/// Errors that can occur during 3D visualization.
#[derive(Debug, Error)]
pub enum Viz3dError {
    #[error("Invalid network configuration: {0}")]
    InvalidConfig(String),

    #[error("Layer not found: {0}")]
    LayerNotFound(String),

    #[error("Connection error: {0}")]
    ConnectionError(String),

    #[error("Render error: {0}")]
    RenderError(String),

    #[error("Data streaming error: {0}")]
    StreamError(String),
}

/// Result type for viz3d operations.
pub type Viz3dResult<T> = Result<T, Viz3dError>;

/// Gradient magnitude for color coding.
#[derive(Debug, Clone, Copy, Default)]
pub struct GradientMagnitude {
    /// Gradient value normalized to [0, 1].
    pub value: f32,
    /// Whether this gradient is considered "hot" (high magnitude).
    pub is_hot: bool,
}

impl GradientMagnitude {
    /// Create a new gradient magnitude.
    pub fn new(value: f32) -> Self {
        let normalized = value.clamp(0.0, 1.0);
        Self {
            value: normalized,
            is_hot: normalized > 0.7,
        }
    }

    /// Convert to color (blue -> green -> yellow -> red).
    pub fn to_color(&self) -> [f32; 4] {
        let v = self.value;
        if v < 0.25 {
            // Blue to cyan
            let t = v * 4.0;
            [0.0, t, 1.0, 1.0]
        } else if v < 0.5 {
            // Cyan to green
            let t = (v - 0.25) * 4.0;
            [0.0, 1.0, 1.0 - t, 1.0]
        } else if v < 0.75 {
            // Green to yellow
            let t = (v - 0.5) * 4.0;
            [t, 1.0, 0.0, 1.0]
        } else {
            // Yellow to red
            let t = (v - 0.75) * 4.0;
            [1.0, 1.0 - t, 0.0, 1.0]
        }
    }
}

/// Animation state for forward/backward pass visualization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AnimationPhase {
    /// No animation running.
    #[default]
    Idle,
    /// Forward pass animation (input -> output).
    Forward,
    /// Backward pass animation (output -> input).
    Backward,
    /// Both forward and backward (training loop).
    Training,
}

/// Training data point for real-time streaming.
#[derive(Debug, Clone, Default)]
pub struct TrainingDataPoint {
    /// Current training step.
    pub step: u64,
    /// Loss value.
    pub loss: f32,
    /// Gradient norms per layer.
    pub gradient_norms: Vec<f32>,
    /// Activation statistics per layer.
    pub activation_stats: Vec<(f32, f32)>, // (mean, std)
    /// Attention weights (if applicable).
    pub attention_weights: Option<Vec<Vec<f32>>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gradient_magnitude() {
        let low = GradientMagnitude::new(0.2);
        assert!(!low.is_hot);

        let high = GradientMagnitude::new(0.9);
        assert!(high.is_hot);

        let clamped = GradientMagnitude::new(1.5);
        assert!((clamped.value - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_gradient_color() {
        let low = GradientMagnitude::new(0.0);
        let color = low.to_color();
        // Should be blue
        assert!(color[2] > color[0]);

        let high = GradientMagnitude::new(1.0);
        let color = high.to_color();
        // Should be red
        assert!(color[0] > color[1]);
    }
}
