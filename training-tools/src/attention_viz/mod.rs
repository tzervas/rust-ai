//! Attention Pattern and Network Architecture Visualization
//!
//! This module provides visualization tools for transformer models:
//! - Attention pattern heat maps (2D)
//! - Flow diagrams (3D connections between tokens)
//! - Head-by-head breakdown
//! - Network architecture visualization
//! - Gradient flow analysis

mod attention;
mod flow;
mod heatmap;
mod network;

pub use attention::{
    AttentionHead, AttentionPattern, AttentionViz, HeadAggregation, HeadAnalysis, HeadPatternType,
};
pub use flow::{FlowConfig, FlowConnection, FlowDiagram, FlowNode, FlowRenderer, FlowStatistics};
pub use heatmap::{ColorMap, HeatMap, HeatMapConfig, HeatMapRenderer};
pub use network::{
    Connection, ConnectionType, GradientHealth, LayerType, LayerViz, NetworkArchViz,
    NetworkRenderer, NetworkVizConfig,
};

use thiserror::Error;

/// Errors that can occur during visualization operations.
#[derive(Debug, Error)]
pub enum VizError {
    #[error("Invalid attention weights: {0}")]
    InvalidWeights(String),

    #[error("Shape mismatch: expected {expected}, got {got}")]
    ShapeMismatch { expected: String, got: String },

    #[error("Empty sequence: cannot visualize attention for empty input")]
    EmptySequence,

    #[error("Invalid layer configuration: {0}")]
    InvalidLayer(String),

    #[error("Render error: {0}")]
    RenderError(String),

    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),
}

/// Result type for visualization operations.
pub type VizResult<T> = Result<T, VizError>;

/// Output format for rendered visualizations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    /// ASCII art for terminal display
    Ascii,
    /// SVG vector graphics
    Svg,
    /// HTML with embedded CSS/JS for interactive viewing
    Html,
    /// JSON data for external visualization tools
    Json,
}

impl Default for OutputFormat {
    fn default() -> Self {
        Self::Ascii
    }
}

/// Common trait for all visualization renderers.
pub trait Renderer {
    /// The type of visualization this renderer produces.
    type Output: std::fmt::Debug;

    /// Render to the specified format.
    fn render(&self, format: OutputFormat) -> VizResult<Self::Output>;

    /// Render to ASCII (terminal-friendly output).
    fn to_ascii(&self) -> VizResult<String> {
        match self.render(OutputFormat::Ascii) {
            Ok(output) => Ok(format!("{:?}", output)),
            Err(e) => Err(e),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_output_format_default() {
        assert_eq!(OutputFormat::default(), OutputFormat::Ascii);
    }
}
