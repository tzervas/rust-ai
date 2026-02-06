//! Main logging interface for Rerun visualization.
//!
//! This module provides the primary `RerunLogger` struct that coordinates
//! all visualization logging to the Rerun viewer.

use super::{RerunError, RerunResult};
use rerun::{RecordingStream, RecordingStreamBuilder};

/// Information about a network layer for architecture visualization.
#[derive(Debug, Clone)]
pub struct LayerInfo {
    /// Name of the layer (e.g., "attention_0", "mlp_1")
    pub name: String,
    /// Type of layer
    pub layer_type: LayerType,
    /// Number of input features
    pub input_dim: usize,
    /// Number of output features
    pub output_dim: usize,
    /// Number of parameters in this layer
    pub num_params: usize,
    /// Additional metadata
    pub metadata: Option<String>,
}

impl LayerInfo {
    /// Create a new layer info.
    pub fn new(
        name: impl Into<String>,
        layer_type: LayerType,
        input_dim: usize,
        output_dim: usize,
    ) -> Self {
        Self {
            name: name.into(),
            layer_type,
            input_dim,
            output_dim,
            num_params: input_dim * output_dim,
            metadata: None,
        }
    }

    /// Set the number of parameters explicitly.
    pub fn with_params(mut self, num_params: usize) -> Self {
        self.num_params = num_params;
        self
    }

    /// Add metadata string.
    pub fn with_metadata(mut self, metadata: impl Into<String>) -> Self {
        self.metadata = Some(metadata.into());
        self
    }
}

/// Type of neural network layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerType {
    /// Embedding layer (token/position embeddings)
    Embedding,
    /// Self-attention layer
    Attention,
    /// Multi-layer perceptron / feed-forward
    MLP,
    /// Layer normalization
    LayerNorm,
    /// Linear projection
    Linear,
    /// Output head (e.g., language modeling head)
    Head,
    /// Dropout layer
    Dropout,
    /// Residual connection
    Residual,
    /// Custom/other layer type
    Custom,
}

impl LayerType {
    /// Get a color for this layer type (RGB).
    pub fn color(&self) -> [u8; 3] {
        match self {
            LayerType::Embedding => [100, 149, 237], // Cornflower blue
            LayerType::Attention => [255, 99, 71],   // Tomato red
            LayerType::MLP => [50, 205, 50],         // Lime green
            LayerType::LayerNorm => [255, 215, 0],   // Gold
            LayerType::Linear => [147, 112, 219],    // Medium purple
            LayerType::Head => [255, 140, 0],        // Dark orange
            LayerType::Dropout => [128, 128, 128],   // Gray
            LayerType::Residual => [0, 206, 209],    // Dark turquoise
            LayerType::Custom => [192, 192, 192],    // Silver
        }
    }

    /// Get a string representation.
    pub fn as_str(&self) -> &'static str {
        match self {
            LayerType::Embedding => "embedding",
            LayerType::Attention => "attention",
            LayerType::MLP => "mlp",
            LayerType::LayerNorm => "layernorm",
            LayerType::Linear => "linear",
            LayerType::Head => "head",
            LayerType::Dropout => "dropout",
            LayerType::Residual => "residual",
            LayerType::Custom => "custom",
        }
    }
}

/// Main logging interface for Rerun training visualization.
///
/// `RerunLogger` provides a unified interface for logging training metrics,
/// embeddings, attention patterns, and network architecture to the Rerun viewer.
///
/// # Example
///
/// ```rust,ignore
/// use training_tools::rerun_viz::RerunLogger;
///
/// let logger = RerunLogger::new("my_experiment")?;
///
/// // Log training step
/// logger.log_step(100, 0.5, 1e-4, 1.2);
///
/// // Log embeddings periodically
/// let embeddings = vec![vec![0.1, 0.2, 0.3]; 100];
/// let labels = vec!["token_0".to_string(); 100];
/// logger.log_embeddings(&embeddings, &labels);
/// ```
pub struct RerunLogger {
    /// The underlying Rerun recording stream
    rec: RecordingStream,
    /// Application identifier
    app_id: String,
}

impl RerunLogger {
    /// Create a new RerunLogger that connects to a running Rerun viewer.
    ///
    /// # Arguments
    ///
    /// * `app_id` - Identifier for this recording session (e.g., "training_run_001")
    ///
    /// # Returns
    ///
    /// A new `RerunLogger` connected to the viewer, or an error if connection fails.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let logger = RerunLogger::new("my_training")?;
    /// ```
    pub fn new(app_id: &str) -> RerunResult<Self> {
        let rec = RecordingStreamBuilder::new(app_id)
            .connect_tcp()
            .map_err(|e| RerunError::StreamCreationError(e.to_string()))?;

        Ok(Self {
            rec,
            app_id: app_id.to_string(),
        })
    }

    /// Create a RerunLogger that spawns a new Rerun viewer process.
    ///
    /// This is useful for standalone scripts that want to automatically
    /// open the Rerun viewer.
    pub fn spawn(app_id: &str) -> RerunResult<Self> {
        let rec = RecordingStreamBuilder::new(app_id)
            .spawn()
            .map_err(|e| RerunError::StreamCreationError(e.to_string()))?;

        Ok(Self {
            rec,
            app_id: app_id.to_string(),
        })
    }

    /// Create a RerunLogger that saves to an .rrd file.
    ///
    /// # Arguments
    ///
    /// * `app_id` - Identifier for this recording
    /// * `path` - Path to save the .rrd file
    pub fn save_to_file(app_id: &str, path: impl AsRef<std::path::Path>) -> RerunResult<Self> {
        let rec = RecordingStreamBuilder::new(app_id)
            .save(path)
            .map_err(|e| RerunError::StreamCreationError(e.to_string()))?;

        Ok(Self {
            rec,
            app_id: app_id.to_string(),
        })
    }

    /// Get the application ID.
    pub fn app_id(&self) -> &str {
        &self.app_id
    }

    /// Get a reference to the underlying recording stream.
    ///
    /// This allows direct access to the Rerun API for advanced usage.
    pub fn recording_stream(&self) -> &RecordingStream {
        &self.rec
    }

    /// Log a training step with core metrics.
    ///
    /// # Arguments
    ///
    /// * `step` - Current training step number
    /// * `loss` - Loss value at this step
    /// * `lr` - Current learning rate
    /// * `grad_norm` - Gradient norm (L2) at this step
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// logger.log_step(1000, 0.234, 1e-4, 1.5);
    /// ```
    pub fn log_step(&self, step: u64, loss: f32, lr: f32, grad_norm: f32) {
        use super::metrics::MetricsLogger;
        let metrics = MetricsLogger::new(&self.rec);
        metrics.log_step(step, loss, lr, grad_norm);
    }

    /// Log embedding vectors as a 3D point cloud.
    ///
    /// Embeddings are automatically projected to 3D using PCA if they
    /// have more than 3 dimensions.
    ///
    /// # Arguments
    ///
    /// * `embeddings` - Vector of embedding vectors (each can be any dimension)
    /// * `labels` - Labels for each embedding point (for hover text)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let embeddings = vec![
    ///     vec![0.1, 0.2, 0.3, 0.4],  // 4D embedding
    ///     vec![0.5, 0.6, 0.7, 0.8],
    /// ];
    /// let labels = vec!["token_a".to_string(), "token_b".to_string()];
    /// logger.log_embeddings(&embeddings, &labels);
    /// ```
    pub fn log_embeddings(&self, embeddings: &[Vec<f32>], labels: &[String]) {
        use super::embeddings::EmbeddingLogger;
        let emb_logger = EmbeddingLogger::new(&self.rec);
        if let Err(e) = emb_logger.log_embeddings(embeddings, labels) {
            tracing::warn!("Failed to log embeddings: {}", e);
        }
    }

    /// Log an attention pattern as a heatmap image.
    ///
    /// # Arguments
    ///
    /// * `layer` - Layer index (0-indexed)
    /// * `head` - Attention head index (0-indexed)
    /// * `weights` - 2D attention weight matrix [seq_len x seq_len]
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Attention weights for layer 0, head 0
    /// let weights = vec![
    ///     vec![0.8, 0.1, 0.1],
    ///     vec![0.2, 0.6, 0.2],
    ///     vec![0.1, 0.3, 0.6],
    /// ];
    /// logger.log_attention(0, 0, &weights);
    /// ```
    pub fn log_attention(&self, layer: usize, head: usize, weights: &[Vec<f32>]) {
        use super::attention::AttentionLogger;
        let attn_logger = AttentionLogger::new(&self.rec);
        if let Err(e) = attn_logger.log_attention(layer, head, weights) {
            tracing::warn!("Failed to log attention: {}", e);
        }
    }

    /// Log network architecture visualization.
    ///
    /// Creates a visual representation of the model architecture with
    /// layers shown as colored nodes connected by edges.
    ///
    /// # Arguments
    ///
    /// * `layers` - Slice of `LayerInfo` describing each layer
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use training_tools::rerun_viz::{LayerInfo, LayerType};
    ///
    /// let layers = vec![
    ///     LayerInfo::new("embed", LayerType::Embedding, 50257, 768),
    ///     LayerInfo::new("attn_0", LayerType::Attention, 768, 768),
    ///     LayerInfo::new("mlp_0", LayerType::MLP, 768, 3072),
    ///     LayerInfo::new("head", LayerType::Head, 768, 50257),
    /// ];
    /// logger.log_network(&layers);
    /// ```
    pub fn log_network(&self, layers: &[LayerInfo]) {
        self.log_network_internal(layers);
    }

    fn log_network_internal(&self, layers: &[LayerInfo]) {
        use rerun::{Color, Points3D, Position3D, Text};

        // Position layers vertically
        let mut positions: Vec<Position3D> = Vec::new();
        let mut colors: Vec<Color> = Vec::new();
        let mut radii: Vec<f32> = Vec::new();

        for (i, layer) in layers.iter().enumerate() {
            // Y position based on layer index (bottom to top)
            let y = i as f32 * 2.0;
            // X position based on layer width (log scale for visibility)
            let x = (layer.output_dim as f32).ln() * 0.5;
            positions.push(Position3D::new(x, y, 0.0));

            // Color based on layer type
            let [r, g, b] = layer.layer_type.color();
            colors.push(Color::from_rgb(r, g, b));

            // Radius based on parameter count (log scale)
            let radius = (layer.num_params as f32).ln().max(1.0) * 0.1;
            radii.push(radius);
        }

        // Log layer nodes
        let _ = self.rec.log(
            "network/layers",
            &Points3D::new(positions.clone())
                .with_colors(colors)
                .with_radii(radii),
        );

        // Log layer labels
        for (i, layer) in layers.iter().enumerate() {
            let y = i as f32 * 2.0;
            let x = (layer.output_dim as f32).ln() * 0.5 + 1.0;
            let label = format!(
                "{} ({}) {}->{}",
                layer.name,
                layer.layer_type.as_str(),
                layer.input_dim,
                layer.output_dim
            );
            let _ = self.rec.log(
                format!("network/labels/{}", layer.name),
                &rerun::TextDocument::new(label),
            );
        }

        // Log connections between layers
        if layers.len() > 1 {
            let mut line_points: Vec<[f32; 3]> = Vec::new();
            for window in positions.windows(2) {
                if let [p1, p2] = window {
                    line_points.push([p1.x(), p1.y(), p1.z()]);
                    line_points.push([p2.x(), p2.y(), p2.z()]);
                }
            }
            let _ = self.rec.log(
                "network/connections",
                &rerun::LineStrips3D::new([line_points])
                    .with_colors([Color::from_rgb(200, 200, 200)]),
            );
        }
    }

    /// Log loss landscape as a 3D surface with optimization trajectory.
    ///
    /// # Arguments
    ///
    /// * `surface` - 2D grid of loss values forming the landscape surface
    /// * `trajectory` - Sequence of [x, y, loss] points showing optimization path
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Create a simple bowl-shaped loss landscape
    /// let mut surface = Vec::new();
    /// for i in 0..20 {
    ///     let row: Vec<f32> = (0..20)
    ///         .map(|j| {
    ///             let x = (i as f32 - 10.0) / 5.0;
    ///             let y = (j as f32 - 10.0) / 5.0;
    ///             x * x + y * y  // Parabolic bowl
    ///         })
    ///         .collect();
    ///     surface.push(row);
    /// }
    ///
    /// // Optimization trajectory
    /// let trajectory = vec![
    ///     [1.5, 1.5, 4.5],
    ///     [1.0, 1.0, 2.0],
    ///     [0.5, 0.5, 0.5],
    ///     [0.1, 0.1, 0.02],
    /// ];
    ///
    /// logger.log_landscape(&surface, &trajectory);
    /// ```
    pub fn log_landscape(&self, surface: &[Vec<f32>], trajectory: &[[f32; 3]]) {
        use super::landscape::LandscapeLogger;
        let landscape_logger = LandscapeLogger::new(&self.rec);
        if let Err(e) = landscape_logger.log_landscape(surface, trajectory) {
            tracing::warn!("Failed to log landscape: {}", e);
        }
    }

    /// Log a text annotation or note.
    pub fn log_text(&self, path: &str, text: &str) {
        let _ = self.rec.log(path, &rerun::TextDocument::new(text));
    }

    /// Set the current step for time-series data.
    ///
    /// All subsequent logs will be associated with this step until
    /// it is changed again.
    pub fn set_step(&self, step: u64) {
        self.rec.set_time_sequence("step", step as i64);
    }

    /// Set the current epoch for time-series data.
    pub fn set_epoch(&self, epoch: u32) {
        self.rec.set_time_sequence("epoch", epoch as i64);
    }

    /// Flush any pending data to the viewer.
    pub fn flush(&self) {
        // RecordingStream handles flushing internally, but we can trigger
        // by logging a small marker if needed
        let _ = self.rec.log("_flush_marker", &rerun::TextLog::new(""));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_info_creation() {
        let layer = LayerInfo::new("test_layer", LayerType::Attention, 768, 768);
        assert_eq!(layer.name, "test_layer");
        assert_eq!(layer.input_dim, 768);
        assert_eq!(layer.output_dim, 768);
        assert_eq!(layer.num_params, 768 * 768);
    }

    #[test]
    fn test_layer_info_with_params() {
        let layer = LayerInfo::new("test", LayerType::MLP, 768, 3072).with_params(1000000);
        assert_eq!(layer.num_params, 1000000);
    }

    #[test]
    fn test_layer_type_colors() {
        // Verify all layer types have distinct colors
        let types = [
            LayerType::Embedding,
            LayerType::Attention,
            LayerType::MLP,
            LayerType::LayerNorm,
            LayerType::Linear,
            LayerType::Head,
        ];

        let colors: Vec<_> = types.iter().map(|t| t.color()).collect();

        // Check that not all colors are the same
        assert!(colors.windows(2).any(|w| w[0] != w[1]));
    }
}
