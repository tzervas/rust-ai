//! Inference Visualization Module
//!
//! Provides comprehensive visualization tools for understanding how transformer
//! models process input during inference.
//!
//! # Features
//!
//! - **Token Flow**: Track how tokens flow through layers
//! - **Activation Patterns**: Visualize activation magnitudes and statistics
//! - **Attention Contributions**: Show which attention heads contribute most
//! - **Output Analysis**: Display probability distributions and top predictions
//! - **3D Flow Visualization**: Generate 3D representations of information flow
//!
//! # Example
//!
//! ```rust,ignore
//! use training_tools::inference_viz::{InferenceViz, ModelConfig};
//! use candle_core::Tensor;
//!
//! // Create visualizer with model configuration
//! let config = ModelConfig::new(12, 768, 12, 50257);
//! let mut viz = InferenceViz::new(config);
//!
//! // During forward pass, record activations
//! viz.record_layer(0, &layer_0_output);
//! viz.record_attention(0, 0, &attention_weights);
//! viz.record_output(&logits);
//!
//! // Generate visualizations
//! let token_flow = viz.get_token_flow(0);
//! let heatmap = viz.get_layer_heatmap(0);
//! let top_k = viz.get_top_predictions(10);
//! ```

mod flow;
mod heatmap;
mod predictions;
mod recorder;
mod render;

pub use flow::{AttentionGraph, TokenFlow};
pub use heatmap::LayerHeatmap;
pub use predictions::{OutputAnalysis, PredictionEntry};
pub use recorder::{ActivationRecorder, AttentionRecorder, LayerRecord};
pub use render::{FlowNode3D, FlowNodeType, FlowRenderer3D, RenderConfig};

use candle_core::{Device, Tensor};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors that can occur during inference visualization.
#[derive(Debug, Error)]
pub enum InferenceVizError {
    #[error("Invalid layer index: {index} (model has {num_layers} layers)")]
    InvalidLayerIndex { index: usize, num_layers: usize },

    #[error("Invalid token index: {index} (sequence has {seq_len} tokens)")]
    InvalidTokenIndex { index: usize, seq_len: usize },

    #[error("No activations recorded for layer {0}")]
    NoActivationsForLayer(usize),

    #[error("No attention patterns recorded")]
    NoAttentionPatterns,

    #[error("No output logits recorded")]
    NoOutputLogits,

    #[error("Shape mismatch: expected {expected}, got {got}")]
    ShapeMismatch { expected: String, got: String },

    #[error("Tensor error: {0}")]
    TensorError(#[from] candle_core::Error),

    #[error("Empty sequence: cannot visualize empty input")]
    EmptySequence,

    #[error("Missing tokenizer for text decoding")]
    MissingTokenizer,
}

/// Result type for inference visualization operations.
pub type Result<T> = std::result::Result<T, InferenceVizError>;

/// Configuration describing the model architecture.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Number of transformer layers
    pub num_layers: usize,
    /// Hidden dimension size
    pub hidden_size: usize,
    /// Number of attention heads per layer
    pub num_attention_heads: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// Intermediate size (for MLP layers)
    pub intermediate_size: usize,
}

impl ModelConfig {
    /// Create a new model configuration.
    pub fn new(
        num_layers: usize,
        hidden_size: usize,
        num_attention_heads: usize,
        vocab_size: usize,
    ) -> Self {
        Self {
            num_layers,
            hidden_size,
            num_attention_heads,
            vocab_size,
            max_seq_len: 2048,
            intermediate_size: hidden_size * 4,
        }
    }

    /// Create configuration with custom intermediate size.
    pub fn with_intermediate(mut self, intermediate_size: usize) -> Self {
        self.intermediate_size = intermediate_size;
        self
    }

    /// Create configuration with custom max sequence length.
    pub fn with_max_seq_len(mut self, max_seq_len: usize) -> Self {
        self.max_seq_len = max_seq_len;
        self
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        // GPT-2 small defaults
        Self::new(12, 768, 12, 50257)
    }
}

/// Type of layer in the transformer model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LayerType {
    /// Embedding layer
    Embedding,
    /// Self-attention layer
    Attention,
    /// Layer normalization
    LayerNorm,
    /// Feed-forward MLP layer
    FeedForward,
    /// Residual connection
    Residual,
    /// Output projection (language model head)
    Output,
}

impl std::fmt::Display for LayerType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LayerType::Embedding => write!(f, "Embedding"),
            LayerType::Attention => write!(f, "Attention"),
            LayerType::LayerNorm => write!(f, "LayerNorm"),
            LayerType::FeedForward => write!(f, "FeedForward"),
            LayerType::Residual => write!(f, "Residual"),
            LayerType::Output => write!(f, "Output"),
        }
    }
}

/// Statistics for activation values.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct ActivationStats {
    /// Mean activation value
    pub mean: f32,
    /// Standard deviation
    pub std: f32,
    /// Maximum activation value
    pub max: f32,
    /// Minimum activation value
    pub min: f32,
    /// Fraction of near-zero values (|x| < threshold)
    pub sparsity: f32,
}

impl ActivationStats {
    /// Compute statistics from activation values.
    pub fn from_values(values: &[f32], sparsity_threshold: f32) -> Self {
        if values.is_empty() {
            return Self::default();
        }

        let n = values.len() as f32;
        let mean = values.iter().sum::<f32>() / n;

        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
        let std = variance.sqrt();

        let max = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min = values.iter().cloned().fold(f32::INFINITY, f32::min);

        let near_zero_count = values
            .iter()
            .filter(|&&x| x.abs() < sparsity_threshold)
            .count();
        let sparsity = near_zero_count as f32 / n;

        Self {
            mean,
            std,
            max,
            min,
            sparsity,
        }
    }

    /// Compute statistics from a tensor.
    pub fn from_tensor(tensor: &Tensor) -> Result<Self> {
        let values: Vec<f32> = tensor.flatten_all()?.to_vec1()?;
        Ok(Self::from_values(&values, 0.01))
    }
}

/// Layer activation data with input/output and statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerActivation {
    /// Index of the layer
    pub layer_idx: usize,
    /// Type of the layer
    pub layer_type: LayerType,
    /// Input activation values (flattened or summarized)
    pub input_activations: Vec<f32>,
    /// Output activation values (flattened or summarized)
    pub output_activations: Vec<f32>,
    /// Statistics for input activations
    pub input_stats: ActivationStats,
    /// Statistics for output activations
    pub output_stats: ActivationStats,
}

impl LayerActivation {
    /// Create a new layer activation record.
    pub fn new(
        layer_idx: usize,
        layer_type: LayerType,
        input_activations: Vec<f32>,
        output_activations: Vec<f32>,
    ) -> Self {
        let input_stats = ActivationStats::from_values(&input_activations, 0.01);
        let output_stats = ActivationStats::from_values(&output_activations, 0.01);

        Self {
            layer_idx,
            layer_type,
            input_activations,
            output_activations,
            input_stats,
            output_stats,
        }
    }

    /// Compute the change in activation magnitude through this layer.
    pub fn magnitude_change(&self) -> f32 {
        if self.input_stats.mean.abs() < 1e-10 {
            return 0.0;
        }
        (self.output_stats.mean - self.input_stats.mean) / self.input_stats.mean.abs()
    }
}

/// Attention pattern from a single head.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionPattern {
    /// Layer index
    pub layer_idx: usize,
    /// Head index within the layer
    pub head_idx: usize,
    /// Attention weights matrix (seq_len x seq_len)
    pub weights: Vec<Vec<f32>>,
    /// Sequence length
    pub seq_len: usize,
}

impl AttentionPattern {
    /// Create a new attention pattern.
    pub fn new(layer_idx: usize, head_idx: usize, weights: Vec<Vec<f32>>) -> Self {
        let seq_len = weights.len();
        Self {
            layer_idx,
            head_idx,
            weights,
            seq_len,
        }
    }

    /// Get attention from source to target position.
    pub fn attention(&self, source: usize, target: usize) -> Option<f32> {
        self.weights
            .get(source)
            .and_then(|row| row.get(target).copied())
    }

    /// Get total attention received by a position (column sum).
    pub fn attention_received(&self, position: usize) -> f32 {
        self.weights
            .iter()
            .map(|row| row.get(position).unwrap_or(&0.0))
            .sum()
    }

    /// Get total attention given by a position (row sum, should be ~1.0).
    pub fn attention_given(&self, position: usize) -> f32 {
        self.weights
            .get(position)
            .map(|row| row.iter().sum())
            .unwrap_or(0.0)
    }

    /// Find the position receiving maximum attention from a given source.
    pub fn max_attention_target(&self, source: usize) -> Option<(usize, f32)> {
        self.weights.get(source).and_then(|row| {
            row.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, &val)| (idx, val))
        })
    }

    /// Compute average attention span (how far attention reaches on average).
    pub fn average_attention_span(&self) -> f32 {
        let mut total_span = 0.0f32;
        let mut count = 0;

        for (i, row) in self.weights.iter().enumerate() {
            for (j, &weight) in row.iter().enumerate() {
                if weight > 0.01 {
                    total_span += (i as f32 - j as f32).abs() * weight;
                    count += 1;
                }
            }
        }

        if count == 0 {
            0.0
        } else {
            total_span / count as f32
        }
    }
}

/// Main inference visualization struct.
///
/// Collects activation data during forward pass and generates visualizations.
pub struct InferenceViz {
    /// Model configuration
    pub model_config: ModelConfig,
    /// Recorded layer activations
    pub layer_activations: Vec<LayerActivation>,
    /// Recorded attention patterns
    pub attention_patterns: Vec<AttentionPattern>,
    /// Output logits (vocabulary probabilities)
    pub output_logits: Vec<f32>,
    /// Token texts (if available)
    pub tokens: Vec<String>,
    /// Token IDs
    pub token_ids: Vec<u32>,
    /// Device for tensor operations
    device: Device,
}

impl InferenceViz {
    /// Create a new inference visualizer with the given model configuration.
    pub fn new(model_config: ModelConfig) -> Self {
        Self {
            model_config,
            layer_activations: Vec::new(),
            attention_patterns: Vec::new(),
            output_logits: Vec::new(),
            tokens: Vec::new(),
            token_ids: Vec::new(),
            device: Device::Cpu,
        }
    }

    /// Set the device for tensor operations.
    pub fn with_device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    /// Set the input tokens (for visualization labels).
    pub fn set_tokens(&mut self, tokens: Vec<String>, token_ids: Vec<u32>) {
        self.tokens = tokens;
        self.token_ids = token_ids;
    }

    /// Clear all recorded data for a new inference.
    pub fn clear(&mut self) {
        self.layer_activations.clear();
        self.attention_patterns.clear();
        self.output_logits.clear();
        self.tokens.clear();
        self.token_ids.clear();
    }

    /// Record layer activations during forward pass.
    ///
    /// # Arguments
    /// * `layer_idx` - Index of the layer (0-based)
    /// * `layer_type` - Type of the layer
    /// * `input` - Input tensor to the layer
    /// * `output` - Output tensor from the layer
    pub fn record_layer(
        &mut self,
        layer_idx: usize,
        layer_type: LayerType,
        input: &Tensor,
        output: &Tensor,
    ) -> Result<()> {
        // Flatten and convert to Vec<f32>
        let input_vals: Vec<f32> = input.flatten_all()?.to_vec1()?;
        let output_vals: Vec<f32> = output.flatten_all()?.to_vec1()?;

        let activation = LayerActivation::new(layer_idx, layer_type, input_vals, output_vals);
        self.layer_activations.push(activation);

        Ok(())
    }

    /// Record layer activations with just the output tensor.
    ///
    /// Uses the previous layer's output as input (or zeros for first layer).
    pub fn record_layer_output(
        &mut self,
        layer_idx: usize,
        layer_type: LayerType,
        output: &Tensor,
    ) -> Result<()> {
        let output_vals: Vec<f32> = output.flatten_all()?.to_vec1()?;

        // Use previous layer's output as input, or empty for first layer
        let input_vals = if layer_idx > 0 && !self.layer_activations.is_empty() {
            self.layer_activations
                .last()
                .map(|l| l.output_activations.clone())
                .unwrap_or_default()
        } else {
            Vec::new()
        };

        let activation = LayerActivation::new(layer_idx, layer_type, input_vals, output_vals);
        self.layer_activations.push(activation);

        Ok(())
    }

    /// Record attention weights from a specific layer and head.
    ///
    /// # Arguments
    /// * `layer_idx` - Layer index
    /// * `head_idx` - Attention head index
    /// * `weights` - Attention weight tensor (seq_len x seq_len)
    pub fn record_attention(
        &mut self,
        layer_idx: usize,
        head_idx: usize,
        weights: &Tensor,
    ) -> Result<()> {
        let shape = weights.dims();
        if shape.len() < 2 {
            return Err(InferenceVizError::ShapeMismatch {
                expected: "2D tensor (seq_len x seq_len)".to_string(),
                got: format!("{:?}", shape),
            });
        }

        let seq_len = shape[shape.len() - 2];
        let weights_2d: Vec<Vec<f32>> = (0..seq_len)
            .map(|i| {
                weights
                    .narrow(shape.len() - 2, i, 1)
                    .map(|t| t.flatten_all())
                    .and_then(|t| t.map(|t| t.to_vec1()))
                    .map(|v| v.unwrap_or_default())
                    .unwrap_or_default()
            })
            .collect();

        let pattern = AttentionPattern::new(layer_idx, head_idx, weights_2d);
        self.attention_patterns.push(pattern);

        Ok(())
    }

    /// Record output logits for prediction analysis.
    ///
    /// # Arguments
    /// * `logits` - Output logits tensor (vocab_size or batch x seq x vocab)
    pub fn record_output(&mut self, logits: &Tensor) -> Result<()> {
        // Get the last token's logits if batched
        let last_logits = if logits.dims().len() > 1 {
            let seq_dim = logits.dims().len() - 2;
            let seq_len = logits.dim(seq_dim)?;
            logits.narrow(seq_dim, seq_len - 1, 1)?.squeeze(seq_dim)?
        } else {
            logits.clone()
        };

        self.output_logits = last_logits.flatten_all()?.to_vec1()?;
        Ok(())
    }

    /// Get token flow analysis for a specific token position.
    pub fn get_token_flow(&self, token_idx: usize) -> Result<TokenFlow> {
        if self.tokens.is_empty() {
            return Err(InferenceVizError::EmptySequence);
        }

        if token_idx >= self.tokens.len() {
            return Err(InferenceVizError::InvalidTokenIndex {
                index: token_idx,
                seq_len: self.tokens.len(),
            });
        }

        flow::compute_token_flow(self, token_idx)
    }

    /// Get heatmap data for a specific layer.
    pub fn get_layer_heatmap(&self, layer_idx: usize) -> Result<Vec<Vec<f32>>> {
        if layer_idx >= self.model_config.num_layers {
            return Err(InferenceVizError::InvalidLayerIndex {
                index: layer_idx,
                num_layers: self.model_config.num_layers,
            });
        }

        heatmap::compute_layer_heatmap(self, layer_idx)
    }

    /// Get the full attention graph for all layers.
    pub fn get_attention_graph(&self) -> Result<AttentionGraph> {
        if self.attention_patterns.is_empty() {
            return Err(InferenceVizError::NoAttentionPatterns);
        }

        flow::compute_attention_graph(self)
    }

    /// Get top K predictions with probabilities.
    ///
    /// # Arguments
    /// * `k` - Number of top predictions to return
    /// * `vocab` - Optional vocabulary for token decoding
    pub fn get_top_predictions(
        &self,
        k: usize,
        vocab: Option<&[String]>,
    ) -> Result<Vec<(String, f32)>> {
        if self.output_logits.is_empty() {
            return Err(InferenceVizError::NoOutputLogits);
        }

        predictions::get_top_k_predictions(&self.output_logits, k, vocab)
    }

    /// Generate 3D flow visualization data.
    pub fn to_3d_flow(&self) -> Result<Vec<FlowNode3D>> {
        render::generate_3d_flow(self)
    }

    /// Get activation statistics for all recorded layers.
    pub fn get_all_activation_stats(
        &self,
    ) -> Vec<(usize, LayerType, ActivationStats, ActivationStats)> {
        self.layer_activations
            .iter()
            .map(|la| (la.layer_idx, la.layer_type, la.input_stats, la.output_stats))
            .collect()
    }

    /// Get attention patterns aggregated by layer.
    pub fn get_attention_by_layer(&self, layer_idx: usize) -> Vec<&AttentionPattern> {
        self.attention_patterns
            .iter()
            .filter(|p| p.layer_idx == layer_idx)
            .collect()
    }

    /// Compute overall model interpretability metrics.
    pub fn interpretability_summary(&self) -> InterpretabilitySummary {
        // Compute average sparsity across all layers
        let avg_sparsity = if self.layer_activations.is_empty() {
            0.0
        } else {
            self.layer_activations
                .iter()
                .map(|la| la.output_stats.sparsity)
                .sum::<f32>()
                / self.layer_activations.len() as f32
        };

        // Compute attention concentration (how focused attention is)
        let attention_concentration = if self.attention_patterns.is_empty() {
            0.0
        } else {
            self.attention_patterns
                .iter()
                .map(|p| {
                    p.weights
                        .iter()
                        .map(|row| {
                            let max = row.iter().cloned().fold(0.0f32, f32::max);
                            max
                        })
                        .sum::<f32>()
                        / p.seq_len as f32
                })
                .sum::<f32>()
                / self.attention_patterns.len() as f32
        };

        // Compute layer-wise magnitude changes
        let magnitude_changes: Vec<f32> = self
            .layer_activations
            .iter()
            .map(|la| la.magnitude_change())
            .collect();

        InterpretabilitySummary {
            avg_sparsity,
            attention_concentration,
            magnitude_changes,
            num_layers_recorded: self.layer_activations.len(),
            num_attention_heads_recorded: self.attention_patterns.len(),
        }
    }
}

/// Summary of model interpretability metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterpretabilitySummary {
    /// Average activation sparsity (0-1, higher = more sparse)
    pub avg_sparsity: f32,
    /// Average attention concentration (0-1, higher = more focused)
    pub attention_concentration: f32,
    /// Magnitude change at each layer
    pub magnitude_changes: Vec<f32>,
    /// Number of layers recorded
    pub num_layers_recorded: usize,
    /// Number of attention heads recorded
    pub num_attention_heads_recorded: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_config_default() {
        let config = ModelConfig::default();
        assert_eq!(config.num_layers, 12);
        assert_eq!(config.hidden_size, 768);
        assert_eq!(config.num_attention_heads, 12);
    }

    #[test]
    fn test_activation_stats() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = ActivationStats::from_values(&values, 0.01);

        assert!((stats.mean - 3.0).abs() < 1e-6);
        assert!((stats.max - 5.0).abs() < 1e-6);
        assert!((stats.min - 1.0).abs() < 1e-6);
        assert!((stats.sparsity - 0.0).abs() < 1e-6); // No values near zero
    }

    #[test]
    fn test_activation_stats_with_sparsity() {
        let values = vec![0.0, 0.001, 0.005, 1.0, 2.0];
        let stats = ActivationStats::from_values(&values, 0.01);

        // 3 out of 5 values are near zero
        assert!((stats.sparsity - 0.6).abs() < 1e-6);
    }

    #[test]
    fn test_attention_pattern() {
        let weights = vec![
            vec![0.5, 0.3, 0.2],
            vec![0.1, 0.6, 0.3],
            vec![0.2, 0.2, 0.6],
        ];

        let pattern = AttentionPattern::new(0, 0, weights);

        assert_eq!(pattern.attention(0, 1), Some(0.3));
        assert!((pattern.attention_given(0) - 1.0).abs() < 1e-6);
        assert_eq!(pattern.max_attention_target(0), Some((0, 0.5)));
    }

    #[test]
    fn test_layer_activation() {
        let input = vec![1.0, 2.0, 3.0];
        let output = vec![2.0, 4.0, 6.0];

        let activation = LayerActivation::new(0, LayerType::FeedForward, input, output);

        assert_eq!(activation.layer_idx, 0);
        assert!((activation.input_stats.mean - 2.0).abs() < 1e-6);
        assert!((activation.output_stats.mean - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_inference_viz_new() {
        let config = ModelConfig::new(6, 512, 8, 32000);
        let viz = InferenceViz::new(config);

        assert_eq!(viz.model_config.num_layers, 6);
        assert!(viz.layer_activations.is_empty());
        assert!(viz.attention_patterns.is_empty());
    }
}
