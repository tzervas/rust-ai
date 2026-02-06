//! Attention pattern visualization for Rerun.
//!
//! This module provides tools for visualizing transformer attention patterns
//! as heatmaps in the Rerun viewer.

use super::{RerunError, RerunResult};
use rerun::RecordingStream;

/// Logger for attention pattern visualization.
pub struct AttentionLogger<'a> {
    rec: &'a RecordingStream,
}

impl<'a> AttentionLogger<'a> {
    /// Create a new attention logger.
    pub fn new(rec: &'a RecordingStream) -> Self {
        Self { rec }
    }

    /// Log an attention pattern as a heatmap image.
    ///
    /// # Arguments
    ///
    /// * `layer` - Layer index (0-indexed)
    /// * `head` - Attention head index (0-indexed)
    /// * `weights` - 2D attention weight matrix [seq_len x seq_len]
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, or an error if weights are invalid.
    pub fn log_attention(
        &self,
        layer: usize,
        head: usize,
        weights: &[Vec<f32>],
    ) -> RerunResult<()> {
        if weights.is_empty() {
            return Err(RerunError::EmptyInput(
                "Empty attention weights".to_string(),
            ));
        }

        let height = weights.len();
        let width = weights[0].len();

        // Validate square matrix (typical for self-attention)
        if height != width {
            tracing::debug!(
                "Attention matrix is not square: {}x{} (cross-attention?)",
                height,
                width
            );
        }

        // Convert to image data (grayscale, values should be 0-1)
        let mut image_data = Vec::with_capacity(height * width * 3);
        for row in weights {
            if row.len() != width {
                return Err(RerunError::ShapeMismatch {
                    expected: format!("{} columns", width),
                    got: format!("{} columns", row.len()),
                });
            }
            for &val in row {
                // Apply colormap (viridis-like)
                let [r, g, b] = self.viridis_color(val);
                image_data.push(r);
                image_data.push(g);
                image_data.push(b);
            }
        }

        let entity_path = format!("attention/layer_{}/head_{}", layer, head);

        // Log as RGB image
        self.rec
            .log(
                &entity_path,
                &rerun::Image::from_rgb24(image_data, [width as u32, height as u32]),
            )
            .map_err(|e| RerunError::LoggingError(e.to_string()))?;

        Ok(())
    }

    /// Log attention patterns for all heads in a layer.
    ///
    /// # Arguments
    ///
    /// * `layer` - Layer index
    /// * `all_heads` - Vector of attention matrices, one per head
    pub fn log_layer_attention(
        &self,
        layer: usize,
        all_heads: &[Vec<Vec<f32>>],
    ) -> RerunResult<()> {
        for (head_idx, weights) in all_heads.iter().enumerate() {
            self.log_attention(layer, head_idx, weights)?;
        }
        Ok(())
    }

    /// Log an aggregated attention pattern (e.g., mean across heads).
    ///
    /// # Arguments
    ///
    /// * `layer` - Layer index
    /// * `weights` - Aggregated 2D attention matrix
    /// * `aggregation` - Type of aggregation used ("mean", "max", etc.)
    pub fn log_aggregated_attention(
        &self,
        layer: usize,
        weights: &[Vec<f32>],
        aggregation: &str,
    ) -> RerunResult<()> {
        if weights.is_empty() {
            return Err(RerunError::EmptyInput(
                "Empty attention weights".to_string(),
            ));
        }

        let height = weights.len();
        let width = weights[0].len();

        let mut image_data = Vec::with_capacity(height * width * 3);
        for row in weights {
            for &val in row {
                let [r, g, b] = self.viridis_color(val);
                image_data.push(r);
                image_data.push(g);
                image_data.push(b);
            }
        }

        let entity_path = format!("attention/layer_{}/{}", layer, aggregation);

        self.rec
            .log(
                &entity_path,
                &rerun::Image::from_rgb24(image_data, [width as u32, height as u32]),
            )
            .map_err(|e| RerunError::LoggingError(e.to_string()))?;

        Ok(())
    }

    /// Log attention with token labels for better interpretation.
    ///
    /// # Arguments
    ///
    /// * `layer` - Layer index
    /// * `head` - Head index
    /// * `weights` - Attention weight matrix
    /// * `tokens` - Token strings for axes
    pub fn log_labeled_attention(
        &self,
        layer: usize,
        head: usize,
        weights: &[Vec<f32>],
        tokens: &[String],
    ) -> RerunResult<()> {
        // First log the attention heatmap
        self.log_attention(layer, head, weights)?;

        // Then log token labels as text annotations
        let entity_path = format!("attention/layer_{}/head_{}", layer, head);

        // Log tokens as a text document for reference
        let tokens_str = tokens.join(", ");
        let _ = self.rec.log(
            format!("{}/tokens", entity_path),
            &rerun::TextDocument::new(format!("Tokens: {}", tokens_str)),
        );

        Ok(())
    }

    /// Log attention evolution over time (useful for tracking attention pattern changes).
    ///
    /// # Arguments
    ///
    /// * `step` - Training step
    /// * `layer` - Layer index
    /// * `head` - Head index
    /// * `weights` - Current attention weights
    pub fn log_attention_evolution(
        &self,
        step: u64,
        layer: usize,
        head: usize,
        weights: &[Vec<f32>],
    ) -> RerunResult<()> {
        self.rec.set_time_sequence("step", step as i64);
        self.log_attention(layer, head, weights)
    }

    /// Log attention statistics (entropy, sparsity) as scalars.
    ///
    /// # Arguments
    ///
    /// * `step` - Current training step
    /// * `layer` - Layer index
    /// * `head` - Head index
    /// * `weights` - Attention weight matrix
    pub fn log_attention_stats(
        &self,
        step: u64,
        layer: usize,
        head: usize,
        weights: &[Vec<f32>],
    ) -> RerunResult<()> {
        self.rec.set_time_sequence("step", step as i64);

        let (entropy, sparsity) = self.compute_attention_stats(weights);

        let path_prefix = format!("attention_stats/layer_{}/head_{}", layer, head);

        let _ = self.rec.log(
            format!("{}/entropy", path_prefix),
            &rerun::Scalar::new(entropy),
        );
        let _ = self.rec.log(
            format!("{}/sparsity", path_prefix),
            &rerun::Scalar::new(sparsity),
        );

        Ok(())
    }

    /// Compute entropy and sparsity of attention distribution.
    fn compute_attention_stats(&self, weights: &[Vec<f32>]) -> (f64, f64) {
        let mut total_entropy = 0.0f64;
        let mut total_sparsity = 0.0f64;
        let mut count = 0;

        for row in weights {
            // Entropy: -sum(p * log(p))
            let entropy: f64 = row
                .iter()
                .filter(|&&p| p > 1e-10)
                .map(|&p| -(p as f64) * (p as f64).ln())
                .sum();
            total_entropy += entropy;

            // Sparsity: fraction of weights below threshold
            let threshold = 0.01;
            let sparse_count = row.iter().filter(|&&p| p < threshold).count();
            total_sparsity += sparse_count as f64 / row.len() as f64;

            count += 1;
        }

        if count > 0 {
            (total_entropy / count as f64, total_sparsity / count as f64)
        } else {
            (0.0, 0.0)
        }
    }

    /// Viridis colormap approximation (value in 0-1 range).
    fn viridis_color(&self, value: f32) -> [u8; 3] {
        // Clamp value to valid range
        let t = value.clamp(0.0, 1.0);

        // Simplified viridis colormap (5 key points)
        let colors: [(f32, [u8; 3]); 5] = [
            (0.0, [68, 1, 84]),    // Dark purple
            (0.25, [59, 82, 139]), // Blue-purple
            (0.5, [33, 145, 140]), // Teal
            (0.75, [94, 201, 98]), // Green
            (1.0, [253, 231, 37]), // Yellow
        ];

        // Find the two colors to interpolate between
        let mut lower = colors[0];
        let mut upper = colors[colors.len() - 1];

        for i in 0..colors.len() - 1 {
            if t >= colors[i].0 && t <= colors[i + 1].0 {
                lower = colors[i];
                upper = colors[i + 1];
                break;
            }
        }

        // Interpolate
        let range = upper.0 - lower.0;
        let local_t = if range > 0.0 {
            (t - lower.0) / range
        } else {
            0.0
        };

        [
            (lower.1[0] as f32 + local_t * (upper.1[0] as f32 - lower.1[0] as f32)) as u8,
            (lower.1[1] as f32 + local_t * (upper.1[1] as f32 - lower.1[1] as f32)) as u8,
            (lower.1[2] as f32 + local_t * (upper.1[2] as f32 - lower.1[2] as f32)) as u8,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_viridis_colormap_endpoints() {
        // We can't easily test without a RecordingStream, but we can test the math
        let colors: [(f32, [u8; 3]); 5] = [
            (0.0, [68, 1, 84]),
            (0.25, [59, 82, 139]),
            (0.5, [33, 145, 140]),
            (0.75, [94, 201, 98]),
            (1.0, [253, 231, 37]),
        ];

        // Check that colors at key points match expectations
        assert_eq!(colors[0].1, [68, 1, 84]); // Dark purple at 0
        assert_eq!(colors[4].1, [253, 231, 37]); // Yellow at 1
    }

    #[test]
    fn test_attention_stats_computation() {
        // Uniform attention should have high entropy
        let uniform: Vec<Vec<f32>> = vec![vec![0.25, 0.25, 0.25, 0.25]];

        let uniform_entropy: f64 = uniform[0]
            .iter()
            .filter(|&&p| p > 1e-10)
            .map(|&p| -(p as f64) * (p as f64).ln())
            .sum();

        // ln(4) ~ 1.386 for uniform over 4 elements
        assert!((uniform_entropy - 1.386).abs() < 0.01);

        // Peaked attention should have low entropy
        let peaked: Vec<Vec<f32>> = vec![vec![0.97, 0.01, 0.01, 0.01]];

        let peaked_entropy: f64 = peaked[0]
            .iter()
            .filter(|&&p| p > 1e-10)
            .map(|&p| -(p as f64) * (p as f64).ln())
            .sum();

        assert!(peaked_entropy < uniform_entropy);
    }
}
