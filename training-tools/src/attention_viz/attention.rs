//! Attention pattern structures and analysis.
//!
//! Provides data structures for representing transformer attention patterns
//! and utilities for analyzing attention across heads and layers.

use super::{VizError, VizResult};
use serde::{Deserialize, Serialize};

/// Represents a single attention head's weights.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionHead {
    /// Attention weights matrix [seq_len, seq_len].
    /// weights[i][j] represents attention from position i to position j.
    pub weights: Vec<Vec<f32>>,
    /// Index of this head within its layer.
    pub head_idx: usize,
}

impl AttentionHead {
    /// Create a new attention head with the given weights.
    pub fn new(weights: Vec<Vec<f32>>, head_idx: usize) -> VizResult<Self> {
        // Validate square matrix
        let seq_len = weights.len();
        if seq_len == 0 {
            return Err(VizError::EmptySequence);
        }

        for (i, row) in weights.iter().enumerate() {
            if row.len() != seq_len {
                return Err(VizError::ShapeMismatch {
                    expected: format!("row {} to have {} columns", i, seq_len),
                    got: format!("{} columns", row.len()),
                });
            }
        }

        Ok(Self { weights, head_idx })
    }

    /// Get the sequence length this attention head operates on.
    pub fn seq_len(&self) -> usize {
        self.weights.len()
    }

    /// Get attention weight from position `from` to position `to`.
    pub fn get_attention(&self, from: usize, to: usize) -> Option<f32> {
        self.weights.get(from).and_then(|row| row.get(to).copied())
    }

    /// Get the maximum attention weight in this head.
    pub fn max_attention(&self) -> f32 {
        self.weights
            .iter()
            .flat_map(|row| row.iter())
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max)
    }

    /// Get the minimum attention weight in this head.
    pub fn min_attention(&self) -> f32 {
        self.weights
            .iter()
            .flat_map(|row| row.iter())
            .cloned()
            .fold(f32::INFINITY, f32::min)
    }

    /// Get the mean attention weight in this head.
    pub fn mean_attention(&self) -> f32 {
        let total: f32 = self.weights.iter().flat_map(|row| row.iter()).sum();
        let count = self.weights.len() * self.weights.len();
        total / count as f32
    }

    /// Get attention entropy for each query position.
    /// Higher entropy means more uniform attention distribution.
    pub fn attention_entropy(&self) -> Vec<f32> {
        self.weights
            .iter()
            .map(|row| {
                let sum: f32 = row.iter().sum();
                if sum == 0.0 {
                    return 0.0;
                }
                -row.iter()
                    .map(|&w| {
                        let p = w / sum;
                        if p > 0.0 {
                            p * p.ln()
                        } else {
                            0.0
                        }
                    })
                    .sum::<f32>()
            })
            .collect()
    }

    /// Find the top-k attended positions for each query position.
    pub fn top_k_attended(&self, k: usize) -> Vec<Vec<(usize, f32)>> {
        self.weights
            .iter()
            .map(|row| {
                let mut indexed: Vec<(usize, f32)> =
                    row.iter().enumerate().map(|(i, &w)| (i, w)).collect();
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                indexed.truncate(k);
                indexed
            })
            .collect()
    }

    /// Check if this head exhibits diagonal (local) attention pattern.
    /// Returns a score from 0.0 (not diagonal) to 1.0 (perfectly diagonal).
    pub fn diagonal_score(&self) -> f32 {
        let seq_len = self.seq_len();
        if seq_len == 0 {
            return 0.0;
        }

        let mut diagonal_sum = 0.0;
        let mut total_sum = 0.0;

        for (i, row) in self.weights.iter().enumerate() {
            for (j, &w) in row.iter().enumerate() {
                total_sum += w;
                // Consider positions within 1 step of diagonal
                if i.abs_diff(j) <= 1 {
                    diagonal_sum += w;
                }
            }
        }

        if total_sum == 0.0 {
            0.0
        } else {
            diagonal_sum / total_sum
        }
    }
}

/// Methods for aggregating attention across multiple heads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HeadAggregation {
    /// Take the mean across all heads.
    Mean,
    /// Take the maximum across all heads.
    Max,
    /// Take the minimum across all heads.
    Min,
    /// Sum all heads (useful for total attention).
    Sum,
}

/// Attention pattern for a specific transformer layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionPattern {
    /// Name or index identifier for this pattern.
    pub name: String,
    /// All attention heads at this layer.
    pub heads: Vec<AttentionHead>,
    /// Sequence length.
    pub seq_len: usize,
    /// Token labels (if available).
    pub tokens: Option<Vec<String>>,
}

impl AttentionPattern {
    /// Create a new attention pattern.
    pub fn new(name: String, heads: Vec<AttentionHead>) -> VizResult<Self> {
        if heads.is_empty() {
            return Err(VizError::InvalidWeights(
                "Must have at least one attention head".to_string(),
            ));
        }

        let seq_len = heads[0].seq_len();
        for (i, head) in heads.iter().enumerate() {
            if head.seq_len() != seq_len {
                return Err(VizError::ShapeMismatch {
                    expected: format!("head {} to have seq_len {}", i, seq_len),
                    got: format!("seq_len {}", head.seq_len()),
                });
            }
        }

        Ok(Self {
            name,
            heads,
            seq_len,
            tokens: None,
        })
    }

    /// Set token labels for visualization.
    pub fn with_tokens(mut self, tokens: Vec<String>) -> VizResult<Self> {
        if tokens.len() != self.seq_len {
            return Err(VizError::ShapeMismatch {
                expected: format!("{} tokens", self.seq_len),
                got: format!("{} tokens", tokens.len()),
            });
        }
        self.tokens = Some(tokens);
        Ok(self)
    }

    /// Number of attention heads.
    pub fn num_heads(&self) -> usize {
        self.heads.len()
    }

    /// Aggregate attention across all heads using the specified method.
    pub fn aggregate(&self, method: HeadAggregation) -> Vec<Vec<f32>> {
        let mut result = vec![vec![0.0; self.seq_len]; self.seq_len];

        match method {
            HeadAggregation::Mean => {
                for head in &self.heads {
                    for (i, row) in head.weights.iter().enumerate() {
                        for (j, &w) in row.iter().enumerate() {
                            result[i][j] += w / self.heads.len() as f32;
                        }
                    }
                }
            }
            HeadAggregation::Max => {
                for i in 0..self.seq_len {
                    for j in 0..self.seq_len {
                        result[i][j] = self
                            .heads
                            .iter()
                            .map(|h| h.weights[i][j])
                            .fold(f32::NEG_INFINITY, f32::max);
                    }
                }
            }
            HeadAggregation::Min => {
                for i in 0..self.seq_len {
                    for j in 0..self.seq_len {
                        result[i][j] = self
                            .heads
                            .iter()
                            .map(|h| h.weights[i][j])
                            .fold(f32::INFINITY, f32::min);
                    }
                }
            }
            HeadAggregation::Sum => {
                for head in &self.heads {
                    for (i, row) in head.weights.iter().enumerate() {
                        for (j, &w) in row.iter().enumerate() {
                            result[i][j] += w;
                        }
                    }
                }
            }
        }

        result
    }
}

/// Complete attention visualization for a transformer model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionViz {
    /// Attention heads at this layer.
    pub heads: Vec<AttentionHead>,
    /// Layer index in the model.
    pub layer_idx: usize,
    /// Token labels for sequence positions.
    pub tokens: Option<Vec<String>>,
    /// Batch index (if from a batch).
    pub batch_idx: Option<usize>,
}

impl AttentionViz {
    /// Create a new attention visualization.
    pub fn new(heads: Vec<AttentionHead>, layer_idx: usize) -> VizResult<Self> {
        if heads.is_empty() {
            return Err(VizError::InvalidWeights(
                "Must have at least one attention head".to_string(),
            ));
        }

        // Validate all heads have same sequence length
        let seq_len = heads[0].seq_len();
        for (i, head) in heads.iter().enumerate() {
            if head.seq_len() != seq_len {
                return Err(VizError::ShapeMismatch {
                    expected: format!("head {} to have seq_len {}", i, seq_len),
                    got: format!("seq_len {}", head.seq_len()),
                });
            }
        }

        Ok(Self {
            heads,
            layer_idx,
            tokens: None,
            batch_idx: None,
        })
    }

    /// Set token labels for this visualization.
    pub fn with_tokens(mut self, tokens: Vec<String>) -> Self {
        self.tokens = Some(tokens);
        self
    }

    /// Set batch index for this visualization.
    pub fn with_batch_idx(mut self, batch_idx: usize) -> Self {
        self.batch_idx = Some(batch_idx);
        self
    }

    /// Get the sequence length.
    pub fn seq_len(&self) -> usize {
        self.heads.first().map(|h| h.seq_len()).unwrap_or(0)
    }

    /// Get the number of heads.
    pub fn num_heads(&self) -> usize {
        self.heads.len()
    }

    /// Get a specific head by index.
    pub fn get_head(&self, idx: usize) -> Option<&AttentionHead> {
        self.heads.get(idx)
    }

    /// Convert to an AttentionPattern for more analysis.
    pub fn to_pattern(&self) -> AttentionPattern {
        let name = format!("Layer {}", self.layer_idx);
        let mut pattern = AttentionPattern {
            name,
            heads: self.heads.clone(),
            seq_len: self.seq_len(),
            tokens: self.tokens.clone(),
        };
        if let Some(ref tokens) = self.tokens {
            pattern.tokens = Some(tokens.clone());
        }
        pattern
    }

    /// Get aggregated attention weights.
    pub fn aggregate(&self, method: HeadAggregation) -> Vec<Vec<f32>> {
        self.to_pattern().aggregate(method)
    }

    /// Find heads that focus on specific patterns.
    pub fn analyze_heads(&self) -> Vec<HeadAnalysis> {
        self.heads
            .iter()
            .map(|head| {
                let diagonal_score = head.diagonal_score();
                let entropy = head.attention_entropy();
                let mean_entropy = entropy.iter().sum::<f32>() / entropy.len() as f32;

                let pattern_type = if diagonal_score > 0.7 {
                    HeadPatternType::Local
                } else if mean_entropy > 2.0 {
                    HeadPatternType::Broad
                } else {
                    HeadPatternType::Focused
                };

                HeadAnalysis {
                    head_idx: head.head_idx,
                    diagonal_score,
                    mean_entropy,
                    pattern_type,
                }
            })
            .collect()
    }
}

/// Analysis results for a single attention head.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeadAnalysis {
    /// Head index.
    pub head_idx: usize,
    /// Score indicating how diagonal/local the attention pattern is.
    pub diagonal_score: f32,
    /// Mean attention entropy across positions.
    pub mean_entropy: f32,
    /// Detected attention pattern type.
    pub pattern_type: HeadPatternType,
}

/// Classification of attention head patterns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HeadPatternType {
    /// Local/diagonal attention (attends to nearby positions).
    Local,
    /// Broad/diffuse attention (relatively uniform).
    Broad,
    /// Focused attention (attends to specific positions).
    Focused,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_attention_weights(seq_len: usize) -> Vec<Vec<f32>> {
        // Create a softmax-like distribution for each row
        (0..seq_len)
            .map(|i| {
                let mut row = vec![0.0; seq_len];
                // Focus attention on position i and neighbors
                for j in 0..seq_len {
                    let dist = (i as i32 - j as i32).abs() as f32;
                    row[j] = (-dist).exp();
                }
                // Normalize to sum to 1
                let sum: f32 = row.iter().sum();
                row.iter_mut().for_each(|x| *x /= sum);
                row
            })
            .collect()
    }

    #[test]
    fn test_attention_head_creation() {
        let weights = sample_attention_weights(4);
        let head = AttentionHead::new(weights.clone(), 0).unwrap();
        assert_eq!(head.seq_len(), 4);
        assert_eq!(head.head_idx, 0);
    }

    #[test]
    fn test_attention_head_validation() {
        // Empty weights should fail
        let result = AttentionHead::new(vec![], 0);
        assert!(result.is_err());

        // Non-square matrix should fail
        let result = AttentionHead::new(vec![vec![0.5, 0.5], vec![0.3, 0.4, 0.3]], 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_attention_head_statistics() {
        let weights = vec![vec![0.1, 0.9], vec![0.5, 0.5]];
        let head = AttentionHead::new(weights, 0).unwrap();

        assert_eq!(head.max_attention(), 0.9);
        assert_eq!(head.min_attention(), 0.1);
        assert!((head.mean_attention() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_attention_head_top_k() {
        let weights = vec![vec![0.1, 0.5, 0.3, 0.1], vec![0.25, 0.25, 0.25, 0.25]];
        // Pad to make square
        let weights: Vec<Vec<f32>> = (0..4)
            .map(|i| {
                if i < 2 {
                    let mut row = weights[i].clone();
                    row.resize(4, 0.0);
                    row
                } else {
                    vec![0.25; 4]
                }
            })
            .collect();

        let head = AttentionHead::new(weights, 0).unwrap();
        let top_2 = head.top_k_attended(2);

        assert_eq!(top_2[0][0].0, 1); // Position 1 has highest weight (0.5)
        assert_eq!(top_2[0][1].0, 2); // Position 2 has second highest (0.3)
    }

    #[test]
    fn test_attention_viz_creation() {
        let heads: Vec<AttentionHead> = (0..4)
            .map(|i| AttentionHead::new(sample_attention_weights(8), i).unwrap())
            .collect();

        let viz = AttentionViz::new(heads, 0).unwrap();
        assert_eq!(viz.num_heads(), 4);
        assert_eq!(viz.seq_len(), 8);
        assert_eq!(viz.layer_idx, 0);
    }

    #[test]
    fn test_head_analysis() {
        let heads: Vec<AttentionHead> = (0..2)
            .map(|i| AttentionHead::new(sample_attention_weights(8), i).unwrap())
            .collect();

        let viz = AttentionViz::new(heads, 0).unwrap();
        let analysis = viz.analyze_heads();

        assert_eq!(analysis.len(), 2);
        // Our sample weights are diagonal-focused
        assert!(analysis[0].diagonal_score > 0.5);
    }

    #[test]
    fn test_aggregation_methods() {
        let weights1 = vec![vec![0.2, 0.8], vec![0.6, 0.4]];
        let weights2 = vec![vec![0.4, 0.6], vec![0.3, 0.7]];

        let head1 = AttentionHead::new(weights1, 0).unwrap();
        let head2 = AttentionHead::new(weights2, 1).unwrap();

        let viz = AttentionViz::new(vec![head1, head2], 0).unwrap();

        // Test mean aggregation
        let mean = viz.aggregate(HeadAggregation::Mean);
        assert!((mean[0][0] - 0.3).abs() < 0.01); // (0.2 + 0.4) / 2

        // Test max aggregation
        let max = viz.aggregate(HeadAggregation::Max);
        assert!((max[0][0] - 0.4).abs() < 0.01);
        assert!((max[0][1] - 0.8).abs() < 0.01);
    }
}
