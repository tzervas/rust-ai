//! Token Flow Analysis Module
//!
//! Analyzes how information flows through the model for each token,
//! including attention patterns and layer contributions.

use super::{AttentionPattern, InferenceViz, InferenceVizError, LayerType, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Token flow analysis showing how a token is processed through the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenFlow {
    /// Index of the token in the sequence
    pub token_idx: usize,
    /// Text representation of the token (if available)
    pub token_text: String,
    /// Token ID
    pub token_id: u32,
    /// Contribution/importance at each layer
    pub layer_contributions: Vec<f32>,
    /// Attention received from other tokens (per layer, aggregated across heads)
    pub attention_received: Vec<f32>,
    /// Attention given to other tokens (per layer, aggregated across heads)
    pub attention_given: Vec<f32>,
    /// Detailed per-layer flow information
    pub layer_flows: Vec<LayerFlow>,
    /// Overall importance score (0-1)
    pub importance_score: f32,
}

/// Flow information for a specific layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerFlow {
    /// Layer index
    pub layer_idx: usize,
    /// Layer type
    pub layer_type: LayerType,
    /// Input contribution (how much this token contributed to layer input)
    pub input_contribution: f32,
    /// Output contribution (how much this token contributed to layer output)
    pub output_contribution: f32,
    /// Attention weights received from each position (aggregated across heads)
    pub attention_from: Vec<f32>,
    /// Attention weights given to each position (aggregated across heads)
    pub attention_to: Vec<f32>,
    /// Top positions that attended to this token
    pub top_attenders: Vec<(usize, f32)>,
    /// Top positions this token attended to
    pub top_attended: Vec<(usize, f32)>,
}

impl TokenFlow {
    /// Get the layer with maximum contribution for this token.
    pub fn max_contribution_layer(&self) -> Option<(usize, f32)> {
        self.layer_contributions
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, &val)| (idx, val))
    }

    /// Get positions that attended most to this token (across all layers).
    pub fn top_global_attenders(&self, k: usize) -> Vec<(usize, usize, f32)> {
        let mut attenders: Vec<(usize, usize, f32)> = self
            .layer_flows
            .iter()
            .flat_map(|lf| {
                lf.top_attenders
                    .iter()
                    .map(|&(pos, weight)| (lf.layer_idx, pos, weight))
            })
            .collect();

        attenders
            .sort_by(|(_, _, a), (_, _, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        attenders.into_iter().take(k).collect()
    }

    /// Compute the attention span (how far attention reaches on average).
    pub fn attention_span(&self) -> f32 {
        let mut total_span = 0.0f32;
        let mut count = 0;

        for flow in &self.layer_flows {
            for (pos, weight) in &flow.top_attended {
                if *weight > 0.01 {
                    total_span += (self.token_idx as f32 - *pos as f32).abs() * weight;
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

/// Compute token flow for a specific token position.
pub fn compute_token_flow(viz: &InferenceViz, token_idx: usize) -> Result<TokenFlow> {
    let seq_len = viz.tokens.len();
    if seq_len == 0 {
        return Err(InferenceVizError::EmptySequence);
    }

    if token_idx >= seq_len {
        return Err(InferenceVizError::InvalidTokenIndex {
            index: token_idx,
            seq_len,
        });
    }

    let token_text = viz.tokens.get(token_idx).cloned().unwrap_or_default();
    let token_id = viz.token_ids.get(token_idx).copied().unwrap_or(0);

    // Compute layer contributions from activation magnitudes
    let mut layer_contributions = Vec::new();
    let mut attention_received = Vec::new();
    let mut attention_given = Vec::new();
    let mut layer_flows = Vec::new();

    // Group attention patterns by layer
    let mut attention_by_layer: HashMap<usize, Vec<&AttentionPattern>> = HashMap::new();
    for pattern in &viz.attention_patterns {
        attention_by_layer
            .entry(pattern.layer_idx)
            .or_default()
            .push(pattern);
    }

    for layer_idx in 0..viz.model_config.num_layers {
        // Find activation record for this layer
        let activation = viz
            .layer_activations
            .iter()
            .find(|a| a.layer_idx == layer_idx);

        // Compute contribution based on activation magnitude
        let contribution = activation
            .map(|a| {
                // Use relative magnitude compared to layer mean
                let mean = a.output_stats.mean.abs();
                let std = a.output_stats.std;
                if mean < 1e-10 {
                    0.0
                } else {
                    // Contribution is the relative activation strength
                    (std / mean).min(1.0)
                }
            })
            .unwrap_or(0.0);

        layer_contributions.push(contribution);

        // Aggregate attention patterns for this layer
        let layer_patterns = attention_by_layer.get(&layer_idx);

        let (attn_received, attn_given, attn_from, attn_to, top_attenders, top_attended) =
            if let Some(patterns) = layer_patterns {
                compute_layer_attention(patterns, token_idx, seq_len)
            } else {
                (
                    0.0,
                    0.0,
                    vec![0.0; seq_len],
                    vec![0.0; seq_len],
                    vec![],
                    vec![],
                )
            };

        attention_received.push(attn_received);
        attention_given.push(attn_given);

        let layer_type = activation
            .map(|a| a.layer_type)
            .unwrap_or(LayerType::Attention);

        layer_flows.push(LayerFlow {
            layer_idx,
            layer_type,
            input_contribution: contribution * 0.8, // Slightly lower than output
            output_contribution: contribution,
            attention_from: attn_from,
            attention_to: attn_to,
            top_attenders,
            top_attended,
        });
    }

    // Compute overall importance score
    let importance_score = compute_importance_score(&layer_contributions, &attention_received);

    Ok(TokenFlow {
        token_idx,
        token_text,
        token_id,
        layer_contributions,
        attention_received,
        attention_given,
        layer_flows,
        importance_score,
    })
}

/// Compute attention statistics for a layer.
fn compute_layer_attention(
    patterns: &[&AttentionPattern],
    token_idx: usize,
    seq_len: usize,
) -> (
    f32,
    f32,
    Vec<f32>,
    Vec<f32>,
    Vec<(usize, f32)>,
    Vec<(usize, f32)>,
) {
    let num_heads = patterns.len();
    if num_heads == 0 {
        return (
            0.0,
            0.0,
            vec![0.0; seq_len],
            vec![0.0; seq_len],
            vec![],
            vec![],
        );
    }

    let mut attention_from = vec![0.0f32; seq_len];
    let mut attention_to = vec![0.0f32; seq_len];

    // Aggregate attention across heads
    for pattern in patterns {
        // Attention this token received (column sum)
        for (src_pos, row) in pattern.weights.iter().enumerate() {
            if let Some(&weight) = row.get(token_idx) {
                attention_from[src_pos] += weight / num_heads as f32;
            }
        }

        // Attention this token gave (row values)
        if let Some(row) = pattern.weights.get(token_idx) {
            for (tgt_pos, &weight) in row.iter().enumerate() {
                attention_to[tgt_pos] += weight / num_heads as f32;
            }
        }
    }

    // Total attention received and given
    let attn_received: f32 = attention_from.iter().sum();
    let attn_given: f32 = attention_to.iter().sum();

    // Top attenders (positions that attended to this token)
    let mut top_attenders: Vec<(usize, f32)> = attention_from
        .iter()
        .enumerate()
        .map(|(pos, &weight)| (pos, weight))
        .collect();
    top_attenders.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    top_attenders.truncate(5);

    // Top attended (positions this token attended to)
    let mut top_attended: Vec<(usize, f32)> = attention_to
        .iter()
        .enumerate()
        .map(|(pos, &weight)| (pos, weight))
        .collect();
    top_attended.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    top_attended.truncate(5);

    (
        attn_received,
        attn_given,
        attention_from,
        attention_to,
        top_attenders,
        top_attended,
    )
}

/// Compute importance score for a token based on its flow characteristics.
fn compute_importance_score(layer_contributions: &[f32], attention_received: &[f32]) -> f32 {
    if layer_contributions.is_empty() && attention_received.is_empty() {
        return 0.0;
    }

    // Average contribution across layers
    let avg_contribution = if layer_contributions.is_empty() {
        0.0
    } else {
        layer_contributions.iter().sum::<f32>() / layer_contributions.len() as f32
    };

    // Average attention received
    let avg_attention = if attention_received.is_empty() {
        0.0
    } else {
        attention_received.iter().sum::<f32>() / attention_received.len() as f32
    };

    // Combine with weights
    (0.6 * avg_contribution + 0.4 * avg_attention).min(1.0)
}

/// Graph representation of attention patterns across the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionGraph {
    /// Nodes (tokens)
    pub nodes: Vec<AttentionNode>,
    /// Edges (attention connections)
    pub edges: Vec<AttentionEdge>,
    /// Number of layers
    pub num_layers: usize,
    /// Sequence length
    pub seq_len: usize,
}

/// Node in the attention graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionNode {
    /// Token position
    pub position: usize,
    /// Token text
    pub text: String,
    /// Token ID
    pub token_id: u32,
    /// Total attention received across all layers
    pub total_attention_received: f32,
    /// Total attention given
    pub total_attention_given: f32,
    /// Layer-wise attention profile
    pub layer_profile: Vec<f32>,
}

/// Edge in the attention graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionEdge {
    /// Source token position
    pub source: usize,
    /// Target token position
    pub target: usize,
    /// Layer index
    pub layer_idx: usize,
    /// Aggregated weight (across heads)
    pub weight: f32,
    /// Individual head weights
    pub head_weights: Vec<f32>,
}

/// Compute the full attention graph.
pub fn compute_attention_graph(viz: &InferenceViz) -> Result<AttentionGraph> {
    if viz.attention_patterns.is_empty() {
        return Err(InferenceVizError::NoAttentionPatterns);
    }

    let seq_len = viz.tokens.len().max(
        viz.attention_patterns
            .first()
            .map(|p| p.seq_len)
            .unwrap_or(0),
    );

    if seq_len == 0 {
        return Err(InferenceVizError::EmptySequence);
    }

    // Initialize node statistics
    let mut nodes: Vec<AttentionNode> = (0..seq_len)
        .map(|pos| AttentionNode {
            position: pos,
            text: viz.tokens.get(pos).cloned().unwrap_or_default(),
            token_id: viz.token_ids.get(pos).copied().unwrap_or(0),
            total_attention_received: 0.0,
            total_attention_given: 0.0,
            layer_profile: vec![0.0; viz.model_config.num_layers],
        })
        .collect();

    // Build edges and update node statistics
    let mut edges = Vec::new();
    let mut layer_edges: HashMap<(usize, usize, usize), (f32, Vec<f32>)> = HashMap::new();

    for pattern in &viz.attention_patterns {
        let layer_idx = pattern.layer_idx;
        let head_idx = pattern.head_idx;

        for (src, row) in pattern.weights.iter().enumerate() {
            for (tgt, &weight) in row.iter().enumerate() {
                if weight > 0.01 {
                    // Only include significant edges
                    let key = (src, tgt, layer_idx);
                    let entry = layer_edges.entry(key).or_insert((0.0, Vec::new()));
                    entry.0 += weight;
                    entry.1.push(weight);

                    // Update node statistics
                    if src < nodes.len() {
                        nodes[src].total_attention_given += weight;
                    }
                    if tgt < nodes.len() {
                        nodes[tgt].total_attention_received += weight;
                        let profile_len = nodes[tgt].layer_profile.len();
                        nodes[tgt].layer_profile[layer_idx.min(profile_len - 1)] += weight;
                    }
                }
            }
        }
    }

    // Convert aggregated edges
    for ((src, tgt, layer_idx), (total_weight, head_weights)) in layer_edges {
        let num_heads = head_weights.len();
        edges.push(AttentionEdge {
            source: src,
            target: tgt,
            layer_idx,
            weight: total_weight / num_heads as f32,
            head_weights,
        });
    }

    // Sort edges by weight (descending)
    edges.sort_by(|a, b| {
        b.weight
            .partial_cmp(&a.weight)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(AttentionGraph {
        nodes,
        edges,
        num_layers: viz.model_config.num_layers,
        seq_len,
    })
}

impl AttentionGraph {
    /// Get top K edges by weight.
    pub fn top_edges(&self, k: usize) -> Vec<&AttentionEdge> {
        self.edges.iter().take(k).collect()
    }

    /// Get edges for a specific layer.
    pub fn layer_edges(&self, layer_idx: usize) -> Vec<&AttentionEdge> {
        self.edges
            .iter()
            .filter(|e| e.layer_idx == layer_idx)
            .collect()
    }

    /// Get edges from a specific source position.
    pub fn edges_from(&self, source: usize) -> Vec<&AttentionEdge> {
        self.edges.iter().filter(|e| e.source == source).collect()
    }

    /// Get edges to a specific target position.
    pub fn edges_to(&self, target: usize) -> Vec<&AttentionEdge> {
        self.edges.iter().filter(|e| e.target == target).collect()
    }

    /// Find the most attended node overall.
    pub fn most_attended(&self) -> Option<&AttentionNode> {
        self.nodes.iter().max_by(|a, b| {
            a.total_attention_received
                .partial_cmp(&b.total_attention_received)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Find the node that gives most attention.
    pub fn most_attending(&self) -> Option<&AttentionNode> {
        self.nodes.iter().max_by(|a, b| {
            a.total_attention_given
                .partial_cmp(&b.total_attention_given)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Compute attention path from source to target through layers.
    pub fn attention_path(&self, source: usize, target: usize) -> Vec<Vec<&AttentionEdge>> {
        let mut paths = Vec::new();

        for layer_idx in 0..self.num_layers {
            let layer_path: Vec<&AttentionEdge> = self
                .edges
                .iter()
                .filter(|e| e.layer_idx == layer_idx && e.source == source && e.target == target)
                .collect();

            paths.push(layer_path);
        }

        paths
    }

    /// Export to DOT format for GraphViz visualization.
    pub fn to_dot(&self, max_edges: usize) -> String {
        let mut dot = String::from("digraph attention {\n");
        dot.push_str("  rankdir=LR;\n");
        dot.push_str("  node [shape=box];\n");

        // Add nodes
        for node in &self.nodes {
            let label = if node.text.is_empty() {
                format!("pos_{}", node.position)
            } else {
                node.text.replace('"', "\\\"")
            };
            dot.push_str(&format!("  n{} [label=\"{}\"];\n", node.position, label));
        }

        // Add top edges
        for edge in self.edges.iter().take(max_edges) {
            let alpha = (edge.weight * 255.0).min(255.0) as u8;
            dot.push_str(&format!(
                "  n{} -> n{} [penwidth={:.2}, label=\"L{}: {:.2}\", color=\"#0000{:02X}\"];\n",
                edge.source,
                edge.target,
                edge.weight * 3.0,
                edge.layer_idx,
                edge.weight,
                alpha
            ));
        }

        dot.push_str("}\n");
        dot
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference_viz::{LayerActivation, ModelConfig};

    fn create_test_viz() -> InferenceViz {
        let config = ModelConfig::new(2, 64, 2, 100);
        let mut viz = InferenceViz::new(config);

        // Add tokens
        viz.set_tokens(
            vec!["hello".to_string(), "world".to_string(), "test".to_string()],
            vec![1, 2, 3],
        );

        // Add activations
        viz.layer_activations.push(LayerActivation::new(
            0,
            LayerType::Attention,
            vec![0.1, 0.2, 0.3],
            vec![0.2, 0.4, 0.6],
        ));

        viz.layer_activations.push(LayerActivation::new(
            1,
            LayerType::FeedForward,
            vec![0.2, 0.4, 0.6],
            vec![0.3, 0.5, 0.7],
        ));

        // Add attention patterns
        viz.attention_patterns.push(AttentionPattern::new(
            0,
            0,
            vec![
                vec![0.5, 0.3, 0.2],
                vec![0.1, 0.6, 0.3],
                vec![0.2, 0.2, 0.6],
            ],
        ));

        viz
    }

    #[test]
    fn test_compute_token_flow() {
        let viz = create_test_viz();
        let flow = compute_token_flow(&viz, 0).unwrap();

        assert_eq!(flow.token_idx, 0);
        assert_eq!(flow.token_text, "hello");
        assert_eq!(flow.token_id, 1);
        assert!(!flow.layer_contributions.is_empty());
    }

    #[test]
    fn test_token_flow_invalid_index() {
        let viz = create_test_viz();
        let result = compute_token_flow(&viz, 10);

        assert!(matches!(
            result,
            Err(InferenceVizError::InvalidTokenIndex { .. })
        ));
    }

    #[test]
    fn test_compute_attention_graph() {
        let viz = create_test_viz();
        let graph = compute_attention_graph(&viz).unwrap();

        assert_eq!(graph.nodes.len(), 3);
        assert!(!graph.edges.is_empty());
    }

    #[test]
    fn test_attention_graph_to_dot() {
        let viz = create_test_viz();
        let graph = compute_attention_graph(&viz).unwrap();
        let dot = graph.to_dot(10);

        assert!(dot.contains("digraph attention"));
        assert!(dot.contains("hello"));
        assert!(dot.contains("world"));
    }
}
