//! Layer Heatmap Visualization Module
//!
//! Provides heatmap generation for visualizing activation patterns
//! at each layer of the model.

use super::{
    AttentionPattern, InferenceViz, InferenceVizError, LayerActivation, LayerType, Result,
};
use serde::{Deserialize, Serialize};

/// Heatmap data for a layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerHeatmap {
    /// Layer index
    pub layer_idx: usize,
    /// Layer type
    pub layer_type: LayerType,
    /// Heatmap data (2D grid of values)
    pub data: Vec<Vec<f32>>,
    /// Number of rows
    pub rows: usize,
    /// Number of columns
    pub cols: usize,
    /// Minimum value in the heatmap
    pub min_val: f32,
    /// Maximum value in the heatmap
    pub max_val: f32,
    /// Row labels (if applicable)
    pub row_labels: Vec<String>,
    /// Column labels (if applicable)
    pub col_labels: Vec<String>,
}

impl LayerHeatmap {
    /// Create a new heatmap from 2D data.
    pub fn new(
        layer_idx: usize,
        layer_type: LayerType,
        data: Vec<Vec<f32>>,
        row_labels: Vec<String>,
        col_labels: Vec<String>,
    ) -> Self {
        let rows = data.len();
        let cols = data.first().map(|r| r.len()).unwrap_or(0);

        let (min_val, max_val) = data
            .iter()
            .flatten()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), &val| {
                (min.min(val), max.max(val))
            });

        Self {
            layer_idx,
            layer_type,
            data,
            rows,
            cols,
            min_val,
            max_val,
            row_labels,
            col_labels,
        }
    }

    /// Get normalized value at position (0-1 range).
    pub fn normalized_value(&self, row: usize, col: usize) -> Option<f32> {
        let val = self.data.get(row).and_then(|r| r.get(col))?;
        let range = self.max_val - self.min_val;
        if range.abs() < 1e-10 {
            Some(0.5)
        } else {
            Some((val - self.min_val) / range)
        }
    }

    /// Get a row of the heatmap.
    pub fn row(&self, idx: usize) -> Option<&[f32]> {
        self.data.get(idx).map(|r| r.as_slice())
    }

    /// Get a column of the heatmap.
    pub fn column(&self, idx: usize) -> Vec<f32> {
        self.data
            .iter()
            .filter_map(|row| row.get(idx).copied())
            .collect()
    }

    /// Compute row-wise statistics.
    pub fn row_stats(&self) -> Vec<RowStats> {
        self.data
            .iter()
            .enumerate()
            .map(|(idx, row)| {
                let sum: f32 = row.iter().sum();
                let mean = if row.is_empty() {
                    0.0
                } else {
                    sum / row.len() as f32
                };
                let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let max_col = row
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i)
                    .unwrap_or(0);

                RowStats {
                    row_idx: idx,
                    sum,
                    mean,
                    max,
                    max_col,
                }
            })
            .collect()
    }

    /// Render to ASCII art.
    pub fn to_ascii(&self, width: usize, height: usize) -> String {
        let chars = [' ', '.', ':', '-', '=', '+', '*', '#', '%', '@'];
        let mut output = String::new();

        // Compute cell dimensions
        let cell_rows = (self.rows as f32 / height as f32).ceil() as usize;
        let cell_cols = (self.cols as f32 / width as f32).ceil() as usize;

        for row_block in 0..height.min(self.rows) {
            for col_block in 0..width.min(self.cols) {
                // Average value in this block
                let mut sum = 0.0f32;
                let mut count = 0;

                for r in 0..cell_rows {
                    for c in 0..cell_cols {
                        let row = row_block * cell_rows + r;
                        let col = col_block * cell_cols + c;
                        if let Some(val) = self.data.get(row).and_then(|r| r.get(col)) {
                            sum += val;
                            count += 1;
                        }
                    }
                }

                let avg = if count > 0 { sum / count as f32 } else { 0.0 };
                let normalized = if (self.max_val - self.min_val).abs() < 1e-10 {
                    0.5
                } else {
                    (avg - self.min_val) / (self.max_val - self.min_val)
                };

                let char_idx = (normalized * (chars.len() - 1) as f32).round() as usize;
                output.push(chars[char_idx.min(chars.len() - 1)]);
            }
            output.push('\n');
        }

        output
    }

    /// Export to SVG format.
    pub fn to_svg(&self, width: usize, height: usize, colormap: &str) -> String {
        let cell_width = width as f32 / self.cols as f32;
        let cell_height = height as f32 / self.rows as f32;

        let mut svg = format!(
            r#"<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}">"#,
            width, height
        );

        svg.push_str(&format!(
            r#"<style>.label {{ font-family: monospace; font-size: 10px; }}</style>"#
        ));

        for (row_idx, row) in self.data.iter().enumerate() {
            for (col_idx, &val) in row.iter().enumerate() {
                let normalized = if (self.max_val - self.min_val).abs() < 1e-10 {
                    0.5
                } else {
                    (val - self.min_val) / (self.max_val - self.min_val)
                };

                let color = value_to_color(normalized, colormap);
                let x = col_idx as f32 * cell_width;
                let y = row_idx as f32 * cell_height;

                svg.push_str(&format!(
                    r#"<rect x="{:.1}" y="{:.1}" width="{:.1}" height="{:.1}" fill="{}" />"#,
                    x, y, cell_width, cell_height, color
                ));
            }
        }

        svg.push_str("</svg>");
        svg
    }
}

/// Statistics for a row in the heatmap.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RowStats {
    /// Row index
    pub row_idx: usize,
    /// Sum of values
    pub sum: f32,
    /// Mean value
    pub mean: f32,
    /// Maximum value
    pub max: f32,
    /// Column with maximum value
    pub max_col: usize,
}

/// Compute layer heatmap from inference visualization data.
pub fn compute_layer_heatmap(viz: &InferenceViz, layer_idx: usize) -> Result<Vec<Vec<f32>>> {
    // Try to get attention pattern for this layer first
    let attention_patterns: Vec<&AttentionPattern> = viz
        .attention_patterns
        .iter()
        .filter(|p| p.layer_idx == layer_idx)
        .collect();

    if !attention_patterns.is_empty() {
        // Use attention weights as heatmap
        // Average across heads
        let seq_len = attention_patterns[0].seq_len;
        let num_heads = attention_patterns.len();

        let mut heatmap = vec![vec![0.0f32; seq_len]; seq_len];

        for pattern in &attention_patterns {
            for (i, row) in pattern.weights.iter().enumerate() {
                for (j, &weight) in row.iter().enumerate() {
                    heatmap[i][j] += weight / num_heads as f32;
                }
            }
        }

        return Ok(heatmap);
    }

    // Fall back to activation-based heatmap
    let activation = viz
        .layer_activations
        .iter()
        .find(|a| a.layer_idx == layer_idx)
        .ok_or(InferenceVizError::NoActivationsForLayer(layer_idx))?;

    // Reshape activations into a 2D grid
    let values = &activation.output_activations;
    let total = values.len();

    if total == 0 {
        return Err(InferenceVizError::NoActivationsForLayer(layer_idx));
    }

    // Try to make a square-ish grid
    let cols = (total as f32).sqrt().ceil() as usize;
    let rows = (total + cols - 1) / cols;

    let mut heatmap = Vec::with_capacity(rows);
    for row_idx in 0..rows {
        let start = row_idx * cols;
        let end = (start + cols).min(total);
        let row: Vec<f32> = values[start..end].to_vec();
        heatmap.push(row);
    }

    Ok(heatmap)
}

/// Build a comprehensive layer heatmap with metadata.
pub fn build_layer_heatmap(viz: &InferenceViz, layer_idx: usize) -> Result<LayerHeatmap> {
    let data = compute_layer_heatmap(viz, layer_idx)?;

    let layer_type = viz
        .layer_activations
        .iter()
        .find(|a| a.layer_idx == layer_idx)
        .map(|a| a.layer_type)
        .unwrap_or(LayerType::Attention);

    // Generate labels
    let rows = data.len();
    let cols = data.first().map(|r| r.len()).unwrap_or(0);

    let row_labels: Vec<String> = if rows <= viz.tokens.len() {
        viz.tokens.iter().take(rows).cloned().collect()
    } else {
        (0..rows).map(|i| format!("r{}", i)).collect()
    };

    let col_labels: Vec<String> = if cols <= viz.tokens.len() {
        viz.tokens.iter().take(cols).cloned().collect()
    } else {
        (0..cols).map(|i| format!("c{}", i)).collect()
    };

    Ok(LayerHeatmap::new(
        layer_idx, layer_type, data, row_labels, col_labels,
    ))
}

/// Convert a normalized value (0-1) to a color string.
fn value_to_color(normalized: f32, colormap: &str) -> String {
    let normalized = normalized.clamp(0.0, 1.0);

    match colormap {
        "viridis" => {
            // Simplified viridis approximation
            let r = (68.0 + normalized * (253.0 - 68.0)) as u8;
            let g = (1.0 + normalized * (231.0 - 1.0)) as u8;
            let b = (84.0 + (1.0 - normalized) * (70.0)) as u8;
            format!("#{:02X}{:02X}{:02X}", r, g, b)
        }
        "plasma" => {
            // Simplified plasma approximation
            let r = (13.0 + normalized * (240.0 - 13.0)) as u8;
            let g = (8.0 + normalized * (249.0 - 8.0)) as u8;
            let b = (135.0 - normalized * (70.0)) as u8;
            format!("#{:02X}{:02X}{:02X}", r, g, b)
        }
        "coolwarm" => {
            // Blue to white to red
            if normalized < 0.5 {
                let t = normalized * 2.0;
                let r = (59.0 + t * (221.0 - 59.0)) as u8;
                let g = (76.0 + t * (221.0 - 76.0)) as u8;
                let b = (192.0 + t * (221.0 - 192.0)) as u8;
                format!("#{:02X}{:02X}{:02X}", r, g, b)
            } else {
                let t = (normalized - 0.5) * 2.0;
                let r = (221.0 + t * (180.0 - 221.0)) as u8;
                let g = (221.0 - t * (183.0)) as u8;
                let b = (221.0 - t * (183.0)) as u8;
                format!("#{:02X}{:02X}{:02X}", r, g, b)
            }
        }
        _ => {
            // Grayscale fallback
            let v = (normalized * 255.0) as u8;
            format!("#{:02X}{:02X}{:02X}", v, v, v)
        }
    }
}

/// Multi-layer heatmap comparison.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeatmapComparison {
    /// Heatmaps for each layer
    pub heatmaps: Vec<LayerHeatmap>,
    /// Similarity matrix between layers
    pub similarity_matrix: Vec<Vec<f32>>,
    /// Layer indices
    pub layer_indices: Vec<usize>,
}

impl HeatmapComparison {
    /// Create a comparison from multiple heatmaps.
    pub fn new(heatmaps: Vec<LayerHeatmap>) -> Self {
        let n = heatmaps.len();
        let layer_indices: Vec<usize> = heatmaps.iter().map(|h| h.layer_idx).collect();

        // Compute similarity matrix using cosine similarity
        let mut similarity_matrix = vec![vec![0.0f32; n]; n];

        for i in 0..n {
            for j in i..n {
                let sim = heatmap_similarity(&heatmaps[i], &heatmaps[j]);
                similarity_matrix[i][j] = sim;
                similarity_matrix[j][i] = sim;
            }
        }

        Self {
            heatmaps,
            similarity_matrix,
            layer_indices,
        }
    }

    /// Find the most similar layer pair.
    pub fn most_similar_pair(&self) -> Option<(usize, usize, f32)> {
        let n = self.similarity_matrix.len();
        let mut best: Option<(usize, usize, f32)> = None;

        for i in 0..n {
            for j in (i + 1)..n {
                let sim = self.similarity_matrix[i][j];
                if best.is_none() || sim > best.unwrap().2 {
                    best = Some((i, j, sim));
                }
            }
        }

        best
    }

    /// Find the most different layer pair.
    pub fn most_different_pair(&self) -> Option<(usize, usize, f32)> {
        let n = self.similarity_matrix.len();
        let mut worst: Option<(usize, usize, f32)> = None;

        for i in 0..n {
            for j in (i + 1)..n {
                let sim = self.similarity_matrix[i][j];
                if worst.is_none() || sim < worst.unwrap().2 {
                    worst = Some((i, j, sim));
                }
            }
        }

        worst
    }
}

/// Compute similarity between two heatmaps using flattened cosine similarity.
fn heatmap_similarity(a: &LayerHeatmap, b: &LayerHeatmap) -> f32 {
    let flat_a: Vec<f32> = a.data.iter().flatten().copied().collect();
    let flat_b: Vec<f32> = b.data.iter().flatten().copied().collect();

    if flat_a.len() != flat_b.len() || flat_a.is_empty() {
        return 0.0;
    }

    let dot: f32 = flat_a.iter().zip(flat_b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = flat_a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = flat_b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a < 1e-10 || norm_b < 1e-10 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_heatmap_new() {
        let data = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];

        let heatmap = LayerHeatmap::new(
            0,
            LayerType::Attention,
            data,
            vec!["r0".to_string(), "r1".to_string()],
            vec!["c0".to_string(), "c1".to_string(), "c2".to_string()],
        );

        assert_eq!(heatmap.rows, 2);
        assert_eq!(heatmap.cols, 3);
        assert!((heatmap.min_val - 1.0).abs() < 1e-6);
        assert!((heatmap.max_val - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_normalized_value() {
        let data = vec![vec![0.0, 5.0, 10.0]];

        let heatmap = LayerHeatmap::new(0, LayerType::Attention, data, vec![], vec![]);

        assert!((heatmap.normalized_value(0, 0).unwrap() - 0.0).abs() < 1e-6);
        assert!((heatmap.normalized_value(0, 1).unwrap() - 0.5).abs() < 1e-6);
        assert!((heatmap.normalized_value(0, 2).unwrap() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_to_ascii() {
        let data = vec![
            vec![0.0, 0.5, 1.0],
            vec![0.25, 0.5, 0.75],
            vec![1.0, 0.5, 0.0],
        ];

        let heatmap = LayerHeatmap::new(0, LayerType::Attention, data, vec![], vec![]);
        let ascii = heatmap.to_ascii(3, 3);

        assert!(!ascii.is_empty());
        assert!(ascii.contains('\n'));
    }

    #[test]
    fn test_row_stats() {
        let data = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];

        let heatmap = LayerHeatmap::new(0, LayerType::Attention, data, vec![], vec![]);
        let stats = heatmap.row_stats();

        assert_eq!(stats.len(), 2);
        assert!((stats[0].sum - 6.0).abs() < 1e-6);
        assert_eq!(stats[0].max_col, 2);
    }

    #[test]
    fn test_value_to_color() {
        let color = value_to_color(0.5, "grayscale");
        assert!(color.starts_with('#'));
        assert_eq!(color.len(), 7);
    }

    #[test]
    fn test_heatmap_similarity() {
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let heatmap1 = LayerHeatmap::new(0, LayerType::Attention, data.clone(), vec![], vec![]);
        let heatmap2 = LayerHeatmap::new(1, LayerType::Attention, data, vec![], vec![]);

        let sim = heatmap_similarity(&heatmap1, &heatmap2);
        assert!((sim - 1.0).abs() < 1e-6); // Identical heatmaps should have similarity 1.0
    }
}
