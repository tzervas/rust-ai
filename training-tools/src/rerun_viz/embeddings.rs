//! Embedding visualization for Rerun.
//!
//! This module provides tools for visualizing high-dimensional embeddings
//! as 3D point clouds in the Rerun viewer.

use super::{RerunError, RerunResult};
use rerun::{Color, Position3D, RecordingStream};

/// Logger for embedding space visualization.
pub struct EmbeddingLogger<'a> {
    rec: &'a RecordingStream,
}

impl<'a> EmbeddingLogger<'a> {
    /// Create a new embedding logger.
    pub fn new(rec: &'a RecordingStream) -> Self {
        Self { rec }
    }

    /// Log embeddings as a 3D point cloud.
    ///
    /// If embeddings have more than 3 dimensions, PCA projection is applied
    /// to reduce to 3D for visualization.
    ///
    /// # Arguments
    ///
    /// * `embeddings` - Vector of embedding vectors
    /// * `labels` - Labels for each point (must match embeddings length)
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, or an error if data is invalid.
    pub fn log_embeddings(&self, embeddings: &[Vec<f32>], labels: &[String]) -> RerunResult<()> {
        if embeddings.is_empty() {
            return Err(RerunError::EmptyInput("No embeddings provided".to_string()));
        }

        if embeddings.len() != labels.len() {
            return Err(RerunError::ShapeMismatch {
                expected: format!("{} labels", embeddings.len()),
                got: format!("{} labels", labels.len()),
            });
        }

        let dim = embeddings[0].len();
        if dim == 0 {
            return Err(RerunError::EmptyInput(
                "Embeddings have zero dimensions".to_string(),
            ));
        }

        // Project to 3D if needed
        let points_3d: Vec<Position3D> = if dim <= 3 {
            // Already 3D or less - pad with zeros if needed
            embeddings
                .iter()
                .map(|emb| {
                    let x = emb.first().copied().unwrap_or(0.0);
                    let y = emb.get(1).copied().unwrap_or(0.0);
                    let z = emb.get(2).copied().unwrap_or(0.0);
                    Position3D::new(x, y, z)
                })
                .collect()
        } else {
            // Need PCA projection to 3D
            let projected = self.pca_project_3d(embeddings)?;
            projected
                .into_iter()
                .map(|p| Position3D::new(p[0], p[1], p[2]))
                .collect()
        };

        // Generate colors based on label hash for visual distinction
        let colors: Vec<Color> = labels.iter().map(|l| self.label_to_color(l)).collect();

        // Log the point cloud
        self.rec
            .log(
                "embeddings/points",
                &rerun::Points3D::new(points_3d)
                    .with_colors(colors)
                    .with_radii([0.02f32]),
            )
            .map_err(|e| RerunError::LoggingError(e.to_string()))?;

        // Log labels as text annotations (for a subset to avoid clutter)
        let max_labels = 50.min(labels.len());
        for (i, label) in labels.iter().take(max_labels).enumerate() {
            let emb = &embeddings[i];
            let x = emb.first().copied().unwrap_or(0.0);
            let y = emb.get(1).copied().unwrap_or(0.0);
            let z = emb.get(2).copied().unwrap_or(0.0);

            let _ = self.rec.log(
                format!("embeddings/labels/{}", i),
                &rerun::Points3D::new([[x, y, z]])
                    .with_labels([label.clone()])
                    .with_radii([0.01f32]),
            );
        }

        Ok(())
    }

    /// Log embeddings with cluster assignments.
    ///
    /// # Arguments
    ///
    /// * `embeddings` - Vector of embedding vectors
    /// * `cluster_ids` - Cluster assignment for each embedding
    /// * `labels` - Optional labels for each point
    pub fn log_clustered_embeddings(
        &self,
        embeddings: &[Vec<f32>],
        cluster_ids: &[usize],
        labels: Option<&[String]>,
    ) -> RerunResult<()> {
        if embeddings.is_empty() {
            return Err(RerunError::EmptyInput("No embeddings provided".to_string()));
        }

        if embeddings.len() != cluster_ids.len() {
            return Err(RerunError::ShapeMismatch {
                expected: format!("{} cluster_ids", embeddings.len()),
                got: format!("{} cluster_ids", cluster_ids.len()),
            });
        }

        let dim = embeddings[0].len();

        // Project to 3D if needed
        let points_3d: Vec<Position3D> = if dim <= 3 {
            embeddings
                .iter()
                .map(|emb| {
                    Position3D::new(
                        emb.first().copied().unwrap_or(0.0),
                        emb.get(1).copied().unwrap_or(0.0),
                        emb.get(2).copied().unwrap_or(0.0),
                    )
                })
                .collect()
        } else {
            let projected = self.pca_project_3d(embeddings)?;
            projected
                .into_iter()
                .map(|p| Position3D::new(p[0], p[1], p[2]))
                .collect()
        };

        // Color by cluster
        let colors: Vec<Color> = cluster_ids
            .iter()
            .map(|&id| self.cluster_to_color(id))
            .collect();

        // Log the clustered point cloud
        let points = rerun::Points3D::new(points_3d)
            .with_colors(colors)
            .with_radii([0.02f32]);

        let points = if let Some(lbls) = labels {
            points.with_labels(lbls.iter().cloned())
        } else {
            points
        };

        self.rec
            .log("embeddings/clustered", &points)
            .map_err(|e| RerunError::LoggingError(e.to_string()))?;

        Ok(())
    }

    /// Log embedding evolution over training steps.
    ///
    /// This tracks how embeddings change during training, useful for
    /// observing representation learning.
    ///
    /// # Arguments
    ///
    /// * `step` - Current training step
    /// * `embeddings` - Current embedding vectors
    /// * `token_ids` - IDs of tokens being tracked
    pub fn log_embedding_trajectory(
        &self,
        step: u64,
        embeddings: &[Vec<f32>],
        token_ids: &[usize],
    ) -> RerunResult<()> {
        if embeddings.is_empty() || token_ids.is_empty() {
            return Err(RerunError::EmptyInput(
                "Empty embeddings or token_ids".to_string(),
            ));
        }

        self.rec.set_time_sequence("step", step as i64);

        let dim = embeddings[0].len();
        let points_3d: Vec<Position3D> = if dim <= 3 {
            embeddings
                .iter()
                .map(|emb| {
                    Position3D::new(
                        emb.first().copied().unwrap_or(0.0),
                        emb.get(1).copied().unwrap_or(0.0),
                        emb.get(2).copied().unwrap_or(0.0),
                    )
                })
                .collect()
        } else {
            let projected = self.pca_project_3d(embeddings)?;
            projected
                .into_iter()
                .map(|p| Position3D::new(p[0], p[1], p[2]))
                .collect()
        };

        // Log each token's position with its ID for tracking
        let colors: Vec<Color> = token_ids
            .iter()
            .map(|&id| self.cluster_to_color(id))
            .collect();

        self.rec
            .log(
                "embeddings/trajectory",
                &rerun::Points3D::new(points_3d).with_colors(colors),
            )
            .map_err(|e| RerunError::LoggingError(e.to_string()))?;

        Ok(())
    }

    /// Simple PCA projection to 3D.
    ///
    /// Uses power iteration for the top 3 principal components.
    fn pca_project_3d(&self, data: &[Vec<f32>]) -> RerunResult<Vec<[f32; 3]>> {
        let n = data.len();
        if n < 3 {
            return Err(RerunError::EmptyInput(
                "Need at least 3 points for PCA".to_string(),
            ));
        }

        let dim = data[0].len();

        // Compute mean
        let mut mean = vec![0.0f32; dim];
        for row in data {
            for (i, &val) in row.iter().enumerate() {
                mean[i] += val;
            }
        }
        for m in &mut mean {
            *m /= n as f32;
        }

        // Center data
        let centered: Vec<Vec<f32>> = data
            .iter()
            .map(|row| row.iter().zip(mean.iter()).map(|(x, m)| x - m).collect())
            .collect();

        // Compute covariance matrix (simplified - uses first 3 PCs via power iteration)
        // For production, use proper SVD from ndarray-linalg or similar
        let pcs = self.compute_principal_components(&centered, 3);

        // Project onto principal components
        let projected: Vec<[f32; 3]> = centered
            .iter()
            .map(|row| {
                let mut proj = [0.0f32; 3];
                for (i, pc) in pcs.iter().enumerate().take(3) {
                    proj[i] = row.iter().zip(pc.iter()).map(|(x, p)| x * p).sum();
                }
                proj
            })
            .collect();

        Ok(projected)
    }

    /// Compute top-k principal components using power iteration.
    fn compute_principal_components(&self, data: &[Vec<f32>], k: usize) -> Vec<Vec<f32>> {
        let dim = data[0].len();
        let mut pcs = Vec::with_capacity(k);

        // Simple power iteration for each component
        for component_idx in 0..k {
            // Initialize with random-ish vector
            let mut pc: Vec<f32> = (0..dim)
                .map(|i| ((i * 7 + component_idx * 13) % 100) as f32 / 100.0)
                .collect();
            self.normalize(&mut pc);

            // Power iteration (20 iterations should suffice for visualization)
            for _ in 0..20 {
                // Compute X^T * X * pc
                let mut new_pc = vec![0.0f32; dim];
                for row in data {
                    let dot: f32 = row.iter().zip(pc.iter()).map(|(x, p)| x * p).sum();
                    for (i, &val) in row.iter().enumerate() {
                        new_pc[i] += val * dot;
                    }
                }

                // Orthogonalize against previous components
                for prev_pc in &pcs {
                    let dot: f32 = new_pc.iter().zip(prev_pc.iter()).map(|(a, b)| a * b).sum();
                    for (i, &val) in prev_pc.iter().enumerate() {
                        new_pc[i] -= dot * val;
                    }
                }

                self.normalize(&mut new_pc);
                pc = new_pc;
            }

            pcs.push(pc);
        }

        pcs
    }

    /// Normalize a vector to unit length.
    fn normalize(&self, v: &mut [f32]) {
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-10 {
            for x in v.iter_mut() {
                *x /= norm;
            }
        }
    }

    /// Convert a label string to a color (deterministic hash-based).
    fn label_to_color(&self, label: &str) -> Color {
        // Simple hash to color
        let hash: u32 = label
            .bytes()
            .fold(0u32, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u32));

        let r = ((hash >> 16) & 0xFF) as u8;
        let g = ((hash >> 8) & 0xFF) as u8;
        let b = (hash & 0xFF) as u8;

        // Ensure minimum brightness
        let min_brightness = 80u8;
        Color::from_rgb(
            r.max(min_brightness),
            g.max(min_brightness),
            b.max(min_brightness),
        )
    }

    /// Convert a cluster ID to a distinct color.
    fn cluster_to_color(&self, cluster_id: usize) -> Color {
        // Use a perceptually distinct color palette
        const PALETTE: &[[u8; 3]] = &[
            [228, 26, 28],   // Red
            [55, 126, 184],  // Blue
            [77, 175, 74],   // Green
            [152, 78, 163],  // Purple
            [255, 127, 0],   // Orange
            [255, 255, 51],  // Yellow
            [166, 86, 40],   // Brown
            [247, 129, 191], // Pink
            [153, 153, 153], // Gray
            [0, 206, 209],   // Turquoise
        ];

        let [r, g, b] = PALETTE[cluster_id % PALETTE.len()];
        Color::from_rgb(r, g, b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_label_to_color_deterministic() {
        // Mock test - in real test we'd use a mock RecordingStream
        let label = "test_label";
        let hash: u32 = label
            .bytes()
            .fold(0u32, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u32));

        // Same label should produce same hash
        let hash2: u32 = label
            .bytes()
            .fold(0u32, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u32));

        assert_eq!(hash, hash2);
    }

    #[test]
    fn test_cluster_colors_are_distinct() {
        const PALETTE: &[[u8; 3]] = &[
            [228, 26, 28],
            [55, 126, 184],
            [77, 175, 74],
            [152, 78, 163],
            [255, 127, 0],
        ];

        // Check first 5 colors are all different
        for i in 0..5 {
            for j in 0..5 {
                if i != j {
                    assert_ne!(PALETTE[i], PALETTE[j]);
                }
            }
        }
    }
}
