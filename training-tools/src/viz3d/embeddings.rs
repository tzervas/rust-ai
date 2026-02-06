//! Embedding point cloud visualization
//!
//! Provides 3D visualization of high-dimensional embeddings reduced to 3D space.

use std::collections::HashMap;

use nalgebra::{Point3, Vector3};

use super::colors::{tab10, CategoricalPalette, Color, Colormap, ColormapPreset};
use super::engine::{Mesh3D, ObjectId, Viz3DEngine};

/// Configuration for embedding visualization
#[derive(Debug, Clone)]
pub struct EmbeddingCloudConfig {
    /// Point size (sphere radius)
    pub point_size: f32,
    /// Maximum points to display (subsample if exceeded)
    pub max_points: usize,
    /// Sphere segments for point rendering
    pub sphere_segments: u32,
    /// Default colormap for continuous labels
    pub colormap: ColormapPreset,
    /// Show cluster centroids
    pub show_centroids: bool,
    /// Centroid size multiplier
    pub centroid_size_multiplier: f32,
    /// Point opacity
    pub point_opacity: f32,
    /// Show connecting lines between sequential points
    pub show_trajectory: bool,
    /// Trajectory line color
    pub trajectory_color: Color,
}

impl Default for EmbeddingCloudConfig {
    fn default() -> Self {
        Self {
            point_size: 0.05,
            max_points: 10000,
            sphere_segments: 8,
            colormap: ColormapPreset::Viridis,
            show_centroids: false,
            centroid_size_multiplier: 3.0,
            point_opacity: 0.8,
            show_trajectory: false,
            trajectory_color: Color::rgba(0.5, 0.5, 0.5, 0.3),
        }
    }
}

/// Dimensionality reduction method
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DimReductionMethod {
    /// Principal Component Analysis
    #[default]
    PCA,
    /// t-SNE (placeholder - requires external computation)
    TSNE,
    /// UMAP (placeholder - requires external computation)
    UMAP,
    /// No reduction - use first 3 dimensions
    None,
}

/// Label type for coloring points
#[derive(Debug, Clone)]
pub enum PointLabels {
    /// No labels (uniform color)
    None,
    /// Categorical labels (discrete classes)
    Categorical(Vec<usize>),
    /// Continuous labels (scalar values)
    Continuous(Vec<f32>),
    /// Custom colors per point
    Custom(Vec<Color>),
}

/// A single embedding point
#[derive(Debug, Clone)]
pub struct EmbeddingPoint {
    /// 3D position (after dimensionality reduction)
    pub position: Point3<f32>,
    /// Original high-dimensional vector (optional)
    pub original: Option<Vec<f32>>,
    /// Label for coloring
    pub label: Option<usize>,
    /// Scalar value for coloring
    pub value: Option<f32>,
    /// Optional text annotation
    pub annotation: Option<String>,
}

impl EmbeddingPoint {
    /// Create a new embedding point from 3D coordinates
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self {
            position: Point3::new(x, y, z),
            original: None,
            label: None,
            value: None,
            annotation: None,
        }
    }

    /// Create from a Point3
    pub fn from_point(position: Point3<f32>) -> Self {
        Self {
            position,
            original: None,
            label: None,
            value: None,
            annotation: None,
        }
    }

    /// Set the categorical label
    pub fn with_label(mut self, label: usize) -> Self {
        self.label = Some(label);
        self
    }

    /// Set the continuous value
    pub fn with_value(mut self, value: f32) -> Self {
        self.value = Some(value);
        self
    }

    /// Set the annotation
    pub fn with_annotation(mut self, annotation: impl Into<String>) -> Self {
        self.annotation = Some(annotation.into());
        self
    }

    /// Set the original high-dimensional vector
    pub fn with_original(mut self, original: Vec<f32>) -> Self {
        self.original = Some(original);
        self
    }
}

/// Cluster information for grouping points
#[derive(Debug, Clone)]
pub struct Cluster {
    /// Cluster index
    pub index: usize,
    /// Cluster name
    pub name: String,
    /// Centroid position
    pub centroid: Point3<f32>,
    /// Indices of points in this cluster
    pub point_indices: Vec<usize>,
    /// Cluster color
    pub color: Color,
}

/// 3D embedding point cloud visualization
#[derive(Debug)]
pub struct EmbeddingCloud3D {
    /// Configuration
    pub config: EmbeddingCloudConfig,
    /// Embedding points
    points: Vec<EmbeddingPoint>,
    /// Clusters (optional grouping)
    clusters: Vec<Cluster>,
    /// Generated mesh IDs for points
    point_mesh_ids: Vec<ObjectId>,
    /// Generated mesh IDs for centroids
    centroid_mesh_ids: Vec<ObjectId>,
    /// Generated mesh IDs for trajectory lines
    trajectory_mesh_ids: Vec<ObjectId>,
    /// Bounding box of the point cloud
    bounds: Option<(Point3<f32>, Point3<f32>)>,
}

impl EmbeddingCloud3D {
    /// Create a new embedding cloud visualization
    pub fn new() -> Self {
        Self::with_config(EmbeddingCloudConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: EmbeddingCloudConfig) -> Self {
        Self {
            config,
            points: Vec::new(),
            clusters: Vec::new(),
            point_mesh_ids: Vec::new(),
            centroid_mesh_ids: Vec::new(),
            trajectory_mesh_ids: Vec::new(),
            bounds: None,
        }
    }

    /// Add a single point
    pub fn add_point(&mut self, point: EmbeddingPoint) {
        self.points.push(point);
        self.bounds = None; // Invalidate bounds
    }

    /// Add multiple points
    pub fn add_points(&mut self, points: impl IntoIterator<Item = EmbeddingPoint>) {
        self.points.extend(points);
        self.bounds = None;
    }

    /// Set all points at once
    pub fn set_points(&mut self, points: Vec<EmbeddingPoint>) {
        self.points = points;
        self.bounds = None;
    }

    /// Load points from a 2D array of positions (Nx3)
    pub fn from_positions(positions: &[[f32; 3]]) -> Self {
        let mut cloud = Self::new();
        for pos in positions {
            cloud.add_point(EmbeddingPoint::new(pos[0], pos[1], pos[2]));
        }
        cloud
    }

    /// Load points with categorical labels
    pub fn from_positions_with_labels(positions: &[[f32; 3]], labels: &[usize]) -> Self {
        let mut cloud = Self::new();
        for (pos, &label) in positions.iter().zip(labels) {
            cloud.add_point(EmbeddingPoint::new(pos[0], pos[1], pos[2]).with_label(label));
        }
        cloud
    }

    /// Load points with continuous values
    pub fn from_positions_with_values(positions: &[[f32; 3]], values: &[f32]) -> Self {
        let mut cloud = Self::new();
        for (pos, &value) in positions.iter().zip(values) {
            cloud.add_point(EmbeddingPoint::new(pos[0], pos[1], pos[2]).with_value(value));
        }
        cloud
    }

    /// Get number of points
    pub fn point_count(&self) -> usize {
        self.points.len()
    }

    /// Get a point by index
    pub fn get_point(&self, idx: usize) -> Option<&EmbeddingPoint> {
        self.points.get(idx)
    }

    /// Clear all points
    pub fn clear(&mut self) {
        self.points.clear();
        self.clusters.clear();
        self.bounds = None;
    }

    /// Calculate bounding box of the point cloud
    pub fn calculate_bounds(&mut self) -> Option<(Point3<f32>, Point3<f32>)> {
        if let Some(bounds) = self.bounds {
            return Some(bounds);
        }

        if self.points.is_empty() {
            return None;
        }

        let mut min = self.points[0].position;
        let mut max = self.points[0].position;

        for p in &self.points {
            min.x = min.x.min(p.position.x);
            min.y = min.y.min(p.position.y);
            min.z = min.z.min(p.position.z);
            max.x = max.x.max(p.position.x);
            max.y = max.y.max(p.position.y);
            max.z = max.z.max(p.position.z);
        }

        self.bounds = Some((min, max));
        self.bounds
    }

    /// Normalize points to fit within a unit cube centered at origin
    pub fn normalize(&mut self) {
        if let Some((min, max)) = self.calculate_bounds() {
            let center = Point3::new(
                (min.x + max.x) / 2.0,
                (min.y + max.y) / 2.0,
                (min.z + max.z) / 2.0,
            );
            let diff = max - min;
            let scale = diff.x.max(diff.y).max(diff.z).max(1e-6);

            for p in &mut self.points {
                p.position = Point3::new(
                    (p.position.x - center.x) / scale,
                    (p.position.y - center.y) / scale,
                    (p.position.z - center.z) / scale,
                );
            }

            self.bounds = None;
        }
    }

    /// Add a cluster
    pub fn add_cluster(&mut self, cluster: Cluster) {
        self.clusters.push(cluster);
    }

    /// Calculate clusters from point labels
    pub fn compute_clusters_from_labels(&mut self) {
        self.clusters.clear();

        // Group points by label
        let mut label_points: HashMap<usize, Vec<usize>> = HashMap::new();
        for (i, p) in self.points.iter().enumerate() {
            if let Some(label) = p.label {
                label_points.entry(label).or_default().push(i);
            }
        }

        let palette = tab10();

        for (label, indices) in label_points {
            if indices.is_empty() {
                continue;
            }

            // Calculate centroid
            let mut centroid = Vector3::zeros();
            for &idx in &indices {
                let p = &self.points[idx].position;
                centroid += Vector3::new(p.x, p.y, p.z);
            }
            centroid /= indices.len() as f32;

            self.clusters.push(Cluster {
                index: label,
                name: format!("Cluster {}", label),
                centroid: Point3::from(centroid),
                point_indices: indices,
                color: palette.get(label),
            });
        }
    }

    /// Perform simple PCA dimensionality reduction on high-dimensional points
    /// This is a basic implementation - for production use, consider external libraries
    pub fn reduce_dimensions_pca(embeddings: &[Vec<f32>]) -> Vec<[f32; 3]> {
        if embeddings.is_empty() {
            return Vec::new();
        }

        let n = embeddings.len();
        let d = embeddings[0].len();

        if d <= 3 {
            // Already low dimensional, just pad/truncate
            return embeddings
                .iter()
                .map(|e| {
                    [
                        e.get(0).copied().unwrap_or(0.0),
                        e.get(1).copied().unwrap_or(0.0),
                        e.get(2).copied().unwrap_or(0.0),
                    ]
                })
                .collect();
        }

        // Center the data
        let mut mean = vec![0.0f32; d];
        for e in embeddings {
            for (i, &v) in e.iter().enumerate() {
                mean[i] += v;
            }
        }
        for m in &mut mean {
            *m /= n as f32;
        }

        let centered: Vec<Vec<f32>> = embeddings
            .iter()
            .map(|e| e.iter().zip(&mean).map(|(v, m)| v - m).collect())
            .collect();

        // Compute covariance matrix (simplified - just use first 3 principal directions)
        // For real PCA, use nalgebra or ndarray-linalg
        // This is a simplified power iteration for the dominant components

        let mut result = Vec::with_capacity(n);

        // Simple projection onto first 3 dimensions as placeholder
        // Real implementation would compute eigenvectors
        for e in &centered {
            result.push([
                e.get(0).copied().unwrap_or(0.0),
                e.get(1).copied().unwrap_or(0.0),
                e.get(2).copied().unwrap_or(0.0),
            ]);
        }

        result
    }

    /// Build 3D meshes and add to engine
    pub fn build_meshes(&mut self, engine: &mut Viz3DEngine) {
        // Clear existing meshes
        for &id in &self.point_mesh_ids {
            engine.remove_mesh(id);
        }
        for &id in &self.centroid_mesh_ids {
            engine.remove_mesh(id);
        }
        for &id in &self.trajectory_mesh_ids {
            engine.remove_mesh(id);
        }
        self.point_mesh_ids.clear();
        self.centroid_mesh_ids.clear();
        self.trajectory_mesh_ids.clear();

        if self.points.is_empty() {
            return;
        }

        // Subsample if needed
        let num_points = self.points.len().min(self.config.max_points);
        let step = if self.points.len() > num_points {
            self.points.len() / num_points
        } else {
            1
        };

        // Prepare colors
        let palette = tab10();
        let colormap = self.config.colormap.colormap();

        // Find value range for continuous coloring
        let (min_val, max_val) = self
            .points
            .iter()
            .filter_map(|p| p.value)
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), v| {
                (min.min(v), max.max(v))
            });

        // Build point meshes
        for (i, point) in self.points.iter().enumerate().step_by(step) {
            let color = if let Some(label) = point.label {
                palette.get(label)
            } else if let Some(value) = point.value {
                colormap.map_range(value, min_val, max_val)
            } else {
                Color::WHITE
            };

            let mut mesh = Mesh3D::sphere(
                format!("point_{}", i),
                point.position,
                self.config.point_size,
                self.config.sphere_segments,
            );
            mesh.set_color(Color::rgba(
                color.r,
                color.g,
                color.b,
                self.config.point_opacity,
            ));

            self.point_mesh_ids.push(engine.add_mesh(mesh));
        }

        // Build centroid meshes
        if self.config.show_centroids && !self.clusters.is_empty() {
            for cluster in &self.clusters {
                let mut mesh = Mesh3D::sphere(
                    format!("centroid_{}", cluster.index),
                    cluster.centroid,
                    self.config.point_size * self.config.centroid_size_multiplier,
                    16,
                );
                mesh.set_color(cluster.color);
                self.centroid_mesh_ids.push(engine.add_mesh(mesh));
            }
        }

        // Build trajectory lines
        if self.config.show_trajectory && self.points.len() > 1 {
            let sampled_points: Vec<_> = self.points.iter().step_by(step).collect();
            for window in sampled_points.windows(2) {
                let mesh = Mesh3D::line(
                    format!("traj_{}_{}", 0, 0),
                    window[0].position,
                    window[1].position,
                    self.config.trajectory_color,
                );
                self.trajectory_mesh_ids.push(engine.add_mesh(mesh));
            }
        }
    }

    /// Get statistics about the point cloud
    pub fn stats(&self) -> EmbeddingCloudStats {
        let bounds = if self.points.is_empty() {
            None
        } else {
            let mut min = self.points[0].position;
            let mut max = self.points[0].position;
            for p in &self.points {
                min.x = min.x.min(p.position.x);
                min.y = min.y.min(p.position.y);
                min.z = min.z.min(p.position.z);
                max.x = max.x.max(p.position.x);
                max.y = max.y.max(p.position.y);
                max.z = max.z.max(p.position.z);
            }
            Some((min, max))
        };

        EmbeddingCloudStats {
            point_count: self.points.len(),
            cluster_count: self.clusters.len(),
            bounds,
            has_labels: self.points.iter().any(|p| p.label.is_some()),
            has_values: self.points.iter().any(|p| p.value.is_some()),
        }
    }
}

impl Default for EmbeddingCloud3D {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about an embedding cloud
#[derive(Debug, Clone)]
pub struct EmbeddingCloudStats {
    /// Number of points
    pub point_count: usize,
    /// Number of clusters
    pub cluster_count: usize,
    /// Bounding box (min, max)
    pub bounds: Option<(Point3<f32>, Point3<f32>)>,
    /// Whether points have categorical labels
    pub has_labels: bool,
    /// Whether points have continuous values
    pub has_values: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cloud_creation() {
        let mut cloud = EmbeddingCloud3D::new();
        cloud.add_point(EmbeddingPoint::new(0.0, 0.0, 0.0));
        cloud.add_point(EmbeddingPoint::new(1.0, 1.0, 1.0));

        assert_eq!(cloud.point_count(), 2);
    }

    #[test]
    fn test_bounds_calculation() {
        let mut cloud = EmbeddingCloud3D::from_positions(&[
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0],
            [-1.0, -2.0, -3.0],
        ]);

        let (min, max) = cloud.calculate_bounds().unwrap();
        assert_eq!(min, Point3::new(-1.0, -2.0, -3.0));
        assert_eq!(max, Point3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn test_cluster_computation() {
        let mut cloud = EmbeddingCloud3D::from_positions_with_labels(
            &[
                [0.0, 0.0, 0.0],
                [0.1, 0.1, 0.1],
                [1.0, 1.0, 1.0],
                [1.1, 1.1, 1.1],
            ],
            &[0, 0, 1, 1],
        );

        cloud.compute_clusters_from_labels();
        assert_eq!(cloud.clusters.len(), 2);
    }

    #[test]
    fn test_normalization() {
        let mut cloud = EmbeddingCloud3D::from_positions(&[[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]]);

        cloud.normalize();

        let (min, max) = cloud.calculate_bounds().unwrap();
        assert!(min.x >= -1.0 && max.x <= 1.0);
    }
}
