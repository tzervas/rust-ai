//! Loss landscape 3D visualization module.
//!
//! This module provides tools for visualizing the loss landscape of neural networks:
//! - Generate 3D loss surface by perturbing weights in two random directions
//! - Track optimization trajectory through the landscape
//! - Compute gradient fields for visualization
//! - Export mesh data for 3D rendering
//!
//! # Example
//!
//! ```rust,ignore
//! use training_tools::landscape::{LossLandscape, LandscapeConfig, DirectionMethod};
//!
//! // Define a simple loss function
//! let loss_fn = |weights: &[f32]| -> f32 {
//!     weights.iter().map(|w| w * w).sum::<f32>() / weights.len() as f32
//! };
//!
//! // Generate the landscape
//! let weights = vec![0.5; 100];
//! let config = LandscapeConfig::default();
//! let mut landscape = LossLandscape::compute(&weights, loss_fn, &config);
//!
//! // Track training progress
//! landscape.add_trajectory_point(&weights, 0.25);
//!
//! // Export for rendering
//! let (vertices, indices) = landscape.to_mesh();
//! ```

mod gradient;
mod mesh;
mod surface;
mod trajectory;

pub use gradient::{GradientField, GradientSample};
pub use mesh::MeshExporter;
pub use surface::{DirectionMethod, LandscapeConfig, LossSurface};
pub use trajectory::{TrajectoryPoint, TrajectoryTracker};

use std::ops::Range;

/// A complete 3D loss landscape visualization.
///
/// Contains the loss surface, optimization trajectory, and gradient information
/// for visualization and analysis.
#[derive(Debug, Clone)]
pub struct LossLandscape {
    /// 2D grid of loss values [y][x] -> loss
    pub surface: Vec<Vec<f32>>,
    /// Range of x-axis perturbations
    pub x_range: (f32, f32),
    /// Range of y-axis perturbations
    pub y_range: (f32, f32),
    /// Training path through the landscape: [x, y, loss]
    pub trajectory: Vec<[f32; 3]>,
    /// Direction vectors used for perturbation
    direction1: Vec<f32>,
    direction2: Vec<f32>,
    /// Center weights (origin of the landscape)
    center_weights: Vec<f32>,
    /// Grid resolution
    resolution: usize,
}

impl LossLandscape {
    /// Compute a loss landscape by perturbing weights in two directions.
    ///
    /// # Arguments
    /// * `weights` - Current weight vector (center of the landscape)
    /// * `loss_fn` - Function that computes loss for a given weight vector
    /// * `config` - Configuration for landscape generation
    ///
    /// # Returns
    /// A new `LossLandscape` with computed surface values.
    pub fn compute<F>(weights: &[f32], loss_fn: F, config: &LandscapeConfig) -> Self
    where
        F: Fn(&[f32]) -> f32,
    {
        let loss_surface = LossSurface::new(config.clone());
        let (surface, dir1, dir2) = loss_surface.compute(weights, &loss_fn);

        let half_range = config.range;
        LossLandscape {
            surface,
            x_range: (-half_range, half_range),
            y_range: (-half_range, half_range),
            trajectory: Vec::new(),
            direction1: dir1,
            direction2: dir2,
            center_weights: weights.to_vec(),
            resolution: config.resolution,
        }
    }

    /// Add a point to the optimization trajectory.
    ///
    /// Projects the given weights onto the 2D landscape plane and records
    /// the position and loss value.
    ///
    /// # Arguments
    /// * `weights` - Weight vector at this training step
    /// * `loss` - Loss value at this step
    pub fn add_trajectory_point(&mut self, weights: &[f32], loss: f32) {
        let (x, y) = self.project_to_plane(weights);
        self.trajectory.push([x, y, loss]);
    }

    /// Convert the landscape to a 3D mesh for rendering.
    ///
    /// # Returns
    /// A tuple of (vertices, triangle_indices) where:
    /// - vertices: Vec<[f32; 3]> with [x, y, z=loss] coordinates
    /// - indices: Vec<[u32; 3]> with triangle vertex indices
    pub fn to_mesh(&self) -> (Vec<[f32; 3]>, Vec<[u32; 3]>) {
        let exporter = MeshExporter::new(&self.surface, self.x_range, self.y_range);
        exporter.generate_mesh()
    }

    /// Compute the gradient (slope) at a specific point on the landscape.
    ///
    /// Uses finite differences to estimate the gradient.
    ///
    /// # Arguments
    /// * `x` - X coordinate in landscape space
    /// * `y` - Y coordinate in landscape space
    ///
    /// # Returns
    /// Gradient vector [dz/dx, dz/dy]
    pub fn gradient_at(&self, x: f32, y: f32) -> [f32; 2] {
        let field = GradientField::from_surface(&self.surface, self.x_range, self.y_range);
        field.gradient_at(x, y)
    }

    /// Get the loss value at a specific point on the landscape.
    ///
    /// Interpolates between grid points for smooth values.
    ///
    /// # Arguments
    /// * `x` - X coordinate in landscape space
    /// * `y` - Y coordinate in landscape space
    ///
    /// # Returns
    /// Interpolated loss value, or None if out of bounds
    pub fn loss_at(&self, x: f32, y: f32) -> Option<f32> {
        self.interpolate(x, y)
    }

    /// Get the x-axis range of the landscape.
    pub fn x_range(&self) -> Range<f32> {
        self.x_range.0..self.x_range.1
    }

    /// Get the y-axis range of the landscape.
    pub fn y_range(&self) -> Range<f32> {
        self.y_range.0..self.y_range.1
    }

    /// Get the grid resolution.
    pub fn resolution(&self) -> usize {
        self.resolution
    }

    /// Get the minimum loss value in the landscape.
    pub fn min_loss(&self) -> f32 {
        self.surface
            .iter()
            .flat_map(|row| row.iter())
            .copied()
            .fold(f32::INFINITY, f32::min)
    }

    /// Get the maximum loss value in the landscape.
    pub fn max_loss(&self) -> f32 {
        self.surface
            .iter()
            .flat_map(|row| row.iter())
            .copied()
            .fold(f32::NEG_INFINITY, f32::max)
    }

    /// Get the center weights (origin of the landscape).
    pub fn center_weights(&self) -> &[f32] {
        &self.center_weights
    }

    /// Get the first perturbation direction.
    pub fn direction1(&self) -> &[f32] {
        &self.direction1
    }

    /// Get the second perturbation direction.
    pub fn direction2(&self) -> &[f32] {
        &self.direction2
    }

    /// Sample the gradient field at regular intervals.
    ///
    /// # Arguments
    /// * `sample_resolution` - Number of sample points per axis
    ///
    /// # Returns
    /// Vector of gradient samples for visualization
    pub fn sample_gradient_field(&self, sample_resolution: usize) -> Vec<GradientSample> {
        let field = GradientField::from_surface(&self.surface, self.x_range, self.y_range);
        field.sample(sample_resolution)
    }

    // ========== Internal Methods ==========

    /// Project a weight vector onto the 2D landscape plane.
    fn project_to_plane(&self, weights: &[f32]) -> (f32, f32) {
        if weights.len() != self.center_weights.len() {
            return (0.0, 0.0);
        }

        // Compute delta from center
        let delta: Vec<f32> = weights
            .iter()
            .zip(&self.center_weights)
            .map(|(w, c)| w - c)
            .collect();

        // Project onto direction vectors
        let x = dot_product(&delta, &self.direction1);
        let y = dot_product(&delta, &self.direction2);

        (x, y)
    }

    /// Bilinear interpolation of loss value at a point.
    fn interpolate(&self, x: f32, y: f32) -> Option<f32> {
        let res = self.resolution;
        if res == 0 {
            return None;
        }

        // Convert to grid coordinates
        let x_step = (self.x_range.1 - self.x_range.0) / (res - 1) as f32;
        let y_step = (self.y_range.1 - self.y_range.0) / (res - 1) as f32;

        let fx = (x - self.x_range.0) / x_step;
        let fy = (y - self.y_range.0) / y_step;

        // Check bounds
        if fx < 0.0 || fx > (res - 1) as f32 || fy < 0.0 || fy > (res - 1) as f32 {
            return None;
        }

        // Get grid cell indices
        let x0 = fx.floor() as usize;
        let y0 = fy.floor() as usize;
        let x1 = (x0 + 1).min(res - 1);
        let y1 = (y0 + 1).min(res - 1);

        // Fractional parts
        let tx = fx - x0 as f32;
        let ty = fy - y0 as f32;

        // Bilinear interpolation
        let v00 = self.surface[y0][x0];
        let v10 = self.surface[y0][x1];
        let v01 = self.surface[y1][x0];
        let v11 = self.surface[y1][x1];

        let v0 = v00 * (1.0 - tx) + v10 * tx;
        let v1 = v01 * (1.0 - tx) + v11 * tx;
        let value = v0 * (1.0 - ty) + v1 * ty;

        Some(value)
    }
}

/// Compute the dot product of two vectors.
#[inline]
fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quadratic_loss(weights: &[f32]) -> f32 {
        weights.iter().map(|w| w * w).sum::<f32>() / weights.len() as f32
    }

    #[test]
    fn test_landscape_creation() {
        let weights = vec![0.0; 10];
        let config = LandscapeConfig {
            resolution: 10,
            range: 1.0,
            direction_method: DirectionMethod::Random,
        };

        let landscape = LossLandscape::compute(&weights, quadratic_loss, &config);

        assert_eq!(landscape.surface.len(), 10);
        assert_eq!(landscape.surface[0].len(), 10);
        assert_eq!(landscape.resolution(), 10);
    }

    #[test]
    fn test_loss_interpolation() {
        let weights = vec![0.0; 10];
        let config = LandscapeConfig::default();
        let landscape = LossLandscape::compute(&weights, quadratic_loss, &config);

        // Center should have minimum loss for quadratic
        let center_loss = landscape.loss_at(0.0, 0.0);
        assert!(center_loss.is_some());
    }

    #[test]
    fn test_trajectory_tracking() {
        let weights = vec![0.0; 10];
        let config = LandscapeConfig::default();
        let mut landscape = LossLandscape::compute(&weights, quadratic_loss, &config);

        landscape.add_trajectory_point(&weights, 0.0);
        landscape.add_trajectory_point(&vec![0.1; 10], 0.01);

        assert_eq!(landscape.trajectory.len(), 2);
    }

    #[test]
    fn test_mesh_generation() {
        let weights = vec![0.0; 10];
        let config = LandscapeConfig {
            resolution: 5,
            range: 1.0,
            direction_method: DirectionMethod::Random,
        };

        let landscape = LossLandscape::compute(&weights, quadratic_loss, &config);
        let (vertices, indices) = landscape.to_mesh();

        // 5x5 grid = 25 vertices
        assert_eq!(vertices.len(), 25);
        // 4x4 cells * 2 triangles = 32 triangles
        assert_eq!(indices.len(), 32);
    }

    #[test]
    fn test_gradient_computation() {
        let weights = vec![0.0; 10];
        let config = LandscapeConfig::default();
        let landscape = LossLandscape::compute(&weights, quadratic_loss, &config);

        let gradient = landscape.gradient_at(0.0, 0.0);
        // At center of quadratic, gradient should be near zero
        assert!(gradient[0].abs() < 0.1);
        assert!(gradient[1].abs() < 0.1);
    }

    #[test]
    fn test_min_max_loss() {
        let weights = vec![0.0; 10];
        let config = LandscapeConfig::default();
        let landscape = LossLandscape::compute(&weights, quadratic_loss, &config);

        let min_loss = landscape.min_loss();
        let max_loss = landscape.max_loss();

        assert!(min_loss < max_loss);
        assert!(min_loss >= 0.0); // Quadratic is non-negative
    }
}
