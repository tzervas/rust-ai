//! Loss surface computation via weight perturbation.
//!
//! Generates a 2D grid of loss values by perturbing weights along two directions.
//! Supports multiple methods for choosing perturbation directions:
//! - Random orthonormal directions
//! - PCA of weight change history
//! - Gradient-aligned directions

use rand::Rng;
use rand_distr::{Distribution, StandardNormal};

/// Method for selecting perturbation directions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DirectionMethod {
    /// Random orthonormal directions (normalized Gaussian)
    #[default]
    Random,
    /// Principal Component Analysis of weight change history
    PCA,
    /// Directions aligned with gradient history
    GradientBased,
}

/// Configuration for loss landscape generation.
#[derive(Debug, Clone)]
pub struct LandscapeConfig {
    /// Grid resolution (number of points per axis)
    pub resolution: usize,
    /// Perturbation range (distance from center in each direction)
    pub range: f32,
    /// Method for selecting perturbation directions
    pub direction_method: DirectionMethod,
}

impl Default for LandscapeConfig {
    fn default() -> Self {
        Self {
            resolution: 50,
            range: 1.0,
            direction_method: DirectionMethod::Random,
        }
    }
}

impl LandscapeConfig {
    /// Create a new configuration with specified resolution and range.
    pub fn new(resolution: usize, range: f32) -> Self {
        Self {
            resolution,
            range,
            direction_method: DirectionMethod::Random,
        }
    }

    /// Set the direction method.
    pub fn with_direction_method(mut self, method: DirectionMethod) -> Self {
        self.direction_method = method;
        self
    }
}

/// Computes loss surfaces by perturbing weights along two directions.
pub struct LossSurface {
    config: LandscapeConfig,
    /// Optional weight change history for PCA/gradient methods
    weight_history: Option<Vec<Vec<f32>>>,
    /// Optional gradient history for gradient-based method
    gradient_history: Option<Vec<Vec<f32>>>,
}

impl LossSurface {
    /// Create a new loss surface generator.
    pub fn new(config: LandscapeConfig) -> Self {
        Self {
            config,
            weight_history: None,
            gradient_history: None,
        }
    }

    /// Set weight change history for PCA direction method.
    pub fn with_weight_history(mut self, history: Vec<Vec<f32>>) -> Self {
        self.weight_history = Some(history);
        self
    }

    /// Set gradient history for gradient-based direction method.
    pub fn with_gradient_history(mut self, history: Vec<Vec<f32>>) -> Self {
        self.gradient_history = Some(history);
        self
    }

    /// Compute the loss surface.
    ///
    /// # Arguments
    /// * `center_weights` - Weight vector at the center of the landscape
    /// * `loss_fn` - Function to compute loss for a weight vector
    ///
    /// # Returns
    /// Tuple of (surface grid, direction1, direction2)
    pub fn compute<F>(
        &self,
        center_weights: &[f32],
        loss_fn: F,
    ) -> (Vec<Vec<f32>>, Vec<f32>, Vec<f32>)
    where
        F: Fn(&[f32]) -> f32,
    {
        let dim = center_weights.len();

        // Generate perturbation directions
        let (dir1, dir2) = match self.config.direction_method {
            DirectionMethod::Random => self.random_directions(dim),
            DirectionMethod::PCA => self.pca_directions(dim),
            DirectionMethod::GradientBased => self.gradient_directions(dim),
        };

        // Compute the surface grid
        let surface = self.compute_surface(center_weights, &dir1, &dir2, &loss_fn);

        (surface, dir1, dir2)
    }

    /// Compute the loss surface grid.
    fn compute_surface<F>(
        &self,
        center: &[f32],
        dir1: &[f32],
        dir2: &[f32],
        loss_fn: F,
    ) -> Vec<Vec<f32>>
    where
        F: Fn(&[f32]) -> f32,
    {
        let res = self.config.resolution;
        let range = self.config.range;
        let step = 2.0 * range / (res - 1) as f32;

        let mut surface = vec![vec![0.0; res]; res];
        let mut perturbed = center.to_vec();

        for j in 0..res {
            let y = -range + j as f32 * step;
            for i in 0..res {
                let x = -range + i as f32 * step;

                // Perturb weights: w' = w + x*d1 + y*d2
                for (k, w) in perturbed.iter_mut().enumerate() {
                    *w = center[k] + x * dir1[k] + y * dir2[k];
                }

                surface[j][i] = loss_fn(&perturbed);
            }
        }

        surface
    }

    /// Generate random orthonormal directions.
    fn random_directions(&self, dim: usize) -> (Vec<f32>, Vec<f32>) {
        let mut rng = rand::thread_rng();

        // Generate first random direction
        let mut dir1: Vec<f32> = (0..dim)
            .map(|_| {
                let val: f64 = StandardNormal.sample(&mut rng);
                val as f32
            })
            .collect();
        normalize(&mut dir1);

        // Generate second random direction and orthogonalize
        let mut dir2: Vec<f32> = (0..dim)
            .map(|_| {
                let val: f64 = StandardNormal.sample(&mut rng);
                val as f32
            })
            .collect();
        orthogonalize(&mut dir2, &dir1);
        normalize(&mut dir2);

        (dir1, dir2)
    }

    /// Generate directions from PCA of weight change history.
    fn pca_directions(&self, dim: usize) -> (Vec<f32>, Vec<f32>) {
        if let Some(ref history) = self.weight_history {
            if history.len() >= 2 {
                // Compute weight deltas
                let deltas: Vec<Vec<f32>> = history
                    .windows(2)
                    .map(|w| w[1].iter().zip(&w[0]).map(|(a, b)| a - b).collect())
                    .collect();

                if !deltas.is_empty() {
                    // Use power iteration to find principal components
                    let (pc1, pc2) = power_iteration_pca(&deltas, dim);
                    return (pc1, pc2);
                }
            }
        }

        // Fallback to random directions
        self.random_directions(dim)
    }

    /// Generate directions based on gradient history.
    fn gradient_directions(&self, dim: usize) -> (Vec<f32>, Vec<f32>) {
        if let Some(ref history) = self.gradient_history {
            if history.len() >= 2 {
                // First direction: mean gradient direction
                let mut dir1 = vec![0.0f32; dim];
                for grad in history {
                    for (d, g) in dir1.iter_mut().zip(grad) {
                        *d += g;
                    }
                }
                normalize(&mut dir1);

                // Second direction: gradient variance direction
                let mut dir2 = vec![0.0f32; dim];
                for grad in history {
                    for (d, (g, mean)) in dir2.iter_mut().zip(grad.iter().zip(&dir1)) {
                        *d += (g - mean).powi(2);
                    }
                }
                orthogonalize(&mut dir2, &dir1);
                normalize(&mut dir2);

                return (dir1, dir2);
            }
        }

        // Fallback to random directions
        self.random_directions(dim)
    }
}

/// Normalize a vector to unit length.
fn normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-10 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Orthogonalize vector v with respect to u (v = v - proj_u(v)).
fn orthogonalize(v: &mut [f32], u: &[f32]) {
    let dot: f32 = v.iter().zip(u).map(|(a, b)| a * b).sum();
    for (vi, ui) in v.iter_mut().zip(u) {
        *vi -= dot * ui;
    }
}

/// Simple power iteration for finding top 2 principal components.
fn power_iteration_pca(data: &[Vec<f32>], dim: usize) -> (Vec<f32>, Vec<f32>) {
    let mut rng = rand::thread_rng();
    let n_iterations = 50;

    // Initialize random vector
    let mut v1: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() - 0.5).collect();
    normalize(&mut v1);

    // Power iteration for first PC
    for _ in 0..n_iterations {
        let mut new_v = vec![0.0f32; dim];
        for row in data {
            let dot: f32 = row.iter().zip(&v1).map(|(a, b)| a * b).sum();
            for (nv, r) in new_v.iter_mut().zip(row) {
                *nv += dot * r;
            }
        }
        normalize(&mut new_v);
        v1 = new_v;
    }

    // Initialize second vector orthogonal to first
    let mut v2: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() - 0.5).collect();
    orthogonalize(&mut v2, &v1);
    normalize(&mut v2);

    // Power iteration for second PC (with deflation)
    for _ in 0..n_iterations {
        let mut new_v = vec![0.0f32; dim];
        for row in data {
            let dot: f32 = row.iter().zip(&v2).map(|(a, b)| a * b).sum();
            for (nv, r) in new_v.iter_mut().zip(row) {
                *nv += dot * r;
            }
        }
        orthogonalize(&mut new_v, &v1);
        normalize(&mut new_v);
        v2 = new_v;
    }

    (v1, v2)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_loss(weights: &[f32]) -> f32 {
        weights.iter().map(|w| w * w).sum()
    }

    #[test]
    fn test_random_directions_orthogonal() {
        let surface = LossSurface::new(LandscapeConfig::default());
        let (dir1, dir2) = surface.random_directions(100);

        // Check unit length
        let norm1: f32 = dir1.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm2: f32 = dir2.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm1 - 1.0).abs() < 1e-5);
        assert!((norm2 - 1.0).abs() < 1e-5);

        // Check orthogonality
        let dot: f32 = dir1.iter().zip(&dir2).map(|(a, b)| a * b).sum();
        assert!(dot.abs() < 1e-5);
    }

    #[test]
    fn test_surface_computation() {
        let config = LandscapeConfig {
            resolution: 10,
            range: 1.0,
            direction_method: DirectionMethod::Random,
        };
        let surface = LossSurface::new(config);
        let center = vec![0.0; 5];

        let (grid, _, _) = surface.compute(&center, simple_loss);

        assert_eq!(grid.len(), 10);
        assert_eq!(grid[0].len(), 10);

        // Center should have minimum loss for quadratic
        let center_loss = grid[5][5]; // Approximately center
        let corner_loss = grid[0][0];
        assert!(center_loss < corner_loss);
    }

    #[test]
    fn test_pca_fallback() {
        let config = LandscapeConfig {
            resolution: 5,
            range: 0.5,
            direction_method: DirectionMethod::PCA,
        };
        // No history set, should fallback to random
        let surface = LossSurface::new(config);
        let center = vec![0.0; 10];

        let (grid, dir1, dir2) = surface.compute(&center, simple_loss);

        // Should still produce valid output
        assert_eq!(grid.len(), 5);
        assert_eq!(dir1.len(), 10);
        assert_eq!(dir2.len(), 10);
    }

    #[test]
    fn test_pca_with_history() {
        let config = LandscapeConfig {
            resolution: 5,
            range: 0.5,
            direction_method: DirectionMethod::PCA,
        };

        let history = vec![
            vec![0.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0],
            vec![2.0, 0.1, 0.0],
            vec![3.0, 0.2, 0.1],
        ];

        let surface = LossSurface::new(config).with_weight_history(history);
        let center = vec![3.0, 0.2, 0.1];

        let (grid, dir1, _) = surface.compute(&center, simple_loss);

        assert_eq!(grid.len(), 5);
        // First PC should be mostly aligned with x-axis (largest variance)
        assert!(dir1[0].abs() > 0.9);
    }

    #[test]
    fn test_config_builder() {
        let config =
            LandscapeConfig::new(100, 2.0).with_direction_method(DirectionMethod::GradientBased);

        assert_eq!(config.resolution, 100);
        assert_eq!(config.range, 2.0);
        assert_eq!(config.direction_method, DirectionMethod::GradientBased);
    }

    #[test]
    fn test_normalize() {
        let mut v = vec![3.0, 4.0];
        normalize(&mut v);
        assert!((v[0] - 0.6).abs() < 1e-5);
        assert!((v[1] - 0.8).abs() < 1e-5);
    }

    #[test]
    fn test_orthogonalize() {
        let u = vec![1.0, 0.0];
        let mut v = vec![1.0, 1.0];
        orthogonalize(&mut v, &u);

        // v should now be perpendicular to u
        let dot: f32 = v.iter().zip(&u).map(|(a, b)| a * b).sum();
        assert!(dot.abs() < 1e-5);
    }
}
