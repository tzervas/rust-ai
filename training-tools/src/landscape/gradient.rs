//! Gradient field computation and visualization.
//!
//! Computes gradient (slope) information from the loss surface for
//! visualizing optimization directions and local curvature.

/// A gradient sample at a specific point in the landscape.
#[derive(Debug, Clone, Copy)]
pub struct GradientSample {
    /// X coordinate in landscape space
    pub x: f32,
    /// Y coordinate in landscape space
    pub y: f32,
    /// Loss value at this point
    pub loss: f32,
    /// Gradient vector [dL/dx, dL/dy]
    pub gradient: [f32; 2],
    /// Gradient magnitude
    pub magnitude: f32,
    /// Gradient direction (radians, 0 = positive x)
    pub direction: f32,
}

impl GradientSample {
    /// Create a new gradient sample.
    pub fn new(x: f32, y: f32, loss: f32, gradient: [f32; 2]) -> Self {
        let magnitude = (gradient[0] * gradient[0] + gradient[1] * gradient[1]).sqrt();
        let direction = gradient[1].atan2(gradient[0]);

        Self {
            x,
            y,
            loss,
            gradient,
            magnitude,
            direction,
        }
    }

    /// Get the normalized gradient direction.
    pub fn normalized_direction(&self) -> [f32; 2] {
        if self.magnitude < 1e-10 {
            [0.0, 0.0]
        } else {
            [
                self.gradient[0] / self.magnitude,
                self.gradient[1] / self.magnitude,
            ]
        }
    }

    /// Get the steepest descent direction (negative gradient).
    pub fn descent_direction(&self) -> [f32; 2] {
        let norm = self.normalized_direction();
        [-norm[0], -norm[1]]
    }

    /// Convert to 3D position for rendering.
    pub fn to_3d(&self) -> [f32; 3] {
        [self.x, self.y, self.loss]
    }

    /// Get an arrow endpoint for visualization.
    ///
    /// # Arguments
    /// * `scale` - Scale factor for arrow length
    /// * `descent` - If true, point in descent direction
    pub fn arrow_endpoint(&self, scale: f32, descent: bool) -> [f32; 3] {
        let dir = if descent {
            self.descent_direction()
        } else {
            self.normalized_direction()
        };

        let len = self.magnitude.min(1.0) * scale;
        [self.x + dir[0] * len, self.y + dir[1] * len, self.loss]
    }
}

/// Computes gradient field from a loss surface.
pub struct GradientField {
    /// Grid of gradient vectors [y][x] -> [dL/dx, dL/dy]
    gradients: Vec<Vec<[f32; 2]>>,
    /// Reference to surface values for interpolation
    surface: Vec<Vec<f32>>,
    /// X-axis range
    x_range: (f32, f32),
    /// Y-axis range
    y_range: (f32, f32),
}

impl GradientField {
    /// Compute gradient field from a loss surface.
    ///
    /// # Arguments
    /// * `surface` - 2D grid of loss values [y][x]
    /// * `x_range` - (min, max) of x-axis
    /// * `y_range` - (min, max) of y-axis
    pub fn from_surface(surface: &[Vec<f32>], x_range: (f32, f32), y_range: (f32, f32)) -> Self {
        if surface.is_empty() || surface[0].is_empty() {
            return Self {
                gradients: Vec::new(),
                surface: Vec::new(),
                x_range,
                y_range,
            };
        }

        let height = surface.len();
        let width = surface[0].len();
        let dx = (x_range.1 - x_range.0) / (width - 1) as f32;
        let dy = (y_range.1 - y_range.0) / (height - 1) as f32;

        let mut gradients = vec![vec![[0.0f32; 2]; width]; height];

        for j in 0..height {
            for i in 0..width {
                // Central differences where possible, forward/backward at edges
                let dL_dx = if i == 0 {
                    (surface[j][i + 1] - surface[j][i]) / dx
                } else if i == width - 1 {
                    (surface[j][i] - surface[j][i - 1]) / dx
                } else {
                    (surface[j][i + 1] - surface[j][i - 1]) / (2.0 * dx)
                };

                let dL_dy = if j == 0 {
                    (surface[j + 1][i] - surface[j][i]) / dy
                } else if j == height - 1 {
                    (surface[j][i] - surface[j - 1][i]) / dy
                } else {
                    (surface[j + 1][i] - surface[j - 1][i]) / (2.0 * dy)
                };

                gradients[j][i] = [dL_dx, dL_dy];
            }
        }

        Self {
            gradients,
            surface: surface.to_vec(),
            x_range,
            y_range,
        }
    }

    /// Get the gradient at a specific point using bilinear interpolation.
    pub fn gradient_at(&self, x: f32, y: f32) -> [f32; 2] {
        if self.gradients.is_empty() {
            return [0.0, 0.0];
        }

        let height = self.gradients.len();
        let width = self.gradients[0].len();

        // Convert to grid coordinates
        let dx = (self.x_range.1 - self.x_range.0) / (width - 1) as f32;
        let dy = (self.y_range.1 - self.y_range.0) / (height - 1) as f32;

        let fx = (x - self.x_range.0) / dx;
        let fy = (y - self.y_range.0) / dy;

        // Clamp to grid bounds
        let fx = fx.clamp(0.0, (width - 1) as f32);
        let fy = fy.clamp(0.0, (height - 1) as f32);

        let i0 = fx.floor() as usize;
        let j0 = fy.floor() as usize;
        let i1 = (i0 + 1).min(width - 1);
        let j1 = (j0 + 1).min(height - 1);

        let tx = fx - i0 as f32;
        let ty = fy - j0 as f32;

        // Bilinear interpolation of gradient components
        let g00 = self.gradients[j0][i0];
        let g10 = self.gradients[j0][i1];
        let g01 = self.gradients[j1][i0];
        let g11 = self.gradients[j1][i1];

        let gx = g00[0] * (1.0 - tx) * (1.0 - ty)
            + g10[0] * tx * (1.0 - ty)
            + g01[0] * (1.0 - tx) * ty
            + g11[0] * tx * ty;

        let gy = g00[1] * (1.0 - tx) * (1.0 - ty)
            + g10[1] * tx * (1.0 - ty)
            + g01[1] * (1.0 - tx) * ty
            + g11[1] * tx * ty;

        [gx, gy]
    }

    /// Get the loss value at a specific point using bilinear interpolation.
    pub fn loss_at(&self, x: f32, y: f32) -> f32 {
        if self.surface.is_empty() {
            return 0.0;
        }

        let height = self.surface.len();
        let width = self.surface[0].len();

        let dx = (self.x_range.1 - self.x_range.0) / (width - 1) as f32;
        let dy = (self.y_range.1 - self.y_range.0) / (height - 1) as f32;

        let fx = ((x - self.x_range.0) / dx).clamp(0.0, (width - 1) as f32);
        let fy = ((y - self.y_range.0) / dy).clamp(0.0, (height - 1) as f32);

        let i0 = fx.floor() as usize;
        let j0 = fy.floor() as usize;
        let i1 = (i0 + 1).min(width - 1);
        let j1 = (j0 + 1).min(height - 1);

        let tx = fx - i0 as f32;
        let ty = fy - j0 as f32;

        let v00 = self.surface[j0][i0];
        let v10 = self.surface[j0][i1];
        let v01 = self.surface[j1][i0];
        let v11 = self.surface[j1][i1];

        v00 * (1.0 - tx) * (1.0 - ty)
            + v10 * tx * (1.0 - ty)
            + v01 * (1.0 - tx) * ty
            + v11 * tx * ty
    }

    /// Sample the gradient field at regular intervals.
    ///
    /// # Arguments
    /// * `resolution` - Number of samples per axis
    ///
    /// # Returns
    /// Vector of gradient samples
    pub fn sample(&self, resolution: usize) -> Vec<GradientSample> {
        if resolution < 2 || self.gradients.is_empty() {
            return Vec::new();
        }

        let dx = (self.x_range.1 - self.x_range.0) / (resolution - 1) as f32;
        let dy = (self.y_range.1 - self.y_range.0) / (resolution - 1) as f32;

        let mut samples = Vec::with_capacity(resolution * resolution);

        for j in 0..resolution {
            let y = self.y_range.0 + j as f32 * dy;
            for i in 0..resolution {
                let x = self.x_range.0 + i as f32 * dx;
                let gradient = self.gradient_at(x, y);
                let loss = self.loss_at(x, y);
                samples.push(GradientSample::new(x, y, loss, gradient));
            }
        }

        samples
    }

    /// Get the maximum gradient magnitude in the field.
    pub fn max_magnitude(&self) -> f32 {
        self.gradients
            .iter()
            .flat_map(|row| row.iter())
            .map(|g| (g[0] * g[0] + g[1] * g[1]).sqrt())
            .fold(0.0f32, f32::max)
    }

    /// Get the average gradient magnitude.
    pub fn avg_magnitude(&self) -> f32 {
        if self.gradients.is_empty() {
            return 0.0;
        }

        let total: f32 = self
            .gradients
            .iter()
            .flat_map(|row| row.iter())
            .map(|g| (g[0] * g[0] + g[1] * g[1]).sqrt())
            .sum();

        let count = self.gradients.len() * self.gradients[0].len();
        total / count as f32
    }

    /// Find local minima in the gradient field.
    ///
    /// Points where gradient magnitude is below threshold and
    /// surrounded by higher gradients.
    pub fn find_minima(&self, threshold: f32) -> Vec<(f32, f32, f32)> {
        if self.gradients.is_empty() || self.gradients.len() < 3 {
            return Vec::new();
        }

        let height = self.gradients.len();
        let width = self.gradients[0].len();
        let dx = (self.x_range.1 - self.x_range.0) / (width - 1) as f32;
        let dy = (self.y_range.1 - self.y_range.0) / (height - 1) as f32;

        let mut minima = Vec::new();

        for j in 1..(height - 1) {
            for i in 1..(width - 1) {
                let g = &self.gradients[j][i];
                let mag = (g[0] * g[0] + g[1] * g[1]).sqrt();

                if mag > threshold {
                    continue;
                }

                // Check if local minimum (loss lower than neighbors)
                let loss = self.surface[j][i];
                let is_minimum = loss < self.surface[j - 1][i]
                    && loss < self.surface[j + 1][i]
                    && loss < self.surface[j][i - 1]
                    && loss < self.surface[j][i + 1];

                if is_minimum {
                    let x = self.x_range.0 + i as f32 * dx;
                    let y = self.y_range.0 + j as f32 * dy;
                    minima.push((x, y, loss));
                }
            }
        }

        minima
    }

    /// Find saddle points (where gradient is low but curvature changes sign).
    pub fn find_saddle_points(&self, threshold: f32) -> Vec<(f32, f32, f32)> {
        if self.gradients.is_empty() || self.gradients.len() < 3 {
            return Vec::new();
        }

        let height = self.gradients.len();
        let width = self.gradients[0].len();
        let dx = (self.x_range.1 - self.x_range.0) / (width - 1) as f32;
        let dy = (self.y_range.1 - self.y_range.0) / (height - 1) as f32;

        let mut saddles = Vec::new();

        for j in 1..(height - 1) {
            for i in 1..(width - 1) {
                let g = &self.gradients[j][i];
                let mag = (g[0] * g[0] + g[1] * g[1]).sqrt();

                if mag > threshold {
                    continue;
                }

                // Check for saddle: minimum in one direction, maximum in other
                let loss = self.surface[j][i];
                let is_x_min = loss < self.surface[j][i - 1] && loss < self.surface[j][i + 1];
                let is_x_max = loss > self.surface[j][i - 1] && loss > self.surface[j][i + 1];
                let is_y_min = loss < self.surface[j - 1][i] && loss < self.surface[j + 1][i];
                let is_y_max = loss > self.surface[j - 1][i] && loss > self.surface[j + 1][i];

                let is_saddle = (is_x_min && is_y_max) || (is_x_max && is_y_min);

                if is_saddle {
                    let x = self.x_range.0 + i as f32 * dx;
                    let y = self.y_range.0 + j as f32 * dy;
                    saddles.push((x, y, loss));
                }
            }
        }

        saddles
    }

    /// Compute the Hessian (second derivatives) at a point.
    ///
    /// Returns [[d2L/dx2, d2L/dxdy], [d2L/dxdy, d2L/dy2]]
    pub fn hessian_at(&self, x: f32, y: f32) -> [[f32; 2]; 2] {
        if self.gradients.is_empty() {
            return [[0.0; 2]; 2];
        }

        let height = self.gradients.len();
        let width = self.gradients[0].len();
        let dx = (self.x_range.1 - self.x_range.0) / (width - 1) as f32;
        let dy = (self.y_range.1 - self.y_range.0) / (height - 1) as f32;

        // Get gradients at nearby points
        let eps_x = dx * 0.5;
        let eps_y = dy * 0.5;

        let g_xp = self.gradient_at(x + eps_x, y);
        let g_xm = self.gradient_at(x - eps_x, y);
        let g_yp = self.gradient_at(x, y + eps_y);
        let g_ym = self.gradient_at(x, y - eps_y);

        // Second derivatives
        let d2L_dx2 = (g_xp[0] - g_xm[0]) / (2.0 * eps_x);
        let d2L_dy2 = (g_yp[1] - g_ym[1]) / (2.0 * eps_y);
        let d2L_dxdy = (g_yp[0] - g_ym[0]) / (2.0 * eps_y);

        [[d2L_dx2, d2L_dxdy], [d2L_dxdy, d2L_dy2]]
    }

    /// Classify the curvature at a point.
    ///
    /// Returns the eigenvalues of the Hessian, which indicate:
    /// - Both positive: local minimum (bowl)
    /// - Both negative: local maximum (hill)
    /// - Mixed signs: saddle point
    pub fn curvature_at(&self, x: f32, y: f32) -> (f32, f32) {
        let h = self.hessian_at(x, y);

        // Eigenvalues of 2x2 symmetric matrix
        let trace = h[0][0] + h[1][1];
        let det = h[0][0] * h[1][1] - h[0][1] * h[1][0];

        let discriminant = trace * trace - 4.0 * det;
        if discriminant < 0.0 {
            // Complex eigenvalues (shouldn't happen for real Hessian)
            return (trace / 2.0, trace / 2.0);
        }

        let sqrt_disc = discriminant.sqrt();
        let lambda1 = (trace + sqrt_disc) / 2.0;
        let lambda2 = (trace - sqrt_disc) / 2.0;

        (lambda1, lambda2)
    }

    /// Generate arrow data for visualization.
    ///
    /// # Arguments
    /// * `resolution` - Number of arrows per axis
    /// * `scale` - Scale factor for arrow length
    /// * `descent` - If true, show descent directions
    ///
    /// # Returns
    /// Vector of (start, end) 3D points for arrows
    pub fn arrow_field(
        &self,
        resolution: usize,
        scale: f32,
        descent: bool,
    ) -> Vec<([f32; 3], [f32; 3])> {
        self.sample(resolution)
            .into_iter()
            .filter(|s| s.magnitude > 1e-6)
            .map(|s| (s.to_3d(), s.arrow_endpoint(scale, descent)))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quadratic_surface() -> Vec<Vec<f32>> {
        // L(x,y) = x^2 + y^2 centered at origin
        let res = 11;
        let mut surface = vec![vec![0.0; res]; res];

        for j in 0..res {
            for i in 0..res {
                let x = (i as f32 - 5.0) / 5.0;
                let y = (j as f32 - 5.0) / 5.0;
                surface[j][i] = x * x + y * y;
            }
        }

        surface
    }

    fn saddle_surface() -> Vec<Vec<f32>> {
        // L(x,y) = x^2 - y^2 (saddle at origin)
        let res = 11;
        let mut surface = vec![vec![0.0; res]; res];

        for j in 0..res {
            for i in 0..res {
                let x = (i as f32 - 5.0) / 5.0;
                let y = (j as f32 - 5.0) / 5.0;
                surface[j][i] = x * x - y * y;
            }
        }

        surface
    }

    #[test]
    fn test_gradient_computation() {
        let surface = quadratic_surface();
        let field = GradientField::from_surface(&surface, (-1.0, 1.0), (-1.0, 1.0));

        // At center, gradient should be near zero
        let g_center = field.gradient_at(0.0, 0.0);
        assert!(g_center[0].abs() < 0.1);
        assert!(g_center[1].abs() < 0.1);

        // At x=0.5, gradient should point positive x
        let g_right = field.gradient_at(0.5, 0.0);
        assert!(g_right[0] > 0.5);
    }

    #[test]
    fn test_gradient_sample() {
        let sample = GradientSample::new(1.0, 2.0, 0.5, [3.0, 4.0]);

        assert_eq!(sample.x, 1.0);
        assert_eq!(sample.y, 2.0);
        assert_eq!(sample.loss, 0.5);
        assert!((sample.magnitude - 5.0).abs() < 1e-5);

        let norm = sample.normalized_direction();
        assert!((norm[0] - 0.6).abs() < 1e-5);
        assert!((norm[1] - 0.8).abs() < 1e-5);
    }

    #[test]
    fn test_sample_field() {
        let surface = quadratic_surface();
        let field = GradientField::from_surface(&surface, (-1.0, 1.0), (-1.0, 1.0));

        let samples = field.sample(5);

        assert_eq!(samples.len(), 25); // 5x5 grid
    }

    #[test]
    fn test_find_minima() {
        let surface = quadratic_surface();
        let field = GradientField::from_surface(&surface, (-1.0, 1.0), (-1.0, 1.0));

        let minima = field.find_minima(0.5);

        // Should find minimum near center
        assert!(!minima.is_empty());
        let (x, y, _) = minima[0];
        assert!(x.abs() < 0.3);
        assert!(y.abs() < 0.3);
    }

    #[test]
    fn test_find_saddle() {
        let surface = saddle_surface();
        let field = GradientField::from_surface(&surface, (-1.0, 1.0), (-1.0, 1.0));

        let saddles = field.find_saddle_points(0.5);

        // Should find saddle near center
        assert!(!saddles.is_empty());
        let (x, y, _) = saddles[0];
        assert!(x.abs() < 0.3);
        assert!(y.abs() < 0.3);
    }

    #[test]
    fn test_curvature() {
        let surface = quadratic_surface();
        let field = GradientField::from_surface(&surface, (-1.0, 1.0), (-1.0, 1.0));

        let (l1, l2) = field.curvature_at(0.0, 0.0);

        // For x^2 + y^2, both eigenvalues should be positive
        assert!(l1 > 0.0);
        assert!(l2 > 0.0);
    }

    #[test]
    fn test_curvature_saddle() {
        let surface = saddle_surface();
        let field = GradientField::from_surface(&surface, (-1.0, 1.0), (-1.0, 1.0));

        let (l1, l2) = field.curvature_at(0.0, 0.0);

        // For x^2 - y^2, eigenvalues should have opposite signs
        assert!(l1 * l2 < 0.0);
    }

    #[test]
    fn test_max_magnitude() {
        let surface = quadratic_surface();
        let field = GradientField::from_surface(&surface, (-1.0, 1.0), (-1.0, 1.0));

        let max_mag = field.max_magnitude();

        // Max should be at corners
        assert!(max_mag > 1.0);
    }

    #[test]
    fn test_arrow_field() {
        let surface = quadratic_surface();
        let field = GradientField::from_surface(&surface, (-1.0, 1.0), (-1.0, 1.0));

        let arrows = field.arrow_field(5, 0.1, true);

        // Most points should have arrows (except near center)
        assert!(!arrows.is_empty());
    }

    #[test]
    fn test_hessian() {
        let surface = quadratic_surface();
        let field = GradientField::from_surface(&surface, (-1.0, 1.0), (-1.0, 1.0));

        let h = field.hessian_at(0.0, 0.0);

        // For x^2 + y^2, d2L/dx2 = 2, d2L/dy2 = 2, d2L/dxdy = 0
        assert!(h[0][0] > 0.0);
        assert!(h[1][1] > 0.0);
        assert!(h[0][1].abs() < 0.5); // Cross derivative should be small
    }

    #[test]
    fn test_empty_field() {
        let surface: Vec<Vec<f32>> = vec![];
        let field = GradientField::from_surface(&surface, (-1.0, 1.0), (-1.0, 1.0));

        assert_eq!(field.gradient_at(0.0, 0.0), [0.0, 0.0]);
        assert!(field.sample(5).is_empty());
    }
}
