//! Loss landscape visualization
//!
//! Provides 3D surface visualization of loss functions for analyzing
//! optimization landscapes, local minima, and training dynamics.

use nalgebra::{Point3, Vector3};

use super::colors::{Color, Colormap, ColormapPreset};
use super::engine::{Mesh3D, ObjectId, Vertex3D, Viz3DEngine};

/// Configuration for loss landscape visualization
#[derive(Debug, Clone)]
pub struct LossLandscapeConfig {
    /// Grid resolution along each axis
    pub resolution: usize,
    /// Physical size of the landscape
    pub size: f32,
    /// Vertical scale factor for loss values
    pub height_scale: f32,
    /// Colormap for loss values
    pub colormap: ColormapPreset,
    /// Show wireframe overlay
    pub show_wireframe: bool,
    /// Wireframe color
    pub wireframe_color: Color,
    /// Surface opacity
    pub surface_opacity: f32,
    /// Show trajectory path
    pub show_trajectory: bool,
    /// Trajectory line color
    pub trajectory_color: Color,
    /// Trajectory point size
    pub trajectory_point_size: f32,
    /// Logarithmic scale for loss values
    pub log_scale: bool,
    /// Clip minimum loss value
    pub loss_min: Option<f32>,
    /// Clip maximum loss value
    pub loss_max: Option<f32>,
}

impl Default for LossLandscapeConfig {
    fn default() -> Self {
        Self {
            resolution: 50,
            size: 10.0,
            height_scale: 1.0,
            colormap: ColormapPreset::Plasma,
            show_wireframe: false,
            wireframe_color: Color::rgba(0.0, 0.0, 0.0, 0.3),
            surface_opacity: 0.9,
            show_trajectory: true,
            trajectory_color: Color::rgb(1.0, 1.0, 1.0),
            trajectory_point_size: 0.1,
            log_scale: false,
            loss_min: None,
            loss_max: None,
        }
    }
}

/// A point in the optimization trajectory
#[derive(Debug, Clone)]
pub struct TrajectoryPoint {
    /// Position in parameter space (x, y)
    pub position: [f32; 2],
    /// Loss value at this position
    pub loss: f32,
    /// Training step number
    pub step: usize,
    /// Learning rate at this step
    pub learning_rate: Option<f32>,
    /// Gradient magnitude at this step
    pub gradient_norm: Option<f32>,
}

impl TrajectoryPoint {
    /// Create a new trajectory point
    pub fn new(x: f32, y: f32, loss: f32, step: usize) -> Self {
        Self {
            position: [x, y],
            loss,
            step,
            learning_rate: None,
            gradient_norm: None,
        }
    }

    /// Set learning rate
    pub fn with_learning_rate(mut self, lr: f32) -> Self {
        self.learning_rate = Some(lr);
        self
    }

    /// Set gradient norm
    pub fn with_gradient_norm(mut self, norm: f32) -> Self {
        self.gradient_norm = Some(norm);
        self
    }
}

/// Method for generating the loss surface
pub enum SurfaceSource {
    /// Direct grid of loss values
    Grid(Vec<Vec<f32>>),
    /// Analytical function
    Function(Box<dyn Fn(f32, f32) -> f32 + Send + Sync>),
    /// Interpolated from sample points
    Interpolated(Vec<([f32; 2], f32)>),
}

// Manual Debug implementation since closures don't implement Debug
impl std::fmt::Debug for SurfaceSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SurfaceSource::Grid(grid) => f.debug_tuple("Grid").field(&grid.len()).finish(),
            SurfaceSource::Function(_) => f.debug_tuple("Function").field(&"<closure>").finish(),
            SurfaceSource::Interpolated(points) => {
                f.debug_tuple("Interpolated").field(&points.len()).finish()
            }
        }
    }
}

/// 3D loss landscape visualization
pub struct LossLandscape3D {
    /// Configuration
    pub config: LossLandscapeConfig,
    /// Surface data source
    surface: Option<SurfaceSource>,
    /// Optimization trajectory
    trajectory: Vec<TrajectoryPoint>,
    /// Generated surface mesh ID
    surface_mesh_id: Option<ObjectId>,
    /// Generated wireframe mesh ID
    wireframe_mesh_id: Option<ObjectId>,
    /// Generated trajectory mesh IDs
    trajectory_mesh_ids: Vec<ObjectId>,
    /// Cached loss range
    loss_range: Option<(f32, f32)>,
}

impl std::fmt::Debug for LossLandscape3D {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LossLandscape3D")
            .field("config", &self.config)
            .field("surface", &self.surface)
            .field("trajectory_len", &self.trajectory.len())
            .finish()
    }
}

impl LossLandscape3D {
    /// Create a new loss landscape visualization
    pub fn new() -> Self {
        Self::with_config(LossLandscapeConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: LossLandscapeConfig) -> Self {
        Self {
            config,
            surface: None,
            trajectory: Vec::new(),
            surface_mesh_id: None,
            wireframe_mesh_id: None,
            trajectory_mesh_ids: Vec::new(),
            loss_range: None,
        }
    }

    /// Set the surface from a grid of loss values
    pub fn set_surface_grid(&mut self, grid: Vec<Vec<f32>>) {
        self.surface = Some(SurfaceSource::Grid(grid));
        self.loss_range = None;
    }

    /// Set the surface from an analytical function
    pub fn set_surface_function<F>(&mut self, f: F)
    where
        F: Fn(f32, f32) -> f32 + Send + Sync + 'static,
    {
        self.surface = Some(SurfaceSource::Function(Box::new(f)));
        self.loss_range = None;
    }

    /// Set the surface from sample points (will be interpolated)
    pub fn set_surface_points(&mut self, points: Vec<([f32; 2], f32)>) {
        self.surface = Some(SurfaceSource::Interpolated(points));
        self.loss_range = None;
    }

    /// Add a trajectory point
    pub fn add_trajectory_point(&mut self, point: TrajectoryPoint) {
        self.trajectory.push(point);
    }

    /// Set the entire trajectory
    pub fn set_trajectory(&mut self, trajectory: Vec<TrajectoryPoint>) {
        self.trajectory = trajectory;
    }

    /// Clear the trajectory
    pub fn clear_trajectory(&mut self) {
        self.trajectory.clear();
    }

    /// Get the current trajectory
    pub fn trajectory(&self) -> &[TrajectoryPoint] {
        &self.trajectory
    }

    /// Sample the loss function at a point
    fn sample_loss(&self, x: f32, y: f32) -> f32 {
        match &self.surface {
            Some(SurfaceSource::Grid(grid)) => {
                if grid.is_empty() || grid[0].is_empty() {
                    return 0.0;
                }
                // Bilinear interpolation
                let half = self.config.size / 2.0;
                let nx = grid[0].len();
                let ny = grid.len();

                let fx =
                    ((x + half) / self.config.size * (nx - 1) as f32).clamp(0.0, (nx - 1) as f32);
                let fy =
                    ((y + half) / self.config.size * (ny - 1) as f32).clamp(0.0, (ny - 1) as f32);

                let ix = fx.floor() as usize;
                let iy = fy.floor() as usize;
                let tx = fx - ix as f32;
                let ty = fy - iy as f32;

                let ix1 = (ix + 1).min(nx - 1);
                let iy1 = (iy + 1).min(ny - 1);

                let v00 = grid[iy][ix];
                let v10 = grid[iy][ix1];
                let v01 = grid[iy1][ix];
                let v11 = grid[iy1][ix1];

                let v0 = v00 * (1.0 - tx) + v10 * tx;
                let v1 = v01 * (1.0 - tx) + v11 * tx;

                v0 * (1.0 - ty) + v1 * ty
            }
            Some(SurfaceSource::Function(f)) => f(x, y),
            Some(SurfaceSource::Interpolated(points)) => {
                // Inverse distance weighting interpolation
                if points.is_empty() {
                    return 0.0;
                }

                let mut weight_sum = 0.0;
                let mut value_sum = 0.0;

                for (pos, val) in points {
                    let dx = x - pos[0];
                    let dy = y - pos[1];
                    let dist_sq = dx * dx + dy * dy;

                    if dist_sq < 1e-10 {
                        return *val;
                    }

                    let weight = 1.0 / dist_sq;
                    weight_sum += weight;
                    value_sum += weight * val;
                }

                value_sum / weight_sum
            }
            None => 0.0,
        }
    }

    /// Calculate loss range
    fn calculate_loss_range(&mut self) -> (f32, f32) {
        if let Some(range) = self.loss_range {
            return range;
        }

        let mut min_loss = f32::INFINITY;
        let mut max_loss = f32::NEG_INFINITY;

        let half = self.config.size / 2.0;
        let step = self.config.size / self.config.resolution as f32;

        for i in 0..=self.config.resolution {
            for j in 0..=self.config.resolution {
                let x = -half + i as f32 * step;
                let y = -half + j as f32 * step;
                let loss = self.sample_loss(x, y);

                if loss.is_finite() {
                    min_loss = min_loss.min(loss);
                    max_loss = max_loss.max(loss);
                }
            }
        }

        // Apply config limits
        if let Some(min) = self.config.loss_min {
            min_loss = min;
        }
        if let Some(max) = self.config.loss_max {
            max_loss = max;
        }

        self.loss_range = Some((min_loss, max_loss));
        (min_loss, max_loss)
    }

    /// Transform loss value for visualization
    fn transform_loss(&self, loss: f32, min_loss: f32, max_loss: f32) -> f32 {
        let loss = loss.clamp(min_loss, max_loss);

        if self.config.log_scale && min_loss > 0.0 {
            let log_min = min_loss.ln();
            let log_max = max_loss.ln();
            let log_loss = loss.max(1e-10).ln();
            (log_loss - log_min) / (log_max - log_min)
        } else {
            (loss - min_loss) / (max_loss - min_loss).max(1e-10)
        }
    }

    /// Build 3D meshes and add to engine
    pub fn build_meshes(&mut self, engine: &mut Viz3DEngine) {
        // Clear existing meshes
        if let Some(id) = self.surface_mesh_id {
            engine.remove_mesh(id);
        }
        if let Some(id) = self.wireframe_mesh_id {
            engine.remove_mesh(id);
        }
        for &id in &self.trajectory_mesh_ids {
            engine.remove_mesh(id);
        }
        self.surface_mesh_id = None;
        self.wireframe_mesh_id = None;
        self.trajectory_mesh_ids.clear();

        if self.surface.is_none() {
            return;
        }

        let (min_loss, max_loss) = self.calculate_loss_range();
        let colormap = self.config.colormap.colormap();

        // Build surface mesh
        let mut mesh = Mesh3D::new("loss_surface");
        let half = self.config.size / 2.0;
        let step = self.config.size / self.config.resolution as f32;
        let res = self.config.resolution;

        // Generate vertices
        for i in 0..=res {
            for j in 0..=res {
                let x = -half + i as f32 * step;
                let z = -half + j as f32 * step;
                let loss = self.sample_loss(x, z);
                let normalized = self.transform_loss(loss, min_loss, max_loss);
                let y = normalized * self.config.height_scale;

                let color = colormap.map(normalized);

                mesh.vertices.push(Vertex3D {
                    position: Point3::new(x, y, z),
                    normal: Vector3::new(0.0, 1.0, 0.0), // Will be computed properly below
                    color: Color::rgba(color.r, color.g, color.b, self.config.surface_opacity),
                    uv: [i as f32 / res as f32, j as f32 / res as f32],
                });
            }
        }

        // Generate indices and compute normals
        for i in 0..res {
            for j in 0..res {
                let base = i * (res + 1) + j;

                // Two triangles per quad
                mesh.indices.extend_from_slice(&[
                    base as u32,
                    (base + res + 1) as u32,
                    (base + 1) as u32,
                    (base + 1) as u32,
                    (base + res + 1) as u32,
                    (base + res + 2) as u32,
                ]);
            }
        }

        // Compute proper normals
        self.compute_normals(&mut mesh);

        self.surface_mesh_id = Some(engine.add_mesh(mesh));

        // Build wireframe mesh
        if self.config.show_wireframe {
            let mut wireframe = Mesh3D::new("loss_wireframe");
            wireframe.wireframe = true;

            // Copy vertices
            for v in &engine
                .get_mesh(self.surface_mesh_id.unwrap())
                .unwrap()
                .vertices
            {
                wireframe.vertices.push(Vertex3D {
                    position: v.position,
                    normal: v.normal,
                    color: self.config.wireframe_color,
                    uv: v.uv,
                });
            }

            // Grid lines only
            for i in 0..=res {
                for j in 0..res {
                    let base = i * (res + 1) + j;
                    wireframe.indices.push(base as u32);
                    wireframe.indices.push((base + 1) as u32);
                }
            }
            for j in 0..=res {
                for i in 0..res {
                    let base = i * (res + 1) + j;
                    wireframe.indices.push(base as u32);
                    wireframe.indices.push((base + res + 1) as u32);
                }
            }

            self.wireframe_mesh_id = Some(engine.add_mesh(wireframe));
        }

        // Build trajectory
        if self.config.show_trajectory && !self.trajectory.is_empty() {
            // Trajectory lines
            let trajectory_colormap = ColormapPreset::Inferno.colormap();

            for window in self.trajectory.windows(2) {
                let t0 = window[0].step as f32 / self.trajectory.last().unwrap().step.max(1) as f32;
                let t1 = window[1].step as f32 / self.trajectory.last().unwrap().step.max(1) as f32;

                let loss0 = self.transform_loss(window[0].loss, min_loss, max_loss);
                let loss1 = self.transform_loss(window[1].loss, min_loss, max_loss);

                let p0 = Point3::new(
                    window[0].position[0],
                    loss0 * self.config.height_scale + 0.01, // Slight offset above surface
                    window[0].position[1],
                );
                let p1 = Point3::new(
                    window[1].position[0],
                    loss1 * self.config.height_scale + 0.01,
                    window[1].position[1],
                );

                let color = trajectory_colormap.map((t0 + t1) / 2.0);
                let mesh = Mesh3D::line(format!("traj_line_{}", window[0].step), p0, p1, color);
                self.trajectory_mesh_ids.push(engine.add_mesh(mesh));
            }

            // Trajectory points
            for (i, point) in self.trajectory.iter().enumerate() {
                let t = point.step as f32 / self.trajectory.last().unwrap().step.max(1) as f32;
                let loss = self.transform_loss(point.loss, min_loss, max_loss);

                let pos = Point3::new(
                    point.position[0],
                    loss * self.config.height_scale + 0.02,
                    point.position[1],
                );

                let mut mesh = Mesh3D::sphere(
                    format!("traj_point_{}", i),
                    pos,
                    self.config.trajectory_point_size,
                    8,
                );
                mesh.set_color(trajectory_colormap.map(t));
                self.trajectory_mesh_ids.push(engine.add_mesh(mesh));
            }
        }
    }

    /// Compute vertex normals from face normals
    fn compute_normals(&self, mesh: &mut Mesh3D) {
        // Reset normals
        for v in &mut mesh.vertices {
            v.normal = Vector3::zeros();
        }

        // Accumulate face normals
        for chunk in mesh.indices.chunks(3) {
            if chunk.len() != 3 {
                continue;
            }

            let i0 = chunk[0] as usize;
            let i1 = chunk[1] as usize;
            let i2 = chunk[2] as usize;

            let p0 = mesh.vertices[i0].position;
            let p1 = mesh.vertices[i1].position;
            let p2 = mesh.vertices[i2].position;

            let v1 = p1 - p0;
            let v2 = p2 - p0;
            let normal = v1.cross(&v2);

            mesh.vertices[i0].normal += normal;
            mesh.vertices[i1].normal += normal;
            mesh.vertices[i2].normal += normal;
        }

        // Normalize
        for v in &mut mesh.vertices {
            let len = v.normal.norm();
            if len > 1e-10 {
                v.normal /= len;
            } else {
                v.normal = Vector3::new(0.0, 1.0, 0.0);
            }
        }
    }

    /// Create a Rosenbrock function landscape (common optimization test function)
    pub fn rosenbrock(a: f32, b: f32) -> Self {
        let mut landscape = Self::new();
        landscape.set_surface_function(move |x, y| {
            let term1 = (a - x).powi(2);
            let term2 = b * (y - x.powi(2)).powi(2);
            term1 + term2
        });
        landscape.config.log_scale = true;
        landscape
    }

    /// Create a Rastrigin function landscape (multimodal test function)
    pub fn rastrigin(n: f32) -> Self {
        use std::f32::consts::PI;
        let mut landscape = Self::new();
        landscape.set_surface_function(move |x, y| {
            n * 2.0
                + (x.powi(2) - n * (2.0 * PI * x).cos())
                + (y.powi(2) - n * (2.0 * PI * y).cos())
        });
        landscape
    }

    /// Create a simple quadratic bowl landscape
    pub fn quadratic_bowl() -> Self {
        let mut landscape = Self::new();
        landscape.set_surface_function(|x, y| x.powi(2) + y.powi(2));
        landscape
    }

    /// Get landscape statistics
    pub fn stats(&self) -> LandscapeStats {
        let (min_loss, max_loss) = self.loss_range.unwrap_or((0.0, 1.0));

        let trajectory_start = self.trajectory.first().map(|p| p.loss);
        let trajectory_end = self.trajectory.last().map(|p| p.loss);

        LandscapeStats {
            resolution: self.config.resolution,
            loss_range: (min_loss, max_loss),
            trajectory_length: self.trajectory.len(),
            trajectory_start,
            trajectory_end,
        }
    }
}

impl Default for LossLandscape3D {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about a loss landscape
#[derive(Debug, Clone)]
pub struct LandscapeStats {
    /// Grid resolution
    pub resolution: usize,
    /// Loss value range
    pub loss_range: (f32, f32),
    /// Number of trajectory points
    pub trajectory_length: usize,
    /// Starting loss value
    pub trajectory_start: Option<f32>,
    /// Ending loss value
    pub trajectory_end: Option<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_landscape_creation() {
        let landscape = LossLandscape3D::quadratic_bowl();
        assert!(landscape.surface.is_some());
    }

    #[test]
    fn test_sample_function() {
        let mut landscape = LossLandscape3D::new();
        landscape.set_surface_function(|x, y| x + y);

        let loss = landscape.sample_loss(1.0, 2.0);
        assert!((loss - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_trajectory() {
        let mut landscape = LossLandscape3D::quadratic_bowl();
        landscape.add_trajectory_point(TrajectoryPoint::new(1.0, 1.0, 2.0, 0));
        landscape.add_trajectory_point(TrajectoryPoint::new(0.5, 0.5, 0.5, 1));
        landscape.add_trajectory_point(TrajectoryPoint::new(0.0, 0.0, 0.0, 2));

        assert_eq!(landscape.trajectory().len(), 3);
    }

    #[test]
    fn test_rosenbrock() {
        let landscape = LossLandscape3D::rosenbrock(1.0, 100.0);
        // Global minimum at (1, 1)
        let loss = landscape.sample_loss(1.0, 1.0);
        assert!(loss < 0.01);
    }
}
