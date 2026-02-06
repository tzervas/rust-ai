//! Core 3D visualization engine using three-d
//!
//! Provides the main rendering infrastructure for scientific visualization.

use std::collections::HashMap;

use nalgebra::{Point3, Vector3};

use super::colors::{Color, Colormap, ColormapPreset};

/// Configuration for the 3D visualization engine
#[derive(Debug, Clone)]
pub struct Viz3DConfig {
    /// Window title
    pub title: String,
    /// Window width in pixels
    pub width: u32,
    /// Window height in pixels
    pub height: u32,
    /// Background color
    pub background_color: Color,
    /// Enable anti-aliasing
    pub antialias: bool,
    /// Default colormap for scalar visualization
    pub default_colormap: ColormapPreset,
    /// Enable grid display
    pub show_grid: bool,
    /// Enable axes display
    pub show_axes: bool,
    /// Camera field of view in degrees
    pub camera_fov: f32,
    /// Camera near plane
    pub camera_near: f32,
    /// Camera far plane
    pub camera_far: f32,
}

impl Default for Viz3DConfig {
    fn default() -> Self {
        Self {
            title: "Rust AI Visualization".to_string(),
            width: 1280,
            height: 720,
            background_color: Color::rgb(0.1, 0.1, 0.15),
            antialias: true,
            default_colormap: ColormapPreset::Viridis,
            show_grid: true,
            show_axes: true,
            camera_fov: 45.0,
            camera_near: 0.1,
            camera_far: 1000.0,
        }
    }
}

/// Camera controls for 3D navigation
#[derive(Debug, Clone)]
pub struct Camera3D {
    /// Camera position in world space
    pub position: Point3<f32>,
    /// Point the camera is looking at
    pub target: Point3<f32>,
    /// Up direction
    pub up: Vector3<f32>,
    /// Field of view in degrees
    pub fov: f32,
    /// Near clipping plane
    pub near: f32,
    /// Far clipping plane
    pub far: f32,
    /// Orbit distance from target
    pub orbit_distance: f32,
    /// Horizontal orbit angle (azimuth) in radians
    pub azimuth: f32,
    /// Vertical orbit angle (elevation) in radians
    pub elevation: f32,
}

impl Default for Camera3D {
    fn default() -> Self {
        let distance = 10.0;
        let azimuth = std::f32::consts::FRAC_PI_4; // 45 degrees
        let elevation = std::f32::consts::FRAC_PI_6; // 30 degrees

        let position = Self::orbit_position(distance, azimuth, elevation, Point3::origin());

        Self {
            position,
            target: Point3::origin(),
            up: Vector3::new(0.0, 1.0, 0.0),
            fov: 45.0,
            near: 0.1,
            far: 1000.0,
            orbit_distance: distance,
            azimuth,
            elevation,
        }
    }
}

impl Camera3D {
    /// Create a new camera at specified position looking at target
    pub fn new(position: Point3<f32>, target: Point3<f32>) -> Self {
        let diff = position - target;
        let distance = diff.norm();
        let azimuth = diff.z.atan2(diff.x);
        let elevation = (diff.y / distance).asin();

        Self {
            position,
            target,
            orbit_distance: distance,
            azimuth,
            elevation,
            ..Default::default()
        }
    }

    /// Calculate camera position from orbit parameters
    fn orbit_position(
        distance: f32,
        azimuth: f32,
        elevation: f32,
        target: Point3<f32>,
    ) -> Point3<f32> {
        let x = distance * elevation.cos() * azimuth.cos();
        let y = distance * elevation.sin();
        let z = distance * elevation.cos() * azimuth.sin();
        Point3::new(target.x + x, target.y + y, target.z + z)
    }

    /// Orbit the camera horizontally (change azimuth)
    pub fn orbit_horizontal(&mut self, delta: f32) {
        self.azimuth += delta;
        self.update_position();
    }

    /// Orbit the camera vertically (change elevation)
    pub fn orbit_vertical(&mut self, delta: f32) {
        self.elevation = (self.elevation + delta).clamp(
            -std::f32::consts::FRAC_PI_2 + 0.01,
            std::f32::consts::FRAC_PI_2 - 0.01,
        );
        self.update_position();
    }

    /// Zoom the camera (change orbit distance)
    pub fn zoom(&mut self, factor: f32) {
        self.orbit_distance = (self.orbit_distance * factor).max(0.1);
        self.update_position();
    }

    /// Pan the camera (move target position)
    pub fn pan(&mut self, dx: f32, dy: f32) {
        // Calculate camera right and up vectors
        let forward = (self.target - self.position).normalize();
        let right = forward.cross(&self.up).normalize();
        let cam_up = right.cross(&forward);

        let offset = right * dx + cam_up * dy;
        self.target += offset;
        self.update_position();
    }

    /// Update camera position from orbit parameters
    fn update_position(&mut self) {
        self.position = Self::orbit_position(
            self.orbit_distance,
            self.azimuth,
            self.elevation,
            self.target,
        );
    }

    /// Get the view matrix
    pub fn view_matrix(&self) -> nalgebra::Matrix4<f32> {
        nalgebra::Matrix4::look_at_rh(&self.position, &self.target, &self.up)
    }

    /// Reset camera to default orbit view
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Set camera to look at a bounding box
    pub fn fit_to_bounds(&mut self, min: Point3<f32>, max: Point3<f32>) {
        let center = Point3::new(
            (min.x + max.x) / 2.0,
            (min.y + max.y) / 2.0,
            (min.z + max.z) / 2.0,
        );
        let size = (max - min).norm();

        self.target = center;
        self.orbit_distance = size * 2.0;
        self.update_position();
    }
}

/// A 3D scene object identifier
pub type ObjectId = u64;

/// Represents a vertex in 3D space with optional attributes
#[derive(Debug, Clone, Copy)]
pub struct Vertex3D {
    /// Position in 3D space
    pub position: Point3<f32>,
    /// Normal vector for lighting
    pub normal: Vector3<f32>,
    /// Vertex color
    pub color: Color,
    /// Texture coordinates (if applicable)
    pub uv: [f32; 2],
}

impl Default for Vertex3D {
    fn default() -> Self {
        Self {
            position: Point3::origin(),
            normal: Vector3::new(0.0, 1.0, 0.0),
            color: Color::WHITE,
            uv: [0.0, 0.0],
        }
    }
}

/// Mesh data for 3D rendering
#[derive(Debug, Clone)]
pub struct Mesh3D {
    /// Unique identifier
    pub id: ObjectId,
    /// Mesh name
    pub name: String,
    /// Vertex data
    pub vertices: Vec<Vertex3D>,
    /// Triangle indices
    pub indices: Vec<u32>,
    /// Is the mesh visible
    pub visible: bool,
    /// Wireframe mode
    pub wireframe: bool,
    /// Transparency (0.0 = invisible, 1.0 = opaque)
    pub opacity: f32,
}

impl Mesh3D {
    /// Create an empty mesh
    pub fn new(name: impl Into<String>) -> Self {
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        Self {
            id: COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst),
            name: name.into(),
            vertices: Vec::new(),
            indices: Vec::new(),
            visible: true,
            wireframe: false,
            opacity: 1.0,
        }
    }

    /// Create a sphere mesh
    pub fn sphere(
        name: impl Into<String>,
        center: Point3<f32>,
        radius: f32,
        segments: u32,
    ) -> Self {
        let mut mesh = Self::new(name);

        // Generate sphere vertices
        for lat in 0..=segments {
            let theta = std::f32::consts::PI * lat as f32 / segments as f32;
            let sin_theta = theta.sin();
            let cos_theta = theta.cos();

            for lon in 0..=segments {
                let phi = 2.0 * std::f32::consts::PI * lon as f32 / segments as f32;
                let sin_phi = phi.sin();
                let cos_phi = phi.cos();

                let x = cos_phi * sin_theta;
                let y = cos_theta;
                let z = sin_phi * sin_theta;

                mesh.vertices.push(Vertex3D {
                    position: Point3::new(
                        center.x + radius * x,
                        center.y + radius * y,
                        center.z + radius * z,
                    ),
                    normal: Vector3::new(x, y, z),
                    color: Color::WHITE,
                    uv: [lon as f32 / segments as f32, lat as f32 / segments as f32],
                });
            }
        }

        // Generate indices
        for lat in 0..segments {
            for lon in 0..segments {
                let first = lat * (segments + 1) + lon;
                let second = first + segments + 1;

                mesh.indices.push(first);
                mesh.indices.push(second);
                mesh.indices.push(first + 1);

                mesh.indices.push(second);
                mesh.indices.push(second + 1);
                mesh.indices.push(first + 1);
            }
        }

        mesh
    }

    /// Create a box mesh
    pub fn cube(name: impl Into<String>, center: Point3<f32>, size: Vector3<f32>) -> Self {
        let mut mesh = Self::new(name);
        let half = size / 2.0;

        // 8 corners
        let corners = [
            Point3::new(center.x - half.x, center.y - half.y, center.z - half.z),
            Point3::new(center.x + half.x, center.y - half.y, center.z - half.z),
            Point3::new(center.x + half.x, center.y + half.y, center.z - half.z),
            Point3::new(center.x - half.x, center.y + half.y, center.z - half.z),
            Point3::new(center.x - half.x, center.y - half.y, center.z + half.z),
            Point3::new(center.x + half.x, center.y - half.y, center.z + half.z),
            Point3::new(center.x + half.x, center.y + half.y, center.z + half.z),
            Point3::new(center.x - half.x, center.y + half.y, center.z + half.z),
        ];

        // 6 faces with proper normals
        let faces = [
            ([0, 1, 2, 3], Vector3::new(0.0, 0.0, -1.0)), // Front
            ([5, 4, 7, 6], Vector3::new(0.0, 0.0, 1.0)),  // Back
            ([4, 0, 3, 7], Vector3::new(-1.0, 0.0, 0.0)), // Left
            ([1, 5, 6, 2], Vector3::new(1.0, 0.0, 0.0)),  // Right
            ([3, 2, 6, 7], Vector3::new(0.0, 1.0, 0.0)),  // Top
            ([4, 5, 1, 0], Vector3::new(0.0, -1.0, 0.0)), // Bottom
        ];

        for (indices, normal) in faces {
            let base = mesh.vertices.len() as u32;
            for &i in &indices {
                mesh.vertices.push(Vertex3D {
                    position: corners[i],
                    normal,
                    color: Color::WHITE,
                    uv: [0.0, 0.0],
                });
            }
            mesh.indices
                .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        }

        mesh
    }

    /// Create a cylinder mesh
    pub fn cylinder(
        name: impl Into<String>,
        base: Point3<f32>,
        height: f32,
        radius: f32,
        segments: u32,
    ) -> Self {
        let mut mesh = Self::new(name);

        // Side vertices
        for i in 0..=segments {
            let angle = 2.0 * std::f32::consts::PI * i as f32 / segments as f32;
            let x = angle.cos();
            let z = angle.sin();

            // Bottom vertex
            mesh.vertices.push(Vertex3D {
                position: Point3::new(base.x + radius * x, base.y, base.z + radius * z),
                normal: Vector3::new(x, 0.0, z),
                color: Color::WHITE,
                uv: [i as f32 / segments as f32, 0.0],
            });

            // Top vertex
            mesh.vertices.push(Vertex3D {
                position: Point3::new(base.x + radius * x, base.y + height, base.z + radius * z),
                normal: Vector3::new(x, 0.0, z),
                color: Color::WHITE,
                uv: [i as f32 / segments as f32, 1.0],
            });
        }

        // Side indices
        for i in 0..segments {
            let base_idx = i * 2;
            mesh.indices.extend_from_slice(&[
                base_idx,
                base_idx + 2,
                base_idx + 1,
                base_idx + 1,
                base_idx + 2,
                base_idx + 3,
            ]);
        }

        mesh
    }

    /// Create a line mesh (rendered as thin cylinders or GL lines)
    pub fn line(
        name: impl Into<String>,
        start: Point3<f32>,
        end: Point3<f32>,
        color: Color,
    ) -> Self {
        let mut mesh = Self::new(name);

        mesh.vertices.push(Vertex3D {
            position: start,
            normal: Vector3::new(0.0, 1.0, 0.0),
            color,
            uv: [0.0, 0.0],
        });

        mesh.vertices.push(Vertex3D {
            position: end,
            normal: Vector3::new(0.0, 1.0, 0.0),
            color,
            uv: [1.0, 0.0],
        });

        mesh.indices.extend_from_slice(&[0, 1]);

        mesh
    }

    /// Calculate bounding box
    pub fn bounds(&self) -> Option<(Point3<f32>, Point3<f32>)> {
        if self.vertices.is_empty() {
            return None;
        }

        let mut min = self.vertices[0].position;
        let mut max = self.vertices[0].position;

        for v in &self.vertices {
            min.x = min.x.min(v.position.x);
            min.y = min.y.min(v.position.y);
            min.z = min.z.min(v.position.z);
            max.x = max.x.max(v.position.x);
            max.y = max.y.max(v.position.y);
            max.z = max.z.max(v.position.z);
        }

        Some((min, max))
    }

    /// Apply a color to all vertices
    pub fn set_color(&mut self, color: Color) {
        for v in &mut self.vertices {
            v.color = color;
        }
    }

    /// Apply colors from a scalar field using a colormap
    pub fn color_by_scalar(&mut self, values: &[f32], colormap: &Colormap) {
        if values.len() != self.vertices.len() {
            return;
        }

        let min = values.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        for (i, v) in self.vertices.iter_mut().enumerate() {
            v.color = colormap.map_range(values[i], min, max);
        }
    }
}

/// Light source for the scene
#[derive(Debug, Clone)]
pub struct Light3D {
    /// Light type
    pub kind: LightKind,
    /// Light color
    pub color: Color,
    /// Light intensity (0.0 - 1.0+)
    pub intensity: f32,
}

/// Types of lights
#[derive(Debug, Clone)]
pub enum LightKind {
    /// Ambient light (uniform everywhere)
    Ambient,
    /// Directional light (like the sun)
    Directional { direction: Vector3<f32> },
    /// Point light (emits in all directions from a position)
    Point { position: Point3<f32>, range: f32 },
}

impl Default for Light3D {
    fn default() -> Self {
        Self {
            kind: LightKind::Directional {
                direction: Vector3::new(-1.0, -1.0, -1.0).normalize(),
            },
            color: Color::WHITE,
            intensity: 1.0,
        }
    }
}

/// Main 3D visualization engine
///
/// This is a logical representation of the 3D scene that can be rendered
/// using the three-d crate when the `viz3d` feature is enabled.
#[derive(Debug)]
pub struct Viz3DEngine {
    /// Engine configuration
    pub config: Viz3DConfig,
    /// Camera
    pub camera: Camera3D,
    /// Scene meshes indexed by ID
    meshes: HashMap<ObjectId, Mesh3D>,
    /// Scene lights
    lights: Vec<Light3D>,
    /// Default colormap
    colormap: Colormap,
    /// Frame counter
    frame: u64,
    /// Is the engine running
    running: bool,
}

impl Viz3DEngine {
    /// Create a new visualization engine with default configuration
    pub fn new() -> Self {
        Self::with_config(Viz3DConfig::default())
    }

    /// Create a new visualization engine with custom configuration
    pub fn with_config(config: Viz3DConfig) -> Self {
        let colormap = config.default_colormap.colormap();

        Self {
            config,
            camera: Camera3D::default(),
            meshes: HashMap::new(),
            lights: vec![
                Light3D {
                    kind: LightKind::Ambient,
                    color: Color::WHITE,
                    intensity: 0.3,
                },
                Light3D::default(),
            ],
            colormap,
            frame: 0,
            running: false,
        }
    }

    /// Add a mesh to the scene
    pub fn add_mesh(&mut self, mesh: Mesh3D) -> ObjectId {
        let id = mesh.id;
        self.meshes.insert(id, mesh);
        id
    }

    /// Remove a mesh from the scene
    pub fn remove_mesh(&mut self, id: ObjectId) -> Option<Mesh3D> {
        self.meshes.remove(&id)
    }

    /// Get a mesh by ID
    pub fn get_mesh(&self, id: ObjectId) -> Option<&Mesh3D> {
        self.meshes.get(&id)
    }

    /// Get a mutable mesh by ID
    pub fn get_mesh_mut(&mut self, id: ObjectId) -> Option<&mut Mesh3D> {
        self.meshes.get_mut(&id)
    }

    /// Get all mesh IDs
    pub fn mesh_ids(&self) -> impl Iterator<Item = ObjectId> + '_ {
        self.meshes.keys().copied()
    }

    /// Clear all meshes
    pub fn clear(&mut self) {
        self.meshes.clear();
    }

    /// Add a light to the scene
    pub fn add_light(&mut self, light: Light3D) {
        self.lights.push(light);
    }

    /// Get the current colormap
    pub fn colormap(&self) -> &Colormap {
        &self.colormap
    }

    /// Set the colormap
    pub fn set_colormap(&mut self, preset: ColormapPreset) {
        self.colormap = preset.colormap();
    }

    /// Fit camera to show all scene content
    pub fn fit_camera_to_scene(&mut self) {
        let mut min = Point3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY);
        let mut max = Point3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
        let mut has_content = false;

        for mesh in self.meshes.values() {
            if !mesh.visible {
                continue;
            }
            if let Some((mesh_min, mesh_max)) = mesh.bounds() {
                min.x = min.x.min(mesh_min.x);
                min.y = min.y.min(mesh_min.y);
                min.z = min.z.min(mesh_min.z);
                max.x = max.x.max(mesh_max.x);
                max.y = max.y.max(mesh_max.y);
                max.z = max.z.max(mesh_max.z);
                has_content = true;
            }
        }

        if has_content {
            self.camera.fit_to_bounds(min, max);
        }
    }

    /// Get current frame number
    pub fn frame(&self) -> u64 {
        self.frame
    }

    /// Advance the frame counter
    pub fn advance_frame(&mut self) {
        self.frame += 1;
    }

    /// Check if engine is running
    pub fn is_running(&self) -> bool {
        self.running
    }

    /// Create axes helper meshes
    pub fn create_axes(&mut self, size: f32) -> [ObjectId; 3] {
        let x_axis = Mesh3D::line(
            "X Axis",
            Point3::origin(),
            Point3::new(size, 0.0, 0.0),
            Color::RED,
        );
        let y_axis = Mesh3D::line(
            "Y Axis",
            Point3::origin(),
            Point3::new(0.0, size, 0.0),
            Color::GREEN,
        );
        let z_axis = Mesh3D::line(
            "Z Axis",
            Point3::origin(),
            Point3::new(0.0, 0.0, size),
            Color::BLUE,
        );

        [
            self.add_mesh(x_axis),
            self.add_mesh(y_axis),
            self.add_mesh(z_axis),
        ]
    }

    /// Create a grid helper mesh
    pub fn create_grid(&mut self, size: f32, divisions: u32) -> Vec<ObjectId> {
        let mut ids = Vec::new();
        let step = size / divisions as f32;
        let half = size / 2.0;
        let color = Color::rgba(0.5, 0.5, 0.5, 0.5);

        for i in 0..=divisions {
            let pos = -half + step * i as f32;

            // Lines along X
            let line_x = Mesh3D::line(
                format!("Grid X {}", i),
                Point3::new(-half, 0.0, pos),
                Point3::new(half, 0.0, pos),
                color,
            );
            ids.push(self.add_mesh(line_x));

            // Lines along Z
            let line_z = Mesh3D::line(
                format!("Grid Z {}", i),
                Point3::new(pos, 0.0, -half),
                Point3::new(pos, 0.0, half),
                color,
            );
            ids.push(self.add_mesh(line_z));
        }

        ids
    }

    /// Get scene statistics
    pub fn stats(&self) -> SceneStats {
        let mut vertex_count = 0;
        let mut triangle_count = 0;

        for mesh in self.meshes.values() {
            vertex_count += mesh.vertices.len();
            triangle_count += mesh.indices.len() / 3;
        }

        SceneStats {
            mesh_count: self.meshes.len(),
            vertex_count,
            triangle_count,
            light_count: self.lights.len(),
        }
    }
}

impl Default for Viz3DEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Scene statistics
#[derive(Debug, Clone, Copy)]
pub struct SceneStats {
    /// Number of meshes
    pub mesh_count: usize,
    /// Total vertex count
    pub vertex_count: usize,
    /// Total triangle count
    pub triangle_count: usize,
    /// Number of lights
    pub light_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_creation() {
        let engine = Viz3DEngine::new();
        assert_eq!(engine.meshes.len(), 0);
        assert!(engine.lights.len() >= 1);
    }

    #[test]
    fn test_add_remove_mesh() {
        let mut engine = Viz3DEngine::new();
        let mesh = Mesh3D::sphere("test", Point3::origin(), 1.0, 8);
        let id = engine.add_mesh(mesh);

        assert!(engine.get_mesh(id).is_some());
        assert!(engine.remove_mesh(id).is_some());
        assert!(engine.get_mesh(id).is_none());
    }

    #[test]
    fn test_camera_orbit() {
        let mut camera = Camera3D::default();
        let initial_pos = camera.position;

        camera.orbit_horizontal(0.1);
        assert_ne!(camera.position.x, initial_pos.x);
    }

    #[test]
    fn test_mesh_bounds() {
        let mesh = Mesh3D::cube(
            "test",
            Point3::new(1.0, 2.0, 3.0),
            Vector3::new(2.0, 2.0, 2.0),
        );
        let (min, max) = mesh.bounds().unwrap();

        assert!((min.x - 0.0).abs() < 0.01);
        assert!((max.x - 2.0).abs() < 0.01);
    }
}
