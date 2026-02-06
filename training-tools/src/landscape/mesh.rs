//! 3D mesh generation from loss surface data.
//!
//! Converts the 2D grid of loss values into triangle meshes suitable
//! for 3D rendering. Supports multiple output formats and level-of-detail.

/// Exports loss surfaces as 3D meshes.
pub struct MeshExporter<'a> {
    /// Reference to the surface grid [y][x]
    surface: &'a [Vec<f32>],
    /// X-axis range
    x_range: (f32, f32),
    /// Y-axis range
    y_range: (f32, f32),
}

impl<'a> MeshExporter<'a> {
    /// Create a new mesh exporter.
    ///
    /// # Arguments
    /// * `surface` - 2D grid of loss values
    /// * `x_range` - (min, max) of x-axis
    /// * `y_range` - (min, max) of y-axis
    pub fn new(surface: &'a [Vec<f32>], x_range: (f32, f32), y_range: (f32, f32)) -> Self {
        Self {
            surface,
            x_range,
            y_range,
        }
    }

    /// Generate a triangle mesh from the surface.
    ///
    /// # Returns
    /// Tuple of (vertices, indices) where:
    /// - vertices: Vec<[f32; 3]> with [x, y, z=loss] coordinates
    /// - indices: Vec<[u32; 3]> with triangle vertex indices
    pub fn generate_mesh(&self) -> (Vec<[f32; 3]>, Vec<[u32; 3]>) {
        if self.surface.is_empty() || self.surface[0].is_empty() {
            return (Vec::new(), Vec::new());
        }

        let height = self.surface.len();
        let width = self.surface[0].len();

        // Generate vertices
        let mut vertices = Vec::with_capacity(height * width);
        let x_step = (self.x_range.1 - self.x_range.0) / (width - 1) as f32;
        let y_step = (self.y_range.1 - self.y_range.0) / (height - 1) as f32;

        for j in 0..height {
            let y = self.y_range.0 + j as f32 * y_step;
            for i in 0..width {
                let x = self.x_range.0 + i as f32 * x_step;
                let z = self.surface[j][i];
                vertices.push([x, y, z]);
            }
        }

        // Generate triangle indices
        let mut indices = Vec::with_capacity((height - 1) * (width - 1) * 2);
        for j in 0..(height - 1) {
            for i in 0..(width - 1) {
                let v00 = (j * width + i) as u32;
                let v10 = (j * width + i + 1) as u32;
                let v01 = ((j + 1) * width + i) as u32;
                let v11 = ((j + 1) * width + i + 1) as u32;

                // Two triangles per cell
                indices.push([v00, v10, v01]);
                indices.push([v10, v11, v01]);
            }
        }

        (vertices, indices)
    }

    /// Generate mesh with vertex normals.
    ///
    /// # Returns
    /// Tuple of (vertices, normals, indices)
    pub fn generate_mesh_with_normals(&self) -> (Vec<[f32; 3]>, Vec<[f32; 3]>, Vec<[u32; 3]>) {
        let (vertices, indices) = self.generate_mesh();

        if vertices.is_empty() {
            return (vertices, Vec::new(), indices);
        }

        let height = self.surface.len();
        let width = self.surface[0].len();

        // Compute normals using finite differences
        let mut normals = vec![[0.0f32; 3]; vertices.len()];
        let x_step = (self.x_range.1 - self.x_range.0) / (width - 1) as f32;
        let y_step = (self.y_range.1 - self.y_range.0) / (height - 1) as f32;

        for j in 0..height {
            for i in 0..width {
                let idx = j * width + i;

                // Gradient in x direction
                let dz_dx = if i == 0 {
                    (self.surface[j][i + 1] - self.surface[j][i]) / x_step
                } else if i == width - 1 {
                    (self.surface[j][i] - self.surface[j][i - 1]) / x_step
                } else {
                    (self.surface[j][i + 1] - self.surface[j][i - 1]) / (2.0 * x_step)
                };

                // Gradient in y direction
                let dz_dy = if j == 0 {
                    (self.surface[j + 1][i] - self.surface[j][i]) / y_step
                } else if j == height - 1 {
                    (self.surface[j][i] - self.surface[j - 1][i]) / y_step
                } else {
                    (self.surface[j + 1][i] - self.surface[j - 1][i]) / (2.0 * y_step)
                };

                // Normal = cross product of tangent vectors
                // T_x = (1, 0, dz/dx), T_y = (0, 1, dz/dy)
                // N = T_x x T_y = (-dz/dx, -dz/dy, 1)
                let mut n = [-dz_dx, -dz_dy, 1.0];
                let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
                n[0] /= len;
                n[1] /= len;
                n[2] /= len;

                normals[idx] = n;
            }
        }

        (vertices, normals, indices)
    }

    /// Generate a decimated (lower resolution) mesh.
    ///
    /// # Arguments
    /// * `factor` - Decimation factor (e.g., 2 = half resolution)
    ///
    /// # Returns
    /// Decimated mesh (vertices, indices)
    pub fn generate_decimated(&self, factor: usize) -> (Vec<[f32; 3]>, Vec<[u32; 3]>) {
        if self.surface.is_empty() || factor == 0 {
            return self.generate_mesh();
        }

        let height = self.surface.len();
        let width = self.surface[0].len();
        let new_height = (height + factor - 1) / factor;
        let new_width = (width + factor - 1) / factor;

        if new_height < 2 || new_width < 2 {
            return self.generate_mesh();
        }

        // Sample vertices at reduced resolution
        let mut vertices = Vec::with_capacity(new_height * new_width);
        let x_step = (self.x_range.1 - self.x_range.0) / (width - 1) as f32;
        let y_step = (self.y_range.1 - self.y_range.0) / (height - 1) as f32;

        for jj in 0..new_height {
            let j = (jj * factor).min(height - 1);
            let y = self.y_range.0 + j as f32 * y_step;

            for ii in 0..new_width {
                let i = (ii * factor).min(width - 1);
                let x = self.x_range.0 + i as f32 * x_step;
                let z = self.surface[j][i];
                vertices.push([x, y, z]);
            }
        }

        // Generate triangle indices
        let mut indices = Vec::with_capacity((new_height - 1) * (new_width - 1) * 2);
        for jj in 0..(new_height - 1) {
            for ii in 0..(new_width - 1) {
                let v00 = (jj * new_width + ii) as u32;
                let v10 = (jj * new_width + ii + 1) as u32;
                let v01 = ((jj + 1) * new_width + ii) as u32;
                let v11 = ((jj + 1) * new_width + ii + 1) as u32;

                indices.push([v00, v10, v01]);
                indices.push([v10, v11, v01]);
            }
        }

        (vertices, indices)
    }

    /// Generate wireframe lines instead of triangles.
    ///
    /// # Returns
    /// Vector of line segment endpoints
    pub fn generate_wireframe(&self) -> Vec<([f32; 3], [f32; 3])> {
        if self.surface.is_empty() || self.surface[0].is_empty() {
            return Vec::new();
        }

        let height = self.surface.len();
        let width = self.surface[0].len();
        let x_step = (self.x_range.1 - self.x_range.0) / (width - 1) as f32;
        let y_step = (self.y_range.1 - self.y_range.0) / (height - 1) as f32;

        let mut lines = Vec::new();

        // Horizontal lines
        for j in 0..height {
            let y = self.y_range.0 + j as f32 * y_step;
            for i in 0..(width - 1) {
                let x1 = self.x_range.0 + i as f32 * x_step;
                let x2 = self.x_range.0 + (i + 1) as f32 * x_step;
                let z1 = self.surface[j][i];
                let z2 = self.surface[j][i + 1];
                lines.push(([x1, y, z1], [x2, y, z2]));
            }
        }

        // Vertical lines
        for i in 0..width {
            let x = self.x_range.0 + i as f32 * x_step;
            for j in 0..(height - 1) {
                let y1 = self.y_range.0 + j as f32 * y_step;
                let y2 = self.y_range.0 + (j + 1) as f32 * y_step;
                let z1 = self.surface[j][i];
                let z2 = self.surface[j + 1][i];
                lines.push(([x, y1, z1], [x, y2, z2]));
            }
        }

        lines
    }

    /// Generate contour lines at specified z-levels.
    ///
    /// # Arguments
    /// * `levels` - Z-values at which to generate contours
    ///
    /// # Returns
    /// Vector of (level, line_segments) tuples
    pub fn generate_contours(&self, levels: &[f32]) -> Vec<(f32, Vec<([f32; 2], [f32; 2])>)> {
        if self.surface.is_empty() || self.surface[0].is_empty() {
            return Vec::new();
        }

        let height = self.surface.len();
        let width = self.surface[0].len();
        let x_step = (self.x_range.1 - self.x_range.0) / (width - 1) as f32;
        let y_step = (self.y_range.1 - self.y_range.0) / (height - 1) as f32;

        levels
            .iter()
            .map(|&level| {
                let mut segments = Vec::new();

                // March through each cell using marching squares
                for j in 0..(height - 1) {
                    for i in 0..(width - 1) {
                        let x0 = self.x_range.0 + i as f32 * x_step;
                        let x1 = x0 + x_step;
                        let y0 = self.y_range.0 + j as f32 * y_step;
                        let y1 = y0 + y_step;

                        let z00 = self.surface[j][i];
                        let z10 = self.surface[j][i + 1];
                        let z01 = self.surface[j + 1][i];
                        let z11 = self.surface[j + 1][i + 1];

                        // Marching squares case index
                        let case = ((z00 >= level) as u8)
                            | (((z10 >= level) as u8) << 1)
                            | (((z01 >= level) as u8) << 2)
                            | (((z11 >= level) as u8) << 3);

                        // Interpolation helper
                        let lerp = |p1: f32, p2: f32, v1: f32, v2: f32| -> f32 {
                            if (v2 - v1).abs() < 1e-10 {
                                (p1 + p2) / 2.0
                            } else {
                                p1 + (level - v1) / (v2 - v1) * (p2 - p1)
                            }
                        };

                        // Edge midpoints
                        let bottom = || [lerp(x0, x1, z00, z10), y0];
                        let top = || [lerp(x0, x1, z01, z11), y1];
                        let left = || [x0, lerp(y0, y1, z00, z01)];
                        let right = || [x1, lerp(y0, y1, z10, z11)];

                        match case {
                            0 | 15 => {} // No contour
                            1 | 14 => segments.push((left(), bottom())),
                            2 | 13 => segments.push((bottom(), right())),
                            3 | 12 => segments.push((left(), right())),
                            4 | 11 => segments.push((top(), left())),
                            5 | 10 => {
                                // Saddle point: two segments
                                segments.push((left(), bottom()));
                                segments.push((top(), right()));
                            }
                            6 | 9 => segments.push((bottom(), top())),
                            7 | 8 => segments.push((top(), right())),
                            _ => {}
                        }
                    }
                }

                (level, segments)
            })
            .collect()
    }

    /// Export mesh to OBJ format string.
    pub fn to_obj(&self) -> String {
        let (vertices, normals, indices) = self.generate_mesh_with_normals();
        let mut obj = String::new();

        obj.push_str("# Loss Landscape Mesh\n");
        obj.push_str(&format!("# Vertices: {}\n", vertices.len()));
        obj.push_str(&format!("# Triangles: {}\n\n", indices.len()));

        // Vertices
        for v in &vertices {
            obj.push_str(&format!("v {} {} {}\n", v[0], v[1], v[2]));
        }
        obj.push('\n');

        // Normals
        for n in &normals {
            obj.push_str(&format!("vn {} {} {}\n", n[0], n[1], n[2]));
        }
        obj.push('\n');

        // Faces (OBJ indices are 1-based)
        for idx in &indices {
            obj.push_str(&format!(
                "f {}//{} {}//{} {}//{}",
                idx[0] + 1,
                idx[0] + 1,
                idx[1] + 1,
                idx[1] + 1,
                idx[2] + 1,
                idx[2] + 1
            ));
            obj.push('\n');
        }

        obj
    }

    /// Export mesh to PLY format string.
    pub fn to_ply(&self) -> String {
        let (vertices, indices) = self.generate_mesh();
        let mut ply = String::new();

        // Header
        ply.push_str("ply\n");
        ply.push_str("format ascii 1.0\n");
        ply.push_str(&format!("element vertex {}\n", vertices.len()));
        ply.push_str("property float x\n");
        ply.push_str("property float y\n");
        ply.push_str("property float z\n");
        ply.push_str(&format!("element face {}\n", indices.len()));
        ply.push_str("property list uchar int vertex_indices\n");
        ply.push_str("end_header\n");

        // Vertices
        for v in &vertices {
            ply.push_str(&format!("{} {} {}\n", v[0], v[1], v[2]));
        }

        // Faces
        for idx in &indices {
            ply.push_str(&format!("3 {} {} {}\n", idx[0], idx[1], idx[2]));
        }

        ply
    }
}

/// Color map for visualizing loss values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorMap {
    /// Jet colormap (blue to red)
    Jet,
    /// Viridis colormap (purple to yellow)
    Viridis,
    /// Plasma colormap (purple to yellow-orange)
    Plasma,
    /// Grayscale
    Grayscale,
}

impl ColorMap {
    /// Map a normalized value [0, 1] to RGB color.
    pub fn map(&self, t: f32) -> [f32; 3] {
        let t = t.clamp(0.0, 1.0);

        match self {
            ColorMap::Jet => {
                let r = (1.5 - 4.0 * (t - 0.75).abs()).clamp(0.0, 1.0);
                let g = (1.5 - 4.0 * (t - 0.5).abs()).clamp(0.0, 1.0);
                let b = (1.5 - 4.0 * (t - 0.25).abs()).clamp(0.0, 1.0);
                [r, g, b]
            }
            ColorMap::Viridis => {
                // Simplified viridis approximation
                let r = (0.267 + 0.329 * t + 1.101 * t * t - 1.090 * t * t * t).clamp(0.0, 1.0);
                let g = (0.005 + 1.404 * t - 0.214 * t * t).clamp(0.0, 1.0);
                let b = (0.329 + 1.421 * t - 1.601 * t * t + 0.513 * t * t * t).clamp(0.0, 1.0);
                [r, g, b]
            }
            ColorMap::Plasma => {
                // Simplified plasma approximation
                let r = (0.050 + 2.460 * t - 1.557 * t * t).clamp(0.0, 1.0);
                let g = (0.030 + 0.171 * t + 1.378 * t * t - 1.076 * t * t * t).clamp(0.0, 1.0);
                let b = (0.533 + 0.671 * t - 1.950 * t * t + 0.965 * t * t * t).clamp(0.0, 1.0);
                [r, g, b]
            }
            ColorMap::Grayscale => [t, t, t],
        }
    }
}

impl<'a> MeshExporter<'a> {
    /// Generate mesh with vertex colors based on loss values.
    ///
    /// # Arguments
    /// * `colormap` - Color mapping to use
    ///
    /// # Returns
    /// Tuple of (vertices, colors, indices)
    pub fn generate_mesh_with_colors(
        &self,
        colormap: ColorMap,
    ) -> (Vec<[f32; 3]>, Vec<[f32; 3]>, Vec<[u32; 3]>) {
        let (vertices, indices) = self.generate_mesh();

        if vertices.is_empty() {
            return (vertices, Vec::new(), indices);
        }

        // Find min/max for normalization
        let min_z = vertices.iter().map(|v| v[2]).fold(f32::INFINITY, f32::min);
        let max_z = vertices
            .iter()
            .map(|v| v[2])
            .fold(f32::NEG_INFINITY, f32::max);
        let range = (max_z - min_z).max(1e-6);

        // Generate colors
        let colors: Vec<[f32; 3]> = vertices
            .iter()
            .map(|v| {
                let t = (v[2] - min_z) / range;
                colormap.map(t)
            })
            .collect();

        (vertices, colors, indices)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_surface() -> Vec<Vec<f32>> {
        // 4x4 grid with quadratic bowl shape
        vec![
            vec![2.0, 1.0, 1.0, 2.0],
            vec![1.0, 0.5, 0.5, 1.0],
            vec![1.0, 0.5, 0.5, 1.0],
            vec![2.0, 1.0, 1.0, 2.0],
        ]
    }

    #[test]
    fn test_mesh_generation() {
        let surface = sample_surface();
        let exporter = MeshExporter::new(&surface, (-1.0, 1.0), (-1.0, 1.0));

        let (vertices, indices) = exporter.generate_mesh();

        assert_eq!(vertices.len(), 16); // 4x4 grid
        assert_eq!(indices.len(), 18); // 3x3 cells * 2 triangles
    }

    #[test]
    fn test_mesh_with_normals() {
        let surface = sample_surface();
        let exporter = MeshExporter::new(&surface, (-1.0, 1.0), (-1.0, 1.0));

        let (vertices, normals, _) = exporter.generate_mesh_with_normals();

        assert_eq!(vertices.len(), normals.len());

        // All normals should be unit length
        for n in &normals {
            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            assert!((len - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_decimated_mesh() {
        let surface = sample_surface();
        let exporter = MeshExporter::new(&surface, (-1.0, 1.0), (-1.0, 1.0));

        let (vertices, _) = exporter.generate_decimated(2);

        assert_eq!(vertices.len(), 4); // 2x2 grid after decimation
    }

    #[test]
    fn test_wireframe() {
        let surface = sample_surface();
        let exporter = MeshExporter::new(&surface, (-1.0, 1.0), (-1.0, 1.0));

        let lines = exporter.generate_wireframe();

        // 4 rows * 3 horizontal + 4 cols * 3 vertical = 24
        assert_eq!(lines.len(), 24);
    }

    #[test]
    fn test_contours() {
        let surface = sample_surface();
        let exporter = MeshExporter::new(&surface, (-1.0, 1.0), (-1.0, 1.0));

        let contours = exporter.generate_contours(&[0.75, 1.5]);

        assert_eq!(contours.len(), 2);
        // Each level should have some segments
        for (_, segments) in &contours {
            assert!(!segments.is_empty());
        }
    }

    #[test]
    fn test_obj_export() {
        let surface = sample_surface();
        let exporter = MeshExporter::new(&surface, (-1.0, 1.0), (-1.0, 1.0));

        let obj = exporter.to_obj();

        assert!(obj.contains("# Loss Landscape Mesh"));
        assert!(obj.contains("v "));
        assert!(obj.contains("vn "));
        assert!(obj.contains("f "));
    }

    #[test]
    fn test_ply_export() {
        let surface = sample_surface();
        let exporter = MeshExporter::new(&surface, (-1.0, 1.0), (-1.0, 1.0));

        let ply = exporter.to_ply();

        assert!(ply.contains("ply"));
        assert!(ply.contains("format ascii"));
        assert!(ply.contains("element vertex"));
        assert!(ply.contains("element face"));
    }

    #[test]
    fn test_colormap_jet() {
        let cmap = ColorMap::Jet;

        let blue = cmap.map(0.0);
        let red = cmap.map(1.0);

        // Blue at t=0
        assert!(blue[2] > blue[0]);
        // Red at t=1
        assert!(red[0] > red[2]);
    }

    #[test]
    fn test_mesh_with_colors() {
        let surface = sample_surface();
        let exporter = MeshExporter::new(&surface, (-1.0, 1.0), (-1.0, 1.0));

        let (vertices, colors, _) = exporter.generate_mesh_with_colors(ColorMap::Viridis);

        assert_eq!(vertices.len(), colors.len());

        // All colors should be in [0, 1]
        for c in &colors {
            assert!(c[0] >= 0.0 && c[0] <= 1.0);
            assert!(c[1] >= 0.0 && c[1] <= 1.0);
            assert!(c[2] >= 0.0 && c[2] <= 1.0);
        }
    }

    #[test]
    fn test_empty_surface() {
        let surface: Vec<Vec<f32>> = vec![];
        let exporter = MeshExporter::new(&surface, (-1.0, 1.0), (-1.0, 1.0));

        let (vertices, indices) = exporter.generate_mesh();
        assert!(vertices.is_empty());
        assert!(indices.is_empty());
    }
}
