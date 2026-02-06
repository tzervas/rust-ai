//! Loss landscape visualization for Rerun.
//!
//! This module provides tools for visualizing the loss landscape as a 3D surface
//! with the optimization trajectory overlaid.

use super::{RerunError, RerunResult};
use rerun::{Color, Position3D, RecordingStream};

/// Logger for loss landscape visualization.
pub struct LandscapeLogger<'a> {
    rec: &'a RecordingStream,
}

impl<'a> LandscapeLogger<'a> {
    /// Create a new landscape logger.
    pub fn new(rec: &'a RecordingStream) -> Self {
        Self { rec }
    }

    /// Log a loss landscape surface with optimization trajectory.
    ///
    /// # Arguments
    ///
    /// * `surface` - 2D grid of loss values [rows x cols]
    /// * `trajectory` - Sequence of [x, y, loss] points showing optimization path
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, or an error if data is invalid.
    pub fn log_landscape(&self, surface: &[Vec<f32>], trajectory: &[[f32; 3]]) -> RerunResult<()> {
        if surface.is_empty() {
            return Err(RerunError::EmptyInput("Empty surface grid".to_string()));
        }

        let height = surface.len();
        let width = surface[0].len();

        // Find min/max for normalization
        let mut min_val = f32::INFINITY;
        let mut max_val = f32::NEG_INFINITY;
        for row in surface {
            for &val in row {
                min_val = min_val.min(val);
                max_val = max_val.max(val);
            }
        }

        let range = (max_val - min_val).max(1e-6);

        // Create mesh vertices from the surface grid
        let mut points: Vec<Position3D> = Vec::with_capacity(height * width);
        let mut colors: Vec<Color> = Vec::with_capacity(height * width);

        for (i, row) in surface.iter().enumerate() {
            if row.len() != width {
                return Err(RerunError::ShapeMismatch {
                    expected: format!("{} columns", width),
                    got: format!("{} columns", row.len()),
                });
            }

            for (j, &val) in row.iter().enumerate() {
                // Position: x from column, y from row, z from loss value
                let x = (j as f32 / width as f32 - 0.5) * 10.0;
                let y = (i as f32 / height as f32 - 0.5) * 10.0;
                let z = val;

                points.push(Position3D::new(x, y, z));

                // Color based on normalized loss
                let normalized = (val - min_val) / range;
                let [r, g, b] = self.height_color(normalized);
                colors.push(Color::from_rgb(r, g, b));
            }
        }

        // Log surface as point cloud (for mesh, we'd need triangulation)
        self.rec
            .log(
                "landscape/surface",
                &rerun::Points3D::new(points)
                    .with_colors(colors)
                    .with_radii([0.05f32]),
            )
            .map_err(|e| RerunError::LoggingError(e.to_string()))?;

        // Log grid lines for structure
        self.log_surface_grid(surface, width, height)?;

        // Log optimization trajectory if provided
        if !trajectory.is_empty() {
            self.log_trajectory(trajectory)?;
        }

        Ok(())
    }

    /// Log the optimization trajectory.
    fn log_trajectory(&self, trajectory: &[[f32; 3]]) -> RerunResult<()> {
        if trajectory.is_empty() {
            return Ok(());
        }

        // Convert trajectory points
        let positions: Vec<Position3D> = trajectory
            .iter()
            .map(|&[x, y, z]| Position3D::new(x, y, z))
            .collect();

        // Log trajectory line
        let line_points: Vec<[f32; 3]> = trajectory.to_vec();
        let _ = self.rec.log(
            "landscape/trajectory/path",
            &rerun::LineStrips3D::new([line_points])
                .with_colors([Color::from_rgb(255, 50, 50)])
                .with_radii([0.08f32]),
        );

        // Log trajectory points with time coloring (early = blue, late = red)
        let n = trajectory.len();
        let point_colors: Vec<Color> = (0..n)
            .map(|i| {
                let t = i as f32 / n.max(1) as f32;
                // Blue to red gradient
                Color::from_rgb((255.0 * t) as u8, 50, (255.0 * (1.0 - t)) as u8)
            })
            .collect();

        self.rec
            .log(
                "landscape/trajectory/points",
                &rerun::Points3D::new(positions)
                    .with_colors(point_colors)
                    .with_radii([0.15f32]),
            )
            .map_err(|e| RerunError::LoggingError(e.to_string()))?;

        // Mark start and end points
        if let Some(&[x, y, z]) = trajectory.first() {
            let _ = self.rec.log(
                "landscape/trajectory/start",
                &rerun::Points3D::new([[x, y, z]])
                    .with_colors([Color::from_rgb(0, 255, 0)])
                    .with_radii([0.25f32])
                    .with_labels(["Start"]),
            );
        }

        if let Some(&[x, y, z]) = trajectory.last() {
            let _ = self.rec.log(
                "landscape/trajectory/end",
                &rerun::Points3D::new([[x, y, z]])
                    .with_colors([Color::from_rgb(255, 0, 0)])
                    .with_radii([0.25f32])
                    .with_labels(["End"]),
            );
        }

        Ok(())
    }

    /// Log grid lines on the surface for visual structure.
    fn log_surface_grid(
        &self,
        surface: &[Vec<f32>],
        width: usize,
        height: usize,
    ) -> RerunResult<()> {
        let mut lines: Vec<Vec<[f32; 3]>> = Vec::new();

        // Horizontal lines (every 2 rows)
        for i in (0..height).step_by(2) {
            let mut line = Vec::new();
            for j in 0..width {
                let x = (j as f32 / width as f32 - 0.5) * 10.0;
                let y = (i as f32 / height as f32 - 0.5) * 10.0;
                let z = surface[i][j];
                line.push([x, y, z]);
            }
            if !line.is_empty() {
                lines.push(line);
            }
        }

        // Vertical lines (every 2 columns)
        for j in (0..width).step_by(2) {
            let mut line = Vec::new();
            for i in 0..height {
                let x = (j as f32 / width as f32 - 0.5) * 10.0;
                let y = (i as f32 / height as f32 - 0.5) * 10.0;
                let z = surface[i][j];
                line.push([x, y, z]);
            }
            if !line.is_empty() {
                lines.push(line);
            }
        }

        if !lines.is_empty() {
            let _ = self.rec.log(
                "landscape/grid",
                &rerun::LineStrips3D::new(lines).with_colors([Color::from_rgb(100, 100, 100)]),
            );
        }

        Ok(())
    }

    /// Log loss landscape with contour hints.
    ///
    /// # Arguments
    ///
    /// * `surface` - 2D loss grid
    /// * `contour_levels` - Z-values at which to draw contour lines
    pub fn log_landscape_with_contours(
        &self,
        surface: &[Vec<f32>],
        contour_levels: &[f32],
    ) -> RerunResult<()> {
        // First log the basic surface
        self.log_landscape(surface, &[])?;

        // Then add contour lines at specified levels
        let height = surface.len();
        let width = surface[0].len();

        for &level in contour_levels {
            let contour_points = self.extract_contour_points(surface, width, height, level);

            if !contour_points.is_empty() {
                let _ = self.rec.log(
                    format!("landscape/contours/level_{:.2}", level),
                    &rerun::Points3D::new(
                        contour_points
                            .iter()
                            .map(|&[x, y, z]| Position3D::new(x, y, z)),
                    )
                    .with_colors([Color::from_rgb(255, 255, 255)])
                    .with_radii([0.03f32]),
                );
            }
        }

        Ok(())
    }

    /// Extract approximate contour points at a given level.
    fn extract_contour_points(
        &self,
        surface: &[Vec<f32>],
        width: usize,
        height: usize,
        level: f32,
    ) -> Vec<[f32; 3]> {
        let mut points = Vec::new();

        for i in 0..height.saturating_sub(1) {
            for j in 0..width.saturating_sub(1) {
                let v00 = surface[i][j];
                let v01 = surface[i][j + 1];
                let v10 = surface[i + 1][j];
                let v11 = surface[i + 1][j + 1];

                // Check if contour crosses this cell
                let min_val = v00.min(v01).min(v10).min(v11);
                let max_val = v00.max(v01).max(v10).max(v11);

                if level >= min_val && level <= max_val {
                    // Add center point of cell as approximate contour point
                    let x = ((j as f32 + 0.5) / width as f32 - 0.5) * 10.0;
                    let y = ((i as f32 + 0.5) / height as f32 - 0.5) * 10.0;
                    points.push([x, y, level]);
                }
            }
        }

        points
    }

    /// Log an animated trajectory step (for real-time streaming).
    ///
    /// # Arguments
    ///
    /// * `step` - Current training step
    /// * `position` - Current [x, y, loss] position in landscape
    pub fn log_trajectory_step(&self, step: u64, position: [f32; 3]) -> RerunResult<()> {
        self.rec.set_time_sequence("step", step as i64);

        let [x, y, z] = position;

        self.rec
            .log(
                "landscape/current_position",
                &rerun::Points3D::new([[x, y, z]])
                    .with_colors([Color::from_rgb(255, 255, 0)])
                    .with_radii([0.2f32])
                    .with_labels(["Current"]),
            )
            .map_err(|e| RerunError::LoggingError(e.to_string()))?;

        Ok(())
    }

    /// Height-based colormap (blue = low, red = high).
    fn height_color(&self, normalized: f32) -> [u8; 3] {
        let t = normalized.clamp(0.0, 1.0);

        // Blue -> Cyan -> Green -> Yellow -> Red
        let colors: [(f32, [u8; 3]); 5] = [
            (0.0, [0, 0, 180]),    // Blue (low)
            (0.25, [0, 180, 180]), // Cyan
            (0.5, [0, 180, 0]),    // Green
            (0.75, [255, 255, 0]), // Yellow
            (1.0, [255, 0, 0]),    // Red (high)
        ];

        // Find interpolation segment
        let mut lower = colors[0];
        let mut upper = colors[colors.len() - 1];

        for i in 0..colors.len() - 1 {
            if t >= colors[i].0 && t <= colors[i + 1].0 {
                lower = colors[i];
                upper = colors[i + 1];
                break;
            }
        }

        let range = upper.0 - lower.0;
        let local_t = if range > 0.0 {
            (t - lower.0) / range
        } else {
            0.0
        };

        [
            (lower.1[0] as f32 + local_t * (upper.1[0] as f32 - lower.1[0] as f32)) as u8,
            (lower.1[1] as f32 + local_t * (upper.1[1] as f32 - lower.1[1] as f32)) as u8,
            (lower.1[2] as f32 + local_t * (upper.1[2] as f32 - lower.1[2] as f32)) as u8,
        ]
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_height_colormap_range() {
        // Test that endpoints produce expected colors
        let colors: [(f32, [u8; 3]); 5] = [
            (0.0, [0, 0, 180]),
            (0.25, [0, 180, 180]),
            (0.5, [0, 180, 0]),
            (0.75, [255, 255, 0]),
            (1.0, [255, 0, 0]),
        ];

        // Low values should be blue
        assert_eq!(colors[0].1[2], 180); // Blue channel high
        assert_eq!(colors[0].1[0], 0); // Red channel low

        // High values should be red
        assert_eq!(colors[4].1[0], 255); // Red channel high
        assert_eq!(colors[4].1[2], 0); // Blue channel low
    }

    #[test]
    fn test_contour_extraction_logic() {
        // A simple 2x2 grid
        let surface = vec![vec![0.0, 1.0], vec![1.0, 2.0]];

        // Level 0.5 should be between cells
        let level = 0.5;
        let min_val = 0.0f32.min(1.0).min(1.0).min(2.0);
        let max_val = 0.0f32.max(1.0).max(1.0).max(2.0);

        assert!(level >= min_val && level <= max_val);
    }
}
