//! Optimization trajectory tracking through the loss landscape.
//!
//! Tracks the path of optimization through parameter space and projects
//! it onto the 2D loss landscape for visualization.

use std::collections::VecDeque;

/// A point on the optimization trajectory.
#[derive(Debug, Clone, Copy)]
pub struct TrajectoryPoint {
    /// X coordinate in landscape space
    pub x: f32,
    /// Y coordinate in landscape space
    pub y: f32,
    /// Loss value at this point
    pub loss: f32,
    /// Training step number
    pub step: usize,
    /// Learning rate at this step
    pub learning_rate: f32,
    /// Gradient norm at this step
    pub gradient_norm: f32,
}

impl TrajectoryPoint {
    /// Create a new trajectory point with minimal information.
    pub fn new(x: f32, y: f32, loss: f32, step: usize) -> Self {
        Self {
            x,
            y,
            loss,
            step,
            learning_rate: 0.0,
            gradient_norm: 0.0,
        }
    }

    /// Create a trajectory point with full information.
    pub fn with_metadata(
        x: f32,
        y: f32,
        loss: f32,
        step: usize,
        learning_rate: f32,
        gradient_norm: f32,
    ) -> Self {
        Self {
            x,
            y,
            loss,
            step,
            learning_rate,
            gradient_norm,
        }
    }

    /// Convert to a 3D coordinate [x, y, loss].
    pub fn to_3d(&self) -> [f32; 3] {
        [self.x, self.y, self.loss]
    }
}

/// Tracks the optimization trajectory through the loss landscape.
///
/// Maintains a history of weight vectors and projects them onto
/// the 2D landscape defined by two direction vectors.
#[derive(Debug, Clone)]
pub struct TrajectoryTracker {
    /// Recorded trajectory points
    points: Vec<TrajectoryPoint>,
    /// Center weights (origin of the landscape)
    center_weights: Vec<f32>,
    /// First perturbation direction
    direction1: Vec<f32>,
    /// Second perturbation direction
    direction2: Vec<f32>,
    /// Maximum number of points to keep (0 = unlimited)
    max_points: usize,
    /// Optional sliding window for recent points only
    window: Option<VecDeque<TrajectoryPoint>>,
    /// Window size for sliding window mode
    window_size: usize,
}

impl TrajectoryTracker {
    /// Create a new trajectory tracker.
    ///
    /// # Arguments
    /// * `center_weights` - Weight vector at the center of the landscape
    /// * `direction1` - First perturbation direction (unit vector)
    /// * `direction2` - Second perturbation direction (unit vector)
    pub fn new(center_weights: Vec<f32>, direction1: Vec<f32>, direction2: Vec<f32>) -> Self {
        Self {
            points: Vec::new(),
            center_weights,
            direction1,
            direction2,
            max_points: 0,
            window: None,
            window_size: 0,
        }
    }

    /// Set a maximum number of points to keep.
    pub fn with_max_points(mut self, max: usize) -> Self {
        self.max_points = max;
        self
    }

    /// Enable sliding window mode for animation.
    ///
    /// Only keeps the most recent `size` points, useful for
    /// animating training progress.
    pub fn with_sliding_window(mut self, size: usize) -> Self {
        self.window_size = size;
        self.window = Some(VecDeque::with_capacity(size));
        self
    }

    /// Record a new point on the trajectory.
    ///
    /// # Arguments
    /// * `weights` - Current weight vector
    /// * `loss` - Current loss value
    /// * `step` - Current training step
    pub fn record(&mut self, weights: &[f32], loss: f32, step: usize) {
        self.record_with_metadata(weights, loss, step, 0.0, 0.0);
    }

    /// Record a new point with additional metadata.
    ///
    /// # Arguments
    /// * `weights` - Current weight vector
    /// * `loss` - Current loss value
    /// * `step` - Current training step
    /// * `learning_rate` - Learning rate at this step
    /// * `gradient_norm` - Gradient norm at this step
    pub fn record_with_metadata(
        &mut self,
        weights: &[f32],
        loss: f32,
        step: usize,
        learning_rate: f32,
        gradient_norm: f32,
    ) {
        let (x, y) = self.project_to_plane(weights);
        let point = TrajectoryPoint::with_metadata(x, y, loss, step, learning_rate, gradient_norm);

        if let Some(ref mut window) = self.window {
            if window.len() >= self.window_size {
                window.pop_front();
            }
            window.push_back(point);
        }

        self.points.push(point);

        // Enforce max_points limit
        if self.max_points > 0 && self.points.len() > self.max_points {
            self.points.remove(0);
        }
    }

    /// Get all recorded trajectory points.
    pub fn points(&self) -> &[TrajectoryPoint] {
        &self.points
    }

    /// Get points in the sliding window (for animation).
    pub fn window_points(&self) -> Vec<TrajectoryPoint> {
        if let Some(ref window) = self.window {
            window.iter().copied().collect()
        } else {
            self.points.clone()
        }
    }

    /// Get the trajectory as 3D coordinates.
    pub fn to_3d_path(&self) -> Vec<[f32; 3]> {
        self.points.iter().map(|p| p.to_3d()).collect()
    }

    /// Get the most recent point.
    pub fn current(&self) -> Option<&TrajectoryPoint> {
        self.points.last()
    }

    /// Get the total path length in the landscape.
    pub fn path_length(&self) -> f32 {
        if self.points.len() < 2 {
            return 0.0;
        }

        self.points
            .windows(2)
            .map(|w| {
                let dx = w[1].x - w[0].x;
                let dy = w[1].y - w[0].y;
                (dx * dx + dy * dy).sqrt()
            })
            .sum()
    }

    /// Get the net displacement from start to current position.
    pub fn net_displacement(&self) -> f32 {
        if self.points.len() < 2 {
            return 0.0;
        }

        let first = &self.points[0];
        let last = self.points.last().unwrap();
        let dx = last.x - first.x;
        let dy = last.y - first.y;
        (dx * dx + dy * dy).sqrt()
    }

    /// Compute the efficiency ratio (net displacement / path length).
    ///
    /// Values close to 1.0 indicate direct optimization paths,
    /// lower values indicate wandering or oscillation.
    pub fn efficiency(&self) -> f32 {
        let length = self.path_length();
        if length < 1e-10 {
            return 1.0;
        }
        self.net_displacement() / length
    }

    /// Get loss improvement over the trajectory.
    pub fn loss_improvement(&self) -> f32 {
        if self.points.len() < 2 {
            return 0.0;
        }
        self.points[0].loss - self.points.last().unwrap().loss
    }

    /// Compute velocity (change in position per step).
    pub fn velocity(&self) -> Option<[f32; 2]> {
        if self.points.len() < 2 {
            return None;
        }

        let p1 = &self.points[self.points.len() - 2];
        let p2 = self.points.last().unwrap();

        let steps = (p2.step - p1.step).max(1) as f32;
        Some([(p2.x - p1.x) / steps, (p2.y - p1.y) / steps])
    }

    /// Compute acceleration (change in velocity).
    pub fn acceleration(&self) -> Option<[f32; 2]> {
        if self.points.len() < 3 {
            return None;
        }

        let p0 = &self.points[self.points.len() - 3];
        let p1 = &self.points[self.points.len() - 2];
        let p2 = self.points.last().unwrap();

        let dt1 = (p1.step - p0.step).max(1) as f32;
        let dt2 = (p2.step - p1.step).max(1) as f32;

        let v1 = [(p1.x - p0.x) / dt1, (p1.y - p0.y) / dt1];
        let v2 = [(p2.x - p1.x) / dt2, (p2.y - p1.y) / dt2];

        let dt = (dt1 + dt2) / 2.0;
        Some([(v2[0] - v1[0]) / dt, (v2[1] - v1[1]) / dt])
    }

    /// Detect oscillation in the trajectory.
    ///
    /// Returns the number of direction reversals in the recent history.
    pub fn detect_oscillations(&self, lookback: usize) -> usize {
        let start = self.points.len().saturating_sub(lookback);
        if self.points.len() - start < 3 {
            return 0;
        }

        let mut reversals = 0;
        for i in (start + 2)..self.points.len() {
            let v1_x = self.points[i - 1].x - self.points[i - 2].x;
            let v1_y = self.points[i - 1].y - self.points[i - 2].y;
            let v2_x = self.points[i].x - self.points[i - 1].x;
            let v2_y = self.points[i].y - self.points[i - 1].y;

            // Dot product of consecutive velocity vectors
            let dot = v1_x * v2_x + v1_y * v2_y;
            if dot < 0.0 {
                reversals += 1;
            }
        }

        reversals
    }

    /// Clear all recorded points.
    pub fn clear(&mut self) {
        self.points.clear();
        if let Some(ref mut window) = self.window {
            window.clear();
        }
    }

    /// Get segment data for line rendering.
    ///
    /// Returns pairs of consecutive points for drawing line segments.
    pub fn segments(&self) -> Vec<([f32; 3], [f32; 3])> {
        self.points
            .windows(2)
            .map(|w| (w[0].to_3d(), w[1].to_3d()))
            .collect()
    }

    /// Get colored segments based on loss improvement.
    ///
    /// Returns tuples of (start, end, improvement_ratio) where
    /// improvement_ratio is in [-1, 1] (positive = loss decreased).
    pub fn colored_segments(&self) -> Vec<([f32; 3], [f32; 3], f32)> {
        if self.points.is_empty() {
            return Vec::new();
        }

        let max_loss = self
            .points
            .iter()
            .map(|p| p.loss)
            .fold(f32::NEG_INFINITY, f32::max);
        let min_loss = self
            .points
            .iter()
            .map(|p| p.loss)
            .fold(f32::INFINITY, f32::min);
        let loss_range = (max_loss - min_loss).max(1e-6);

        self.points
            .windows(2)
            .map(|w| {
                let improvement = (w[0].loss - w[1].loss) / loss_range;
                (w[0].to_3d(), w[1].to_3d(), improvement.clamp(-1.0, 1.0))
            })
            .collect()
    }

    // ========== Internal Methods ==========

    /// Project a weight vector onto the 2D landscape plane.
    fn project_to_plane(&self, weights: &[f32]) -> (f32, f32) {
        if weights.len() != self.center_weights.len() {
            return (0.0, 0.0);
        }

        // Compute delta from center
        let mut x = 0.0;
        let mut y = 0.0;

        for i in 0..weights.len() {
            let delta = weights[i] - self.center_weights[i];
            x += delta * self.direction1[i];
            y += delta * self.direction2[i];
        }

        (x, y)
    }
}

/// Statistics about trajectory behavior.
#[derive(Debug, Clone, Copy)]
pub struct TrajectoryStats {
    /// Total path length
    pub path_length: f32,
    /// Net displacement
    pub net_displacement: f32,
    /// Efficiency ratio (0-1)
    pub efficiency: f32,
    /// Total loss improvement
    pub loss_improvement: f32,
    /// Number of oscillations detected
    pub oscillation_count: usize,
    /// Average step size
    pub avg_step_size: f32,
}

impl TrajectoryTracker {
    /// Compute comprehensive trajectory statistics.
    pub fn stats(&self) -> TrajectoryStats {
        let path_length = self.path_length();
        let net_displacement = self.net_displacement();
        let efficiency = self.efficiency();
        let loss_improvement = self.loss_improvement();
        let oscillation_count = self.detect_oscillations(50);

        let avg_step_size = if self.points.len() > 1 {
            path_length / (self.points.len() - 1) as f32
        } else {
            0.0
        };

        TrajectoryStats {
            path_length,
            net_displacement,
            efficiency,
            loss_improvement,
            oscillation_count,
            avg_step_size,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_directions(dim: usize) -> (Vec<f32>, Vec<f32>) {
        let mut dir1 = vec![0.0; dim];
        let mut dir2 = vec![0.0; dim];
        dir1[0] = 1.0;
        if dim > 1 {
            dir2[1] = 1.0;
        }
        (dir1, dir2)
    }

    #[test]
    fn test_trajectory_creation() {
        let (dir1, dir2) = unit_directions(5);
        let tracker = TrajectoryTracker::new(vec![0.0; 5], dir1, dir2);
        assert!(tracker.points().is_empty());
    }

    #[test]
    fn test_record_points() {
        let (dir1, dir2) = unit_directions(5);
        let mut tracker = TrajectoryTracker::new(vec![0.0; 5], dir1, dir2);

        tracker.record(&[1.0, 0.0, 0.0, 0.0, 0.0], 1.0, 0);
        tracker.record(&[2.0, 0.0, 0.0, 0.0, 0.0], 0.5, 1);

        assert_eq!(tracker.points().len(), 2);
        assert_eq!(tracker.points()[0].x, 1.0);
        assert_eq!(tracker.points()[1].x, 2.0);
    }

    #[test]
    fn test_path_length() {
        let (dir1, dir2) = unit_directions(5);
        let mut tracker = TrajectoryTracker::new(vec![0.0; 5], dir1, dir2);

        tracker.record(&[0.0, 0.0, 0.0, 0.0, 0.0], 1.0, 0);
        tracker.record(&[1.0, 0.0, 0.0, 0.0, 0.0], 0.8, 1);
        tracker.record(&[1.0, 1.0, 0.0, 0.0, 0.0], 0.6, 2);

        let length = tracker.path_length();
        assert!((length - 2.0).abs() < 1e-5); // 1 + 1 = 2
    }

    #[test]
    fn test_efficiency() {
        let (dir1, dir2) = unit_directions(5);
        let mut tracker = TrajectoryTracker::new(vec![0.0; 5], dir1, dir2);

        // Direct path
        tracker.record(&[0.0, 0.0, 0.0, 0.0, 0.0], 1.0, 0);
        tracker.record(&[1.0, 0.0, 0.0, 0.0, 0.0], 0.5, 1);

        assert!((tracker.efficiency() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_oscillation_detection() {
        let (dir1, dir2) = unit_directions(5);
        let mut tracker = TrajectoryTracker::new(vec![0.0; 5], dir1, dir2);

        // Oscillating path
        for i in 0..10 {
            let x = if i % 2 == 0 { 0.0 } else { 1.0 };
            tracker.record(&[x, 0.0, 0.0, 0.0, 0.0], 1.0 - i as f32 * 0.1, i);
        }

        let oscillations = tracker.detect_oscillations(10);
        assert!(oscillations >= 4);
    }

    #[test]
    fn test_sliding_window() {
        let (dir1, dir2) = unit_directions(5);
        let mut tracker = TrajectoryTracker::new(vec![0.0; 5], dir1, dir2).with_sliding_window(3);

        for i in 0..10 {
            tracker.record(&[i as f32, 0.0, 0.0, 0.0, 0.0], 1.0, i);
        }

        let window_points = tracker.window_points();
        assert_eq!(window_points.len(), 3);
        assert_eq!(window_points[0].x, 7.0);
        assert_eq!(window_points[2].x, 9.0);
    }

    #[test]
    fn test_max_points() {
        let (dir1, dir2) = unit_directions(5);
        let mut tracker = TrajectoryTracker::new(vec![0.0; 5], dir1, dir2).with_max_points(5);

        for i in 0..10 {
            tracker.record(&[i as f32, 0.0, 0.0, 0.0, 0.0], 1.0, i);
        }

        assert_eq!(tracker.points().len(), 5);
    }

    #[test]
    fn test_velocity() {
        let (dir1, dir2) = unit_directions(5);
        let mut tracker = TrajectoryTracker::new(vec![0.0; 5], dir1, dir2);

        tracker.record(&[0.0, 0.0, 0.0, 0.0, 0.0], 1.0, 0);
        tracker.record(&[2.0, 0.0, 0.0, 0.0, 0.0], 0.5, 1);

        let vel = tracker.velocity().unwrap();
        assert!((vel[0] - 2.0).abs() < 1e-5);
        assert!(vel[1].abs() < 1e-5);
    }

    #[test]
    fn test_colored_segments() {
        let (dir1, dir2) = unit_directions(5);
        let mut tracker = TrajectoryTracker::new(vec![0.0; 5], dir1, dir2);

        tracker.record(&[0.0, 0.0, 0.0, 0.0, 0.0], 1.0, 0);
        tracker.record(&[1.0, 0.0, 0.0, 0.0, 0.0], 0.5, 1);

        let segments = tracker.colored_segments();
        assert_eq!(segments.len(), 1);
        assert!(segments[0].2 > 0.0); // Loss decreased, positive improvement
    }

    #[test]
    fn test_stats() {
        let (dir1, dir2) = unit_directions(5);
        let mut tracker = TrajectoryTracker::new(vec![0.0; 5], dir1, dir2);

        for i in 0..10 {
            tracker.record(&[i as f32, 0.0, 0.0, 0.0, 0.0], 1.0 - i as f32 * 0.1, i);
        }

        let stats = tracker.stats();
        assert!(stats.path_length > 0.0);
        assert!(stats.loss_improvement > 0.0);
        assert!(stats.efficiency > 0.9);
    }
}
