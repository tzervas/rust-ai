//! 3D Layout algorithms for token positioning

/// Configuration for layout algorithms
#[derive(Debug, Clone)]
pub struct LayoutConfig {
    /// Space dimensions [width, height, depth]
    pub dimensions: [f32; 3],
    /// Repulsion strength between nodes
    pub repulsion: f32,
    /// Attraction strength for connected nodes
    pub attraction: f32,
    /// Damping factor for force simulation
    pub damping: f32,
    /// Maximum iterations for force-directed layout
    pub max_iterations: usize,
    /// Minimum energy threshold for convergence
    pub convergence_threshold: f32,
    /// Whether to use embeddings for initial positioning
    pub use_embeddings: bool,
}

impl Default for LayoutConfig {
    fn default() -> Self {
        Self {
            dimensions: [100.0, 100.0, 100.0],
            repulsion: 1000.0,
            attraction: 0.01,
            damping: 0.9,
            max_iterations: 500,
            convergence_threshold: 0.01,
            use_embeddings: true,
        }
    }
}

/// Trait for 3D layout algorithms
pub trait Layout3D {
    /// Compute 3D positions for tokens
    ///
    /// # Arguments
    /// * `embeddings` - Optional embedding vectors for each token
    /// * `frequencies` - Frequency/importance weight for each token
    ///
    /// # Returns
    /// A vector of 3D positions [x, y, z] for each token
    fn compute(&self, embeddings: &[Option<&Vec<f32>>], frequencies: &[f32]) -> Vec<[f32; 3]>;
}

/// Force-directed layout using Fruchterman-Reingold algorithm
#[derive(Debug, Clone)]
pub struct ForceDirectedLayout {
    config: LayoutConfig,
}

impl ForceDirectedLayout {
    /// Create a new force-directed layout
    pub fn new(config: LayoutConfig) -> Self {
        Self { config }
    }

    /// Project high-dimensional embeddings to 3D using PCA-like projection
    fn project_to_3d(&self, embeddings: &[Option<&Vec<f32>>], n: usize) -> Vec<[f32; 3]> {
        // If embeddings available, use first 3 principal components (simplified)
        let mut positions = Vec::with_capacity(n);

        for (i, emb) in embeddings.iter().enumerate() {
            if let Some(e) = emb {
                if e.len() >= 3 {
                    // Use first 3 dimensions directly (simplified PCA)
                    positions.push([e[0], e[1], e[2]]);
                } else {
                    // Pad with zeros
                    let mut pos = [0.0f32; 3];
                    for (j, &v) in e.iter().take(3).enumerate() {
                        pos[j] = v;
                    }
                    positions.push(pos);
                }
            } else {
                // No embedding: use deterministic position based on index
                let angle1 = (i as f32 * 2.399) % std::f32::consts::TAU; // Golden angle
                let angle2 = (i as f32 * 1.618) % std::f32::consts::PI;
                let r = (i as f32 / n as f32).sqrt() * 50.0;

                positions.push([
                    r * angle2.sin() * angle1.cos(),
                    r * angle2.sin() * angle1.sin(),
                    r * angle2.cos(),
                ]);
            }
        }

        // Normalize to fit in layout dimensions
        self.normalize_positions(&mut positions);
        positions
    }

    /// Normalize positions to fit within configured dimensions
    fn normalize_positions(&self, positions: &mut [[f32; 3]]) {
        if positions.is_empty() {
            return;
        }

        // Find bounding box
        let mut min = [f32::MAX; 3];
        let mut max = [f32::MIN; 3];

        for pos in positions.iter() {
            for i in 0..3 {
                min[i] = min[i].min(pos[i]);
                max[i] = max[i].max(pos[i]);
            }
        }

        // Scale to fit dimensions
        let range = [
            (max[0] - min[0]).max(1.0),
            (max[1] - min[1]).max(1.0),
            (max[2] - min[2]).max(1.0),
        ];

        let scale = [
            self.config.dimensions[0] / range[0],
            self.config.dimensions[1] / range[1],
            self.config.dimensions[2] / range[2],
        ];

        // Use smallest scale to maintain aspect ratio
        let uniform_scale = scale[0].min(scale[1]).min(scale[2]) * 0.8;

        for pos in positions.iter_mut() {
            for i in 0..3 {
                pos[i] = (pos[i] - min[i] - range[i] / 2.0) * uniform_scale
                    + self.config.dimensions[i] / 2.0;
            }
        }
    }
}

impl Layout3D for ForceDirectedLayout {
    fn compute(&self, embeddings: &[Option<&Vec<f32>>], frequencies: &[f32]) -> Vec<[f32; 3]> {
        let n = embeddings.len();
        if n == 0 {
            return Vec::new();
        }

        // Initialize positions
        let mut positions = if self.config.use_embeddings {
            self.project_to_3d(embeddings, n)
        } else {
            // Random-ish initial positions using golden angle
            (0..n)
                .map(|i| {
                    let t = i as f32 / n as f32;
                    let angle1 = t * std::f32::consts::TAU * 6.0;
                    let angle2 = t * std::f32::consts::PI;
                    let r = t.sqrt() * 50.0;
                    [
                        r * angle2.sin() * angle1.cos(),
                        r * angle2.sin() * angle1.sin(),
                        r * angle2.cos(),
                    ]
                })
                .collect()
        };

        // Velocity for each node
        let mut velocities = vec![[0.0f32; 3]; n];

        // Optimal distance between nodes
        let volume = self.config.dimensions.iter().product::<f32>();
        let k = (volume / n as f32).powf(1.0 / 3.0);

        // Temperature for simulated annealing
        let mut temperature = self.config.dimensions[0] / 10.0;
        let cooling_rate = temperature / self.config.max_iterations as f32;

        for _iteration in 0..self.config.max_iterations {
            let mut forces = vec![[0.0f32; 3]; n];

            // Calculate repulsive forces between all pairs
            for i in 0..n {
                for j in (i + 1)..n {
                    let dx = positions[i][0] - positions[j][0];
                    let dy = positions[i][1] - positions[j][1];
                    let dz = positions[i][2] - positions[j][2];

                    let dist = (dx * dx + dy * dy + dz * dz).sqrt().max(0.01);

                    // Repulsion: F = k^2 / d
                    let repulsion = self.config.repulsion * k * k / (dist * dist);

                    // Weight by inverse frequency (less frequent = more repulsion)
                    let weight_i = 1.0 / (1.0 + frequencies[i] / 1000.0);
                    let weight_j = 1.0 / (1.0 + frequencies[j] / 1000.0);

                    let fx = dx / dist * repulsion;
                    let fy = dy / dist * repulsion;
                    let fz = dz / dist * repulsion;

                    forces[i][0] += fx * weight_i;
                    forces[i][1] += fy * weight_i;
                    forces[i][2] += fz * weight_i;

                    forces[j][0] -= fx * weight_j;
                    forces[j][1] -= fy * weight_j;
                    forces[j][2] -= fz * weight_j;
                }
            }

            // Calculate attractive forces based on embedding similarity
            if self.config.use_embeddings {
                for i in 0..n {
                    for j in (i + 1)..n {
                        if let (Some(emb_i), Some(emb_j)) = (&embeddings[i], &embeddings[j]) {
                            // Cosine similarity
                            let similarity = cosine_similarity(emb_i, emb_j);

                            if similarity > 0.5 {
                                // Attract similar tokens
                                let dx = positions[j][0] - positions[i][0];
                                let dy = positions[j][1] - positions[i][1];
                                let dz = positions[j][2] - positions[i][2];

                                let dist = (dx * dx + dy * dy + dz * dz).sqrt().max(0.01);
                                let attraction =
                                    self.config.attraction * dist * dist / k * similarity;

                                let fx = dx / dist * attraction;
                                let fy = dy / dist * attraction;
                                let fz = dz / dist * attraction;

                                forces[i][0] += fx;
                                forces[i][1] += fy;
                                forces[i][2] += fz;

                                forces[j][0] -= fx;
                                forces[j][1] -= fy;
                                forces[j][2] -= fz;
                            }
                        }
                    }
                }
            }

            // Center gravity - pull towards center
            let center = [
                self.config.dimensions[0] / 2.0,
                self.config.dimensions[1] / 2.0,
                self.config.dimensions[2] / 2.0,
            ];

            for i in 0..n {
                let dx = center[0] - positions[i][0];
                let dy = center[1] - positions[i][1];
                let dz = center[2] - positions[i][2];

                // Gravity proportional to importance
                let gravity = 0.01 * frequencies[i] / 1000.0;

                forces[i][0] += dx * gravity;
                forces[i][1] += dy * gravity;
                forces[i][2] += dz * gravity;
            }

            // Apply forces with temperature limiting
            let mut max_displacement = 0.0f32;

            for i in 0..n {
                // Update velocity with damping
                velocities[i][0] =
                    (velocities[i][0] + forces[i][0]) * self.config.damping;
                velocities[i][1] =
                    (velocities[i][1] + forces[i][1]) * self.config.damping;
                velocities[i][2] =
                    (velocities[i][2] + forces[i][2]) * self.config.damping;

                // Limit by temperature
                let vel_magnitude = (velocities[i][0].powi(2)
                    + velocities[i][1].powi(2)
                    + velocities[i][2].powi(2))
                .sqrt();

                if vel_magnitude > temperature {
                    let scale = temperature / vel_magnitude;
                    velocities[i][0] *= scale;
                    velocities[i][1] *= scale;
                    velocities[i][2] *= scale;
                }

                // Update position
                positions[i][0] += velocities[i][0];
                positions[i][1] += velocities[i][1];
                positions[i][2] += velocities[i][2];

                // Keep within bounds
                positions[i][0] = positions[i][0]
                    .max(0.0)
                    .min(self.config.dimensions[0]);
                positions[i][1] = positions[i][1]
                    .max(0.0)
                    .min(self.config.dimensions[1]);
                positions[i][2] = positions[i][2]
                    .max(0.0)
                    .min(self.config.dimensions[2]);

                max_displacement = max_displacement.max(vel_magnitude);
            }

            // Cool down
            temperature -= cooling_rate;
            temperature = temperature.max(0.01);

            // Check for convergence
            if max_displacement < self.config.convergence_threshold {
                tracing::debug!("Force-directed layout converged at iteration {}", _iteration);
                break;
            }
        }

        positions
    }
}

/// Spherical layout - arrange tokens on a sphere
#[derive(Debug, Clone)]
pub struct SphericalLayout {
    config: LayoutConfig,
}

impl SphericalLayout {
    /// Create a new spherical layout
    pub fn new(config: LayoutConfig) -> Self {
        Self { config }
    }
}

impl Layout3D for SphericalLayout {
    fn compute(&self, embeddings: &[Option<&Vec<f32>>], frequencies: &[f32]) -> Vec<[f32; 3]> {
        let n = embeddings.len();
        if n == 0 {
            return Vec::new();
        }

        let center = [
            self.config.dimensions[0] / 2.0,
            self.config.dimensions[1] / 2.0,
            self.config.dimensions[2] / 2.0,
        ];

        let radius = self.config.dimensions[0].min(self.config.dimensions[1])
            .min(self.config.dimensions[2])
            / 2.0
            * 0.8;

        // Sort tokens by frequency for spiral placement
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| {
            frequencies[b]
                .partial_cmp(&frequencies[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut positions = vec![[0.0f32; 3]; n];

        // Use Fibonacci sphere for even distribution
        let golden_ratio = (1.0 + 5.0_f32.sqrt()) / 2.0;

        for (rank, &original_idx) in indices.iter().enumerate() {
            let i = rank as f32;

            // Fibonacci sphere point
            let theta = 2.0 * std::f32::consts::PI * i / golden_ratio;
            let phi = (1.0 - 2.0 * (i + 0.5) / n as f32).acos();

            // Adjust radius based on frequency (more frequent = closer to center)
            let freq_factor = frequencies[original_idx] / frequencies[indices[0]].max(1.0);
            let r = radius * (0.3 + 0.7 * (1.0 - freq_factor.sqrt()));

            positions[original_idx] = [
                center[0] + r * phi.sin() * theta.cos(),
                center[1] + r * phi.sin() * theta.sin(),
                center[2] + r * phi.cos(),
            ];
        }

        positions
    }
}

/// Calculate cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}

/// Grid-based layout for structured visualization
#[derive(Debug, Clone)]
pub struct GridLayout {
    config: LayoutConfig,
    /// Number of cells in each dimension
    pub grid_size: [usize; 3],
}

impl GridLayout {
    /// Create a new grid layout
    pub fn new(config: LayoutConfig, grid_size: [usize; 3]) -> Self {
        Self { config, grid_size }
    }

    /// Auto-calculate grid size based on token count
    pub fn auto(config: LayoutConfig, n: usize) -> Self {
        let side = (n as f32).cbrt().ceil() as usize;
        Self {
            config,
            grid_size: [side, side, side],
        }
    }
}

impl Layout3D for GridLayout {
    fn compute(&self, embeddings: &[Option<&Vec<f32>>], frequencies: &[f32]) -> Vec<[f32; 3]> {
        let n = embeddings.len();
        if n == 0 {
            return Vec::new();
        }

        // Sort by frequency
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| {
            frequencies[b]
                .partial_cmp(&frequencies[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let cell_size = [
            self.config.dimensions[0] / self.grid_size[0] as f32,
            self.config.dimensions[1] / self.grid_size[1] as f32,
            self.config.dimensions[2] / self.grid_size[2] as f32,
        ];

        let mut positions = vec![[0.0f32; 3]; n];

        for (rank, &original_idx) in indices.iter().enumerate() {
            // Convert linear index to 3D grid coordinates
            let x = rank % self.grid_size[0];
            let y = (rank / self.grid_size[0]) % self.grid_size[1];
            let z = rank / (self.grid_size[0] * self.grid_size[1]);

            positions[original_idx] = [
                (x as f32 + 0.5) * cell_size[0],
                (y as f32 + 0.5) * cell_size[1],
                (z as f32 + 0.5) * cell_size[2],
            ];
        }

        positions
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_force_directed_basic() {
        let embeddings: Vec<Option<&Vec<f32>>> = vec![None, None, None];
        let frequencies = vec![100.0, 50.0, 25.0];

        let layout = ForceDirectedLayout::new(LayoutConfig::default());
        let positions = layout.compute(&embeddings, &frequencies);

        assert_eq!(positions.len(), 3);
        // All positions should be within bounds
        for pos in &positions {
            assert!(pos[0] >= 0.0 && pos[0] <= 100.0);
            assert!(pos[1] >= 0.0 && pos[1] <= 100.0);
            assert!(pos[2] >= 0.0 && pos[2] <= 100.0);
        }
    }

    #[test]
    fn test_force_directed_with_embeddings() {
        let emb1 = vec![1.0, 0.0, 0.0];
        let emb2 = vec![0.9, 0.1, 0.0]; // Similar to emb1
        let emb3 = vec![0.0, 0.0, 1.0]; // Different

        let embeddings: Vec<Option<&Vec<f32>>> =
            vec![Some(&emb1), Some(&emb2), Some(&emb3)];
        let frequencies = vec![100.0, 100.0, 100.0];

        let layout = ForceDirectedLayout::new(LayoutConfig::default());
        let positions = layout.compute(&embeddings, &frequencies);

        assert_eq!(positions.len(), 3);
    }

    #[test]
    fn test_spherical_layout() {
        let embeddings: Vec<Option<&Vec<f32>>> = vec![None; 10];
        let frequencies: Vec<f32> = (0..10).map(|i| (100 - i * 10) as f32).collect();

        let layout = SphericalLayout::new(LayoutConfig::default());
        let positions = layout.compute(&embeddings, &frequencies);

        assert_eq!(positions.len(), 10);

        // Check all points are roughly within the sphere
        let center = [50.0, 50.0, 50.0];
        for pos in &positions {
            let dist = ((pos[0] - center[0]).powi(2)
                + (pos[1] - center[1]).powi(2)
                + (pos[2] - center[2]).powi(2))
            .sqrt();
            assert!(dist <= 50.0, "Point outside sphere: dist={}", dist);
        }
    }

    #[test]
    fn test_grid_layout() {
        let embeddings: Vec<Option<&Vec<f32>>> = vec![None; 8];
        let frequencies = vec![1.0; 8];

        let layout = GridLayout::new(LayoutConfig::default(), [2, 2, 2]);
        let positions = layout.compute(&embeddings, &frequencies);

        assert_eq!(positions.len(), 8);

        // All positions should be distinct
        for i in 0..positions.len() {
            for j in (i + 1)..positions.len() {
                let dist = ((positions[i][0] - positions[j][0]).powi(2)
                    + (positions[i][1] - positions[j][1]).powi(2)
                    + (positions[i][2] - positions[j][2]).powi(2))
                .sqrt();
                assert!(dist > 0.1, "Overlapping positions");
            }
        }
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.001);

        let c = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &c).abs() < 0.001); // Orthogonal
    }
}
