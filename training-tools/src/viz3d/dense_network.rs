//! Dense Neural Network Visualization
//!
//! Provides high-performance visualization for dense networks with millions
//! of synaptic connections, inspired by classic MNIST perceptron visualizations.
//!
//! # Features
//!
//! - **Efficient Rendering**: LOD (Level of Detail), instancing, and connection sampling
//! - **Stacked Layer Representation**: Rectangular grids with depth positioning
//! - **Weight-Based Opacity**: Connection brightness based on weight magnitude
//! - **Statistics Overlay**: Network type, layer counts, synapse statistics
//! - **Depth Fog**: Visual clarity for distant connections
//!
//! # Example
//!
//! ```rust,ignore
//! use training_tools::viz3d::dense_network::{DenseNetworkViz, DenseVizConfig};
//!
//! // Create from layer sizes (MNIST-style: 784 -> 256 -> 128 -> 10)
//! let mut viz = DenseNetworkViz::from_layer_sizes(&[784, 256, 128, 10]);
//!
//! // Or create from actual weight matrices
//! let weights = vec![/* ... */];
//! let viz = DenseNetworkViz::from_weight_matrices(&weights);
//!
//! // Render to image buffer
//! let image = viz.render(1920, 1080, 1.0);
//! ```

use super::colors::{Color, Colormap, ColormapPreset};

/// Connection color mode determines how synapse colors are computed
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ConnectionColorMode {
    /// Brightness proportional to |weight|
    #[default]
    WeightMagnitude,
    /// Red for negative weights, blue for positive weights
    WeightSign,
    /// Color gradient based on source/destination layers
    LayerGradient,
    /// Color based on activation strength through connection
    Activation,
}

/// Grid arrangement for neurons within a layer
#[derive(Debug, Clone, Copy)]
pub struct GridArrangement {
    /// Number of rows in the grid
    pub rows: usize,
    /// Number of columns in the grid
    pub cols: usize,
    /// Spacing between neurons (in world units)
    pub spacing: f32,
}

impl GridArrangement {
    /// Create a new grid arrangement
    pub fn new(rows: usize, cols: usize, spacing: f32) -> Self {
        Self {
            rows,
            cols,
            spacing,
        }
    }

    /// Auto-arrange neurons into an approximately square grid
    pub fn auto(neuron_count: usize, spacing: f32) -> Self {
        let cols = (neuron_count as f32).sqrt().ceil() as usize;
        let rows = (neuron_count + cols - 1) / cols;
        Self {
            rows,
            cols,
            spacing,
        }
    }

    /// Get the position of a neuron at the given index
    pub fn neuron_position(&self, index: usize) -> (f32, f32) {
        let row = index / self.cols;
        let col = index % self.cols;

        // Center the grid
        let x = (col as f32 - (self.cols - 1) as f32 / 2.0) * self.spacing;
        let y = (row as f32 - (self.rows - 1) as f32 / 2.0) * self.spacing;

        (x, y)
    }

    /// Get the total width of the grid
    pub fn width(&self) -> f32 {
        (self.cols - 1) as f32 * self.spacing
    }

    /// Get the total height of the grid
    pub fn height(&self) -> f32 {
        (self.rows - 1) as f32 * self.spacing
    }
}

impl Default for GridArrangement {
    fn default() -> Self {
        Self::new(1, 1, 0.1)
    }
}

/// A dense layer in the network
#[derive(Debug, Clone)]
pub struct DenseLayer {
    /// Number of neurons in this layer
    pub neurons: usize,
    /// Z position (depth) in world coordinates
    pub position_z: f32,
    /// How neurons are arranged in 2D
    pub grid_arrangement: GridArrangement,
    /// Optional activation values for each neuron (normalized 0-1)
    pub activations: Option<Vec<f32>>,
    /// Layer name for display
    pub name: String,
}

impl DenseLayer {
    /// Create a new dense layer
    pub fn new(neurons: usize, position_z: f32) -> Self {
        Self {
            neurons,
            position_z,
            grid_arrangement: GridArrangement::auto(neurons, 0.05),
            activations: None,
            name: String::new(),
        }
    }

    /// Create with custom grid arrangement
    pub fn with_grid(neurons: usize, position_z: f32, grid: GridArrangement) -> Self {
        Self {
            neurons,
            position_z,
            grid_arrangement: grid,
            activations: None,
            name: String::new(),
        }
    }

    /// Set the layer name
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Get 3D position of a neuron
    pub fn neuron_position_3d(&self, index: usize) -> [f32; 3] {
        let (x, y) = self.grid_arrangement.neuron_position(index);
        [x, y, self.position_z]
    }
}

/// Configuration for dense network visualization
#[derive(Debug, Clone)]
pub struct DenseVizConfig {
    /// Percentage of connections to show (0.0 - 1.0) for performance
    pub connection_sample_rate: f32,
    /// Only show connections with |weight| above this threshold
    pub min_weight_threshold: f32,
    /// Maximum number of connections to render (hard limit)
    pub max_connections_shown: usize,
    /// Enable Level of Detail based on zoom level
    pub use_lod: bool,
    /// Density of depth fog (0.0 = no fog, 1.0 = heavy fog)
    pub fog_density: f32,
    /// How to color connections
    pub connection_color_mode: ConnectionColorMode,
    /// Base color for connections (modified by color mode)
    pub connection_base_color: Color,
    /// Neuron color for inactive neurons
    pub neuron_color: Color,
    /// Neuron color for highly active neurons
    pub neuron_active_color: Color,
    /// Background color
    pub background_color: Color,
    /// Line width for connections (in pixels at zoom 1.0)
    pub connection_line_width: f32,
    /// Neuron radius (in world units)
    pub neuron_radius: f32,
    /// Whether to show neuron dots
    pub show_neurons: bool,
    /// Whether to use instanced rendering for connections
    pub use_instancing: bool,
    /// LOD thresholds: (zoom_level, sample_rate) pairs
    pub lod_levels: Vec<(f32, f32)>,
    /// Colormap for weight visualization
    pub colormap: ColormapPreset,
}

impl Default for DenseVizConfig {
    fn default() -> Self {
        Self {
            connection_sample_rate: 0.01, // Show 1% by default
            min_weight_threshold: 0.01,
            max_connections_shown: 500_000,
            use_lod: true,
            fog_density: 0.3,
            connection_color_mode: ConnectionColorMode::WeightMagnitude,
            connection_base_color: Color::rgba(0.8, 0.85, 1.0, 0.15),
            neuron_color: Color::rgb(0.3, 0.3, 0.4),
            neuron_active_color: Color::rgb(1.0, 0.9, 0.3),
            background_color: Color::rgb(0.02, 0.02, 0.05),
            connection_line_width: 1.0,
            neuron_radius: 0.02,
            show_neurons: true,
            use_instancing: true,
            lod_levels: vec![
                (0.5, 0.001), // Far zoom: 0.1% connections
                (1.0, 0.01),  // Normal zoom: 1% connections
                (2.0, 0.05),  // Close zoom: 5% connections
                (5.0, 0.2),   // Very close: 20% connections
            ],
            colormap: ColormapPreset::Plasma,
        }
    }
}

impl DenseVizConfig {
    /// Create config optimized for millions of connections
    pub fn for_large_network() -> Self {
        Self {
            connection_sample_rate: 0.001, // Show 0.1%
            max_connections_shown: 100_000,
            fog_density: 0.5,
            use_lod: true,
            use_instancing: true,
            ..Default::default()
        }
    }

    /// Create config for smaller networks (show more detail)
    pub fn for_small_network() -> Self {
        Self {
            connection_sample_rate: 0.1, // Show 10%
            max_connections_shown: 1_000_000,
            fog_density: 0.2,
            use_lod: true,
            ..Default::default()
        }
    }
}

/// Network statistics for overlay display
#[derive(Debug, Clone, Default)]
pub struct NetworkStats {
    /// Network architecture type (e.g., "Perceptron", "MLP", "Transformer")
    pub network_type: String,
    /// Dataset name (e.g., "MNIST", "CIFAR-10")
    pub dataset: String,
    /// Number of hidden layers
    pub hidden_layers: usize,
    /// Total neurons in hidden layers
    pub hidden_neurons: usize,
    /// Total number of synapses (connections)
    pub total_synapses: u64,
    /// Percentage of synapses currently shown
    pub synapses_shown_pct: f32,
    /// Number of synapses currently rendered
    pub synapses_shown: usize,
    /// Learning method (e.g., "Backpropagation", "Hebbian")
    pub learning_method: String,
    /// Current training epoch
    pub current_epoch: Option<usize>,
    /// Current training accuracy
    pub accuracy: Option<f32>,
    /// Current loss value
    pub loss: Option<f32>,
}

impl NetworkStats {
    /// Create stats for a classic MNIST perceptron
    pub fn mnist_perceptron(hidden_sizes: &[usize]) -> Self {
        let hidden_neurons: usize = hidden_sizes.iter().sum();

        // Calculate total synapses
        let mut total_synapses = 784u64 * hidden_sizes.first().copied().unwrap_or(10) as u64;
        for window in hidden_sizes.windows(2) {
            total_synapses += window[0] as u64 * window[1] as u64;
        }
        if let Some(&last) = hidden_sizes.last() {
            total_synapses += last as u64 * 10; // Output layer
        }

        Self {
            network_type: "Perceptron".to_string(),
            dataset: "MNIST".to_string(),
            hidden_layers: hidden_sizes.len(),
            hidden_neurons,
            total_synapses,
            synapses_shown_pct: 0.0,
            synapses_shown: 0,
            learning_method: "Backpropagation".to_string(),
            current_epoch: None,
            accuracy: None,
            loss: None,
        }
    }
}

/// A connection between two neurons for rendering
#[derive(Debug, Clone, Copy)]
pub struct ConnectionInstance {
    /// 3D position of source neuron
    pub source: [f32; 3],
    /// 3D position of target neuron
    pub target: [f32; 3],
    /// Weight value (for color/opacity)
    pub weight: f32,
    /// Computed color (RGBA)
    pub color: [f32; 4],
}

/// Weight matrix between two layers
#[derive(Debug, Clone)]
pub struct WeightMatrix {
    /// Weights[i][j] = weight from neuron j in prev layer to neuron i in next layer
    pub weights: Vec<Vec<f32>>,
    /// Source layer index
    pub from_layer: usize,
    /// Target layer index
    pub to_layer: usize,
    /// Cached max weight magnitude for normalization
    max_magnitude: f32,
}

impl WeightMatrix {
    /// Create a new weight matrix
    pub fn new(weights: Vec<Vec<f32>>, from_layer: usize, to_layer: usize) -> Self {
        let max_magnitude = weights
            .iter()
            .flat_map(|row| row.iter())
            .map(|w| w.abs())
            .fold(0.0f32, f32::max);

        Self {
            weights,
            from_layer,
            to_layer,
            max_magnitude,
        }
    }

    /// Create a random weight matrix for visualization
    pub fn random(from_size: usize, to_size: usize, from_layer: usize, to_layer: usize) -> Self {
        // Simple deterministic pseudo-random based on indices
        let weights: Vec<Vec<f32>> = (0..to_size)
            .map(|i| {
                (0..from_size)
                    .map(|j| {
                        let seed = (i * 7919 + j * 7927) as f32;
                        let val = (seed * 0.001).sin() * (seed * 0.0013).cos();
                        val * 2.0 // Scale to roughly [-2, 2]
                    })
                    .collect()
            })
            .collect();

        Self::new(weights, from_layer, to_layer)
    }

    /// Number of connections in this matrix
    pub fn connection_count(&self) -> usize {
        self.weights.iter().map(|row| row.len()).sum()
    }

    /// Get normalized weight (0-1 range based on max magnitude)
    pub fn normalized_weight(&self, row: usize, col: usize) -> f32 {
        if self.max_magnitude == 0.0 {
            return 0.0;
        }
        self.weights[row][col].abs() / self.max_magnitude
    }

    /// Get weight sign (-1, 0, or 1)
    pub fn weight_sign(&self, row: usize, col: usize) -> i8 {
        let w = self.weights[row][col];
        if w > 0.0 {
            1
        } else if w < 0.0 {
            -1
        } else {
            0
        }
    }
}

/// Main visualization structure for dense networks
#[derive(Debug)]
pub struct DenseNetworkViz {
    /// Layers in the network
    pub layers: Vec<DenseLayer>,
    /// Weight matrices between layers
    pub weight_matrices: Vec<WeightMatrix>,
    /// Visualization configuration
    pub config: DenseVizConfig,
    /// Network statistics
    pub stats: NetworkStats,
    /// Cached connection instances for rendering
    cached_connections: Vec<ConnectionInstance>,
    /// Whether cache needs rebuild
    cache_dirty: bool,
    /// Current LOD level index
    current_lod: usize,
    /// Colormap instance
    colormap: Colormap,
}

impl DenseNetworkViz {
    /// Create a visualization from layer sizes
    ///
    /// Automatically generates random weights for display purposes
    pub fn from_layer_sizes(sizes: &[usize]) -> Self {
        let layer_spacing = 2.0;
        let total_depth = (sizes.len() - 1) as f32 * layer_spacing;

        let layers: Vec<DenseLayer> = sizes
            .iter()
            .enumerate()
            .map(|(i, &size)| {
                let z = -total_depth / 2.0 + i as f32 * layer_spacing;
                let name = if i == 0 {
                    "Input".to_string()
                } else if i == sizes.len() - 1 {
                    "Output".to_string()
                } else {
                    format!("Hidden {}", i)
                };
                DenseLayer::new(size, z).with_name(name)
            })
            .collect();

        // Generate random weight matrices
        let weight_matrices: Vec<WeightMatrix> = sizes
            .windows(2)
            .enumerate()
            .map(|(i, window)| WeightMatrix::random(window[0], window[1], i, i + 1))
            .collect();

        let total_synapses: u64 = weight_matrices
            .iter()
            .map(|m| m.connection_count() as u64)
            .sum();

        let hidden_neurons: usize = if sizes.len() > 2 {
            sizes[1..sizes.len() - 1].iter().sum()
        } else {
            0
        };

        let stats = NetworkStats {
            network_type: "Dense MLP".to_string(),
            hidden_layers: sizes.len().saturating_sub(2),
            hidden_neurons,
            total_synapses,
            ..Default::default()
        };

        let config = if total_synapses > 1_000_000 {
            DenseVizConfig::for_large_network()
        } else {
            DenseVizConfig::default()
        };

        let colormap = config.colormap.colormap();

        Self {
            layers,
            weight_matrices,
            config,
            stats,
            cached_connections: Vec::new(),
            cache_dirty: true,
            current_lod: 1,
            colormap,
        }
    }

    /// Create from actual weight matrices
    ///
    /// weights[i] contains the weight matrix from layer i to layer i+1
    /// Each inner Vec<f32> is a row of weights (to_neuron, from_neurons)
    pub fn from_weight_matrices(weights: &[Vec<Vec<f32>>]) -> Self {
        if weights.is_empty() {
            return Self::from_layer_sizes(&[1]);
        }

        // Infer layer sizes from weight matrices
        let mut sizes = Vec::with_capacity(weights.len() + 1);
        sizes.push(weights[0].first().map(|r| r.len()).unwrap_or(1));
        for w in weights {
            sizes.push(w.len());
        }

        let mut viz = Self::from_layer_sizes(&sizes);

        // Replace random weights with actual weights
        viz.weight_matrices = weights
            .iter()
            .enumerate()
            .map(|(i, w)| WeightMatrix::new(w.clone(), i, i + 1))
            .collect();

        viz.cache_dirty = true;
        viz
    }

    /// Set weights for a specific layer connection
    pub fn set_weights(&mut self, layer_idx: usize, weights: &[Vec<f32>]) {
        if layer_idx < self.weight_matrices.len() {
            self.weight_matrices[layer_idx] =
                WeightMatrix::new(weights.to_vec(), layer_idx, layer_idx + 1);
            self.cache_dirty = true;
        }
    }

    /// Set activations for a layer
    pub fn set_activations(&mut self, layer_idx: usize, activations: &[f32]) {
        if layer_idx < self.layers.len() && activations.len() == self.layers[layer_idx].neurons {
            self.layers[layer_idx].activations = Some(activations.to_vec());
            self.cache_dirty = true;
        }
    }

    /// Update visualization configuration
    pub fn set_config(&mut self, config: DenseVizConfig) {
        self.colormap = config.colormap.colormap();
        self.config = config;
        self.cache_dirty = true;
    }

    /// Get effective sample rate based on zoom and LOD
    fn effective_sample_rate(&self, zoom: f32) -> f32 {
        if !self.config.use_lod {
            return self.config.connection_sample_rate;
        }

        // Find appropriate LOD level
        for &(lod_zoom, rate) in &self.config.lod_levels {
            if zoom <= lod_zoom {
                return rate;
            }
        }

        // Beyond highest LOD, use the base rate
        self.config.connection_sample_rate
    }

    /// Sample connections for rendering (performance optimization)
    fn sample_connections(&self, max_count: usize, sample_rate: f32) -> Vec<ConnectionInstance> {
        let mut connections = Vec::new();
        let threshold = self.config.min_weight_threshold;

        // Deterministic sampling using weight-based priority
        for matrix in &self.weight_matrices {
            let from_layer = &self.layers[matrix.from_layer];
            let to_layer = &self.layers[matrix.to_layer];

            for (to_idx, row) in matrix.weights.iter().enumerate() {
                for (from_idx, &weight) in row.iter().enumerate() {
                    if weight.abs() < threshold {
                        continue;
                    }

                    // Importance-based sampling: higher weights more likely to be shown
                    let importance = weight.abs() / matrix.max_magnitude.max(1e-6);
                    let sample_prob = sample_rate * (0.5 + 0.5 * importance);

                    // Deterministic "random" based on indices
                    let hash = (to_idx * 31337 + from_idx * 7919 + matrix.from_layer * 4999) as f32;
                    let pseudo_random = (hash * 0.0001).fract();

                    if pseudo_random < sample_prob {
                        let source = from_layer.neuron_position_3d(from_idx);
                        let target = to_layer.neuron_position_3d(to_idx);

                        let color = self.compute_connection_color(
                            weight,
                            importance,
                            matrix.from_layer,
                            matrix.to_layer,
                            source[2],
                            target[2],
                        );

                        connections.push(ConnectionInstance {
                            source,
                            target,
                            weight,
                            color,
                        });

                        if connections.len() >= max_count {
                            return connections;
                        }
                    }
                }
            }
        }

        connections
    }

    /// Compute connection color based on configuration
    fn compute_connection_color(
        &self,
        weight: f32,
        normalized_weight: f32,
        from_layer: usize,
        to_layer: usize,
        from_z: f32,
        to_z: f32,
    ) -> [f32; 4] {
        let base = self.config.connection_base_color;

        match self.config.connection_color_mode {
            ConnectionColorMode::WeightMagnitude => {
                // Brightness based on weight magnitude
                let intensity = normalized_weight.sqrt(); // sqrt for better visual distribution
                let color = self.colormap.map(intensity);
                let alpha = (base.a * (0.3 + 0.7 * intensity)).min(1.0);
                [color.r, color.g, color.b, alpha]
            }
            ConnectionColorMode::WeightSign => {
                // Red for negative, blue for positive
                let intensity = normalized_weight.sqrt();
                let (r, g, b) = if weight > 0.0 {
                    (0.2 * intensity, 0.4 * intensity, 1.0 * intensity) // Blue
                } else {
                    (1.0 * intensity, 0.2 * intensity, 0.2 * intensity) // Red
                };
                let alpha = (base.a * (0.3 + 0.7 * intensity)).min(1.0);
                [r, g, b, alpha]
            }
            ConnectionColorMode::LayerGradient => {
                // Color based on layer position
                let total_layers = self.layers.len() as f32;
                let layer_progress = (from_layer + to_layer) as f32 / (2.0 * total_layers);
                let color = self.colormap.map(layer_progress);
                let alpha = base.a * normalized_weight.sqrt();
                [color.r, color.g, color.b, alpha]
            }
            ConnectionColorMode::Activation => {
                // Color based on activation at source
                let activation = self.layers[from_layer]
                    .activations
                    .as_ref()
                    .map(|a| a.get(0).copied().unwrap_or(0.5))
                    .unwrap_or(0.5);
                let color = self.colormap.map(activation);
                let alpha = base.a * normalized_weight.sqrt();
                [color.r, color.g, color.b, alpha]
            }
        }
    }

    /// Apply depth fog to a color
    fn apply_fog(&self, color: [f32; 4], depth: f32, camera_z: f32) -> [f32; 4] {
        if self.config.fog_density == 0.0 {
            return color;
        }

        let distance = (depth - camera_z).abs();
        let fog_factor = (-self.config.fog_density * distance * 0.5).exp();
        let fog_color = self.config.background_color;

        [
            color[0] * fog_factor + fog_color.r * (1.0 - fog_factor),
            color[1] * fog_factor + fog_color.g * (1.0 - fog_factor),
            color[2] * fog_factor + fog_color.b * (1.0 - fog_factor),
            color[3] * fog_factor,
        ]
    }

    /// Rebuild the connection cache
    fn rebuild_cache(&mut self, zoom: f32) {
        let sample_rate = self.effective_sample_rate(zoom);
        let max_connections = self.config.max_connections_shown;

        self.cached_connections = self.sample_connections(max_connections, sample_rate);

        // Update stats
        self.stats.synapses_shown = self.cached_connections.len();
        self.stats.synapses_shown_pct = if self.stats.total_synapses > 0 {
            self.stats.synapses_shown as f32 / self.stats.total_synapses as f32 * 100.0
        } else {
            0.0
        };

        self.cache_dirty = false;
    }

    /// Render the network to an RGBA image buffer
    ///
    /// Returns a Vec<u8> with RGBA values for each pixel
    pub fn render(&mut self, width: u32, height: u32, zoom: f32) -> Vec<u8> {
        // Rebuild cache if needed
        if self.cache_dirty {
            self.rebuild_cache(zoom);
        }

        let mut buffer = vec![0u8; (width * height * 4) as usize];

        // Fill background
        let bg = &self.config.background_color;
        for chunk in buffer.chunks_exact_mut(4) {
            chunk[0] = (bg.r * 255.0) as u8;
            chunk[1] = (bg.g * 255.0) as u8;
            chunk[2] = (bg.b * 255.0) as u8;
            chunk[3] = 255;
        }

        // Camera setup (simple orthographic projection)
        let aspect = width as f32 / height as f32;
        let view_width = 10.0 / zoom;
        let view_height = view_width / aspect;
        let camera_z = -10.0;

        // Project 3D to 2D screen coordinates
        let project = |pos: [f32; 3]| -> Option<(i32, i32)> {
            // Simple perspective projection
            let depth = pos[2] - camera_z;
            if depth <= 0.1 {
                return None;
            }
            let scale = 5.0 / depth * zoom;
            let screen_x = ((pos[0] * scale / view_width + 0.5) * width as f32) as i32;
            let screen_y = ((-pos[1] * scale / view_height + 0.5) * height as f32) as i32;
            Some((screen_x, screen_y))
        };

        // Draw connections (using Bresenham's line algorithm with alpha blending)
        for conn in &self.cached_connections {
            let start = match project(conn.source) {
                Some(p) => p,
                None => continue,
            };
            let end = match project(conn.target) {
                Some(p) => p,
                None => continue,
            };

            // Apply fog based on average depth
            let avg_depth = (conn.source[2] + conn.target[2]) / 2.0;
            let color = self.apply_fog(conn.color, avg_depth, camera_z);

            // Draw line with alpha blending
            draw_line_alpha(&mut buffer, width, height, start, end, color);
        }

        // Draw neurons if enabled
        if self.config.show_neurons {
            for layer in &self.layers {
                for i in 0..layer.neurons {
                    let pos = layer.neuron_position_3d(i);
                    if let Some((x, y)) = project(pos) {
                        // Activation-based color
                        let activation = layer
                            .activations
                            .as_ref()
                            .and_then(|a| a.get(i).copied())
                            .unwrap_or(0.0);

                        let base_color = &self.config.neuron_color;
                        let active_color = &self.config.neuron_active_color;
                        let color = base_color.lerp(active_color, activation);

                        // Draw as small circle
                        let radius =
                            (self.config.neuron_radius * zoom * width as f32 / 10.0) as i32;
                        let radius = radius.max(1).min(5);
                        draw_filled_circle(&mut buffer, width, height, x, y, radius, color);
                    }
                }
            }
        }

        buffer
    }

    /// Render with statistics overlay
    pub fn render_with_overlay(&mut self, width: u32, height: u32, zoom: f32) -> Vec<u8> {
        let mut buffer = self.render(width, height, zoom);

        // Draw overlay text (simplified - in real impl would use font rendering)
        let overlay_lines = self.format_stats_overlay();

        // For now, just add a semi-transparent overlay box in the corner
        let box_width = 280;
        let box_height = 20 + overlay_lines.len() as u32 * 18;
        let box_x = 10;
        let box_y = 10;

        // Draw overlay background
        for y in box_y..(box_y + box_height).min(height) {
            for x in box_x..(box_x + box_width).min(width) {
                let idx = ((y * width + x) * 4) as usize;
                if idx + 3 < buffer.len() {
                    // Semi-transparent dark background
                    buffer[idx] = (buffer[idx] as f32 * 0.3) as u8;
                    buffer[idx + 1] = (buffer[idx + 1] as f32 * 0.3) as u8;
                    buffer[idx + 2] = (buffer[idx + 2] as f32 * 0.3) as u8;
                }
            }
        }

        buffer
    }

    /// Format statistics for overlay display
    pub fn format_stats_overlay(&self) -> Vec<String> {
        let mut lines = Vec::new();

        if !self.stats.network_type.is_empty() {
            lines.push(format!("Type: {}", self.stats.network_type));
        }
        if !self.stats.dataset.is_empty() {
            lines.push(format!("Dataset: {}", self.stats.dataset));
        }

        lines.push(format!(
            "Layers: {} ({} hidden)",
            self.layers.len(),
            self.stats.hidden_layers
        ));
        lines.push(format!("Hidden neurons: {}", self.stats.hidden_neurons));
        lines.push(format!(
            "Synapses: {} total",
            format_number(self.stats.total_synapses)
        ));
        lines.push(format!(
            "Shown: {} ({:.2}%)",
            format_number(self.stats.synapses_shown as u64),
            self.stats.synapses_shown_pct
        ));

        if !self.stats.learning_method.is_empty() {
            lines.push(format!("Learning: {}", self.stats.learning_method));
        }
        if let Some(epoch) = self.stats.current_epoch {
            lines.push(format!("Epoch: {}", epoch));
        }
        if let Some(acc) = self.stats.accuracy {
            lines.push(format!("Accuracy: {:.2}%", acc * 100.0));
        }
        if let Some(loss) = self.stats.loss {
            lines.push(format!("Loss: {:.4}", loss));
        }

        lines
    }

    /// Get layer by index
    pub fn get_layer(&self, index: usize) -> Option<&DenseLayer> {
        self.layers.get(index)
    }

    /// Get mutable layer by index
    pub fn get_layer_mut(&mut self, index: usize) -> Option<&mut DenseLayer> {
        self.cache_dirty = true;
        self.layers.get_mut(index)
    }

    /// Get total synapse count
    pub fn total_synapses(&self) -> u64 {
        self.stats.total_synapses
    }

    /// Get currently shown synapse count
    pub fn shown_synapses(&self) -> usize {
        self.stats.synapses_shown
    }

    /// Force cache rebuild on next render
    pub fn invalidate_cache(&mut self) {
        self.cache_dirty = true;
    }
}

/// Format large numbers with K/M/B suffixes
fn format_number(n: u64) -> String {
    if n >= 1_000_000_000 {
        format!("{:.1}B", n as f64 / 1_000_000_000.0)
    } else if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}

/// Draw a line with alpha blending using Bresenham's algorithm
fn draw_line_alpha(
    buffer: &mut [u8],
    width: u32,
    height: u32,
    start: (i32, i32),
    end: (i32, i32),
    color: [f32; 4],
) {
    let (mut x0, mut y0) = start;
    let (x1, y1) = end;

    let dx = (x1 - x0).abs();
    let dy = -(y1 - y0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;

    loop {
        // Plot pixel with alpha blending
        if x0 >= 0 && x0 < width as i32 && y0 >= 0 && y0 < height as i32 {
            let idx = ((y0 as u32 * width + x0 as u32) * 4) as usize;
            if idx + 3 < buffer.len() {
                let alpha = color[3];
                let inv_alpha = 1.0 - alpha;

                buffer[idx] = ((color[0] * 255.0 * alpha) + (buffer[idx] as f32 * inv_alpha)) as u8;
                buffer[idx + 1] =
                    ((color[1] * 255.0 * alpha) + (buffer[idx + 1] as f32 * inv_alpha)) as u8;
                buffer[idx + 2] =
                    ((color[2] * 255.0 * alpha) + (buffer[idx + 2] as f32 * inv_alpha)) as u8;
            }
        }

        if x0 == x1 && y0 == y1 {
            break;
        }

        let e2 = 2 * err;
        if e2 >= dy {
            if x0 == x1 {
                break;
            }
            err += dy;
            x0 += sx;
        }
        if e2 <= dx {
            if y0 == y1 {
                break;
            }
            err += dx;
            y0 += sy;
        }
    }
}

/// Draw a filled circle
fn draw_filled_circle(
    buffer: &mut [u8],
    width: u32,
    height: u32,
    cx: i32,
    cy: i32,
    radius: i32,
    color: Color,
) {
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            if dx * dx + dy * dy <= radius * radius {
                let x = cx + dx;
                let y = cy + dy;
                if x >= 0 && x < width as i32 && y >= 0 && y < height as i32 {
                    let idx = ((y as u32 * width + x as u32) * 4) as usize;
                    if idx + 3 < buffer.len() {
                        buffer[idx] = (color.r * 255.0) as u8;
                        buffer[idx + 1] = (color.g * 255.0) as u8;
                        buffer[idx + 2] = (color.b * 255.0) as u8;
                        buffer[idx + 3] = 255;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_arrangement_auto() {
        let grid = GridArrangement::auto(784, 0.1);
        assert!(grid.rows * grid.cols >= 784);
        assert!(grid.cols >= grid.rows); // Approximately square
    }

    #[test]
    fn test_grid_position() {
        let grid = GridArrangement::new(3, 3, 1.0);
        let (x, y) = grid.neuron_position(4); // Center position
        assert!((x - 0.0).abs() < 0.01);
        assert!((y - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_from_layer_sizes() {
        let viz = DenseNetworkViz::from_layer_sizes(&[784, 256, 128, 10]);
        assert_eq!(viz.layers.len(), 4);
        assert_eq!(viz.weight_matrices.len(), 3);

        // Check synapse count: 784*256 + 256*128 + 128*10
        let expected = 784 * 256 + 256 * 128 + 128 * 10;
        assert_eq!(viz.stats.total_synapses, expected as u64);
    }

    #[test]
    fn test_weight_matrix_random() {
        let matrix = WeightMatrix::random(100, 50, 0, 1);
        assert_eq!(matrix.weights.len(), 50);
        assert_eq!(matrix.weights[0].len(), 100);
        assert!(matrix.max_magnitude > 0.0);
    }

    #[test]
    fn test_effective_sample_rate() {
        let viz = DenseNetworkViz::from_layer_sizes(&[100, 50, 10]);

        // Low zoom should give lower sample rate
        let rate_low = viz.effective_sample_rate(0.3);
        let rate_high = viz.effective_sample_rate(3.0);

        assert!(rate_low < rate_high);
    }

    #[test]
    fn test_render_dimensions() {
        let mut viz = DenseNetworkViz::from_layer_sizes(&[10, 5, 2]);
        let buffer = viz.render(100, 100, 1.0);

        assert_eq!(buffer.len(), 100 * 100 * 4);
    }

    #[test]
    fn test_stats_overlay() {
        let viz = DenseNetworkViz::from_layer_sizes(&[784, 256, 10]);
        let lines = viz.format_stats_overlay();

        assert!(!lines.is_empty());
        assert!(lines.iter().any(|l| l.contains("Layers")));
        assert!(lines.iter().any(|l| l.contains("Synapses")));
    }

    #[test]
    fn test_format_number() {
        assert_eq!(format_number(500), "500");
        assert_eq!(format_number(1500), "1.5K");
        assert_eq!(format_number(1_500_000), "1.5M");
        assert_eq!(format_number(1_500_000_000), "1.5B");
    }

    #[test]
    fn test_connection_color_modes() {
        let mut viz = DenseNetworkViz::from_layer_sizes(&[10, 5, 2]);

        // Test each color mode renders without panic
        for mode in [
            ConnectionColorMode::WeightMagnitude,
            ConnectionColorMode::WeightSign,
            ConnectionColorMode::LayerGradient,
            ConnectionColorMode::Activation,
        ] {
            viz.config.connection_color_mode = mode;
            viz.cache_dirty = true;
            let _ = viz.render(50, 50, 1.0);
        }
    }

    #[test]
    fn test_set_weights() {
        let mut viz = DenseNetworkViz::from_layer_sizes(&[3, 2, 1]);

        let new_weights = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        viz.set_weights(0, &new_weights);

        assert_eq!(viz.weight_matrices[0].weights[0][0], 1.0);
        assert_eq!(viz.weight_matrices[0].weights[1][2], 6.0);
    }

    #[test]
    fn test_set_activations() {
        let mut viz = DenseNetworkViz::from_layer_sizes(&[3, 2]);

        viz.set_activations(0, &[0.5, 0.8, 0.2]);
        assert!(viz.layers[0].activations.is_some());

        let acts = viz.layers[0].activations.as_ref().unwrap();
        assert_eq!(acts[1], 0.8);
    }

    #[test]
    fn test_large_network_config() {
        let config = DenseVizConfig::for_large_network();
        assert!(config.connection_sample_rate < 0.01);
        assert!(config.use_lod);
    }
}
