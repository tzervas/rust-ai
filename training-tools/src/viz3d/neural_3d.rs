//! Professional 3D Neural Network Visualization with Glowing Nodes and Particle Effects
//!
//! This module provides a sci-fi style neural network visualization featuring:
//! - Glowing blue/cyan nodes with bloom effects
//! - Purple/pink connection lines with weight-based opacity
//! - Animated particle system for data flow visualization
//! - Binary/hex data overlay on the sides
//! - Force-directed 3D layout for node positioning
//! - Layer-wise organization (input -> hidden -> output)
//!
//! # Example
//!
//! ```rust,ignore
//! use training_tools::viz3d::neural_3d::{Neural3DScene, RenderConfig};
//!
//! // Create scene from layer architecture
//! let mut scene = Neural3DScene::from_architecture(&[4, 8, 8, 2]);
//!
//! // Animate and render
//! scene.update(0.016); // 60fps delta
//! let pixels = scene.render(1920, 1080);
//! ```

use std::f32::consts::PI;

use rand::Rng;

// ============================================================================
// Core Types
// ============================================================================

/// A 3D node representing a neuron in the network
#[derive(Debug, Clone)]
pub struct Node3D {
    /// Position in 3D space [x, y, z]
    pub position: [f32; 3],
    /// Node radius for rendering
    pub radius: f32,
    /// Base color (RGBA)
    pub color: [f32; 4],
    /// Glow intensity multiplier (0.0 = no glow, 1.0+ = bright glow)
    pub glow_intensity: f32,
    /// Which layer this node belongs to
    pub layer_idx: usize,
    /// Optional text label
    pub label: Option<String>,
    /// Current activation value (used for animation, 0.0-1.0)
    pub activation: f32,
    /// Velocity for force-directed layout
    velocity: [f32; 3],
    /// Whether this node is currently receiving a forward pass signal
    pub is_active: bool,
}

impl Default for Node3D {
    fn default() -> Self {
        Self {
            position: [0.0, 0.0, 0.0],
            radius: 0.3,
            color: CYAN_GLOW,
            glow_intensity: 0.8,
            layer_idx: 0,
            label: None,
            activation: 0.0,
            velocity: [0.0, 0.0, 0.0],
            is_active: false,
        }
    }
}

impl Node3D {
    /// Create a new node at the specified position
    pub fn new(position: [f32; 3], layer_idx: usize) -> Self {
        Self {
            position,
            layer_idx,
            ..Default::default()
        }
    }

    /// Set the glow color based on activation
    pub fn set_activation_color(&mut self) {
        let t = self.activation;
        // Interpolate from cyan (inactive) to white-blue (active)
        self.color = [
            CYAN_GLOW[0] + t * 0.3,
            CYAN_GLOW[1] + t * 0.2,
            CYAN_GLOW[2],
            CYAN_GLOW[3],
        ];
        self.glow_intensity = 0.5 + t * 1.5;
    }
}

/// A connection between two nodes
#[derive(Debug, Clone)]
pub struct Connection3D {
    /// Index of source node
    pub source_idx: usize,
    /// Index of target node
    pub target_idx: usize,
    /// Connection weight (affects opacity and color)
    pub weight: f32,
    /// Base color (RGBA)
    pub color: [f32; 4],
    /// Particles flowing along this connection
    pub flow_particles: Vec<Particle>,
    /// Number of particles to maintain on this connection
    particle_count: usize,
}

impl Connection3D {
    /// Create a new connection between nodes
    pub fn new(source_idx: usize, target_idx: usize, weight: f32) -> Self {
        let abs_weight = weight.abs();
        let particle_count = (abs_weight * 3.0).ceil() as usize;

        // Color based on weight sign: positive = purple, negative = orange
        let color = if weight >= 0.0 {
            [
                PURPLE_CONNECTION[0],
                PURPLE_CONNECTION[1],
                PURPLE_CONNECTION[2],
                (abs_weight * 0.8).min(0.9),
            ]
        } else {
            [0.9, 0.4, 0.1, (abs_weight * 0.8).min(0.9)]
        };

        Self {
            source_idx,
            target_idx,
            weight,
            color,
            flow_particles: Vec::with_capacity(particle_count),
            particle_count,
        }
    }

    /// Update particle positions along the connection
    pub fn update_particles(&mut self, dt: f32, source_pos: [f32; 3], target_pos: [f32; 3]) {
        let speed = 0.5 + self.weight.abs() * 0.5;

        // Update existing particles
        for particle in &mut self.flow_particles {
            particle.age += dt;
            particle.progress += speed * dt;

            if particle.progress >= 1.0 {
                particle.progress = 0.0;
                particle.age = 0.0;
            }

            // Interpolate position along the connection
            let t = particle.progress;
            particle.position = [
                source_pos[0] + t * (target_pos[0] - source_pos[0]),
                source_pos[1] + t * (target_pos[1] - source_pos[1]),
                source_pos[2] + t * (target_pos[2] - source_pos[2]),
            ];
        }

        // Spawn new particles if needed
        let mut rng = rand::thread_rng();
        while self.flow_particles.len() < self.particle_count {
            self.flow_particles.push(Particle {
                position: source_pos,
                velocity: [0.0, 0.0, 0.0],
                color: PARTICLE_COLOR,
                size: 0.05 + rng.gen::<f32>() * 0.05,
                age: 0.0,
                progress: rng.gen::<f32>(),
            });
        }
    }
}

/// A single particle in the visualization
#[derive(Debug, Clone)]
pub struct Particle {
    /// Current position [x, y, z]
    pub position: [f32; 3],
    /// Velocity vector [vx, vy, vz]
    pub velocity: [f32; 3],
    /// Particle color (RGBA)
    pub color: [f32; 4],
    /// Particle size
    pub size: f32,
    /// Time since spawn
    pub age: f32,
    /// Progress along connection (0.0-1.0) for flow particles
    progress: f32,
}

impl Default for Particle {
    fn default() -> Self {
        Self {
            position: [0.0, 0.0, 0.0],
            velocity: [0.0, 0.0, 0.0],
            color: PARTICLE_COLOR,
            size: 0.05,
            age: 0.0,
            progress: 0.0,
        }
    }
}

/// Background particle system for ambient effects
#[derive(Debug, Clone)]
pub struct ParticleSystem {
    /// All particles in the system
    pub particles: Vec<Particle>,
    /// Particles emitted per second
    pub emission_rate: f32,
    /// Particle lifetime in seconds
    pub lifetime: f32,
    /// Bounding box for particle spawn [min, max]
    bounds: ([f32; 3], [f32; 3]),
    /// Time accumulator for emission
    emission_accumulator: f32,
}

impl Default for ParticleSystem {
    fn default() -> Self {
        Self {
            particles: Vec::with_capacity(200),
            emission_rate: 20.0,
            lifetime: 5.0,
            bounds: ([-10.0, -5.0, -5.0], [10.0, 5.0, 5.0]),
            emission_accumulator: 0.0,
        }
    }
}

impl ParticleSystem {
    /// Create a particle system with custom bounds
    pub fn new(emission_rate: f32, lifetime: f32, bounds: ([f32; 3], [f32; 3])) -> Self {
        let max_particles = (emission_rate * lifetime) as usize + 10;
        Self {
            particles: Vec::with_capacity(max_particles),
            emission_rate,
            lifetime,
            bounds,
            emission_accumulator: 0.0,
        }
    }

    /// Update all particles
    pub fn update(&mut self, dt: f32) {
        let mut rng = rand::thread_rng();

        // Update existing particles
        self.particles.retain_mut(|p| {
            p.age += dt;
            if p.age >= self.lifetime {
                return false;
            }

            // Slow upward drift with slight random motion
            p.position[0] += p.velocity[0] * dt;
            p.position[1] += p.velocity[1] * dt + 0.1 * dt;
            p.position[2] += p.velocity[2] * dt;

            // Fade out near end of life
            let life_ratio = p.age / self.lifetime;
            p.color[3] = (1.0 - life_ratio) * 0.3;

            true
        });

        // Emit new particles
        self.emission_accumulator += self.emission_rate * dt;
        while self.emission_accumulator >= 1.0 {
            self.emission_accumulator -= 1.0;

            let (min, max) = self.bounds;
            self.particles.push(Particle {
                position: [
                    min[0] + rng.gen::<f32>() * (max[0] - min[0]),
                    min[1] + rng.gen::<f32>() * (max[1] - min[1]),
                    min[2] + rng.gen::<f32>() * (max[2] - min[2]),
                ],
                velocity: [
                    (rng.gen::<f32>() - 0.5) * 0.2,
                    rng.gen::<f32>() * 0.1,
                    (rng.gen::<f32>() - 0.5) * 0.2,
                ],
                color: [0.2, 0.6, 0.9, 0.3],
                size: 0.02 + rng.gen::<f32>() * 0.03,
                age: 0.0,
                progress: 0.0,
            });
        }
    }
}

/// 3D camera for the scene
#[derive(Debug, Clone)]
pub struct Camera3D {
    /// Camera position [x, y, z]
    pub position: [f32; 3],
    /// Look-at target [x, y, z]
    pub target: [f32; 3],
    /// Up vector [x, y, z]
    pub up: [f32; 3],
    /// Field of view in radians
    pub fov: f32,
    /// Near clipping plane
    pub near: f32,
    /// Far clipping plane
    pub far: f32,
    /// Orbit angle (azimuth) in radians
    pub azimuth: f32,
    /// Orbit angle (elevation) in radians
    pub elevation: f32,
    /// Distance from target
    pub distance: f32,
}

impl Default for Camera3D {
    fn default() -> Self {
        let distance = 15.0;
        let azimuth = PI / 4.0;
        let elevation = PI / 6.0;

        let position = Self::orbit_position(distance, azimuth, elevation, [0.0, 0.0, 0.0]);

        Self {
            position,
            target: [0.0, 0.0, 0.0],
            up: [0.0, 1.0, 0.0],
            fov: PI / 4.0,
            near: 0.1,
            far: 100.0,
            azimuth,
            elevation,
            distance,
        }
    }
}

impl Camera3D {
    /// Calculate camera position from orbit parameters
    fn orbit_position(distance: f32, azimuth: f32, elevation: f32, target: [f32; 3]) -> [f32; 3] {
        let x = distance * elevation.cos() * azimuth.cos();
        let y = distance * elevation.sin();
        let z = distance * elevation.cos() * azimuth.sin();
        [target[0] + x, target[1] + y, target[2] + z]
    }

    /// Orbit the camera
    pub fn orbit(&mut self, delta_azimuth: f32, delta_elevation: f32) {
        self.azimuth += delta_azimuth;
        self.elevation = (self.elevation + delta_elevation).clamp(-PI / 2.0 + 0.1, PI / 2.0 - 0.1);
        self.update_position();
    }

    /// Zoom the camera
    pub fn zoom(&mut self, factor: f32) {
        self.distance = (self.distance * factor).clamp(1.0, 50.0);
        self.update_position();
    }

    fn update_position(&mut self) {
        self.position =
            Self::orbit_position(self.distance, self.azimuth, self.elevation, self.target);
    }

    /// Get the view matrix as a flat 16-element array (column-major)
    pub fn view_matrix(&self) -> [f32; 16] {
        let forward = normalize([
            self.target[0] - self.position[0],
            self.target[1] - self.position[1],
            self.target[2] - self.position[2],
        ]);
        let right = normalize(cross(forward, self.up));
        let up = cross(right, forward);

        [
            right[0],
            up[0],
            -forward[0],
            0.0,
            right[1],
            up[1],
            -forward[1],
            0.0,
            right[2],
            up[2],
            -forward[2],
            0.0,
            -dot(right, self.position),
            -dot(up, self.position),
            dot(forward, self.position),
            1.0,
        ]
    }

    /// Get the projection matrix as a flat 16-element array (column-major)
    pub fn projection_matrix(&self, aspect: f32) -> [f32; 16] {
        let f = 1.0 / (self.fov / 2.0).tan();
        let nf = 1.0 / (self.near - self.far);

        [
            f / aspect,
            0.0,
            0.0,
            0.0,
            0.0,
            f,
            0.0,
            0.0,
            0.0,
            0.0,
            (self.far + self.near) * nf,
            -1.0,
            0.0,
            0.0,
            2.0 * self.far * self.near * nf,
            0.0,
        ]
    }
}

/// Rendering configuration
#[derive(Debug, Clone)]
pub struct RenderConfig {
    /// Background color (RGBA)
    pub background_color: [f32; 4],
    /// Ambient light intensity (0.0-1.0)
    pub ambient_light: f32,
    /// Bloom/glow effect intensity
    pub bloom_intensity: f32,
    /// Enable particle glow effects
    pub particle_glow: bool,
    /// Show node labels
    pub show_labels: bool,
    /// Show binary/hex data overlay
    pub show_data_overlay: bool,
    /// Overlay text (for binary/hex display)
    pub overlay_data: Vec<String>,
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self {
            background_color: DARK_BACKGROUND,
            ambient_light: 0.2,
            bloom_intensity: 1.2,
            particle_glow: true,
            show_labels: false,
            show_data_overlay: true,
            overlay_data: Vec::new(),
        }
    }
}

// ============================================================================
// Color Constants (Sci-Fi Theme)
// ============================================================================

/// Glowing cyan for nodes
const CYAN_GLOW: [f32; 4] = [0.2, 0.8, 1.0, 1.0];

/// Purple/pink for connections
const PURPLE_CONNECTION: [f32; 4] = [0.7, 0.3, 0.9, 0.6];

/// Particle color
const PARTICLE_COLOR: [f32; 4] = [0.3, 0.7, 1.0, 0.5];

/// Dark blue-black background
const DARK_BACKGROUND: [f32; 4] = [0.02, 0.02, 0.05, 1.0];

// ============================================================================
// Main Scene Structure
// ============================================================================

/// The main 3D neural network scene
#[derive(Debug, Clone)]
pub struct Neural3DScene {
    /// All nodes in the network
    pub nodes: Vec<Node3D>,
    /// All connections between nodes
    pub connections: Vec<Connection3D>,
    /// Background particle system
    pub particles: ParticleSystem,
    /// Scene camera
    pub camera: Camera3D,
    /// Rendering configuration
    pub config: RenderConfig,
    /// Layer sizes (for reference)
    layer_sizes: Vec<usize>,
    /// Animation time accumulator
    time: f32,
    /// Current forward pass layer (for animation)
    forward_pass_layer: Option<usize>,
    /// Forward pass progress (0.0-1.0)
    forward_pass_progress: f32,
}

impl Neural3DScene {
    /// Create a scene from layer architecture (e.g., [4, 8, 8, 2])
    pub fn from_architecture(layers: &[usize]) -> Self {
        let mut nodes = Vec::new();
        let mut node_idx = 0;

        // Calculate total width for centering
        let num_layers = layers.len();
        let layer_spacing = 3.0;
        let total_width = (num_layers - 1) as f32 * layer_spacing;

        // Create nodes for each layer
        for (layer_idx, &layer_size) in layers.iter().enumerate() {
            let x = -total_width / 2.0 + layer_idx as f32 * layer_spacing;

            // Arrange nodes in a circle within the layer
            let node_spacing = 0.8;
            let layer_height = (layer_size - 1) as f32 * node_spacing;

            for node_in_layer in 0..layer_size {
                let y = -layer_height / 2.0 + node_in_layer as f32 * node_spacing;
                let z = 0.0;

                let mut node = Node3D::new([x, y, z], layer_idx);
                node.label = Some(format!("L{}N{}", layer_idx, node_in_layer));

                // Color variation by layer
                let layer_hue = layer_idx as f32 / num_layers as f32;
                node.color = [0.2 + layer_hue * 0.3, 0.7 - layer_hue * 0.2, 1.0, 1.0];

                nodes.push(node);
                node_idx += 1;
            }
        }

        // Create connections between adjacent layers
        let mut connections = Vec::new();
        let mut rng = rand::thread_rng();

        let mut src_start = 0;
        for layer_idx in 0..num_layers - 1 {
            let src_size = layers[layer_idx];
            let dst_size = layers[layer_idx + 1];
            let dst_start = src_start + src_size;

            for src in 0..src_size {
                for dst in 0..dst_size {
                    // Random weight between -1 and 1
                    let weight = rng.gen::<f32>() * 2.0 - 1.0;
                    connections.push(Connection3D::new(src_start + src, dst_start + dst, weight));
                }
            }

            src_start = dst_start;
        }

        // Create particle system bounds based on network extent
        let x_extent = total_width / 2.0 + 2.0;
        let y_extent = layers.iter().max().copied().unwrap_or(1) as f32 * 0.5 + 2.0;
        let particles = ParticleSystem::new(
            30.0,
            4.0,
            ([-x_extent, -y_extent, -3.0], [x_extent, y_extent, 3.0]),
        );

        let mut scene = Self {
            nodes,
            connections,
            particles,
            camera: Camera3D::default(),
            config: RenderConfig::default(),
            layer_sizes: layers.to_vec(),
            time: 0.0,
            forward_pass_layer: None,
            forward_pass_progress: 0.0,
        };

        // Generate some overlay data
        scene.generate_overlay_data();

        scene
    }

    /// Create a scene from model weights
    pub fn from_model_weights(weights: &[Vec<Vec<f32>>]) -> Self {
        // Infer layer sizes from weights
        let mut layer_sizes = Vec::new();

        if let Some(first_layer) = weights.first() {
            if let Some(first_row) = first_layer.first() {
                layer_sizes.push(first_row.len());
            }
            layer_sizes.push(first_layer.len());
        }

        for layer in weights.iter().skip(1) {
            layer_sizes.push(layer.len());
        }

        let mut scene = Self::from_architecture(&layer_sizes);

        // Update connection weights from actual model
        let mut conn_idx = 0;
        for layer_weights in weights {
            for row in layer_weights {
                for &weight in row {
                    if conn_idx < scene.connections.len() {
                        scene.connections[conn_idx].weight = weight;
                        // Update color based on weight
                        let abs_weight = weight.abs().min(1.0);
                        scene.connections[conn_idx].color[3] = (abs_weight * 0.8).max(0.1);
                    }
                    conn_idx += 1;
                }
            }
        }

        scene
    }

    /// Generate binary/hex overlay data
    fn generate_overlay_data(&mut self) {
        let mut rng = rand::thread_rng();
        let mut data = Vec::new();

        for _ in 0..20 {
            let line: String = (0..32)
                .map(|_| if rng.gen::<bool>() { '1' } else { '0' })
                .collect();
            data.push(line);
        }

        for _ in 0..10 {
            let hex: String = (0..16)
                .map(|_| {
                    let v: u8 = rng.gen();
                    format!("{:02X}", v)
                })
                .collect::<Vec<_>>()
                .join(" ");
            data.push(hex);
        }

        self.config.overlay_data = data;
    }

    // ========================================================================
    // Animation
    // ========================================================================

    /// Update the scene (call each frame)
    pub fn update(&mut self, dt: f32) {
        self.time += dt;

        // Update background particles
        self.particles.update(dt);

        // Update connection particles
        for conn in &mut self.connections {
            let src_pos = self.nodes[conn.source_idx].position;
            let dst_pos = self.nodes[conn.target_idx].position;
            conn.update_particles(dt, src_pos, dst_pos);
        }

        // Animate node glow (subtle pulsing)
        for node in &mut self.nodes {
            let pulse = 0.1 * (self.time * 2.0 + node.layer_idx as f32).sin();
            node.glow_intensity = 0.7 + pulse + node.activation * 0.5;
        }

        // Forward pass animation
        if let Some(layer) = self.forward_pass_layer {
            self.forward_pass_progress += dt * 2.0;

            if self.forward_pass_progress >= 1.0 {
                // Deactivate current layer, activate next
                for node in &mut self.nodes {
                    if node.layer_idx == layer {
                        node.activation = node.activation * 0.9;
                        node.is_active = false;
                    }
                }

                if layer + 1 < self.layer_sizes.len() {
                    self.forward_pass_layer = Some(layer + 1);
                    self.forward_pass_progress = 0.0;

                    for node in &mut self.nodes {
                        if node.layer_idx == layer + 1 {
                            node.is_active = true;
                        }
                    }
                } else {
                    self.forward_pass_layer = None;
                }
            }

            // Update activations during pass
            for node in &mut self.nodes {
                if node.layer_idx == layer {
                    node.activation = (1.0 - self.forward_pass_progress).max(0.0);
                } else if node.layer_idx == layer + 1 {
                    node.activation = self.forward_pass_progress;
                }
                node.set_activation_color();
            }
        }
    }

    /// Set node activations (for visualization)
    pub fn set_activations(&mut self, activations: &[Vec<f32>]) {
        let mut node_idx = 0;
        for layer_acts in activations {
            for &act in layer_acts {
                if node_idx < self.nodes.len() {
                    self.nodes[node_idx].activation = act.clamp(0.0, 1.0);
                    self.nodes[node_idx].set_activation_color();
                }
                node_idx += 1;
            }
        }
    }

    /// Animate a forward pass through the network
    pub fn animate_forward_pass(&mut self, input: &[f32]) {
        // Set input layer activations
        let input_size = self.layer_sizes.first().copied().unwrap_or(0);
        for (i, &val) in input.iter().enumerate() {
            if i < input_size && i < self.nodes.len() {
                self.nodes[i].activation = val.clamp(0.0, 1.0);
                self.nodes[i].is_active = true;
                self.nodes[i].set_activation_color();
            }
        }

        // Start the animation
        self.forward_pass_layer = Some(0);
        self.forward_pass_progress = 0.0;
    }

    // ========================================================================
    // Force-Directed Layout
    // ========================================================================

    /// Apply force-directed layout to optimize node positions
    pub fn apply_force_layout(&mut self, iterations: usize) {
        const REPULSION: f32 = 2.0;
        const ATTRACTION: f32 = 0.1;
        const DAMPING: f32 = 0.9;
        const MAX_VELOCITY: f32 = 0.5;

        for _ in 0..iterations {
            // Calculate forces
            let n = self.nodes.len();
            let mut forces: Vec<[f32; 3]> = vec![[0.0, 0.0, 0.0]; n];

            // Repulsion between all nodes
            for i in 0..n {
                for j in (i + 1)..n {
                    let dx = self.nodes[j].position[0] - self.nodes[i].position[0];
                    let dy = self.nodes[j].position[1] - self.nodes[i].position[1];
                    let dz = self.nodes[j].position[2] - self.nodes[i].position[2];
                    let dist_sq = dx * dx + dy * dy + dz * dz + 0.01;
                    let dist = dist_sq.sqrt();

                    let force = REPULSION / dist_sq;
                    let fx = force * dx / dist;
                    let fy = force * dy / dist;
                    let fz = force * dz / dist;

                    forces[i][0] -= fx;
                    forces[i][1] -= fy;
                    forces[i][2] -= fz;
                    forces[j][0] += fx;
                    forces[j][1] += fy;
                    forces[j][2] += fz;
                }
            }

            // Attraction along connections
            for conn in &self.connections {
                let src = &self.nodes[conn.source_idx];
                let dst = &self.nodes[conn.target_idx];

                let dx = dst.position[0] - src.position[0];
                let dy = dst.position[1] - src.position[1];
                let dz = dst.position[2] - src.position[2];
                let dist = (dx * dx + dy * dy + dz * dz).sqrt();

                if dist > 0.1 {
                    let force = ATTRACTION * dist * conn.weight.abs();
                    let fx = force * dx / dist;
                    let fy = force * dy / dist;
                    let fz = force * dz / dist;

                    forces[conn.source_idx][0] += fx;
                    forces[conn.source_idx][1] += fy;
                    forces[conn.source_idx][2] += fz;
                    forces[conn.target_idx][0] -= fx;
                    forces[conn.target_idx][1] -= fy;
                    forces[conn.target_idx][2] -= fz;
                }
            }

            // Apply forces (but keep X position fixed for layer structure)
            for (i, node) in self.nodes.iter_mut().enumerate() {
                node.velocity[0] = 0.0; // Keep layer X position fixed
                node.velocity[1] = (node.velocity[1] + forces[i][1]) * DAMPING;
                node.velocity[2] = (node.velocity[2] + forces[i][2]) * DAMPING;

                // Clamp velocity
                for v in &mut node.velocity {
                    *v = v.clamp(-MAX_VELOCITY, MAX_VELOCITY);
                }

                node.position[1] += node.velocity[1];
                node.position[2] += node.velocity[2];
            }
        }
    }

    // ========================================================================
    // Rendering
    // ========================================================================

    /// Render the scene to an RGBA pixel buffer
    ///
    /// This is a software renderer for offline/headless rendering.
    /// For real-time rendering, use with a GPU-based renderer.
    pub fn render(&self, width: u32, height: u32) -> Vec<u8> {
        let mut pixels = vec![0u8; (width * height * 4) as usize];

        // Fill background
        let bg = &self.config.background_color;
        for y in 0..height {
            for x in 0..width {
                let idx = ((y * width + x) * 4) as usize;
                pixels[idx] = (bg[0] * 255.0) as u8;
                pixels[idx + 1] = (bg[1] * 255.0) as u8;
                pixels[idx + 2] = (bg[2] * 255.0) as u8;
                pixels[idx + 3] = 255;
            }
        }

        let aspect = width as f32 / height as f32;
        let view = self.camera.view_matrix();
        let proj = self.camera.projection_matrix(aspect);

        // Render background particles first (no depth test for simplicity)
        for particle in &self.particles.particles {
            let screen_pos = self.project_point(particle.position, &view, &proj, width, height);
            if let Some((sx, sy, _)) = screen_pos {
                self.draw_glow_point(
                    &mut pixels,
                    width,
                    height,
                    sx,
                    sy,
                    particle.size * 50.0,
                    particle.color,
                );
            }
        }

        // Render connections
        for conn in &self.connections {
            let src_pos = self.nodes[conn.source_idx].position;
            let dst_pos = self.nodes[conn.target_idx].position;

            let src_screen = self.project_point(src_pos, &view, &proj, width, height);
            let dst_screen = self.project_point(dst_pos, &view, &proj, width, height);

            if let (Some((sx1, sy1, _)), Some((sx2, sy2, _))) = (src_screen, dst_screen) {
                self.draw_line(&mut pixels, width, height, sx1, sy1, sx2, sy2, conn.color);
            }

            // Render flow particles on connections
            for particle in &conn.flow_particles {
                let screen_pos = self.project_point(particle.position, &view, &proj, width, height);
                if let Some((sx, sy, _)) = screen_pos {
                    self.draw_glow_point(
                        &mut pixels,
                        width,
                        height,
                        sx,
                        sy,
                        particle.size * 80.0,
                        [0.5, 0.8, 1.0, 0.8],
                    );
                }
            }
        }

        // Render nodes with glow
        for node in &self.nodes {
            let screen_pos = self.project_point(node.position, &view, &proj, width, height);
            if let Some((sx, sy, depth)) = screen_pos {
                // Size varies with depth
                let size = node.radius * 100.0 / (1.0 + depth * 0.5);
                let glow_size = size * (1.0 + node.glow_intensity);

                // Draw outer glow
                let glow_color = [
                    node.color[0] * 0.5,
                    node.color[1] * 0.5,
                    node.color[2] * 0.5,
                    0.3 * node.glow_intensity,
                ];
                self.draw_glow_point(
                    &mut pixels,
                    width,
                    height,
                    sx,
                    sy,
                    glow_size * 2.0,
                    glow_color,
                );

                // Draw node core
                self.draw_glow_point(&mut pixels, width, height, sx, sy, size, node.color);
            }
        }

        // Render data overlay on sides
        if self.config.show_data_overlay {
            self.render_data_overlay(&mut pixels, width, height);
        }

        pixels
    }

    /// Project a 3D point to screen coordinates
    fn project_point(
        &self,
        pos: [f32; 3],
        view: &[f32; 16],
        proj: &[f32; 16],
        width: u32,
        height: u32,
    ) -> Option<(i32, i32, f32)> {
        // Transform by view matrix
        let vx = view[0] * pos[0] + view[4] * pos[1] + view[8] * pos[2] + view[12];
        let vy = view[1] * pos[0] + view[5] * pos[1] + view[9] * pos[2] + view[13];
        let vz = view[2] * pos[0] + view[6] * pos[1] + view[10] * pos[2] + view[14];
        let vw = view[3] * pos[0] + view[7] * pos[1] + view[11] * pos[2] + view[15];

        // Transform by projection matrix
        let px = proj[0] * vx + proj[4] * vy + proj[8] * vz + proj[12] * vw;
        let py = proj[1] * vx + proj[5] * vy + proj[9] * vz + proj[13] * vw;
        let pz = proj[2] * vx + proj[6] * vy + proj[10] * vz + proj[14] * vw;
        let pw = proj[3] * vx + proj[7] * vy + proj[11] * vz + proj[15] * vw;

        // Perspective divide
        if pw.abs() < 0.0001 {
            return None;
        }
        let ndx = px / pw;
        let ndy = py / pw;
        let depth = pz / pw;

        // Clip to view frustum
        if ndx < -1.0 || ndx > 1.0 || ndy < -1.0 || ndy > 1.0 || depth < -1.0 || depth > 1.0 {
            return None;
        }

        // Convert to screen coordinates
        let sx = ((ndx + 1.0) * 0.5 * width as f32) as i32;
        let sy = ((1.0 - ndy) * 0.5 * height as f32) as i32;

        Some((sx, sy, depth))
    }

    /// Draw a glowing point (soft circle)
    fn draw_glow_point(
        &self,
        pixels: &mut [u8],
        width: u32,
        height: u32,
        cx: i32,
        cy: i32,
        radius: f32,
        color: [f32; 4],
    ) {
        let r = radius as i32;
        let radius_sq = radius * radius;

        for dy in -r..=r {
            for dx in -r..=r {
                let x = cx + dx;
                let y = cy + dy;

                if x < 0 || x >= width as i32 || y < 0 || y >= height as i32 {
                    continue;
                }

                let dist_sq = (dx * dx + dy * dy) as f32;
                if dist_sq > radius_sq {
                    continue;
                }

                // Smooth falloff
                let dist = dist_sq.sqrt();
                let falloff = 1.0 - (dist / radius).powf(2.0);
                let alpha = color[3] * falloff * self.config.bloom_intensity.min(1.5);

                let idx = ((y as u32 * width + x as u32) * 4) as usize;

                // Additive blending for glow effect
                let old_r = pixels[idx] as f32 / 255.0;
                let old_g = pixels[idx + 1] as f32 / 255.0;
                let old_b = pixels[idx + 2] as f32 / 255.0;

                let new_r = (old_r + color[0] * alpha).min(1.0);
                let new_g = (old_g + color[1] * alpha).min(1.0);
                let new_b = (old_b + color[2] * alpha).min(1.0);

                pixels[idx] = (new_r * 255.0) as u8;
                pixels[idx + 1] = (new_g * 255.0) as u8;
                pixels[idx + 2] = (new_b * 255.0) as u8;
            }
        }
    }

    /// Draw a line between two points
    fn draw_line(
        &self,
        pixels: &mut [u8],
        width: u32,
        height: u32,
        x1: i32,
        y1: i32,
        x2: i32,
        y2: i32,
        color: [f32; 4],
    ) {
        let dx = (x2 - x1).abs();
        let dy = (y2 - y1).abs();
        let sx = if x1 < x2 { 1 } else { -1 };
        let sy = if y1 < y2 { 1 } else { -1 };
        let mut err = dx - dy;

        let mut x = x1;
        let mut y = y1;

        loop {
            if x >= 0 && x < width as i32 && y >= 0 && y < height as i32 {
                let idx = ((y as u32 * width + x as u32) * 4) as usize;

                // Alpha blend
                let alpha = color[3];
                let old_r = pixels[idx] as f32 / 255.0;
                let old_g = pixels[idx + 1] as f32 / 255.0;
                let old_b = pixels[idx + 2] as f32 / 255.0;

                pixels[idx] = ((old_r * (1.0 - alpha) + color[0] * alpha) * 255.0) as u8;
                pixels[idx + 1] = ((old_g * (1.0 - alpha) + color[1] * alpha) * 255.0) as u8;
                pixels[idx + 2] = ((old_b * (1.0 - alpha) + color[2] * alpha) * 255.0) as u8;
            }

            if x == x2 && y == y2 {
                break;
            }

            let e2 = 2 * err;
            if e2 > -dy {
                err -= dy;
                x += sx;
            }
            if e2 < dx {
                err += dx;
                y += sy;
            }
        }
    }

    /// Render binary/hex data overlay on screen edges
    fn render_data_overlay(&self, pixels: &mut [u8], width: u32, height: u32) {
        // Simple text rendering using colored bars to represent data
        let line_height = 12;
        let char_width = 6;
        let margin = 10;

        // Left side - binary data
        for (i, line) in self.config.overlay_data.iter().take(15).enumerate() {
            let y = margin + i as u32 * line_height;
            for (j, ch) in line.chars().take(24).enumerate() {
                let x = margin + j as u32 * char_width;

                // Draw a small colored rectangle for each character
                let brightness = if ch == '1' || ch.is_ascii_hexdigit() {
                    0.3 + (self.time * 2.0 + i as f32 + j as f32).sin().abs() * 0.2
                } else {
                    0.1
                };

                let color = [0.0, brightness * 0.8, brightness, 0.5];
                self.draw_char_rect(
                    pixels,
                    width,
                    height,
                    x as i32,
                    y as i32,
                    char_width - 1,
                    line_height - 2,
                    color,
                );
            }
        }

        // Right side - hex data
        // Check if there's enough space for the right-side overlay
        let right_side_width = margin + 24 * char_width;
        if width > right_side_width {
            for (i, line) in self
                .config
                .overlay_data
                .iter()
                .skip(15)
                .take(10)
                .enumerate()
            {
                let y = margin + i as u32 * line_height;
                for (j, ch) in line.chars().take(24).enumerate() {
                    let x = width - margin - 24 * char_width + j as u32 * char_width;

                    let brightness = if ch.is_ascii_hexdigit() {
                        0.3 + (self.time * 1.5 + i as f32 - j as f32).cos().abs() * 0.2
                    } else {
                        0.05
                    };

                    let color = [brightness * 0.5, 0.0, brightness, 0.5];
                    self.draw_char_rect(
                        pixels,
                        width,
                        height,
                        x as i32,
                        y as i32,
                        char_width - 1,
                        line_height - 2,
                        color,
                    );
                }
            }
        }
    }

    /// Draw a small rectangle for text representation
    fn draw_char_rect(
        &self,
        pixels: &mut [u8],
        width: u32,
        height: u32,
        x: i32,
        y: i32,
        w: u32,
        h: u32,
        color: [f32; 4],
    ) {
        for dy in 0..h {
            for dx in 0..w {
                let px = x + dx as i32;
                let py = y + dy as i32;

                if px < 0 || px >= width as i32 || py < 0 || py >= height as i32 {
                    continue;
                }

                let idx = ((py as u32 * width + px as u32) * 4) as usize;
                let alpha = color[3];

                let old_r = pixels[idx] as f32 / 255.0;
                let old_g = pixels[idx + 1] as f32 / 255.0;
                let old_b = pixels[idx + 2] as f32 / 255.0;

                pixels[idx] = ((old_r + color[0] * alpha).min(1.0) * 255.0) as u8;
                pixels[idx + 1] = ((old_g + color[1] * alpha).min(1.0) * 255.0) as u8;
                pixels[idx + 2] = ((old_b + color[2] * alpha).min(1.0) * 255.0) as u8;
            }
        }
    }
}

// ============================================================================
// Math Helpers
// ============================================================================

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len < 0.0001 {
        [0.0, 0.0, 1.0]
    } else {
        [v[0] / len, v[1] / len, v[2] / len]
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scene_from_architecture() {
        let scene = Neural3DScene::from_architecture(&[4, 8, 4, 2]);

        // Should have 4 + 8 + 4 + 2 = 18 nodes
        assert_eq!(scene.nodes.len(), 18);

        // Should have connections: 4*8 + 8*4 + 4*2 = 32 + 32 + 8 = 72
        assert_eq!(scene.connections.len(), 72);
    }

    #[test]
    fn test_node_activation() {
        let mut node = Node3D::default();
        assert_eq!(node.activation, 0.0);

        node.activation = 0.5;
        node.set_activation_color();

        // Glow should increase with activation
        assert!(node.glow_intensity > 0.5);
    }

    #[test]
    fn test_camera_orbit() {
        let mut camera = Camera3D::default();
        let initial_pos = camera.position;

        camera.orbit(0.1, 0.0);
        assert_ne!(camera.position[0], initial_pos[0]);
    }

    #[test]
    fn test_particle_system_update() {
        let mut system = ParticleSystem::default();
        assert!(system.particles.is_empty());

        // After update, particles should be spawned
        system.update(1.0);
        assert!(!system.particles.is_empty());
    }

    #[test]
    fn test_render_creates_pixels() {
        let scene = Neural3DScene::from_architecture(&[2, 3, 2]);
        let pixels = scene.render(100, 100);

        assert_eq!(pixels.len(), 100 * 100 * 4);

        // Should not be all black (background is dark blue)
        let has_color = pixels.chunks(4).any(|p| p[0] > 0 || p[1] > 0 || p[2] > 5);
        assert!(has_color);
    }

    #[test]
    fn test_force_layout() {
        let mut scene = Neural3DScene::from_architecture(&[3, 4, 3]);

        let initial_positions: Vec<[f32; 3]> = scene.nodes.iter().map(|n| n.position).collect();

        scene.apply_force_layout(10);

        // Y and Z positions should change (X is fixed for layers)
        let positions_changed =
            scene
                .nodes
                .iter()
                .zip(initial_positions.iter())
                .any(|(n, init)| {
                    (n.position[1] - init[1]).abs() > 0.01 || (n.position[2] - init[2]).abs() > 0.01
                });

        assert!(positions_changed);
    }

    #[test]
    fn test_forward_pass_animation() {
        let mut scene = Neural3DScene::from_architecture(&[2, 3, 1]);
        scene.animate_forward_pass(&[1.0, 0.5]);

        // First layer should be active
        assert!(scene.nodes[0].is_active);
        assert!(scene.nodes[1].is_active);
        assert_eq!(scene.forward_pass_layer, Some(0));
    }

    #[test]
    fn test_connection_particles() {
        let mut conn = Connection3D::new(0, 1, 0.8);
        conn.update_particles(0.5, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]);

        assert!(!conn.flow_particles.is_empty());
        // Particles should be along the connection
        for p in &conn.flow_particles {
            assert!(p.position[0] >= -0.1 && p.position[0] <= 1.1);
        }
    }

    #[test]
    fn test_math_helpers() {
        let a = [1.0, 0.0, 0.0];
        let b = [0.0, 1.0, 0.0];

        let c = cross(a, b);
        assert!((c[2] - 1.0).abs() < 0.001);

        let d = dot(a, b);
        assert!(d.abs() < 0.001);

        let n = normalize([3.0, 4.0, 0.0]);
        assert!((n[0] - 0.6).abs() < 0.001);
        assert!((n[1] - 0.8).abs() < 0.001);
    }
}
