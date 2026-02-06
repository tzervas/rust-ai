//! Layer mesh generation for neural network visualization.
//!
//! Creates 3D representations of neural network layers.

use bevy::prelude::*;

use super::{bevy_app::NetworkConfig, AnimationState, GradientMagnitude, SelectedLayer};

/// Shape of the layer mesh.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LayerShape {
    /// Box/cuboid shape (good for dense layers).
    #[default]
    Box,
    /// Sphere (good for embedding layers).
    Sphere,
    /// Cylinder (good for attention layers).
    Cylinder,
    /// Flat plane (good for input/output).
    Plane,
    /// Torus (good for recurrent layers).
    Torus,
}

/// Style configuration for layer rendering.
#[derive(Debug, Clone)]
pub struct LayerStyle {
    /// Base color of the layer.
    pub base_color: Color,
    /// Highlight color when selected.
    pub highlight_color: Color,
    /// Transparency (0 = opaque, 1 = invisible).
    pub alpha: f32,
    /// Whether to show wireframe.
    pub wireframe: bool,
    /// Emission intensity for glow effect.
    pub emission: f32,
}

impl Default for LayerStyle {
    fn default() -> Self {
        Self {
            base_color: Color::srgb(0.2, 0.5, 0.8),
            highlight_color: Color::srgb(1.0, 0.8, 0.2),
            alpha: 0.8,
            wireframe: false,
            emission: 0.0,
        }
    }
}

/// Component for neural network layers.
#[derive(Component)]
pub struct NeuralLayer {
    /// Layer index in the network.
    pub index: usize,
    /// Layer name.
    pub name: String,
    /// Number of neurons/features in this layer.
    pub size: usize,
    /// Layer shape.
    pub shape: LayerShape,
    /// Layer style.
    pub style: LayerStyle,
    /// Current gradient magnitude (for color coding).
    pub gradient: GradientMagnitude,
    /// Whether this layer is currently active in animation.
    pub is_active: bool,
}

/// Mesh data for a neural layer.
#[derive(Debug, Clone)]
pub struct LayerMesh {
    /// Position in 3D space.
    pub position: Vec3,
    /// Scale of the mesh.
    pub scale: Vec3,
    /// Rotation.
    pub rotation: Quat,
}

impl Default for LayerMesh {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            scale: Vec3::ONE,
            rotation: Quat::IDENTITY,
        }
    }
}

/// Builder for creating layer meshes.
pub struct LayerMeshBuilder {
    /// Network configuration.
    config: NetworkConfig,
    /// Spacing between layers.
    layer_spacing: f32,
    /// Maximum width for layer visualization.
    max_width: f32,
    /// Height scaling factor.
    height_scale: f32,
}

impl LayerMeshBuilder {
    /// Create a new builder.
    pub fn new(config: NetworkConfig) -> Self {
        Self {
            config,
            layer_spacing: 5.0,
            max_width: 20.0,
            height_scale: 0.01,
        }
    }

    /// Set spacing between layers.
    pub fn with_spacing(mut self, spacing: f32) -> Self {
        self.layer_spacing = spacing;
        self
    }

    /// Set maximum width.
    pub fn with_max_width(mut self, width: f32) -> Self {
        self.max_width = width;
        self
    }

    /// Calculate position for a layer.
    pub fn calculate_position(&self, layer_index: usize) -> Vec3 {
        let num_layers = self.config.layer_sizes.len();
        let total_depth = (num_layers - 1) as f32 * self.layer_spacing;
        let z = layer_index as f32 * self.layer_spacing - total_depth / 2.0;
        Vec3::new(0.0, 0.0, z)
    }

    /// Calculate scale for a layer based on its size.
    pub fn calculate_scale(&self, layer_size: usize) -> Vec3 {
        let max_size = self.config.layer_sizes.iter().max().copied().unwrap_or(1) as f32;
        let relative_size = (layer_size as f32 / max_size).sqrt();

        let width = relative_size * self.max_width;
        let height = (layer_size as f32 * self.height_scale).max(0.5);
        let depth = 1.0;

        Vec3::new(width, height, depth)
    }

    /// Determine appropriate shape for a layer.
    pub fn determine_shape(&self, layer_index: usize, layer_name: &str) -> LayerShape {
        let name_lower = layer_name.to_lowercase();

        if name_lower.contains("embed") {
            LayerShape::Sphere
        } else if name_lower.contains("attn") || name_lower.contains("attention") {
            LayerShape::Cylinder
        } else if layer_index == 0 {
            LayerShape::Plane
        } else if layer_index == self.config.layer_sizes.len() - 1 {
            LayerShape::Plane
        } else {
            LayerShape::Box
        }
    }

    /// Determine color for a layer.
    pub fn determine_color(&self, layer_index: usize, layer_name: &str) -> Color {
        let name_lower = layer_name.to_lowercase();

        if name_lower.contains("embed") {
            Color::srgb(0.3, 0.7, 0.9) // Cyan
        } else if name_lower.contains("attn") || name_lower.contains("attention") {
            Color::srgb(0.9, 0.5, 0.3) // Orange
        } else if name_lower.contains("ffn") || name_lower.contains("mlp") {
            Color::srgb(0.5, 0.8, 0.4) // Green
        } else if layer_index == 0 {
            Color::srgb(0.4, 0.4, 0.9) // Blue
        } else if layer_index == self.config.layer_sizes.len() - 1 {
            Color::srgb(0.9, 0.4, 0.4) // Red
        } else {
            Color::srgb(0.6, 0.6, 0.7) // Gray
        }
    }

    /// Build all layer meshes.
    pub fn build_all(&self) -> Vec<(LayerMesh, NeuralLayer)> {
        self.config
            .layer_sizes
            .iter()
            .enumerate()
            .map(|(i, &size)| {
                let name = self
                    .config
                    .layer_names
                    .get(i)
                    .cloned()
                    .unwrap_or_else(|| format!("Layer {}", i));

                let shape = self.determine_shape(i, &name);
                let color = self.determine_color(i, &name);

                let mesh = LayerMesh {
                    position: self.calculate_position(i),
                    scale: self.calculate_scale(size),
                    rotation: Quat::IDENTITY,
                };

                let layer = NeuralLayer {
                    index: i,
                    name,
                    size,
                    shape,
                    style: LayerStyle {
                        base_color: color,
                        ..Default::default()
                    },
                    gradient: GradientMagnitude::default(),
                    is_active: false,
                };

                (mesh, layer)
            })
            .collect()
    }
}

/// Plugin for layer systems.
pub struct LayerPlugin;

impl Plugin for LayerPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, spawn_layers)
            .add_systems(Update, (update_layer_colors, update_layer_animation));
    }
}

/// Spawn layer meshes.
fn spawn_layers(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    config: Res<NetworkConfig>,
) {
    let builder = LayerMeshBuilder::new((*config).clone());
    let layers = builder.build_all();

    for (mesh_data, layer) in layers {
        let mesh = match layer.shape {
            LayerShape::Box => Mesh::from(Cuboid::new(1.0, 1.0, 1.0)),
            LayerShape::Sphere => Mesh::from(Sphere::new(0.5)),
            LayerShape::Cylinder => Mesh::from(Cylinder::new(0.5, 1.0)),
            LayerShape::Plane => Mesh::from(Plane3d::new(Vec3::Y, Vec2::new(1.0, 1.0))),
            LayerShape::Torus => Mesh::from(Torus::new(0.3, 0.5)),
        };

        let material = StandardMaterial {
            base_color: layer.style.base_color.with_alpha(layer.style.alpha),
            alpha_mode: AlphaMode::Blend,
            perceptual_roughness: 0.5,
            metallic: 0.3,
            ..default()
        };

        commands.spawn((
            Mesh3d(meshes.add(mesh)),
            MeshMaterial3d(materials.add(material)),
            Transform::from_translation(mesh_data.position)
                .with_scale(mesh_data.scale)
                .with_rotation(mesh_data.rotation),
            layer,
        ));
    }

    info!("Spawned {} layers", config.layer_sizes.len());
}

/// Update layer colors based on selection and gradients.
fn update_layer_colors(
    selected: Res<SelectedLayer>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    query: Query<(&NeuralLayer, &MeshMaterial3d<StandardMaterial>)>,
) {
    for (layer, material_handle) in query.iter() {
        if let Some(material) = materials.get_mut(&material_handle.0) {
            let is_selected = selected.index == Some(layer.index);

            let base_color = if is_selected {
                layer.style.highlight_color
            } else if layer.gradient.value > 0.01 {
                let grad_color = layer.gradient.to_color();
                Color::srgb(grad_color[0], grad_color[1], grad_color[2])
            } else {
                layer.style.base_color
            };

            material.base_color = base_color.with_alpha(layer.style.alpha);

            // Add emission for active/selected layers
            if is_selected || layer.is_active {
                material.emissive = LinearRgba::rgb(
                    base_color.to_linear().red * 0.3,
                    base_color.to_linear().green * 0.3,
                    base_color.to_linear().blue * 0.3,
                );
            } else {
                material.emissive = LinearRgba::BLACK;
            }
        }
    }
}

/// Update layer animation state.
fn update_layer_animation(
    animation_state: Res<AnimationState>,
    mut query: Query<&mut NeuralLayer>,
) {
    for mut layer in query.iter_mut() {
        layer.is_active = layer.index == animation_state.current_layer;
    }
}
