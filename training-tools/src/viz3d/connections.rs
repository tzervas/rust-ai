//! Connection rendering between neural network layers.
//!
//! Renders connections as lines, curves, or flowing particles.

use bevy::prelude::*;

use super::{
    bevy_app::NetworkConfig, layers::NeuralLayer, AnimationPhase, AnimationState, GradientMagnitude,
};

/// Style of connection rendering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ConnectionStyle {
    /// Simple straight lines.
    #[default]
    Lines,
    /// Bezier curves.
    Curves,
    /// Flowing particles.
    Particles,
    /// Gradient flow (colored by gradient magnitude).
    GradientFlow,
}

/// Connection between two layers.
#[derive(Component)]
pub struct Connection {
    /// Source layer index.
    pub from_layer: usize,
    /// Target layer index.
    pub to_layer: usize,
    /// Connection strength/weight.
    pub weight: f32,
    /// Gradient magnitude for this connection.
    pub gradient: GradientMagnitude,
    /// Whether this connection is active in animation.
    pub is_active: bool,
    /// Connection style.
    pub style: ConnectionStyle,
}

/// Bundle for spawning connections.
#[derive(Bundle)]
pub struct ConnectionBundle {
    /// The connection component.
    pub connection: Connection,
    /// Transform for positioning.
    pub transform: Transform,
    /// Global transform.
    pub global_transform: GlobalTransform,
    /// Visibility.
    pub visibility: Visibility,
    /// Inherited visibility.
    pub inherited_visibility: InheritedVisibility,
    /// View visibility.
    pub view_visibility: ViewVisibility,
}

impl ConnectionBundle {
    /// Create a new connection bundle.
    pub fn new(from_layer: usize, to_layer: usize) -> Self {
        Self {
            connection: Connection {
                from_layer,
                to_layer,
                weight: 1.0,
                gradient: GradientMagnitude::default(),
                is_active: false,
                style: ConnectionStyle::Lines,
            },
            transform: Transform::IDENTITY,
            global_transform: GlobalTransform::IDENTITY,
            visibility: Visibility::Visible,
            inherited_visibility: InheritedVisibility::VISIBLE,
            view_visibility: ViewVisibility::default(),
        }
    }
}

/// Renderer for connections.
#[derive(Resource, Default)]
pub struct ConnectionRenderer {
    /// Default connection style.
    pub default_style: ConnectionStyle,
    /// Line width.
    pub line_width: f32,
    /// Particle count per connection.
    pub particle_count: usize,
    /// Particle speed.
    pub particle_speed: f32,
    /// Connection color.
    pub color: Color,
    /// Active connection color.
    pub active_color: Color,
}

impl ConnectionRenderer {
    /// Create a new connection renderer.
    pub fn new() -> Self {
        Self {
            default_style: ConnectionStyle::Lines,
            line_width: 2.0,
            particle_count: 10,
            particle_speed: 5.0,
            color: Color::srgba(0.5, 0.5, 0.6, 0.3),
            active_color: Color::srgba(1.0, 0.8, 0.2, 0.8),
        }
    }
}

/// Particle for flowing animation.
#[derive(Component)]
pub struct FlowParticle {
    /// Connection this particle belongs to.
    pub connection_index: usize,
    /// Progress along the connection (0.0 to 1.0).
    pub progress: f32,
    /// Particle speed.
    pub speed: f32,
    /// Direction (true = forward, false = backward).
    pub forward: bool,
}

/// Plugin for connection systems.
pub struct ConnectionPlugin;

impl Plugin for ConnectionPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ConnectionRenderer>()
            .add_systems(Startup, spawn_connections)
            .add_systems(
                Update,
                (
                    update_connection_animation,
                    draw_connections_gizmos,
                    update_flow_particles,
                ),
            );
    }
}

/// Spawn connection entities.
fn spawn_connections(mut commands: Commands, config: Res<NetworkConfig>) {
    if !config.show_connections {
        return;
    }

    // Create connections between adjacent layers
    for i in 0..config.layer_sizes.len().saturating_sub(1) {
        commands.spawn(ConnectionBundle::new(i, i + 1));
    }

    info!(
        "Spawned {} connections",
        config.layer_sizes.len().saturating_sub(1)
    );
}

/// Update connection animation state.
fn update_connection_animation(
    animation_state: Res<AnimationState>,
    mut connections: Query<&mut Connection>,
) {
    for mut connection in connections.iter_mut() {
        let is_active = match animation_state.phase {
            AnimationPhase::Forward => {
                connection.from_layer == animation_state.current_layer
                    || connection.to_layer == animation_state.current_layer
            }
            AnimationPhase::Backward => {
                connection.from_layer == animation_state.current_layer
                    || connection.to_layer == animation_state.current_layer
            }
            AnimationPhase::Training => {
                connection.from_layer == animation_state.current_layer
                    || connection.to_layer == animation_state.current_layer
            }
            AnimationPhase::Idle => false,
        };

        connection.is_active = is_active;
    }
}

/// Draw connections using gizmos.
fn draw_connections_gizmos(
    mut gizmos: Gizmos,
    connections: Query<&Connection>,
    layers: Query<(&NeuralLayer, &Transform)>,
    renderer: Res<ConnectionRenderer>,
    animation_state: Res<AnimationState>,
) {
    // Build layer position lookup
    let layer_positions: std::collections::HashMap<usize, Vec3> = layers
        .iter()
        .map(|(layer, transform)| (layer.index, transform.translation))
        .collect();

    for connection in connections.iter() {
        let Some(from_pos) = layer_positions.get(&connection.from_layer) else {
            continue;
        };
        let Some(to_pos) = layer_positions.get(&connection.to_layer) else {
            continue;
        };

        let color = if connection.is_active {
            renderer.active_color
        } else if connection.gradient.value > 0.01 {
            let grad_color = connection.gradient.to_color();
            Color::srgba(grad_color[0], grad_color[1], grad_color[2], 0.5)
        } else {
            renderer.color
        };

        match connection.style {
            ConnectionStyle::Lines => {
                gizmos.line(*from_pos, *to_pos, color);
            }
            ConnectionStyle::Curves => {
                // Draw a bezier curve
                let mid = (*from_pos + *to_pos) / 2.0;
                let control = mid + Vec3::Y * 3.0;

                // Approximate bezier with line segments
                let segments = 20;
                for i in 0..segments {
                    let t1 = i as f32 / segments as f32;
                    let t2 = (i + 1) as f32 / segments as f32;

                    let p1 = bezier_point(*from_pos, control, *to_pos, t1);
                    let p2 = bezier_point(*from_pos, control, *to_pos, t2);

                    gizmos.line(p1, p2, color);
                }
            }
            ConnectionStyle::Particles | ConnectionStyle::GradientFlow => {
                // Particles are handled separately
                gizmos.line(*from_pos, *to_pos, color.with_alpha(0.1));

                // Draw animated particles
                if connection.is_active {
                    let progress = animation_state.progress;
                    let particle_pos = from_pos.lerp(*to_pos, progress);
                    gizmos.sphere(Isometry3d::from_translation(particle_pos), 0.3, color);
                }
            }
        }
    }
}

/// Calculate point on quadratic bezier curve.
fn bezier_point(p0: Vec3, p1: Vec3, p2: Vec3, t: f32) -> Vec3 {
    let t_inv = 1.0 - t;
    p0 * t_inv * t_inv + p1 * 2.0 * t_inv * t + p2 * t * t
}

/// Update flow particle positions.
fn update_flow_particles(
    time: Res<Time>,
    animation_state: Res<AnimationState>,
    mut particles: Query<(&mut FlowParticle, &mut Transform)>,
    connections: Query<&Connection>,
    layers: Query<(&NeuralLayer, &Transform), Without<FlowParticle>>,
) {
    if animation_state.phase == AnimationPhase::Idle {
        return;
    }

    // Build layer position lookup
    let layer_positions: std::collections::HashMap<usize, Vec3> = layers
        .iter()
        .map(|(layer, transform)| (layer.index, transform.translation))
        .collect();

    for (mut particle, mut transform) in particles.iter_mut() {
        // Update progress
        particle.progress += time.delta_secs() * particle.speed;

        if particle.progress > 1.0 {
            particle.progress = 0.0;
            particle.forward = !particle.forward; // Reverse direction
        }

        // Find the connection and update position
        for connection in connections.iter() {
            if connection.from_layer == particle.connection_index
                || connection.to_layer == particle.connection_index
            {
                let Some(from_pos) = layer_positions.get(&connection.from_layer) else {
                    continue;
                };
                let Some(to_pos) = layer_positions.get(&connection.to_layer) else {
                    continue;
                };

                let progress = if particle.forward {
                    particle.progress
                } else {
                    1.0 - particle.progress
                };

                transform.translation = from_pos.lerp(*to_pos, progress);
                break;
            }
        }
    }
}
