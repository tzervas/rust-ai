//! Bevy application setup for 3D neural network visualization.

use bevy::prelude::*;
use std::sync::{Arc, Mutex};

use super::{
    camera::CameraPlugin, connections::ConnectionPlugin, layers::LayerPlugin,
    ui_overlay::UiOverlayPlugin, AnimationPhase, TrainingDataPoint,
};

/// Configuration for the neural network visualization.
#[derive(Debug, Clone, Resource)]
pub struct NetworkConfig {
    /// Layer sizes (number of neurons per layer).
    pub layer_sizes: Vec<usize>,
    /// Layer names (optional, defaults to "Layer N").
    pub layer_names: Vec<String>,
    /// Whether to show connections between all layers.
    pub show_connections: bool,
    /// Whether to animate forward/backward passes.
    pub animate_passes: bool,
    /// Animation speed multiplier.
    pub animation_speed: f32,
    /// Background color.
    pub background_color: Color,
    /// Whether this is a transformer model (enables attention viz).
    pub is_transformer: bool,
    /// Number of attention heads (for transformers).
    pub num_attention_heads: usize,
    /// Hidden dimension size.
    pub hidden_dim: usize,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            layer_sizes: vec![768, 3072, 3072, 768], // Typical transformer FFN
            layer_names: vec![
                "Input".to_string(),
                "FFN Up".to_string(),
                "FFN Down".to_string(),
                "Output".to_string(),
            ],
            show_connections: true,
            animate_passes: true,
            animation_speed: 1.0,
            background_color: Color::srgb(0.1, 0.1, 0.15),
            is_transformer: true,
            num_attention_heads: 12,
            hidden_dim: 768,
        }
    }
}

impl NetworkConfig {
    /// Create a transformer configuration.
    pub fn transformer(num_layers: usize, hidden_dim: usize, num_heads: usize) -> Self {
        let mut layer_sizes = Vec::new();
        let mut layer_names = Vec::new();

        // Input embedding
        layer_sizes.push(hidden_dim);
        layer_names.push("Embedding".to_string());

        // Transformer blocks
        for i in 0..num_layers {
            // Attention
            layer_sizes.push(hidden_dim);
            layer_names.push(format!("Attn {}", i + 1));

            // FFN
            layer_sizes.push(hidden_dim * 4);
            layer_names.push(format!("FFN {} Up", i + 1));
            layer_sizes.push(hidden_dim);
            layer_names.push(format!("FFN {} Down", i + 1));
        }

        // Output
        layer_sizes.push(hidden_dim);
        layer_names.push("Output".to_string());

        Self {
            layer_sizes,
            layer_names,
            is_transformer: true,
            num_attention_heads: num_heads,
            hidden_dim,
            ..Default::default()
        }
    }

    /// Create an MLP configuration.
    pub fn mlp(layer_sizes: Vec<usize>) -> Self {
        let layer_names = layer_sizes
            .iter()
            .enumerate()
            .map(|(i, _)| {
                if i == 0 {
                    "Input".to_string()
                } else if i == layer_sizes.len() - 1 {
                    "Output".to_string()
                } else {
                    format!("Hidden {}", i)
                }
            })
            .collect();

        Self {
            layer_sizes,
            layer_names,
            is_transformer: false,
            ..Default::default()
        }
    }
}

/// Shared state for real-time training data.
#[derive(Resource, Default)]
pub struct TrainingDataStream {
    /// Latest training data point.
    pub latest: Arc<Mutex<Option<TrainingDataPoint>>>,
    /// History of training data points.
    pub history: Arc<Mutex<Vec<TrainingDataPoint>>>,
    /// Maximum history size.
    pub max_history: usize,
}

impl TrainingDataStream {
    /// Create a new training data stream.
    pub fn new(max_history: usize) -> Self {
        Self {
            latest: Arc::new(Mutex::new(None)),
            history: Arc::new(Mutex::new(Vec::with_capacity(max_history))),
            max_history,
        }
    }

    /// Push a new data point.
    pub fn push(&self, data: TrainingDataPoint) {
        if let Ok(mut latest) = self.latest.lock() {
            *latest = Some(data.clone());
        }
        if let Ok(mut history) = self.history.lock() {
            if history.len() >= self.max_history {
                history.remove(0);
            }
            history.push(data);
        }
    }

    /// Get the latest data point.
    pub fn get_latest(&self) -> Option<TrainingDataPoint> {
        self.latest.lock().ok()?.clone()
    }
}

/// Current animation state.
#[derive(Resource, Default)]
pub struct AnimationState {
    /// Current animation phase.
    pub phase: AnimationPhase,
    /// Animation progress (0.0 to 1.0).
    pub progress: f32,
    /// Current layer being animated.
    pub current_layer: usize,
    /// Whether animation is paused.
    pub paused: bool,
}

/// Currently selected layer for highlighting.
#[derive(Resource, Default)]
pub struct SelectedLayer {
    /// Index of the selected layer (None if no selection).
    pub index: Option<usize>,
}

/// Main Bevy plugin for 3D visualization.
pub struct Viz3dPlugin;

impl Plugin for Viz3dPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<NetworkConfig>()
            .init_resource::<TrainingDataStream>()
            .init_resource::<AnimationState>()
            .init_resource::<SelectedLayer>()
            .add_plugins((CameraPlugin, LayerPlugin, ConnectionPlugin, UiOverlayPlugin))
            .add_systems(Startup, setup_scene)
            .add_systems(Update, (animate_training, handle_keyboard_input));
    }
}

/// Main visualization application.
pub struct Viz3dApp;

impl Viz3dApp {
    /// Run the visualization application with the given configuration.
    pub fn run(config: NetworkConfig) {
        App::new()
            .add_plugins(DefaultPlugins.set(WindowPlugin {
                primary_window: Some(Window {
                    title: "Neural Network 3D Viewer".to_string(),
                    resolution: (1280.0, 720.0).into(),
                    ..default()
                }),
                ..default()
            }))
            .insert_resource(config)
            .add_plugins(Viz3dPlugin)
            .run();
    }

    /// Run with default configuration.
    pub fn run_default() {
        Self::run(NetworkConfig::default());
    }
}

/// Setup the initial scene.
fn setup_scene(mut commands: Commands, config: Res<NetworkConfig>) {
    // Ambient light
    commands.insert_resource(AmbientLight {
        color: Color::WHITE,
        brightness: 300.0,
    });

    // Directional light for shadows
    commands.spawn((
        DirectionalLight {
            illuminance: 10000.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(10.0, 20.0, 10.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    // Point light for better depth perception
    commands.spawn((
        PointLight {
            color: Color::srgb(0.8, 0.8, 1.0),
            intensity: 500_000.0,
            range: 100.0,
            ..default()
        },
        Transform::from_xyz(-10.0, 10.0, -10.0),
    ));

    // Clear color
    commands.insert_resource(ClearColor(config.background_color));

    info!(
        "Scene setup complete with {} layers",
        config.layer_sizes.len()
    );
}

/// Animate training forward/backward passes.
fn animate_training(
    time: Res<Time>,
    config: Res<NetworkConfig>,
    mut state: ResMut<AnimationState>,
) {
    if state.paused || state.phase == AnimationPhase::Idle {
        return;
    }

    let speed = config.animation_speed;
    state.progress += time.delta_secs() * speed * 0.5;

    if state.progress >= 1.0 {
        state.progress = 0.0;
        state.current_layer += 1;

        let num_layers = config.layer_sizes.len();

        match state.phase {
            AnimationPhase::Forward => {
                if state.current_layer >= num_layers {
                    state.current_layer = num_layers.saturating_sub(1);
                    state.phase = AnimationPhase::Backward;
                }
            }
            AnimationPhase::Backward => {
                if state.current_layer == 0 || state.current_layer >= num_layers {
                    state.current_layer = 0;
                    state.phase = AnimationPhase::Forward;
                }
            }
            AnimationPhase::Training => {
                if state.current_layer >= num_layers {
                    state.current_layer = 0;
                }
            }
            AnimationPhase::Idle => {}
        }
    }
}

/// Handle keyboard input for controls.
fn handle_keyboard_input(
    keys: Res<ButtonInput<KeyCode>>,
    mut state: ResMut<AnimationState>,
    mut selected: ResMut<SelectedLayer>,
    config: Res<NetworkConfig>,
) {
    // Toggle pause
    if keys.just_pressed(KeyCode::Space) {
        state.paused = !state.paused;
    }

    // Start forward animation
    if keys.just_pressed(KeyCode::KeyF) {
        state.phase = AnimationPhase::Forward;
        state.current_layer = 0;
        state.progress = 0.0;
        state.paused = false;
    }

    // Start backward animation
    if keys.just_pressed(KeyCode::KeyB) {
        state.phase = AnimationPhase::Backward;
        state.current_layer = config.layer_sizes.len().saturating_sub(1);
        state.progress = 0.0;
        state.paused = false;
    }

    // Start training animation
    if keys.just_pressed(KeyCode::KeyT) {
        state.phase = AnimationPhase::Training;
        state.current_layer = 0;
        state.progress = 0.0;
        state.paused = false;
    }

    // Stop animation
    if keys.just_pressed(KeyCode::Escape) {
        state.phase = AnimationPhase::Idle;
        state.paused = true;
    }

    // Layer selection with number keys
    for (i, key) in [
        KeyCode::Digit1,
        KeyCode::Digit2,
        KeyCode::Digit3,
        KeyCode::Digit4,
        KeyCode::Digit5,
        KeyCode::Digit6,
        KeyCode::Digit7,
        KeyCode::Digit8,
        KeyCode::Digit9,
    ]
    .iter()
    .enumerate()
    {
        if keys.just_pressed(*key) {
            if i < config.layer_sizes.len() {
                selected.index = Some(i);
            }
        }
    }

    // Clear selection
    if keys.just_pressed(KeyCode::Digit0) {
        selected.index = None;
    }

    // Navigate layers
    if keys.just_pressed(KeyCode::ArrowUp) || keys.just_pressed(KeyCode::KeyK) {
        if let Some(idx) = selected.index {
            if idx + 1 < config.layer_sizes.len() {
                selected.index = Some(idx + 1);
            }
        } else {
            selected.index = Some(0);
        }
    }

    if keys.just_pressed(KeyCode::ArrowDown) || keys.just_pressed(KeyCode::KeyJ) {
        if let Some(idx) = selected.index {
            if idx > 0 {
                selected.index = Some(idx - 1);
            }
        } else {
            selected.index = Some(config.layer_sizes.len().saturating_sub(1));
        }
    }
}
