//! UI overlay for controls and information display.
//!
//! Uses Bevy UI for control panels and information overlays.

use bevy::prelude::*;

use super::{
    bevy_app::{NetworkConfig, TrainingDataStream},
    layers::NeuralLayer,
    AnimationPhase, AnimationState, SelectedLayer,
};

/// UI state resource.
#[derive(Resource, Default)]
pub struct UiState {
    /// Whether the control panel is visible.
    pub show_controls: bool,
    /// Whether layer info is visible.
    pub show_layer_info: bool,
    /// Whether training stats are visible.
    pub show_training_stats: bool,
    /// Whether help is visible.
    pub show_help: bool,
}

/// Control panel marker.
#[derive(Component)]
pub struct ControlPanel;

/// Layer info panel marker.
#[derive(Component)]
pub struct LayerInfoPanel;

/// Training stats panel marker.
#[derive(Component)]
pub struct TrainingStatsPanel;

/// Help panel marker.
#[derive(Component)]
pub struct HelpPanel;

/// UI overlay component.
#[derive(Component)]
pub struct UiOverlay;

/// Text component for dynamic updates.
#[derive(Component)]
pub struct DynamicText {
    /// Text identifier.
    pub id: String,
}

/// Plugin for UI overlay systems.
pub struct UiOverlayPlugin;

impl Plugin for UiOverlayPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<UiState>()
            .add_systems(Startup, setup_ui)
            .add_systems(
                Update,
                (
                    toggle_ui_visibility,
                    update_animation_status,
                    update_layer_info,
                    update_training_stats,
                ),
            );
    }
}

/// Setup UI elements.
fn setup_ui(mut commands: Commands, mut ui_state: ResMut<UiState>) {
    ui_state.show_controls = true;
    ui_state.show_help = true;

    // Root UI node
    commands
        .spawn((
            Node {
                width: Val::Percent(100.0),
                height: Val::Percent(100.0),
                flex_direction: FlexDirection::Column,
                justify_content: JustifyContent::SpaceBetween,
                ..default()
            },
            UiOverlay,
        ))
        .with_children(|parent| {
            // Top bar with title and controls
            parent
                .spawn(Node {
                    width: Val::Percent(100.0),
                    height: Val::Px(40.0),
                    flex_direction: FlexDirection::Row,
                    justify_content: JustifyContent::SpaceBetween,
                    align_items: AlignItems::Center,
                    padding: UiRect::all(Val::Px(10.0)),
                    ..default()
                })
                .insert(BackgroundColor(Color::srgba(0.1, 0.1, 0.15, 0.8)))
                .with_children(|parent| {
                    // Title
                    parent.spawn((
                        Text::new("Neural Network 3D Viewer"),
                        TextFont {
                            font_size: 20.0,
                            ..default()
                        },
                        TextColor(Color::WHITE),
                    ));

                    // Animation status
                    parent.spawn((
                        Text::new("Idle"),
                        TextFont {
                            font_size: 16.0,
                            ..default()
                        },
                        TextColor(Color::srgb(0.7, 0.7, 0.7)),
                        DynamicText {
                            id: "animation_status".to_string(),
                        },
                    ));
                });

            // Main content area
            parent
                .spawn(Node {
                    width: Val::Percent(100.0),
                    flex_grow: 1.0,
                    flex_direction: FlexDirection::Row,
                    justify_content: JustifyContent::SpaceBetween,
                    ..default()
                })
                .with_children(|parent| {
                    // Left panel - Layer info
                    parent
                        .spawn((
                            Node {
                                width: Val::Px(250.0),
                                height: Val::Auto,
                                flex_direction: FlexDirection::Column,
                                padding: UiRect::all(Val::Px(10.0)),
                                margin: UiRect::all(Val::Px(10.0)),
                                ..default()
                            },
                            BackgroundColor(Color::srgba(0.1, 0.1, 0.15, 0.7)),
                            LayerInfoPanel,
                        ))
                        .with_children(|parent| {
                            parent.spawn((
                                Text::new("Layer Info"),
                                TextFont {
                                    font_size: 16.0,
                                    ..default()
                                },
                                TextColor(Color::srgb(0.8, 0.8, 0.2)),
                            ));

                            parent.spawn((
                                Text::new("Select a layer (1-9)"),
                                TextFont {
                                    font_size: 14.0,
                                    ..default()
                                },
                                TextColor(Color::srgb(0.7, 0.7, 0.7)),
                                DynamicText {
                                    id: "layer_info".to_string(),
                                },
                            ));
                        });

                    // Right panel - Training stats
                    parent
                        .spawn((
                            Node {
                                width: Val::Px(250.0),
                                height: Val::Auto,
                                flex_direction: FlexDirection::Column,
                                padding: UiRect::all(Val::Px(10.0)),
                                margin: UiRect::all(Val::Px(10.0)),
                                ..default()
                            },
                            BackgroundColor(Color::srgba(0.1, 0.1, 0.15, 0.7)),
                            TrainingStatsPanel,
                        ))
                        .with_children(|parent| {
                            parent.spawn((
                                Text::new("Training Stats"),
                                TextFont {
                                    font_size: 16.0,
                                    ..default()
                                },
                                TextColor(Color::srgb(0.2, 0.8, 0.4)),
                            ));

                            parent.spawn((
                                Text::new("No training data"),
                                TextFont {
                                    font_size: 14.0,
                                    ..default()
                                },
                                TextColor(Color::srgb(0.7, 0.7, 0.7)),
                                DynamicText {
                                    id: "training_stats".to_string(),
                                },
                            ));
                        });
                });

            // Bottom bar with help
            parent
                .spawn((
                    Node {
                        width: Val::Percent(100.0),
                        height: Val::Auto,
                        flex_direction: FlexDirection::Column,
                        padding: UiRect::all(Val::Px(10.0)),
                        ..default()
                    },
                    BackgroundColor(Color::srgba(0.1, 0.1, 0.15, 0.8)),
                    HelpPanel,
                ))
                .with_children(|parent| {
                    parent.spawn((
                        Text::new(concat!(
                            "Controls: ",
                            "Right-click+drag=Rotate | ",
                            "Left-click+drag=Pan | ",
                            "Scroll=Zoom | ",
                            "1-9=Select layer | ",
                            "F=Forward | ",
                            "B=Backward | ",
                            "T=Training | ",
                            "Space=Pause | ",
                            "R=Reset | ",
                            "H=Toggle help"
                        )),
                        TextFont {
                            font_size: 12.0,
                            ..default()
                        },
                        TextColor(Color::srgb(0.6, 0.6, 0.6)),
                    ));
                });
        });

    info!("UI setup complete");
}

/// Toggle UI panel visibility.
fn toggle_ui_visibility(
    keys: Res<ButtonInput<KeyCode>>,
    mut ui_state: ResMut<UiState>,
    mut help_panels: Query<&mut Visibility, With<HelpPanel>>,
    mut layer_panels: Query<&mut Visibility, (With<LayerInfoPanel>, Without<HelpPanel>)>,
    mut stats_panels: Query<
        &mut Visibility,
        (
            With<TrainingStatsPanel>,
            Without<HelpPanel>,
            Without<LayerInfoPanel>,
        ),
    >,
) {
    // Toggle help
    if keys.just_pressed(KeyCode::KeyH) {
        ui_state.show_help = !ui_state.show_help;
        for mut visibility in help_panels.iter_mut() {
            *visibility = if ui_state.show_help {
                Visibility::Visible
            } else {
                Visibility::Hidden
            };
        }
    }

    // Toggle layer info
    if keys.just_pressed(KeyCode::KeyI) {
        ui_state.show_layer_info = !ui_state.show_layer_info;
        for mut visibility in layer_panels.iter_mut() {
            *visibility = if ui_state.show_layer_info {
                Visibility::Visible
            } else {
                Visibility::Hidden
            };
        }
    }

    // Toggle training stats
    if keys.just_pressed(KeyCode::KeyS) && !keys.pressed(KeyCode::ControlLeft) {
        ui_state.show_training_stats = !ui_state.show_training_stats;
        for mut visibility in stats_panels.iter_mut() {
            *visibility = if ui_state.show_training_stats {
                Visibility::Visible
            } else {
                Visibility::Hidden
            };
        }
    }
}

/// Update animation status text.
fn update_animation_status(
    animation_state: Res<AnimationState>,
    config: Res<NetworkConfig>,
    mut texts: Query<(&mut Text, &DynamicText)>,
) {
    for (mut text, dynamic) in texts.iter_mut() {
        if dynamic.id == "animation_status" {
            let status = match animation_state.phase {
                AnimationPhase::Idle => "Idle".to_string(),
                AnimationPhase::Forward => format!(
                    "Forward Pass: Layer {}/{}{}",
                    animation_state.current_layer + 1,
                    config.layer_sizes.len(),
                    if animation_state.paused {
                        " (Paused)"
                    } else {
                        ""
                    }
                ),
                AnimationPhase::Backward => format!(
                    "Backward Pass: Layer {}/{}{}",
                    animation_state.current_layer + 1,
                    config.layer_sizes.len(),
                    if animation_state.paused {
                        " (Paused)"
                    } else {
                        ""
                    }
                ),
                AnimationPhase::Training => format!(
                    "Training: Layer {}/{}{}",
                    animation_state.current_layer + 1,
                    config.layer_sizes.len(),
                    if animation_state.paused {
                        " (Paused)"
                    } else {
                        ""
                    }
                ),
            };
            text.0 = status;
        }
    }
}

/// Update layer info panel.
fn update_layer_info(
    selected: Res<SelectedLayer>,
    layers: Query<&NeuralLayer>,
    mut texts: Query<(&mut Text, &DynamicText)>,
) {
    for (mut text, dynamic) in texts.iter_mut() {
        if dynamic.id == "layer_info" {
            if let Some(idx) = selected.index {
                // Find the selected layer
                let layer_info = layers
                    .iter()
                    .find(|l| l.index == idx)
                    .map(|layer| {
                        format!(
                            "Name: {}\nIndex: {}\nSize: {}\nShape: {:?}\nGradient: {:.4}\nActive: {}",
                            layer.name,
                            layer.index,
                            layer.size,
                            layer.shape,
                            layer.gradient.value,
                            if layer.is_active { "Yes" } else { "No" }
                        )
                    })
                    .unwrap_or_else(|| "Layer not found".to_string());

                text.0 = layer_info;
            } else {
                text.0 = "Select a layer (1-9)\nor use J/K to navigate".to_string();
            }
        }
    }
}

/// Update training stats panel.
fn update_training_stats(
    training_data: Res<TrainingDataStream>,
    mut texts: Query<(&mut Text, &DynamicText)>,
) {
    for (mut text, dynamic) in texts.iter_mut() {
        if dynamic.id == "training_stats" {
            if let Some(data) = training_data.get_latest() {
                let avg_gradient = if data.gradient_norms.is_empty() {
                    0.0
                } else {
                    data.gradient_norms.iter().sum::<f32>() / data.gradient_norms.len() as f32
                };

                let stats = format!(
                    "Step: {}\nLoss: {:.6}\nAvg Gradient: {:.6}\nLayers: {}",
                    data.step,
                    data.loss,
                    avg_gradient,
                    data.gradient_norms.len()
                );
                text.0 = stats;
            } else {
                text.0 = "No training data\n\nConnect training to\nstream live updates".to_string();
            }
        }
    }
}
