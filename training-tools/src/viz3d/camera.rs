//! Camera controls for 3D visualization.
//!
//! Provides orbit, pan, and zoom functionality for exploring the neural network.

use bevy::input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll};
use bevy::prelude::*;
use std::f32::consts::PI;

/// Camera control mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CameraMode {
    /// Orbit around a focal point.
    #[default]
    Orbit,
    /// Free camera (WASD + mouse look).
    Free,
    /// Pan mode (drag to move).
    Pan,
    /// Follow a specific layer.
    Follow,
}

/// Orbit camera controller component.
#[derive(Component)]
pub struct OrbitCamera {
    /// Distance from the focal point.
    pub distance: f32,
    /// Horizontal angle (yaw) in radians.
    pub yaw: f32,
    /// Vertical angle (pitch) in radians.
    pub pitch: f32,
    /// Focal point to orbit around.
    pub focus: Vec3,
    /// Mouse sensitivity for rotation.
    pub sensitivity: f32,
    /// Zoom sensitivity.
    pub zoom_sensitivity: f32,
    /// Pan sensitivity.
    pub pan_sensitivity: f32,
    /// Minimum distance.
    pub min_distance: f32,
    /// Maximum distance.
    pub max_distance: f32,
    /// Minimum pitch angle.
    pub min_pitch: f32,
    /// Maximum pitch angle.
    pub max_pitch: f32,
    /// Current control mode.
    pub mode: CameraMode,
    /// Smooth damping factor (0 = instant, 1 = no movement).
    pub damping: f32,
    /// Target distance for smooth zoom.
    target_distance: f32,
    /// Target yaw for smooth rotation.
    target_yaw: f32,
    /// Target pitch for smooth rotation.
    target_pitch: f32,
    /// Target focus for smooth pan.
    target_focus: Vec3,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self {
            distance: 30.0,
            yaw: 0.0,
            pitch: PI / 6.0, // 30 degrees
            focus: Vec3::ZERO,
            sensitivity: 0.005,
            zoom_sensitivity: 2.0,
            pan_sensitivity: 0.02,
            min_distance: 5.0,
            max_distance: 200.0,
            min_pitch: -PI / 2.0 + 0.1,
            max_pitch: PI / 2.0 - 0.1,
            mode: CameraMode::Orbit,
            damping: 0.1,
            target_distance: 30.0,
            target_yaw: 0.0,
            target_pitch: PI / 6.0,
            target_focus: Vec3::ZERO,
        }
    }
}

impl OrbitCamera {
    /// Calculate camera position from orbital parameters.
    pub fn calculate_position(&self) -> Vec3 {
        let x = self.distance * self.pitch.cos() * self.yaw.sin();
        let y = self.distance * self.pitch.sin();
        let z = self.distance * self.pitch.cos() * self.yaw.cos();
        self.focus + Vec3::new(x, y, z)
    }

    /// Reset camera to default view.
    pub fn reset(&mut self) {
        self.target_distance = 30.0;
        self.target_yaw = 0.0;
        self.target_pitch = PI / 6.0;
        self.target_focus = Vec3::ZERO;
    }

    /// Focus on a specific position.
    pub fn focus_on(&mut self, position: Vec3) {
        self.target_focus = position;
    }

    /// Set the zoom level.
    pub fn set_zoom(&mut self, distance: f32) {
        self.target_distance = distance.clamp(self.min_distance, self.max_distance);
    }
}

/// Camera controller for free movement.
#[derive(Component)]
pub struct CameraController {
    /// Movement speed.
    pub speed: f32,
    /// Sprint multiplier.
    pub sprint_multiplier: f32,
    /// Mouse sensitivity.
    pub sensitivity: f32,
}

impl Default for CameraController {
    fn default() -> Self {
        Self {
            speed: 10.0,
            sprint_multiplier: 2.5,
            sensitivity: 0.003,
        }
    }
}

/// Plugin for camera systems.
pub struct CameraPlugin;

impl Plugin for CameraPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup_camera)
            .add_systems(Update, (orbit_camera_system, camera_keyboard_controls));
    }
}

/// Setup the main camera.
fn setup_camera(mut commands: Commands) {
    let orbit = OrbitCamera::default();
    let position = orbit.calculate_position();

    commands.spawn((
        Camera3d::default(),
        Transform::from_translation(position).looking_at(orbit.focus, Vec3::Y),
        orbit,
        CameraController::default(),
    ));

    info!("Camera initialized at {:?}", position);
}

/// Orbit camera control system.
fn orbit_camera_system(
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    accumulated_mouse_motion: Res<AccumulatedMouseMotion>,
    accumulated_mouse_scroll: Res<AccumulatedMouseScroll>,
    mut query: Query<(&mut OrbitCamera, &mut Transform)>,
    time: Res<Time>,
) {
    for (mut orbit, mut transform) in query.iter_mut() {
        let delta = accumulated_mouse_motion.delta;
        let scroll = accumulated_mouse_scroll.delta.y;

        // Handle rotation (right mouse button or middle button)
        if mouse_buttons.pressed(MouseButton::Right) || mouse_buttons.pressed(MouseButton::Middle) {
            orbit.target_yaw -= delta.x * orbit.sensitivity;
            orbit.target_pitch += delta.y * orbit.sensitivity;
            orbit.target_pitch = orbit.target_pitch.clamp(orbit.min_pitch, orbit.max_pitch);
        }

        // Handle panning (left mouse button + shift or middle button + shift)
        if mouse_buttons.pressed(MouseButton::Left) {
            let right = transform.right();
            let up = transform.up();
            let pan_delta =
                right * (-delta.x * orbit.pan_sensitivity) + up * (delta.y * orbit.pan_sensitivity);
            orbit.target_focus += pan_delta * orbit.distance * 0.01;
        }

        // Handle zoom (scroll wheel)
        if scroll.abs() > 0.0 {
            let zoom_factor = 1.0 - scroll * orbit.zoom_sensitivity * 0.1;
            orbit.target_distance =
                (orbit.target_distance * zoom_factor).clamp(orbit.min_distance, orbit.max_distance);
        }

        // Smooth damping
        let dt = time.delta_secs();
        let factor = 1.0 - (1.0 - orbit.damping).powf(dt * 60.0);

        orbit.distance = orbit.distance + (orbit.target_distance - orbit.distance) * factor;
        orbit.yaw = orbit.yaw + (orbit.target_yaw - orbit.yaw) * factor;
        orbit.pitch = orbit.pitch + (orbit.target_pitch - orbit.pitch) * factor;
        orbit.focus = orbit.focus + (orbit.target_focus - orbit.focus) * factor;

        // Update transform
        let position = orbit.calculate_position();
        transform.translation = position;
        transform.look_at(orbit.focus, Vec3::Y);
    }
}

/// Keyboard controls for camera.
fn camera_keyboard_controls(
    keys: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
    mut query: Query<(&mut OrbitCamera, &CameraController, &Transform)>,
) {
    for (mut orbit, controller, transform) in query.iter_mut() {
        let dt = time.delta_secs();
        let speed = if keys.pressed(KeyCode::ShiftLeft) {
            controller.speed * controller.sprint_multiplier
        } else {
            controller.speed
        };

        // WASD movement in free mode
        if orbit.mode == CameraMode::Free {
            let forward = transform.forward();
            let right = transform.right();

            let mut movement = Vec3::ZERO;

            if keys.pressed(KeyCode::KeyW) {
                movement += *forward;
            }
            if keys.pressed(KeyCode::KeyS) {
                movement -= *forward;
            }
            if keys.pressed(KeyCode::KeyA) {
                movement -= *right;
            }
            if keys.pressed(KeyCode::KeyD) {
                movement += *right;
            }
            if keys.pressed(KeyCode::KeyQ) {
                movement -= Vec3::Y;
            }
            if keys.pressed(KeyCode::KeyE) {
                movement += Vec3::Y;
            }

            if movement.length_squared() > 0.0 {
                orbit.target_focus += movement.normalize() * speed * dt;
            }
        }

        // Reset view
        if keys.just_pressed(KeyCode::KeyR) {
            orbit.reset();
        }

        // Toggle camera mode
        if keys.just_pressed(KeyCode::KeyC) {
            orbit.mode = match orbit.mode {
                CameraMode::Orbit => CameraMode::Free,
                CameraMode::Free => CameraMode::Pan,
                CameraMode::Pan => CameraMode::Follow,
                CameraMode::Follow => CameraMode::Orbit,
            };
            info!("Camera mode: {:?}", orbit.mode);
        }

        // Quick zoom presets
        if keys.just_pressed(KeyCode::Numpad1) {
            orbit.set_zoom(10.0);
        }
        if keys.just_pressed(KeyCode::Numpad2) {
            orbit.set_zoom(30.0);
        }
        if keys.just_pressed(KeyCode::Numpad3) {
            orbit.set_zoom(60.0);
        }
        if keys.just_pressed(KeyCode::Numpad4) {
            orbit.set_zoom(100.0);
        }

        // Quick view angles
        if keys.just_pressed(KeyCode::Numpad7) {
            // Top view
            orbit.target_pitch = PI / 2.0 - 0.1;
            orbit.target_yaw = 0.0;
        }
        if keys.just_pressed(KeyCode::Numpad5) {
            // Front view
            orbit.target_pitch = 0.0;
            orbit.target_yaw = 0.0;
        }
        if keys.just_pressed(KeyCode::Numpad9) {
            // Side view
            orbit.target_pitch = 0.0;
            orbit.target_yaw = PI / 2.0;
        }
    }
}
