//! Training state tracking and persistence.
//!
//! Tracks active training runs, their status, and enables monitoring
//! and recovery from interruptions.

use std::collections::HashMap;
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::Command;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Git information captured at training start.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GitInfo {
    /// Git commit SHA (from `git rev-parse HEAD`)
    pub commit_sha: Option<String>,
    /// Git branch name (from `git branch --show-current`)
    pub branch: Option<String>,
}

/// Capture current git commit SHA and branch name.
///
/// Returns `GitInfo` with `None` fields if git commands fail
/// (e.g., not in a git repository).
pub fn capture_git_info() -> GitInfo {
    let commit_sha = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| {
            String::from_utf8(output.stdout)
                .ok()
                .map(|s| s.trim().to_string())
        })
        .filter(|s| !s.is_empty());

    let branch = Command::new("git")
        .args(["branch", "--show-current"])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| {
            String::from_utf8(output.stdout)
                .ok()
                .map(|s| s.trim().to_string())
        })
        .filter(|s| !s.is_empty());

    GitInfo { commit_sha, branch }
}

/// Status of a training run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TrainingStatus {
    /// Training is initializing
    Initializing,
    /// Training is actively running
    Running,
    /// Training is paused
    Paused,
    /// Training completed successfully
    Completed,
    /// Training failed with error
    Failed,
    /// Training was cancelled
    Cancelled,
}

impl TrainingStatus {
    /// Check if training is still active (can be monitored).
    pub fn is_active(&self) -> bool {
        matches!(self, Self::Initializing | Self::Running | Self::Paused)
    }

    /// Check if training has finished (completed, failed, or cancelled).
    pub fn is_finished(&self) -> bool {
        matches!(self, Self::Completed | Self::Failed | Self::Cancelled)
    }
}

/// Phase of hybrid predictive training.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TrainingPhase {
    Warmup,
    Full,
    Predict,
    Correct,
}

impl std::fmt::Display for TrainingPhase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Warmup => write!(f, "WARMUP"),
            Self::Full => write!(f, "FULL"),
            Self::Predict => write!(f, "PREDICT"),
            Self::Correct => write!(f, "CORRECT"),
        }
    }
}

/// Phase transition event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseTransition {
    /// Step when transition occurred
    pub step: u64,
    /// Previous phase
    pub from_phase: TrainingPhase,
    /// New phase
    pub to_phase: TrainingPhase,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

/// Tracks loss history to compute velocity and acceleration metrics.
///
/// Maintains a sliding window of recent loss values and computes first/second
/// derivatives (velocity/acceleration) on demand.
#[derive(Debug, Clone)]
pub struct LossDynamicsTracker {
    /// Circular buffer of recent loss values
    loss_history: Vec<f32>,
    /// Maximum window size
    window_size: usize,
}

impl LossDynamicsTracker {
    /// Create a new loss dynamics tracker.
    ///
    /// # Arguments
    /// * `window_size` - Number of recent losses to keep (recommended 30-50)
    ///
    /// # Example
    /// ```
    /// use training_tools::LossDynamicsTracker;
    ///
    /// let mut tracker = LossDynamicsTracker::new(30);
    /// let (v, a) = tracker.update(2.5);
    /// ```
    pub fn new(window_size: usize) -> Self {
        Self {
            loss_history: Vec::with_capacity(window_size + 1),
            window_size,
        }
    }

    /// Update tracker with a new loss value and compute velocity/acceleration.
    ///
    /// Returns `(velocity, acceleration)` tuple.
    /// - `velocity`: Negative = loss improving, positive = loss worsening
    /// - `acceleration`: Negative = slowing improvement, positive = accelerating improvement
    ///
    /// # Example
    /// ```
    /// use training_tools::LossDynamicsTracker;
    ///
    /// let mut tracker = LossDynamicsTracker::new(10);
    /// let losses = vec![2.5, 2.4, 2.3, 2.2, 2.1];
    /// for loss in losses {
    ///     let (velocity, acceleration) = tracker.update(loss);
    ///     println!("v={}, a={}", velocity, acceleration);
    /// }
    /// ```
    pub fn update(&mut self, loss: f32) -> (f32, f32) {
        self.loss_history.push(loss);
        if self.loss_history.len() > self.window_size {
            self.loss_history.remove(0);
        }
        calculate_loss_dynamics(&self.loss_history, self.window_size)
    }

    /// Get current loss history.
    pub fn history(&self) -> &[f32] {
        &self.loss_history
    }

    /// Clear the history.
    pub fn reset(&mut self) {
        self.loss_history.clear();
    }
}

/// Metrics for a single training step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepMetrics {
    /// Step number
    pub step: u64,
    /// Loss value
    pub loss: f32,
    /// Gradient norm
    pub gradient_norm: f32,
    /// Current phase
    pub phase: TrainingPhase,
    /// Whether gradients were predicted (not computed)
    pub was_predicted: bool,
    /// Prediction error (if predicted)
    pub prediction_error: Option<f32>,
    /// Step duration in milliseconds
    pub step_time_ms: f64,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Tokens processed in this step (batch_size * seq_length)
    #[serde(default)]
    pub tokens_this_step: u64,
    /// Cumulative tokens trained so far
    #[serde(default)]
    pub total_tokens_trained: u64,
    /// Estimated tokens remaining to train
    #[serde(default)]
    pub tokens_remaining: u64,
    /// Predictor confidence (0.0 - 1.0)
    #[serde(default)]
    pub confidence: f32,
    /// Current learning rate
    #[serde(default)]
    pub learning_rate: f32,
    /// Perplexity (exp(loss))
    #[serde(default)]
    pub perplexity: f32,

    // Generalization health metrics
    /// Training-validation gap (train_loss - val_loss, if validation available)
    #[serde(default)]
    pub train_val_gap: Option<f32>,
    /// Rate of loss change (rolling average derivative)
    #[serde(default)]
    pub loss_velocity: f32,
    /// Rate of velocity change (second derivative)
    #[serde(default)]
    pub loss_acceleration: f32,
    /// Diversity of gradient magnitudes (entropy measure, optional)
    #[serde(default)]
    pub gradient_entropy: Option<f32>,

    /// Per-layer gradient norms (layer_name -> norm).
    /// Enables tracking gradient flow through specific layers.
    #[serde(default)]
    pub layer_gradients: Option<HashMap<String, f32>>,

    /// Per-layer gradient statistics (computed from layer_gradients).
    /// Summarizes gradient health across all layers.
    #[serde(default)]
    pub layer_gradient_stats: Option<LayerGradientStats>,
}

impl StepMetrics {
    /// Update velocity and acceleration fields from recent loss history.
    ///
    /// This is a helper method to populate `loss_velocity` and `loss_acceleration`
    /// based on recent loss values. Call this after creating a StepMetrics instance
    /// if you have access to recent loss history.
    ///
    /// # Arguments
    /// - `recent_losses`: Recent loss values (ordered chronologically, current loss should be last)
    /// - `window_size`: Number of recent losses to use for calculation (default: 30)
    ///
    /// # Example
    /// ```
    /// use training_tools::{StepMetrics, TrainingPhase};
    /// use chrono::Utc;
    ///
    /// let mut metrics = StepMetrics {
    ///     step: 100,
    ///     loss: 2.3,
    ///     gradient_norm: 0.5,
    ///     phase: TrainingPhase::Full,
    ///     was_predicted: false,
    ///     prediction_error: None,
    ///     step_time_ms: 150.0,
    ///     timestamp: Utc::now(),
    ///     tokens_this_step: 4096,
    ///     total_tokens_trained: 409600,
    ///     tokens_remaining: 1000000,
    ///     confidence: 0.9,
    ///     learning_rate: 1e-4,
    ///     perplexity: 10.0,
    ///     train_val_gap: None,
    ///     loss_velocity: 0.0,
    ///     loss_acceleration: 0.0,
    ///     gradient_entropy: None,
    /// };
    ///
    /// let recent_losses = vec![2.5, 2.4, 2.35, 2.3];
    /// metrics.update_dynamics(&recent_losses, 10);
    /// assert!(metrics.loss_velocity < 0.0); // Loss is decreasing
    /// ```
    pub fn update_dynamics(&mut self, recent_losses: &[f32], window_size: usize) {
        let (velocity, acceleration) = calculate_loss_dynamics(recent_losses, window_size);
        self.loss_velocity = velocity;
        self.loss_acceleration = acceleration;
    }
}

/// Calculate loss velocity and acceleration from loss history.
///
/// Uses linear regression for velocity (smoothed rate of change) and
/// windowed velocity differences for acceleration.
///
/// # Arguments
/// - `losses`: Slice of recent loss values (ordered chronologically)
/// - `window_size`: Number of recent losses to analyze (recommended 20-50)
///
/// # Returns
/// - `(velocity, acceleration)` tuple where:
///   - `velocity`: Negative = improving loss, positive = worsening loss
///   - `acceleration`: Negative = slowing improvement, positive = accelerating improvement
///
/// # Example
/// ```
/// use training_tools::calculate_loss_dynamics;
///
/// let losses = vec![2.5, 2.4, 2.3, 2.25, 2.2, 2.18, 2.15];
/// let (velocity, acceleration) = calculate_loss_dynamics(&losses, 5);
/// assert!(velocity < 0.0); // Loss decreasing (improving)
/// ```
pub fn calculate_loss_dynamics(losses: &[f32], window_size: usize) -> (f32, f32) {
    if losses.is_empty() {
        return (0.0, 0.0);
    }

    // Take the last `window_size` losses
    let window_start = losses.len().saturating_sub(window_size);
    let windowed_losses = &losses[window_start..];

    if windowed_losses.len() < 2 {
        return (0.0, 0.0);
    }

    // Calculate velocity using linear regression slope
    let n = windowed_losses.len() as f32;
    let x_values: Vec<f32> = (0..windowed_losses.len()).map(|i| i as f32).collect();

    // Linear regression: y = mx + b, we want m (slope)
    let x_mean = x_values.iter().sum::<f32>() / n;
    let y_mean = windowed_losses.iter().sum::<f32>() / n;

    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for i in 0..windowed_losses.len() {
        let x_diff = x_values[i] - x_mean;
        let y_diff = windowed_losses[i] - y_mean;
        numerator += x_diff * y_diff;
        denominator += x_diff * x_diff;
    }

    let velocity = if denominator.abs() > 1e-10 {
        numerator / denominator
    } else {
        0.0
    };

    // Calculate acceleration from velocity changes
    // Split window in half and compute velocity for each half
    let acceleration = if windowed_losses.len() >= 6 {
        let mid = windowed_losses.len() / 2;
        let first_half = &windowed_losses[..mid];
        let second_half = &windowed_losses[mid..];

        // Velocity for first half
        let n1 = first_half.len() as f32;
        let x1: Vec<f32> = (0..first_half.len()).map(|i| i as f32).collect();
        let x1_mean = x1.iter().sum::<f32>() / n1;
        let y1_mean = first_half.iter().sum::<f32>() / n1;
        let mut num1 = 0.0;
        let mut den1 = 0.0;
        for i in 0..first_half.len() {
            let x_diff = x1[i] - x1_mean;
            let y_diff = first_half[i] - y1_mean;
            num1 += x_diff * y_diff;
            den1 += x_diff * x_diff;
        }
        let v1 = if den1.abs() > 1e-10 { num1 / den1 } else { 0.0 };

        // Velocity for second half
        let n2 = second_half.len() as f32;
        let x2: Vec<f32> = (0..second_half.len()).map(|i| i as f32).collect();
        let x2_mean = x2.iter().sum::<f32>() / n2;
        let y2_mean = second_half.iter().sum::<f32>() / n2;
        let mut num2 = 0.0;
        let mut den2 = 0.0;
        for i in 0..second_half.len() {
            let x_diff = x2[i] - x2_mean;
            let y_diff = second_half[i] - y2_mean;
            num2 += x_diff * y_diff;
            den2 += x_diff * x_diff;
        }
        let v2 = if den2.abs() > 1e-10 { num2 / den2 } else { 0.0 };

        // Acceleration = change in velocity
        v2 - v1
    } else {
        0.0
    };

    (velocity, acceleration)
}

/// Per-layer gradient statistics for diagnosing training issues.
///
/// Tracks aggregate statistics across all layers to identify:
/// - Vanishing gradients (norm < 1e-6)
/// - Exploding gradients (norm > 100.0)
/// - Overall gradient distribution health
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct LayerGradientStats {
    /// Maximum gradient norm across all layers
    pub max_norm: f32,
    /// Minimum gradient norm across all layers
    pub min_norm: f32,
    /// Mean gradient norm across all layers
    pub mean_norm: f32,
    /// Layers with gradient norm < 1e-6 (vanishing gradients)
    pub vanishing_layers: Vec<String>,
    /// Layers with gradient norm > 100.0 (exploding gradients)
    pub exploding_layers: Vec<String>,
}

impl LayerGradientStats {
    /// Threshold below which gradients are considered vanishing.
    pub const VANISHING_THRESHOLD: f32 = 1e-6;
    /// Threshold above which gradients are considered exploding.
    pub const EXPLODING_THRESHOLD: f32 = 100.0;

    /// Compute layer gradient statistics from a map of layer names to gradient norms.
    ///
    /// # Example
    /// ```
    /// use std::collections::HashMap;
    /// use training_tools::LayerGradientStats;
    ///
    /// let mut gradients = HashMap::new();
    /// gradients.insert("layer_0".to_string(), 0.5);
    /// gradients.insert("layer_1".to_string(), 0.3);
    /// gradients.insert("layer_2".to_string(), 1e-8); // vanishing
    ///
    /// let stats = LayerGradientStats::from_layer_gradients(&gradients);
    /// assert_eq!(stats.vanishing_layers, vec!["layer_2".to_string()]);
    /// assert!(stats.max_norm > 0.4);
    /// ```
    pub fn from_layer_gradients(layer_gradients: &HashMap<String, f32>) -> Self {
        if layer_gradients.is_empty() {
            return Self::default();
        }

        let norms: Vec<f32> = layer_gradients.values().copied().collect();
        let max_norm = norms.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let min_norm = norms.iter().copied().fold(f32::INFINITY, f32::min);
        let mean_norm = norms.iter().sum::<f32>() / norms.len() as f32;

        let vanishing_layers: Vec<String> = layer_gradients
            .iter()
            .filter(|(_, &norm)| norm < Self::VANISHING_THRESHOLD)
            .map(|(name, _)| name.clone())
            .collect();

        let exploding_layers: Vec<String> = layer_gradients
            .iter()
            .filter(|(_, &norm)| norm > Self::EXPLODING_THRESHOLD)
            .map(|(name, _)| name.clone())
            .collect();

        Self {
            max_norm,
            min_norm,
            mean_norm,
            vanishing_layers,
            exploding_layers,
        }
    }

    /// Check if there are any problematic layers (vanishing or exploding).
    pub fn has_problems(&self) -> bool {
        !self.vanishing_layers.is_empty() || !self.exploding_layers.is_empty()
    }

    /// Get the ratio of max to min gradient norm (spread indicator).
    /// Returns None if min_norm is zero or negative.
    pub fn gradient_spread(&self) -> Option<f32> {
        if self.min_norm > 0.0 {
            Some(self.max_norm / self.min_norm)
        } else {
            None
        }
    }
}

/// Generalization health status for detecting overfitting/underfitting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GeneralizationHealth {
    /// Model is learning well, no signs of overfitting/underfitting
    Healthy,
    /// Model may be underfitting (high loss, slow learning)
    Underfitting,
    /// Model may be overfitting (train-val gap, decreasing gradient diversity)
    Overfitting,
    /// Insufficient data to determine health
    Unknown,
}

/// Configuration for a training run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Config schema version for tracking hyperparameter changes
    #[serde(default = "default_config_version")]
    pub config_version: u32,
    /// SHA-256 hash of the serialized config (first 12 hex chars)
    #[serde(default)]
    pub config_hash: String,
    /// Model size identifier
    pub model_size: String,
    /// Number of parameters
    pub num_parameters: usize,
    /// Hidden dimension
    pub hidden_size: usize,
    /// Number of layers
    pub num_layers: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Maximum sequence length
    pub max_seq_length: usize,
    /// Batch size
    pub batch_size: usize,
    /// Learning rate
    pub learning_rate: f32,
    /// Maximum training steps
    pub max_steps: u64,
    /// Gradient checkpointing enabled
    pub gradient_checkpointing: bool,
    /// Checkpoint interval (layers)
    pub checkpoint_interval: usize,
    /// Device (cpu/cuda:N)
    pub device: String,
}

/// Default config version (1).
fn default_config_version() -> u32 {
    1
}

/// Compute a SHA-256 hash of a TrainingConfig (first 12 hex characters).
///
/// This excludes the `config_hash` field itself to avoid circular dependency.
/// Allows comparing configs across runs to detect hyperparameter changes.
///
/// # Example
///
/// ```
/// use training_tools::TrainingConfig;
/// use training_tools::compute_config_hash;
///
/// let config = TrainingConfig {
///     config_version: 1,
///     config_hash: String::new(),
///     model_size: "100m".to_string(),
///     num_parameters: 100_000_000,
///     hidden_size: 768,
///     num_layers: 12,
///     num_heads: 12,
///     max_seq_length: 512,
///     batch_size: 8,
///     learning_rate: 1e-4,
///     max_steps: 10000,
///     gradient_checkpointing: true,
///     checkpoint_interval: 4,
///     device: "cuda:0".to_string(),
/// };
/// let hash = compute_config_hash(&config);
/// assert_eq!(hash.len(), 12);
/// ```
pub fn compute_config_hash(config: &TrainingConfig) -> String {
    // Create a copy without config_hash to avoid circular dependency
    let hashable = TrainingConfig {
        config_version: config.config_version,
        config_hash: String::new(), // Exclude from hash computation
        model_size: config.model_size.clone(),
        num_parameters: config.num_parameters,
        hidden_size: config.hidden_size,
        num_layers: config.num_layers,
        num_heads: config.num_heads,
        max_seq_length: config.max_seq_length,
        batch_size: config.batch_size,
        learning_rate: config.learning_rate,
        max_steps: config.max_steps,
        gradient_checkpointing: config.gradient_checkpointing,
        checkpoint_interval: config.checkpoint_interval,
        device: config.device.clone(),
    };

    // Serialize to JSON and compute SHA-256
    let json = serde_json::to_string(&hashable).unwrap_or_default();
    let mut hasher = Sha256::new();
    hasher.update(json.as_bytes());
    let result = hasher.finalize();

    // Return first 12 hex characters
    format!("{:x}", result)[..12].to_string()
}

/// Checkpoint save/upload event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointEvent {
    /// Checkpoint step number
    pub step: u64,
    /// When checkpoint save started
    pub save_started: DateTime<Utc>,
    /// When checkpoint save completed
    pub save_completed: Option<DateTime<Utc>>,
    /// Checkpoint file size in bytes
    pub size_bytes: Option<u64>,
    /// When HuggingFace upload started (if any)
    pub upload_started: Option<DateTime<Utc>>,
    /// When HuggingFace upload completed
    pub upload_completed: Option<DateTime<Utc>>,
    /// HuggingFace URL if uploaded
    pub hf_url: Option<String>,
    /// Path to checkpoint file
    pub path: PathBuf,
}

impl CheckpointEvent {
    /// Create a new checkpoint event.
    pub fn new(step: u64, path: PathBuf) -> Self {
        Self {
            step,
            save_started: Utc::now(),
            save_completed: None,
            size_bytes: None,
            upload_started: None,
            upload_completed: None,
            hf_url: None,
            path,
        }
    }

    /// Mark save as complete.
    pub fn save_complete(&mut self, size_bytes: u64) {
        self.save_completed = Some(Utc::now());
        self.size_bytes = Some(size_bytes);
    }

    /// Mark upload as started.
    pub fn upload_start(&mut self) {
        self.upload_started = Some(Utc::now());
    }

    /// Mark upload as complete.
    pub fn upload_complete(&mut self, url: &str) {
        self.upload_completed = Some(Utc::now());
        self.hf_url = Some(url.to_string());
    }

    /// Get save duration.
    pub fn save_duration(&self) -> Option<chrono::Duration> {
        self.save_completed.map(|end| end - self.save_started)
    }

    /// Get upload duration.
    pub fn upload_duration(&self) -> Option<chrono::Duration> {
        match (self.upload_started, self.upload_completed) {
            (Some(start), Some(end)) => Some(end - start),
            _ => None,
        }
    }
}

/// A training run with all its metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingRun {
    /// Unique run identifier
    pub run_id: String,
    /// Human-readable run name
    pub run_name: String,
    /// Current status
    pub status: TrainingStatus,
    /// Training configuration
    pub config: TrainingConfig,
    /// Start time
    pub started_at: DateTime<Utc>,
    /// Last update time
    pub updated_at: DateTime<Utc>,
    /// End time (if finished)
    pub ended_at: Option<DateTime<Utc>>,
    /// Current step
    pub current_step: u64,
    /// Current loss
    pub current_loss: f32,
    /// Current phase
    pub current_phase: TrainingPhase,
    /// Total forward passes
    pub total_forward: u64,
    /// Total backward passes
    pub total_backward: u64,
    /// Best loss achieved
    pub best_loss: f32,
    /// Step at best loss
    pub best_step: u64,
    /// Error message (if failed)
    pub error_message: Option<String>,
    /// Path to run directory
    pub run_dir: PathBuf,
    /// Path to metrics log file
    pub metrics_file: PathBuf,
    /// Path to latest checkpoint
    pub latest_checkpoint: Option<PathBuf>,
    /// HuggingFace repo (if uploaded)
    pub hf_repo: Option<String>,
    /// GPU memory usage in bytes
    pub gpu_memory_used: Option<u64>,
    /// GPU memory total in bytes
    pub gpu_memory_total: Option<u64>,
    /// GPU temperature in Celsius
    pub gpu_temperature: Option<u32>,
    /// GPU power draw in watts
    pub gpu_power_draw: Option<f32>,
    /// GPU utilization percentage
    pub gpu_utilization: Option<u32>,
    /// Peak GPU memory usage in bytes
    pub gpu_memory_peak: Option<u64>,
    /// Checkpoint events
    #[serde(default)]
    pub checkpoints: Vec<CheckpointEvent>,
    /// Estimated steps per second
    pub steps_per_second: Option<f64>,
    /// Phase transition timestamps
    #[serde(default)]
    pub phase_transitions: Vec<PhaseTransition>,
    /// Total tokens trained so far
    #[serde(default)]
    pub total_tokens_trained: u64,
    /// Tokens per step (batch_size * seq_length)
    #[serde(default)]
    pub tokens_per_step: u64,
    /// Tokens per second (throughput)
    #[serde(default)]
    pub tokens_per_second: Option<f64>,
    /// Git commit SHA at training start
    #[serde(default)]
    pub git_commit_sha: Option<String>,
    /// Git branch name at training start
    #[serde(default)]
    pub git_branch: Option<String>,
}

impl TrainingRun {
    /// Create a new training run.
    pub fn new(run_name: &str, config: TrainingConfig, run_dir: PathBuf) -> Self {
        let run_id = uuid::Uuid::new_v4().to_string();
        let now = Utc::now();
        let metrics_file = run_dir.join("metrics.jsonl");
        let tokens_per_step = (config.batch_size * config.max_seq_length) as u64;
        let git_info = capture_git_info();

        Self {
            run_id,
            run_name: run_name.to_string(),
            status: TrainingStatus::Initializing,
            config,
            started_at: now,
            updated_at: now,
            ended_at: None,
            current_step: 0,
            current_loss: f32::INFINITY,
            current_phase: TrainingPhase::Warmup,
            total_forward: 0,
            total_backward: 0,
            best_loss: f32::INFINITY,
            best_step: 0,
            error_message: None,
            run_dir,
            metrics_file,
            latest_checkpoint: None,
            hf_repo: None,
            gpu_memory_used: None,
            gpu_memory_total: None,
            gpu_temperature: None,
            gpu_power_draw: None,
            gpu_utilization: None,
            gpu_memory_peak: None,
            checkpoints: Vec::new(),
            steps_per_second: None,
            phase_transitions: Vec::new(),
            total_tokens_trained: 0,
            tokens_per_step,
            tokens_per_second: None,
            git_commit_sha: git_info.commit_sha,
            git_branch: git_info.branch,
        }
    }

    /// Record a phase transition.
    pub fn record_phase_transition(&mut self, from: TrainingPhase, to: TrainingPhase) {
        self.phase_transitions.push(PhaseTransition {
            step: self.current_step,
            from_phase: from,
            to_phase: to,
            timestamp: Utc::now(),
        });
    }

    /// Start a new checkpoint.
    pub fn start_checkpoint(&mut self, path: PathBuf) -> usize {
        let event = CheckpointEvent::new(self.current_step, path);
        self.checkpoints.push(event);
        self.checkpoints.len() - 1
    }

    /// Complete a checkpoint save.
    pub fn complete_checkpoint_save(&mut self, idx: usize, size_bytes: u64) {
        if let Some(event) = self.checkpoints.get_mut(idx) {
            event.save_complete(size_bytes);
        }
    }

    /// Start checkpoint upload.
    pub fn start_checkpoint_upload(&mut self, idx: usize) {
        if let Some(event) = self.checkpoints.get_mut(idx) {
            event.upload_start();
        }
    }

    /// Complete checkpoint upload.
    pub fn complete_checkpoint_upload(&mut self, idx: usize, url: &str) {
        if let Some(event) = self.checkpoints.get_mut(idx) {
            event.upload_complete(url);
        }
    }

    /// Get estimated time to completion.
    pub fn eta(&self) -> Option<chrono::Duration> {
        let steps_per_sec = self.steps_per_second?;
        if steps_per_sec <= 0.0 {
            return None;
        }
        let remaining = self.config.max_steps.saturating_sub(self.current_step);
        let secs = remaining as f64 / steps_per_sec;
        Some(chrono::Duration::seconds(secs as i64))
    }

    /// Format ETA as string.
    pub fn eta_string(&self) -> String {
        match self.eta() {
            Some(eta) => {
                let secs = eta.num_seconds();
                if secs < 60 {
                    format!("{}s", secs)
                } else if secs < 3600 {
                    format!("{}m {}s", secs / 60, secs % 60)
                } else {
                    format!("{}h {}m", secs / 3600, (secs % 3600) / 60)
                }
            }
            None => "calculating...".to_string(),
        }
    }

    /// Update with new step metrics.
    pub fn update_step(&mut self, metrics: &StepMetrics) {
        self.current_step = metrics.step;
        self.current_loss = metrics.loss;
        self.current_phase = metrics.phase;
        self.total_forward += 1;
        if !metrics.was_predicted {
            self.total_backward += 1;
        }
        self.updated_at = Utc::now();

        // Update token tracking
        if metrics.tokens_this_step > 0 {
            self.total_tokens_trained = metrics.total_tokens_trained;
            self.tokens_per_step = metrics.tokens_this_step;
        }

        if metrics.loss < self.best_loss {
            self.best_loss = metrics.loss;
            self.best_step = metrics.step;
        }
    }

    /// Get backward reduction percentage.
    pub fn backward_reduction(&self) -> f64 {
        if self.total_forward == 0 {
            0.0
        } else {
            100.0 - (100.0 * self.total_backward as f64 / self.total_forward as f64)
        }
    }

    /// Get training progress as percentage.
    pub fn progress(&self) -> f64 {
        if self.config.max_steps == 0 {
            0.0
        } else {
            100.0 * self.current_step as f64 / self.config.max_steps as f64
        }
    }

    /// Get elapsed training time.
    pub fn elapsed(&self) -> chrono::Duration {
        let end = self.ended_at.unwrap_or_else(Utc::now);
        end - self.started_at
    }

    /// Calculate current loss velocity and acceleration from recent metrics.
    ///
    /// Analyzes the last 30 steps (or fewer if not available) to compute:
    /// - Velocity: Rate of loss change (negative = improving)
    /// - Acceleration: Rate of velocity change (negative = slowing improvement)
    ///
    /// Returns `(0.0, 0.0)` if insufficient data is available.
    pub fn loss_dynamics(&self) -> (f32, f32) {
        // Use last 30 steps for dynamics calculation
        let window_size = 30;

        match self.read_recent_metrics(window_size) {
            Ok(metrics) if metrics.len() >= 2 => {
                let losses: Vec<f32> = metrics.iter().map(|m| m.loss).collect();
                calculate_loss_dynamics(&losses, window_size)
            }
            _ => (0.0, 0.0),
        }
    }

    /// Get tokens per second.
    pub fn tokens_per_second(&self) -> f64 {
        let elapsed_secs = self.elapsed().num_seconds() as f64;
        if elapsed_secs == 0.0 {
            0.0
        } else {
            let tokens = self.current_step as f64
                * self.config.batch_size as f64
                * self.config.max_seq_length as f64;
            tokens / elapsed_secs
        }
    }

    /// Save run state to file.
    pub fn save(&self) -> anyhow::Result<()> {
        let state_file = self.run_dir.join("run_state.json");
        let json = serde_json::to_string_pretty(self)?;
        fs::write(&state_file, json)?;
        Ok(())
    }

    /// Load run state from file.
    pub fn load(run_dir: &Path) -> anyhow::Result<Self> {
        let state_file = run_dir.join("run_state.json");
        let json = fs::read_to_string(&state_file)?;
        let run: Self = serde_json::from_str(&json)?;
        Ok(run)
    }

    /// Read recent metrics from the metrics log file.
    pub fn read_recent_metrics(&self, count: usize) -> anyhow::Result<Vec<StepMetrics>> {
        if !self.metrics_file.exists() {
            return Ok(Vec::new());
        }

        let file = fs::File::open(&self.metrics_file)?;
        let reader = BufReader::new(file);

        let mut all_metrics: Vec<StepMetrics> = reader
            .lines()
            .filter_map(|line| line.ok())
            .filter_map(|line| serde_json::from_str(&line).ok())
            .collect();

        // Return last N metrics
        if all_metrics.len() > count {
            all_metrics = all_metrics.split_off(all_metrics.len() - count);
        }

        Ok(all_metrics)
    }

    /// Assess generalization health based on recent metrics.
    ///
    /// Analyzes recent training metrics to detect signs of overfitting or underfitting:
    /// - Overfitting: Large train-val gap, decreasing gradient entropy, slowing loss velocity
    /// - Underfitting: Consistently high loss, positive loss acceleration (loss increasing)
    /// - Healthy: Good progress, stable metrics, no concerning trends
    pub fn generalization_health(&self) -> GeneralizationHealth {
        // Need at least 10 steps for meaningful analysis
        let recent_metrics = match self.read_recent_metrics(10) {
            Ok(m) if m.len() >= 5 => m,
            _ => return GeneralizationHealth::Unknown,
        };

        // Check for train-validation gap (strong overfitting signal)
        let has_large_gap = recent_metrics
            .iter()
            .filter_map(|m| m.train_val_gap)
            .any(|gap| gap > 0.5); // Gap > 0.5 is concerning

        // Check loss velocity trend (should be negative in healthy training)
        let avg_velocity: f32 = recent_metrics.iter().map(|m| m.loss_velocity).sum::<f32>()
            / recent_metrics.len() as f32;

        // Check loss acceleration (positive = loss increasing = bad)
        let avg_acceleration: f32 = recent_metrics
            .iter()
            .map(|m| m.loss_acceleration)
            .sum::<f32>()
            / recent_metrics.len() as f32;

        // Check gradient entropy trend (decreasing = memorization)
        let entropy_trend = if recent_metrics.len() >= 6 {
            let first_half: Vec<_> = recent_metrics
                .iter()
                .take(recent_metrics.len() / 2)
                .filter_map(|m| m.gradient_entropy)
                .collect();
            let second_half: Vec<_> = recent_metrics
                .iter()
                .skip(recent_metrics.len() / 2)
                .filter_map(|m| m.gradient_entropy)
                .collect();

            if !first_half.is_empty() && !second_half.is_empty() {
                let first_avg: f32 = first_half.iter().sum::<f32>() / first_half.len() as f32;
                let second_avg: f32 = second_half.iter().sum::<f32>() / second_half.len() as f32;
                Some(second_avg - first_avg)
            } else {
                None
            }
        } else {
            None
        };

        // Decision logic
        if has_large_gap {
            // Clear overfitting signal from train-val gap
            GeneralizationHealth::Overfitting
        } else if let Some(trend) = entropy_trend {
            // Entropy decreasing significantly = overfitting
            if trend < -0.1 && avg_velocity > -0.01 {
                GeneralizationHealth::Overfitting
            } else if avg_acceleration > 0.001 {
                // Loss increasing = underfitting
                GeneralizationHealth::Underfitting
            } else {
                GeneralizationHealth::Healthy
            }
        } else if avg_acceleration > 0.001 {
            // Loss increasing without other signals
            GeneralizationHealth::Underfitting
        } else if avg_velocity.abs() < 0.0001 && self.current_loss > 3.0 {
            // Stuck at high loss = underfitting
            GeneralizationHealth::Underfitting
        } else {
            GeneralizationHealth::Healthy
        }
    }

    /// Get the gradient norm history for a specific layer.
    ///
    /// Reads all metrics from the metrics file and extracts the gradient norm
    /// for the specified layer at each step where it was recorded.
    ///
    /// # Arguments
    /// * `layer` - The layer name to get gradient history for
    ///
    /// # Returns
    /// A vector of gradient norms in chronological order (oldest first).
    /// Returns an empty vector if no metrics exist or the layer wasn't tracked.
    ///
    /// # Example
    /// ```ignore
    /// let history = run.layer_gradient_history("transformer.layer_0.attention");
    /// if history.len() > 10 {
    ///     let recent_mean = history.iter().rev().take(10).sum::<f32>() / 10.0;
    ///     println!("Recent average gradient norm: {}", recent_mean);
    /// }
    /// ```
    pub fn layer_gradient_history(&self, layer: &str) -> Vec<f32> {
        // Read all available metrics
        let all_metrics = match self.read_recent_metrics(usize::MAX) {
            Ok(m) => m,
            Err(_) => return Vec::new(),
        };

        all_metrics
            .iter()
            .filter_map(|m| {
                m.layer_gradients
                    .as_ref()
                    .and_then(|lg| lg.get(layer).copied())
            })
            .collect()
    }

    /// Identify layers with problematic gradients across training history.
    ///
    /// Analyzes all recorded metrics to find layers that have experienced
    /// vanishing or exploding gradients at any point during training.
    ///
    /// # Returns
    /// A tuple of `(vanishing_layers, exploding_layers)` where:
    /// - `vanishing_layers`: Layer names that had gradient norm < 1e-6
    /// - `exploding_layers`: Layer names that had gradient norm > 100.0
    ///
    /// Layers are returned sorted alphabetically for consistent output.
    ///
    /// # Example
    /// ```ignore
    /// let (vanishing, exploding) = run.problematic_layers();
    /// if !vanishing.is_empty() {
    ///     println!("Warning: {} layers have vanishing gradients", vanishing.len());
    ///     for layer in &vanishing {
    ///         println!("  - {}", layer);
    ///     }
    /// }
    /// ```
    pub fn problematic_layers(&self) -> (Vec<String>, Vec<String>) {
        use std::collections::HashSet;

        let all_metrics = match self.read_recent_metrics(usize::MAX) {
            Ok(m) => m,
            Err(_) => return (Vec::new(), Vec::new()),
        };

        let mut vanishing_set: HashSet<String> = HashSet::new();
        let mut exploding_set: HashSet<String> = HashSet::new();

        for metric in &all_metrics {
            if let Some(ref stats) = metric.layer_gradient_stats {
                vanishing_set.extend(stats.vanishing_layers.iter().cloned());
                exploding_set.extend(stats.exploding_layers.iter().cloned());
            }
        }

        let mut vanishing: Vec<String> = vanishing_set.into_iter().collect();
        let mut exploding: Vec<String> = exploding_set.into_iter().collect();

        vanishing.sort();
        exploding.sort();

        (vanishing, exploding)
    }
}

/// Manager for discovering and tracking training runs.
#[derive(Debug)]
pub struct RunManager {
    /// Base directory for training runs
    runs_dir: PathBuf,
    /// Cached runs
    runs: HashMap<String, TrainingRun>,
}

impl RunManager {
    /// Create a new run manager.
    pub fn new(runs_dir: PathBuf) -> Self {
        Self {
            runs_dir,
            runs: HashMap::new(),
        }
    }

    /// Get the runs directory.
    pub fn runs_dir(&self) -> &Path {
        &self.runs_dir
    }

    /// Discover all training runs in the runs directory.
    pub fn discover_runs(&mut self) -> anyhow::Result<()> {
        self.runs.clear();

        if !self.runs_dir.exists() {
            return Ok(());
        }

        for entry in fs::read_dir(&self.runs_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_dir() {
                if let Ok(run) = TrainingRun::load(&path) {
                    self.runs.insert(run.run_id.clone(), run);
                }
            }
        }

        Ok(())
    }

    /// Get all discovered runs.
    pub fn runs(&self) -> impl Iterator<Item = &TrainingRun> {
        self.runs.values()
    }

    /// Get active runs (still training).
    pub fn active_runs(&self) -> impl Iterator<Item = &TrainingRun> {
        self.runs.values().filter(|r| r.status.is_active())
    }

    /// Get a run by ID.
    pub fn get_run(&self, run_id: &str) -> Option<&TrainingRun> {
        self.runs.get(run_id)
    }

    /// Get a mutable run by ID.
    pub fn get_run_mut(&mut self, run_id: &str) -> Option<&mut TrainingRun> {
        self.runs.get_mut(run_id)
    }

    /// Create a new run directory.
    pub fn create_run_dir(&self, run_name: &str) -> anyhow::Result<PathBuf> {
        let timestamp = Utc::now().format("%Y%m%d_%H%M%S");
        let dir_name = format!("{}_{}", run_name, timestamp);
        let run_dir = self.runs_dir.join(dir_name);
        fs::create_dir_all(&run_dir)?;
        Ok(run_dir)
    }

    /// Register a new run.
    pub fn register_run(&mut self, run: TrainingRun) {
        self.runs.insert(run.run_id.clone(), run);
    }

    /// Remove a run from tracking (does not delete files).
    pub fn unregister_run(&mut self, run_id: &str) -> Option<TrainingRun> {
        self.runs.remove(run_id)
    }

    /// Detect and mark stale runs as cancelled.
    ///
    /// A run is considered stale if:
    /// - Status is "running" or "initializing"
    /// - The metrics file hasn't been modified in `stale_threshold` duration
    /// - OR the run has 0 steps and was started more than `stale_threshold` ago
    ///
    /// Returns the number of runs marked as stale.
    pub fn mark_stale_runs(&mut self, stale_threshold: chrono::Duration) -> usize {
        let now = Utc::now();
        let mut stale_count = 0;

        for run in self.runs.values_mut() {
            if !run.status.is_active() {
                continue;
            }

            let is_stale = if run.current_step == 0 {
                // Run never started - check if it's been too long since creation
                now - run.started_at > stale_threshold
            } else {
                // Check metrics file modification time
                if let Ok(metadata) = std::fs::metadata(&run.metrics_file) {
                    if let Ok(modified) = metadata.modified() {
                        let modified_dt: DateTime<Utc> = modified.into();
                        now - modified_dt > stale_threshold
                    } else {
                        // Can't get mtime, check updated_at
                        now - run.updated_at > stale_threshold
                    }
                } else {
                    // No metrics file, check updated_at
                    now - run.updated_at > stale_threshold
                }
            };

            if is_stale {
                run.status = TrainingStatus::Cancelled;
                run.error_message =
                    Some("Automatically marked as stale - no updates detected".to_string());
                run.ended_at = Some(now);
                stale_count += 1;

                // Try to save the updated state
                let _ = run.save();
            }
        }

        stale_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_loss_dynamics_decreasing() {
        // Steadily decreasing loss (good training)
        let losses = vec![2.5, 2.4, 2.3, 2.2, 2.1, 2.0];
        let (velocity, _acceleration) = calculate_loss_dynamics(&losses, 10);

        // Velocity should be negative (loss decreasing)
        assert!(
            velocity < 0.0,
            "Velocity should be negative for decreasing loss"
        );
        assert!(
            velocity.abs() > 0.05,
            "Velocity magnitude should be significant"
        );
    }

    #[test]
    fn test_calculate_loss_dynamics_increasing() {
        // Increasing loss (bad training)
        let losses = vec![2.0, 2.1, 2.2, 2.3, 2.4, 2.5];
        let (velocity, _acceleration) = calculate_loss_dynamics(&losses, 10);

        // Velocity should be positive (loss increasing)
        assert!(
            velocity > 0.0,
            "Velocity should be positive for increasing loss"
        );
    }

    #[test]
    fn test_calculate_loss_dynamics_plateau() {
        // Flat loss (plateau)
        let losses = vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0];
        let (velocity, acceleration) = calculate_loss_dynamics(&losses, 10);

        // Velocity should be near zero
        assert!(
            velocity.abs() < 0.01,
            "Velocity should be near zero for plateau"
        );
        assert!(
            acceleration.abs() < 0.01,
            "Acceleration should be near zero for plateau"
        );
    }

    #[test]
    fn test_calculate_loss_dynamics_acceleration() {
        // Loss decreasing, then slowing (negative acceleration)
        let losses = vec![3.0, 2.8, 2.6, 2.45, 2.35, 2.3, 2.28, 2.27];
        let (_velocity, acceleration) = calculate_loss_dynamics(&losses, 10);

        // Acceleration should be positive (velocity becoming less negative = slowing improvement)
        assert!(
            acceleration > 0.0,
            "Acceleration should be positive when improvement slows"
        );
    }

    #[test]
    fn test_calculate_loss_dynamics_empty() {
        let losses = vec![];
        let (velocity, acceleration) = calculate_loss_dynamics(&losses, 10);
        assert_eq!(velocity, 0.0);
        assert_eq!(acceleration, 0.0);
    }

    #[test]
    fn test_calculate_loss_dynamics_single() {
        let losses = vec![2.5];
        let (velocity, acceleration) = calculate_loss_dynamics(&losses, 10);
        assert_eq!(velocity, 0.0);
        assert_eq!(acceleration, 0.0);
    }

    #[test]
    fn test_calculate_loss_dynamics_window_size() {
        // Window size larger than data
        let losses = vec![2.5, 2.4, 2.3];
        let (velocity1, _) = calculate_loss_dynamics(&losses, 10);
        let (velocity2, _) = calculate_loss_dynamics(&losses, 3);

        // Should produce same result (uses available data)
        assert!((velocity1 - velocity2).abs() < 0.01);
    }

    #[test]
    fn test_training_run_loss_dynamics() {
        use std::path::PathBuf;

        // Create a test run
        let config = TrainingConfig {
            config_version: 1,
            config_hash: String::new(),
            model_size: "test".to_string(),
            num_parameters: 1000,
            hidden_size: 64,
            num_layers: 2,
            num_heads: 2,
            max_seq_length: 128,
            batch_size: 4,
            learning_rate: 1e-4,
            max_steps: 100,
            gradient_checkpointing: false,
            checkpoint_interval: 10,
            device: "cpu".to_string(),
        };

        let run = TrainingRun::new("test_run", config, PathBuf::from("/tmp/test"));

        // Should return (0.0, 0.0) when no metrics available
        let (velocity, acceleration) = run.loss_dynamics();
        assert_eq!(velocity, 0.0);
        assert_eq!(acceleration, 0.0);
    }

    #[test]
    fn test_layer_gradient_stats_default() {
        let stats = LayerGradientStats::default();
        assert_eq!(stats.max_norm, 0.0);
        assert_eq!(stats.min_norm, 0.0);
        assert_eq!(stats.mean_norm, 0.0);
        assert!(stats.vanishing_layers.is_empty());
        assert!(stats.exploding_layers.is_empty());
        assert!(!stats.has_problems());
    }

    #[test]
    fn test_layer_gradient_stats_from_layer_gradients() {
        let mut gradients = HashMap::new();
        gradients.insert("layer_0".to_string(), 0.5);
        gradients.insert("layer_1".to_string(), 0.3);
        gradients.insert("layer_2".to_string(), 0.4);

        let stats = LayerGradientStats::from_layer_gradients(&gradients);

        assert!((stats.max_norm - 0.5).abs() < 1e-6);
        assert!((stats.min_norm - 0.3).abs() < 1e-6);
        assert!((stats.mean_norm - 0.4).abs() < 1e-6);
        assert!(stats.vanishing_layers.is_empty());
        assert!(stats.exploding_layers.is_empty());
        assert!(!stats.has_problems());
    }

    #[test]
    fn test_layer_gradient_stats_vanishing() {
        let mut gradients = HashMap::new();
        gradients.insert("layer_0".to_string(), 0.5);
        gradients.insert("layer_1".to_string(), 1e-8);

        let stats = LayerGradientStats::from_layer_gradients(&gradients);

        assert_eq!(stats.vanishing_layers.len(), 1);
        assert!(stats.vanishing_layers.contains(&"layer_1".to_string()));
        assert!(stats.has_problems());
    }

    #[test]
    fn test_layer_gradient_stats_exploding() {
        let mut gradients = HashMap::new();
        gradients.insert("layer_0".to_string(), 0.5);
        gradients.insert("layer_1".to_string(), 150.0);

        let stats = LayerGradientStats::from_layer_gradients(&gradients);

        assert_eq!(stats.exploding_layers.len(), 1);
        assert!(stats.exploding_layers.contains(&"layer_1".to_string()));
        assert!(stats.has_problems());
    }

    #[test]
    fn test_layer_gradient_stats_gradient_spread() {
        let mut gradients = HashMap::new();
        gradients.insert("layer_0".to_string(), 0.1);
        gradients.insert("layer_1".to_string(), 1.0);

        let stats = LayerGradientStats::from_layer_gradients(&gradients);

        let spread = stats.gradient_spread().unwrap();
        assert!((spread - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_layer_gradient_stats_gradient_spread_zero() {
        let mut gradients = HashMap::new();
        gradients.insert("layer_0".to_string(), 0.0);
        gradients.insert("layer_1".to_string(), 1.0);

        let stats = LayerGradientStats::from_layer_gradients(&gradients);

        assert!(stats.gradient_spread().is_none());
    }

    #[test]
    fn test_loss_dynamics_tracker_creation() {
        let tracker = LossDynamicsTracker::new(30);
        assert!(tracker.history().is_empty());
    }

    #[test]
    fn test_loss_dynamics_tracker_decreasing_loss() {
        let mut tracker = LossDynamicsTracker::new(10);
        let losses = vec![2.5, 2.4, 2.3, 2.2, 2.1, 2.0];

        for loss in losses {
            let (velocity, _) = tracker.update(loss);
            // Once we have enough data, velocity should be negative
            if tracker.history().len() >= 3 {
                assert!(
                    velocity <= 0.0,
                    "Velocity should be non-positive for decreasing loss sequence"
                );
            }
        }
    }

    #[test]
    fn test_loss_dynamics_tracker_window_size() {
        let mut tracker = LossDynamicsTracker::new(5);

        // Add more than window size elements
        for i in 0..10 {
            tracker.update(10.0 - i as f32);
        }

        // History should not exceed window size
        assert!(
            tracker.history().len() <= 5,
            "History length should not exceed window size"
        );
    }

    #[test]
    fn test_loss_dynamics_tracker_reset() {
        let mut tracker = LossDynamicsTracker::new(10);
        tracker.update(2.5);
        tracker.update(2.4);
        assert!(!tracker.history().is_empty());

        tracker.reset();
        assert!(tracker.history().is_empty());
    }

    #[test]
    fn test_loss_dynamics_tracker_acceleration() {
        let mut tracker = LossDynamicsTracker::new(20);

        // Simulate improving loss that is slowing down
        let losses = vec![3.0, 2.8, 2.6, 2.45, 2.35, 2.3, 2.28, 2.27, 2.26, 2.26];

        let mut last_acceleration = 0.0;
        for loss in losses {
            let (_, accel) = tracker.update(loss);
            last_acceleration = accel;
        }

        // When improvement slows, acceleration should become positive
        if tracker.history().len() >= 6 {
            assert!(
                last_acceleration >= 0.0,
                "Acceleration should be non-negative when improvement is slowing"
            );
        }
    }
}
