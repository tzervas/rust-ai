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
}
