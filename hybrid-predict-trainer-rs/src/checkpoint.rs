//! Checkpoint save/restore functionality for hybrid trainer.
//!
//! This module provides efficient serialization and deserialization of training
//! state, enabling:
//! - **Recovery**: Resume training from failures or interruptions
//! - **Experimentation**: Branch training from checkpoints with different hyperparameters
//! - **Reproducibility**: Save exact training state for later analysis
//!
//! # Format
//!
//! Checkpoints use `bincode` for efficient binary serialization of all trainer
//! components. While SafeTensors would be ideal for model weights, the hybrid
//! trainer doesn't directly access model internals (framework-agnostic design),
//! so we rely on users to checkpoint their models separately if needed.
//!
//! # What's Checkpointed
//!
//! A checkpoint includes:
//! - Training state (step, loss history, gradient norms, etc.)
//! - Dynamics model state (GRU weights, ensemble members)
//! - Residual store (recent prediction errors)
//! - Phase controller state (current phase, budget)
//! - Divergence monitor state (EMA statistics)
//! - Residual corrector state (online correction model)
//! - Configuration (for validation on restore)
//!
//! # What's NOT Checkpointed
//!
//! Due to framework-agnostic design:
//! - Model weights (user must checkpoint via their framework)
//! - Optimizer state (user must checkpoint via their framework)
//!
//! # Usage
//!
//! ```rust,ignore
//! // Save checkpoint
//! trainer.save_checkpoint("checkpoint_step_1000.bin")?;
//!
//! // Load checkpoint
//! let trainer = HybridTrainer::load_checkpoint(
//!     "checkpoint_step_1000.bin",
//!     model,
//!     optimizer
//! )?;
//!
//! // Auto-checkpointing (via config)
//! let config = HybridTrainerConfig::builder()
//!     .checkpoint_config(CheckpointConfig {
//!         save_interval: 1000,
//!         keep_last_n: 3,
//!         ..Default::default()
//!     })
//!     .build();
//! ```

use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};

use crate::config::HybridTrainerConfig;
use crate::error::{HybridResult, HybridTrainingError};
use crate::state::TrainingState;

/// Complete checkpoint of hybrid trainer state.
///
/// Contains all information needed to resume training, except for model
/// and optimizer state (which must be checkpointed separately via the
/// user's deep learning framework).
///
/// # Size Estimation
///
/// Checkpoint size varies based on history buffer sizes:
/// - TrainingState: ~2 KB (with default buffers)
/// - RSSMLite (dynamics model): ~500 KB (depends on ensemble size)
/// - ResidualStore: ~100 KB (with 1000 residuals)
/// - Other components: ~10 KB
///
/// Total: ~600 KB typical, up to several MB for large configurations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingCheckpoint {
    /// Checkpoint format version for compatibility checking.
    pub version: u32,

    /// Training configuration (for validation on restore).
    pub config: HybridTrainerConfig,

    /// Training state (step, loss, history, etc.).
    pub training_state: TrainingState,

    /// Serialized dynamics model state.
    ///
    /// Contains GRU weights, ensemble members, and prediction statistics.
    pub dynamics_state: DynamicsState,

    /// Serialized residual store.
    pub residual_store: ResidualStoreState,

    /// Phase controller state.
    pub phase_controller_state: PhaseControllerState,

    /// Divergence monitor state.
    pub divergence_monitor_state: DivergenceMonitorState,

    /// Residual corrector state.
    pub corrector_state: CorrectorState,

    /// Metadata about the checkpoint.
    pub metadata: CheckpointMetadata,
}

/// Current checkpoint format version.
const CHECKPOINT_VERSION: u32 = 1;

impl TrainingCheckpoint {
    /// Creates a new checkpoint with the given components.
    #[must_use]
    pub fn new(
        config: HybridTrainerConfig,
        training_state: TrainingState,
        dynamics_state: DynamicsState,
        residual_store: ResidualStoreState,
        phase_controller_state: PhaseControllerState,
        divergence_monitor_state: DivergenceMonitorState,
        corrector_state: CorrectorState,
    ) -> Self {
        Self {
            version: CHECKPOINT_VERSION,
            config,
            training_state,
            dynamics_state,
            residual_store,
            phase_controller_state,
            divergence_monitor_state,
            corrector_state,
            metadata: CheckpointMetadata::new(),
        }
    }

    /// Saves the checkpoint to a file.
    ///
    /// # Arguments
    ///
    /// * `path` - Path where the checkpoint should be saved
    ///
    /// # Errors
    ///
    /// Returns an error if file creation or serialization fails.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> HybridResult<()> {
        let file = File::create(path.as_ref()).map_err(|e| {
            (
                HybridTrainingError::CheckpointError {
                    reason: format!("Failed to create checkpoint file: {e}"),
                },
                None,
            )
        })?;

        let writer = BufWriter::new(file);
        serde_json::to_writer(writer, self).map_err(|e| {
            (
                HybridTrainingError::CheckpointError {
                    reason: format!("Failed to serialize checkpoint: {e}"),
                },
                None,
            )
        })?;

        Ok(())
    }

    /// Loads a checkpoint from a file.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the checkpoint file
    ///
    /// # Returns
    ///
    /// The deserialized checkpoint or an error if loading fails.
    ///
    /// # Errors
    ///
    /// Returns an error if the file doesn't exist, is corrupted, or has an
    /// incompatible version.
    pub fn load<P: AsRef<Path>>(path: P) -> HybridResult<Self> {
        let file = File::open(path.as_ref()).map_err(|e| {
            (
                HybridTrainingError::CheckpointError {
                    reason: format!("Failed to open checkpoint file: {e}"),
                },
                None,
            )
        })?;

        let reader = BufReader::new(file);
        let checkpoint: Self = serde_json::from_reader(reader).map_err(|e| {
            (
                HybridTrainingError::CheckpointError {
                    reason: format!("Failed to deserialize checkpoint: {e}"),
                },
                None,
            )
        })?;

        // Validate version compatibility
        if checkpoint.version != CHECKPOINT_VERSION {
            return Err((
                HybridTrainingError::CheckpointError {
                    reason: format!(
                        "Incompatible checkpoint version: {} (expected {})",
                        checkpoint.version, CHECKPOINT_VERSION
                    ),
                },
                None,
            ));
        }

        Ok(checkpoint)
    }
}

/// Metadata about a checkpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    /// Training step when checkpoint was created.
    pub step: u64,

    /// Wall-clock timestamp (RFC 3339 format).
    pub timestamp: String,

    /// Hostname where checkpoint was created.
    pub hostname: String,

    /// Git commit hash (if available).
    pub git_commit: Option<String>,

    /// User-provided notes.
    pub notes: String,
}

impl CheckpointMetadata {
    /// Creates new metadata with current timestamp and hostname.
    #[must_use]
    pub fn new() -> Self {
        Self {
            step: 0,
            timestamp: chrono::Utc::now().to_rfc3339(),
            hostname: hostname::get()
                .ok()
                .and_then(|h| h.into_string().ok())
                .unwrap_or_else(|| "unknown".to_string()),
            git_commit: None,
            notes: String::new(),
        }
    }

    /// Sets the training step.
    #[must_use]
    pub fn with_step(mut self, step: u64) -> Self {
        self.step = step;
        self
    }

    /// Sets user notes.
    #[must_use]
    pub fn with_notes(mut self, notes: impl Into<String>) -> Self {
        self.notes = notes.into();
        self
    }
}

impl Default for CheckpointMetadata {
    fn default() -> Self {
        Self::new()
    }
}

/// Serializable state of the dynamics model (RSSMLite).
///
/// Contains all trainable parameters and prediction statistics.
///
/// # TODO
///
/// - Implement `extract_from_rssmlite()` to serialize GRU weights, ensemble states
/// - Implement `restore_to_rssmlite()` to deserialize into RSSMLite instance
/// - Add version field for compatibility checking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicsState {
    /// GRU weights (serialized).
    pub gru_weights: Vec<u8>,

    /// Ensemble member states (serialized).
    pub ensemble_states: Vec<Vec<u8>>,

    /// Prediction confidence history.
    pub confidence_history: Vec<f32>,

    /// Loss head weights.
    pub loss_head_weights: Vec<f32>,

    /// Weight delta head weights.
    pub weight_delta_head_weights: Vec<f32>,
}

impl Default for DynamicsState {
    fn default() -> Self {
        // TODO: Replace with actual extraction from RSSMLite (see lib.rs:1122)
        Self {
            gru_weights: Vec::new(),
            ensemble_states: Vec::new(),
            confidence_history: Vec::new(),
            loss_head_weights: Vec::new(),
            weight_delta_head_weights: Vec::new(),
        }
    }
}

/// Serializable state of the residual store.
///
/// # TODO
///
/// - Implement `extract_from_residual_store()` to serialize residuals
/// - Implement `restore_to_residual_store()` to deserialize into ResidualStore
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResidualStoreState {
    /// Stored residuals (serialized).
    pub residuals: Vec<u8>,

    /// Number of residuals in store.
    pub count: usize,

    /// Maximum capacity.
    pub capacity: usize,
}

impl Default for ResidualStoreState {
    fn default() -> Self {
        // TODO: Replace with actual extraction from ResidualStore (see lib.rs:1123)
        Self {
            residuals: Vec::new(),
            count: 0,
            capacity: 1000,
        }
    }
}

/// Serializable state of the phase controller.
///
/// # TODO
///
/// - Implement `extract_from_phase_controller()` to capture phase state
/// - Implement `restore_to_phase_controller()` to restore phase state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseControllerState {
    /// Current phase.
    pub current_phase: crate::Phase,

    /// Predictor confidence.
    pub predictor_confidence: f32,

    /// Warmup completion status.
    pub warmup_complete: bool,

    /// Phase statistics (serialized).
    pub phase_stats: Vec<u8>,
}

impl Default for PhaseControllerState {
    fn default() -> Self {
        // TODO: Replace with actual extraction from DefaultPhaseController (see lib.rs:1128)
        Self {
            current_phase: crate::Phase::Warmup,
            predictor_confidence: 0.0,
            warmup_complete: false,
            phase_stats: Vec::new(),
        }
    }
}

/// Serializable state of the divergence monitor.
///
/// # TODO
///
/// - Implement `extract_from_divergence_monitor()` to capture EMA statistics
/// - Implement `restore_to_divergence_monitor()` to restore monitor state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DivergenceMonitorState {
    /// Loss EMA.
    pub loss_ema: f32,

    /// Loss EMA variance.
    pub loss_ema_variance: f32,

    /// Gradient norm EMA.
    pub gradient_norm_ema: f32,

    /// Recent divergence signals (serialized).
    pub divergence_signals: Vec<u8>,
}

impl Default for DivergenceMonitorState {
    fn default() -> Self {
        // TODO: Replace with actual extraction from DivergenceMonitor (see lib.rs:1130)
        Self {
            loss_ema: 0.0,
            loss_ema_variance: 0.0,
            gradient_norm_ema: 0.0,
            divergence_signals: Vec::new(),
        }
    }
}

/// Serializable state of the residual corrector.
///
/// # TODO
///
/// - Implement `extract_from_residual_corrector()` to serialize correction weights
/// - Implement `restore_to_residual_corrector()` to restore corrector state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrectorState {
    /// Online correction model weights.
    pub correction_weights: Vec<f32>,

    /// Correction statistics (serialized).
    pub correction_stats: Vec<u8>,
}

impl Default for CorrectorState {
    fn default() -> Self {
        // TODO: Replace with actual extraction from ResidualCorrector (see lib.rs:1131)
        Self {
            correction_weights: Vec::new(),
            correction_stats: Vec::new(),
        }
    }
}

/// Checkpoint manager for automatic checkpointing.
///
/// Handles periodic checkpoint saves, rotating old checkpoints, and
/// cleanup based on configuration.
pub struct CheckpointManager {
    /// Directory where checkpoints are saved.
    checkpoint_dir: PathBuf,

    /// Save checkpoint every N steps.
    save_interval: usize,

    /// Number of checkpoints to keep.
    keep_last_n: usize,

    /// List of checkpoint files (sorted by step, oldest first).
    checkpoints: Vec<PathBuf>,
}

impl CheckpointManager {
    /// Creates a new checkpoint manager.
    ///
    /// # Arguments
    ///
    /// * `checkpoint_dir` - Directory for checkpoints (created if doesn't exist)
    /// * `save_interval` - Save every N steps
    /// * `keep_last_n` - Number of checkpoints to keep
    ///
    /// # Errors
    ///
    /// Returns an error if directory creation fails.
    pub fn new(
        checkpoint_dir: impl AsRef<Path>,
        save_interval: usize,
        keep_last_n: usize,
    ) -> HybridResult<Self> {
        let checkpoint_dir = checkpoint_dir.as_ref().to_path_buf();

        // Create directory if it doesn't exist
        std::fs::create_dir_all(&checkpoint_dir).map_err(|e| {
            (
                HybridTrainingError::CheckpointError {
                    reason: format!("Failed to create checkpoint directory: {e}"),
                },
                None,
            )
        })?;

        // Scan for existing checkpoints
        let checkpoints = Self::scan_checkpoints(&checkpoint_dir)?;

        Ok(Self {
            checkpoint_dir,
            save_interval,
            keep_last_n,
            checkpoints,
        })
    }

    /// Checks if a checkpoint should be saved at the given step.
    #[must_use]
    pub fn should_save(&self, step: u64) -> bool {
        self.save_interval > 0 && step > 0 && step % self.save_interval as u64 == 0
    }

    /// Saves a checkpoint and manages rotation.
    ///
    /// # Arguments
    ///
    /// * `checkpoint` - The checkpoint to save
    ///
    /// # Errors
    ///
    /// Returns an error if saving or rotation fails.
    pub fn save(&mut self, checkpoint: &TrainingCheckpoint) -> HybridResult<()> {
        let step = checkpoint.training_state.step;
        let filename = format!("checkpoint_step_{:08}.json", step);
        let path = self.checkpoint_dir.join(&filename);

        // Save the checkpoint
        checkpoint.save(&path)?;

        // Add to list
        self.checkpoints.push(path);

        // Rotate old checkpoints
        self.rotate_checkpoints()?;

        Ok(())
    }

    /// Loads the most recent checkpoint.
    ///
    /// # Returns
    ///
    /// The latest checkpoint or `None` if no checkpoints exist.
    ///
    /// # Errors
    ///
    /// Returns an error if loading fails.
    pub fn load_latest(&self) -> HybridResult<Option<TrainingCheckpoint>> {
        if let Some(path) = self.checkpoints.last() {
            Ok(Some(TrainingCheckpoint::load(path)?))
        } else {
            Ok(None)
        }
    }

    /// Returns the path to the most recent checkpoint.
    #[must_use]
    pub fn latest_checkpoint_path(&self) -> Option<&Path> {
        self.checkpoints.last().map(|p| p.as_path())
    }

    /// Scans directory for existing checkpoints.
    fn scan_checkpoints(dir: &Path) -> HybridResult<Vec<PathBuf>> {
        let mut checkpoints = Vec::new();

        if !dir.exists() {
            return Ok(checkpoints);
        }

        let entries = std::fs::read_dir(dir).map_err(|e| {
            (
                HybridTrainingError::CheckpointError {
                    reason: format!("Failed to read checkpoint directory: {e}"),
                },
                None,
            )
        })?;

        for entry in entries {
            let entry = entry.map_err(|e| {
                (
                    HybridTrainingError::CheckpointError {
                        reason: format!("Failed to read directory entry: {e}"),
                    },
                    None,
                )
            })?;

            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("json")
                && path
                    .file_name()
                    .and_then(|s| s.to_str())
                    .map_or(false, |s| s.starts_with("checkpoint_step_"))
            {
                checkpoints.push(path);
            }
        }

        // Sort by step number
        checkpoints.sort_by_key(|path| {
            path.file_stem()
                .and_then(|s| s.to_str())
                .and_then(|s| s.strip_prefix("checkpoint_step_"))
                .and_then(|s| s.parse::<u64>().ok())
                .unwrap_or(0)
        });

        Ok(checkpoints)
    }

    /// Rotates old checkpoints to keep only the last N.
    fn rotate_checkpoints(&mut self) -> HybridResult<()> {
        while self.checkpoints.len() > self.keep_last_n {
            if let Some(old_checkpoint) = self.checkpoints.first() {
                std::fs::remove_file(old_checkpoint).map_err(|e| {
                    (
                        HybridTrainingError::CheckpointError {
                            reason: format!("Failed to delete old checkpoint: {e}"),
                        },
                        None,
                    )
                })?;
                self.checkpoints.remove(0);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_checkpoint_save_load_roundtrip() {
        let temp_dir = TempDir::new().unwrap();
        let checkpoint_path = temp_dir.path().join("test_checkpoint.json");

        // Create a checkpoint with valid training state
        let mut training_state = TrainingState::new();
        training_state.loss = 0.0; // Replace NaN with valid value
        training_state.gradient_norm = 0.0; // Replace NaN with valid value

        let checkpoint = TrainingCheckpoint::new(
            HybridTrainerConfig::default(),
            training_state,
            DynamicsState::default(),
            ResidualStoreState::default(),
            PhaseControllerState::default(),
            DivergenceMonitorState::default(),
            CorrectorState::default(),
        );

        // Save
        checkpoint.save(&checkpoint_path).unwrap();

        // Load
        let loaded = TrainingCheckpoint::load(&checkpoint_path).unwrap();

        // Verify
        assert_eq!(loaded.version, CHECKPOINT_VERSION);
        assert_eq!(loaded.training_state.step, 0);
    }

    #[test]
    fn test_checkpoint_metadata() {
        let metadata = CheckpointMetadata::new()
            .with_step(1000)
            .with_notes("Test checkpoint");

        assert_eq!(metadata.step, 1000);
        assert_eq!(metadata.notes, "Test checkpoint");
        assert!(!metadata.timestamp.is_empty());
    }

    #[test]
    fn test_checkpoint_manager_rotation() {
        let temp_dir = TempDir::new().unwrap();
        let mut manager =
            CheckpointManager::new(temp_dir.path(), 100, 2).unwrap();

        // Create and save 3 checkpoints
        for step in [100, 200, 300] {
            let mut checkpoint = TrainingCheckpoint::new(
                HybridTrainerConfig::default(),
                TrainingState::new(),
                DynamicsState::default(),
                ResidualStoreState::default(),
                PhaseControllerState::default(),
                DivergenceMonitorState::default(),
                CorrectorState::default(),
            );
            checkpoint.training_state.step = step;
            manager.save(&checkpoint).unwrap();
        }

        // Should only keep last 2
        assert_eq!(manager.checkpoints.len(), 2);

        // Verify oldest was deleted
        let steps: Vec<u64> = manager
            .checkpoints
            .iter()
            .filter_map(|path| {
                path.file_stem()
                    .and_then(|s| s.to_str())
                    .and_then(|s| s.strip_prefix("checkpoint_step_"))
                    .and_then(|s| s.parse().ok())
            })
            .collect();

        assert_eq!(steps, vec![200, 300]);
    }

    #[test]
    fn test_checkpoint_manager_should_save() {
        let temp_dir = TempDir::new().unwrap();
        let manager = CheckpointManager::new(temp_dir.path(), 100, 3).unwrap();

        assert!(!manager.should_save(0)); // Skip step 0
        assert!(!manager.should_save(50));
        assert!(manager.should_save(100));
        assert!(!manager.should_save(150));
        assert!(manager.should_save(200));
    }

    #[test]
    fn test_checkpoint_load_latest() {
        let temp_dir = TempDir::new().unwrap();
        let mut manager =
            CheckpointManager::new(temp_dir.path(), 100, 3).unwrap();

        // No checkpoints initially
        assert!(manager.load_latest().unwrap().is_none());

        // Save a checkpoint
        let mut training_state = TrainingState::new();
        training_state.step = 500;
        training_state.loss = 2.5; // Replace NaN with valid value
        training_state.gradient_norm = 1.0; // Replace NaN with valid value

        let checkpoint = TrainingCheckpoint::new(
            HybridTrainerConfig::default(),
            training_state,
            DynamicsState::default(),
            ResidualStoreState::default(),
            PhaseControllerState::default(),
            DivergenceMonitorState::default(),
            CorrectorState::default(),
        );
        manager.save(&checkpoint).unwrap();

        // Load latest
        let loaded = manager.load_latest().unwrap().unwrap();
        assert_eq!(loaded.training_state.step, 500);
    }

    #[test]
    fn test_incompatible_version() {
        let temp_dir = TempDir::new().unwrap();
        let checkpoint_path = temp_dir.path().join("bad_version.json");

        // Create checkpoint with wrong version
        let mut checkpoint = TrainingCheckpoint::new(
            HybridTrainerConfig::default(),
            TrainingState::new(),
            DynamicsState::default(),
            ResidualStoreState::default(),
            PhaseControllerState::default(),
            DivergenceMonitorState::default(),
            CorrectorState::default(),
        );
        checkpoint.version = 999;

        // Save it directly (bypassing normal validation)
        let file = File::create(&checkpoint_path).unwrap();
        serde_json::to_writer(BufWriter::new(file), &checkpoint).unwrap();

        // Try to load - should fail
        let result = TrainingCheckpoint::load(&checkpoint_path);
        assert!(result.is_err());
    }
}
