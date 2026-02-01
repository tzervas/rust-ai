//! Checkpoint management with compression and HuggingFace upload.
//!
//! Provides utilities for:
//! - Saving and loading model checkpoints
//! - Compressing metadata and residuals
//! - Uploading checkpoints to HuggingFace
//! - Cleaning up local files after upload

use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use serde::{Deserialize, Serialize};

use crate::hf::{HuggingFaceUploader, ModelCardData};
use crate::training_state::TrainingRun;

/// Checkpoint metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    /// Step at which checkpoint was created
    pub step: u64,
    /// Loss at checkpoint
    pub loss: f32,
    /// Best loss seen so far
    pub best_loss: f32,
    /// Training phase
    pub phase: String,
    /// Total forward passes
    pub total_forward: u64,
    /// Total backward passes
    pub total_backward: u64,
    /// Timestamp
    pub timestamp: String,
    /// Model configuration hash (for compatibility check)
    pub config_hash: String,
    /// Optimizer state included
    pub has_optimizer_state: bool,
    /// Compressed size in bytes
    pub compressed_size: Option<u64>,
}

/// Residual data from predictive training.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResidualData {
    /// Step range covered
    pub step_range: (u64, u64),
    /// Prediction errors per step
    pub prediction_errors: Vec<f32>,
    /// Correction magnitudes
    pub correction_magnitudes: Vec<f32>,
    /// Compressed
    pub compressed: bool,
}

/// Checkpoint save options.
#[derive(Debug, Clone)]
pub struct CheckpointOptions {
    /// Save optimizer state
    pub save_optimizer: bool,
    /// Compress checkpoint
    pub compress: bool,
    /// Compression level (0-9)
    pub compression_level: u32,
    /// Upload to HuggingFace after save
    pub upload_to_hf: bool,
    /// Delete local after upload
    pub delete_after_upload: bool,
    /// Keep last N local checkpoints
    pub keep_last_n: usize,
}

impl Default for CheckpointOptions {
    fn default() -> Self {
        Self {
            save_optimizer: true,
            compress: true,
            compression_level: 6,
            upload_to_hf: false,
            delete_after_upload: false,
            keep_last_n: 3,
        }
    }
}

/// Manages model checkpoints with compression and cleanup.
pub struct CheckpointManager {
    /// Base directory for checkpoints
    checkpoints_dir: PathBuf,
    /// Directory for compressed residuals
    residuals_dir: PathBuf,
    /// HuggingFace uploader (optional)
    hf_uploader: Option<HuggingFaceUploader>,
    /// Default options
    options: CheckpointOptions,
    /// Tracked checkpoints
    checkpoints: Vec<CheckpointMetadata>,
}

impl CheckpointManager {
    /// Create a new checkpoint manager.
    pub fn new(
        checkpoints_dir: PathBuf,
        hf_uploader: Option<HuggingFaceUploader>,
        options: CheckpointOptions,
    ) -> anyhow::Result<Self> {
        let residuals_dir = checkpoints_dir.join("residuals");
        fs::create_dir_all(&checkpoints_dir)?;
        fs::create_dir_all(&residuals_dir)?;

        let mut manager = Self {
            checkpoints_dir,
            residuals_dir,
            hf_uploader,
            options,
            checkpoints: Vec::new(),
        };

        // Load existing checkpoints
        manager.discover_checkpoints()?;

        Ok(manager)
    }

    /// Discover existing checkpoints in the directory.
    fn discover_checkpoints(&mut self) -> anyhow::Result<()> {
        self.checkpoints.clear();

        for entry in fs::read_dir(&self.checkpoints_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_file() {
                if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                    if name.starts_with("checkpoint_") && name.ends_with(".meta.json") {
                        if let Ok(content) = fs::read_to_string(&path) {
                            if let Ok(meta) = serde_json::from_str::<CheckpointMetadata>(&content) {
                                self.checkpoints.push(meta);
                            }
                        }
                    }
                }
            }
        }

        // Sort by step
        self.checkpoints.sort_by_key(|c| c.step);

        Ok(())
    }

    /// Get checkpoint path for a step.
    fn checkpoint_path(&self, step: u64, compressed: bool) -> PathBuf {
        let ext = if compressed {
            "safetensors.gz"
        } else {
            "safetensors"
        };
        self.checkpoints_dir
            .join(format!("checkpoint_{}.{}", step, ext))
    }

    /// Get metadata path for a step.
    fn metadata_path(&self, step: u64) -> PathBuf {
        self.checkpoints_dir
            .join(format!("checkpoint_{}.meta.json", step))
    }

    /// Get optimizer state path for a step.
    fn optimizer_path(&self, step: u64, compressed: bool) -> PathBuf {
        let ext = if compressed {
            "optimizer.gz"
        } else {
            "optimizer"
        };
        self.checkpoints_dir
            .join(format!("checkpoint_{}.{}", step, ext))
    }

    /// Save a checkpoint.
    pub fn save_checkpoint(
        &mut self,
        run: &TrainingRun,
        model_weights: &[u8],
        optimizer_state: Option<&[u8]>,
    ) -> anyhow::Result<PathBuf> {
        let step = run.current_step;
        let compressed = self.options.compress;

        // Save model weights
        let model_path = self.checkpoint_path(step, compressed);
        if compressed {
            self.compress_and_write(&model_path, model_weights)?;
        } else {
            fs::write(&model_path, model_weights)?;
        }

        // Save optimizer state if provided
        if self.options.save_optimizer {
            if let Some(opt_state) = optimizer_state {
                let opt_path = self.optimizer_path(step, compressed);
                if compressed {
                    self.compress_and_write(&opt_path, opt_state)?;
                } else {
                    fs::write(&opt_path, opt_state)?;
                }
            }
        }

        // Create metadata
        let compressed_size = if compressed {
            Some(fs::metadata(&model_path)?.len())
        } else {
            None
        };

        let metadata = CheckpointMetadata {
            step,
            loss: run.current_loss,
            best_loss: run.best_loss,
            phase: format!("{:?}", run.current_phase),
            total_forward: run.total_forward,
            total_backward: run.total_backward,
            timestamp: chrono::Utc::now().to_rfc3339(),
            config_hash: self.compute_config_hash(&run.config),
            has_optimizer_state: optimizer_state.is_some() && self.options.save_optimizer,
            compressed_size,
        };

        // Save metadata
        let meta_path = self.metadata_path(step);
        fs::write(&meta_path, serde_json::to_string_pretty(&metadata)?)?;

        self.checkpoints.push(metadata);

        // Cleanup old checkpoints
        self.cleanup_old_checkpoints()?;

        // Upload to HuggingFace if configured
        if self.options.upload_to_hf {
            if let Some(ref uploader) = self.hf_uploader {
                // Would need model card data - skip for now
                tracing::info!("HuggingFace upload configured but skipped (need model card)");
            }
        }

        Ok(model_path)
    }

    /// Load a checkpoint.
    pub fn load_checkpoint(&self, step: u64) -> anyhow::Result<(Vec<u8>, Option<Vec<u8>>)> {
        // Try compressed first, then uncompressed
        let model_path = self.checkpoint_path(step, true);
        let model_weights = if model_path.exists() {
            self.decompress_and_read(&model_path)?
        } else {
            let uncompressed_path = self.checkpoint_path(step, false);
            fs::read(&uncompressed_path)?
        };

        // Load optimizer if exists
        let opt_path = self.optimizer_path(step, true);
        let optimizer_state = if opt_path.exists() {
            Some(self.decompress_and_read(&opt_path)?)
        } else {
            let uncompressed_opt = self.optimizer_path(step, false);
            if uncompressed_opt.exists() {
                Some(fs::read(&uncompressed_opt)?)
            } else {
                None
            }
        };

        Ok((model_weights, optimizer_state))
    }

    /// Get the latest checkpoint step.
    pub fn latest_checkpoint(&self) -> Option<u64> {
        self.checkpoints.last().map(|c| c.step)
    }

    /// Get all checkpoint steps.
    pub fn checkpoint_steps(&self) -> Vec<u64> {
        self.checkpoints.iter().map(|c| c.step).collect()
    }

    /// Get checkpoint metadata.
    pub fn get_metadata(&self, step: u64) -> Option<&CheckpointMetadata> {
        self.checkpoints.iter().find(|c| c.step == step)
    }

    /// Compress and write data to file.
    fn compress_and_write(&self, path: &Path, data: &[u8]) -> anyhow::Result<()> {
        let file = File::create(path)?;
        let mut encoder =
            GzEncoder::new(file, Compression::new(self.options.compression_level));
        encoder.write_all(data)?;
        encoder.finish()?;
        Ok(())
    }

    /// Read and decompress data from file.
    fn decompress_and_read(&self, path: &Path) -> anyhow::Result<Vec<u8>> {
        let file = File::open(path)?;
        let mut decoder = GzDecoder::new(file);
        let mut data = Vec::new();
        decoder.read_to_end(&mut data)?;
        Ok(data)
    }

    /// Compute a hash of the model config for compatibility checking.
    fn compute_config_hash(
        &self,
        config: &crate::training_state::TrainingConfig,
    ) -> String {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        config.hidden_size.hash(&mut hasher);
        config.num_layers.hash(&mut hasher);
        config.num_heads.hash(&mut hasher);
        format!("{:016x}", hasher.finish())
    }

    /// Cleanup old checkpoints, keeping only the last N.
    fn cleanup_old_checkpoints(&mut self) -> anyhow::Result<()> {
        if self.checkpoints.len() <= self.options.keep_last_n {
            return Ok(());
        }

        let to_remove = self.checkpoints.len() - self.options.keep_last_n;
        let removed: Vec<_> = self.checkpoints.drain(..to_remove).collect();

        for meta in removed {
            // Delete checkpoint files
            for compressed in [true, false] {
                let model_path = self.checkpoint_path(meta.step, compressed);
                if model_path.exists() {
                    fs::remove_file(&model_path)?;
                }

                let opt_path = self.optimizer_path(meta.step, compressed);
                if opt_path.exists() {
                    fs::remove_file(&opt_path)?;
                }
            }

            // Delete metadata
            let meta_path = self.metadata_path(meta.step);
            if meta_path.exists() {
                fs::remove_file(&meta_path)?;
            }

            tracing::info!("Cleaned up checkpoint at step {}", meta.step);
        }

        Ok(())
    }

    /// Save residual data with compression.
    pub fn save_residuals(&self, residuals: &ResidualData) -> anyhow::Result<PathBuf> {
        let filename = format!(
            "residuals_{}_{}.json.gz",
            residuals.step_range.0, residuals.step_range.1
        );
        let path = self.residuals_dir.join(filename);

        let json = serde_json::to_string(residuals)?;
        self.compress_and_write(&path, json.as_bytes())?;

        Ok(path)
    }

    /// Load residual data.
    pub fn load_residuals(&self, step_start: u64, step_end: u64) -> anyhow::Result<ResidualData> {
        let filename = format!("residuals_{}_{}.json.gz", step_start, step_end);
        let path = self.residuals_dir.join(filename);

        let data = self.decompress_and_read(&path)?;
        let residuals: ResidualData = serde_json::from_slice(&data)?;

        Ok(residuals)
    }

    /// Get total disk usage of checkpoints.
    pub fn total_disk_usage(&self) -> anyhow::Result<u64> {
        let mut total = 0u64;

        for entry in fs::read_dir(&self.checkpoints_dir)? {
            let entry = entry?;
            if entry.path().is_file() {
                total += entry.metadata()?.len();
            }
        }

        for entry in fs::read_dir(&self.residuals_dir)? {
            let entry = entry?;
            if entry.path().is_file() {
                total += entry.metadata()?.len();
            }
        }

        Ok(total)
    }

    /// Upload checkpoint to HuggingFace and optionally delete local.
    pub fn upload_and_cleanup(
        &mut self,
        step: u64,
        run: &TrainingRun,
        model_card: &ModelCardData,
    ) -> anyhow::Result<Option<String>> {
        let uploader = match &self.hf_uploader {
            Some(u) => u,
            None => return Ok(None),
        };

        // Find checkpoint path
        let checkpoint_path = if self.checkpoint_path(step, true).exists() {
            self.checkpoint_path(step, true)
        } else if self.checkpoint_path(step, false).exists() {
            self.checkpoint_path(step, false)
        } else {
            anyhow::bail!("Checkpoint not found for step {}", step);
        };

        // Upload
        let repo_id = uploader.upload_checkpoint(run, &checkpoint_path, model_card)?;

        // Delete local if configured
        if self.options.delete_after_upload {
            for compressed in [true, false] {
                let model_path = self.checkpoint_path(step, compressed);
                if model_path.exists() {
                    fs::remove_file(&model_path)?;
                }

                let opt_path = self.optimizer_path(step, compressed);
                if opt_path.exists() {
                    fs::remove_file(&opt_path)?;
                }
            }

            let meta_path = self.metadata_path(step);
            if meta_path.exists() {
                fs::remove_file(&meta_path)?;
            }

            // Remove from tracked checkpoints
            self.checkpoints.retain(|c| c.step != step);

            tracing::info!(
                "Uploaded and cleaned up checkpoint at step {} -> {}",
                step,
                repo_id
            );
        }

        Ok(Some(repo_id))
    }
}
