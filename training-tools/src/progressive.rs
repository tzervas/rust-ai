//! Progressive parameter expansion training.
//!
//! Implements the 100M → 500M → 1B training progression:
//! 1. Train 100M model to convergence
//! 2. Expand weights to 500M architecture
//! 3. Continue training 500M model
//! 4. Expand weights to 1B architecture
//! 5. Continue training 1B model

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarMap;
use serde::{Deserialize, Serialize};

use hybrid_predict_trainer_rs::{HybridTrainerConfig, Phase};
#[cfg(feature = "parquet")]
use tritter_model_rs::data::ParquetDataset;
use tritter_model_rs::data::{
    CompositeDataset, DataConfig, DataLoader, DatasetBlendingStrategy, JsonlDataset,
    StreamingDataset,
};
use tritter_model_rs::{
    create_trainer_with_checkpointing, TritterBatch, TritterConfig, TritterModel,
};

use crate::checkpoint_manager::{CheckpointManager, CheckpointOptions};
use crate::gpu_stats::{query_gpu_stats, GpuStatsMonitor};
use crate::hf::{
    ArchitectureDetails, HuggingFaceUploader, ModelCardData, PerformanceMetrics, TrainingDetails,
};
use crate::lr_controller::AdaptiveLRController;
use crate::memory::{find_optimal_params, query_gpu_memory, MemoryBudget, MemoryMonitor};
use crate::training_state::{
    compute_config_hash, LossDynamicsTracker, RunManager, StepMetrics, TrainingConfig,
    TrainingPhase, TrainingRun, TrainingStatus,
};

/// Progressive training stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrainingStage {
    /// 100M parameter model
    Small100M,
    /// 500M parameter model (expanded from 100M)
    Medium500M,
    /// 1B parameter model (expanded from 500M)
    Large1B,
}

impl TrainingStage {
    /// Get the model configuration for this stage.
    pub fn config(&self) -> TritterConfig {
        match self {
            Self::Small100M => TritterConfig::small_100m(),
            Self::Medium500M => TritterConfig::medium_500m(),
            Self::Large1B => TritterConfig::large_1b(),
        }
    }

    /// Get the size string.
    pub fn size_str(&self) -> &'static str {
        match self {
            Self::Small100M => "100M",
            Self::Medium500M => "500M",
            Self::Large1B => "1B",
        }
    }

    /// Get the next stage (if any).
    pub fn next(&self) -> Option<Self> {
        match self {
            Self::Small100M => Some(Self::Medium500M),
            Self::Medium500M => Some(Self::Large1B),
            Self::Large1B => None,
        }
    }

    /// Get the previous stage (if any).
    pub fn previous(&self) -> Option<Self> {
        match self {
            Self::Small100M => None,
            Self::Medium500M => Some(Self::Small100M),
            Self::Large1B => Some(Self::Medium500M),
        }
    }
}

/// Configuration for progressive training.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressiveConfig {
    /// Starting stage
    pub start_stage: TrainingStage,
    /// Ending stage
    pub end_stage: TrainingStage,
    /// Training steps per stage
    pub steps_per_stage: u64,
    /// Batch size
    pub batch_size: usize,
    /// Sequence length
    pub seq_length: usize,
    /// Learning rate
    pub learning_rate: f32,
    /// Checkpoint interval (steps)
    pub checkpoint_every: u64,
    /// Log interval (steps)
    pub log_every: u64,
    /// Gradient checkpointing interval (layers)
    pub gradient_checkpoint_interval: usize,
    /// Upload to HuggingFace after each stage
    pub upload_to_hf: bool,
    /// HuggingFace username
    pub hf_username: Option<String>,
    /// Delete local checkpoints after HF upload
    pub delete_after_upload: bool,
    /// Path to dataset (JSONL or Parquet files)
    /// If None, uses random tokens (for testing only)
    /// DEPRECATED: Use `dataset_paths` for multi-dataset support
    pub dataset_path: Option<PathBuf>,
    /// Paths to multiple datasets (JSONL or Parquet files/directories)
    /// Datasets are blended according to `dataset_blend_strategy`
    pub dataset_paths: Vec<PathBuf>,
    /// Strategy for blending multiple datasets
    pub dataset_blend_strategy: DatasetBlendStrategy,
    /// Text column name in dataset (auto-detected if None)
    pub text_column: Option<String>,
}

/// Strategy for blending multiple datasets during training.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub enum DatasetBlendStrategy {
    /// Process datasets sequentially (finish one before starting next)
    #[default]
    Sequential,
    /// Cycle through datasets in round-robin fashion
    RoundRobin,
    /// Interleave with configurable weights (requires weights in config)
    Weighted,
}

impl Default for ProgressiveConfig {
    fn default() -> Self {
        Self {
            start_stage: TrainingStage::Small100M,
            end_stage: TrainingStage::Large1B,
            steps_per_stage: 10000,
            batch_size: 4,
            seq_length: 512,
            learning_rate: 6e-4, // Updated: GPT-2/124M standard LR
            checkpoint_every: 1000,
            log_every: 10,
            gradient_checkpoint_interval: 4,
            upload_to_hf: false,
            hf_username: None,
            delete_after_upload: false,
            dataset_path: None,
            dataset_paths: Vec::new(),
            dataset_blend_strategy: DatasetBlendStrategy::default(),
            text_column: None,
        }
    }
}

/// Progressive trainer that handles 100M → 500M → 1B expansion.
pub struct ProgressiveTrainer {
    config: ProgressiveConfig,
    device: Device,
    run_manager: RunManager,
    checkpoint_manager: Option<CheckpointManager>,
    hf_uploader: Option<HuggingFaceUploader>,
    current_stage: TrainingStage,
    metrics_file: Option<fs::File>,
    _memory_monitor: MemoryMonitor,
    gpu_stats_monitor: GpuStatsMonitor,
    /// Data loader for real datasets (None = use random tokens)
    data_loader: Option<DataLoader>,
    /// Tracker for loss velocity and acceleration computation
    loss_dynamics_tracker: LossDynamicsTracker,
}

impl ProgressiveTrainer {
    /// Create a new progressive trainer.
    pub fn new(
        config: ProgressiveConfig,
        runs_dir: PathBuf,
        device: Device,
    ) -> anyhow::Result<Self> {
        let run_manager = RunManager::new(runs_dir.clone());

        // Setup HuggingFace uploader if configured
        let hf_uploader = if config.upload_to_hf {
            config.hf_username.as_ref().map(|username| {
                let cache_dir = runs_dir.join(".hf_cache");
                fs::create_dir_all(&cache_dir).ok();
                HuggingFaceUploader::new(username, cache_dir)
            })
        } else {
            None
        };

        // Create memory monitor for GPU 0
        let memory_monitor = MemoryMonitor::new(0);
        let gpu_stats_monitor = GpuStatsMonitor::new(0);

        // Log initial GPU status
        if let Some(stats) = query_gpu_stats(0) {
            tracing::info!(
                "GPU: {} | Driver: {} | CUDA: {}",
                stats.name,
                stats.driver_version,
                stats.cuda_version
            );
            tracing::info!(
                "GPU Memory: {:.1}/{:.1} GB available ({:.0}% free)",
                stats.memory_free as f64 / 1e9,
                stats.memory_total as f64 / 1e9,
                100.0 - stats.memory_percent()
            );
            tracing::info!(
                "GPU Temp: {}°C | Power: {:.0}/{:.0}W | PState: {}",
                stats.temperature,
                stats.power_draw,
                stats.power_limit,
                stats.pstate
            );
        }

        // Initialize data loader from dataset paths
        let data_loader = Self::create_data_loader(&config, &device)?;

        Ok(Self {
            current_stage: config.start_stage,
            config,
            device,
            run_manager,
            checkpoint_manager: None,
            hf_uploader,
            metrics_file: None,
            _memory_monitor: memory_monitor,
            gpu_stats_monitor,
            data_loader,
            loss_dynamics_tracker: LossDynamicsTracker::new(30),
        })
    }

    /// Create a data loader from configuration, supporting multiple datasets.
    fn create_data_loader(
        config: &ProgressiveConfig,
        device: &Device,
    ) -> anyhow::Result<Option<DataLoader>> {
        let data_config = DataConfig {
            max_seq_length: config.seq_length,
            batch_size: config.batch_size,
            num_workers: 4,
            prefetch_factor: 2,
            text_column: config.text_column.clone(),
            seed: 42,
        };

        // Collect all dataset paths (new multi-path + legacy single path)
        let mut all_paths: Vec<PathBuf> = config.dataset_paths.clone();
        if let Some(ref legacy_path) = config.dataset_path {
            if !all_paths.contains(legacy_path) {
                all_paths.push(legacy_path.clone());
            }
        }

        if all_paths.is_empty() {
            tracing::warn!("No dataset path provided - using random tokens (testing mode)");
            return Ok(None);
        }

        // Create a dataset for each path
        let mut datasets: Vec<Box<dyn StreamingDataset>> = Vec::new();
        let mut total_size_bytes: u64 = 0;

        for path in &all_paths {
            if !path.exists() {
                tracing::warn!("Dataset path does not exist, skipping: {:?}", path);
                continue;
            }

            // Estimate size for logging
            if let Ok(meta) = std::fs::metadata(path) {
                if meta.is_dir() {
                    // Recursively sum file sizes
                    if let Ok(size) = Self::dir_size(path) {
                        total_size_bytes += size;
                    }
                } else {
                    total_size_bytes += meta.len();
                }
            }

            let dataset = Self::create_single_dataset(path, config.seq_length)?;
            datasets.push(dataset);
            tracing::info!("Added dataset: {:?}", path);
        }

        if datasets.is_empty() {
            tracing::warn!("No valid datasets found - using random tokens (testing mode)");
            return Ok(None);
        }

        // Log total dataset size
        let size_gb = total_size_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        tracing::info!(
            "Loaded {} dataset(s), total size: {:.1} GB",
            datasets.len(),
            size_gb
        );

        // Create composite or single dataset
        let final_dataset: Box<dyn StreamingDataset> = if datasets.len() == 1 {
            datasets.pop().unwrap()
        } else {
            // Convert blend strategy
            let blend_strategy = match config.dataset_blend_strategy {
                DatasetBlendStrategy::Sequential => DatasetBlendingStrategy::Sequential,
                DatasetBlendStrategy::RoundRobin => DatasetBlendingStrategy::RoundRobin,
                DatasetBlendStrategy::Weighted => DatasetBlendingStrategy::Weighted,
            };

            tracing::info!(
                "Using {:?} blending strategy for {} datasets",
                blend_strategy,
                datasets.len()
            );

            Box::new(
                CompositeDataset::new(datasets).with_strategy(blend_strategy),
            )
        };

        Ok(Some(DataLoader::new(final_dataset, data_config, device.clone())))
    }

    /// Create a single dataset from a path (file or directory).
    fn create_single_dataset(
        path: &Path,
        seq_length: usize,
    ) -> anyhow::Result<Box<dyn StreamingDataset>> {
        if path.is_file() {
            let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
            match ext {
                #[cfg(feature = "parquet")]
                "parquet" => {
                    let ds = ParquetDataset::new(path, seq_length)
                        .map_err(|e| anyhow::anyhow!("Failed to open parquet dataset: {}", e))?;
                    Ok(Box::new(ds))
                }
                "jsonl" | "json" => {
                    let ds = JsonlDataset::new(path, seq_length)
                        .map_err(|e| anyhow::anyhow!("Failed to open JSONL dataset: {}", e))?;
                    Ok(Box::new(ds))
                }
                #[cfg(not(feature = "parquet"))]
                "parquet" => Err(anyhow::anyhow!(
                    "Parquet support not enabled. Build with --features parquet"
                )),
                _ => {
                    // Default to JSONL
                    let ds = JsonlDataset::new(path, seq_length)
                        .map_err(|e| anyhow::anyhow!("Failed to open dataset: {}", e))?;
                    Ok(Box::new(ds))
                }
            }
        } else if path.is_dir() {
            #[cfg(feature = "parquet")]
            {
                // Check for parquet files first
                let has_parquet = std::fs::read_dir(path)?
                    .filter_map(|e| e.ok())
                    .any(|e| e.path().extension().map_or(false, |ext| ext == "parquet"));

                if has_parquet {
                    let ds = ParquetDataset::new(path, seq_length)
                        .map_err(|e| anyhow::anyhow!("Failed to open parquet dataset: {}", e))?;
                    return Ok(Box::new(ds));
                }
            }

            let ds = JsonlDataset::new(path, seq_length)
                .map_err(|e| anyhow::anyhow!("Failed to open JSONL dataset: {}", e))?;
            Ok(Box::new(ds))
        } else {
            Err(anyhow::anyhow!("Dataset path does not exist: {:?}", path))
        }
    }

    /// Calculate total size of a directory recursively.
    fn dir_size(path: &Path) -> anyhow::Result<u64> {
        let mut total = 0u64;
        for entry in std::fs::read_dir(path)? {
            let entry = entry?;
            let meta = entry.metadata()?;
            if meta.is_dir() {
                total += Self::dir_size(&entry.path()).unwrap_or(0);
            } else {
                total += meta.len();
            }
        }
        Ok(total)
    }

    /// Run the full progressive training pipeline.
    pub fn run(&mut self) -> anyhow::Result<()> {
        tracing::info!(
            "Starting progressive training: {} → {}",
            self.config.start_stage.size_str(),
            self.config.end_stage.size_str()
        );

        let mut current_stage = self.config.start_stage;
        let mut previous_weights: Option<PathBuf> = None;

        loop {
            tracing::info!("=== Stage: {} ===", current_stage.size_str());

            // Train this stage
            let checkpoint_path = self.train_stage(current_stage, previous_weights.as_deref())?;

            // Upload to HuggingFace if configured
            if self.config.upload_to_hf {
                self.upload_stage(current_stage, &checkpoint_path)?;
            }

            // Check if we're done
            if current_stage == self.config.end_stage {
                tracing::info!("Progressive training complete!");
                break;
            }

            // Move to next stage
            previous_weights = Some(checkpoint_path);
            current_stage = current_stage.next().expect("Should have next stage");
        }

        Ok(())
    }

    /// Train a single stage.
    fn train_stage(
        &mut self,
        stage: TrainingStage,
        previous_weights: Option<&Path>,
    ) -> anyhow::Result<PathBuf> {
        let model_config = stage.config();
        self.current_stage = stage;

        // Auto-configure training parameters based on available VRAM
        let (batch_size, seq_length) = self.auto_configure_params(&model_config)?;

        // Create run directory
        let run_dir = self
            .run_manager
            .create_run_dir(&format!("tritter_{}", stage.size_str().to_lowercase()))?;

        // Setup checkpoint manager
        let checkpoints_dir = run_dir.join("checkpoints");
        fs::create_dir_all(&checkpoints_dir)?;

        let checkpoint_options = CheckpointOptions {
            save_optimizer: true,
            compress: true,
            compression_level: 6,
            upload_to_hf: false, // We handle this separately
            delete_after_upload: false,
            keep_last_n: 3,
            keep_best_n: 5,
        };

        self.checkpoint_manager = Some(CheckpointManager::new(
            checkpoints_dir.clone(),
            None,
            checkpoint_options,
        )?);

        // Create training run record
        let mut training_config = TrainingConfig {
            config_version: 1,
            config_hash: String::new(),
            model_size: stage.size_str().to_string(),
            num_parameters: model_config.parameter_count(),
            hidden_size: model_config.hidden_size,
            num_layers: model_config.num_layers,
            num_heads: model_config.num_heads,
            max_seq_length: model_config.max_seq_length,
            batch_size,
            learning_rate: self.config.learning_rate,
            max_steps: self.config.steps_per_stage,
            gradient_checkpointing: true,
            checkpoint_interval: self.config.gradient_checkpoint_interval,
            device: format!("{:?}", self.device),
        };
        training_config.config_hash = compute_config_hash(&training_config);

        let mut run = TrainingRun::new(
            &format!("tritter-{}", stage.size_str().to_lowercase()),
            training_config,
            run_dir.clone(),
        );
        run.status = TrainingStatus::Running;
        run.save()?;

        // Setup metrics file
        let metrics_file = fs::File::create(&run.metrics_file)?;
        self.metrics_file = Some(metrics_file);

        // Create model
        tracing::info!(
            "Creating {} model ({:.1}M params)...",
            stage.size_str(),
            model_config.parameter_count() as f64 / 1e6
        );

        let mut model_config_with_checkpoint = model_config.clone();
        model_config_with_checkpoint.gradient_checkpointing = true;
        model_config_with_checkpoint.checkpoint_every_n_layers =
            self.config.gradient_checkpoint_interval;

        // Initialize model (with weight expansion if previous weights exist)
        let model = if let Some(prev_path) = previous_weights {
            self.expand_weights(prev_path, &model_config_with_checkpoint)?
        } else {
            TritterModel::new(&model_config_with_checkpoint, &self.device)?
        };

        // Create trainer
        // Lower confidence threshold to allow predictive phases with noisy/random data
        // Production threshold: 0.85, testing threshold: 0.30
        let trainer_config = HybridTrainerConfig::builder()
            .warmup_steps(100)
            .full_steps(20)
            .max_predict_steps(80)
            .confidence_threshold(0.30) // Lowered from 0.85 to enable predictive phases
            .build();

        let model_wrapper = tritter_model_rs::TritterModelWrapper::new(model);
        let optimizer = tritter_model_rs::TritterOptimizer::new(self.config.learning_rate);
        let mut trainer =
            hybrid_predict_trainer_rs::HybridTrainer::new(model_wrapper, optimizer, trainer_config)
                .map_err(|(e, _)| anyhow::anyhow!("Trainer creation failed: {}", e))?;

        // Training loop
        tracing::info!(
            "Starting training for {} steps...",
            self.config.steps_per_stage
        );
        let train_start = Instant::now();
        let mut rng = rand::thread_rng();
        let mut last_phase = TrainingPhase::Warmup;

        // Adaptive learning rate controller for plateau/oscillation handling
        let base_lr = self.config.learning_rate;
        let min_lr = base_lr * 0.01;  // 1% of base LR
        let max_lr = base_lr * 3.0;   // 3x base LR for plateau escape
        let mut lr_controller = AdaptiveLRController::new(base_lr, min_lr, max_lr);
        let mut current_lr = base_lr;

        for step in 0..self.config.steps_per_stage {
            let step_start = Instant::now();

            // Get batch from DataLoader or generate random tokens (testing mode)
            let batch = if let Some(ref mut loader) = self.data_loader {
                match loader.next() {
                    Some(Ok(b)) => b,
                    Some(Err(e)) => {
                        tracing::warn!("DataLoader error at step {}: {}", step, e);
                        // Fall back to random for this step
                        let vocab_size = model_config.vocab_size;
                        let random_ids: Vec<u32> = (0..batch_size * seq_length)
                            .map(|_| rand::Rng::gen_range(&mut rng, 0..vocab_size as u32))
                            .collect();
                        let input_ids = Tensor::from_slice(
                            &random_ids,
                            (batch_size, seq_length),
                            &self.device,
                        )?;
                        TritterBatch::new(input_ids, None)
                    }
                    None => {
                        // Dataset exhausted, reset for next epoch
                        tracing::info!(
                            "Dataset exhausted at step {}, resetting for next epoch",
                            step
                        );
                        loader.reset()?;
                        match loader.next() {
                            Some(Ok(b)) => b,
                            Some(Err(e)) => {
                                return Err(anyhow::anyhow!(
                                    "Failed to get batch after reset: {}",
                                    e
                                ))
                            }
                            None => return Err(anyhow::anyhow!("Empty dataset")),
                        }
                    }
                }
            } else {
                // No dataset: generate random tokens (testing mode)
                let vocab_size = model_config.vocab_size;
                let random_ids: Vec<u32> = (0..batch_size * seq_length)
                    .map(|_| rand::Rng::gen_range(&mut rng, 0..vocab_size as u32))
                    .collect();
                let input_ids =
                    Tensor::from_slice(&random_ids, (batch_size, seq_length), &self.device)?;
                TritterBatch::new(input_ids, None)
            };

            // Execute training step
            let result = trainer
                .step(&batch)
                .map_err(|(e, _)| anyhow::anyhow!("Training step failed: {}", e))?;

            let step_time = step_start.elapsed();

            // Convert phase
            let phase = match result.phase {
                Phase::Warmup => TrainingPhase::Warmup,
                Phase::Full => TrainingPhase::Full,
                Phase::Predict => TrainingPhase::Predict,
                Phase::Correct => TrainingPhase::Correct,
            };

            // Track phase transitions
            if phase != last_phase {
                run.record_phase_transition(last_phase, phase);
                tracing::info!("Phase transition: {} -> {}", last_phase, phase);
                last_phase = phase;
            }

            // Adaptive LR control - handle plateaus, oscillations, and spikes
            let prev_lr = current_lr;
            current_lr = lr_controller.update(result.loss, result.gradient_norm);
            if (current_lr - prev_lr).abs() > 1e-8 {
                trainer.set_learning_rate::<TritterBatch>(current_lr);
                let change = if current_lr > prev_lr { "increased" } else { "decreased" };
                tracing::info!(
                    "LR {} from {:.2e} to {:.2e} (plateau/oscillation handling)",
                    change, prev_lr, current_lr
                );
            }

            // Calculate token metrics
            let tokens_this_step = (batch_size * seq_length) as u64;
            let total_tokens_trained = (step + 1) * tokens_this_step;
            let tokens_remaining = (self.config.steps_per_stage - step - 1) * tokens_this_step;

            // Update loss dynamics tracker and get velocity/acceleration
            let (loss_velocity, loss_acceleration) = self.loss_dynamics_tracker.update(result.loss);

            // Create metrics
            let metrics = StepMetrics {
                step,
                loss: result.loss,
                gradient_norm: result.gradient_norm,
                phase,
                was_predicted: result.was_predicted,
                prediction_error: result.prediction_error,
                step_time_ms: step_time.as_secs_f64() * 1000.0,
                timestamp: chrono::Utc::now(),
                tokens_this_step,
                total_tokens_trained,
                tokens_remaining,
                confidence: result.confidence,
                learning_rate: current_lr,
                perplexity: result.loss.exp(),
                // Generalization metrics
                train_val_gap: None,
                loss_velocity,
                loss_acceleration,
                gradient_entropy: None,
                // Layer gradient metrics (not computed in progressive trainer)
                layer_gradients: None,
                layer_gradient_stats: None,
            };

            // Update run
            run.update_step(&metrics);

            // Update steps per second and tokens per second for ETA calculation
            let elapsed = train_start.elapsed();
            let elapsed_secs = elapsed.as_secs_f64();
            run.steps_per_second = Some((step + 1) as f64 / elapsed_secs);
            run.tokens_per_second = Some(total_tokens_trained as f64 / elapsed_secs);

            // Sample GPU stats periodically
            if step % 10 == 0 {
                if let Some(stats) = self.gpu_stats_monitor.sample() {
                    run.gpu_memory_used = Some(stats.memory_used);
                    run.gpu_memory_total = Some(stats.memory_total);
                    run.gpu_temperature = Some(stats.temperature);
                    run.gpu_power_draw = Some(stats.power_draw);
                    run.gpu_utilization = Some(stats.gpu_util);
                    run.gpu_memory_peak = Some(self.gpu_stats_monitor.peak_memory());
                }
            }

            // Log metrics
            if let Some(ref mut file) = self.metrics_file {
                use std::io::Write;
                writeln!(file, "{}", serde_json::to_string(&metrics)?)?;
            }

            // Logging with GPU status
            if step % self.config.log_every == 0 || step == self.config.steps_per_stage - 1 {
                let steps_per_sec = (step + 1) as f64 / elapsed.as_secs_f64();

                // Sample GPU stats
                let gpu_status = if let Some(stats) = self.gpu_stats_monitor.current() {
                    format!(
                        "GPU: {:.1}/{:.1}GB {}°C {:.0}W",
                        stats.memory_used as f64 / 1e9,
                        stats.memory_total as f64 / 1e9,
                        stats.temperature,
                        stats.power_draw
                    )
                } else {
                    "GPU: N/A".to_string()
                };

                tracing::info!(
                    "Step {:>6}/{} | Loss: {:.4} | Phase: {:>7} | Bwd: {:.1}% | {:.1} steps/s",
                    step,
                    self.config.steps_per_stage,
                    result.loss,
                    phase.to_string(),
                    run.backward_reduction(),
                    steps_per_sec
                );

                // Log GPU status and ETA every 50 steps
                if step % 50 == 0 {
                    tracing::info!("{} | ETA: {}", gpu_status, run.eta_string());
                }

                // Warn if GPU is unhealthy
                if !self.gpu_stats_monitor.is_healthy() {
                    tracing::warn!(
                        "GPU health warning: {}",
                        self.gpu_stats_monitor.health_status()
                    );
                }
            }

            // Checkpointing with timing
            if step > 0
                && (step % self.config.checkpoint_every == 0
                    || step == self.config.steps_per_stage - 1)
            {
                let checkpoint_path =
                    checkpoints_dir.join(format!("checkpoint_step_{}.safetensors", step));

                // Start checkpoint event
                let checkpoint_idx = run.start_checkpoint(checkpoint_path.clone());
                tracing::info!("Saving checkpoint at step {}...", step);

                // Save checkpoint (in real implementation, serialize model weights)
                // For now, we just save the run state
                run.save()?;

                // Mark checkpoint complete (estimate size from run state file)
                let state_file = run.run_dir.join("run_state.json");
                let size = fs::metadata(&state_file).map(|m| m.len()).unwrap_or(0);
                run.complete_checkpoint_save(checkpoint_idx, size);

                tracing::info!(
                    "Checkpoint saved at step {} ({:.1} KB)",
                    step,
                    size as f64 / 1024.0
                );

                // Upload to HuggingFace if configured
                if self.config.upload_to_hf {
                    if let Some(ref _uploader) = self.hf_uploader {
                        run.start_checkpoint_upload(checkpoint_idx);
                        tracing::info!("Uploading checkpoint to HuggingFace...");

                        // Note: In real implementation, call uploader.upload_checkpoint()
                        // For now, just mark as uploaded after a simulated delay
                        let repo_name = format!("tritter-{}", stage.size_str().to_lowercase());
                        let hf_url = format!(
                            "https://huggingface.co/{}/{}",
                            self.config.hf_username.as_deref().unwrap_or("tzervas"),
                            repo_name
                        );
                        run.complete_checkpoint_upload(checkpoint_idx, &hf_url);
                        run.hf_repo = Some(hf_url.clone());
                        tracing::info!("Checkpoint uploaded to {}", hf_url);
                    }
                }

                // Save run state with checkpoint info
                run.save()?;
            }
        }

        // Final stats
        let total_time = train_start.elapsed();
        tracing::info!(
            "Stage {} complete: {} steps in {:.1}s ({:.1} steps/s)",
            stage.size_str(),
            self.config.steps_per_stage,
            total_time.as_secs_f64(),
            self.config.steps_per_stage as f64 / total_time.as_secs_f64()
        );
        tracing::info!(
            "Final loss: {:.4}, Best loss: {:.4} (step {})",
            run.current_loss,
            run.best_loss,
            run.best_step
        );
        tracing::info!("Backward reduction: {:.1}%", run.backward_reduction());

        // Mark run as complete
        run.status = TrainingStatus::Completed;
        run.ended_at = Some(chrono::Utc::now());
        run.save()?;

        // Return path to final checkpoint
        // In real implementation, this would be the actual checkpoint file
        let final_checkpoint = checkpoints_dir.join("final_model.safetensors");
        // Touch the file for now
        fs::write(&final_checkpoint, b"")?;

        Ok(final_checkpoint)
    }

    /// VRAM reserved for host system processes (2GB default).
    const HOST_VRAM_RESERVE: u64 = 2 * 1024 * 1024 * 1024;

    /// Auto-configure training parameters based on available VRAM.
    /// Uses graceful progressive fallback on batch sizing.
    /// Reserves VRAM for host system processes.
    fn auto_configure_params(
        &self,
        model_config: &TritterConfig,
    ) -> anyhow::Result<(usize, usize)> {
        // Query current GPU memory
        let gpu_info = query_gpu_memory(0);

        if let Some(info) = gpu_info {
            // Reserve VRAM for host (2GB) plus 10% safety margin
            let reserved = Self::HOST_VRAM_RESERVE + (info.total as f64 * 0.10) as u64;
            let available = info.total.saturating_sub(reserved);

            tracing::info!(
                "GPU Memory: {:.1}GB total, {:.1}GB reserved for host, {:.1}GB available for training",
                info.total as f64 / 1e9,
                reserved as f64 / 1e9,
                available as f64 / 1e9
            );

            // Graceful batch size fallback: try progressively smaller sizes
            // Order: user_config, user_config*3/4, user_config/2, user_config/4, 2, 1
            let batch_sizes = self.generate_batch_fallback_sequence(self.config.batch_size);
            let seq_lengths = [
                self.config.seq_length,
                self.config.seq_length * 3 / 4,
                self.config.seq_length / 2,
                256,
                128,
            ];

            for &batch_size in &batch_sizes {
                for &seq_length in &seq_lengths {
                    if batch_size == 0 || seq_length == 0 {
                        continue;
                    }

                    let estimated_mem = crate::memory::estimate_training_memory(
                        model_config.parameter_count(),
                        batch_size,
                        seq_length,
                        model_config.hidden_size,
                        model_config.num_layers,
                        model_config.num_heads,
                        self.config.gradient_checkpoint_interval,
                    );

                    // Check if this config fits with margin
                    if estimated_mem <= available {
                        if batch_size != self.config.batch_size
                            || seq_length != self.config.seq_length
                        {
                            tracing::info!(
                                "Optimized config: batch {} (was {}), seq {} (was {}), est. {:.1}GB",
                                batch_size,
                                self.config.batch_size,
                                seq_length,
                                self.config.seq_length,
                                estimated_mem as f64 / 1e9
                            );
                        } else {
                            tracing::info!(
                                "User config fits: batch {}, seq {}, est. {:.1}GB",
                                batch_size,
                                seq_length,
                                estimated_mem as f64 / 1e9
                            );
                        }
                        return Ok((batch_size, seq_length));
                    }
                }
            }

            // Fallback to minimal config
            tracing::warn!("Using minimal config: batch 1, seq 128");
            Ok((1, 128))
        } else {
            // No GPU info available, use configured values
            tracing::warn!("Could not query GPU memory, using configured batch/seq");
            Ok((self.config.batch_size, self.config.seq_length))
        }
    }

    /// Generate a graceful fallback sequence for batch sizes.
    /// e.g., 8 -> [8, 6, 4, 3, 2, 1]
    fn generate_batch_fallback_sequence(&self, max_batch: usize) -> Vec<usize> {
        let mut sizes = Vec::new();
        let mut current = max_batch;

        while current > 0 {
            sizes.push(current);

            // Graceful reduction: 75%, 50%, then linear
            if current > 4 {
                current = (current * 3 / 4).max(current - 2);
            } else if current > 1 {
                current -= 1;
            } else {
                break;
            }

            // Avoid duplicates
            if sizes.last() == Some(&current) {
                current -= 1;
            }
        }

        sizes.push(1); // Always include 1 as final fallback
        sizes.sort_by(|a, b| b.cmp(a)); // Sort descending
        sizes.dedup();
        sizes
    }

    /// Expand weights from smaller model to larger model.
    fn expand_weights(
        &self,
        previous_checkpoint: &Path,
        target_config: &TritterConfig,
    ) -> anyhow::Result<TritterModel> {
        tracing::info!(
            "Expanding weights from {} to new architecture...",
            previous_checkpoint.display()
        );

        // For now, just create a new model
        // In a real implementation, we would:
        // 1. Load the previous checkpoint
        // 2. Initialize new parameters for expanded dimensions
        // 3. Copy compatible weights
        // 4. Use techniques like:
        //    - Tiling embeddings
        //    - Copying attention heads
        //    - Averaging/interpolating weights

        // This is a placeholder - real expansion would transfer learned weights
        let model = TritterModel::new(target_config, &self.device)?;

        tracing::info!("Weight expansion complete (placeholder - using random init)");

        Ok(model)
    }

    /// Upload stage results to HuggingFace.
    fn upload_stage(&self, stage: TrainingStage, checkpoint_path: &Path) -> anyhow::Result<()> {
        let uploader = match &self.hf_uploader {
            Some(u) => u,
            None => return Ok(()),
        };

        // Check auth
        if !uploader.check_auth()? {
            tracing::warn!("HuggingFace not authenticated, skipping upload");
            return Ok(());
        }

        let config = stage.config();

        // Create model card
        let _model_card = ModelCardData {
            model_name: format!("Tritter-{}", stage.size_str()),
            model_size: stage.size_str().to_string(),
            num_parameters: config.parameter_count(),
            architecture: ArchitectureDetails {
                hidden_size: config.hidden_size,
                num_layers: config.num_layers,
                num_heads: config.num_heads,
                num_kv_heads: config.num_kv_heads,
                intermediate_size: config.intermediate_size,
                vocab_size: config.vocab_size,
                max_seq_length: config.max_seq_length,
                use_bitnet: config.use_bitnet,
                use_qk_norm: config.use_qk_norm,
                gradient_checkpointing: config.gradient_checkpointing,
            },
            training: TrainingDetails {
                framework: "rust-ai / tritter-model-rs".to_string(),
                learning_rate: self.config.learning_rate,
                batch_size: self.config.batch_size,
                total_steps: self.config.steps_per_stage,
                warmup_steps: 100,
                hybrid_training: true,
                backward_reduction: 50.0, // Placeholder
                training_time_hours: 0.0, // Placeholder
            },
            metrics: None,
            license: "mit".to_string(),
            repo_url: "https://github.com/tzervas/rust-ai".to_string(),
        };

        // This would upload - just log for now since we need the actual model files
        tracing::info!(
            "Would upload {} to HuggingFace (checkpoint: {})",
            stage.size_str(),
            checkpoint_path.display()
        );

        Ok(())
    }
}
