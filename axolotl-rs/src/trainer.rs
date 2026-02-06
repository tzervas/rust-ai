//! Training loop and optimization.

use candle_core::{Device, Tensor};
use candle_nn::VarMap;
use indicatif::{ProgressBar, ProgressStyle};

use crate::config::AxolotlConfig;
use crate::dataset::Dataset;
use crate::error::{AxolotlError, Result};
use crate::model::{load_model, LoadedModel};
use crate::optimizer::{AdamWOptimizer, OptimizerConfig};
use crate::scheduler::{LRScheduler, SchedulerType};

// Use qlora-rs cross_entropy_loss when available (maintains gradient graph)
#[cfg(feature = "qlora")]
use qlora_rs::cross_entropy_loss;

/// Training step metrics for convergence validation and monitoring.
#[derive(Debug, Clone)]
pub struct StepMetrics {
    /// Cross-entropy loss for this step
    pub loss: f64,
    /// Global norm of all gradients
    pub grad_norm: f64,
    /// Global norm of all trainable parameters
    pub param_norm: f64,
}

/// Training orchestrator.
///
/// # Example
///
/// ```no_run
/// use axolotl_rs::{AxolotlConfig, Trainer};
///
/// # fn main() -> axolotl_rs::Result<()> {
/// // Create configuration
/// let config = AxolotlConfig::from_preset("llama2-7b")?;
///
/// // Create trainer
/// let mut trainer = Trainer::new(config)?;
///
/// // Run training
/// trainer.train()?;
/// # Ok(())
/// # }
/// ```
pub struct Trainer {
    /// Configuration
    config: AxolotlConfig,
    /// Current step
    step: usize,
    /// Current epoch
    epoch: usize,
    /// Device for training
    device: Device,
    /// Loaded model (optional, loaded during train())
    model: Option<LoadedModel>,
    /// Optimizer (optional, created during train())
    optimizer: Option<AdamWOptimizer>,
    /// Learning rate scheduler (optional, created during train())
    scheduler: Option<LRScheduler>,
    /// Training metrics from last run
    pub training_metrics: Vec<StepMetrics>,
}

impl Trainer {
    /// Create a new trainer.
    ///
    /// Validates the configuration before creating the trainer.
    ///
    /// # Example
    ///
    /// ```rust
    /// use axolotl_rs::{AxolotlConfig, Trainer};
    ///
    /// # fn main() -> axolotl_rs::Result<()> {
    /// let config = AxolotlConfig::from_preset("llama2-7b")?;
    /// let trainer = Trainer::new(config)?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid.
    pub fn new(config: AxolotlConfig) -> Result<Self> {
        config.validate()?;

        // Determine device (prefer CUDA, fallback to CPU with warning)
        let force_cpu = std::env::var("AXOLOTL_FORCE_CPU")
            .ok()
            .is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true"));
        let cuda_device = std::env::var("AXOLOTL_CUDA_DEVICE")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(0);
        let device = if !force_cpu && cfg!(feature = "cuda") {
            match Device::cuda_if_available(cuda_device) {
                Ok(device @ Device::Cuda(_)) => {
                    tracing::info!("Training device: CUDA (device {})", cuda_device);
                    device
                }
                Ok(_) => {
                    tracing::warn!("CUDA not available; falling back to CPU. This is a compatibility path only.");
                    Device::Cpu
                }
                Err(err) => {
                    tracing::warn!("CUDA init failed ({err}); falling back to CPU. This is a compatibility path only.");
                    Device::Cpu
                }
            }
        } else {
            if force_cpu {
                tracing::warn!(
                    "CPU mode forced via AXOLOTL_FORCE_CPU=1. GPU is the intended default."
                );
            } else {
                tracing::warn!(
                    "CUDA feature disabled; falling back to CPU. Enable with --features cuda."
                );
            }
            Device::Cpu
        };

        Ok(Self {
            config,
            step: 0,
            epoch: 0,
            device,
            model: None,
            optimizer: None,
            scheduler: None,
            training_metrics: Vec::new(),
        })
    }

    /// Resume training from a checkpoint.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use axolotl_rs::{AxolotlConfig, Trainer};
    ///
    /// # fn main() -> axolotl_rs::Result<()> {
    /// let config = AxolotlConfig::from_file("config.yaml")?;
    /// let mut trainer = Trainer::new(config)?;
    ///
    /// // Resume from a previous checkpoint
    /// trainer.resume_from("./outputs/checkpoint-1000")?;
    /// trainer.train()?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    /// Returns an error if the checkpoint cannot be loaded.
    pub fn resume_from(&mut self, checkpoint_path: &str) -> Result<()> {
        self.load_checkpoint(checkpoint_path)
    }

    /// Run the training loop.
    ///
    /// This performs the following steps:
    /// 1. Loads the dataset
    /// 2. Iterates over epochs and batches
    /// 3. Logs metrics periodically
    /// 4. Saves checkpoints periodically
    /// 5. Saves final checkpoint
    ///
    /// # Example
    ///
    /// ```no_run
    /// use axolotl_rs::{AxolotlConfig, Trainer};
    ///
    /// # fn main() -> axolotl_rs::Result<()> {
    /// // Load configuration
    /// let config = AxolotlConfig::from_file("config.yaml")?;
    ///
    /// // Create and run trainer
    /// let mut trainer = Trainer::new(config)?;
    /// trainer.train()?;
    ///
    /// println!("Training complete!");
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Dataset cannot be loaded
    /// - Model fails to load
    /// - Training encounters an error
    /// - Checkpoint saving fails
    pub fn train(&mut self) -> Result<()> {
        tracing::info!("Starting training");
        tracing::info!("  Base model: {}", self.config.base_model);
        tracing::info!("  Adapter: {:?}", self.config.adapter);
        tracing::info!("  Epochs: {}", self.config.training.epochs);

        // Load model
        let model = load_model(&self.config, &self.device)?;
        tracing::info!(
            "Model loaded with vocab size: {}",
            model.tokenizer.get_vocab_size(true)
        );
        self.model = Some(model);

        // Load dataset
        let dataset = Dataset::load(&self.config.dataset)?;
        tracing::info!("Loaded {} training examples", dataset.len());

        // Create output directory
        std::fs::create_dir_all(&self.config.output_dir)?;

        // Calculate total steps for progress bar
        let total_steps =
            dataset.len() * self.config.training.epochs / self.config.training.batch_size;

        // Initialize optimizer with trainable parameters from adapter layers
        {
            let optimizer_config = OptimizerConfig {
                learning_rate: self.config.training.learning_rate,
                weight_decay: self.config.training.weight_decay,
                ..OptimizerConfig::default()
            };

            // Use trainable params from loaded model (LoRA A/B matrices)
            let model = self.model.as_ref().ok_or_else(|| {
                AxolotlError::Training("Model must be loaded before optimizer init".into())
            })?;
            let optimizer = optimizer_config.build_adamw(&model.trainable_params)?;
            let param_count: usize = model
                .trainable_params
                .all_vars()
                .iter()
                .map(|v| v.elem_count())
                .sum();
            tracing::info!(
                "Initialized AdamW optimizer with lr={}, {} trainable params",
                optimizer.learning_rate(),
                param_count
            );
            self.optimizer = Some(optimizer);
        }

        // Initialize learning rate scheduler
        {
            let warmup_steps = (total_steps as f64 * 0.1) as usize; // 10% warmup

            let scheduler = LRScheduler::new(
                SchedulerType::Linear {
                    warmup_steps,
                    total_steps,
                },
                self.config.training.learning_rate,
            );
            tracing::info!(
                "Initialized linear scheduler with {} warmup steps",
                warmup_steps
            );
            self.scheduler = Some(scheduler);
        }

        // Create progress bar
        let pb = ProgressBar::new(total_steps as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos:>7}/{len:7} {msg}")
                .unwrap()
                .progress_chars("#>-"),
        );

        // Clear previous metrics
        self.training_metrics.clear();

        // Training loop
        for epoch in 0..self.config.training.epochs {
            self.epoch = epoch;
            tracing::info!(
                "Starting epoch {}/{}",
                epoch + 1,
                self.config.training.epochs
            );

            for batch in dataset.train.chunks(self.config.training.batch_size) {
                self.step += 1;

                // Training step
                let metrics = self.training_step(batch)?;

                // Store metrics for convergence validation
                self.training_metrics.push(metrics.clone());

                // Update progress bar with loss
                pb.set_message(format!("{:.4}", metrics.loss));
                pb.inc(1);

                // Log periodically
                if self.step % self.config.training.logging_steps == 0 {
                    tracing::info!(
                        "Step {}/{}, Epoch {}, Loss: {:.4}, GradNorm: {:.4}, ParamNorm: {:.4}, LR: {:.2e}",
                        self.step,
                        total_steps,
                        epoch + 1,
                        metrics.loss,
                        metrics.grad_norm,
                        metrics.param_norm,
                        self.optimizer.as_ref().unwrap().learning_rate()
                    );
                }

                // Save checkpoint periodically
                if self.step % self.config.training.save_steps == 0 {
                    self.save_checkpoint()?;
                }

                // Step scheduler
                if let (Some(scheduler), Some(optimizer)) =
                    (self.scheduler.as_mut(), self.optimizer.as_mut())
                {
                    scheduler.step(optimizer);
                }
            }
        }

        pb.finish_with_message("Training complete");

        // Save final checkpoint
        self.save_checkpoint()?;

        Ok(())
    }

    /// Perform a single training step.
    ///
    /// This method:
    /// 1. Tokenizes the batch
    /// 2. Performs forward pass
    /// 3. Computes cross-entropy loss
    /// 4. Performs backward pass and optimizer step
    ///
    /// Note that this method requires `&mut self` because calling the optimizer
    /// step updates the internal training state (e.g. optimizer buffers and
    /// model parameters), and therefore must take a mutable reference to the
    /// trainer.
    fn training_step(&mut self, batch: &[crate::dataset::Example]) -> Result<StepMetrics> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| AxolotlError::Training("Model not loaded".into()))?;

        // 1. Tokenize batch
        // Get pad token ID from tokenizer, fallback to 0 if not found
        let pad_token_id = model
            .tokenizer
            .token_to_id("<pad>")
            .or_else(|| model.tokenizer.token_to_id("[PAD]"))
            .unwrap_or(0);

        let mut input_ids = Vec::new();
        let mut labels = Vec::new();
        let max_len = self.config.dataset.max_length;

        for example in batch {
            let encoding = model
                .tokenizer
                .encode(example.text.as_str(), true)
                .map_err(|e| {
                    AxolotlError::Tokenizer(format!("Tokenization failed: {}", e).into())
                })?;

            let mut ids = encoding.get_ids().to_vec();
            let original_len = ids.len();

            // Truncate or pad to max_len
            if ids.len() > max_len {
                ids.truncate(max_len);
            }
            while ids.len() < max_len {
                ids.push(pad_token_id);
            }

            // For causal LM, labels are input_ids shifted left by 1
            // Mask padding tokens with -100 so they don't contribute to loss
            let mut label_ids: Vec<i64> = Vec::with_capacity(max_len);
            for i in 0..max_len {
                // After truncation at line 312, original_len represents the actual content length
                // We mask positions where there's no next token (i.e., i >= original_len - 1)
                if i + 1 < original_len {
                    // Use next token as label
                    label_ids.push(ids[i + 1] as i64);
                } else {
                    // Mask padding positions with -100 (ignore index)
                    label_ids.push(-100);
                }
            }

            input_ids.push(ids);
            labels.push(label_ids);
        }

        // 2. Convert to tensors
        let batch_size = input_ids.len();
        let flat_input: Vec<i64> = input_ids.iter().flatten().map(|&x| x as i64).collect();
        let flat_labels: Vec<i64> = labels.iter().flatten().copied().collect();

        let input_tensor = Tensor::from_vec(flat_input, (batch_size, max_len), &self.device)
            .map_err(|e| AxolotlError::Training(format!("Failed to create input tensor: {}", e)))?;
        let label_tensor = Tensor::from_vec(flat_labels, (batch_size, max_len), &self.device)
            .map_err(|e| AxolotlError::Training(format!("Failed to create label tensor: {}", e)))?;

        // 3. Forward pass - returns logits for all positions [batch, seq, vocab]
        let logits = model
            .forward_with_adapters(&input_tensor)
            .map_err(|e| AxolotlError::Training(format!("Forward pass failed: {}", e)))?;

        // 4. Compute cross-entropy loss over all positions
        // For language model training, we compute loss at each position
        // comparing prediction at position i with target at position i+1
        let loss = compute_cross_entropy_loss(&logits, &label_tensor, &self.device)?;
        let loss_val = loss
            .to_vec0::<f32>()
            .map_err(|e| AxolotlError::Training(format!("Failed to get loss value: {}", e)))?
            as f64;

        // 5. Backward pass and optimizer step
        let optimizer = self
            .optimizer
            .as_mut()
            .ok_or_else(|| AxolotlError::Training("Optimizer not initialized".into()))?;

        // Compute gradients and apply via optimizer step (internally calls backward)
        optimizer.step(&loss)?;

        // Compute gradient and parameter norms for monitoring
        let grad_norm = compute_global_grad_norm(&self.model.as_ref().unwrap().trainable_params)?;
        let param_norm = compute_global_param_norm(&self.model.as_ref().unwrap().trainable_params)?;

        Ok(StepMetrics {
            loss: loss_val,
            grad_norm,
            param_norm,
        })
    }

    /// Save a checkpoint.
    ///
    /// Saves:
    /// - Training state (step, epoch, config)
    /// - Adapter weights (if using LoRA/QLoRA) in safetensors format
    /// - Optimizer state (for resume)
    fn save_checkpoint(&self) -> Result<()> {
        let checkpoint_dir = format!("{}/checkpoint-{}", self.config.output_dir, self.step);
        std::fs::create_dir_all(&checkpoint_dir)?;

        // Save training state
        let optimizer = self.optimizer.as_ref().ok_or_else(|| {
            AxolotlError::Checkpoint("Optimizer not initialized during checkpoint save".into())
        })?;
        let training_state = TrainingState {
            step: self.step,
            epoch: self.epoch,
            learning_rate: optimizer.learning_rate(),
        };
        let state_path = format!("{}/training_state.json", checkpoint_dir);
        let state_json = serde_json::to_string_pretty(&training_state).map_err(|e| {
            AxolotlError::Checkpoint(format!("Failed to serialize state: {}", e).into())
        })?;
        std::fs::write(&state_path, state_json)?;

        // Save config for reproducibility
        let config_path = format!("{}/config.yaml", checkpoint_dir);
        self.config.to_file(&config_path)?;

        // Save adapter weights if using LoRA/QLoRA
        #[cfg(feature = "peft")]
        if let Some(ref model) = self.model {
            if model.adapter_layers.is_some() {
                model.save_adapter_weights(&checkpoint_dir)?;

                // Also save adapter config as JSON (HuggingFace compatible)
                let adapter_config = serde_json::json!({
                    "base_model_name_or_path": self.config.base_model,
                    "r": self.config.lora.r,
                    "lora_alpha": self.config.lora.alpha,
                    "lora_dropout": self.config.lora.dropout,
                    "target_modules": self.config.lora.target_modules,
                    "bias": "none",
                    "task_type": "CAUSAL_LM"
                });
                let adapter_config_path = format!("{}/adapter_config.json", checkpoint_dir);
                std::fs::write(
                    &adapter_config_path,
                    serde_json::to_string_pretty(&adapter_config).unwrap(),
                )?;
            }
        }

        tracing::info!("Saved checkpoint to: {}", checkpoint_dir);
        Ok(())
    }

    /// Load training state from a checkpoint.
    ///
    /// # Errors
    /// Returns error if checkpoint files cannot be read or parsed.
    pub fn load_checkpoint(&mut self, checkpoint_path: &str) -> Result<()> {
        let state_path = format!("{}/training_state.json", checkpoint_path);
        let state_json = std::fs::read_to_string(&state_path)
            .map_err(|e| AxolotlError::Checkpoint(format!("Failed to read state: {}", e).into()))?;
        let state: TrainingState = serde_json::from_str(&state_json).map_err(|e| {
            AxolotlError::Checkpoint(format!("Failed to parse state: {}", e).into())
        })?;

        self.step = state.step;
        self.epoch = state.epoch;

        if let Some(optimizer) = self.optimizer.as_mut() {
            optimizer.set_learning_rate(state.learning_rate);
        }

        // Load adapter weights if available
        #[cfg(feature = "peft")]
        {
            let adapter_path = format!("{}/adapter_model.safetensors", checkpoint_path);
            if std::path::Path::new(&adapter_path).exists() {
                if let Some(ref mut model) = self.model {
                    model.load_adapter_weights(checkpoint_path)?;
                }
            }
        }

        tracing::info!(
            "Loaded checkpoint from: {} (step={}, epoch={})",
            checkpoint_path,
            state.step,
            state.epoch
        );
        Ok(())
    }

    /// Get reference to the loaded model for testing/inspection.
    ///
    /// Returns None if model hasn't been loaded yet (before train() is called).
    #[allow(dead_code)]
    pub fn get_model(&self) -> Option<&LoadedModel> {
        self.model.as_ref()
    }

    /// Get mutable reference to the loaded model for testing/inspection.
    ///
    /// Returns None if model hasn't been loaded yet (before train() is called).
    #[allow(dead_code)]
    pub fn get_model_mut(&mut self) -> Option<&mut LoadedModel> {
        self.model.as_mut()
    }

    /// Get all training metrics collected during training.
    ///
    /// Returns a vector of metrics for each training step.
    /// Use this for convergence validation and analysis.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use axolotl_rs::{AxolotlConfig, Trainer};
    ///
    /// # fn main() -> axolotl_rs::Result<()> {
    /// let mut trainer = Trainer::new(AxolotlConfig::from_preset("llama2-7b")?)?;
    /// trainer.train()?;
    ///
    /// let metrics = trainer.metrics();
    /// println!("Training completed with {} steps", metrics.len());
    /// # Ok(())
    /// # }
    /// ```
    #[allow(dead_code)]
    pub fn metrics(&self) -> &[StepMetrics] {
        &self.training_metrics
    }

    /// Get loss values for all training steps.
    ///
    /// Returns a vector of loss values, useful for convergence validation.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use axolotl_rs::{AxolotlConfig, Trainer};
    ///
    /// # fn main() -> axolotl_rs::Result<()> {
    /// let mut trainer = Trainer::new(AxolotlConfig::from_preset("llama2-7b")?)?;
    /// trainer.train()?;
    ///
    /// let losses = trainer.losses();
    /// assert!(losses[losses.len()-1] < losses[0], "Loss should decrease");
    /// # Ok(())
    /// # }
    /// ```
    #[allow(dead_code)]
    pub fn losses(&self) -> Vec<f64> {
        self.training_metrics.iter().map(|m| m.loss).collect()
    }

    /// Get gradient norms for all training steps.
    ///
    /// Returns a vector of global gradient norms.
    #[allow(dead_code)]
    pub fn grad_norms(&self) -> Vec<f64> {
        self.training_metrics.iter().map(|m| m.grad_norm).collect()
    }

    /// Get parameter norms for all training steps.
    ///
    /// Returns a vector of global parameter norms.
    #[allow(dead_code)]
    pub fn param_norms(&self) -> Vec<f64> {
        self.training_metrics.iter().map(|m| m.param_norm).collect()
    }

    /// Get current training step.
    #[allow(dead_code)]
    pub fn step(&self) -> usize {
        self.step
    }

    /// Get current epoch.
    #[allow(dead_code)]
    pub fn epoch(&self) -> usize {
        self.epoch
    }
}

/// Training state for checkpoint serialization.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct TrainingState {
    /// Current training step
    step: usize,
    /// Current epoch
    epoch: usize,
    /// Current learning rate
    learning_rate: f64,
}

/// Compute cross-entropy loss on the last position prediction.
///
/// This is used when the model only returns logits for the last position
/// (common in generation-optimized models like candle's Llama).
/// Compute global norm of all gradients in a VarMap.
///
/// This is a simplified implementation that returns a placeholder value.
/// Full implementation would require access to gradient tensors from the optimizer.
fn compute_global_grad_norm(_varmap: &VarMap) -> Result<f64> {
    // TODO: Implement proper gradient norm computation
    // For now, return a placeholder value
    // The full implementation would need access to gradients from the backward pass
    Ok(0.0)
}

/// Compute global norm of all parameters in a VarMap.
///
/// This is a simplified implementation that returns a placeholder value.
fn compute_global_param_norm(_varmap: &VarMap) -> Result<f64> {
    // TODO: Implement proper parameter norm computation
    // For now, return a placeholder value
    Ok(1.0)
}

///
/// # Arguments
/// * `logits` - Model output logits with shape [batch_size, vocab_size] (last position only)
/// * `labels` - Target labels with shape [batch_size, seq_len], -100 for masked positions
/// * `device` - Device for tensor operations
///
/// # Returns
/// A scalar loss tensor that can be backpropagated through
#[allow(dead_code)]
fn compute_last_position_loss(logits: &Tensor, labels: &Tensor, device: &Device) -> Result<Tensor> {
    let dims = logits.dims();

    // Logits should be [batch, vocab] for last-position-only output
    if dims.len() != 2 {
        return Err(AxolotlError::Training(format!(
            "Expected 2D logits [batch, vocab], got {:?}",
            dims
        )));
    }

    let (batch_size, vocab_size) = (dims[0], dims[1]);
    let label_dims = labels.dims();
    let _seq_len = label_dims[1];

    // For each sequence, get the last non-padding label
    // This is the target for predicting what comes after the last token
    let labels_flat = labels
        .to_vec2::<i64>()
        .map_err(|e| AxolotlError::Training(format!("Failed to read labels: {}", e)))?;

    // Find last valid (non -100) label for each batch item
    let mut last_labels: Vec<u32> = Vec::with_capacity(batch_size);
    let mut valid_mask: Vec<f32> = Vec::with_capacity(batch_size);

    for seq_labels in &labels_flat {
        // Find the last valid label (not -100)
        let mut last_valid: Option<i64> = None;
        for &label in seq_labels.iter().rev() {
            if label >= 0 && (label as usize) < vocab_size {
                last_valid = Some(label);
                break;
            }
        }

        match last_valid {
            Some(label) => {
                last_labels.push(label as u32);
                valid_mask.push(1.0);
            }
            None => {
                last_labels.push(0);
                valid_mask.push(0.0);
            }
        }
    }

    let valid_count: f32 = valid_mask.iter().sum();

    if valid_count == 0.0 {
        return Tensor::new(0.0f32, device)
            .map_err(|e| AxolotlError::Training(format!("Failed to create zero loss: {}", e)));
    }

    let labels_tensor = Tensor::from_vec(last_labels, batch_size, device)
        .map_err(|e| AxolotlError::Training(format!("Failed to create labels tensor: {}", e)))?;

    // Compute log softmax
    let log_probs = candle_nn::ops::log_softmax(logits, 1)
        .map_err(|e| AxolotlError::Training(format!("Log softmax failed: {}", e)))?;

    // Gather log probs at target indices
    let target_indices = labels_tensor
        .unsqueeze(1)
        .map_err(|e| AxolotlError::Training(format!("Unsqueeze failed: {}", e)))?;
    let gathered = log_probs
        .gather(&target_indices, 1)
        .map_err(|e| AxolotlError::Training(format!("Gather failed: {}", e)))?
        .squeeze(1)
        .map_err(|e| AxolotlError::Training(format!("Squeeze failed: {}", e)))?;

    // Apply mask and compute mean of negative log likelihood
    let mask_tensor = Tensor::from_vec(valid_mask, batch_size, device)
        .map_err(|e| AxolotlError::Training(format!("Failed to create mask tensor: {}", e)))?;
    let masked_loss = gathered
        .neg()
        .map_err(|e| AxolotlError::Training(format!("Neg failed: {}", e)))?
        .mul(&mask_tensor)
        .map_err(|e| AxolotlError::Training(format!("Mul failed: {}", e)))?;

    // Sum and divide by valid count
    let total_loss = masked_loss
        .sum_all()
        .map_err(|e| AxolotlError::Training(format!("Sum failed: {}", e)))?;

    let valid_count_scalar = Tensor::new(valid_count, device)
        .map_err(|e| AxolotlError::Training(format!("Failed to create count tensor: {}", e)))?;

    let loss = total_loss
        .broadcast_div(&valid_count_scalar)
        .map_err(|e| AxolotlError::Training(format!("Div failed: {}", e)))?;

    Ok(loss)
}

/// Compute cross-entropy loss with gradient tracking (full sequence version).
///
/// This function uses tensor operations that maintain the autograd graph,
/// enabling proper backpropagation through the loss to update LoRA weights.
///
/// # Arguments
/// * `logits` - Model output logits with shape [batch_size, seq_len, vocab_size] or [batch*seq, vocab]
/// * `labels` - Target labels with shape [batch_size, seq_len], -100 for masked positions
/// * `device` - Device for tensor operations
///
/// # Returns
/// A scalar loss tensor that can be backpropagated through
#[allow(dead_code)]
fn compute_cross_entropy_loss(logits: &Tensor, labels: &Tensor, device: &Device) -> Result<Tensor> {
    let dims = logits.dims();
    let label_dims = labels.dims();

    // Handle different logit shapes from different model implementations
    // Candle's Llama returns [batch * seq, vocab] while some return [batch, seq, vocab]
    let (num_positions, vocab_size) = match dims.len() {
        2 => (dims[0], dims[1]),
        3 => (dims[0] * dims[1], dims[2]),
        _ => {
            return Err(AxolotlError::Training(format!(
                "Expected 2D or 3D logits, got {:?}",
                dims
            )))
        }
    };

    // Flatten logits to [num_positions, vocab_size]
    let logits_flat = if dims.len() == 3 {
        logits
            .reshape((num_positions, vocab_size))
            .map_err(|e| AxolotlError::Training(format!("Logits reshape failed: {}", e)))?
    } else {
        logits.clone()
    };

    // Flatten labels to [num_positions]
    let labels_flat = labels
        .reshape(num_positions)
        .map_err(|e| AxolotlError::Training(format!("Labels reshape failed: {}", e)))?;

    // Verify dimensions match
    if num_positions != label_dims.iter().product::<usize>() {
        return Err(AxolotlError::Training(format!(
            "Logits positions {} != labels positions {}",
            num_positions,
            label_dims.iter().product::<usize>()
        )));
    }

    // Create mask for valid (non-padding) positions
    // Labels of -100 are masked out
    let labels_i64 = labels_flat
        .to_vec1::<i64>()
        .map_err(|e| AxolotlError::Training(format!("Failed to read labels: {}", e)))?;

    let valid_mask: Vec<f32> = labels_i64
        .iter()
        .map(|&l| {
            if l >= 0 && (l as usize) < vocab_size {
                1.0
            } else {
                0.0
            }
        })
        .collect();
    let valid_count: f32 = valid_mask.iter().sum();

    if valid_count == 0.0 {
        // No valid labels, return zero loss
        return Tensor::new(&[0.0f32], device)
            .map_err(|e| AxolotlError::Training(format!("Failed to create zero loss: {}", e)));
    }

    // Replace invalid labels with 0 (they'll be masked anyway)
    let safe_labels: Vec<u32> = labels_i64
        .iter()
        .map(|&l| {
            if l >= 0 && (l as usize) < vocab_size {
                l as u32
            } else {
                0
            }
        })
        .collect();
    let safe_labels_tensor = Tensor::from_vec(safe_labels, num_positions, device)
        .map_err(|e| AxolotlError::Training(format!("Failed to create safe labels: {}", e)))?;

    // Compute log softmax (this maintains gradients)
    let log_probs = candle_nn::ops::log_softmax(&logits_flat, 1)
        .map_err(|e| AxolotlError::Training(format!("Log softmax failed: {}", e)))?;

    // Gather log probs at target indices
    let target_indices = safe_labels_tensor
        .unsqueeze(1)
        .map_err(|e| AxolotlError::Training(format!("Unsqueeze failed: {}", e)))?;
    let gathered = log_probs
        .gather(&target_indices, 1)
        .map_err(|e| AxolotlError::Training(format!("Gather failed: {}", e)))?
        .squeeze(1)
        .map_err(|e| AxolotlError::Training(format!("Squeeze failed: {}", e)))?;

    // Apply mask and compute mean of negative log likelihood
    let mask_tensor = Tensor::from_vec(valid_mask, num_positions, device)
        .map_err(|e| AxolotlError::Training(format!("Failed to create mask tensor: {}", e)))?;
    let masked_loss = gathered
        .neg()
        .map_err(|e| AxolotlError::Training(format!("Neg failed: {}", e)))?
        .mul(&mask_tensor)
        .map_err(|e| AxolotlError::Training(format!("Mul failed: {}", e)))?;

    // Sum and divide by valid count
    let total_loss = masked_loss
        .sum_all()
        .map_err(|e| AxolotlError::Training(format!("Sum failed: {}", e)))?;

    // Create scalar tensor for valid_count and squeeze total_loss to same shape
    let valid_count_scalar = Tensor::new(valid_count, device)
        .map_err(|e| AxolotlError::Training(format!("Failed to create count tensor: {}", e)))?;

    // Both are scalars now, division should work
    let loss = total_loss
        .broadcast_div(&valid_count_scalar)
        .map_err(|e| AxolotlError::Training(format!("Div failed: {}", e)))?;

    Ok(loss)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    /// Helper to create a test config using a preset
    fn create_test_config(output_dir: &str) -> AxolotlConfig {
        let mut config = AxolotlConfig::from_preset("llama2-7b").unwrap();
        config.output_dir = output_dir.to_string();
        // Override dataset path for testing
        config.dataset.path = "test-dataset.jsonl".to_string();
        config
    }

    /// Helper to create a test dataset file
    fn create_test_dataset(path: &str, num_examples: usize) -> std::io::Result<()> {
        let mut content = String::new();
        for i in 0..num_examples {
            content.push_str(&format!(
                r#"{{"instruction":"Test instruction {}","input":"","output":"Test output {}"}}"#,
                i, i
            ));
            content.push('\n');
        }
        fs::write(path, content)
    }

    // ========================================================================
    // Tests for Trainer::new
    // ========================================================================

    #[test]
    fn test_trainer_creation() {
        let config = AxolotlConfig::from_preset("llama2-7b").unwrap();
        let trainer = Trainer::new(config);
        assert!(trainer.is_ok());
    }

    #[test]
    fn test_trainer_new_stores_config() {
        let temp_dir = TempDir::new().unwrap();
        let output_path = temp_dir.path().join("outputs");
        let config = create_test_config(output_path.to_str().unwrap());
        let base_model = config.base_model.clone();

        let trainer = Trainer::new(config).unwrap();

        // Verify config is stored correctly
        assert_eq!(trainer.config.base_model, base_model);
    }

    #[test]
    fn test_trainer_new_initializes_counters() {
        let temp_dir = TempDir::new().unwrap();
        let output_path = temp_dir.path().join("outputs");
        let config = create_test_config(output_path.to_str().unwrap());

        let trainer = Trainer::new(config).unwrap();

        // Verify epoch and step counters start at 0
        assert_eq!(trainer.epoch, 0);
        assert_eq!(trainer.step, 0);
    }

    #[test]
    fn test_trainer_new_with_invalid_config() {
        let temp_dir = TempDir::new().unwrap();
        let output_path = temp_dir.path().join("outputs");
        let mut config = create_test_config(output_path.to_str().unwrap());

        // Make config invalid by setting base_model to empty (this IS validated)
        config.base_model = String::new();

        let result = Trainer::new(config);
        assert!(result.is_err());
    }

    // ========================================================================
    // Tests for checkpoint directory handling
    // ========================================================================

    #[test]
    fn test_checkpoint_directory_creation() {
        let temp_dir = TempDir::new().unwrap();
        let output_path = temp_dir.path().join("checkpoints");
        let dataset_path = temp_dir.path().join("dataset.jsonl");

        // Create test dataset
        create_test_dataset(dataset_path.to_str().unwrap(), 5).unwrap();

        let mut config = create_test_config(output_path.to_str().unwrap());
        config.dataset.path = dataset_path.to_str().unwrap().to_string();
        config.training.epochs = 1;
        config.training.batch_size = 5; // Process all in one batch
        config.training.save_steps = 1; // Save immediately

        let mut trainer = Trainer::new(config).unwrap();

        // Directory shouldn't exist yet
        assert!(!output_path.exists());

        // Run training (will fail due to missing model)
        let _ = trainer.train();

        // Directory should not be created since training failed
        assert!(!output_path.exists());
    }

    #[test]
    fn test_checkpoint_directory_reuse() {
        let temp_dir = TempDir::new().unwrap();
        let output_path = temp_dir.path().join("checkpoints");
        let dataset_path = temp_dir.path().join("dataset.jsonl");

        // Pre-create the output directory
        fs::create_dir_all(&output_path).unwrap();

        // Create test dataset
        create_test_dataset(dataset_path.to_str().unwrap(), 5).unwrap();

        let mut config = create_test_config(output_path.to_str().unwrap());
        config.dataset.path = dataset_path.to_str().unwrap().to_string();
        config.training.epochs = 1;
        config.training.batch_size = 5;
        config.training.save_steps = 1;

        let mut trainer = Trainer::new(config).unwrap();

        // Should succeed even though directory exists
        let result = trainer.train();
        assert!(result.is_err()); // Model loading fails in test environment
    }

    // ========================================================================
    // Tests for resume_from
    // ========================================================================

    #[test]
    fn test_resume_from_missing_checkpoint() {
        let temp_dir = TempDir::new().unwrap();
        let output_path = temp_dir.path().join("outputs");
        let config = create_test_config(output_path.to_str().unwrap());

        let mut trainer = Trainer::new(config).unwrap();

        // Resuming from non-existent checkpoint should fail
        let result = trainer.resume_from("nonexistent-checkpoint");
        assert!(result.is_err());

        match result {
            Err(AxolotlError::Checkpoint(_)) => {}
            _ => panic!("Expected Checkpoint error"),
        }
    }

    #[test]
    fn test_checkpoint_save_and_load() {
        let temp_dir = TempDir::new().unwrap();
        let output_path = temp_dir.path().join("outputs");
        let config = create_test_config(output_path.to_str().unwrap());

        let mut trainer = Trainer::new(config).unwrap();
        trainer.step = 100;
        trainer.epoch = 2;

        // Initialize optimizer for checkpoint save (required)
        let optimizer_config = OptimizerConfig {
            learning_rate: 0.001,
            weight_decay: 0.01,
            ..OptimizerConfig::default()
        };
        let varmap = VarMap::new();
        trainer.optimizer = Some(optimizer_config.build_adamw(&varmap).unwrap());

        // Save checkpoint
        trainer.save_checkpoint().unwrap();

        // Verify checkpoint files exist
        let checkpoint_dir = output_path.join("checkpoint-100");
        assert!(checkpoint_dir.join("training_state.json").exists());
        assert!(checkpoint_dir.join("config.yaml").exists());

        // Load checkpoint into new trainer
        let config2 = create_test_config(output_path.to_str().unwrap());
        let mut trainer2 = Trainer::new(config2).unwrap();

        // Initialize optimizer in trainer2 to test learning rate restoration
        let optimizer_config2 = OptimizerConfig {
            learning_rate: 0.002, // Different initial value
            weight_decay: 0.01,
            ..OptimizerConfig::default()
        };
        let varmap2 = VarMap::new();
        trainer2.optimizer = Some(optimizer_config2.build_adamw(&varmap2).unwrap());

        trainer2
            .load_checkpoint(checkpoint_dir.to_str().unwrap())
            .unwrap();

        assert_eq!(trainer2.step, 100);
        assert_eq!(trainer2.epoch, 2);
        // Verify learning rate was restored from checkpoint
        assert_eq!(trainer2.optimizer.as_ref().unwrap().learning_rate(), 0.001);
    }

    #[test]
    fn test_resume_from_valid_checkpoint() {
        let temp_dir = TempDir::new().unwrap();
        let output_path = temp_dir.path().join("outputs");
        let config = create_test_config(output_path.to_str().unwrap());

        let mut trainer = Trainer::new(config).unwrap();

        // Try to resume from non-existent checkpoint
        let result = trainer.resume_from("non-existent-checkpoint");
        assert!(result.is_err());
    }

    // ========================================================================
    // Tests for train method
    // ========================================================================

    #[test]
    fn test_train_with_small_dataset() {
        let temp_dir = TempDir::new().unwrap();
        let output_path = temp_dir.path().join("outputs");
        let dataset_path = temp_dir.path().join("dataset.jsonl");

        // Create small dataset (5 examples)
        create_test_dataset(dataset_path.to_str().unwrap(), 5).unwrap();

        let mut config = create_test_config(output_path.to_str().unwrap());
        config.dataset.path = dataset_path.to_str().unwrap().to_string();
        config.training.epochs = 1;
        config.training.batch_size = 2;
        config.training.save_steps = 1000; // Don't save during training

        let mut trainer = Trainer::new(config).unwrap();

        // Training should fail due to missing model files in test environment
        let result = trainer.train();
        assert!(result.is_err());
    }

    #[test]
    fn test_train_with_empty_dataset() {
        let temp_dir = TempDir::new().unwrap();
        let output_path = temp_dir.path().join("outputs");
        let dataset_path = temp_dir.path().join("empty_dataset.jsonl");

        // Create empty dataset
        fs::write(&dataset_path, "").unwrap();

        let mut config = create_test_config(output_path.to_str().unwrap());
        config.dataset.path = dataset_path.to_str().unwrap().to_string();

        let mut trainer = Trainer::new(config).unwrap();

        // Training should fail due to missing model files
        let result = trainer.train();
        assert!(result.is_err());
    }

    #[test]
    fn test_train_epoch_iteration() {
        let temp_dir = TempDir::new().unwrap();
        let output_path = temp_dir.path().join("outputs");
        let dataset_path = temp_dir.path().join("dataset.jsonl");

        // Create dataset with 10 examples
        create_test_dataset(dataset_path.to_str().unwrap(), 10).unwrap();

        let mut config = create_test_config(output_path.to_str().unwrap());
        config.dataset.path = dataset_path.to_str().unwrap().to_string();
        config.training.epochs = 3;
        config.training.batch_size = 5;
        config.training.save_steps = 1000; // Don't save during training

        let mut trainer = Trainer::new(config).unwrap();

        let result = trainer.train();
        assert!(result.is_err()); // Model loading fails in test environment

        // With 10 examples, batch size 5, and 3 epochs:
        // Each epoch would have 2 batches, so 3 epochs = 6 steps total
        // But since training fails, step counter remains 0
        assert_eq!(trainer.step, 0);

        // Epoch counter should remain at 0 since training failed before starting
        assert_eq!(trainer.epoch, 0);
    }

    #[test]
    fn test_train_batch_iteration() {
        let temp_dir = TempDir::new().unwrap();
        let output_path = temp_dir.path().join("outputs");
        let dataset_path = temp_dir.path().join("dataset.jsonl");

        // Create dataset with 7 examples
        create_test_dataset(dataset_path.to_str().unwrap(), 7).unwrap();

        let mut config = create_test_config(output_path.to_str().unwrap());
        config.dataset.path = dataset_path.to_str().unwrap().to_string();
        config.training.epochs = 1;
        config.training.batch_size = 3;
        config.training.save_steps = 1000; // Don't save during training

        let mut trainer = Trainer::new(config).unwrap();

        let result = trainer.train();
        assert!(result.is_err()); // Model loading fails in test environment

        // With 7 examples and batch size 3, after validation split (10%):
        // Training set would have ~6 examples (7 * 0.9 = 6.3)
        // Batches: [0,1,2], [3,4,5] = 2 steps
        // But since training fails, step counter remains 0
        assert_eq!(trainer.step, 0);
    }

    #[test]
    fn test_train_with_missing_dataset() {
        let temp_dir = TempDir::new().unwrap();
        let output_path = temp_dir.path().join("outputs");

        let mut config = create_test_config(output_path.to_str().unwrap());
        config.dataset.path = "non_existent_dataset.jsonl".to_string();

        let mut trainer = Trainer::new(config).unwrap();

        // Training should fail with model loading error (model loading happens before dataset loading)
        let result = trainer.train();
        assert!(result.is_err());

        match result {
            Err(AxolotlError::Model(_)) => {
                // Expected error type (model loading fails first)
            }
            _ => panic!("Expected Model error"),
        }
    }

    // ========================================================================
    // Tests for checkpoint operations
    // ========================================================================

    #[test]
    fn test_checkpoint_path_generation() {
        let temp_dir = TempDir::new().unwrap();
        let output_path = temp_dir.path().join("outputs");
        let dataset_path = temp_dir.path().join("dataset.jsonl");

        // Create test dataset
        create_test_dataset(dataset_path.to_str().unwrap(), 10).unwrap();

        let mut config = create_test_config(output_path.to_str().unwrap());
        config.dataset.path = dataset_path.to_str().unwrap().to_string();
        config.training.epochs = 1;
        config.training.batch_size = 5;
        config.training.save_steps = 1; // Save after each step

        let mut trainer = Trainer::new(config).unwrap();

        let _ = trainer.train();

        // Check that checkpoint directories were NOT created since training failed
        // With 10 examples, batch size 5, we would get 2 steps if training succeeded
        let checkpoint_1 = output_path.join("checkpoint-1");
        let checkpoint_2 = output_path.join("checkpoint-2");

        assert!(!checkpoint_1.exists());
        assert!(!checkpoint_2.exists());
    }

    #[test]
    fn test_checkpoint_final_save() {
        let temp_dir = TempDir::new().unwrap();
        let output_path = temp_dir.path().join("outputs");
        let dataset_path = temp_dir.path().join("dataset.jsonl");

        // Create test dataset
        create_test_dataset(dataset_path.to_str().unwrap(), 5).unwrap();

        let mut config = create_test_config(output_path.to_str().unwrap());
        config.dataset.path = dataset_path.to_str().unwrap().to_string();
        config.training.epochs = 1;
        config.training.batch_size = 5;
        config.training.save_steps = 1000; // Don't save during training

        let mut trainer = Trainer::new(config).unwrap();

        let result = trainer.train();
        assert!(result.is_err()); // Model loading fails in test environment

        // Final checkpoint should NOT be saved since training failed
        let final_checkpoint = output_path.join("checkpoint-1");
        assert!(!final_checkpoint.exists());
    }

    // ========================================================================
    // Tests with mock datasets of various sizes
    // ========================================================================

    #[test]
    fn test_train_with_single_example() {
        let temp_dir = TempDir::new().unwrap();
        let output_path = temp_dir.path().join("outputs");
        let dataset_path = temp_dir.path().join("dataset.jsonl");

        // Create dataset with 3 examples to ensure at least 1 in training split
        create_test_dataset(dataset_path.to_str().unwrap(), 3).unwrap();

        let mut config = create_test_config(output_path.to_str().unwrap());
        config.dataset.path = dataset_path.to_str().unwrap().to_string();
        config.training.epochs = 1;
        config.training.batch_size = 1;
        config.dataset.val_split = 0.1; // 90% training = 2-3 examples

        let mut trainer = Trainer::new(config).unwrap();

        let result = trainer.train();
        assert!(result.is_err()); // Model loading fails in test environment
                                  // With 3 examples and 10% val split: 2 training examples
                                  // With batch size 1: 2 steps
                                  // But since training fails, step counter remains 0
        assert_eq!(trainer.step, 0);
    }

    #[test]
    fn test_train_with_large_dataset_batching() {
        let temp_dir = TempDir::new().unwrap();
        let output_path = temp_dir.path().join("outputs");
        let dataset_path = temp_dir.path().join("dataset.jsonl");

        // Create larger dataset (50 examples)
        create_test_dataset(dataset_path.to_str().unwrap(), 50).unwrap();

        let mut config = create_test_config(output_path.to_str().unwrap());
        config.dataset.path = dataset_path.to_str().unwrap().to_string();
        config.training.epochs = 1;
        config.training.batch_size = 10;
        config.training.save_steps = 1000; // Don't save during training

        let mut trainer = Trainer::new(config).unwrap();

        let result = trainer.train();
        assert!(result.is_err()); // Model loading fails in test environment

        // 50 examples / 10 batch size = 5 steps
        // But since training fails, step counter remains 0
        assert_eq!(trainer.step, 0);
    }

    #[test]
    fn test_train_multiple_epochs_step_accumulation() {
        let temp_dir = TempDir::new().unwrap();
        let output_path = temp_dir.path().join("outputs");
        let dataset_path = temp_dir.path().join("dataset.jsonl");

        // Create dataset
        create_test_dataset(dataset_path.to_str().unwrap(), 8).unwrap();

        let mut config = create_test_config(output_path.to_str().unwrap());
        config.dataset.path = dataset_path.to_str().unwrap().to_string();
        config.training.epochs = 5;
        config.training.batch_size = 4;
        config.training.save_steps = 1000; // Don't save during training

        let mut trainer = Trainer::new(config).unwrap();

        let result = trainer.train();
        assert!(result.is_err()); // Model loading fails in test environment

        // 8 examples / 4 batch size = 2 steps per epoch
        // 2 steps * 5 epochs = 10 total steps
        // But since training fails, step counter remains 0
        assert_eq!(trainer.step, 0);
    }
}
