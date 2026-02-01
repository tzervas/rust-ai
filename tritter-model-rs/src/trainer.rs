//! Integration with hybrid-predict-trainer-rs for predictive training.
//!
//! This module provides implementations of the `Model`, `Optimizer`, and `Batch` traits
//! from hybrid-predict-trainer-rs to enable phase-based predictive training.

use std::collections::HashMap;

use candle_core::{DType, Device, Tensor};

use hybrid_predict_trainer_rs::{
    Batch, GradientInfo, HybridResult, HybridTrainer, HybridTrainerConfig, HybridTrainingError,
    Model, Optimizer,
    state::WeightDelta,
};

use crate::config::TritterConfig;
use crate::error::TritterResult;
use crate::model::TritterModel;

/// Helper to create training error
fn training_error(msg: impl Into<String>) -> (HybridTrainingError, Option<hybrid_predict_trainer_rs::RecoveryAction>) {
    (
        HybridTrainingError::IntegrationError {
            crate_name: "tritter-model-rs".to_string(),
            detail: msg.into(),
        },
        None,
    )
}

/// A batch of training data for Tritter.
#[derive(Debug, Clone)]
pub struct TritterBatch {
    /// Input token IDs: (batch_size, seq_len)
    pub input_ids: Tensor,
    /// Attention mask (optional): (batch_size, seq_len)
    pub attention_mask: Option<Tensor>,
}

impl TritterBatch {
    /// Create a new training batch
    pub fn new(input_ids: Tensor, attention_mask: Option<Tensor>) -> Self {
        Self {
            input_ids,
            attention_mask,
        }
    }

    /// Create batch from u32 slice
    pub fn from_ids(ids: &[u32], batch_size: usize, seq_len: usize, device: &Device) -> TritterResult<Self> {
        let input_ids = Tensor::from_slice(ids, (batch_size, seq_len), device)?;
        Ok(Self::new(input_ids, None))
    }
}

impl Batch for TritterBatch {
    fn batch_size(&self) -> usize {
        self.input_ids.dims()[0]
    }
}

/// Wrapper around TritterModel that implements the hybrid trainer Model trait.
///
/// This wrapper integrates gradient checkpointing for memory-efficient training.
/// When checkpointing is enabled, activations are cached at checkpoint boundaries
/// during the forward pass and cleared after backward.
pub struct TritterModelWrapper {
    model: TritterModel,
    /// Last computed loss (for backward pass)
    last_loss: Option<Tensor>,
    /// Gradient tensors (for optimizer)
    gradients: Option<HashMap<String, Tensor>>,
}

impl TritterModelWrapper {
    /// Create a new model wrapper
    pub fn new(model: TritterModel) -> Self {
        Self {
            model,
            last_loss: None,
            gradients: None,
        }
    }

    /// Get reference to inner model
    pub fn inner(&self) -> &TritterModel {
        &self.model
    }

    /// Get mutable reference to inner model
    pub fn inner_mut(&mut self) -> &mut TritterModel {
        &mut self.model
    }

    /// Check if gradient checkpointing is enabled
    pub fn is_checkpointing_enabled(&self) -> bool {
        self.model.is_checkpointing_enabled()
    }

    /// Enable or disable gradient checkpointing
    pub fn set_checkpointing(&mut self, enabled: bool, interval: Option<usize>) {
        self.model.set_checkpointing(enabled, interval);
    }

    /// Get checkpoint memory usage in bytes
    pub fn checkpoint_memory_usage(&self) -> usize {
        self.model.checkpoint_memory_usage()
    }

    /// Get number of stored checkpoints
    pub fn num_checkpoints(&self) -> usize {
        self.model.num_checkpoints()
    }
}

impl Model<TritterBatch> for TritterModelWrapper {
    fn forward(&mut self, batch: &TritterBatch) -> HybridResult<f32> {
        // Run forward pass with loss computation
        let logits = self.model.forward(&batch.input_ids)
            .map_err(|e| training_error(format!("Forward failed: {}", e)))?;

        // Compute cross-entropy loss
        let loss = self.model.compute_loss(&logits, &batch.input_ids)
            .map_err(|e| training_error(format!("Loss computation failed: {}", e)))?;

        // Store for backward
        self.last_loss = Some(loss.clone());

        // Return scalar loss
        loss.to_scalar::<f32>()
            .map_err(|e| training_error(format!("Loss to scalar failed: {}", e)))
    }

    fn backward(&mut self) -> HybridResult<GradientInfo> {
        let loss = self.last_loss.as_ref()
            .ok_or_else(|| training_error("No loss computed - call forward() first"))?;

        // Compute gradients via backward pass
        // Note: Candle's autograd handles the backward computation through the
        // computation graph. When checkpointing is enabled, we've stored activations
        // at checkpoint boundaries during forward. The backward pass uses these
        // cached tensors and recomputes intermediates as needed.
        let grads = loss.backward()
            .map_err(|e| training_error(format!("Backward failed: {}", e)))?;

        // Clear checkpoint store after backward to free memory
        // This is important for memory efficiency - we don't need the cached
        // activations anymore after gradients are computed
        self.model.clear_checkpoints();

        // Collect gradient info from VarMap
        let var_map = self.model.var_map();
        let data = var_map.data().lock().unwrap();

        let mut gradient_norm_sq = 0.0f32;
        let mut per_param_norms_vec = Vec::new();
        let mut grad_map = HashMap::new();

        for (name, var) in data.iter() {
            if let Some(grad) = grads.get(var) {
                // Store gradient
                grad_map.insert(name.clone(), grad.clone());

                // Compute norm
                let norm_sq: f32 = grad.sqr()
                    .and_then(|t| t.sum_all())
                    .and_then(|t| t.to_scalar())
                    .unwrap_or(0.0);
                gradient_norm_sq += norm_sq;
                per_param_norms_vec.push(norm_sq.sqrt());
            }
        }

        drop(data); // Release lock
        self.gradients = Some(grad_map);

        Ok(GradientInfo {
            loss: loss.to_scalar::<f32>().unwrap_or(f32::NAN),
            gradient_norm: gradient_norm_sq.sqrt(),
            per_param_norms: Some(per_param_norms_vec),
        })
    }

    fn parameter_count(&self) -> usize {
        self.model.parameter_count()
    }

    fn apply_weight_delta(&mut self, delta: &WeightDelta) -> HybridResult<()> {
        let var_map = self.model.var_map();
        let data = var_map.data().lock().unwrap();

        for (name, var) in data.iter() {
            if let Some(delta_vals) = delta.deltas.get(name) {
                // Create delta tensor
                let shape = var.dims();
                let delta_tensor = Tensor::from_slice(delta_vals.as_slice(), shape, var.device())
                    .map_err(|e| training_error(format!("Delta tensor creation failed: {}", e)))?;

                // Scale and apply
                let scaled_delta = (&delta_tensor * delta.scale as f64)
                    .map_err(|e| training_error(format!("Delta scaling failed: {}", e)))?;

                let new_val = (var.as_tensor() + &scaled_delta)
                    .map_err(|e| training_error(format!("Weight update failed: {}", e)))?;

                var.set(&new_val)
                    .map_err(|e| training_error(format!("Weight set failed: {}", e)))?;
            }
        }

        Ok(())
    }
}

/// AdamW optimizer for Tritter training.
pub struct TritterOptimizer {
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    /// First moment estimates
    m: HashMap<String, Tensor>,
    /// Second moment estimates
    v: HashMap<String, Tensor>,
    /// Step counter for bias correction
    t: usize,
}

impl TritterOptimizer {
    /// Create a new AdamW optimizer
    pub fn new(lr: f32) -> Self {
        Self {
            learning_rate: lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
            m: HashMap::new(),
            v: HashMap::new(),
            t: 0,
        }
    }

    /// Create optimizer with custom hyperparameters
    pub fn with_params(lr: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32) -> Self {
        Self {
            learning_rate: lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            m: HashMap::new(),
            v: HashMap::new(),
            t: 0,
        }
    }
}

impl Optimizer<TritterModelWrapper, TritterBatch> for TritterOptimizer {
    fn step(&mut self, model: &mut TritterModelWrapper, _gradients: &GradientInfo) -> HybridResult<()> {
        self.t += 1;

        // Bias correction factors
        let bc1 = 1.0 - self.beta1.powi(self.t as i32);
        let bc2 = 1.0 - self.beta2.powi(self.t as i32);

        // Get gradients from model
        let grads = model.gradients.as_ref()
            .ok_or_else(|| training_error("No gradients available"))?
            .clone();

        let var_map = model.model.var_map();
        let data = var_map.data().lock().unwrap();

        for (name, var) in data.iter() {
            if let Some(grad) = grads.get(name) {
                let device = var.device();
                let shape = var.dims();

                // Initialize moments if needed
                if !self.m.contains_key(name) {
                    self.m.insert(name.clone(), Tensor::zeros(shape, DType::F32, device)
                        .map_err(|e| training_error(e.to_string()))?);
                    self.v.insert(name.clone(), Tensor::zeros(shape, DType::F32, device)
                        .map_err(|e| training_error(e.to_string()))?);
                }

                let m = self.m.get(name).unwrap();
                let v = self.v.get(name).unwrap();

                // Update biased first moment: m = β1 * m + (1 - β1) * g
                let m_new = ((m * self.beta1 as f64).map_err(|e| training_error(e.to_string()))?
                    + (grad * (1.0 - self.beta1) as f64).map_err(|e| training_error(e.to_string()))?)
                    .map_err(|e| training_error(e.to_string()))?;

                // Update biased second moment: v = β2 * v + (1 - β2) * g²
                let grad_sq = grad.sqr().map_err(|e| training_error(e.to_string()))?;
                let v_new = ((v * self.beta2 as f64).map_err(|e| training_error(e.to_string()))?
                    + (&grad_sq * (1.0 - self.beta2) as f64).map_err(|e| training_error(e.to_string()))?)
                    .map_err(|e| training_error(e.to_string()))?;

                // Bias-corrected estimates
                let m_hat = (&m_new / bc1 as f64).map_err(|e| training_error(e.to_string()))?;
                let v_hat = (&v_new / bc2 as f64).map_err(|e| training_error(e.to_string()))?;

                // Compute update: lr * m_hat / (sqrt(v_hat) + eps)
                let v_sqrt = v_hat.sqrt().map_err(|e| training_error(e.to_string()))?;
                let denom = (&v_sqrt + self.eps as f64).map_err(|e| training_error(e.to_string()))?;
                let update = (&m_hat / &denom).map_err(|e| training_error(e.to_string()))?;
                let update = (&update * self.learning_rate as f64).map_err(|e| training_error(e.to_string()))?;

                // AdamW weight decay: w = w - lr * wd * w
                let var_tensor = var.as_tensor();
                let decay = (var_tensor * (self.learning_rate * self.weight_decay) as f64)
                    .map_err(|e: candle_core::Error| training_error(e.to_string()))?;

                // Apply update: w = w - update - decay
                let new_w = ((var_tensor - &update).map_err(|e: candle_core::Error| training_error(e.to_string()))? - &decay)
                    .map_err(|e: candle_core::Error| training_error(e.to_string()))?;

                var.set(&new_w).map_err(|e: candle_core::Error| training_error(e.to_string()))?;

                // Store updated moments
                self.m.insert(name.clone(), m_new);
                self.v.insert(name.clone(), v_new);
            }
        }

        Ok(())
    }

    fn learning_rate(&self) -> f32 {
        self.learning_rate
    }

    fn set_learning_rate(&mut self, lr: f32) {
        self.learning_rate = lr;
    }

    fn zero_grad(&mut self) {
        // Gradients are recomputed on each backward pass, nothing to clear
    }
}

/// Type alias for the complete Tritter trainer
pub type TritterTrainer = HybridTrainer<TritterModelWrapper, TritterOptimizer>;

/// Create a new Tritter trainer with default configuration
pub fn create_trainer(
    config: &TritterConfig,
    learning_rate: f32,
    device: &Device,
) -> TritterResult<TritterTrainer> {
    let model = TritterModel::new(config, device)?;
    let model_wrapper = TritterModelWrapper::new(model);
    let optimizer = TritterOptimizer::new(learning_rate);

    let trainer_config = HybridTrainerConfig::default();
    let trainer = HybridTrainer::new(model_wrapper, optimizer, trainer_config)
        .map_err(|(e, _)| crate::error::TritterError::Training(e.to_string()))?;

    Ok(trainer)
}

/// Create a trainer with custom hybrid training configuration
pub fn create_trainer_with_config(
    model_config: &TritterConfig,
    trainer_config: HybridTrainerConfig,
    learning_rate: f32,
    device: &Device,
) -> TritterResult<TritterTrainer> {
    let model = TritterModel::new(model_config, device)?;
    let model_wrapper = TritterModelWrapper::new(model);
    let optimizer = TritterOptimizer::new(learning_rate);

    let trainer = HybridTrainer::new(model_wrapper, optimizer, trainer_config)
        .map_err(|(e, _)| crate::error::TritterError::Training(e.to_string()))?;

    Ok(trainer)
}

/// Create a trainer with gradient checkpointing explicitly enabled.
///
/// This is a convenience function that ensures gradient checkpointing is active,
/// which reduces memory usage at the cost of additional compute.
///
/// # Arguments
/// * `model_config` - Model configuration (gradient_checkpointing will be enabled)
/// * `trainer_config` - Hybrid trainer configuration
/// * `learning_rate` - Learning rate for AdamW optimizer
/// * `checkpoint_interval` - Cache activations every N layers (4 is typical)
/// * `device` - Device for training (CPU or CUDA)
///
/// # Memory Savings
///
/// With `checkpoint_interval=4` on a 24-layer model:
/// - Memory reduction: ~75% of activation memory
/// - Compute overhead: ~33% additional forward passes
///
/// # Example
///
/// ```no_run
/// use tritter_model_rs::{TritterConfig, trainer::create_trainer_with_checkpointing};
/// use hybrid_predict_trainer_rs::HybridTrainerConfig;
/// use candle_core::Device;
///
/// let model_config = TritterConfig::medium_500m();
/// let trainer_config = HybridTrainerConfig::default();
/// let device = Device::Cpu;
///
/// let trainer = create_trainer_with_checkpointing(
///     &model_config,
///     trainer_config,
///     1e-4,
///     4, // checkpoint every 4 layers
///     &device,
/// ).unwrap();
/// ```
pub fn create_trainer_with_checkpointing(
    model_config: &TritterConfig,
    trainer_config: HybridTrainerConfig,
    learning_rate: f32,
    checkpoint_interval: usize,
    device: &Device,
) -> TritterResult<TritterTrainer> {
    // Clone config and enable checkpointing
    let mut config = model_config.clone();
    config.gradient_checkpointing = true;
    config.checkpoint_every_n_layers = checkpoint_interval.max(1);

    let model = TritterModel::new(&config, device)?;
    let model_wrapper = TritterModelWrapper::new(model);
    let optimizer = TritterOptimizer::new(learning_rate);

    let trainer = HybridTrainer::new(model_wrapper, optimizer, trainer_config)
        .map_err(|(e, _)| crate::error::TritterError::Training(e.to_string()))?;

    Ok(trainer)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_creation() {
        let device = Device::Cpu;
        let ids: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let batch = TritterBatch::from_ids(&ids, 2, 4, &device).unwrap();

        assert_eq!(batch.batch_size(), 2);
    }

    #[test]
    fn test_optimizer_creation() {
        let optimizer = TritterOptimizer::new(1e-4);
        assert_eq!(optimizer.learning_rate(), 1e-4);
    }

    #[test]
    fn test_model_wrapper_checkpointing() {
        let mut config = TritterConfig::test();
        config.gradient_checkpointing = true;
        config.checkpoint_every_n_layers = 1;

        let device = Device::Cpu;
        let model = TritterModel::new(&config, &device).unwrap();
        let wrapper = TritterModelWrapper::new(model);

        assert!(wrapper.is_checkpointing_enabled());
    }

    #[test]
    fn test_model_wrapper_toggle_checkpointing() {
        let config = TritterConfig::test();
        let device = Device::Cpu;
        let model = TritterModel::new(&config, &device).unwrap();
        let mut wrapper = TritterModelWrapper::new(model);

        // Initially disabled
        assert!(!wrapper.is_checkpointing_enabled());

        // Enable
        wrapper.set_checkpointing(true, Some(2));
        assert!(wrapper.is_checkpointing_enabled());

        // Disable
        wrapper.set_checkpointing(false, None);
        assert!(!wrapper.is_checkpointing_enabled());
    }

    #[test]
    fn test_forward_with_checkpointing() {
        let mut config = TritterConfig::test();
        config.gradient_checkpointing = true;
        config.checkpoint_every_n_layers = 1;
        config.num_layers = 4;

        let device = Device::Cpu;
        let model = TritterModel::new(&config, &device).unwrap();
        let mut wrapper = TritterModelWrapper::new(model);

        let ids: Vec<u32> = vec![0; 8];
        let batch = TritterBatch::from_ids(&ids, 2, 4, &device).unwrap();

        // Forward should work and cache checkpoints
        let loss = <TritterModelWrapper as Model<TritterBatch>>::forward(&mut wrapper, &batch);
        assert!(loss.is_ok());

        // Should have stored checkpoints
        assert!(wrapper.num_checkpoints() > 0);
    }

    #[test]
    fn test_backward_clears_checkpoints() {
        let mut config = TritterConfig::test();
        config.gradient_checkpointing = true;
        config.checkpoint_every_n_layers = 1;
        config.num_layers = 4;

        let device = Device::Cpu;
        let model = TritterModel::new(&config, &device).unwrap();
        let mut wrapper = TritterModelWrapper::new(model);

        let ids: Vec<u32> = vec![0; 8];
        let batch = TritterBatch::from_ids(&ids, 2, 4, &device).unwrap();

        // Forward pass
        let _ = <TritterModelWrapper as Model<TritterBatch>>::forward(&mut wrapper, &batch);
        assert!(wrapper.num_checkpoints() > 0);

        // Backward pass should clear checkpoints
        let _ = <TritterModelWrapper as Model<TritterBatch>>::backward(&mut wrapper);
        assert_eq!(wrapper.num_checkpoints(), 0);
    }

    #[test]
    fn test_create_trainer_with_checkpointing() {
        let config = TritterConfig::test();
        let trainer_config = HybridTrainerConfig::default();
        let device = Device::Cpu;

        let trainer = create_trainer_with_checkpointing(
            &config,
            trainer_config,
            1e-4,
            2,
            &device,
        );

        assert!(trainer.is_ok());
    }
}
