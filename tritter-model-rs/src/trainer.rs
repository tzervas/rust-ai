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
        let grads = loss.backward()
            .map_err(|e| training_error(format!("Backward failed: {}", e)))?;

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
}
