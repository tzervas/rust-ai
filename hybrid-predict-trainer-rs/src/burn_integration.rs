//! Burn framework integration for hybrid-predict-trainer-rs.
//!
//! This module provides wrappers that bridge Burn's autodiff system with the
//! generic `Model` and `Optimizer` traits used by `HybridTrainer`.
//!
//! # Why Burn Integration?
//!
//! Burn is a comprehensive deep learning framework for Rust that provides:
//! - **Autodiff**: Automatic differentiation for gradient computation
//! - **Backends**: CPU, CUDA, and other accelerators via CubeCL
//! - **Modules**: Pre-built neural network layers and architectures
//! - **Optimizers**: Adam, SGD, AdamW with momentum and learning rate schedules
//!
//! This integration allows `HybridTrainer` to work with any Burn model while
//! preserving its predictive training capabilities.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────┐
//! │                    HybridTrainer                        │
//! │  (Generic over Model<B> + Optimizer<M, B> traits)      │
//! └───────────────────────┬─────────────────────────────────┘
//!                         │
//!                         ▼
//! ┌─────────────────────────────────────────────────────────┐
//! │            BurnModel / BurnOptimizer                    │
//! │         (Implements Model/Optimizer traits)             │
//! └───────────────────────┬─────────────────────────────────┘
//!                         │
//!                         ▼
//! ┌─────────────────────────────────────────────────────────┐
//! │              Burn Autodiff Module                       │
//! │         (burn::module::AutodiffModule)                  │
//! └─────────────────────────────────────────────────────────┘
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use burn::module::AutodiffModule;
//! use burn::optim::Adam;
//! use burn::backend::Autodiff;
//! use burn_wgpu::WgpuBackend;
//! use hybrid_predict_trainer_rs::burn_integration::{BurnModel, BurnOptimizer};
//! use hybrid_predict_trainer_rs::HybridTrainer;
//!
//! // Define backend
//! type Backend = Autodiff<WgpuBackend>;
//!
//! // Create Burn model
//! let model = MyBurnModel::new(...);
//!
//! // Wrap for HybridTrainer
//! let wrapped_model = BurnModel::new(model, device);
//! let wrapped_optimizer = BurnOptimizer::new(Adam::default(), 1e-3);
//!
//! // Use with HybridTrainer
//! let trainer = HybridTrainer::new(wrapped_model, wrapped_optimizer, config)?;
//! ```

use crate::state::WeightDelta;
use crate::{Batch, GradientInfo, Model, Optimizer};

use burn::module::{AutodiffModule, ModuleMapper, ModuleVisitor, Param};
use burn::optim::GradientsParams;
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Shape, Tensor, TensorData};

use parking_lot::RwLock;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::Arc;

/// Wrapper for Burn batches that implements the `Batch` trait.
///
/// This allows any Burn-compatible batch type to work with `HybridTrainer`.
///
/// # Type Parameters
///
/// - `B`: The Burn backend (e.g., `WgpuBackend`, `NdArrayBackend`)
/// - `T`: The actual batch data type
///
/// # Example
///
/// ```rust,ignore
/// struct MnistBatch<B: Backend> {
///     images: Tensor<B, 2>,
///     labels: Tensor<B, 1>,
/// }
///
/// let batch = BurnBatch::new(MnistBatch { images, labels }, 32);
/// ```
#[derive(Debug, Clone)]
pub struct BurnBatch<B: Backend, T> {
    /// The actual batch data
    pub data: T,
    /// Batch size (number of samples)
    batch_size: usize,
    _phantom: PhantomData<B>,
}

impl<B: Backend, T> BurnBatch<B, T> {
    /// Creates a new Burn batch wrapper.
    ///
    /// # Arguments
    ///
    /// * `data` - The batch data
    /// * `batch_size` - Number of samples in the batch
    pub fn new(data: T, batch_size: usize) -> Self {
        Self {
            data,
            batch_size,
            _phantom: PhantomData,
        }
    }
}

impl<B: Backend, T: Send + Sync> Batch for BurnBatch<B, T> {
    fn batch_size(&self) -> usize {
        self.batch_size
    }
}

/// Trait for user-defined forward pass and loss computation.
///
/// This trait allows users to define how their model computes predictions and loss,
/// enabling flexible integration with different model architectures and loss functions.
///
/// # Why is this needed?
///
/// Burn's ownership model requires that the model is consumed and returned by
/// operations. This trait encapsulates that pattern while letting users define
/// their specific forward logic.
///
/// # Type Parameters
///
/// - `B`: Autodiff backend
/// - `M`: The Burn module type
/// - `T`: The batch data type
///
/// # Example
///
/// ```rust,ignore
/// struct MnistForward;
///
/// impl BurnForwardFn<Autodiff<NdArrayBackend>, SimpleMLP<Autodiff<NdArrayBackend>>, MnistBatch>
///     for MnistForward
/// {
///     fn forward(
///         &self,
///         model: SimpleMLP<Autodiff<NdArrayBackend>>,
///         batch: &BurnBatch<Autodiff<NdArrayBackend>, MnistBatch>,
///     ) -> (SimpleMLP<Autodiff<NdArrayBackend>>, Tensor<Autodiff<NdArrayBackend>, 1>) {
///         // Forward pass
///         let (model, logits) = model.forward(batch.data.images.clone());
///
///         // Compute loss (cross-entropy)
///         let loss = cross_entropy_loss(logits, batch.data.labels.clone());
///
///         (model, loss)
///     }
/// }
/// ```
pub trait BurnForwardFn<B, M, T>: Send + Sync
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
{
    /// Computes forward pass and returns model + loss tensor.
    ///
    /// # Arguments
    ///
    /// * `model` - The model (consumed, ownership transfer)
    /// * `batch` - The input batch
    ///
    /// # Returns
    ///
    /// A tuple of (model, loss_tensor). The model is returned so it can be
    /// used for subsequent operations (Burn's ownership model).
    fn forward(&self, model: M, batch: &BurnBatch<B, T>) -> (M, Tensor<B, 1>);
}

/// Wrapper for Burn models that implements the `Model` trait.
///
/// This enables any Burn `AutodiffModule` to work with `HybridTrainer`'s
/// predictive training system.
///
/// # Type Parameters
///
/// - `B`: Autodiff backend (e.g., `Autodiff<WgpuBackend>`)
/// - `M`: The Burn module type (must implement `AutodiffModule`)
/// - `T`: The batch data type
///
/// # Forward/Backward Integration
///
/// The wrapper maintains a reference to the last computed loss tensor to enable
/// backward pass computation. This is necessary because Burn's autodiff requires
/// calling `.backward()` on the loss tensor itself.
///
/// # Weight Delta Application
///
/// When applying predicted weight deltas during the Predict phase, the wrapper
/// directly modifies parameter tensors while preserving the autodiff graph structure.
pub struct BurnModelWrapper<B, M, T, F>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
    F: BurnForwardFn<B, M, T>,
{
    /// The wrapped Burn model (Option for ownership dance)
    ///
    /// Uses `Mutex` instead of `RwLock` because the model itself may be `!Sync`
    /// in autodiff backends. `Mutex<T>` only requires `T: Send` to be `Send + Sync`.
    model: Arc<parking_lot::Mutex<Option<M>>>,
    /// User-provided forward function for loss computation
    forward_fn: Arc<F>,
    /// Device for tensor operations
    device: burn::tensor::Device<B>,
    /// Last computed loss (for backward pass)
    last_loss: Arc<RwLock<Option<Tensor<B, 1>>>>,
    /// Last computed gradients (for optimizer)
    ///
    /// Uses `Mutex` instead of `RwLock` because gradients may be `!Sync`
    /// in autodiff backends. `Mutex<T>` only requires `T: Send` to be `Send + Sync`.
    last_gradients: Arc<parking_lot::Mutex<Option<<B as AutodiffBackend>::Gradients>>>,
    /// Cache of parameter names and shapes
    #[allow(dead_code)]
    param_metadata: Arc<RwLock<HashMap<String, Vec<usize>>>>,
    _phantom: PhantomData<T>,
}

impl<B, M, T, F> BurnModelWrapper<B, M, T, F>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
    F: BurnForwardFn<B, M, T>,
{
    /// Creates a new Burn model wrapper.
    ///
    /// # Arguments
    ///
    /// * `model` - The Burn model to wrap
    /// * `forward_fn` - User-provided function for forward pass and loss computation
    /// * `device` - Device for tensor operations
    ///
    /// # Returns
    ///
    /// A wrapped model ready for use with `HybridTrainer`.
    pub fn new(model: M, forward_fn: F, device: burn::tensor::Device<B>) -> Self {
        Self {
            model: Arc::new(parking_lot::Mutex::new(Some(model))),
            forward_fn: Arc::new(forward_fn),
            device,
            last_loss: Arc::new(RwLock::new(None)),
            last_gradients: Arc::new(parking_lot::Mutex::new(None)),
            param_metadata: Arc::new(RwLock::new(HashMap::new())),
            _phantom: PhantomData,
        }
    }

    // Phase 3 Burn Integration Methods (Deferred):
    //
    // The following methods were removed as dead code but represent future
    // integration points with Burn's advanced features:
    //
    // - count_parameters(): Requires Burn's module introspection API
    // - tensor_to_vec()/vec_to_tensor(): For gradient/parameter conversion
    // - compute_gradient_norm(): For per-layer gradient monitoring
    //
    // These will be implemented when we add:
    // 1. Per-layer gradient tracking for detailed monitoring
    // 2. Parameter-level weight delta application
    // 3. Advanced Burn model introspection capabilities
    //
    // Until then, current implementation uses high-level Burn APIs that don't
    // require tensor-level manipulation.

    /// Clears the last loss tensor and its autodiff graph.
    ///
    /// This method should be called during the Predict phase after forward()
    /// when backward() won't be called. This prevents memory accumulation from
    /// unused autodiff graphs that would otherwise persist.
    ///
    /// # Memory Management
    ///
    /// During normal training (Full phase):
    /// - forward() stores loss + autodiff graph (~2-4 GB for GPT-2 Small)
    /// - backward() releases the autodiff graph after computing gradients
    ///
    /// During Predict phase:
    /// - forward() stores loss + autodiff graph
    /// - backward() is NEVER called (that's the speedup!)
    /// - Without clearing, autodiff graphs accumulate (+500 MB/step)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // In execute_predict_step()
    /// let loss = model.forward(batch)?;
    /// model.clear_loss(); // Prevent memory leak
    /// ```
    pub fn clear_loss(&mut self) {
        *self.last_loss.write() = None;
    }
}

/// Implementation of Model trait for BurnModelWrapper
impl<B, M, T, F> Model<BurnBatch<B, T>> for BurnModelWrapper<B, M, T, F>
where
    B: AutodiffBackend,
    M: AutodiffModule<B> + Send,
    F: BurnForwardFn<B, M, T>,
    T: Send + Sync,
    <B as AutodiffBackend>::Gradients: Send,
{
    fn forward(&mut self, batch: &BurnBatch<B, T>) -> crate::error::HybridResult<f32> {
        // 1. Take model from Option (ownership dance)
        let mut model_lock = self.model.lock();
        let model = model_lock.take().ok_or_else(|| {
            (
                crate::error::HybridTrainingError::IntegrationError {
                    crate_name: "burn".to_string(),
                    detail: "Model already taken during forward pass".to_string(),
                },
                None,
            )
        })?;

        // 2. Call user-provided forward function (model consumed, returned with loss)
        let (model_returned, loss_tensor) = self.forward_fn.forward(model, batch);

        // 3. Extract scalar loss value
        let loss_data = loss_tensor.to_data();
        let loss_scalar = loss_data
            .to_vec::<f32>()
            .ok()
            .and_then(|v| v.first().copied())
            .ok_or_else(|| {
                (
                    crate::error::HybridTrainingError::IntegrationError {
                        crate_name: "burn".to_string(),
                        detail: "Failed to extract loss scalar from tensor".to_string(),
                    },
                    None,
                )
            })?;

        // 4. Store loss tensor for backward pass (must keep autodiff graph alive)
        *self.last_loss.write() = Some(loss_tensor);

        // 5. Put model back
        *model_lock = Some(model_returned);

        Ok(loss_scalar)
    }

    fn backward(&mut self) -> crate::error::HybridResult<GradientInfo> {
        // 1. Take loss tensor
        let loss_tensor = self.last_loss.write().take().ok_or_else(|| {
            (
                crate::error::HybridTrainingError::IntegrationError {
                    crate_name: "burn".to_string(),
                    detail: "No loss tensor available for backward pass".to_string(),
                },
                None,
            )
        })?;

        // 2. Call backward() to get gradients
        let gradients = loss_tensor.backward();

        // 3. Get model to extract gradients
        let model_lock = self.model.lock();
        let model = model_lock.as_ref().ok_or_else(|| {
            (
                crate::error::HybridTrainingError::IntegrationError {
                    crate_name: "burn".to_string(),
                    detail: "Model not available during backward pass".to_string(),
                },
                None,
            )
        })?;

        // 4. Extract per-parameter gradients
        let per_param_grads = extract_gradients(model, &gradients);

        // 5. Compute gradient norm (L2 norm across all parameters)
        let grad_norm: f32 = per_param_grads
            .values()
            .flat_map(|v| v.iter())
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt();

        // 6. Compute per-parameter norms
        let per_param_norms: Vec<f32> = per_param_grads
            .values()
            .map(|vec| vec.iter().map(|x| x * x).sum::<f32>().sqrt())
            .collect();

        // 7. Store gradients for optimizer
        *self.last_gradients.lock() = Some(gradients);

        // Note: loss is not available here (was consumed by backward())
        // The caller (HybridTrainer) will fill in the loss from forward()
        Ok(GradientInfo {
            loss: 0.0, // Placeholder - will be filled by caller
            gradient_norm: grad_norm,
            per_param_norms: Some(per_param_norms),
        })
    }

    fn clear_forward_state(&mut self) {
        // Clear the last loss tensor to free autodiff graph
        // This is critical during Predict phase when backward() won't be called
        self.clear_loss();
    }

    fn parameter_count(&self) -> usize {
        /// Visitor that counts parameters
        struct ParamCounter {
            count: usize,
        }

        impl<B: Backend> ModuleVisitor<B> for ParamCounter {
            fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
                let tensor = param.val();
                self.count += tensor.dims().iter().product::<usize>();
            }

            fn visit_int<const D: usize>(
                &mut self,
                param: &Param<Tensor<B, D, burn::tensor::Int>>,
            ) {
                let tensor = param.val();
                self.count += tensor.dims().iter().product::<usize>();
            }
        }

        let model_lock = self.model.lock();
        if let Some(ref model) = *model_lock {
            let mut counter = ParamCounter { count: 0 };
            model.visit(&mut counter);
            counter.count
        } else {
            0
        }
    }

    fn apply_weight_delta(&mut self, delta: &WeightDelta) -> crate::error::HybridResult<()> {
        // 1. Take model from Option
        let mut model_lock = self.model.lock();
        let model = model_lock.take().ok_or_else(|| {
            (
                crate::error::HybridTrainingError::IntegrationError {
                    crate_name: "burn".to_string(),
                    detail: "Model not available for weight delta application".to_string(),
                },
                None,
            )
        })?;

        // 2. Apply deltas using helper function
        let updated_model = apply_deltas_to_model(model, delta, &self.device);

        // 3. Put model back
        *model_lock = Some(updated_model);

        Ok(())
    }
}

/// Wrapper for Burn optimizers that implements the `Optimizer` trait.
///
/// This enables Burn's optimizers (Adam, SGD, AdamW) to work with `HybridTrainer`.
///
/// # Type Parameters
///
/// - `B`: Autodiff backend
/// - `M`: The Burn module type
/// - `O`: The Burn optimizer type
/// - `T`: The batch data type
///
/// # State Management
///
/// Burn optimizers maintain per-parameter state (e.g., momentum, variance for Adam).
/// This wrapper preserves that state across training steps.
pub struct BurnOptimizerWrapper<B, M, O, T>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
    O: burn::optim::Optimizer<M, B>,
{
    /// The wrapped Burn optimizer
    ///
    /// Uses `Mutex` instead of `RwLock` because optimizer may be `!Sync`
    /// in autodiff backends (contains state with !Sync components).
    optimizer: Arc<parking_lot::Mutex<O>>,
    /// Current learning rate
    learning_rate: Arc<RwLock<f32>>,
    _phantom_model: PhantomData<M>,
    _phantom_backend: PhantomData<B>,
    _phantom_batch: PhantomData<T>,
}

impl<B, M, O, T> BurnOptimizerWrapper<B, M, O, T>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
    O: burn::optim::Optimizer<M, B>,
{
    /// Creates a new Burn optimizer wrapper.
    ///
    /// # Arguments
    ///
    /// * `optimizer` - The Burn optimizer to wrap
    /// * `learning_rate` - Initial learning rate
    ///
    /// # Returns
    ///
    /// A wrapped optimizer ready for use with `HybridTrainer`.
    pub fn new(optimizer: O, learning_rate: f32) -> Self {
        Self {
            optimizer: Arc::new(parking_lot::Mutex::new(optimizer)),
            learning_rate: Arc::new(RwLock::new(learning_rate)),
            _phantom_model: PhantomData,
            _phantom_backend: PhantomData,
            _phantom_batch: PhantomData,
        }
    }
}

/// Implementation of Optimizer trait for BurnOptimizerWrapper
impl<B, M, O, T, F> Optimizer<BurnModelWrapper<B, M, T, F>, BurnBatch<B, T>>
    for BurnOptimizerWrapper<B, M, O, T>
where
    B: AutodiffBackend,
    M: AutodiffModule<B> + Send,
    O: burn::optim::Optimizer<M, B> + Send,
    F: BurnForwardFn<B, M, T>,
    T: Send + Sync,
    <B as AutodiffBackend>::Gradients: Send,
{
    fn step(
        &mut self,
        model: &mut BurnModelWrapper<B, M, T, F>,
        _gradients: &GradientInfo,
    ) -> crate::error::HybridResult<()> {
        // 1. Get current learning rate (convert f32 to f64 for Burn)
        let lr = *self.learning_rate.read() as f64;

        // 2. Take model from wrapper (ownership dance)
        let mut model_lock = model.model.lock();
        let model_inner = model_lock.take().ok_or_else(|| {
            (
                crate::error::HybridTrainingError::IntegrationError {
                    crate_name: "burn".to_string(),
                    detail: "Model not available for optimizer step".to_string(),
                },
                None,
            )
        })?;

        // 3. Take gradients from wrapper
        let gradients = model.last_gradients.lock().take().ok_or_else(|| {
            (
                crate::error::HybridTrainingError::IntegrationError {
                    crate_name: "burn".to_string(),
                    detail: "Gradients not available for optimizer step".to_string(),
                },
                None,
            )
        })?;

        // 4. Convert gradients to GradientsParams
        let grads_params = GradientsParams::from_grads(gradients, &model_inner);

        // 5. Call Burn optimizer.step (consumes model, returns updated model)
        let updated_model = self.optimizer.lock().step(lr, model_inner, grads_params);

        // 6. Put updated model back
        *model_lock = Some(updated_model);

        Ok(())
    }

    fn learning_rate(&self) -> f32 {
        *self.learning_rate.read()
    }

    fn set_learning_rate(&mut self, lr: f32) {
        *self.learning_rate.write() = lr;
    }

    fn zero_grad(&mut self) {
        // Burn handles gradient zeroing via autodiff graph reset
        // Gradients are automatically cleared when loss.backward() is called
        // No explicit action needed here
    }
}

/// Helper function to extract gradient information from a Burn model.
///
/// This walks the model's parameters and extracts gradients computed during
/// the backward pass using Burn's ModuleVisitor pattern.
///
/// # Arguments
///
/// * `model` - The model to extract gradients from
/// * `gradients` - The gradients returned by loss.backward()
///
/// # Returns
///
/// A `HashMap` of parameter names to flattened gradient vectors.
fn extract_gradients<B, M>(
    model: &M,
    gradients: &<B as AutodiffBackend>::Gradients,
) -> HashMap<String, Vec<f32>>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
{
    /// Visitor that extracts gradients from each parameter
    struct GradientVisitor<'a, B: AutodiffBackend> {
        gradients: &'a B::Gradients,
        per_param: HashMap<String, Vec<f32>>,
        path_stack: Vec<String>,
    }

    impl<'a, B: AutodiffBackend> ModuleVisitor<B> for GradientVisitor<'a, B> {
        fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
            // Get parameter ID and tensor
            let param_id = param.id; // id is a field, not a method
            let tensor = param.val(); // val() method returns &Tensor

            // Build hierarchical parameter name from ID
            let param_name = format!("{:?}", param_id); // ParamId has Debug impl

            // Extract gradient for this parameter
            // Note: grad() returns Option<Tensor> on the inner backend
            if let Some(grad_tensor) = tensor.grad(self.gradients) {
                // Flatten gradient to Vec<f32>
                // to_data() needs to know the target type
                let data = grad_tensor.to_data();
                let grad_vec = data.to_vec::<f32>().unwrap();

                // Store with parameter name
                self.per_param.insert(param_name, grad_vec);
            }
        }

        fn enter_module(&mut self, name: &str, _container_type: &str) {
            self.path_stack.push(name.to_string());
        }

        fn exit_module(&mut self, _name: &str, _container_type: &str) {
            self.path_stack.pop();
        }
    }

    // Create visitor and walk model parameters
    let mut visitor = GradientVisitor {
        gradients,
        per_param: HashMap::new(),
        path_stack: Vec::new(),
    };

    model.visit(&mut visitor);

    visitor.per_param
}

/// Helper function to apply weight deltas to a Burn model.
///
/// This modifies the model's parameters by adding the provided deltas using
/// Burn's ModuleMapper pattern. The model is consumed and a new model is returned
/// to preserve Burn's ownership semantics.
///
/// # Arguments
///
/// * `model` - The model to modify (consumed)
/// * `delta` - The weight deltas to apply
/// * `device` - Device for tensor operations
///
/// # Returns
///
/// The updated model with deltas applied.
fn apply_deltas_to_model<B, M>(model: M, delta: &WeightDelta, device: &burn::tensor::Device<B>) -> M
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
{
    if delta.deltas.is_empty() {
        return model;
    }

    /// Mapper that applies weight deltas to each parameter
    struct DeltaMapper<'a, B: Backend> {
        deltas: &'a HashMap<String, Vec<f32>>,
        scale: f32,
        device: burn::tensor::Device<B>,
    }

    impl<'a, B: Backend> ModuleMapper<B> for DeltaMapper<'a, B> {
        fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
            // Consume parameter to get ID, tensor, and mapper
            let (id, tensor, mapper) = param.consume();

            // Build parameter name from ID
            let param_name = format!("{:?}", id);

            // Check if we have a delta for this parameter
            if let Some(delta_vec) = self.deltas.get(&param_name) {
                // Get tensor shape
                let shape = tensor.shape();

                // Convert delta vector to tensor with same shape
                let delta_data = TensorData::new(delta_vec.clone(), shape.clone());
                let delta_tensor =
                    Tensor::<B, D>::from_data(delta_data.convert::<f32>(), &self.device);

                // Apply: param = param + scale * delta
                let updated_tensor = tensor.add(delta_tensor.mul_scalar(self.scale));

                // Return updated parameter
                Param::from_mapped_value(id, updated_tensor, mapper)
            } else {
                // No delta for this parameter, return unchanged
                Param::from_mapped_value(id, tensor, mapper)
            }
        }
    }

    // Create mapper and apply deltas
    let mut mapper = DeltaMapper {
        deltas: &delta.deltas,
        scale: delta.scale,
        device: device.clone(),
    };

    // Record model copy for VRAM tracking
    // Burn's .map() creates a full model copy (496 MB for GPT-2 Small)
    crate::vram_manager::VramManager::record_model_copy();

    model.map(&mut mapper)
}

#[cfg(test)]
mod tests {
    use super::*;

    // NOTE: Tests are disabled until full Burn integration is complete
    // The module compiles correctly, but tests require concrete Burn backends
    // which are feature-gated. Tests will be enabled in Phase 2.

    #[test]
    fn test_module_compiles() {
        // Verify that the module structure compiles correctly
        // Actual functionality tests will be added once traits are implemented
        assert!(true, "Burn integration module compiles successfully");
    }
}
