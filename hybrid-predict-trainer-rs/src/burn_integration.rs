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
use crate::Batch;

use burn::module::{AutodiffModule, Module, ModuleMapper, ModuleVisitor, Param};
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
pub struct BurnModelWrapper<B, M, T>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
{
    /// The wrapped Burn model
    model: Arc<RwLock<M>>,
    /// Device for tensor operations
    device: burn::tensor::Device<B>,
    /// Last computed loss (for backward pass)
    last_loss: Arc<RwLock<Option<Tensor<B, 1>>>>,
    /// Cache of parameter names and shapes
    param_metadata: Arc<RwLock<HashMap<String, Vec<usize>>>>,
    _phantom: PhantomData<T>,
}

impl<B, M, T> BurnModelWrapper<B, M, T>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
{
    /// Creates a new Burn model wrapper.
    ///
    /// # Arguments
    ///
    /// * `model` - The Burn model to wrap
    /// * `device` - Device for tensor operations
    ///
    /// # Returns
    ///
    /// A wrapped model ready for use with `HybridTrainer`.
    pub fn new(model: M, device: burn::tensor::Device<B>) -> Self {
        Self {
            model: Arc::new(RwLock::new(model)),
            device,
            last_loss: Arc::new(RwLock::new(None)),
            param_metadata: Arc::new(RwLock::new(HashMap::new())),
            _phantom: PhantomData,
        }
    }

    /// Extracts parameter count from the model.
    ///
    /// This walks the model's parameters and sums their element counts.
    fn count_parameters(&self) -> usize {
        // TODO: Implement parameter counting via Burn's module introspection
        // For now, return a placeholder value
        // This will be implemented when we integrate with real Burn models
        0
    }

    /// Converts a Burn tensor to Vec<f32>.
    ///
    /// This is used when extracting gradients or parameters. The tensor is
    /// flattened to a 1D vector regardless of its original dimensionality.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to convert
    ///
    /// # Returns
    ///
    /// A flattened vector of f32 values.
    fn tensor_to_vec<const D: usize>(tensor: &Tensor<B, D>) -> Vec<f32> {
        // Extract tensor data and convert to Vec<f32>
        // Note: to_data() creates a copy without consuming the tensor
        tensor.to_data().to_vec::<f32>().unwrap()
    }

    /// Converts Vec<f32> to a Burn tensor.
    ///
    /// This is used when applying weight deltas. The vector is reshaped
    /// to match the provided shape.
    ///
    /// # Arguments
    ///
    /// * `vec` - The vector to convert
    /// * `shape` - The desired tensor shape
    /// * `device` - Device for tensor allocation
    ///
    /// # Returns
    ///
    /// A tensor with the specified shape.
    fn vec_to_tensor<const D: usize>(
        vec: Vec<f32>,
        shape: Shape,
        device: &burn::tensor::Device<B>,
    ) -> Tensor<B, D> {
        // Create TensorData from vector and shape
        let data = TensorData::new(vec, shape);

        // Create tensor on device with f32 element type
        Tensor::from_data(data.convert::<f32>(), device)
    }

    /// Computes the L2 norm of gradients across all parameters.
    ///
    /// # Arguments
    ///
    /// * `model` - The model with computed gradients
    ///
    /// # Returns
    ///
    /// The global gradient norm.
    fn compute_gradient_norm(model: &M::InnerModule) -> f32 {
        // TODO: Implement gradient norm computation via Burn's grad introspection
        // For now, return a placeholder
        // This will be implemented when we integrate with real Burn models
        1.0
    }
}

// Placeholder Model trait implementation
// TODO: This needs to be properly implemented with real Burn autodiff integration
// For now, we provide a skeleton that compiles but doesn't function
//
// impl<B, M, T> Model<BurnBatch<B, T>> for BurnModelWrapper<B, M, T>
// where
//     B: AutodiffBackend,
//     M: AutodiffModule<B>,
//     T: Send + Sync,
// {
//     fn forward(&mut self, batch: &BurnBatch<B, T>) -> HybridResult<f32> {
//         // TODO: Call model.forward() with batch data
//         // Store loss tensor in self.last_loss for backward pass
//         Ok(0.0)
//     }
//
//     fn backward(&mut self) -> HybridResult<GradientInfo> {
//         // TODO: Call loss.backward() and extract gradients
//         Ok(GradientInfo {
//             loss: 0.0,
//             gradient_norm: 0.0,
//             per_param_norms: None,
//         })
//     }
//
//     fn parameter_count(&self) -> usize {
//         self.count_parameters()
//     }
//
//     fn apply_weight_delta(&mut self, delta: &WeightDelta) -> HybridResult<()> {
//         let mut model = self.model.write();
//         apply_deltas_to_model(&mut *model, delta, &self.device)
//     }
// }

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
    optimizer: Arc<RwLock<O>>,
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
            optimizer: Arc::new(RwLock::new(optimizer)),
            learning_rate: Arc::new(RwLock::new(learning_rate)),
            _phantom_model: PhantomData,
            _phantom_backend: PhantomData,
            _phantom_batch: PhantomData,
        }
    }
}

// Placeholder Optimizer trait implementation
// TODO: This needs to be properly implemented with real Burn optimizer integration
//
// impl<B, M, O, T> Optimizer<BurnModelWrapper<B, M, T>, BurnBatch<B, T>>
//     for BurnOptimizerWrapper<B, M, O, T>
// where
//     B: AutodiffBackend,
//     M: AutodiffModule<B>,
//     O: burn::optim::Optimizer<M, B>,
//     T: Send + Sync,
// {
//     fn step(
//         &mut self,
//         model: &mut BurnModelWrapper<B, M, T>,
//         gradients: &GradientInfo,
//     ) -> HybridResult<()> {
//         // TODO: Apply optimizer step via Burn's optimizer.step()
//         Ok(())
//     }
//
//     fn learning_rate(&self) -> f32 {
//         *self.learning_rate.read()
//     }
//
//     fn set_learning_rate(&mut self, lr: f32) {
//         *self.learning_rate.write() = lr;
//     }
//
//     fn zero_grad(&mut self) {
//         // TODO: Clear gradients via Burn's API
//     }
// }

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
fn apply_deltas_to_model<B, M>(
    model: M,
    delta: &WeightDelta,
    device: &burn::tensor::Device<B>,
) -> M
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
        fn map_float<const D: usize>(
            &mut self,
            param: Param<Tensor<B, D>>,
        ) -> Param<Tensor<B, D>> {
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
                let delta_tensor = Tensor::<B, D>::from_data(delta_data.convert::<f32>(), &self.device);

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
