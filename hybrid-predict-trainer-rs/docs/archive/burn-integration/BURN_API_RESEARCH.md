# Burn 0.20 API Research Findings

**Research Date:** 2026-02-06
**Purpose:** Document Burn 0.20 framework APIs for hybrid-predict-trainer-rs integration
**Agent:** general-purpose research agent

---

## Executive Summary

Burn 0.20 uses an **ownership-based autodiff model** fundamentally different from PyTorch's reference-based approach. Key implications:

1. **Models are consumed and returned** by optimizer.step()
2. **Gradients are returned** by loss.backward(), not stored in tensors
3. **ModuleVisitor pattern** for parameter iteration (read-only)
4. **ModuleMapper pattern** for parameter transformation (ownership transfer)
5. **Type-safe tensor dimensions** enforced at compile time

---

## Core Architecture Patterns

### 1. Autodiff Backend

```rust
use burn::tensor::backend::{Backend, AutodiffBackend};
use burn::module::{Module, AutodiffModule};

// Backend trait hierarchy
trait Backend {
    type FloatElem;
    type Device;
    type FloatTensorPrimitive<const D: usize>;
    // ... other associated types
}

trait AutodiffBackend: Backend {
    type InnerBackend: Backend;
    type Gradients: Gradients<Self>;
}

// Autodiff module wraps inner module
trait AutodiffModule<B: AutodiffBackend>: Module<B> {
    type InnerModule: Module<B::InnerBackend>;

    fn valid(&self) -> Self::InnerModule;
}
```

**Key Insight:** Autodiff tracking is at the backend level, not tensor level. The `AutodiffModule` wraps an `InnerModule` that operates on the non-autodiff backend.

### 2. Tensor Operations

```rust
use burn::tensor::Tensor;

// Tensors are parameterized by backend and dimensionality
struct Tensor<B: Backend, const D: usize> { /* ... */ }

impl<B: Backend, const D: usize> Tensor<B, D> {
    // Data extraction
    fn into_data(self) -> TensorData;
    fn to_data(&self) -> TensorData;

    // Creation
    fn from_data(data: TensorData, device: &B::Device) -> Self;
    fn zeros(shape: Shape<D>, device: &B::Device) -> Self;
    fn ones(shape: Shape<D>, device: &B::Device) -> Self;

    // Operations
    fn add(self, other: Self) -> Self;
    fn mul(self, other: Self) -> Self;
    fn matmul(self, other: Tensor<B, D>) -> Tensor<B, D>;

    // Reshaping
    fn reshape<const D2: usize>(self, shape: Shape<D2>) -> Tensor<B, D2>;
    fn flatten<const D2: usize>(self, start_dim: usize, end_dim: usize) -> Tensor<B, D2>;
}

// TensorData for host-side data
struct TensorData {
    // Methods
    fn to_vec<E: Element>(self) -> Vec<E>;
    fn from_vec<E: Element, S: Into<Shape>>(vec: Vec<E>, shape: S) -> Self;
}
```

**Critical Pattern:** Tensor → Vec<f32> conversion:
```rust
let vec: Vec<f32> = tensor.into_data().to_vec::<f32>();
```

**Critical Pattern:** Vec<f32> → Tensor conversion:
```rust
let tensor = Tensor::<B, 1>::from_data(
    TensorData::from_vec(vec, Shape::new([vec.len()])),
    device
);
```

### 3. Loss and Backward Pass

```rust
use burn::train::loss::{Loss, LossBackward};

// Loss computation returns loss and backward handle
impl<B: AutodiffBackend> Loss<B> for CrossEntropyLoss {
    fn forward(&self, predictions: Tensor<B, 2>, targets: Tensor<B::InnerBackend, 1>)
        -> Tensor<B, 1> {
        // Compute loss with autodiff tracking
        let loss = cross_entropy(predictions, targets);
        loss
    }
}

// Backward pass
let loss_tensor: Tensor<B, 1> = loss_fn.forward(predictions, targets);
let gradients: B::Gradients = loss_tensor.backward();
```

**Key Insight:** `loss.backward()` returns `B::Gradients` struct containing all gradients, not stored in tensors themselves.

### 4. Gradient Access

```rust
use burn::module::Param;

// Gradients struct (opaque, backend-specific)
trait Gradients<B: AutodiffBackend> {
    // Internal representation, not directly accessible
}

// Access gradients via ModuleVisitor after backward
struct GradientExtractor<B: AutodiffBackend> {
    gradients: HashMap<String, Tensor<B::InnerBackend, 1>>,
}

impl<B: AutodiffBackend> ModuleVisitor<B> for GradientExtractor<B> {
    fn visit<const D: usize>(&mut self, id: &ParamId, tensor: &Tensor<B, D>) {
        // Extract gradient for this parameter from stored B::Gradients
        // This requires backend-specific gradient lookup
        // Burn provides: tensor.grad() on autodiff tensors
    }
}
```

**Critical Discovery:** Burn 0.20 has `Tensor::grad()` method on autodiff tensors:

```rust
// After loss.backward(), gradients are stored in the computation graph
let grad: Tensor<B::InnerBackend, D> = param_tensor.grad(&gradients);
```

### 5. Module Parameter Iteration

```rust
use burn::module::{Module, ModuleVisitor, ParamId};

// Read-only parameter iteration
trait ModuleVisitor<B: Backend> {
    fn visit<const D: usize>(&mut self, id: &ParamId, tensor: &Tensor<B, D>);
}

impl<B: Backend, M: Module<B>> M {
    fn visit<V: ModuleVisitor<B>>(&self, visitor: &mut V);
}

// Example: Count parameters
struct ParamCounter {
    count: usize,
}

impl<B: Backend> ModuleVisitor<B> for ParamCounter {
    fn visit<const D: usize>(&mut self, _id: &ParamId, tensor: &Tensor<B, D>) {
        self.count += tensor.dims().iter().product::<usize>();
    }
}

let mut counter = ParamCounter { count: 0 };
model.visit(&mut counter);
println!("Total parameters: {}", counter.count);
```

**Key Pattern:** Use `ModuleVisitor` for read-only parameter access (gradient extraction, counting, etc.)

### 6. Module Parameter Transformation

```rust
use burn::module::{Module, ModuleMapper};

// Ownership-based parameter transformation
trait ModuleMapper<B: Backend> {
    fn map<const D: usize>(&mut self, id: &ParamId, tensor: Tensor<B, D>) -> Tensor<B, D>;
}

impl<B: Backend, M: Module<B>> M {
    fn map<Mapper: ModuleMapper<B>>(self, mapper: &mut Mapper) -> Self;
}

// Example: Apply weight delta
struct WeightDeltaApplier<B: Backend> {
    deltas: HashMap<String, Vec<f32>>,
    device: Device<B>,
}

impl<B: Backend> ModuleMapper<B> for WeightDeltaApplier<B> {
    fn map<const D: usize>(&mut self, id: &ParamId, tensor: Tensor<B, D>) -> Tensor<B, D> {
        let param_name = id.to_string();
        if let Some(delta_vec) = self.deltas.get(&param_name) {
            let delta_tensor = Tensor::from_data(
                TensorData::from_vec(delta_vec.clone(), tensor.shape()),
                &self.device
            );
            tensor.add(delta_tensor)
        } else {
            tensor
        }
    }
}

// Apply deltas
let updated_model = model.map(&mut applier);
```

**Key Pattern:** Use `ModuleMapper` for parameter modification (weight deltas, quantization, etc.)

### 7. Optimizer Interface

```rust
use burn::optim::{Optimizer, OptimizerAdaptor, GradientsParams};

trait Optimizer<M: Module<B>, B: AutodiffBackend>: Send + Sync {
    // Step consumes model and gradients, returns updated model
    fn step(&mut self, lr: f32, model: M, grads: GradientsParams) -> M;
}

// OptimizerAdaptor provides convenience methods
struct OptimizerAdaptor<O, M, B>
where
    O: Optimizer<M, B>,
    M: AutodiffModule<B>,
    B: AutodiffBackend,
{
    optimizer: O,
}

impl<O, M, B> OptimizerAdaptor<O, M, B> {
    pub fn backward_step(&mut self, lr: f32, model: M, loss: Tensor<B, 1>) -> M {
        // 1. Call loss.backward() to get gradients
        let grads = loss.backward();

        // 2. Convert to GradientsParams
        let grads_params = GradientsParams::from_grads(grads, &model);

        // 3. Call optimizer.step() with model (consumes it)
        self.optimizer.step(lr, model, grads_params)
    }
}
```

**Critical Pattern:** Optimizer workflow:
```rust
// 1. Forward pass (model consumed)
let (model, predictions) = model.forward(batch);

// 2. Loss computation
let loss = loss_fn.forward(predictions, targets);

// 3. Backward + step (model consumed, new model returned)
let model = optimizer.backward_step(lr, model, loss);
```

**This is fundamentally different from PyTorch:**
- PyTorch: model is mutated in-place
- Burn: model is consumed and a new model is returned

### 8. Learning Rate Scheduling

```rust
use burn::lr_scheduler::{LrScheduler, LrSchedulerConfig};

trait LrScheduler: Send + Sync {
    fn step(&mut self) -> f32;
}

// Learning rate is passed to optimizer.step() per-call
let lr = scheduler.step();
let model = optimizer.step(lr, model, grads);
```

**Key Insight:** Learning rate is an argument to `step()`, not stored in optimizer. This makes our LR tracking straightforward.

---

## Implementation Implications

### 1. BurnModelWrapper Must Use Option<M>

```rust
pub struct BurnModelWrapper<B, M, T>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
{
    // CRITICAL: Use Option to take ownership during forward/backward
    model: Arc<RwLock<Option<M>>>,
    device: Device<B>,
    last_gradients: Arc<RwLock<Option<B::Gradients>>>,
    _phantom: PhantomData<T>,
}

impl<B, M, T> Model<BurnBatch<B, T>> for BurnModelWrapper<B, M, T> {
    fn forward(&mut self, batch: &BurnBatch<B, T>) -> HybridResult<f32> {
        // 1. Take model out of Option (ownership transfer)
        let model = self.model.write().take().expect("Model already taken");

        // 2. Call user-provided forward function
        let (model, loss_tensor) = forward_fn(model, batch);

        // 3. Extract scalar loss
        let loss_scalar = loss_tensor.to_data().to_vec::<f32>()[0];

        // 4. Store loss tensor for backward
        *self.last_loss.write() = Some(loss_tensor);

        // 5. Put model back
        *self.model.write() = Some(model);

        Ok(loss_scalar)
    }

    fn backward(&mut self) -> HybridResult<GradientInfo> {
        // 1. Take loss tensor
        let loss = self.last_loss.write().take().expect("No loss to backward");

        // 2. Call backward
        let gradients = loss.backward();

        // 3. Store gradients for optimizer
        *self.last_gradients.write() = Some(gradients);

        // 4. Extract gradient info
        // ... use ModuleVisitor to walk parameters and compute norms
    }
}
```

### 2. Need BurnForwardFn Trait

Users must provide a function that:
- Takes ownership of model + batch
- Returns (model, loss_tensor)

```rust
pub trait BurnForwardFn<B, M, T>: Send + Sync
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
{
    fn forward(&self, model: M, batch: &BurnBatch<B, T>) -> (M, Tensor<B, 1>);
}

// User implements this for their model
impl BurnForwardFn<NdArrayBackend, SimpleMLP, MnistBatch> for MnistForward {
    fn forward(&self, model: SimpleMLP, batch: &BurnBatch<...>) -> (...) {
        let (model, logits) = model.forward(batch.inputs);
        let loss = cross_entropy(logits, batch.targets);
        (model, loss)
    }
}
```

### 3. Gradient Extraction via ModuleVisitor

```rust
struct GradientExtractor<B: AutodiffBackend> {
    gradients: B::Gradients,
    per_param_grads: HashMap<String, Vec<f32>>,
    device: Device<B>,
}

impl<B: AutodiffBackend> ModuleVisitor<B> for GradientExtractor<B> {
    fn visit<const D: usize>(&mut self, id: &ParamId, tensor: &Tensor<B, D>) {
        // Get gradient for this parameter
        let grad_tensor: Tensor<B::InnerBackend, D> = tensor.grad(&self.gradients);

        // Convert to Vec<f32>
        let grad_vec: Vec<f32> = grad_tensor.into_data().to_vec();

        // Store with parameter name
        let param_name = id.to_string();
        self.per_param_grads.insert(param_name, grad_vec);
    }
}

// Use in backward()
let mut extractor = GradientExtractor {
    gradients: gradients.clone(),
    per_param_grads: HashMap::new(),
    device: self.device.clone(),
};

let model = self.model.read();
model.as_ref().unwrap().visit(&mut extractor);

// Now extractor.per_param_grads contains all gradients as Vec<f32>
```

### 4. Weight Delta Application via ModuleMapper

```rust
struct WeightDeltaApplier<B: Backend> {
    deltas: HashMap<String, Vec<f32>>,
    device: Device<B>,
    scale: f32,
}

impl<B: Backend> ModuleMapper<B> for WeightDeltaApplier<B> {
    fn map<const D: usize>(&mut self, id: &ParamId, tensor: Tensor<B, D>) -> Tensor<B, D> {
        let param_name = id.to_string();

        if let Some(delta_vec) = self.deltas.get(&param_name) {
            // Convert delta to tensor with same shape
            let shape = tensor.shape();
            let delta_tensor = Tensor::from_data(
                TensorData::from_vec(delta_vec.clone(), shape),
                &self.device
            );

            // Apply: param = param + scale * delta
            tensor.add(delta_tensor.mul_scalar(self.scale))
        } else {
            // No delta for this parameter, return unchanged
            tensor
        }
    }
}

// Use in apply_weight_delta()
let mut applier = WeightDeltaApplier {
    deltas: delta.deltas.clone(),
    device: self.device.clone(),
    scale: delta.scale,
};

let model = self.model.write().take().unwrap();
let updated_model = model.map(&mut applier);
*self.model.write() = Some(updated_model);
```

### 5. Optimizer Wrapper Pattern

```rust
pub struct BurnOptimizerWrapper<B, M, O, T>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
    O: Optimizer<M, B>,
{
    optimizer: Arc<RwLock<OptimizerAdaptor<O, M, B>>>,
    learning_rate: Arc<RwLock<f32>>,
    _phantom: PhantomData<T>,
}

impl<B, M, O, T> Optimizer<BurnModelWrapper<B, M, T>, BurnBatch<B, T>>
    for BurnOptimizerWrapper<B, M, O, T>
{
    fn step(&mut self, model: &mut BurnModelWrapper<B, M, T>, _grads: &GradientInfo)
        -> HybridResult<()> {
        // 1. Get learning rate
        let lr = *self.learning_rate.read();

        // 2. Take model and gradients
        let model_inner = model.model.write().take().unwrap();
        let grads = model.last_gradients.write().take().unwrap();

        // 3. Convert gradients to GradientsParams
        let grads_params = GradientsParams::from_grads(grads, &model_inner);

        // 4. Call optimizer.step (consumes model, returns new model)
        let updated_model = self.optimizer.write().step(lr, model_inner, grads_params);

        // 5. Put model back
        *model.model.write() = Some(updated_model);

        Ok(())
    }
}
```

---

## Parameter Naming Convention

Burn's `ParamId` generates hierarchical names like:
- `"fc1.weight"`
- `"fc1.bias"`
- `"layers.0.weight"`
- `"layers.0.bias"`

**Match these names in WeightDelta HashMap keys.**

Example module hierarchy:
```rust
#[derive(Module, Debug)]
struct SimpleMLP<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
}

// ParamIds will be:
// "fc1.weight" (shape: [128, 784])
// "fc1.bias"   (shape: [128])
// "fc2.weight" (shape: [10, 128])
// "fc2.bias"   (shape: [10])
```

---

## Device Handling

```rust
use burn::tensor::Device;

// Device is generic over backend
struct Device<B: Backend> { /* ... */ }

// Common devices
let cpu_device = Device::<NdArrayBackend>::Cpu;
let cuda_device = Device::<CudaBackend>::Cuda(0); // GPU 0

// Move tensors between devices
let tensor_gpu = tensor_cpu.to_device(&cuda_device);
```

**Important:** All tensors in a computation must be on the same device.

---

## Common Backends

### 1. NdArray (CPU)

```toml
[dependencies]
burn = { version = "0.20", features = ["ndarray"] }
```

```rust
use burn::backend::ndarray::NdArrayBackend;

type Backend = NdArrayBackend<f32>;
let device = Device::<Backend>::Cpu;
```

### 2. WGPU (GPU, cross-platform)

```toml
[dependencies]
burn = { version = "0.20", features = ["wgpu"] }
```

```rust
use burn::backend::wgpu::WgpuBackend;

type Backend = WgpuBackend<f32, i32>;
let device = Device::<Backend>::default(); // Auto-select GPU
```

### 3. CUDA (NVIDIA GPUs)

```toml
[dependencies]
burn = { version = "0.20", features = ["cuda"] }
```

```rust
use burn::backend::cuda::CudaBackend;

type Backend = CudaBackend<f32>;
let device = Device::<Backend>::Cuda(0); // GPU 0
```

### 4. Autodiff Wrapper

```rust
use burn::backend::Autodiff;

// Wrap any backend with autodiff
type AutodiffNdArray = Autodiff<NdArrayBackend<f32>>;
type AutodiffCuda = Autodiff<CudaBackend<f32>>;
```

---

## Testing Recommendations

### Unit Tests: Use NdArray Backend

```rust
#[cfg(test)]
mod tests {
    use burn::backend::ndarray::NdArrayBackend;
    type TestBackend = NdArrayBackend<f32>;

    #[test]
    fn test_forward() {
        let device = Device::<TestBackend>::Cpu;
        // ... test logic
    }
}
```

### Integration Tests: Parameterize Backend

```rust
#[cfg(test)]
mod tests {
    fn test_training<B: AutodiffBackend>() {
        // Generic test that works with any backend
    }

    #[test]
    fn test_training_ndarray() {
        test_training::<Autodiff<NdArrayBackend<f32>>>();
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_training_cuda() {
        test_training::<Autodiff<CudaBackend<f32>>>();
    }
}
```

---

## Error Handling

```rust
use burn::tensor::TensorError;

// Burn operations return Result
let result: Result<Tensor<B, 2>, TensorError> = tensor.try_reshape([10, 10]);

// Convert to HybridResult
result.map_err(|e| {
    (HybridTrainingError::IntegrationError {
        crate_name: "burn".to_string(),
        detail: format!("Tensor operation failed: {}", e),
    }, None)
})?
```

---

## Performance Considerations

### 1. Minimize Device Transfers

```rust
// Bad: Multiple CPU ↔ GPU transfers
let vec = tensor.to_data().to_vec::<f32>(); // GPU → CPU
let new_tensor = Tensor::from_data(..., device); // CPU → GPU

// Good: Batch transfers
let all_vecs: Vec<Vec<f32>> = tensors.iter()
    .map(|t| t.to_data().to_vec())
    .collect();
```

### 2. Use In-Place Operations

```rust
// Prefer in-place when possible
tensor = tensor.add_scalar(1.0); // May reuse memory

// Avoid unnecessary clones
let tensor2 = tensor.clone(); // Copies data
```

### 3. Tensor Fusion

Burn automatically fuses operations where possible, but explicit fusion can help:

```rust
// Burn will fuse: y = (x + 1) * 2
let y = x.add_scalar(1.0).mul_scalar(2.0);
```

---

## Key Differences from PyTorch

| Aspect | PyTorch | Burn |
|--------|---------|------|
| Ownership | Reference-based, shared ownership | Ownership-based, move semantics |
| Gradients | Stored in `.grad` attribute | Returned by `.backward()` |
| Optimizer | Mutates model in-place | Consumes and returns model |
| Type safety | Runtime shape checks | Compile-time dimension checks |
| Device | Runtime device tracking | Backend type parameter |
| Autograd | Implicit via `requires_grad` | Explicit `AutodiffBackend` |

---

## Summary of Critical Patterns

### Tensor Conversion

```rust
// Tensor → Vec<f32>
let vec: Vec<f32> = tensor.into_data().to_vec();

// Vec<f32> → Tensor
let tensor = Tensor::from_data(
    TensorData::from_vec(vec, shape),
    device
);
```

### Ownership Dance

```rust
// Take model from Option
let model = self.model.write().take().unwrap();

// Use model (ownership transfer)
let (model, result) = operation(model);

// Put model back
*self.model.write() = Some(model);
```

### Gradient Extraction

```rust
// After loss.backward()
let gradients = loss.backward();

// Extract per-parameter gradients
struct GradientVisitor { /* ... */ }
impl ModuleVisitor for GradientVisitor {
    fn visit(&mut self, id: &ParamId, tensor: &Tensor<B, D>) {
        let grad = tensor.grad(&self.gradients);
        // Process gradient
    }
}

model.visit(&mut visitor);
```

### Weight Delta Application

```rust
// Apply deltas
struct DeltaMapper { /* ... */ }
impl ModuleMapper for DeltaMapper {
    fn map(&mut self, id: &ParamId, tensor: Tensor<B, D>) -> Tensor<B, D> {
        // Modify and return tensor
    }
}

let updated_model = model.map(&mut mapper);
```

### Optimizer Step

```rust
// Forward
let (model, predictions) = model.forward(batch);

// Loss
let loss = loss_fn.forward(predictions, targets);

// Backward + step (model consumed)
let model = optimizer.backward_step(lr, model, loss);
```

---

## Next Steps for Implementation

1. **Update Cargo.toml** - Consolidate burn dependencies with features
2. **Implement BurnForwardFn trait** - User-defined loss computation
3. **Implement helper functions** - Tensor conversion, gradient extraction, delta application
4. **Implement Model trait for BurnModelWrapper** - Use ownership patterns
5. **Implement Optimizer trait for BurnOptimizerWrapper** - Handle model consumption
6. **Write comprehensive tests** - Cover ownership, gradients, deltas
7. **Add examples** - SimpleMLP MNIST, gradient flow verification

---

## References

- **Burn Book:** https://burn.dev/book/
- **Burn API Docs:** https://docs.rs/burn/0.20.0/burn/
- **Burn Examples:** https://github.com/tracel-ai/burn/tree/main/examples
- **Module System:** https://burn.dev/book/building-blocks/module
- **Autodiff:** https://burn.dev/book/building-blocks/autodiff
- **Optimizer:** https://burn.dev/book/building-blocks/optimizer

---

**Status:** Research complete, ready for Phase 2 implementation
**Confidence:** High - All critical APIs identified and patterns documented
**Risk:** Medium - Ownership dance complexity requires careful testing
