# Burn Integration Status

**Created:** 2026-02-06
**Task:** Implement Burn model wrapper for real training (Task #1)
**Status:** 🟡 Phase 1 Complete - Skeleton & Foundation

---

## What Was Accomplished

### ✅ Module Structure Created

**File:** `src/burn_integration.rs` (420 lines)

Created comprehensive Burn integration module with:

1. **BurnBatch<B, T>** - Wrapper for Burn batches implementing `Batch` trait
2. **BurnModelWrapper<B, M, T>** - Wrapper for Burn `AutodiffModule` models
3. **BurnOptimizerWrapper<B, M, O, T>** - Wrapper for Burn optimizers
4. **Helper Functions** - Tensor conversion, gradient extraction, delta application

### ✅ Example Created

**File:** `examples/burn_mlp_mnist.rs` (150 lines)

Created working example demonstrating:
- SimpleMLP model (784 → 128 → 10) for MNIST
- BurnModelWrapper usage pattern
- Integration with HybridTrainer (commented out pending implementation)

### ✅ Tests Scaffolded

**File:** `tests/burn_integration_test.rs` (120 lines)

Created test file with 12 test stubs covering:
- Wrapper creation and properties
- Forward/backward passes
- Weight delta application
- Optimizer steps
- End-to-end training loop

### ✅ Compilation Verified

- `cargo check` passes ✓
- `cargo check --example burn_mlp_mnist` passes ✓
- `cargo test --test burn_integration_test` passes (2 tests, 10 ignored) ✓

---

## What Remains (Next Steps)

### 🔴 Phase 2: Implement Model Trait

**Target File:** `src/burn_integration.rs`

Need to implement for `BurnModelWrapper`:

```rust
impl<B, M, T> Model<BurnBatch<B, T>> for BurnModelWrapper<B, M, T>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
    T: Send + Sync,
{
    fn forward(&mut self, batch: &BurnBatch<B, T>) -> HybridResult<f32> {
        // 1. Extract model and call model.forward(batch.data)
        // 2. Compute loss (e.g., cross-entropy)
        // 3. Store loss tensor in self.last_loss for backward
        // 4. Return scalar loss value
    }

    fn backward(&mut self) -> HybridResult<GradientInfo> {
        // 1. Get stored loss tensor from self.last_loss
        // 2. Call loss.backward() to trigger autodiff
        // 3. Extract gradients from model parameters
        // 4. Compute gradient norm
        // 5. Return GradientInfo with loss, norm, per-param norms
    }

    fn parameter_count(&self) -> usize {
        // Walk model.parameters() and sum element counts
    }

    fn apply_weight_delta(&mut self, delta: &WeightDelta) -> HybridResult<()> {
        // 1. Lock model
        // 2. For each (param_name, delta_vec) in delta.deltas:
        //    - Find corresponding parameter in model
        //    - Convert delta_vec to Tensor
        //    - Add delta to parameter (param += delta * scale)
        // 3. Preserve autodiff graph
    }
}
```

**Key Challenges:**

1. **Loss Computation** - Need to define how loss is computed from model output
   - Cross-entropy for classification?
   - MSE for regression?
   - User-provided loss function?

2. **Gradient Extraction** - Need to walk Burn's parameter tree and extract gradients
   - Use `model.parameters()` or similar API
   - Convert Tensor gradients to Vec<f32>
   - Compute L2 norm across all parameters

3. **Weight Delta Application** - Need to modify parameters without breaking autodiff
   - Must preserve gradient tracking
   - Need to map WeightDelta param names to Burn parameter paths
   - Handle tensor shape mismatches

### 🔴 Phase 3: Implement Optimizer Trait

**Target File:** `src/burn_integration.rs`

Need to implement for `BurnOptimizerWrapper`:

```rust
impl<B, M, O, T> Optimizer<BurnModelWrapper<B, M, T>, BurnBatch<B, T>>
    for BurnOptimizerWrapper<B, M, O, T>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
    O: burn::optim::Optimizer<M, B>,
    T: Send + Sync,
{
    fn step(&mut self, model: &mut BurnModelWrapper<B, M, T>, _gradients: &GradientInfo) -> HybridResult<()> {
        // 1. Lock optimizer and model
        // 2. Call optimizer.step() with model and gradients
        // 3. Burn optimizers handle gradient application internally
    }

    fn learning_rate(&self) -> f32 {
        *self.learning_rate.read()
    }

    fn set_learning_rate(&mut self, lr: f32) {
        *self.learning_rate.write() = lr;
        // Also update Burn optimizer's LR if applicable
    }

    fn zero_grad(&mut self) {
        // Burn handles this internally via autodiff reset
        // May not need explicit implementation
    }
}
```

**Key Challenges:**

1. **Optimizer API Mismatch** - Burn optimizers have different step() signatures
   - Need to adapt GradientInfo to Burn's gradient format
   - May need to store gradients separately

2. **Learning Rate Updates** - Burn optimizers may handle LR differently
   - Some optimizers have LR as part of state
   - Others use external schedulers
   - Need to sync our LR tracking with Burn's

### 🟡 Phase 4: Helper Function Implementation

**Target Functions in `src/burn_integration.rs`:**

1. `tensor_to_vec()` - Convert Burn Tensor to Vec<f32>
   ```rust
   fn tensor_to_vec(tensor: &Tensor<B, 1>) -> Vec<f32> {
       tensor.to_data().to_vec()
   }
   ```

2. `vec_to_tensor()` - Convert Vec<f32> to Burn Tensor
   ```rust
   fn vec_to_tensor(vec: &[f32], device: &Device<B>) -> Tensor<B, 1> {
       Tensor::from_data(vec, device)
   }
   ```

3. `extract_gradients()` - Walk model and extract gradients
   ```rust
   fn extract_gradients<B, M>(model: &M::InnerModule) -> HashMap<String, Vec<f32>> {
       // Use Burn's parameter iteration API
       // For each parameter: (name, tensor) -> (name, gradient_vec)
   }
   ```

4. `apply_deltas_to_model()` - Modify model parameters
   ```rust
   fn apply_deltas_to_model<B, M>(model: &mut M, delta: &WeightDelta, device: &Device<B>) -> HybridResult<()> {
       // For each delta: find param, convert to tensor, add to param
   }
   ```

---

## Implementation Plan

### Step 1: Research Burn API (2 hours)

**Goal:** Understand Burn 0.20 APIs for:
- Parameter iteration (`module.parameters()` or equivalent)
- Gradient access (how to get gradients after `.backward()`)
- Parameter modification (can we do `param += delta`?)
- Loss computation patterns in Burn

**Resources:**
- Burn 0.20 documentation
- Burn examples in burn repo
- burn::module API docs
- burn::tensor API docs

### Step 2: Implement Helper Functions (1 hour)

**Order:**
1. `tensor_to_vec()` - Simplest, just data conversion
2. `vec_to_tensor()` - Also straightforward
3. `extract_gradients()` - Requires parameter walking
4. `apply_deltas_to_model()` - Most complex, needs parameter modification

**Test each function independently**

### Step 3: Implement Model Trait (2 hours)

**Order:**
1. `parameter_count()` - Use helper, straightforward
2. `forward()` - Need to define loss computation strategy
3. `backward()` - Trigger autodiff, extract gradients
4. `apply_weight_delta()` - Use helper function

**Test with SimpleMLP example**

### Step 4: Implement Optimizer Trait (1 hour)

**Order:**
1. `learning_rate()` / `set_learning_rate()` - Simple getters/setters
2. `zero_grad()` - May be no-op for Burn
3. `step()` - Call Burn optimizer

**Test with Adam optimizer**

### Step 5: Update Example & Tests (1 hour)

- Uncomment training loop in `examples/burn_mlp_mnist.rs`
- Enable ignored tests in `tests/burn_integration_test.rs`
- Add assertions for loss convergence
- Verify all 4 training phases execute

---

## Total Estimated Effort

| Phase | Description | Time | Status |
|-------|-------------|------|--------|
| Phase 1 | Module structure & skeleton | 2h | ✅ Complete |
| Phase 2 | Model trait implementation | 3h | 🔴 Next |
| Phase 3 | Optimizer trait implementation | 1h | 🔴 After P2 |
| Phase 4 | Helper functions | 1h | 🟡 Partial |
| Phase 5 | Testing & validation | 1h | 🔴 Final |
| **Total** | | **8h** | **12% done** |

**Current Progress:** 2h / 8h = 25% complete

---

## VRAM Budget Considerations

**IMPORTANT:** RTX 5080 has 16GB total, but ~2-2.5GB is consumed by OS + desktop environment.

**Safe Available VRAM:** ~13.5-14GB

### Model VRAM Requirements

| Model | Precision | Batch Size | Expected VRAM | Safe? |
|-------|-----------|------------|---------------|-------|
| SimpleMLP | FP32 | 64 | 0.7 GB | ✅ Yes |
| SmolLM2-135M | FP16 | 16 | 4.2 GB | ✅ Yes |
| TinyLlama-1.1B | FP16 | 2 | 8.4 GB | ✅ Yes (with checkpointing) |
| LLaMA-3B | FP16 | 1 | ~22 GB | ❌ Need 3090 Ti |

**Detailed VRAM budget:** See `VRAM_BUDGET.md`

## Known Issues & Questions

### Issue 1: Loss Computation Strategy

**Problem:** How should `forward()` compute loss?

**Options:**
1. **Require user to provide loss function** - Most flexible
2. **Infer from model output shape** - Classification vs regression
3. **Let model return loss directly** - Model includes loss computation

**Decision:** TBD - Need to discuss with user or follow Burn conventions

### Issue 2: Parameter Naming

**Problem:** WeightDelta uses parameter names as HashMap keys. How do we map these to Burn's parameter tree?

**Challenges:**
- Burn may use hierarchical paths (e.g., "fc1.weight", "fc1.bias")
- WeightDelta may use flat names
- Need consistent naming convention

**Solution:** TBD - May need to add parameter name mapping layer

### Issue 3: Autodiff Graph Preservation

**Problem:** Does modifying parameters with `param += delta` break the autodiff graph?

**Concerns:**
- Burn tracks computation graph for gradient flow
- Direct parameter modification may disconnect gradients
- May need to use Burn's parameter update APIs

**Testing:** Need to verify with small example that gradients still flow after delta application

---

## Next Session Plan

**When resuming Task 1:**

1. Start with Step 1 (Research Burn API) - 2 hours
2. Read Burn 0.20 docs thoroughly
3. Find examples of:
   - Parameter iteration
   - Gradient extraction
   - Loss computation
   - Parameter modification
4. Take notes on APIs and patterns
5. Update this document with findings
6. Proceed to Step 2 (Helper functions)

**Or if user wants to switch tasks:**
- Task 2 (Optimizer wrapper) can proceed partially in parallel
- Task 12 (Update CLAUDE.md) is independent and can be done anytime
- Task 10 (Release v0.2.0) should wait until integration is more complete

---

## Files Modified/Created

### Modified
- `src/lib.rs` - Added `pub mod burn_integration;`

### Created
- `src/burn_integration.rs` - Main integration module (420 lines)
- `examples/burn_mlp_mnist.rs` - MNIST example (150 lines)
- `tests/burn_integration_test.rs` - Integration tests (120 lines)
- `BURN_INTEGRATION_STATUS.md` - This file

### Total: 690+ lines of new code

---

**Status:** Foundation complete, ready for Phase 2 implementation
**Next:** Research Burn 0.20 API and implement Model trait
**Blockers:** None - all dependencies available
**Risk:** Medium - Burn API familiarity needed
