# Burn Integration Implementation Plan

**Plan Date:** 2026-02-06
**Manager Agent:** Plan agent (software architect)
**Scope:** Phases 2-5 of Burn integration (Task 1 completion)
**Total Estimated Time:** ~6.25 hours (5.75 hours with parallelism)

---

## Executive Summary

This plan details the systematic implementation of Burn 0.20 framework integration into hybrid-predict-trainer-rs. Phase 1 (foundation/skeleton) is complete. This plan covers Phases 2-5: implementation, testing, and polish.

**Critical Architectural Decisions:**

1. **Ownership Model:** BurnModelWrapper uses `Option<M>` internally to support Burn's ownership-based optimizer pattern
2. **Loss Computation:** Introduce `BurnForwardFn` trait for user-defined loss functions
3. **Gradient Storage:** Store `B::Gradients` after backward() for optimizer consumption
4. **Parameter Naming:** Use `ParamId::to_string()` for consistent WeightDelta key mapping
5. **Backend Features:** Consolidate to single `burn` crate with feature flags (ndarray, cuda, wgpu)

---

## Phase Overview

| Phase | Description | Worker Tasks | Duration | Parallelizable |
|-------|-------------|--------------|----------|----------------|
| Phase 2 | Core Implementation | W2A, W2B, W2C | 2.5h | Partially |
| Phase 3 | Optimizer Implementation | W3A | 0.75h | No |
| Phase 4 | Testing & Validation | W4A, W4B, W4C | 2h | Yes |
| Phase 5 | Documentation & Polish | W5A | 1h | No |

---

## Worker Tasks Breakdown

### W2A: Cargo.toml Updates (15 minutes)

**Goal:** Consolidate Burn dependencies and add feature flags

**Actions:**
1. Replace `burn-core`, `burn-tensor`, etc. with single `burn` crate
2. Add feature flags: `ndarray` (default), `cuda`, `wgpu`
3. Update version to `0.20` (consistent with research)
4. Add `ndarray` as optional dependency for NdArray backend

**Deliverable:**
```toml
[dependencies]
burn = { version = "0.20", features = ["autodiff", "ndarray"] }
ndarray = { version = "0.16", optional = true }

[features]
default = ["ndarray"]
cuda = ["burn/cuda"]
wgpu = ["burn/wgpu"]
```

**Quality Gate G1:** `cargo check` passes, `cargo check --features cuda` passes (if CUDA available)

---

### W2B: Helper Functions + BurnForwardFn Trait (45 minutes)

**Goal:** Implement tensor conversion utilities and user-facing loss function trait

**Actions:**

1. **Implement `tensor_to_vec<B, const D: usize>()`**
   ```rust
   fn tensor_to_vec<B: Backend, const D: usize>(tensor: &Tensor<B, D>) -> Vec<f32> {
       tensor.to_data().to_vec::<f32>()
   }
   ```

2. **Implement `vec_to_tensor<B, const D: usize>()`**
   ```rust
   fn vec_to_tensor<B: Backend, const D: usize>(
       vec: Vec<f32>,
       shape: Shape<D>,
       device: &Device<B>
   ) -> Tensor<B, D> {
       Tensor::from_data(
           TensorData::from_vec(vec, shape),
           device
       )
   }
   ```

3. **Define `BurnForwardFn` trait**
   ```rust
   pub trait BurnForwardFn<B, M, T>: Send + Sync
   where
       B: AutodiffBackend,
       M: AutodiffModule<B>,
   {
       /// Computes forward pass and loss
       /// Takes ownership of model, returns (model, loss_tensor)
       fn forward(&self, model: M, batch: &BurnBatch<B, T>) -> (M, Tensor<B, 1>);
   }
   ```

4. **Implement `extract_gradients<B, M>()`**
   ```rust
   fn extract_gradients<B, M>(
       model: &M,
       gradients: &B::Gradients,
   ) -> HashMap<String, Vec<f32>>
   where
       B: AutodiffBackend,
       M: AutodiffModule<B>,
   {
       struct GradVisitor<B: AutodiffBackend> {
           gradients: B::Gradients,
           per_param: HashMap<String, Vec<f32>>,
       }

       impl<B: AutodiffBackend> ModuleVisitor<B> for GradVisitor<B> {
           fn visit<const D: usize>(&mut self, id: &ParamId, tensor: &Tensor<B, D>) {
               let grad_tensor = tensor.grad(&self.gradients);
               let grad_flat = tensor_to_vec(&grad_tensor.reshape([/* total elements */]));
               self.per_param.insert(id.to_string(), grad_flat);
           }
       }

       let mut visitor = GradVisitor {
           gradients: gradients.clone(),
           per_param: HashMap::new(),
       };
       model.visit(&mut visitor);
       visitor.per_param
   }
   ```

5. **Implement `apply_deltas_to_model<B, M>()`**
   ```rust
   fn apply_deltas_to_model<B, M>(
       model: M,
       delta: &WeightDelta,
       device: &Device<B>,
   ) -> M
   where
       B: Backend,
       M: Module<B>,
   {
       struct DeltaMapper<'a, B: Backend> {
           deltas: &'a HashMap<String, Vec<f32>>,
           scale: f32,
           device: Device<B>,
       }

       impl<'a, B: Backend> ModuleMapper<B> for DeltaMapper<'a, B> {
           fn map<const D: usize>(&mut self, id: &ParamId, tensor: Tensor<B, D>) -> Tensor<B, D> {
               let param_name = id.to_string();
               if let Some(delta_vec) = self.deltas.get(&param_name) {
                   let delta_tensor = vec_to_tensor(
                       delta_vec.clone(),
                       tensor.shape(),
                       &self.device
                   );
                   tensor.add(delta_tensor.mul_scalar(self.scale))
               } else {
                   tensor
               }
           }
       }

       let mut mapper = DeltaMapper {
           deltas: &delta.deltas,
           scale: delta.scale,
           device: device.clone(),
       };

       model.map(&mut mapper)
   }
   ```

**Quality Gate G2:** Unit tests for helper functions pass (see W4A)

---

### W2C: BurnModelWrapper Model Trait Implementation (90 minutes)

**Goal:** Implement `Model<BurnBatch<B, T>>` trait for `BurnModelWrapper`

**Actions:**

1. **Update BurnModelWrapper struct**
   ```rust
   pub struct BurnModelWrapper<B, M, T, F>
   where
       B: AutodiffBackend,
       M: AutodiffModule<B>,
       F: BurnForwardFn<B, M, T>,
   {
       model: Arc<RwLock<Option<M>>>,  // Option for ownership dance
       forward_fn: Arc<F>,              // User-provided loss function
       device: Device<B>,
       last_loss: Arc<RwLock<Option<Tensor<B, 1>>>>,
       last_gradients: Arc<RwLock<Option<B::Gradients>>>,
       param_metadata: Arc<RwLock<HashMap<String, Vec<usize>>>>,
       _phantom: PhantomData<T>,
   }
   ```

2. **Implement `forward()`**
   ```rust
   fn forward(&mut self, batch: &BurnBatch<B, T>) -> HybridResult<f32> {
       // 1. Take model from Option (ownership transfer)
       let mut model_lock = self.model.write();
       let model = model_lock.take()
           .ok_or_else(|| /* error: model already taken */)?;

       // 2. Call user forward function
       let (model_returned, loss_tensor) = self.forward_fn.forward(model, batch);

       // 3. Extract scalar loss value
       let loss_data = loss_tensor.to_data();
       let loss_scalar = loss_data.to_vec::<f32>()
           .get(0)
           .copied()
           .ok_or_else(|| /* error: empty loss tensor */)?;

       // 4. Store loss tensor for backward (must keep autodiff graph)
       *self.last_loss.write() = Some(loss_tensor);

       // 5. Put model back
       *model_lock = Some(model_returned);

       Ok(loss_scalar)
   }
   ```

3. **Implement `backward()`**
   ```rust
   fn backward(&mut self) -> HybridResult<GradientInfo> {
       // 1. Take loss tensor
       let loss_tensor = self.last_loss.write().take()
           .ok_or_else(|| /* error: no loss to backward */)?;

       // 2. Call backward to get gradients
       let gradients: B::Gradients = loss_tensor.backward();

       // 3. Extract per-parameter gradients
       let model_lock = self.model.read();
       let model = model_lock.as_ref()
           .ok_or_else(|| /* error: model missing */)?;

       let per_param_grads = extract_gradients(model, &gradients);

       // 4. Compute gradient norm
       let grad_norm = per_param_grads.values()
           .flat_map(|v| v.iter())
           .map(|x| x * x)
           .sum::<f32>()
           .sqrt();

       // 5. Compute per-parameter norms
       let per_param_norms: HashMap<String, f32> = per_param_grads.iter()
           .map(|(name, vec)| {
               let norm = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
               (name.clone(), norm)
           })
           .collect();

       // 6. Store gradients for optimizer
       *self.last_gradients.write() = Some(gradients);

       // 7. Return GradientInfo (loss was already computed in forward)
       Ok(GradientInfo {
           loss: 0.0, // Placeholder, will be filled by caller
           gradient_norm: grad_norm,
           per_param_norms,
       })
   }
   ```

4. **Implement `parameter_count()`**
   ```rust
   fn parameter_count(&self) -> usize {
       struct ParamCounter {
           count: usize,
       }

       impl<B: Backend> ModuleVisitor<B> for ParamCounter {
           fn visit<const D: usize>(&mut self, _id: &ParamId, tensor: &Tensor<B, D>) {
               self.count += tensor.dims().iter().product::<usize>();
           }
       }

       let model = self.model.read();
       if let Some(ref m) = *model {
           let mut counter = ParamCounter { count: 0 };
           m.visit(&mut counter);
           counter.count
       } else {
           0
       }
   }
   ```

5. **Implement `apply_weight_delta()`**
   ```rust
   fn apply_weight_delta(&mut self, delta: &WeightDelta) -> HybridResult<()> {
       // 1. Take model from Option
       let mut model_lock = self.model.write();
       let model = model_lock.take()
           .ok_or_else(|| /* error: model already taken */)?;

       // 2. Apply deltas using helper function
       let updated_model = apply_deltas_to_model(model, delta, &self.device);

       // 3. Put model back
       *model_lock = Some(updated_model);

       Ok(())
   }
   ```

**Quality Gate G3:** Model trait methods compile, basic forward/backward test passes

---

### W3A: BurnOptimizerWrapper Optimizer Trait Implementation (45 minutes)

**Goal:** Implement `Optimizer<BurnModelWrapper, BurnBatch>` trait for `BurnOptimizerWrapper`

**Actions:**

1. **Update BurnOptimizerWrapper struct**
   ```rust
   pub struct BurnOptimizerWrapper<B, M, O, T, F>
   where
       B: AutodiffBackend,
       M: AutodiffModule<B>,
       O: burn::optim::Optimizer<M, B>,
       F: BurnForwardFn<B, M, T>,
   {
       optimizer: Arc<RwLock<O>>,
       learning_rate: Arc<RwLock<f32>>,
       _phantom: PhantomData<(B, M, T, F)>,
   }
   ```

2. **Implement `step()`**
   ```rust
   fn step(
       &mut self,
       model: &mut BurnModelWrapper<B, M, T, F>,
       _gradients: &GradientInfo,
   ) -> HybridResult<()> {
       // 1. Get learning rate
       let lr = *self.learning_rate.read();

       // 2. Take model and gradients from wrapper
       let mut model_lock = model.model.write();
       let model_inner = model_lock.take()
           .ok_or_else(|| /* error: model missing */)?;

       let gradients = model.last_gradients.write().take()
           .ok_or_else(|| /* error: gradients missing */)?;

       // 3. Convert to GradientsParams
       let grads_params = GradientsParams::from_grads(gradients, &model_inner);

       // 4. Call optimizer.step (consumes model, returns updated model)
       let updated_model = self.optimizer.write().step(lr, model_inner, grads_params);

       // 5. Put model back
       *model_lock = Some(updated_model);

       Ok(())
   }
   ```

3. **Implement `learning_rate()`**
   ```rust
   fn learning_rate(&self) -> f32 {
       *self.learning_rate.read()
   }
   ```

4. **Implement `set_learning_rate()`**
   ```rust
   fn set_learning_rate(&mut self, lr: f32) {
       *self.learning_rate.write() = lr;
   }
   ```

5. **Implement `zero_grad()`**
   ```rust
   fn zero_grad(&mut self) {
       // Burn handles gradient zeroing via autodiff graph reset
       // No explicit action needed
       // Gradients are regenerated on each backward() call
   }
   ```

**Quality Gate G4:** Optimizer trait compiles, basic optimizer step test passes

---

### W4A: Unit Tests for Helper Functions (30 minutes)

**Goal:** Comprehensive tests for tensor conversion and utility functions

**Test Coverage:**

1. **test_tensor_to_vec_1d()**
   - Create 1D tensor, convert to vec, verify values

2. **test_tensor_to_vec_2d()**
   - Create 2D tensor, convert to vec (flattened), verify values

3. **test_vec_to_tensor_1d()**
   - Create vec, convert to 1D tensor, verify shape and values

4. **test_vec_to_tensor_2d()**
   - Create vec, convert to 2D tensor with shape, verify

5. **test_extract_gradients()**
   - Create simple module, run forward+backward, extract gradients
   - Verify gradient keys match parameter names
   - Verify gradient shapes are flattened correctly

6. **test_apply_deltas_to_model()**
   - Create model, create WeightDelta with known values
   - Apply deltas, verify parameters updated correctly
   - Test with scale != 1.0

7. **test_parameter_naming_consistency()**
   - Verify ParamId.to_string() produces expected hierarchical names
   - Test with nested modules

**Implementation Location:** `tests/burn_integration_test.rs`

**Quality Gate G5:** All helper function tests pass

---

### W4B: Integration Tests with SimpleMLP (60 minutes)

**Goal:** End-to-end training scenario with actual model

**Test Coverage:**

1. **test_simple_mlp_forward_backward()**
   - Create SimpleMLP (784 → 128 → 10)
   - Wrap in BurnModelWrapper
   - Run forward pass with dummy batch
   - Run backward pass
   - Verify gradients extracted

2. **test_simple_mlp_optimizer_step()**
   - Create model + optimizer wrapper
   - Run forward → backward → optimizer.step()
   - Verify parameters changed
   - Verify loss decreased (on simple dataset)

3. **test_simple_mlp_weight_delta_application()**
   - Create model
   - Capture initial parameters
   - Apply known weight delta
   - Verify parameters updated by exact delta amount

4. **test_simple_mlp_convergence_on_xor()**
   - Train SimpleMLP on XOR problem (small synthetic dataset)
   - Run 100 steps
   - Verify loss decreases below threshold (e.g., < 0.1)
   - Verify predictions match expected outputs

5. **test_simple_mlp_device_handling()**
   - Test model creation on CPU
   - If CUDA available, test on GPU
   - Verify tensors on correct device

**Implementation Location:** `tests/burn_integration_test.rs` or `examples/burn_mlp_mnist.rs`

**Quality Gate G6:** Integration tests pass, convergence demonstrated

---

### W4C: End-to-End HybridTrainer Test (30 minutes)

**Goal:** Verify BurnModelWrapper works with full HybridTrainer pipeline

**Test Coverage:**

1. **test_hybrid_trainer_with_burn_model()**
   - Create SimpleMLP wrapped in BurnModelWrapper
   - Create BurnOptimizerWrapper with Adam
   - Create HybridTrainerConfig with short phases (warmup=10, full=5, etc.)
   - Create HybridTrainer
   - Run 50 steps through all 4 phases
   - Verify:
     - Phase transitions occur (Warmup → Full → Predict → Correct)
     - Loss tracked correctly
     - No panics or errors
     - Metrics collected

2. **test_burn_model_predict_phase()**
   - Run HybridTrainer to Predict phase
   - Verify weight deltas applied
   - Verify predictions executed without backward pass
   - Verify divergence monitoring works

3. **test_burn_model_correct_phase()**
   - Run HybridTrainer through Predict → Correct
   - Verify residual corrections computed
   - Verify corrected predictions applied

**Implementation Location:** `tests/burn_integration_test.rs`

**Quality Gate G7:** HybridTrainer works end-to-end with Burn models

---

### W5A: Documentation and Polish (60 minutes)

**Goal:** Finalize documentation, examples, and public API

**Actions:**

1. **Update `examples/burn_mlp_mnist.rs`**
   - Uncomment training loop
   - Add comprehensive comments explaining each step
   - Add VRAM monitoring integration
   - Show how to use BurnForwardFn trait

2. **Write `BurnForwardFn` documentation**
   - Add module-level docs explaining the trait
   - Provide 3 examples: classification, regression, custom loss

3. **Write integration guide**
   - Create `docs/BURN_INTEGRATION_GUIDE.md`
   - Step-by-step tutorial for integrating user models
   - Common patterns and pitfalls
   - Device selection guidance

4. **Update `BURN_INTEGRATION_STATUS.md`**
   - Mark Phase 2-5 complete
   - Document final API surface
   - Add performance benchmarks (if available)

5. **Update main `README.md`**
   - Add Burn integration section
   - Link to examples
   - Update feature flags documentation

6. **Run clippy and fix warnings**
   ```bash
   cargo clippy --all-features --tests -- -W clippy::all
   ```

7. **Run rustfmt**
   ```bash
   cargo fmt --all
   ```

8. **Generate and review docs**
   ```bash
   cargo doc --no-deps --open
   ```

**Deliverable:** Polished, documented, production-ready Burn integration

**Quality Gate G8:** Documentation complete, no clippy warnings, examples run successfully

---

## Risk Analysis

### R1: Burn Import Path Changes (Likelihood: Medium, Impact: Low)

**Risk:** Burn 0.20 may have slightly different import paths than documented

**Mitigation:**
- Check actual Burn 0.20 source code on docs.rs
- Test imports incrementally as we add them
- Keep imports localized in burn_integration.rs module

**Contingency:** Update imports based on actual Burn 0.20 API

---

### R2: Ownership Dance Correctness (Likelihood: Medium, Impact: High)

**Risk:** Taking model out of `Option<M>` and putting it back may have subtle bugs

**Mitigation:**
- Write extensive unit tests for ownership patterns
- Test error paths (what if forward() panics?)
- Use RAII guard pattern for automatic cleanup

**Contingency:** Add explicit drop guards or use scopeguard crate

**Example Safety Pattern:**
```rust
struct ModelGuard<'a, M> {
    model: Option<M>,
    lock: &'a RwLock<Option<M>>,
}

impl<'a, M> Drop for ModelGuard<'a, M> {
    fn drop(&mut self) {
        if let Some(m) = self.model.take() {
            *self.lock.write() = Some(m);
        }
    }
}
```

---

### R3: ModuleVisitor Path Names Not Matching WeightDelta Keys (Likelihood: Medium, Impact: Medium)

**Risk:** `ParamId::to_string()` may produce names that don't match our WeightDelta HashMap keys

**Mitigation:**
- Write test that explicitly checks parameter naming
- Document expected naming convention
- Add debug logging for parameter names during weight delta application

**Contingency:** Add parameter name mapping layer if needed

---

### R4: NdArray Backend Test Limitations (Likelihood: Low, Impact: Low)

**Risk:** NdArray backend may have limited operation support compared to CUDA

**Mitigation:**
- Keep tests simple (basic forward/backward, no complex ops)
- Mark advanced tests as `#[cfg(feature = "cuda")]`
- Test on real GPU when available

**Contingency:** Fallback to CUDA-only tests for advanced features

---

### R5: Tensor Reshape in Gradient Extraction (Likelihood: Medium, Impact: Medium)

**Risk:** Flattening multi-dimensional tensors to Vec<f32> may have incorrect shape calculations

**Mitigation:**
- Store original tensor shapes in param_metadata
- Verify total element count matches before/after reshape
- Add shape validation in helper functions

**Contingency:** Use `tensor.dims().iter().product()` for safe element count

---

### R6: Feature Flag Interaction (Likelihood: Low, Impact: Medium)

**Risk:** Different Burn backends (ndarray, cuda, wgpu) may have incompatible types

**Mitigation:**
- Use feature = "ndarray" as default for tests
- Keep backend-specific code isolated
- Test with multiple backends if available

**Contingency:** Use conditional compilation for backend-specific workarounds

---

## Success Metrics

### Functional Requirements

- [x] Phase 1: Module structure created (complete)
- [ ] Phase 2: Model trait fully implemented
- [ ] Phase 3: Optimizer trait fully implemented
- [ ] Phase 4: All tests pass (unit + integration)
- [ ] Phase 5: Documentation complete

### Quality Requirements

- [ ] Zero `cargo clippy` warnings with `--all-features --tests`
- [ ] Zero `cargo fmt` diffs
- [ ] 100% public API documented
- [ ] All integration tests pass on NdArray backend
- [ ] SimpleMLP example runs and converges

### Performance Requirements

- [ ] Forward pass overhead < 5% vs raw Burn
- [ ] Backward pass overhead < 5% vs raw Burn
- [ ] Optimizer step overhead < 3% vs raw Burn
- [ ] Weight delta application < 1ms for 1M parameters

---

## Execution Strategy

### Sequential Dependencies

```
W2A (Cargo.toml)
  ↓
W2B (Helpers + BurnForwardFn)
  ↓
W2C (Model trait) ← Cannot start until W2B complete
  ↓
W3A (Optimizer trait) ← Cannot start until W2C complete
  ↓
W4A (Unit tests) ← Depends on W2B
W4B (Integration tests) ← Depends on W2C, W3A
W4C (E2E tests) ← Depends on W3A
  ↓
W5A (Documentation) ← Final polish after all tests pass
```

### Parallelization Opportunities

- **W4A and W4B can run in parallel** once W2C + W3A complete
- **Multiple test files can be written concurrently** by different agents
- **Documentation (W5A) can start partially during W4** (document completed components)

### Recommended Agent Allocation

1. **Primary Implementation Agent** - Handles W2A → W2C → W3A sequentially
2. **Test Agent 1** - Writes W4A (unit tests) while W3A in progress
3. **Test Agent 2** - Writes W4B (integration tests) after W3A complete
4. **Documentation Agent** - Starts W5A documentation while tests run

---

## Quality Gates Detail

### G1: Cargo Dependencies (After W2A)

**Acceptance Criteria:**
- `cargo check` passes
- `cargo check --features ndarray` passes
- `cargo check --features cuda` passes (if CUDA available)
- All feature combinations compile

**Verification:**
```bash
cargo check --all-features
cargo check --no-default-features
cargo check --features ndarray
cargo check --features cuda
```

---

### G2: Helper Functions Compile (After W2B)

**Acceptance Criteria:**
- All helper functions compile without warnings
- BurnForwardFn trait definition accepted by compiler
- Example BurnForwardFn implementation compiles

**Verification:**
```bash
cargo check --lib
cargo clippy --lib -- -W clippy::all
```

---

### G3: Model Trait Implementation (After W2C)

**Acceptance Criteria:**
- BurnModelWrapper implements Model trait
- All trait methods compile
- Basic forward pass test runs (dummy data)
- Basic backward pass test runs

**Verification:**
```bash
cargo test test_burn_model_wrapper_forward
cargo test test_burn_model_wrapper_backward
```

---

### G4: Optimizer Trait Implementation (After W3A)

**Acceptance Criteria:**
- BurnOptimizerWrapper implements Optimizer trait
- All trait methods compile
- Basic optimizer step test runs
- Learning rate get/set works

**Verification:**
```bash
cargo test test_burn_optimizer_step
cargo test test_burn_optimizer_learning_rate
```

---

### G5: Unit Tests Pass (After W4A)

**Acceptance Criteria:**
- All 7 helper function tests pass
- Test coverage > 80% for burn_integration.rs helpers
- Tests run in < 1 second on CPU

**Verification:**
```bash
cargo test --lib burn_integration::tests
```

---

### G6: Integration Tests Pass (After W4B)

**Acceptance Criteria:**
- SimpleMLP trains on XOR and converges (loss < 0.1)
- Weight delta application verified with known values
- Gradient extraction matches expected shapes
- Tests complete in < 10 seconds

**Verification:**
```bash
cargo test --test burn_integration_test test_simple_mlp
```

---

### G7: End-to-End Test Passes (After W4C)

**Acceptance Criteria:**
- HybridTrainer runs all 4 phases with Burn model
- No panics or errors
- Phase transitions occur as expected
- Metrics collected correctly

**Verification:**
```bash
cargo test --test burn_integration_test test_hybrid_trainer_with_burn_model
```

---

### G8: Documentation Complete (After W5A)

**Acceptance Criteria:**
- `cargo doc` generates docs without warnings
- All public items have documentation
- Examples run successfully
- README updated with Burn integration section
- Zero clippy warnings

**Verification:**
```bash
cargo doc --no-deps --all-features
cargo clippy --all-features --tests -- -W clippy::all
cargo run --example burn_mlp_mnist
cargo fmt --check
```

---

## Timeline Estimate

| Task | Duration | Start | End | Dependencies |
|------|----------|-------|-----|--------------|
| W2A | 15 min | T+0 | T+0.25h | None |
| W2B | 45 min | T+0.25h | T+1h | W2A |
| W2C | 90 min | T+1h | T+2.5h | W2B |
| W3A | 45 min | T+2.5h | T+3.25h | W2C |
| W4A | 30 min | T+3.25h | T+3.75h | W2B (parallel with W3A) |
| W4B | 60 min | T+3.75h | T+4.75h | W3A, W2C |
| W4C | 30 min | T+4.75h | T+5.25h | W3A |
| W5A | 60 min | T+5.25h | T+6.25h | W4A, W4B, W4C |

**Total Sequential Time:** 6.25 hours

**With Parallelism (W4A || W3A):** 5.75 hours

---

## Rollback Plan

If critical issues encountered at any stage:

1. **W2A Failure** - Revert Cargo.toml, use separate burn-core/burn-tensor crates
2. **W2B Failure** - Simplify BurnForwardFn trait, hardcode loss function initially
3. **W2C Failure** - Fall back to mock Model implementation, defer Burn integration
4. **W3A Failure** - Use manual gradient descent instead of Burn optimizers
5. **W4 Failure** - Mark tests as `#[ignore]` and document known issues
6. **W5A Failure** - Document "experimental" status, defer polish to v0.3.0

**Critical Decision Point:** After W2C completion
- If Model trait implementation too complex, consider alternative architecture
- If ownership dance proves unworkable, explore reference-counted approach

---

## Next Session Checklist

When resuming implementation:

- [x] Read BURN_API_RESEARCH.md for API patterns
- [ ] Start with W2A: Update Cargo.toml
- [ ] Run `cargo check` after each task
- [ ] Commit after each quality gate passes
- [ ] Update BURN_INTEGRATION_STATUS.md with progress
- [ ] Monitor VRAM usage during tests (use check_vram_budget example)
- [ ] Keep implementation plan updated with actual times

---

## Completion Criteria

Task 1 (Burn Integration) is COMPLETE when:

1. All 8 worker tasks (W2A-W5A) finished
2. All 8 quality gates (G1-G8) passed
3. `examples/burn_mlp_mnist.rs` runs and trains successfully
4. Documentation reviewed and approved
5. BURN_INTEGRATION_STATUS.md updated to "✅ Phase 5 Complete"
6. Ready to merge to `dev` branch

---

**Plan Status:** Ready for execution
**Manager Agent:** Available for re-consultation if issues arise
**Evaluator:** Will gate each phase per quality criteria
**Estimated Completion:** 6.25 hours of focused development

**Let's begin with W2A: Cargo.toml Updates!**
