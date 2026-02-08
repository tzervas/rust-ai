# Phase 2 Implementation TODOs

**Status**: Framework complete, integration pending
**Last Updated**: 2026-02-07

---

## High Priority: HybridTrainer Integration

### Mixed Precision Integration
**File**: `src/lib.rs` (HybridTrainer::step)
**Effort**: 1-2 days
**Priority**: HIGH

- [ ] Add precision casting in `step()` method based on current phase
  ```rust
  fn step(&mut self) -> HybridResult<f32> {
      let precision = self.config.mixed_precision_config.precision_for_phase(self.current_phase);
      // TODO: Cast model/tensors to target precision
      match precision {
          Precision::Fp32 => { /* ensure FP32 */ },
          Precision::Fp16 => { /* cast to FP16 */ },
          Precision::Bf16 => { /* cast to BF16 */ },
      }
  }
  ```
- [ ] Integrate with Burn's `Device::to_dtype()` for tensor conversion
- [ ] Add precision statistics to `MetricsCollector`
  - Current precision per phase
  - Memory savings estimate
  - Precision switches count
- [ ] Test on GPU (CUDA backend) to validate actual savings
- [ ] Benchmark precision switching overhead

**Blocked by**: Burn backend dtype conversion API understanding

---

### Gradient Accumulation Integration
**File**: `src/lib.rs` (HybridTrainer)
**Effort**: 1-2 days
**Priority**: HIGH

- [ ] Add `GradientAccumulationState` field to `HybridTrainer`
  ```rust
  pub struct HybridTrainer<M, O> {
      // ... existing fields ...
      gradient_accumulation_state: GradientAccumulationState,
  }
  ```
- [ ] Track accumulation in `step()` method
  ```rust
  self.gradient_accumulation_state.accumulate_step();
  if self.gradient_accumulation_state.should_update_weights() {
      // Apply optimizer update
      self.optimizer.step()?;
      self.gradient_accumulation_state.reset();
  } else {
      // Skip optimizer update, just accumulate gradients
  }
  ```
- [ ] Implement learning rate scaling
  - Apply scaled LR when creating optimizer
  - OR normalize gradients before update
- [ ] Update metrics to show:
  - Effective batch size
  - Accumulation progress
  - Micro-batch count
- [ ] Test on real models to validate zero quality impact

**Blocked by**: None (ready to implement)

---

### Predict-Aware Memory Integration
**File**: `src/lib.rs`, `src/predict_aware_memory.rs`
**Effort**: 2-3 days
**Priority**: MEDIUM-HIGH

- [ ] Add `PredictAwareMemoryState` field to `HybridTrainer`
- [ ] Implement CPU offload for optimizer state
  ```rust
  // In enter_predict_phase()
  fn offload_optimizer_to_cpu(&mut self) {
      // TODO: Copy optimizer state tensors to CPU
      // Requires Burn optimizer trait introspection
  }
  ```
- [ ] Implement async restore with CUDA streams
  ```rust
  fn async_restore_optimizer(&mut self) {
      // TODO: Start async transfer from CPU to GPU
      // Use CUDA streams for overlap with computation
  }
  ```
- [ ] Integrate with Burn's optimizer trait for state access
- [ ] Add activation discarding flag during Predict
- [ ] Track offload statistics:
  - Offload time
  - Restore time
  - Memory freed
  - Async restore efficiency
- [ ] Benchmark overhead vs memory savings tradeoff

**Blocked by**:
- Burn optimizer state introspection API
- CUDA stream support in Burn/CubeCL

---

## Medium Priority: Checkpoint State Extraction

### DynamicsState Serialization
**File**: `src/checkpoint.rs`, `src/dynamics.rs`
**Effort**: 1 day
**Priority**: MEDIUM

- [ ] Implement `extract_from_rssmlite()` method
  ```rust
  impl DynamicsState {
      pub fn extract_from_rssmlite(rssm: &RSSMLite) -> Self {
          // TODO: Serialize GRU weights
          // TODO: Serialize ensemble states
          // TODO: Extract confidence history
      }
  }
  ```
- [ ] Implement `restore_to_rssmlite()` method
- [ ] Add version field for compatibility checking
- [ ] Test round-trip serialization

**Blocked by**: None (ready to implement)

---

### ResidualStoreState Serialization
**File**: `src/checkpoint.rs`, `src/residuals.rs`
**Effort**: 0.5 days
**Priority**: MEDIUM

- [ ] Implement `extract_from_residual_store()` method
- [ ] Implement `restore_to_residual_store()` method
- [ ] Test serialization with compressed residuals

**Blocked by**: None (ready to implement)

---

### PhaseControllerState Serialization
**File**: `src/checkpoint.rs`, `src/phases.rs`
**Effort**: 0.5 days
**Priority**: MEDIUM

- [ ] Implement `extract_from_phase_controller()` method
- [ ] Implement `restore_to_phase_controller()` method
- [ ] Serialize phase transition statistics

**Blocked by**: None (ready to implement)

---

### DivergenceMonitorState Serialization
**File**: `src/checkpoint.rs`, `src/divergence.rs`
**Effort**: 0.5 days
**Priority**: MEDIUM

- [ ] Implement `extract_from_divergence_monitor()` method
- [ ] Implement `restore_to_divergence_monitor()` method
- [ ] Serialize EMA statistics

**Blocked by**: None (ready to implement)

---

### CorrectorState Serialization
**File**: `src/checkpoint.rs`, `src/corrector.rs`
**Effort**: 0.5 days
**Priority**: MEDIUM

- [ ] Implement `extract_from_residual_corrector()` method
- [ ] Implement `restore_to_residual_corrector()` method
- [ ] Serialize online correction model weights

**Blocked by**: None (ready to implement)

---

## Low Priority: Future Enhancements

### Auto-Tuning Integration
**File**: `src/lib.rs`
**Effort**: 1 day
**Priority**: LOW

- [ ] Enable auto-tuning fields in HybridTrainerConfig
  ```rust
  // Currently commented out at line 592
  // TODO: Enable when auto_tuning fields are added to struct
  ```
- [ ] Wire weight updates to external access
  ```rust
  // Line 766: TODO: wire to optimizer when available
  ```

**Blocked by**: None (design decision needed)

---

### Compression Support
**File**: `src/residuals.rs`
**Effort**: 2 days
**Priority**: LOW

- [ ] Add compression support for residuals
  ```rust
  // Line 899: TODO: Add compression support
  compressed: Some(CompressedResidual { ... })
  ```
- [ ] Implement codec (bincode or custom)
- [ ] Benchmark compression ratio vs overhead

**Blocked by**: None (optional optimization)

---

### Burn Integration Helpers
**File**: `src/burn_integration.rs`
**Effort**: 1 day
**Priority**: LOW

- [ ] Implement parameter counting via Burn module introspection
  ```rust
  // Line 267: TODO: Implement parameter counting
  fn count_parameters(&self) -> usize { ... }
  ```
- [ ] Implement gradient norm computation via Burn grad introspection
  ```rust
  // Line 330: TODO: Implement gradient norm computation
  fn compute_gradient_norm(model: &M::InnerModule) -> f32 { ... }
  ```

**Blocked by**: Burn module introspection API understanding

---

### VRAM Budget Fields
**File**: `src/vram_budget.rs`, `src/config.rs`
**Effort**: 0.5 days
**Priority**: LOW

- [ ] Add VRAM budget fields to HybridTrainerConfig
  ```rust
  // Line 267 in vram_budget.rs
  // TODO: Add these fields to HybridTrainerConfig in Phase 2:
  // - use_mixed_precision
  // - gradient_accumulation_steps
  // - offload_optimizer_to_cpu
  ```

**Blocked by**: None (design decision needed)

---

## Documentation TODOs

### Mixed Precision Guide
**File**: `docs/MIXED_PRECISION_GUIDE.md`
**Effort**: 0.5 days
**Priority**: MEDIUM

- [ ] Complete integration examples with actual Burn code
- [ ] Add Burn backend validation section
- [ ] Benchmark on real models (1B, 7B params)
- [ ] Add gradient scaling for FP16 stability
- [ ] Add precision statistics to metrics

---

### Memory Optimization Guide
**File**: `docs/MEMORY_OPTIMIZATION_COMPLETE.md`
**Effort**: 0.5 days
**Priority**: MEDIUM

- [ ] Update with real GPU benchmark results
- [ ] Add integration completion checklist
- [ ] Add troubleshooting section based on validation

---

## Testing TODOs

### Integration Tests
**File**: `tests/memory_optimizations_integration.rs` (NEW)
**Effort**: 1 day
**Priority**: HIGH

- [ ] Create end-to-end test with all optimizations enabled
- [ ] Test mixed precision switching during training
- [ ] Test gradient accumulation with micro-batches
- [ ] Test optimizer offload/restore cycle
- [ ] Validate quality preservation (<1% loss)

---

### GPU Tests
**File**: `tests/gpu_memory_tests.rs` (NEW)
**Effort**: 1 day
**Priority**: HIGH

- [ ] Benchmark VRAM usage with nvidia-smi
- [ ] Validate savings match estimates
- [ ] Test on RTX 5080 (16GB VRAM)
- [ ] Test on different model sizes (1B, 7B)

---

## Phase 2B Validation TODOs

### GPT-2 Small (124M params)
**Priority**: CRITICAL
**Effort**: 2 days

- [ ] Implement GPT-2 architecture in Burn
- [ ] Train baseline (vanilla FP32)
- [ ] Train with all optimizations
- [ ] Compare:
  - VRAM usage (nvidia-smi)
  - Wall-clock time
  - Final perplexity
  - Training stability
- [ ] Document results

---

### GPT-2 Medium (350M params)
**Priority**: HIGH
**Effort**: 2 days

- [ ] Same as GPT-2 Small but with larger model
- [ ] Stress-test memory optimizations
- [ ] Validate no OOM on RTX 5080

---

### 1B Parameter Model
**Priority**: HIGH
**Effort**: 3 days

- [ ] Define architecture (GPT-2 XL or custom)
- [ ] Validate fits in 16GB VRAM with optimizations
- [ ] Benchmark end-to-end training
- [ ] Document as primary use case

---

## Summary

**Total Estimated Effort**: 15-20 days

**Critical Path**:
1. Mixed precision integration (1-2 days)
2. Gradient accumulation integration (1-2 days)
3. Predict-aware memory integration (2-3 days)
4. Integration tests (1 day)
5. GPU validation (1 day)
6. GPT-2 Small validation (2 days)
7. 1B model validation (3 days)

**Parallel Work**:
- Checkpoint state extraction (can be done anytime)
- Documentation updates (ongoing)
- Low priority enhancements (future)

---

**Status**: Phase 2A complete (framework), Phase 2B starting (validation)
**Next**: Validate on GPT-2 Small (124M params)
