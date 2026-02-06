# Next Phase: LoRA & QLoRA Implementation & Validation

## Current Status
 GPU training infrastructure complete with 5-tier test suite  
 Adapter backward pass integration fixed  
 Forward pass with adapters implemented  
 Test automation and documentation ready  
 **NEXT:** Implement and validate LoRA training, then QLoRA

---

## Phase 1: LoRA Training Validation (This Phase)

### 1.1 Test Loss Extraction & Convergence Validation
**Purpose:** Validate that loss values can be extracted from trainer for convergence checks

**Tasks:**
- Modify `Trainer::training_step()` to return both loss value and metrics
- Create `TrainingMetrics` struct to track: loss, gradient norms, parameter updates
- Add loss recording to each test tier
- Implement convergence assertions in gpu_utils.rs
- Validate 30% loss decrease over convergence test (100 steps)

**Files to Modify:**
- [src/trainer.rs](src/trainer.rs) - Extract metrics from training loop
- [tests/gpu_training.rs](tests/gpu_training.rs) - Add loss collection and assertions
- [tests/gpu_utils.rs](tests/gpu_utils.rs) - Enhance convergence validation

**Success Criteria:**
-  Loss values collected per step
-  Convergence assertions pass (≥30% decrease)
-  Gradient norms tracked
-  Parameter updates verified

---

### 1.2 LoRA Gradient Flow Verification
**Purpose:** Validate that gradients flow through LoRA weights correctly

**Tasks:**
- Add test to capture initial LoRA weights before/after training step
- Verify LoRA A matrix changes after backward pass
- Verify LoRA B matrix changes after backward pass
- Verify base model weights remain unchanged (frozen)
- Check gradient magnitudes are reasonable (not NaN, not zero)

**Files to Create/Modify:**
- [tests/gpu_training.rs](tests/gpu_training.rs) - Add gradient flow test
- [src/trainer.rs](src/trainer.rs) - Expose LoRA weights for verification

**Success Criteria:**
-  LoRA A weights update after training step
-  LoRA B weights update after training step
-  Base model weights frozen
-  Gradients are finite and non-zero

---

### 1.3 LoRA Checkpoint Save/Load
**Purpose:** Validate adapter weights persist correctly

**Tasks:**
- Implement `save_adapter_weights()` properly in LoadedModel
- Test checkpoint save after N steps
- Load checkpoint and resume training
- Verify loss continues decreasing after resume
- Test adapter_config.json generation (HF compatible)

**Files to Modify:**
- [src/model.rs](src/model.rs) - Complete save_adapter_weights implementation
- [src/trainer.rs](src/trainer.rs) - Checkpoint loading
- [tests/gpu_training.rs](tests/gpu_training.rs) - Add checkpoint test

**Success Criteria:**
-  Checkpoints save without errors
-  Can resume from checkpoint
-  Loss continues decreasing
-  adapter_config.json is valid JSON
-  Safetensors format is correct

---

### 1.4 LoRA with Different Target Modules
**Purpose:** Test LoRA on various projection layers

**Tasks:**
- Test with q_proj only (minimal)
- Test with q_proj + v_proj (standard)
- Test with q_proj + k_proj + v_proj + o_proj (full attention)
- Test with gate_proj + up_proj + down_proj (full MLP)
- Validate different ranks: r=4, r=8, r=16, r=32

**Files to Create:**
- examples/configs/lora_minimal.yaml - q_proj only
- examples/configs/lora_standard.yaml - q_proj + v_proj
- examples/configs/lora_full_attention.yaml - all attention projs
- examples/configs/lora_mlp.yaml - MLP projections

**Success Criteria:**
-  All target module combinations work
-  Loss decreases for each configuration
-  Memory usage scales with rank

---

## Phase 2: QLoRA Implementation & Validation

### 2.1 QLoRA Backward Pass Integration
**Purpose:** Ensure 4-bit quantized base + full-precision LoRA training works

**Tasks:**
- Verify QuantizedLinear properly propagates gradients
- Test backward pass with 4-bit quantization
- Validate quantization doesn't break gradients
- Test with double quantization enabled
- Verify straight-through estimator (STE) works

**Files to Modify:**
- [tests/gpu_training.rs](tests/gpu_training.rs) - Add QLoRA gradient tests
- [qlora-rs/src/qlora.rs](https://github.com/tzervas/qlora-rs) - Verify backward pass

**Success Criteria:**
-  QLoRA gradients flow correctly
-  Loss decreases with quantization
-  No NaN or Inf gradients
-  Training is numerically stable

---

### 2.2 QLoRA Memory Profiling & Optimization
**Purpose:** Validate memory efficiency of 4-bit quantization

**Tasks:**
- Measure peak VRAM for each model tier with QLoRA
- Compare QLoRA vs LoRA memory usage
- Implement memory profiling in tests
- Validate fits within expected VRAM bounds:
  - SmolLM2-135M: < 512 MB
  - TinyLlama-1.1B: < 2 GB
  - LLaMA-7B: < 8 GB (with gradient checkpointing < 12 GB)
- Test with different quantization configs

**Files to Create:**
- tests/memory_profiling.rs - VRAM measurement utilities

**Files to Modify:**
- [tests/gpu_training.rs](tests/gpu_training.rs) - Add memory tracking

**Success Criteria:**
-  Memory usage within expected bounds
-  4-bit quantization reduces VRAM by 50%+
-  Can fit 7B model in 12GB VRAM
-  Double quantization trades speed for memory

---

### 2.3 QLoRA Training Convergence
**Purpose:** Validate QLoRA reaches same convergence as LoRA

**Tasks:**
- Add QLoRA test tiers to gpu_training.rs
- QLoRA SmolLM2: 100 steps, verify 30%+ loss decrease
- QLoRA TinyLlama: 500 steps, verify 40%+ loss decrease
- Compare QLoRA vs LoRA convergence curves
- Validate training is numerically stable (no NaN)

**Files to Modify:**
- [tests/gpu_training.rs](tests/gpu_training.rs) - Add QLoRA test functions

**Success Criteria:**
-  QLoRA converges similarly to LoRA
-  Loss decrease ≥ 30% for 100 steps
-  Training is stable (no NaN/Inf)
-  No quantization artifacts in convergence

---

### 2.4 QLoRA Checkpoint & Inference
**Purpose:** Validate full QLoRA pipeline from training to inference

**Tasks:**
- Save QLoRA checkpoint (quantized base + LoRA adapters)
- Load checkpoint and verify adapter weights
- Test inference with loaded adapters
- Compare outputs with trained vs untrained model
- Validate adapter composition (merge if needed)

**Files to Modify:**
- [src/model.rs](src/model.rs) - QLoRA checkpoint handling

**Success Criteria:**
-  Checkpoints save correctly
-  Can load and resume training
-  Inference works with trained adapters
-  Quality improvement visible in outputs

---

## Detailed Task Breakdown: LoRA Phase

### Task 1: Extract Trainer Metrics
**Effort:** 2 hours

**Changes Needed:**
```rust
// In Trainer::training_step() - return metrics
struct StepMetrics {
    loss: f64,
    loss_scale: f64,  // for mixed precision
    grad_norm: f64,   // l2 norm of gradients
    param_norm: f64,  // l2 norm of parameters
}

// Modify training loop to collect metrics
pub fn train(&mut self) -> Result<TrainingMetrics> { ... }

// In tests, collect and validate
let mut losses = Vec::new();
for step in 0..100 {
    let metrics = trainer.training_step(batch)?;
    losses.push(metrics.loss);
}
assert_loss_convergence(&losses, 0.3, 25);  // 30% decrease
```

---

### Task 2: Gradient Flow Test
**Effort:** 3 hours

**Implementation:**
```rust
#[test]
fn test_lora_gradient_flow() {
    // 1. Create model with LoRA
    let model = LoadedModel::new(...)?;
    
    // 2. Capture initial LoRA weights
    let initial_weights = model.get_lora_weights();
    
    // 3. Run one training step
    trainer.training_step(batch)?;
    
    // 4. Verify weights changed
    let updated_weights = model.get_lora_weights();
    assert!(weights_changed(&initial_weights, &updated_weights));
    
    // 5. Verify base weights frozen
    let base_weights = model.get_base_weights();
    assert_eq!(initial_base_weights, base_weights);
}
```

---

### Task 3: Checkpoint Save/Load
**Effort:** 4 hours

**Implementation:**
```rust
// Save after 50 steps
trainer.save_checkpoint()?;

// Load and resume
trainer.load_checkpoint("./outputs/checkpoint-50")?;
trainer.train()?;

// Verify checkpoint format
let adapter_config = read_json("./outputs/checkpoint-50/adapter_config.json");
assert_eq!(adapter_config["r"], 8);
assert_eq!(adapter_config["lora_alpha"], 16);
```

---

### Task 4: Target Module Variations
**Effort:** 3 hours

**Create 4 configs:**
1. Minimal (q_proj)
2. Standard (q_proj + v_proj)
3. Full Attention (q_proj + k_proj + v_proj + o_proj)
4. MLP (gate_proj + up_proj + down_proj)

---

## Timeline

| Phase | Tasks | Effort | Time |
|-------|-------|--------|------|
| **1: LoRA** | Metrics extraction | 2h | Today |
| | Gradient flow test | 3h | Today |
| | Checkpoint save/load | 4h | Tomorrow |
| | Target module variations | 3h | Tomorrow |
| **2: QLoRA** | Backward pass test | 3h | Thursday |
| | Memory profiling | 4h | Thursday |
| | Convergence validation | 3h | Friday |
| | Checkpoint & inference | 4h | Friday |

**Total: ~30 hours (4-5 days with local GPU testing)**

---

## Success Metrics

### LoRA Phase
-  All 5 test tiers pass with ≥30% loss decrease
-  Gradient flow verified (weights update, base frozen)
-  Checkpoints save/load correctly
-  All target module combinations work
-  Tests compile and run without CUDA errors

### QLoRA Phase
-  QLoRA gradients flow correctly
-  Memory usage within 50% of LoRA
-  Convergence matches LoRA
-  Training numerically stable
-  Full pipeline works (train → checkpoint → inference)

---

## Implementation Order

1. **Start with Task 1: Metrics Extraction**
   - Most fundamental - needed for all convergence checks
   - Unblocks: Gradient flow test, checkpoint test
   - Est: 2 hours

2. **Task 2: Gradient Flow Test**
   - Validates core gradient flow mechanism
   - Unblocks: QLoRA backward pass work
   - Est: 3 hours

3. **Task 3: Checkpoint Save/Load**
   - Ensures training is resumable
   - Required before longer runs
   - Est: 4 hours

4. **Task 4: Target Module Variations**
   - Flexibility for different use cases
   - Less critical but good coverage
   - Est: 3 hours

5. **QLoRA Tasks 2.1-2.4**
   - Start once LoRA phase passing
   - Build on LoRA infrastructure
   - Est: ~14 hours

---

## Risks & Mitigation

| Risk | Mitigation |
|------|-----------|
| Gradient flow broken | Add verbose logging, check Var registration |
| Memory spike on GPU | Profile with nvidia-smi, reduce batch size |
| Checkpoint format incompatible | Test with HF transformers library |
| QLoRA numerical instability | Use straight-through estimator correctly |

---

## References

- Current: [GPU_IMPLEMENTATION_COMPLETE.md](GPU_IMPLEMENTATION_COMPLETE.md)
- LoRA paper: https://arxiv.org/abs/2106.09685
- QLoRA paper: https://arxiv.org/abs/2305.14314
- Peft-rs: [peft-rs/src/](https://github.com/tzervas/peft-rs)
- QLoRA-rs: [qlora-rs/src/](https://github.com/tzervas/qlora-rs)

---

## Next Action

Start Task 1: Metrics Extraction
- Modify `Trainer::training_step()` to return StepMetrics
- Collect loss values in tests
- Add loss convergence assertions
- Test with quick iteration test (10 steps)

**Estimate:** 2 hours to complete
