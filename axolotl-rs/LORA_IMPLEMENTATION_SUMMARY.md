# LoRA Per-Layer Injection Implementation Summary

## Date: January 15-16, 2026

## Goal
Fix the Tier 2 training convergence failure where loss INCREASED from 6.29 to 6.84 (-8.6%) instead of expected 30% decrease.

## Root Cause Analysis (Completed)

Identified 3 CRITICAL issues:

1. **Adapter returns only delta**: `forward_with_adapters()` passed `None` to `lora_layer.forward()`, causing it to return only `lora_out` instead of `base + lora_out`

2. **Single adapter applied to logits**: LoRA was applied post-hoc to final logits instead of per-layer Q,K,V,O projection injection

3. **Broken gradient chain**: Frozen base model prevents proper gradient flow

4. **Weak loss signal**: `compute_last_position_loss()` only uses last token

## Implementation Approach

### Option A: Pre-Merge Weights (Static) - NOT CHOSEN
- Merge LoRA weights into base model before construction: `W_merged = W_base + B @ A * scaling`
- Pros: Works with existing candle-transformers, simple
- Cons: Cannot switch adapters at runtime, gradients may not flow correctly

### Option B: Custom LoraLlama (Dynamic) - CHOSEN BUT INCOMPLETE
- Create custom LLaMA model with per-layer LoRA injection
- LoRA applied at each Q,K,V,O projection during forward pass
- Pros: Full control, runtime adapter switching, proper gradient flow
- Cons: More code, needs to track candle-transformers API changes

## What Was Implemented

### 1. Created `lora_llama.rs` Module (Skeleton Only)
- File: [axolotl-rs/src/lora_llama.rs](axolotl-rs/src/lora_llama.rs)
- Status: **TODO - Needs RoPE/Cache Integration**
- Blocked By:
  - Candle's RoPE is not available as `candle_nn::RotaryEmbedding`
  - Must use `cache.cos` and `cache.sin` tensors (see candle-transformers LLaMA)
  - Cache API uses manual `cache.kvs[layer_idx]` management, not `.append()`
  - Forward signature requires `(index_pos: usize, block_idx: usize)`

### 2. Fixed `forward_with_adapters()` 
- File: [axolotl-rs/src/model.rs#L110-L133](axolotl-rs/src/model.rs#L110)
- Removed broken logits-level LoRA application
- Now returns base model output only (documented as temporary)
- **Status**: Training loop can run, but LoRA not integrated

### 3. Updated Model Loading Flow
- File: [axolotl-rs/src/model.rs#L394-L419](axolotl-rs/src/model.rs#L394)
- Restructured to create adapters before model loading
- Added `lora_params` path for future LoraLlama integration
- Currently falls back to standard LLaMA + separate adapters

##  What Still Needs To Be Done

### Priority 1: Complete LoraLlama Implementation
**File**: `axolotl-rs/src/lora_llama.rs`

**Tasks**:
1. Study candle-transformers Llama implementation:
   - Read `~/.cargo/registry/.../candle-transformers-0.9.1/src/models/llama.rs`
   - Note `apply_rotary_emb(&self, x, index_pos, cache)` pattern
   - Note `cache.cos.narrow(0, index_pos, seq_len)` usage
   - Note manual `cache.kvs[block_idx] = Some((k.clone(), v.clone()))` pattern

2. Implement `LoraAttention`:
   - Match signature: `forward(&self, x, index_pos, block_idx, cache)`
   - Apply RoPE using `cache.cos` and `cache.sin`
   - Inject LoRA after Q,K,V,O projections: `lora.forward(&base_q, Some(&base_q))?`
   - Handle error conversion: `.map_err(|e| candle_core::Error::Msg(...))`

3. Implement `LoraMlp`:
   - Inject LoRA after gate/up/down projections
   - Match candle-transformers MLP signature

4. Implement `LoraTransformerBlock`:
   - Combine LoraAttention + LoraMlp
   - Pre-norm pattern (match candle-transformers)

5. Implement `LoraLlama`:
   - `new_with_lora()` creates internal LoRA layers
   - `forward()` passes through all transformer blocks
   - Implement `Module` trait

6. Enable in model loading:
   - File: [axolotl-rs/src/model.rs#L402](axolotl-rs/src/model.rs#L402)
   - Change `let use_lora_model = false` to `true`
   - Remove fallback warning

### Priority 2: Add Gradient Verification
**File**: `axolotl-rs/src/trainer.rs`

**Task**: After `loss.backward()`, verify gradients exist for LoRA A/B:
```rust
use candle_nn::VarMap;

// After backward pass
let grad_store = loss.grad_store()?;
for var in self.model.trainable_params.all_vars() {
    if let Some(grad) = grad_store.get(&var) {
        let grad_norm = grad.sqr()?.sum_all()?.to_scalar::<f32>()?;
        tracing::debug!("Gradient norm for {}: {}", var.name(), grad_norm);
    } else {
        tracing::warn!("No gradient for {}", var.name());
    }
}
```

### Priority 3: Improve Loss Computation
**File**: `axolotl-rs/src/trainer.rs#L380`

**Task**: Change from last-position-only to full-sequence cross-entropy:
```rust
fn compute_full_sequence_loss(logits: &Tensor, targets: &Tensor) -> Result<Tensor> {
    // logits: [batch, seq_len, vocab]
    // targets: [batch, seq_len]
    
    // Flatten to [batch*seq_len, vocab] and [batch*seq_len]
    let (batch, seq_len, vocab) = logits.dims3()?;
    let logits_flat = logits.reshape((batch * seq_len, vocab))?;
    let targets_flat = targets.reshape(batch * seq_len)?;
    
    // Cross-entropy loss
    let loss = candle_nn::ops::cross_entropy(&logits_flat, &targets_flat)?;
    Ok(loss)
}
```

## Testing Plan

### After LoraLlama Complete:
1. Run Tier 1 test (10 steps, sanity check)
2. Check gradient verification output
3. Run Tier 2 test (50 steps, expect 30% loss decrease)
4. Verify LoRA weights change using `capture_lora_weights()`

### Expected Results:
- Tier 1: Completes without errors, gradients present
- Tier 2: Loss decreases from ~6.3 to ~4.4 (30% reduction)
- Gradients: Non-zero norms for all LoRA A/B matrices

## References

### Candle API
- `~/.cargo/registry/.../candle-transformers-0.9.1/src/models/llama.rs` - Reference implementation
- `candle_nn::rotary_emb::rope(x, &cos, &sin)` - How to apply RoPE
- `Cache` struct - Has `cos`, `sin` tensors and `kvs` vec

### LoRA Implementation  
- `peft-rs/src/adapters/lora.rs` - LoraLayer::forward(input, base_output: Option<&Tensor>)
- When `base_output` is `Some`, returns `base.broadcast_add(&lora_out)`
- When `None`, returns only `lora_out`

### Related Files
- [axolotl-rs/src/model.rs](axolotl-rs/src/model.rs) - Model loading, forward_with_adapters
- [axolotl-rs/src/trainer.rs](axolotl-rs/src/trainer.rs) - Training loop, loss computation
- [axolotl-rs/src/lora_llama.rs](axolotl-rs/src/lora_llama.rs) - Custom LoRA model (TODO)
- [peft-rs/src/adapters/lora.rs](peft-rs/src/adapters/lora.rs) - LoRA layer implementation

## Timeline Estimate

- Complete LoraLlama implementation: **4-6 hours** (study candle API, implement, test)
- Add gradient verification: **1 hour**
- Improve loss computation: **30 minutes**
- Testing and validation: **1-2 hours**

**Total**: 6-9 hours of focused development time

## Notes

- Resized BAR available if memory overhead exceeds 16GB VRAM
- CUDA Compute Cap 89 (RTX 5080)
- Current CPU workaround for model forward passes due to Candle GPU limitations
- 112/112 lib tests passing, 4/4 RMS-Norm tests passing
