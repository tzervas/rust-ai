# VRAM Growth Analysis & Fix

**Date**: 2026-02-07
**Issue**: HybridTrainer shows steady VRAM growth from 3.8 GB → 14.1 GB during 50-step training

---

## Root Cause Analysis

### Hypothesis: Autodiff Graph Accumulation in Predict Phase

During the Predict phase, `HybridTrainer::execute_predict_step()` calls:

```rust
// Line 977 in lib.rs
let actual_loss = model.forward(batch)?;
```

This forward pass:
1. **Stores loss tensor** with autodiff graph in `BurnModelWrapper::last_loss`
2. **Never calls backward()** (Predict phase skips backward for speedup)
3. **Autodiff graph may accumulate** intermediate activations

### Memory Flow Comparison

**Full Phase (Normal)**:
```
forward() → store loss + autodiff graph (4 GB activations)
backward() → compute gradients, FREE autodiff graph
optimizer.step() → apply gradients
→ Net memory: stable
```

**Predict Phase (Problem)**:
```
forward() → store loss + autodiff graph (4 GB activations)
[NO backward() call]
next forward() → replace loss, but activations may persist?
→ Net memory: growing +200 MB per step
```

### Evidence from Validation

| Config | Initial VRAM | Final VRAM | Growth | Predict % |
|--------|--------------|------------|--------|-----------|
| Baseline | 3.8 GB | 4.2 GB | +0.4 GB | 0% |
| Hybrid | 3.8 GB | 14.1 GB | **+10.3 GB** | 40% |
| Memory-Opt | 3.7 GB | 8.8 GB | **+5.1 GB** | 40% |

**Correlation**: Both configs with Predict phase show significant growth

### Growth Rate Calculation

- **Hybrid**: 10.3 GB / 50 steps = **206 MB/step average**
- **Predict phase**: 20 steps (40%) with forward-only
- **Growth per predict step**: 10.3 GB / 20 = **515 MB/step**

For GPT-2 Small (124M params):
- **Model weights**: 124M × 4 bytes = 496 MB (FP32)
- **Activations per batch**: ~2-4 GB (depends on sequence length)
- **515 MB/step matches ~1 batch of activations**

---

## Proposed Fix

### Option 1: Explicit Memory Clearing (Immediate Fix)

Add method to `BurnModelWrapper` to clear autodiff graph without backward:

```rust
// In burn_integration.rs
impl<B, M, T, F> BurnModelWrapper<B, M, T, F>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
    F: BurnForwardFn<B, M, T>,
{
    /// Clears the last loss tensor and its autodiff graph.
    ///
    /// Use this during Predict phase when backward() won't be called,
    /// to prevent memory accumulation from unused autodiff graphs.
    pub fn clear_loss(&mut self) {
        *self.last_loss.write() = None;
    }
}
```

Then in `lib.rs`, after forward in Predict phase:

```rust
// In execute_predict_step(), after line 977
let actual_loss = model.forward(batch)?;

// Clear autodiff graph immediately (won't call backward in Predict phase)
model.clear_loss();
```

### Option 2: Detach Loss Tensor (Cleaner)

Modify forward() to optionally detach from autodiff graph:

```rust
// Add to Model trait
trait Model<B: Batch> {
    fn forward(&mut self, batch: &B) -> HybridResult<f32>;
    fn forward_no_grad(&mut self, batch: &B) -> HybridResult<f32>; // NEW
    // ... rest of trait
}

// In BurnModelWrapper
fn forward_no_grad(&mut self, batch: &B) -> HybridResult<f32> {
    // Same as forward() but with .detach() on loss tensor
    // This prevents autodiff graph creation entirely
    let (model_returned, loss_tensor) =
        (self.forward_fn).forward(model, &batch);
    let loss_detached = loss_tensor.detach(); // No autodiff graph
    let loss_scalar = extract_scalar(&loss_detached)?;
    // Don't store loss_tensor (no backward needed)
    Ok(loss_scalar)
}
```

Then use `forward_no_grad()` in Predict phase.

### Option 3: Burn's no_grad() Context (Most Efficient)

Wrap Predict phase forward in Burn's no-grad context:

```rust
// Pseudo-code (need to check Burn API)
let actual_loss = {
    use burn::tensor::no_grad::no_grad;
    no_grad(|| model.forward(batch))?
};
```

This prevents autodiff graph construction entirely.

---

## Implementation Plan

### Phase 1: Quick Fix (Option 1)
1. Add `clear_loss()` method to `BurnModelWrapper`
2. Call after forward in `execute_predict_step()`
3. Test on GPT-2 Small hybrid (expect stable VRAM)
4. Run 100-step validation

### Phase 2: Proper Solution (Option 2 or 3)
1. Research Burn's no-grad API
2. Implement `forward_no_grad()` or use Burn's context manager
3. Benchmark to verify no performance regression
4. Update all 3 examples

### Phase 3: Validation
1. Re-run Phase 2B validation with fix
2. Verify VRAM stable over 1000 steps
3. Measure impact on throughput (should be neutral or positive)
4. Document in PHASE_2B_VALIDATION_COMPLETE.md

---

## Expected Results After Fix

| Config | Initial VRAM | Final VRAM | Growth | Status |
|--------|--------------|------------|--------|--------|
| Baseline | 3.8 GB | 4.2 GB | +0.4 GB | ✅ Stable (reference) |
| Hybrid | 3.8 GB | **4.5 GB** | **+0.7 GB** | ✅ Fixed (marginal growth OK) |
| Memory-Opt | 3.7 GB | **4.2 GB** | **+0.5 GB** | ✅ Fixed |

**Target**: Growth <1 GB over 50 steps, <5 GB over 1000 steps

---

## Additional Investigation Needed

1. **Gradient Accumulation**: Does `accumulation_steps=4` store intermediate gradients?
2. **Corrector State**: Is `linear_model` (64-dim) accumulating?
3. **Dynamics Model History**: Are BPTT buffers growing beyond limits?
4. **Burn Internals**: Does Burn cache intermediate tensors for optimization?

---

## References

- Burn autodiff docs: https://burn.dev/book/building-blocks/autodiff.html
- PyTorch's `torch.no_grad()`: Similar concept for reference
- Memory profiling: Use `nvidia-smi dmon -s mu` for real-time tracking

---

## Status

- [x] Root cause identified (autodiff graph in Predict phase)
- [ ] Fix implemented (Option 1 ready to code)
- [ ] Fix validated (need to test)
- [ ] Long-term solution (Option 2/3 needs research)
