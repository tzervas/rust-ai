# VRAM Leak Investigation - Executive Summary

**Date**: 2026-02-07
**Status**: ✅ Root cause identified | ⏳ Fix in progress

---

## The Problem

HybridTrainer shows **10 GB VRAM growth** over 50 training steps:
- Baseline (vanilla Burn): 4.2 GB stable ✅
- HybridTrainer: 3.9 GB → 14.1 GB ❌

**Impact**: Blocks longer runs, limits model size

---

## Root Cause

**Burn's `.map()` API creates full model copies (496 MB each)**

Location: `src/burn_integration.rs:799`
```rust
fn apply_deltas_to_model(...) -> M {
    model.map(&mut mapper)  // Creates ENTIRE model copy!
}
```

**Frequency**:
- Predict phase: 1 copy per step (20 steps)
- Correct phase: 1 copy per step (15 steps)
- **Total**: 35 copies × 496 MB = **17 GB allocations**

CUDA doesn't free old copies fast enough → accumulation

---

## Solution: Delta Accumulation

Instead of applying deltas immediately, **accumulate and apply in batches**:

```rust
// Current (broken):
for step in predict_phase {
    model.apply_weight_delta(delta)?;  // 496 MB copy each time
}

// Fixed (batched):
let mut accumulated_delta = WeightDelta::zero();
for step in predict_phase {
    accumulated_delta.merge(delta);  // Just sum, no copies
}
model.apply_weight_delta(&accumulated_delta)?;  // One copy at end
```

**Expected result**: 35 copies → 5 copies = **7× memory reduction**

---

## Implementation Status

### Completed ✅
- Root cause identified (Burn's `.map()` creating copies)
- Comprehensive analysis documented (3 documents)
- GPU coordination plan created
- Quick workaround tested (correction_interval=0) - insufficient

### In Progress ⏳
- Delta accumulation implementation (ETA: 2 hours)

### Blocked ⏸️
- Phase 2B re-validation (awaiting VRAM fix)
- Longer training runs (1000+ steps)
- Larger model scaling (350M-1B params)

---

## Documentation Created

1. **VRAM_GROWTH_ANALYSIS.md**: Initial hypothesis (autodiff graphs) - disproven
2. **VRAM_LEAK_ROOT_CAUSE.md**: Complete analysis + 4 solution options
3. **GPU_COORDINATION_PLAN.md**: Multi-session GPU management strategy
4. **THIS FILE**: Executive summary for quick reference

---

## Immediate Next Steps

1. **Implement delta accumulation** (2 hours, Sonnet)
2. **Validate fix** (50-100 steps, expect 4-5 GB stable)
3. **Re-run Phase 2B validation** (all 3 configs)
4. **Document final results**

---

## GPU Coordination (Multi-Session)

**Current situation**:
- This session: Using 4-14 GB VRAM (leak)
- self-hosted-ai VS Code session: Unknown usage

**Recommendation**: **Time-slice GPU access**
```bash
# This session finishes VRAM fix (~2 hours)
# Then creates checkpoint and releases GPU
# Other session can then use full 16 GB VRAM
```

**Lock file coordination**: `/tmp/gpu_lock_hybrid`

---

## Questions for User

1. Proceed with delta accumulation fix now? (2 hours)
2. Or quick workaround (increase max_predict_steps)?
3. What's the priority of self-hosted-ai GPU tasks?

**Awaiting direction to proceed!**
