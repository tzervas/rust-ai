# CUDA Kernel Implementation Handoff

**Date:** 2026-01-24  
**Status:** Feature branches pushed, ready for testing and completion  
**Total New Lines:** ~7,549 lines of CubeCL kernel code

---

## Repository Branch Mapping

| Repository | Branch | Remote URL | Status |
|------------|--------|------------|--------|
| rust-ai-core | `main` | *needs remote* |  Initial commit, no remote yet |
| trit-vsa | `feature/cuda-kernels-v2` | https://github.com/tzervas/trit-vsa |  Pushed |
| bitnet-quantize | `feature/cuda-kernels-v2` | https://github.com/tzervas/bitnet-quantize |  Pushed |
| peft-rs | `feature/cuda-kernels-v2` | https://github.com/tzervas/peft-rs |  Pushed |
| qlora-rs | `feature/cuda-kernels-v2` | https://github.com/tzervas/qlora-rs |  Pushed |
| unsloth-rs | `feature/cuda-kernels-v2` | https://github.com/tzervas/unsloth-rs |  Pushed |
| vsa-optim-rs | `main` | https://github.com/tzervas/vsa-optim-rs | No changes (depends on trit-vsa GPU) |
| axolotl-rs | `main` | https://github.com/tzervas/axolotl-rs | No changes (integration layer) |
| tritter-accel | `main` | https://github.com/tzervas/tritter-accel | No changes (needs GPU bindings) |

---

## New Crate: rust-ai-core

**Location:** `/home/kang/Documents/projects/rust-ai/rust-ai-core`  
**Lines:** 1,138  
**Status:** Needs GitHub repo creation and remote setup

### Files Created
```
rust-ai-core/
├── Cargo.toml
├── LICENSE-MIT
├── README.md
└── src/
    ├── lib.rs           (72 lines)
    ├── device.rs        (275 lines) - CUDA-first device selection
    ├── error.rs         (224 lines) - Unified CoreError type
    ├── traits.rs        (271 lines) - ValidatableConfig, Quantize, Dequantize, GpuDispatchable
    └── cubecl/
        ├── mod.rs       (35 lines)
        └── interop.rs   (261 lines) - Candle ↔ CubeCL conversion
```

### Remaining Tasks
- [ ] Create GitHub repo `tzervas/rust-ai-core`
- [ ] Push to remote
- [ ] Publish to crates.io as v0.1.0
- [ ] Update all downstream crates to depend on rust-ai-core
- [ ] Replace 8x duplicated `warn_cpu_fallback()` with `rust_ai_core::warn_if_cpu()`

---

## trit-vsa GPU Kernels

**Branch:** `feature/cuda-kernels-v2`  
**Lines Added:** 1,561  
**Feature Flag:** `cuda`

### Files Created
```
src/gpu/
├── mod.rs      (49 lines)
├── kernels.rs  (653 lines) - CubeCL kernel definitions
└── ops.rs      (859 lines) - High-level GPU operation wrappers
```

### Kernels Implemented
| Kernel | Status | Description |
|--------|--------|-------------|
| `BindKernel` |  Implemented | Parallel XOR for VSA binding |
| `UnbindKernel` |  Implemented | Parallel XOR for VSA unbinding |
| `BundleKernel` |  Implemented | Parallel majority vote reduction |
| `DotSimilarityKernel` |  Implemented | GPU popcount + reduce |
| `HammingDistanceKernel` |  Implemented | GPU popcount for distance |

### Remaining Tasks
- [ ] Add GPU unit tests with CUDA hardware
- [ ] Benchmark GPU vs CPU performance
- [ ] Wire `cuda` feature to auto-select GPU ops in public API
- [ ] Add CI with CUDA runner

---

## bitnet-quantize GPU Kernels

**Branch:** `feature/cuda-kernels-v2`  
**Lines Added:** 911  
**Feature Flag:** `cuda`

### Files Modified/Created
```
src/kernels/
├── mod.rs      (56 lines) - Module exports
└── cubecl.rs   (855 lines) - Real CubeCL kernels (was stub)
```

### Kernels Implemented
| Kernel | Status | Description |
|--------|--------|-------------|
| `TernaryMatmulKernel` |  Implemented | Tiled ternary matmul with shared memory |
| `TernaryQuantizeKernel` |  Implemented | AbsMean/AbsMax quantization |
| `TernaryDequantizeKernel` |  Implemented | Scale restoration |

### Remaining Tasks
- [ ] Test with actual CUDA hardware
- [ ] Validate numerical correctness vs CPU path
- [ ] Benchmark against reference implementations
- [ ] Connect to trit-vsa GPU ops for VSA ternary operations

---

## peft-rs GPU Kernels

**Branch:** `feature/cuda-kernels-v2`  
**Lines Added:** 1,100  
**Feature Flag:** `cuda`

### Files Created
```
src/kernels/
├── mod.rs   (59 lines)
├── lora.rs  (695 lines) - Fused LoRA kernels
└── dora.rs  (346 lines) - DoRA weight decomposition kernels
```

### Kernels Implemented
| Kernel | Status | Description |
|--------|--------|-------------|
| `FusedLoraKernel` |  Implemented | Single-pass x @ A @ B |
| `LoraScaleKernel` |  Implemented | Efficient rank scaling |
| `DoraWeightNormKernel` |  Implemented | Weight normalization for DoRA |
| `DoraMagnitudeKernel` |  Implemented | Magnitude computation |

### Remaining Tasks
- [ ] Test kernel launches with CUDA
- [ ] Integrate with existing LoRA/DoRA forward paths
- [ ] Benchmark fused vs sequential (expect 30-50% speedup)
- [ ] Add quantized-LoRA-forward kernel (optional)

---

## qlora-rs GPU Kernels

**Branch:** `feature/cuda-kernels-v2`  
**Lines Added:** 1,356  
**Feature Flag:** `cuda`

### Files Created
```
src/kernels/
├── mod.rs   (79 lines)
├── nf4.rs   (472 lines) - NF4 quantization kernels
├── fp4.rs   (359 lines) - FP4 quantization kernels
└── fused.rs (446 lines) - Fused dequant+matmul
```

### Kernels Implemented
| Kernel | Status | Description |
|--------|--------|-------------|
| `Nf4QuantizeKernel` |  Implemented | NF4 weight quantization |
| `Nf4DequantizeKernel` |  Implemented | Fast GPU dequantization |
| `Fp4QuantizeKernel` |  Implemented | FP4 quantization |
| `Fp4DequantizeKernel` |  Implemented | FP4 dequantization |
| `FusedNf4MatmulKernel` |  Implemented | Dequant + matmul in one pass |
| `DoubleQuantKernel` |  Implemented | Double quantization support |

### Remaining Tasks
- [ ] Validate NF4 lookup table correctness
- [ ] Test double quantization with absmax
- [ ] Benchmark fused matmul vs dequant-then-matmul
- [ ] Critical for QLoRA inference performance

---

## unsloth-rs GPU Kernels

**Branch:** `feature/cuda-kernels-v2`  
**Lines Added:** 1,483 (new fused kernels)  
**Existing:** Flash Attention (~1,000 lines), Ternary kernels (~4,700 lines)

### Files Created/Modified
```
src/kernels/
├── mod.rs                 (modified) - Added fused kernel exports
├── attention_cubecl.rs    (modified) - Fixed kernel launch
├── cubecl/kernel.rs       (modified) - Removed fallback, enabled GPU
├── fused_rmsnorm_rope.rs  (810 lines) - NEW: Fused norm + rotation
└── fused_swiglu.rs        (673 lines) - NEW: Fused gate activation
```

### Kernels Implemented
| Kernel | Status | Description |
|--------|--------|-------------|
| `FlashAttentionKernel` |  Enabled | Removed CPU fallback |
| `FusedRmsnormRopeKernel` |  Implemented | Combined norm + RoPE |
| `FusedSwigluKernel` |  Implemented | Combined gate + activation |
| `TernaryMatmulKernel` |  Existing | Comprehensive ternary matmul |
| `TernaryAttentionKernel` |  Existing | Ternary attention |

### Remaining Tasks
- [ ] Validate Flash Attention numerical accuracy
- [ ] Test fused kernels end-to-end
- [ ] Benchmark memory savings from fused ops
- [ ] Integrate fused kernels into transformer forward pass

---

## Cross-Crate Integration Tasks

### High Priority
1. **Create rust-ai-core GitHub repo** and publish to crates.io
2. **Migrate all crates to rust-ai-core** error types and device selection
3. **CUDA hardware testing** for all kernel implementations
4. **Numerical validation** against CPU reference implementations

### Medium Priority
5. **vsa-optim-rs GPU integration** - Wire to trit-vsa GPU ops
6. **tritter-accel Python bindings** - Expose GPU-accelerated ops
7. **axolotl-rs E2E testing** - Full training loop on GPU
8. **Performance benchmarks** - Establish baselines

### Lower Priority
9. **Google-style docstrings** for all public APIs
10. **CI with CUDA runners** for automated GPU testing
11. **Version bumps** to 2.0.0 for API-breaking changes

---

## Dependency Order for Integration

```
rust-ai-core (v0.1.0)           ← Foundation, publish first
       │
       ▼
trit-vsa (v0.2.0)               ← Add rust-ai-core dep
       │
       ├───────────────┐
       ▼               ▼
bitnet-quantize    vsa-optim-rs  ← Both use trit-vsa GPU
(v0.2.0)           (v0.2.0)
       │
       ▼
peft-rs (v2.0.0)                 ← Add rust-ai-core, cuda kernels
       │
       ▼
qlora-rs (v2.0.0)                ← Add cuda kernels
       │
       ▼
unsloth-rs (v2.0.0)              ← Enable all fused kernels
       │
       ▼
axolotl-rs (v2.0.0)              ← Integration testing
       │
       ▼
tritter-accel (v0.2.0)           ← Python GPU bindings
```

---

## Claude Code Handoff Prompt

```
Continue the CUDA kernel implementation for the rust-ai ecosystem.

## Context
- 8 crates with CubeCL GPU kernels on feature branches
- New `rust-ai-core` crate needs GitHub repo and crates.io publish
- All kernels written but untested on actual CUDA hardware

## Branch Mapping (all feature/cuda-kernels-v2 except rust-ai-core on main)
- rust-ai-core: /home/kang/Documents/projects/rust-ai/rust-ai-core (no remote)
- trit-vsa: https://github.com/tzervas/trit-vsa
- bitnet-quantize: https://github.com/tzervas/bitnet-quantize  
- peft-rs: https://github.com/tzervas/peft-rs
- qlora-rs: https://github.com/tzervas/qlora-rs
- unsloth-rs: https://github.com/tzervas/unsloth-rs

## Immediate Next Steps
1. Create GitHub repo for rust-ai-core, push, publish to crates.io
2. Run `CUDA_COMPUTE_CAP=89 cargo test --features cuda` on each crate
3. Fix any CUDA compilation errors
4. Add GPU unit tests validating kernel correctness
5. Migrate crates to use rust-ai-core::warn_if_cpu() and CoreError

## Kernel Summary (~7,500 lines)
- trit-vsa: bind/unbind/bundle/similarity GPU ops (1,561 lines)
- bitnet-quantize: ternary matmul/quantize (911 lines)
- peft-rs: fused LoRA/DoRA kernels (1,100 lines)
- qlora-rs: NF4/FP4 quantize/dequantize/fused matmul (1,356 lines)
- unsloth-rs: Flash Attention enabled, fused RMSNorm+RoPE, fused SwiGLU (1,483 lines new)
- rust-ai-core: device selection, errors, traits, cubecl interop (1,138 lines)

## 1.0.0 Criteria
- Full CUDA kernel coverage for all ops (no CPU fallbacks in hot path)
- Google-style docstrings with "why" explanations
- Unified API surface via rust-ai-core traits
- Comprehensive GPU tests
- CI with CUDA runners
```

---

## Summary Statistics

| Metric | Value |
|--------|-------|
| Crates with new GPU code | 6 |
| Total new lines of code | ~7,549 |
| New CubeCL kernels | 20+ |
| Feature branches pushed | 5 |
| Remaining to publish | 1 (rust-ai-core) |
| Est. hours to complete | 20-40 (mostly testing) |
