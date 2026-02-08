# Cross-Repository Coordination Plan
## Rust-AI Workspace Multi-Repo Analysis & Enhancement Strategy

**Status:** DRAFT - Comprehensive analysis of 12 workspace repos
**Created:** 2026-02-07
**Last Updated:** 2026-02-08
**Analysis Scope:** Full workspace + tritter variants + excluded crates

---

## Executive Summary

Analysis of the rust-ai workspace reveals **significant code reuse opportunities** and **GPU kernel implementations** that can accelerate hybrid-predict-trainer-rs development. Three crates are currently excluded due to uuid/js-sys conflicts, blocking access to production-ready CubeCL kernels in unsloth-rs and ecosystem orchestration in rust-ai-core.

### Key Findings

1. **Production-Ready GPU Kernels Available** 📦
   - unsloth-rs has Flash Attention CubeCL implementation (150+ LOC)
   - unsloth-rs has ternary matmul with 4 optimization levels
   - Can be extracted to hybrid-predict-trainer-rs with adaptation

2. **Ecosystem Orchestration Layer Exists** 🏗️
   - rust-ai-core defines GpuDispatchable, Quantize, ValidatableConfig traits
   - hybrid-predict-trainer-rs should implement these for ecosystem compatibility

3. **Dependency Resolution Blocking Progress** ⚠️
   - uuid v1.20.0 (burn-core) vs js-sys v0.3.72 (cubecl/wasm-bindgen-futures)
   - Blocks 3 crates: axolotl-rs, rust-ai-core, unsloth-rs
   - Fixable with local cubecl patch or upstream update

4. **Tritter Variants Show Good Architecture** ✅
   - tritter-accel delegates to sister crates (no duplication)
   - tritter-model-rs consumes hybrid-predict-trainer-rs (downstream validation)

---

## Workspace Structure

### Active Members (9 crates)

```
├── bitnet-quantize        [0.2.1] BitNet b1.58 quantization
├── hybrid-predict-trainer-rs [0.2.0] 🎯 Primary focus (THIS CRATE)
├── peft-rs                [1.0.2] LoRA, DoRA, AdaLoRA adapters
├── qlora-rs               [1.0.2] 4-bit quantized LoRA
├── training-tools         [0.1.0] Training utilities and harness
├── trit-vsa               [0.3.0] Balanced ternary + VSA ops (GPU-enabled)
├── tritter-accel          [0.2.2] Python bindings for ternary accel
├── tritter-model-rs       [0.1.0] Ternary transformer models
└── vsa-optim-rs           [0.1.0] VSA gradient compression
```

### Excluded Members (3 crates) - **BLOCKED**

```
├── axolotl-rs             [1.1.0] Fine-tuning orchestration (reqwest → js-sys conflict)
├── rust-ai-core           [0.3.4] 🌟 Ecosystem orchestration (cubecl → uuid conflict)
└── unsloth-rs             [1.0.2] 🌟 GPU-optimized kernels (cubecl → uuid conflict)
```

### Forks (3 directories)

```
├── qlora-candle          Git fork: candle with qlora-gemm integration
├── qlora-gemm            Git fork: GEMM optimizations
└── qlora-paste           Git fork: Paste macro utilities
```

---

## Dependency Graph

### Conceptual Layers

```
Layer 4: Applications & Orchestration
    ├── tritter-model-rs (uses hybrid-predict-trainer-rs)
    ├── axolotl-rs (EXCLUDED - orchestration)
    └── rust-ai-core (EXCLUDED - unifies all below)

Layer 3: Training Infrastructure
    ├── hybrid-predict-trainer-rs 🎯 (uses trit-vsa, bitnet-quantize)
    ├── training-tools (generic harness)
    └── vsa-optim-rs (VSA optimization)

Layer 2: Acceleration & Quantization
    ├── tritter-accel (delegates to trit-vsa, bitnet-quantize, vsa-optim-rs)
    ├── bitnet-quantize (ternary quantization)
    ├── unsloth-rs (EXCLUDED - GPU kernels)
    └── qlora-rs (uses peft-rs)

Layer 1: Foundations
    ├── trit-vsa (ternary arithmetic + VSA, 70% GPU-ready)
    ├── peft-rs (adapter primitives)
    └── candle-core (tensor ops)
```

### Critical Dependencies for hybrid-predict-trainer-rs

**Current:**
```toml
[dependencies]
burn = "0.20.1"          # Deep learning framework
trit-vsa = "0.3"         # Ternary VSA (workspace)
bitnet-quantize = "0.2"  # BitNet quant (workspace)
serde = "1.0.228"
thiserror = "2.0.18"
rand = { version = "0.8", features = ["small_rng"] }
half = "2.7.1"
tokio = { version = "1.49.0", optional = true }

[features]
cuda = ["burn/cuda"]     # NO CubeCL due to uuid conflict
```

**Should Add (after conflict resolution):**
```toml
# Ecosystem integration
rust-ai-core = { path = "../rust-ai-core" }  # GpuDispatchable trait

# GPU kernels (extracted or via dependency)
cubecl = "0.9"           # After uuid fix
cubecl-cuda = "0.9"      # After uuid fix

# Or use unsloth-rs directly
unsloth-rs = { path = "../unsloth-rs", features = ["cuda"] }
```

---

## UUID/JS-SYS Conflict Analysis

### Root Cause

```
cubecl 0.9.0
├── wasm-bindgen-futures (for WASM support)
│   └── js-sys = "0.3.72" (pinned, no "std" feature)
│
burn-core 0.20.1
└── uuid = "1.20.0" (requires js-sys >= 0.3.74 with "std")

CONFLICT: js-sys version + feature mismatch
```

### Why This Matters

- **cubecl** needed for GPU kernels (Flash Attention, GRU, matmul)
- **burn** needed for hybrid-predict-trainer-rs tensor ops
- Can't have both in same workspace currently

### Resolution Options

#### Option 1: Wait for Upstream (LOW EFFORT, SLOW)
- Wait for cubecl to update wasm-bindgen-futures
- Timeline: Unknown (depends on upstream maintenance)
- **Status:** Not acceptable given "time is not a factor, focus on productivity"

#### Option 2: Local CubeCL Patch (MEDIUM EFFORT, FAST)
```toml
[patch.crates-io]
wasm-bindgen-futures = { git = "https://github.com/rustwasm/wasm-bindgen", branch = "main" }
# Use latest wasm-bindgen with js-sys 0.3.74+
```
- Patch at workspace level
- Test that cubecl still works
- **Timeline:** 1-2 hours
- **Risk:** Low (well-tested dependency)

#### Option 3: Fork CubeCL Locally (HIGH EFFORT, FULL CONTROL)
- Fork cubecl repo
- Update wasm-bindgen-futures dependency
- Point workspace at local fork
- **Timeline:** 4-6 hours
- **Benefit:** Full control over updates

#### Option 4: Conditional Features (MEDIUM EFFORT, WORKAROUND)
```toml
# In hybrid-predict-trainer-rs/Cargo.toml
[features]
cuda = ["cubecl/cuda", "cubecl-cuda"]  # Enable only when building standalone
```
- Build hybrid-predict-trainer-rs outside workspace for GPU
- Keep workspace for CPU-only builds
- **Timeline:** 1 hour
- **Limitation:** No GPU in workspace builds

### Recommended Approach: **Option 2 + Option 4**

1. Apply local patch for wasm-bindgen-futures (immediate fix)
2. Use conditional features for flexibility
3. Submit PR to cubecl to update upstream (community contribution)

**Implementation:**
```bash
# 1. Add patch to workspace Cargo.toml
echo '[patch.crates-io]
wasm-bindgen-futures = { git = "https://github.com/rustwasm/wasm-bindgen", branch = "main" }' >> Cargo.toml

# 2. Uncomment cubecl in workspace.dependencies
# 3. Uncomment rust-ai-core, unsloth-rs from workspace.members
# 4. cargo build --workspace
```

---

## Code Extraction Opportunities

### 1. Flash Attention Kernel (unsloth-rs → hybrid-predict-trainer-rs)

**Source:** `/home/kang/Documents/projects/rust-ai/unsloth-rs/src/kernels/cubecl/kernel.rs`

**What:** Production-ready Flash Attention 2 implementation
- 150+ LOC of well-documented CubeCL kernel
- Online softmax with O(N) memory
- Handles arbitrary head dimensions
- Causal and non-causal variants

**Extraction Plan:**
1. Copy `flash_attention_tile` kernel to `hybrid-predict-trainer-rs/src/gpu/kernels/attention.rs`
2. Adapt for GRU attention mechanism (if needed)
3. Add to `GpuAccelerator::predict_batch()` implementation
4. **Effort:** 2-3 hours
5. **Benefit:** Enables GPU-accelerated attention for RSSMLite

**Code Snippet:**
```rust
#[cube(launch)]
fn flash_attention_tile<F: Float + CubeElement>(
    q: &Array<F>,       // Query [batch * heads * seq_len * head_dim]
    k: &Array<F>,       // Key
    v: &Array<F>,       // Value
    out: &mut Array<F>, // Output
    scale: F,           // 1/sqrt(head_dim)
    seq_len_val: u32,
    head_dim_val: u32,
    block_size_val: u32,
) {
    // Tiled computation with online softmax
    // ... (150 lines of optimized kernel)
}
```

### 2. Ternary Matmul Kernels (unsloth-rs → hybrid-predict-trainer-rs)

**Source:** `/home/kang/Documents/projects/rust-ai/unsloth-rs/src/kernels/ternary/matmul_cubecl.rs`

**What:** Four optimization levels for ternary matmul
- Basic popcount kernel (validation)
- Tiled with shared memory
- Vectorized Line<u32> loads
- Sparse with plane skipping (95%+ sparsity)

**Extraction Plan:**
1. Copy ternary matmul kernels to `hybrid-predict-trainer-rs/src/gpu/kernels/ternary.rs`
2. Use for weight delta compression in PredictiveExecutor
3. Integrate with ResidualCorrector for sparse corrections
4. **Effort:** 3-4 hours
5. **Benefit:** 4x speedup for sparse gradient operations

### 3. GpuDispatchable Trait (rust-ai-core → hybrid-predict-trainer-rs)

**Source:** `/home/kang/Documents/projects/rust-ai/rust-ai-core/src/traits.rs`

**What:** Standard trait for CPU/GPU dispatch
```rust
pub trait GpuDispatchable: Send + Sync {
    fn dispatch_gpu(&self, device: &Device) -> Result<Tensor>;
    fn dispatch_cpu(&self) -> Result<Tensor>;
}
```

**Integration Plan:**
1. Add rust-ai-core as dependency (after conflict fix)
2. Implement `GpuDispatchable` for:
   - `RSSMLite` (dynamics model)
   - `ResidualCorrector`
   - `TrainingState::compute_features()`
3. **Effort:** 2-3 hours
4. **Benefit:** Ecosystem compatibility + consistent GPU/CPU fallback

### 4. ValidatableConfig Trait (rust-ai-core → hybrid-predict-trainer-rs)

**Source:** `/home/kang/Documents/projects/rust-ai/rust-ai-core/src/traits.rs`

**What:** Configuration validation interface
```rust
pub trait ValidatableConfig: Clone + Send + Sync {
    fn validate(&self) -> Result<()>;
}
```

**Integration Plan:**
1. Implement for all config types:
   - `HybridTrainerConfig`
   - `PredictorConfig`
   - `DivergenceConfig`
   - `CheckpointConfig` (v0.3.0 modules)
2. Add validation in constructors
3. **Effort:** 1-2 hours
4. **Benefit:** Better error messages, ecosystem consistency

---

## Cross-Repo Enhancement Tasks

### Priority Matrix

| Task | Effort | Impact | Dependencies | Priority |
|------|--------|--------|--------------|----------|
| 1. Resolve uuid/js-sys conflict | Medium | Critical | None | P0 |
| 2. Extract Flash Attention kernel | Low | High | Task 1 | P1 |
| 3. Implement GpuDispatchable | Low | High | Task 1 | P1 |
| 4. Extract ternary matmul | Medium | Medium | Task 1 | P2 |
| 5. Implement ValidatableConfig | Low | Low | Task 1 | P3 |
| 6. Add rust-ai-core dependency | Low | High | Task 1 | P1 |
| 7. Test in tritter-model-rs | Low | High | Tasks 2-6 | P1 |

### Detailed Task Breakdown

#### Task 1: Resolve UUID/JS-SYS Conflict (P0)

**Goal:** Enable cubecl in workspace to unlock GPU kernels

**Steps:**
1. Test wasm-bindgen-futures patch locally
   ```bash
   cd /home/kang/Documents/projects/rust-ai
   # Add patch to Cargo.toml
   # Uncomment cubecl in workspace.dependencies
   cargo build --workspace
   ```
2. If successful, uncomment:
   - rust-ai-core from workspace.members
   - unsloth-rs from workspace.members
   - axolotl-rs from workspace.members (if needed)
3. Run full workspace test suite
   ```bash
   cargo test --workspace --all-features
   ```
4. Document solution in WORKSPACE_DEPENDENCY_FIXES.md
5. Submit PR to cubecl upstream

**Success Criteria:**
- [ ] `cargo build --workspace` succeeds
- [ ] All workspace tests pass
- [ ] CubeCL CUDA features work
- [ ] No regression in CPU-only builds

**Timeline:** 2-4 hours
**Blocker:** None
**Blocked:** All other GPU kernel tasks

---

#### Task 2: Extract Flash Attention Kernel (P1)

**Goal:** Add production Flash Attention to hybrid-predict-trainer-rs

**Source Files:**
- unsloth-rs/src/kernels/cubecl/kernel.rs (flash_attention_tile)
- unsloth-rs/src/kernels/cubecl/config.rs (TileConfig)
- unsloth-rs/src/kernels/cubecl/interop.rs (has_cubecl_cuda_support)

**Target Files:**
- hybrid-predict-trainer-rs/src/gpu/kernels/mod.rs (new)
- hybrid-predict-trainer-rs/src/gpu/kernels/attention.rs (new)
- hybrid-predict-trainer-rs/src/gpu/kernels/config.rs (new)

**Steps:**
1. Create `src/gpu/kernels/` module structure
2. Copy kernel code with proper attribution
3. Adapt for RSSM attention (if different from standard attention)
4. Wire into `GpuAccelerator::predict_batch()`
5. Add unit tests comparing CPU vs GPU output
6. Benchmark on synthetic data

**Code Changes:**
```rust
// src/gpu/kernels/attention.rs
#[cfg(feature = "cuda")]
#[cube(launch)]
fn flash_attention_rssm<F: Float + CubeElement>(
    // ... adapted from unsloth-rs
) {
    // Same algorithm, potentially modified for GRU patterns
}
```

**Testing:**
```rust
#[test]
#[ignore] // Requires GPU
fn test_flash_attention_correctness() {
    let gpu_out = gpu_accelerator.attention(q, k, v)?;
    let cpu_out = cpu_baseline_attention(q, k, v);
    assert_tensor_close(gpu_out, cpu_out, 1e-5);
}
```

**Success Criteria:**
- [ ] Kernel compiles with `cuda` feature
- [ ] CPU vs GPU outputs match within 1e-5
- [ ] 5-10x speedup over CPU baseline
- [ ] Integrates into existing predict_batch() flow

**Timeline:** 3-4 hours
**Depends On:** Task 1
**Blocks:** None

---

#### Task 3: Implement GpuDispatchable Trait (P1)

**Goal:** Make hybrid-predict-trainer-rs compatible with rust-ai-core ecosystem

**Source:** rust-ai-core/src/traits.rs

**Target Types:**
1. `RSSMLite` (dynamics model)
2. `ResidualCorrector`
3. `TrainingState` (for feature computation)
4. `GpuAccelerator` (wrapper)

**Steps:**
1. Add rust-ai-core dependency:
   ```toml
   [dependencies]
   rust-ai-core = { path = "../rust-ai-core" }
   ```
2. Implement trait for each type:
   ```rust
   use rust_ai_core::GpuDispatchable;

   impl GpuDispatchable for RSSMLite {
       fn dispatch_gpu(&self, device: &Device) -> Result<Tensor> {
           self.gpu_accelerator.as_ref()
               .ok_or(Error::NoGpu)?
               .predict_batch(/* ... */)
       }

       fn dispatch_cpu(&self) -> Result<Tensor> {
           // CPU fallback
       }
   }
   ```
3. Add feature flags for rust-ai-core integration:
   ```toml
   [features]
   ecosystem = ["rust-ai-core"]
   ```
4. Update docs to mention ecosystem compatibility

**Success Criteria:**
- [ ] Trait implementations compile
- [ ] CPU fallback works when GPU unavailable
- [ ] GPU dispatch uses CubeCL kernels
- [ ] Compatible with rust-ai-core examples

**Timeline:** 2-3 hours
**Depends On:** Task 1, Task 6
**Blocks:** None

---

#### Task 6: Add rust-ai-core Dependency (P1)

**Goal:** Integrate with ecosystem orchestration layer

**Changes:**
```toml
# hybrid-predict-trainer-rs/Cargo.toml
[dependencies]
rust-ai-core = { path = "../rust-ai-core", version = "0.3" }

[features]
ecosystem = ["rust-ai-core"]
cuda = ["rust-ai-core/cuda", "cubecl/cuda"]
```

**Benefits:**
- Access to `GpuDispatchable`, `ValidatableConfig`, `Quantize` traits
- Consistent error types (`CoreError`)
- Logging infrastructure
- Python/TypeScript binding patterns (future)

**Steps:**
1. Add dependency after Task 1 complete
2. Implement core traits (Tasks 3, 5)
3. Add examples showing ecosystem integration
4. Update README with ecosystem section

**Success Criteria:**
- [ ] rust-ai-core compiles in workspace
- [ ] hybrid-predict-trainer-rs builds with ecosystem feature
- [ ] Traits implemented correctly
- [ ] Examples run without errors

**Timeline:** 1 hour
**Depends On:** Task 1
**Blocks:** Tasks 3, 5

---

#### Task 7: Validate in tritter-model-rs (P1)

**Goal:** Ensure downstream compatibility

**Why:** tritter-model-rs depends on hybrid-predict-trainer-rs, so any breaking changes will show up here first.

**Steps:**
1. After implementing Tasks 2-6, rebuild tritter-model-rs:
   ```bash
   cd /home/kang/Documents/projects/rust-ai/tritter-model-rs
   cargo build --features cuda
   ```
2. Run tritter-model-rs tests:
   ```bash
   cargo test
   ```
3. Run training examples:
   ```bash
   cargo run --example train_100m --features cuda
   ```
4. Check for:
   - Compilation errors
   - Test failures
   - Runtime performance regressions
   - GPU memory issues

**Success Criteria:**
- [ ] tritter-model-rs compiles
- [ ] All tests pass
- [ ] Training examples run
- [ ] GPU utilization matches expectations
- [ ] No memory leaks

**Timeline:** 1-2 hours
**Depends On:** Tasks 2, 3, 6
**Blocks:** None

---

## Workspace-Wide Improvements

### 1. Standardize on rust-ai-core Traits

**Goal:** All crates implement common interfaces

**Affected Crates:**
- hybrid-predict-trainer-rs (GpuDispatchable, ValidatableConfig)
- bitnet-quantize (Quantize, Dequantize)
- trit-vsa (GpuDispatchable for VSA ops)
- vsa-optim-rs (ValidatableConfig)
- peft-rs (ValidatableConfig)
- qlora-rs (Quantize)

**Benefits:**
- Interchangeable implementations
- Consistent error handling
- Easier testing (mock traits)
- Better documentation

**Effort:** 1-2 hours per crate = 6-12 hours total
**Priority:** P2 (after GPU kernel extraction)

---

### 2. Consolidate Documentation

**Goal:** Single source of truth for workspace architecture

**Create:**
- `/docs/WORKSPACE_ARCHITECTURE.md` (this document's home)
- `/docs/TRAIT_IMPLEMENTATIONS.md` (which crate implements what)
- `/docs/DEPENDENCY_GRAPH.md` (visual dependency map)
- `/docs/CUDA_SETUP_GUIDE.md` (GPU prerequisites)

**Benefits:**
- Onboarding new developers
- Understanding cross-crate dependencies
- Debugging build issues

**Effort:** 4-6 hours
**Priority:** P3

---

### 3. Add Workspace-Level CI/CD

**Goal:** Catch cross-repo breaking changes early

**GitHub Actions Workflow:**
```yaml
name: Workspace CI

on: [push, pull_request]

jobs:
  test-workspace:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Install Rust
        uses: actions-rs/toolchain@v1
      - name: Build workspace
        run: cargo build --workspace
      - name: Test workspace
        run: cargo test --workspace
      - name: Clippy
        run: cargo clippy --workspace -- -D warnings

  test-cuda:
    runs-on: [self-hosted, gpu]
    steps:
      - name: Build with CUDA
        run: cargo build --workspace --features cuda
      - name: Test GPU kernels
        run: cargo test --workspace --features cuda -- --ignored
```

**Effort:** 2-3 hours
**Priority:** P2

---

## Multi-Repo Coordination Prompt

**Use this prompt when working across multiple rust-ai crates:**

```
You are coordinating changes across the rust-ai workspace ecosystem.

WORKSPACE STRUCTURE:
- Primary crate: hybrid-predict-trainer-rs
- Sister crates: trit-vsa, bitnet-quantize, vsa-optim-rs, peft-rs, qlora-rs
- Orchestration: rust-ai-core (defines GpuDispatchable, ValidatableConfig traits)
- GPU kernels: unsloth-rs (Flash Attention, ternary matmul CubeCL implementations)
- Downstream: tritter-model-rs (consumes hybrid-predict-trainer-rs)

CURRENT BLOCKERS:
- uuid/js-sys conflict prevents cubecl in workspace
- Resolution: Apply wasm-bindgen-futures patch (see CROSS_REPO_COORDINATION.md Task 1)

CODE EXTRACTION OPPORTUNITIES:
1. Flash Attention kernel from unsloth-rs → hybrid-predict-trainer-rs/src/gpu/kernels/attention.rs
2. Ternary matmul from unsloth-rs → hybrid-predict-trainer-rs/src/gpu/kernels/ternary.rs
3. GpuDispatchable trait from rust-ai-core → implement on RSSMLite, ResidualCorrector
4. ValidatableConfig trait from rust-ai-core → implement on all config types

WORKFLOW:
1. Check if change affects multiple crates (grep across /home/kang/Documents/projects/rust-ai/)
2. Update all affected Cargo.toml files together
3. Run `cargo build --workspace` to catch breaking changes early
4. Test downstream in tritter-model-rs before finalizing
5. Document cross-repo changes in git commit message

TESTING PROTOCOL:
- After changes: `cargo test --workspace`
- GPU changes: `cargo test --workspace --features cuda -- --ignored`
- Breaking changes: Check tritter-model-rs compilation

When asked to "coordinate multi-repo work", refer to this document's task breakdown (Tasks 1-7) and implementation priority matrix.
```

---

## Implementation Roadmap

### Phase 1: Unblock Workspace (Week 1)
- [ ] **Day 1:** Resolve uuid/js-sys conflict (Task 1)
- [ ] **Day 2:** Uncomment excluded crates, test workspace build
- [ ] **Day 3:** Add rust-ai-core dependency (Task 6)
- [ ] **Day 4:** Extract Flash Attention kernel (Task 2)
- [ ] **Day 5:** Implement GpuDispatchable (Task 3)

**Milestone:** Workspace builds with GPU support, hybrid-predict-trainer-rs has Flash Attention kernel

---

### Phase 2: GPU Kernel Integration (Week 2)
- [ ] **Day 1-2:** Extract ternary matmul kernels (Task 4)
- [ ] **Day 3:** Wire kernels into GpuAccelerator
- [ ] **Day 4:** Benchmarking and optimization
- [ ] **Day 5:** Validate in tritter-model-rs (Task 7)

**Milestone:** hybrid-predict-trainer-rs has full GPU acceleration (8.7x speedup target achieved)

---

### Phase 3: Ecosystem Integration (Week 3)
- [ ] **Day 1:** Implement ValidatableConfig (Task 5)
- [ ] **Day 2-3:** Standardize traits across workspace
- [ ] **Day 4:** Consolidate documentation
- [ ] **Day 5:** Add workspace CI/CD

**Milestone:** All crates follow rust-ai-core conventions, comprehensive docs

---

### Phase 4: Validation & Release (Week 4)
- [ ] **Day 1-2:** Full workspace test suite
- [ ] **Day 3:** GPU kernel validation on hardware
- [ ] **Day 4:** Performance benchmarking vs baseline
- [ ] **Day 5:** Release prep (CHANGELOG, version bumps)

**Milestone:** Ready for v0.3.0 release with GPU acceleration

---

## Key Metrics

### Before Optimization
- Workspace members: 9/12 (3 excluded)
- GPU kernels: 0 (placeholders only)
- Ecosystem integration: None
- Cross-repo tests: None

### After Implementation (Target)
- Workspace members: 12/12 (all included)
- GPU kernels: 4 (attention, GRU, ternary matmul, corrections)
- Ecosystem integration: Full (GpuDispatchable, ValidatableConfig)
- Cross-repo tests: Automated CI/CD
- Training speedup: 8.7x (72ms → 8.3ms per step)

---

## Appendices

### A. File Inventory

**unsloth-rs (26 source files):**
- `src/kernels/cubecl/kernel.rs` - Flash Attention (EXTRACT)
- `src/kernels/cubecl/config.rs` - Kernel configs (EXTRACT)
- `src/kernels/ternary/matmul_cubecl.rs` - Ternary matmul (EXTRACT)
- `src/kernels/rope.rs` - RoPE (potential use)
- `src/kernels/rmsnorm.rs` - RMSNorm (potential use)

**rust-ai-core (16 source files):**
- `src/traits.rs` - GpuDispatchable, ValidatableConfig (IMPLEMENT)
- `src/device.rs` - Device abstraction (USE)
- `src/error.rs` - CoreError types (ADOPT)
- `src/cubecl/interop.rs` - CubeCL patterns (REFERENCE)

**hybrid-predict-trainer-rs (current):**
- `src/gpu.rs` - 435 lines of placeholders (REPLACE with real kernels)
- `src/dynamics.rs` - RSSMLite (ADD GpuDispatchable)
- `src/corrector.rs` - ResidualCorrector (ADD GpuDispatchable)
- `src/config.rs` - All configs (ADD ValidatableConfig)

---

### B. Dependency Version Matrix

| Crate | candle | cubecl | burn | uuid | pyo3 |
|-------|--------|--------|------|------|------|
| hybrid-predict-trainer-rs | - | ❌ | 0.20.1 | 1.20.0 | - |
| unsloth-rs | 0.9.2 | 0.9 | - | - | - |
| rust-ai-core | 0.9 | 0.9 | - | - | 0.27 |
| trit-vsa | 0.9.2 | 0.9 | - | - | - |
| tritter-accel | 0.9.2 | - | - | - | 0.27 |

**Conflicts:**
- cubecl 0.9 requires js-sys 0.3.72 (via wasm-bindgen-futures)
- uuid 1.20.0 requires js-sys >= 0.3.74 with "std" feature

---

### C. Code Size Analysis

| Crate | Total LOC | GPU Code | Tests | Examples |
|-------|-----------|----------|-------|----------|
| hybrid-predict-trainer-rs | ~6,800 | 435 (stubs) | 2,273 | ~500 |
| unsloth-rs | ~3,500 | ~1,200 | ~400 | ~200 |
| rust-ai-core | ~2,100 | ~300 | ~400 | ~150 |
| trit-vsa | ~4,800 | ~1,500 | ~800 | ~300 |
| tritter-accel | ~4,263 | 0 (delegates) | ~500 | ~200 |
| tritter-model-rs | ~6,328 | 0 | ~800 | ~600 |

**Total Workspace:** ~40,000 LOC
**GPU-Ready Code:** ~3,000 LOC (mostly in unsloth-rs, trit-vsa)

---

## Conclusion

The rust-ai workspace contains **production-ready GPU kernel implementations** and **ecosystem orchestration infrastructure** that can immediately accelerate hybrid-predict-trainer-rs development. The primary blocker (uuid/js-sys conflict) is **resolvable in 2-4 hours** with a local patch.

**Recommended immediate action:**
1. Apply wasm-bindgen-futures patch (Task 1)
2. Extract Flash Attention kernel (Task 2)
3. Implement GpuDispatchable trait (Task 3)

This will unlock 8.7x training speedup and full ecosystem integration within 1-2 weeks of focused development.

---

**Document Status:** READY FOR IMPLEMENTATION
**Next Review:** After Task 1 completion
**Maintained By:** Claude Sonnet 4.5 + Tyler Zervas
