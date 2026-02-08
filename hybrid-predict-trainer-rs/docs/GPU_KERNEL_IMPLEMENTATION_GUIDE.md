# GPU Kernel Implementation Roadmap

## Current Status

The `src/gpu.rs` module contains **placeholder implementations only**. All GPU acceleration functions are stubs that return empty data structures. This document provides a roadmap for implementing actual CubeCL CUDA kernels.

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│ HybridTrainer (CPU)                                 │
│ ├─ TrainingState                                   │
│ └─ GpuAccelerator<CudaBackend>                     │
│     ├─ encode_state()      → GPU Kernel #1         │
│     ├─ predict_batch()     → GPU Kernel #2         │
│     ├─ compress_residuals()→ GPU Kernel #3         │
│     └─ apply_corrections() → GPU Kernel #4         │
└─────────────────────────────────────────────────────┘
```

## Required Kernels

### 1. State Encoding Kernel
**File:** `src/gpu.rs::encode_state()`
**Priority:** HIGH
**Complexity:** Medium

**Purpose:** Parallel feature extraction from training state

**Input:**
- TrainingState (64-dim feature vector from `compute_features()`)
- Configuration: input_dim=64, output_dim=128, block_size=256

**Output:**
- GpuTensor [batch_size, 128]

**Algorithm:**
```rust
// CubeCL pseudocode
#[cube(launch)]
fn encode_state_kernel(
    input: &Tensor<f32>,      // [batch, 64]
    weights: &Tensor<f32>,    // [64, 128]
    bias: &Tensor<f32>,       // [128]
    output: &mut Tensor<f32>  // [batch, 128]
) {
    let batch_idx = ABSOLUTE_POS_X;
    let feat_idx = ABSOLUTE_POS_Y;

    if batch_idx < input.shape(0) && feat_idx < output.shape(1) {
        let mut sum = bias[feat_idx];
        for i in 0..input.shape(1) {
            sum += input[batch_idx][i] * weights[i][feat_idx];
        }
        output[batch_idx][feat_idx] = relu(sum);
    }
}
```

**Implementation Steps:**
1. Add `cubecl` and `cubecl-cuda` dependencies to Cargo.toml
2. Define CubeCL module structure
3. Implement `encode_state_kernel` using CubeCL macro
4. Add kernel launch logic in `GpuAccelerator::encode_state()`
5. Implement host-device memory transfer
6. Add unit tests with synthetic data
7. Benchmark against CPU baseline

**Estimated LOC:** ~150 lines

---

### 2. GRU Forward Pass Kernel
**File:** `src/gpu.rs::predict_batch()` (calls RSSM forward)
**Priority:** HIGH
**Complexity:** High

**Purpose:** RSSM dynamics model prediction via GRU

**Input:**
- Encoded state tensor [batch, 128]
- GRU hidden state [batch, 256]
- GRU weights (update_gate, reset_gate, candidate)
- Number of prediction steps

**Output:**
- Predicted states [batch, steps, 128]
- Final hidden state [batch, 256]

**Algorithm:**
```rust
// CubeCL pseudocode
#[cube(launch)]
fn gru_forward_kernel(
    input: &Tensor<f32>,           // [batch, input_dim]
    hidden: &mut Tensor<f32>,      // [batch, hidden_dim]
    weights_update: &Tensor<f32>,  // [input_dim + hidden_dim, hidden_dim]
    weights_reset: &Tensor<f32>,
    weights_candidate: &Tensor<f32>,
    output: &mut Tensor<f32>       // [batch, hidden_dim]
) {
    let batch_idx = ABSOLUTE_POS_X;

    // Update gate
    let update = sigmoid(matmul([input, hidden], weights_update));

    // Reset gate
    let reset = sigmoid(matmul([input, hidden], weights_reset));

    // Candidate hidden state
    let candidate = tanh(matmul([input, reset * hidden], weights_candidate));

    // New hidden state
    hidden[batch_idx] = (1.0 - update) * hidden[batch_idx] + update * candidate;
    output[batch_idx] = hidden[batch_idx];
}
```

**Implementation Steps:**
1. Implement helper kernels:
   - `matrix_multiply_kernel` (batched GEMM)
   - `sigmoid_kernel` (element-wise activation)
   - `tanh_kernel` (element-wise activation)
   - `hadamard_product_kernel` (element-wise multiply)
2. Implement `gru_step_kernel` for single timestep
3. Implement `gru_sequence_kernel` for multi-step rollout
4. Add kernel launch in `predict_batch()`
5. Implement RSSM ensemble logic (5 parallel GRUs)
6. Add dropout and layer normalization kernels
7. Test with RSSMLite CPU implementation for correctness
8. Optimize memory access patterns (coalesced reads/writes)

**Estimated LOC:** ~400 lines

---

### 3. Residual Compression Kernel (SVD)
**File:** `src/gpu.rs::compress_residuals()`
**Priority:** Medium
**Complexity:** High

**Purpose:** Low-rank approximation via truncated SVD

**Input:**
- Residual matrix [m, n]
- Target rank k

**Output:**
- U matrix [m, k]
- S vector [k] (singular values)
- V matrix [k, n]

**Algorithm:**
- Use cuSOLVER library for GPU-accelerated SVD
- Alternatively: Implement randomized SVD (Halko 2011)

**Implementation Steps:**
1. Link against cuSOLVER library
2. Implement cuSOLVER wrapper in CubeCL
3. Add memory allocation for workspace
4. Implement truncation logic (keep top-k singular values)
5. Add error handling for numerical stability
6. Test with known low-rank matrices
7. Benchmark against CPU SVD (nalgebra)

**Estimated LOC:** ~200 lines (if using cuSOLVER)

---

### 4. Correction Application Kernel
**File:** `src/gpu.rs::apply_corrections()`
**Priority:** Low
**Complexity:** Low

**Purpose:** Element-wise addition of corrections to predictions

**Input:**
- Predictions tensor [batch, dim]
- Corrections tensor [batch, dim]

**Output:**
- Corrected predictions [batch, dim]

**Algorithm:**
```rust
#[cube(launch)]
fn apply_corrections_kernel(
    predictions: &Tensor<f32>,
    corrections: &Tensor<f32>,
    output: &mut Tensor<f32>
) {
    let idx = ABSOLUTE_POS;
    if idx < predictions.len() {
        output[idx] = predictions[idx] + corrections[idx];
    }
}
```

**Implementation Steps:**
1. Implement element-wise addition kernel
2. Add bounds checking
3. Optimize for memory bandwidth (coalesced access)
4. Test with random tensors

**Estimated LOC:** ~50 lines

---

## Dependencies

### Cargo.toml Additions
```toml
[dependencies]
cubecl = { version = "0.9", features = ["cuda"] }
cubecl-cuda = "0.9"

[features]
cuda = ["cubecl/cuda", "cubecl-cuda"]
```

### System Requirements
- NVIDIA GPU with CUDA Compute Capability ≥ 6.0 (Pascal or newer)
- CUDA Toolkit 11.0+
- cuDNN (for optimized operations)
- cuSOLVER (for SVD kernel)

---

## Testing Strategy

### Unit Tests
- **Correctness:** Compare GPU output vs CPU reference implementation
- **Numerical stability:** Test with edge cases (zeros, large values, near-singular matrices)
- **Memory safety:** Valgrind/CUDA-MEMCHECK for leaks

### Integration Tests
- **End-to-end:** Full prediction cycle on synthetic data
- **Performance:** Benchmark speedup vs CPU baseline
- **Accuracy:** Ensure <1e-5 relative error vs CPU

### Benchmarks
```bash
cargo bench --features cuda -- gpu_kernels
```

Target speedups (vs CPU):
- State encoding: 10-20x
- GRU forward: 5-10x
- SVD compression: 3-5x
- Correction: 20-50x

---

## Development Phases

### Phase 1: Foundation (Week 1-2)
- [ ] Set up CubeCL build environment
- [ ] Implement basic tensor operations (matmul, sigmoid, tanh)
- [ ] Add memory pool management
- [ ] Create test harness with synthetic data

### Phase 2: Core Kernels (Week 3-4)
- [ ] Implement state encoding kernel (#1)
- [ ] Implement GRU forward pass kernel (#2)
- [ ] Validate correctness against CPU RSSMLite

### Phase 3: Advanced Features (Week 5-6)
- [ ] Implement SVD compression kernel (#3)
- [ ] Implement correction kernel (#4)
- [ ] Optimize memory access patterns
- [ ] Add kernel fusion opportunities

### Phase 4: Integration & Optimization (Week 7-8)
- [ ] Integrate into HybridTrainer workflow
- [ ] Add CUDA stream management for concurrency
- [ ] Implement mixed precision (FP16 inference)
- [ ] Profile and optimize bottlenecks
- [ ] Documentation and examples

---

## Performance Targets

### Latency (batch_size=32)
| Operation | CPU (ms) | GPU Target (ms) | Speedup |
|-----------|----------|-----------------|---------|
| State encode | 5.0 | 0.25 | 20x |
| GRU forward (10 steps) | 50.0 | 5.0 | 10x |
| SVD (rank=32) | 15.0 | 3.0 | 5x |
| Correction | 2.0 | 0.04 | 50x |
| **Total** | **72 ms** | **8.29 ms** | **8.7x** |

### Memory Requirements
- State encoding: ~512 KB (weights + activations)
- GRU forward: ~2 MB (weights + hidden states + activations)
- SVD: ~4 MB (workspace + output)
- **Total VRAM:** ~10 MB per batch

---

## References

1. **CubeCL Documentation:** https://github.com/tracel-ai/cubecl
2. **CUDA Best Practices:** NVIDIA CUDA C Programming Guide
3. **GRU Implementation:** "Learning Phrase Representations using RNN Encoder-Decoder" (Cho 2014)
4. **Randomized SVD:** "Finding structure with randomness" (Halko 2011)
5. **Burn Framework:** https://github.com/tracel-ai/burn

---

## Current Limitations

1. **No CubeCL dependency:** Must add to Cargo.toml
2. **No kernel implementations:** All functions are placeholders
3. **No GPU memory management:** Memory pool is stub
4. **No error handling:** CUDA errors not propagated
5. **No benchmarks:** Need criterion benchmarks with `cuda` feature

---

## Next Steps

1. **Immediate:** Add CubeCL dependencies to Cargo.toml
2. **Short-term:** Implement state encoding kernel (highest ROI)
3. **Medium-term:** Implement GRU forward pass (most complex)
4. **Long-term:** Optimize for multi-GPU and async execution

---

## Questions for Review

1. Should we target CUDA only, or add ROCm/Metal support via CubeCL?
2. What minimum CUDA compute capability should we support?
3. Should we implement custom GEMM or use cuBLAS?
4. Is mixed precision (FP16) a requirement for Phase 1?
5. What's the priority order for kernel implementation?

---

**Status:** DRAFT - Ready for technical review
**Last Updated:** 2026-02-07
**Author:** Claude Sonnet 4.5
