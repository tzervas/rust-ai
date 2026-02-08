# Phase 2B: Real Model Validation Plan

**Status**: Ready to begin
**Goal**: Validate memory optimizations on real models (GPT-2 Small → 1B params)
**Hardware**: RTX 5080 (16GB VRAM)
**Timeline**: 5-7 days

---

## MNIST CNN Validation Results ✅

**Completed**: 2026-02-07

### Baseline (Vanilla Burn)
- Time: 810.4s
- Accuracy: 97.45%
- Memory: 195.3 MB

### Conservative (conf=0.60, H=50, interval=15)
- Time: 599.0s (**1.35× speedup**, 26% faster)
- Accuracy: 99.05% (better than baseline!)
- Memory: 161.8 MB (17% savings)
- Backward reduction: 32.8%
- Divergences: 0 ✅

### Aggressive (conf=0.55, H=75, interval=10)
- Time: 431.5s (**1.88× speedup**, 47% faster)
- Accuracy: 99.37% (best!)
- Memory: 189.1 MB
- Backward reduction: 42.8%
- Divergences: 0 ✅

**Key Insight**: Speedups lower than expected on CPU. GPU validation will show true potential (60-78% from previous research).

---

## Phase 2B Objectives

1. **Validate on progressively larger models**
   - 124M params (GPT-2 Small) - baseline
   - 350M params (GPT-2 Medium) - memory stress
   - 1B params (GPT-2 XL scale) - RTX 5080 target

2. **Measure actual VRAM usage**
   - Baseline (vanilla FP32)
   - Each optimization individually
   - All optimizations combined
   - Compare to estimates

3. **Validate quality preservation**
   - Final perplexity within 2% of baseline
   - Training stability (no divergences)
   - Convergence speed

4. **Document production configuration**
   - Recommended settings for RTX 5080
   - Conservative vs Aggressive tradeoffs
   - Memory/quality curves

---

## Milestone 1: GPT-2 Small (124M params)

**Task**: #30
**Duration**: 2 days
**Priority**: CRITICAL

### Step 1.1: Implement Architecture

**File**: `examples/gpt2_small_baseline.rs`

```rust
use burn::nn::{Embedding, Linear, LayerNorm, Dropout};
use burn::module::Module;

#[derive(Module, Debug)]
pub struct GPT2Config {
    vocab_size: usize,        // 50257
    n_positions: usize,       // 1024
    n_ctx: usize,             // 1024
    n_embd: usize,            // 768
    n_layer: usize,           // 12
    n_head: usize,            // 12
    dropout: f64,             // 0.1
}

#[derive(Module, Debug)]
pub struct GPT2Block<B: Backend> {
    ln_1: LayerNorm<B>,
    attn: MultiHeadAttention<B>,
    ln_2: LayerNorm<B>,
    mlp: MLP<B>,
}

#[derive(Module, Debug)]
pub struct GPT2Model<B: Backend> {
    wte: Embedding<B>,        // Token embeddings
    wpe: Embedding<B>,        // Position embeddings
    drop: Dropout,
    h: Vec<GPT2Block<B>>,    // 12 transformer blocks
    ln_f: LayerNorm<B>,
}
```

**Parameters**: ~124M
**Memory (FP32)**: ~500MB model + ~500MB gradients + ~1GB optimizer = ~2GB total

**Success Criteria**:
- Model compiles
- Forward pass works
- Backward pass works
- Can train 100 steps without OOM

---

### Step 1.2: Baseline Training

**File**: `examples/gpt2_small_baseline.rs`

```rust
fn main() -> HybridResult<()> {
    let device = <Cuda<f32>>::default();
    let model = GPT2Model::new(&device);
    let optimizer = AdamConfig::new().init();

    // Synthetic data (initially)
    let batch_size = 8;
    let seq_len = 512;

    for step in 0..1000 {
        let (input, target) = generate_batch(batch_size, seq_len);

        let output = model.forward(input);
        let loss = cross_entropy_loss(output, target);

        let grads = loss.backward();
        optimizer.step(&model, &grads);

        if step % 10 == 0 {
            println!("Step {}: Loss {:.4}", step, loss.into_scalar());
            measure_vram_usage(); // nvidia-smi
        }
    }
}
```

**Metrics to collect**:
- Loss curve
- Perplexity
- Peak VRAM (nvidia-smi)
- Time per step
- Gradient norm

**Success Criteria**:
- Training completes 1000 steps
- Loss converges (decreasing trend)
- Peak VRAM <5GB (should be ~2GB)

---

### Step 1.3: HybridTrainer Integration

**File**: `examples/gpt2_small_hybrid.rs`

```rust
fn main() -> HybridResult<()> {
    let device = <Cuda<f32>>::default();
    let model = GPT2Model::new(&device);
    let optimizer = AdamConfig::new().init();

    // Conservative config
    let config = HybridTrainerConfig::builder()
        .warmup_steps(100)
        .full_steps(20)
        .max_predict_steps(50)
        .confidence_threshold(0.60)
        .correction_interval(15)
        .build();

    let mut trainer = HybridTrainer::new(model, optimizer, config)?;

    for step in 0..1000 {
        let (input, target) = generate_batch(8, 512);

        let loss = trainer.step(|model| {
            let output = model.forward(input);
            cross_entropy_loss(output, target)
        })?;

        if step % 10 == 0 {
            println!("Step {}: Loss {:.4}, Phase: {:?}",
                     step, loss, trainer.current_phase());
            measure_vram_usage();
        }
    }

    let stats = trainer.statistics();
    println!("Final speedup: {:.2}×", stats.wall_clock_speedup);
}
```

**Configs to test**:
1. Conservative: conf=0.60, H=50, interval=15
2. Aggressive: conf=0.55, H=75, interval=10

**Success Criteria**:
- 40-60% speedup on GPU (not CPU!)
- Quality within 2% of baseline
- Zero divergences
- Peak VRAM <5GB

---

### Step 1.4: Memory Optimization Testing

**File**: `examples/gpt2_small_memory_optimized.rs`

Test each optimization individually:

**Test 1: Mixed Precision Only**
```rust
let config = HybridTrainerConfig::builder()
    .mixed_precision_config(MixedPrecisionConfig::aggressive())
    .build();
```
Expected: ~1GB VRAM (50% of 2GB)

**Test 2: Gradient Accumulation Only**
```rust
let config = HybridTrainerConfig::builder()
    .gradient_accumulation_config(GradientAccumulationConfig::aggressive())
    .build();
```
Expected: ~1.4GB VRAM (30% savings on activations)

**Test 3: Predict-Aware Only**
```rust
let config = HybridTrainerConfig::builder()
    .predict_aware_memory_config(PredictAwareMemoryConfig::aggressive())
    .build();
```
Expected: ~1GB during Predict (50% savings on optimizer)

**Test 4: All Combined**
```rust
let config = HybridTrainerConfig::builder()
    .mixed_precision_config(MixedPrecisionConfig::aggressive())
    .gradient_accumulation_config(GradientAccumulationConfig::aggressive())
    .predict_aware_memory_config(PredictAwareMemoryConfig::aggressive())
    .build();
```
Expected: ~0.6GB during Predict (70% savings)

**Success Criteria**:
- VRAM savings within ±10% of estimates
- Quality degradation <1%
- All tests complete without OOM

---

## Milestone 2: GPT-2 Medium (350M params)

**Task**: #31 (continued)
**Duration**: 1 day
**Priority**: HIGH

Same as GPT-2 Small but with:
- n_embd: 1024
- n_layer: 24
- n_head: 16
- Parameters: ~350M

**Memory estimate (FP32)**:
- Model: ~1.4GB
- Gradients: ~1.4GB
- Optimizer: ~2.8GB
- Total: ~5.6GB

**With optimizations**: ~1.5GB during Predict (73% savings)

**Success Criteria**:
- Baseline fits in 16GB VRAM
- Optimized version uses <2GB
- Quality within 2% of baseline

---

## Milestone 3: 1B Parameter Model

**Task**: #32
**Duration**: 2 days
**Priority**: HIGH

**Architecture**: GPT-2 XL scale
- n_embd: 1600
- n_layer: 48
- n_head: 25
- Parameters: ~1B

**Memory estimate (FP32)**:
- Model: ~4GB
- Gradients: ~4GB
- Optimizer: ~8GB
- Total: ~16GB (at limit!)

**With all optimizations**:
- Predict phase: ~2-3GB (85% savings)
- Full phase: ~6-7GB (60% savings)

**Success Criteria**:
- ✅ Fits in 16GB VRAM with optimizations
- ✅ Peak VRAM <14GB (2GB safety margin)
- ✅ Training stable for 1000+ steps
- ✅ Quality within 2% of baseline (if baseline fits)

**This is the primary use case for RTX 5080!**

---

## Deliverables

### Code
- [ ] `examples/gpt2_small_baseline.rs` - Vanilla training
- [ ] `examples/gpt2_small_hybrid.rs` - HybridTrainer
- [ ] `examples/gpt2_small_memory_optimized.rs` - All optimizations
- [ ] `examples/gpt2_medium_validation.rs` - 350M params
- [ ] `examples/gpt2_1b_validation.rs` - 1B params

### Documentation
- [ ] `docs/VALIDATION_RESULTS.md` - Comprehensive results
- [ ] `docs/RTX_5080_GUIDE.md` - Production configuration guide
- [ ] Update `README.md` with validated results

### Metrics
- [ ] VRAM usage charts (nvidia-smi logs)
- [ ] Speedup comparison table
- [ ] Quality degradation analysis
- [ ] Memory savings breakdown

---

## Timeline

| Day | Milestone | Deliverable |
|-----|-----------|-------------|
| 1 | GPT-2 Small architecture | Model compiles, baseline training works |
| 2 | GPT-2 Small validation | HybridTrainer + memory optimizations tested |
| 3 | GPT-2 Medium validation | Larger model stress test |
| 4 | 1B model implementation | Architecture defined |
| 5-6 | 1B model validation | Training completes, metrics collected |
| 7 | Documentation | Results documented, guide published |

**Total**: 7 days (5-7 with buffer)

---

## Success Criteria (Overall)

### Must Have
- ✅ 1B model fits in 16GB VRAM with optimizations
- ✅ Peak VRAM <14GB (2GB safety margin)
- ✅ Quality degradation <2% across all models
- ✅ Training stable (zero divergences)
- ✅ VRAM savings within ±10% of estimates

### Nice to Have
- ⭐ 60-78% speedup on GPU (validated)
- ⭐ Quality degradation <1%
- ⭐ 7B model exploration (if 1B very successful)

---

## Risk Mitigation

### Risk: 1B model doesn't fit even with optimizations
**Likelihood**: Low (estimates show it should fit)
**Mitigation**:
- Start with smaller batch size
- Enable gradient checkpointing
- Test with 768M params first (between Medium and 1B)

### Risk: Quality degradation >2%
**Likelihood**: Medium (mixed precision can affect convergence)
**Mitigation**:
- Use BF16 instead of FP16 (better dynamic range)
- Disable aggressive optimizations if needed
- Fine-tune confidence threshold

### Risk: GPU speedup lower than expected
**Likelihood**: Medium (CPU results were 26-47% vs expected 60-78%)
**Mitigation**:
- Profile to identify bottlenecks
- Optimize predict phase overhead
- Consider async operations

### Risk: Integration complexity
**Likelihood**: High (framework complete but integration pending)
**Mitigation**:
- Prioritize one optimization at a time
- Start with mixed precision (simplest)
- Use Opus for complex integration planning

---

## Next Steps

1. **Create GPT-2 Small architecture** (Task #30)
   - Transformer implementation in Burn
   - Baseline training script
   - Metric collection

2. **Use Opus for complex planning if needed**
   - Architecture design review
   - Integration strategy
   - Performance optimization

3. **Iterate on validation**
   - Small → Medium → 1B
   - Collect comprehensive metrics
   - Document findings

4. **Publish results**
   - Update README with validated results
   - Create RTX 5080 production guide
   - Share benchmarks

---

**Status**: Ready to begin Phase 2B
**Next**: Task #30 - Implement GPT-2 Small architecture
**Goal**: Validate 1B model training on RTX 5080 (16GB VRAM)

---

**Last Updated**: 2026-02-07
**Phase 2A**: ✅ Complete (framework + documentation)
**Phase 2B**: ⏳ Starting (real model validation)
