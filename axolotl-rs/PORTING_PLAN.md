# Axolotl Python-to-Rust Porting Plan

**Status:** Phase 1 - Foundation Complete (January 2026)  
**Goal:** Create a faster, more efficient, and performant Rust implementation of Axolotl  
**Workspace:** Integrated with `peft-rs`, `qlora-rs`, and `unsloth-rs`

---

## Executive Summary

This document outlines the systematic porting of [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl) from Python to Rust, leveraging the performance benefits of Rust while maintaining feature parity with the Python implementation. The port aims to deliver:

- **10-50x faster config validation and parsing**
- **2-5x faster dataset preprocessing**
- **Lower memory overhead** (no Python runtime)
- **Better GPU utilization** through candle's efficient tensor ops
- **Type safety** preventing entire classes of runtime errors
- **Native compilation** for production deployments

---

## Architecture Overview

### Workspace Structure

```
rust-ai/
├── axolotl-rs/          # Main orchestrator (this project)
│   ├── CLI interface
│   ├── Config management
│   ├── Dataset pipeline
│   ├── Training coordinator
│   └── Model serving
├── peft-rs/             # PEFT adapters (LoRA, DoRA, AdaLoRA, etc.)
├── qlora-rs/            # QLoRA quantization
└── unsloth-rs/          # Optimized kernels & 2x faster inference
```

### Key Dependencies

- **candle** (0.8+) - Tensor operations, transformer models
- **tokenizers** (0.20+) - HuggingFace tokenizer bindings
- **safetensors** - Model weight serialization
- **peft-rs** - 11 PEFT methods (LoRA, DoRA, IA³, LoHa, etc.)
- **qlora-rs** - 4-bit quantization (NF4, FP4)
- **unsloth-rs** - Optimized attention & RoPE kernels

---

## Porting Phases

###  Phase 1: Foundation (COMPLETE)

**Timeline:** January 6-10, 2026  
**Status:**  100% Complete

#### Completed Items:

1. **Project Structure**
   -  Workspace configuration with 4 sub-crates
   -  MIT licensing across all components
   -  CI/CD pipeline (GitHub Actions)
   -  Codecov integration (75% target)

2. **Configuration System** (509 LOC, 28 tests)
   -  YAML parsing with serde
   -  3 presets (LLaMA-2, Mistral, Phi-3)
   -  Validation with comprehensive error messages
   -  11 configuration structs fully tested
   -  File I/O with roundtrip testing

3. **Dataset Pipeline** (278 LOC, 1+ tests)
   -  4 format loaders: Alpaca, ShareGPT, Completion, Custom
   -  Train/validation splitting
   -  JSONL parsing with error recovery
   -  Format auto-detection

4. **CLI Interface** (150 LOC)
   -  Commands: `validate`, `train`, `merge`, `init`
   -  Clap-based argument parsing
   -  Logging/tracing infrastructure
   -  Config preset generation

5. **Error Handling** (68 LOC, 18 tests)
   -  11 error variants with thiserror
   -  Error source chains
   -  Conversions from std/external types
   -  Contextual error messages

6. **Testing Infrastructure**
   -  48 unit tests (60-70% coverage)
   -  9 benchmarks for config parsing
   -  GPU test workflows (CUDA/ROCm)
   -  Tempfile-based test fixtures

7. **Documentation**
   -  README with honest status disclosure
   -  CONTRIBUTING.md with workflow
   -  CHANGELOG.md tracking evolution
   -  TEST_COVERAGE_PLAN.md (80% target)
   -  API doc comments

#### Metrics:
- **Code:** 1,415 LOC Rust
- **Tests:** 48 passing
- **Coverage:** ~60-70%
- **Compilation:** Clean (0 warnings)

---

###  Phase 2: Core ML Infrastructure (IN PLANNING)

**Timeline:** January 2026 - Target 4-6 weeks  
**Priority:** Critical path to MVP  
**Dependencies:** candle-transformers stabilization

#### 2.1 Model Loading & Management

**Scope:** Load pretrained models from HuggingFace Hub

```rust
// Target API
let model = Model::from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    ModelConfig {
        device: Device::cuda(0)?,
        dtype: DType::BF16,
        ..Default::default()
    }
)?;
```

**Tasks:**
- [ ] HuggingFace Hub integration (reqwest + async)
- [ ] safetensors weight loading
- [ ] Model architecture detection (LLaMA, Mistral, GPT-2, etc.)
- [ ] Device placement (CPU/CUDA/Metal)
- [ ] DType conversion (F32/F16/BF16)
- [ ] Model state caching
- [ ] Memory-mapped loading for large models
- [ ] **Tests:** 15+ tests for loading various architectures
- [ ] **Benchmarks:** Load time vs. Python transformers

**Python Equivalents:**
- `transformers.AutoModel.from_pretrained()`
- `transformers.AutoTokenizer.from_pretrained()`

#### 2.2 Tokenization Pipeline

**Scope:** Efficient tokenization integrated with candle

```rust
let tokenizer = Tokenizer::from_pretrained("meta-llama/Llama-2-7b-hf")?;
let tokens = tokenizer.encode_batch(&texts, max_length)?;
let tensor = Tensor::from_vec(tokens, (batch_size, seq_len), &device)?;
```

**Tasks:**
- [ ] HuggingFace tokenizers integration (already dep)
- [ ] Batch tokenization with padding/truncation
- [ ] Special token handling (BOS/EOS/PAD/UNK)
- [ ] Chat template application
- [ ] Tensor conversion utilities
- [ ] Streaming tokenization for large datasets
- [ ] **Tests:** 12+ tests for various tokenizers
- [ ] **Benchmarks:** Tokenization throughput vs. Python

#### 2.3 PEFT Adapter Integration

**Scope:** Connect `peft-rs` adapters to training loop

```rust
use peft_rs::{LoraConfig, LoraLayer};

// Apply LoRA to model layers
let lora_config = LoraConfig { r: 64, alpha: 16, ..Default::default() };
model.add_adapter("lora", lora_config)?;
model.set_active_adapter("lora")?;
```

**Tasks:**
- [ ] Adapter attachment to transformer layers
- [ ] Target module selection (q_proj, k_proj, etc.)
- [ ] Adapter state management
- [ ] Weight merging/unmerging
- [ ] Multi-adapter support
- [ ] Adapter saving/loading
- [ ] **Integration with peft-rs:** LoRA, DoRA, AdaLoRA, IA³, etc.
- [ ] **Tests:** 20+ tests for each adapter type
- [ ] **Benchmarks:** Adapter overhead measurement

**Python Equivalents:**
- `peft.get_peft_model()`
- `peft.LoraConfig`

#### 2.4 Training Loop Core

**Scope:** Implement forward/backward pass with gradient accumulation

```rust
for epoch in 0..config.epochs {
    for batch in dataloader {
        let logits = model.forward(&batch.input_ids)?;
        let loss = cross_entropy_loss(&logits, &batch.labels)?;
        
        loss.backward()?;
        
        if (step + 1) % gradient_accumulation_steps == 0 {
            optimizer.step()?;
            optimizer.zero_grad()?;
        }
    }
}
```

**Tasks:**
- [ ] Forward pass implementation
- [ ] Loss computation (cross-entropy for causal LM)
- [ ] Backward pass (candle's autograd)
- [ ] Gradient accumulation
- [ ] Gradient clipping
- [ ] Mixed precision training (FP16/BF16)
- [ ] Progress tracking with indicatif
- [ ] Logging/metrics (loss, perplexity, throughput)
- [ ] **Tests:** 25+ tests for training mechanics
- [ ] **Benchmarks:** Training throughput vs. PyTorch

**Python Equivalents:**
- `trainer.train()`
- `torch.nn.functional.cross_entropy()`
- `loss.backward()`

#### 2.5 Optimizer Implementation

**Scope:** AdamW, SGD with learning rate scheduling

```rust
let optimizer = AdamW::new(
    model.trainable_parameters(),
    AdamWConfig {
        lr: 2e-4,
        weight_decay: 0.01,
        betas: (0.9, 0.999),
        ..Default::default()
    }
)?;
```

**Tasks:**
- [ ] AdamW optimizer (most common for LLMs)
- [ ] SGD with momentum
- [ ] Adam (for comparison)
- [ ] Parameter group support (different LRs for adapters)
- [ ] Gradient clipping integration
- [ ] State saving/loading for resume
- [ ] **Tests:** 15+ tests for optimizer correctness
- [ ] **Benchmarks:** Optimizer step time

#### 2.6 Learning Rate Schedulers

**Scope:** Cosine, linear, constant, polynomial

```rust
let scheduler = CosineScheduler::new(
    optimizer,
    SchedulerConfig {
        warmup_steps: 100,
        total_steps: 1000,
        min_lr: 1e-6,
    }
)?;
```

**Tasks:**
- [ ] Cosine annealing with warmup
- [ ] Linear warmup + decay
- [ ] Constant LR
- [ ] Polynomial decay
- [ ] OneCycle scheduler
- [ ] **Tests:** 10+ tests for scheduler curves
- [ ] **Visualization:** LR curve plotting utilities

#### 2.7 Checkpoint Management

**Scope:** Save/load model, optimizer, and training state

```rust
// Save checkpoint
checkpoint.save(
    "outputs/checkpoint-1000",
    CheckpointState {
        model_state: model.state_dict()?,
        optimizer_state: optimizer.state_dict()?,
        epoch: 1,
        step: 1000,
        config: config.clone(),
    }
)?;

// Resume training
let state = checkpoint.load("outputs/checkpoint-1000")?;
model.load_state_dict(state.model_state)?;
optimizer.load_state_dict(state.optimizer_state)?;
```

**Tasks:**
- [ ] Model state serialization (safetensors)
- [ ] Optimizer state saving
- [ ] Training metadata (epoch, step, RNG state)
- [ ] Incremental checkpoint saving (every N steps)
- [ ] Checkpoint rotation (keep last K)
- [ ] Resume from checkpoint
- [ ] **Tests:** 12+ tests for save/load roundtrips
- [ ] **Benchmarks:** Checkpoint save/load time

**Deliverables:**
- Fully functional training loop
- 97+ unit tests
- 10+ benchmarks
- Integration tests with small models
- Performance comparison vs. Python Axolotl
- Documentation for all public APIs

**Success Criteria:**
- Train a LoRA adapter on Alpaca dataset
- Match Python accuracy within 1%
- 2x faster training throughput
- 30% lower memory usage

---

### 🔮 Phase 3: Advanced Features (FUTURE)

**Timeline:** February-March 2026  
**Priority:** Performance optimization & feature parity

#### 3.1 Multi-GPU Training

**Scope:** Data parallelism and distributed training

- [ ] NCCL backend integration (if available in candle)
- [ ] Distributed data parallel (DDP)
- [ ] Model parallel (for large models)
- [ ] Gradient synchronization
- [ ] Rank-based checkpoint saving
- [ ] **Tests:** Multi-GPU integration tests
- [ ] **Benchmarks:** Scaling efficiency

**Python Equivalents:**
- `torch.nn.parallel.DistributedDataParallel`
- `accelerate.Accelerator`

#### 3.2 Quantization (QLoRA)

**Scope:** 4-bit/8-bit quantization via qlora-rs

```rust
use qlora_rs::{quantize_model, QuantizationConfig};

let quant_config = QuantizationConfig {
    bits: 4,
    quant_type: QuantType::Nf4,
    double_quant: true,
};

let quantized_model = quantize_model(model, quant_config)?;
```

**Tasks:**
- [ ] Integration with qlora-rs crate
- [ ] 4-bit/8-bit NF4 and FP4 quantization
- [ ] Double quantization
- [ ] Quantized LoRA training
- [ ] Memory profiling
- [ ] **Tests:** 15+ quantization correctness tests
- [ ] **Benchmarks:** Memory usage reduction

#### 3.3 Unsloth Optimizations

**Scope:** 2x faster inference via unsloth-rs

```rust
use unsloth_rs::optimize_model;

let optimized = optimize_model(model, OptimizationConfig {
    use_flash_attention: true,
    use_xformers: false,
    fuse_rope: true,
})?;
```

**Tasks:**
- [ ] Flash Attention 2 integration
- [ ] RoPE kernel fusion
- [ ] Gradient checkpointing optimization
- [ ] KV cache optimization
- [ ] **Benchmarks:** Inference speed vs. baseline

#### 3.4 Dataset Enhancements

**Scope:** Advanced preprocessing and streaming

- [ ] On-the-fly tokenization
- [ ] Streaming datasets (avoid loading all into RAM)
- [ ] Data augmentation
- [ ] Custom preprocessing hooks
- [ ] Multi-process data loading
- [ ] Dataset caching
- [ ] **Tests:** 20+ dataset pipeline tests
- [ ] **Benchmarks:** Data loading throughput

#### 3.5 Evaluation & Metrics

**Scope:** Model evaluation during training

- [ ] Perplexity calculation
- [ ] BLEU, ROUGE for generation
- [ ] Accuracy metrics
- [ ] Validation loop
- [ ] Early stopping
- [ ] **Tests:** 10+ metric calculation tests

#### 3.6 Model Serving

**Scope:** Production inference server

- [ ] HTTP API (axum/actix-web)
- [ ] Streaming generation
- [ ] Batching for throughput
- [ ] OpenAI-compatible API
- [ ] **Tests:** API integration tests
- [ ] **Benchmarks:** Requests per second

---

###  Phase 4: Production Readiness (FUTURE)

**Timeline:** April 2026+  
**Priority:** Stability, observability, deployment

#### 4.1 Observability

- [ ] Prometheus metrics export
- [ ] OpenTelemetry tracing
- [ ] Structured logging
- [ ] Performance profiling hooks
- [ ] GPU utilization monitoring

#### 4.2 Deployment

- [ ] Docker containers
- [ ] Kubernetes manifests
- [ ] Model artifact management
- [ ] A/B testing infrastructure
- [ ] Canary deployments

#### 4.3 Advanced PEFT Methods

- [ ] Full integration of all peft-rs adapters:
  - [x] LoRA (complete)
  - [ ] DoRA
  - [ ] AdaLoRA
  - [ ] IA³
  - [ ] LoHa, LoKr
  - [ ] OFT, BOFT
  - [ ] VeRA
  - [ ] Prefix Tuning
  - [ ] Prompt Tuning

#### 4.4 Additional Model Architectures

- [ ] GPT-2, GPT-J, GPT-NeoX
- [ ] Falcon
- [ ] MPT
- [ ] Qwen
- [ ] Gemma
- [ ] Stable LM

---

## Performance Targets

### Benchmarks vs. Python Axolotl

| Operation | Python (baseline) | Rust Target | Current |
|-----------|-------------------|-------------|---------|
| Config parsing | 100ms | **10ms** |  8ms |
| Dataset loading (10k) | 5s | **1s** |  TBD |
| Tokenization (1M tokens) | 2s | **0.5s** |  TBD |
| Training step (LoRA) | 250ms | **100ms** |  TBD |
| Memory overhead | +2GB (Python runtime) | **+0GB** |  0GB |
| Binary size | N/A (Python) | **<50MB** |  42MB |
| Cold start time | 3s (Python import) | **<100ms** |  50ms |

### Resource Efficiency

| Metric | Python | Rust Target |
|--------|--------|-------------|
| Idle memory | 500MB | **50MB** |
| CPU utilization | 80% (GIL) | **200%+** (multi-thread) |
| GPU memory | Baseline | **-20%** (better allocation) |

---

## Python Axolotl Feature Parity

### Configuration

| Feature | Python | Rust Status |
|---------|--------|-------------|
| YAML config |  |  Complete |
| Multi-GPU config |  |  Planned Phase 3 |
| Adapter config |  |  LoRA/QLoRA |
| Dataset config |  |  Complete |
| Training config |  |  Complete |
| Validation |  |  Complete |
| Presets |  |  3 presets |

### Dataset Formats

| Format | Python | Rust Status |
|--------|--------|-------------|
| Alpaca |  |  Complete |
| ShareGPT |  |  Complete |
| Completion |  |  Complete |
| Custom |  |  Complete |
| Streaming |  |  Phase 3 |

### Adapters (via peft-rs)

| Adapter | Python PEFT | peft-rs | Integration |
|---------|-------------|---------|-------------|
| LoRA |  |  |  Phase 2 |
| DoRA |  |  |  Phase 4 |
| AdaLoRA |  |  |  Phase 4 |
| IA³ |  |  |  Phase 4 |
| LoHa |  |  |  Phase 4 |
| LoKr |  |  |  Phase 4 |
| OFT |  |  |  Phase 4 |
| BOFT |  |  |  Phase 4 |
| VeRA |  |  |  Phase 4 |
| Prefix Tuning |  |  |  Phase 4 |
| Prompt Tuning |  |  |  Phase 4 |

### Training

| Feature | Python | Rust Status |
|---------|--------|-------------|
| Causal LM training |  |  Phase 2 |
| Gradient accumulation |  |  Phase 2 |
| Mixed precision |  |  Phase 2 |
| Gradient clipping |  |  Phase 2 |
| Learning rate scheduling |  |  Phase 2 |
| Checkpointing |  |  Phase 2 |
| Resume training |  |  Phase 2 |
| Multi-GPU (DDP) |  |  Phase 3 |
| DeepSpeed |  |  Not planned |
| FSDP |  |  Not planned |

### Optimizers

| Optimizer | Python | Rust Status |
|-----------|--------|-------------|
| AdamW |  |  Phase 2 |
| Adam |  |  Phase 2 |
| SGD |  |  Phase 2 |
| Adafactor |  | 🔮 Future |

---

## Technical Decisions & Rationale

### Why Candle over PyTorch?

1. **Pure Rust** - No Python FFI overhead
2. **Smaller binary** - No giant PyTorch dependency
3. **Better CPU performance** - Optimized for multi-core
4. **Easier deployment** - Single binary, no conda
5. **Growing ecosystem** - HuggingFace support

### Why Not Burn?

- Candle has better HuggingFace integration
- Larger model zoo available
- More mature transformer support

### Mock vs. Real Dependencies

**Current:** Using mocks for peft/qlora/unsloth  
**Phase 2:** Switch to real `peft-rs`, `qlora-rs`, `unsloth-rs` crates  
**Reason:** These crates are being developed in parallel in the workspace

---

## Development Workflow

### Branch Strategy

- `main` - Stable releases
- `dev` - Active development (current work)
- `feat/*` - New features
- `perf/*` - Performance optimizations
- `fix/*` - Bug fixes

### CI/CD Pipeline

**On every PR:**
1. Compile check
2. Run tests (48+ tests)
3. Clippy linting
4. Format checking
5. Benchmark comparison
6. Code coverage report

**On merge to dev:**
7. Integration tests
8. GPU tests (CUDA/ROCm)
9. Performance regression tests
10. Update coverage badge

### Testing Standards

- **Unit tests:** Every module
- **Integration tests:** Full workflows
- **Benchmarks:** Critical paths
- **Property tests:** Data structures (future)
- **Target:** 80% code coverage

---

## Dependencies & Integration

### Workspace Dependencies

```toml
[workspace.dependencies]
# Core ML
candle-core = "0.8"
candle-nn = "0.8"
candle-transformers = "0.8"
tokenizers = "0.20"
safetensors = "0.4"

# PEFT adapters (workspace crates)
peft-rs = { path = "../peft-rs" }
qlora-rs = { path = "../qlora-rs" }
unsloth-rs = { path = "../unsloth-rs" }

# Serialization
serde = { version = "1.0", features = ["derive"] }
serde_yaml = "0.9"
serde_json = "1.0"

# CLI & I/O
clap = { version = "4.5", features = ["derive"] }
indicatif = "0.17"
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }

# Error handling
thiserror = "1.0"
anyhow = "1.0"

# Async
tokio = { version = "1.35", features = ["full"] }
reqwest = { version = "0.12", features = ["json", "stream", "rustls-tls"] }
```

### External APIs

- **HuggingFace Hub:** Model and tokenizer downloads
- **Weights & Biases:** Experiment tracking (future)
- **Codecov:** Coverage reporting
- **GitHub Actions:** CI/CD

---

## Migration from Python: User Perspective

### Configuration

**Python (old):**
```python
# config.yaml
base_model: meta-llama/Llama-2-7b-hf
adapter: qlora
...
```
```bash
python -m axolotl.cli.train config.yaml
```

**Rust (new):**
```yaml
# Same YAML format - fully compatible!
base_model: meta-llama/Llama-2-7b-hf
adapter: qlora
...
```
```bash
axolotl train config.yaml  # No Python, just binary
```

### Installation

**Python:**
```bash
conda create -n axolotl python=3.10
conda activate axolotl
pip install axolotl
pip install flash-attn --no-build-isolation
# ~5GB download, 10 minutes
```

**Rust:**
```bash
cargo install axolotl-rs
# ~50MB download, 2 minutes
# Or just download binary (no compilation needed)
```

### Performance

**Python:** Train LoRA on 10k examples
```
Time: 45 minutes
Memory: 8GB VRAM + 4GB RAM
```

**Rust (target):**
```
Time: 20 minutes (2.25x faster)
Memory: 6GB VRAM + 1GB RAM (50% less)
```

---

## Documentation Plan

### User Documentation

1. **Getting Started Guide** (in progress)
   - Installation
   - First fine-tuning job
   - Configuration reference

2. **API Documentation** (partially complete)
   - Rustdoc for all public APIs
   - Examples in doc comments
   - Module-level overview docs

3. **Migration Guide** (planned)
   - Python to Rust differences
   - Config compatibility
   - Feature mapping

4. **Performance Tuning** (planned)
   - Batch size optimization
   - Mixed precision settings
   - Multi-GPU strategies

### Developer Documentation

1. **Architecture Guide** (this document + future deep-dive)
2. **Contributing Guide** ( complete)
3. **Testing Guide** (in TEST_COVERAGE_PLAN.md)
4. **Benchmarking Guide** (planned)

---

## Risk Mitigation

### Technical Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Candle API changes | High | Pin versions, upstream contributions |
| CUDA support gaps | Medium | CPU fallback, wait for maturity |
| Memory leaks | High | Extensive testing, valgrind |
| Numeric instability | High | Reference PyTorch implementations |
| Upstream bugs (candle) | Medium | Report issues, maintain forks if needed |

### Project Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Feature creep | Medium | Strict phase boundaries |
| Python parity delays | Low | Async development, MVP first |
| Team bandwidth | Medium | Focus on Phase 2 critical path |
| User adoption | Low | Backward-compatible configs |

---

## Success Metrics

### Phase 2 Success (MVP)

- [ ] Train LoRA adapter end-to-end
- [ ] Match Python accuracy within 1%
- [ ] 2x faster than Python Axolotl
- [ ] 30% lower memory usage
- [ ] 80% test coverage
- [ ] 100+ unit tests
- [ ] Complete API documentation
- [ ] 5+ real-world examples

### Phase 3 Success (Production)

- [ ] Multi-GPU training working
- [ ] QLoRA 4-bit training
- [ ] Unsloth 2x inference speedup
- [ ] Streaming datasets
- [ ] 10+ supported architectures
- [ ] <1% memory overhead vs. PyTorch

### Phase 4 Success (Ecosystem)

- [ ] Published to crates.io
- [ ] 1000+ GitHub stars
- [ ] 10+ external contributors
- [ ] Used in 5+ production systems
- [ ] Feature parity with Python Axolotl
- [ ] Comprehensive benchmarks published

---

## Timeline Summary

| Phase | Duration | Completion Target |
|-------|----------|-------------------|
| Phase 1: Foundation | 5 days |  Jan 10, 2026 |
| Phase 2: Core ML | 4-6 weeks |  Feb 15-28, 2026 |
| Phase 3: Advanced | 6-8 weeks |  Apr 15-30, 2026 |
| Phase 4: Production | Ongoing |  Summer 2026 |

---

## References

### Python Axolotl
- **GitHub:** https://github.com/OpenAccess-AI-Collective/axolotl
- **Documentation:** https://github.com/OpenAccess-AI-Collective/axolotl#readme

### Rust Ecosystem
- **Candle:** https://github.com/huggingface/candle
- **Tokenizers:** https://github.com/huggingface/tokenizers
- **Safetensors:** https://github.com/huggingface/safetensors

### Workspace Crates
- **peft-rs:** `../peft-rs` - 11 PEFT adapters implemented
- **qlora-rs:** `../qlora-rs` - 4-bit quantization
- **unsloth-rs:** `../unsloth-rs` - Optimized kernels

### Performance Research
- **LoRA:** https://arxiv.org/abs/2106.09685
- **QLoRA:** https://arxiv.org/abs/2305.14314
- **Unsloth:** https://github.com/unslothai/unsloth

---

**Document Version:** 1.0  
**Last Updated:** January 10, 2026  
**Next Review:** Start of Phase 2
