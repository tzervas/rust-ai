# Rust-AI Subsidiary Projects Index

This document indexes the subsidiary projects in the rust-ai workspace, providing an overview, git status, and functional parity status against their Python originals.

**See also**: [NEXT_PHASE_PLAN.md](NEXT_PHASE_PLAN.md) — Detailed implementation plan for single-GPU E2E fine-tuning

---

# Detailed Parity Analysis (January 2026)

## Summary Table

| Library | Parity | Status | Critical Gaps |
|---------|--------|--------|---------------|
| **peft-rs** | **100%** |  Production Ready | None (rsLoRA, LoftQ added) |
| **qlora-rs** | **85%** |  Training Ready | 3/2/1-bit quant (post-parity) |
| **axolotl-rs** | **65%** | 🟡 Adapter Integration | E2E validation needed |
| **unsloth-rs** | **40%** | 🟠 CPU Reference Only | CubeCL GPU kernels (deferred) |

## Recent Progress (Session)

| Task | Status | Notes |
|------|--------|-------|
|  Model wrapping with adapters | Complete | `AdapterLayers`, `create_adapter_layers()` |
|  Trainer adapter integration | Complete | safetensors checkpoint format |
|  E2E test suite | Complete | `tests/e2e_qlora.rs` (5 tests) |
|  Workspace patch alignment | Complete | Local peft-rs/qlora-rs for all deps |

## Next Steps Priority

**Goal**: Validate E2E fine-tuning on real model + dataset

| Priority | Task | Effort | Status |
|----------|------|--------|--------|
| 1 | ~~Wrap LLaMA layers with LoRA/QLoRA adapters~~ | ~~2 days~~ |  Done |
| 2 | ~~Connect trainer to adapter forward/backward~~ | ~~2 days~~ |  Done |
| 3 | ~~Implement adapter checkpoint save/load~~ | ~~1 day~~ |  Done |
| 4 | ~~E2E validation test~~ | ~~1-2 days~~ |  Done |
| 5 | Validate with TinyLLaMA on small Alpaca | 1 day | Next |
| 6 | Scale to LLaMA-7B with full Alpaca | 2 days | Pending |

**Total**: ~5-8 days to E2E correctness

---

## 1. peft-rs — 100% Parity 

**Python Original**: [HuggingFace PEFT](https://github.com/huggingface/peft) (20.5k , 290 contributors)

###  Implemented Features

| Feature | Python PEFT | peft-rs | Notes |
|---------|-------------|---------|-------|
| **LoRA** |  |  | Full implementation with scaling, dropout |
| **DoRA** |  |  | Weight-decomposed LoRA via `use_dora` flag |
| **AdaLoRA** |  |  | SVD-based adaptive rank with importance scores |
| **IA³** |  |  | Learned rescaling vectors for K/V/FFN |
| **LoHa** |  |  | Hadamard product decomposition |
| **LoKr** |  |  | Kronecker product decomposition |
| **OFT** |  |  | Orthogonal fine-tuning |
| **BOFT** |  |  | Butterfly orthogonal fine-tuning |
| **VeRA** |  |  | Frozen random matrices + trainable scaling |
| **Prefix Tuning** |  |  | Learnable prefix vectors |
| **Prompt Tuning** |  |  | Soft prompts |
| **Save/Load** |  |  | safetensors + HF-compatible JSON config |
| **Merge/Unmerge** |  |  | Full `Mergeable` trait implementation |
| **Multi-Adapter** |  |  | `AdapterRegistry` for managing multiple adapters |
| **Training Utils** |  |  | LR schedules, parameter counting, training state |
| **rsLoRA** |  |  | `use_rslora` flag for `alpha/sqrt(r)` scaling |
| **LoftQ Init** |  |  | `loftq_iterations` for quantization-aware init |

###  Missing Features (0%)

| Feature | Priority | Effort | Blocker? |
|---------|----------|--------|----------|
| **Quantization integration** (4-bit/8-bit base) | Medium | 3 days | No (use qlora-rs) |
| **EETQ/GPTQ backend support** | Low | 1 week | No |
| **Transformers `add_adapter` API** | Low | 2 days | No |

### Success Criteria for 100%
- [ ] Add `use_rslora: bool` to LoraConfig with scaling adjustment
- [ ] Implement LoftQ init (SVD + quantization-aware)
- [ ] Document integration path with qlora-rs for quantized base

---

## 2. qlora-rs — 85% Parity 

**Python Original**: [QLoRA](https://github.com/artidoro/qlora) (10.8k ) + [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)

###  Implemented Features

| Feature | Python QLoRA | qlora-rs | Notes |
|---------|--------------|----------|-------|
| **NF4 Quantization** |  |  | 16-level optimal normal float |
| **Double Quantization** |  |  | Quantize the scales themselves |
| **Per-Tensor Blocking** |  |  | Configurable block size (default 64) |
| **Per-Channel Strategy** |  |  | Alternative quantization strategy |
| **QuantizedLinear** |  |  | Fused quantized base + LoRA adapter |
| **Dequantization** |  |  | Both single and double quant paths |
| **Inference Forward** |  |  | Working 2D/3D input handling |
| **GGUF Export** |  |  | Export for llama.cpp deployment |
| **peft-rs Integration** |  |  | Uses `LoraLayer` from peft-rs |
| **Training Loop** |  |  | Full forward/backward with gradient accumulation |
| **PagedAdamW Optimizer** |  |  | CPU offloading for optimizer states |
| **Gradient Accumulation** |  |  | Configurable accumulation steps |

###  Missing Features (15%)

| Feature | Priority | Effort | Blocker? |
|---------|----------|--------|----------|
| **Gradient Checkpointing** | High | 3 days | No |
| **FP4 Quantization** | Low | 2 days | No |
| **8-bit Quantization** | Medium | 3 days | No |
| **3/2/1-bit Quantization** | Medium | 1 week | Extended |
| **Mixed Precision** | Medium | 3 days | No |
| **Multi-GPU Sharding** | High | 2 weeks | Deferred |

### Current State
```
 quantize_nf4() - Working with tests
 dequantize_nf4() - Both single/double quant
 QuantizedLinear::forward() - Inference works
 QLoraTrainer::training_step() - Full backward pass
 QLoraTrainer::training_step_lm() - Cross-entropy loss for LM
 PagedAdamW - CPU paging for optimizer states
 Gradient accumulation - Configurable steps
```

### Success Criteria for 100%
- [x] Implement full training loop with backward pass
- [x] Add paged_adamw_32bit optimizer (CPU paging for optimizer states)
- [ ] Validate: Fine-tune LLaMA-7B on Alpaca with 4-bit base
- [ ] Match Python QLoRA memory usage (<24GB for 7B model)
- [ ] Add gradient checkpointing for memory efficiency

---

## 3. axolotl-rs — 65% Parity 🟡

**Python Original**: [Axolotl](https://github.com/axolotl-ai-cloud/axolotl) (11.1k , 226 contributors)

###  Implemented Features

| Feature | Python Axolotl | axolotl-rs | Notes |
|---------|----------------|------------|-------|
| **YAML Config Parsing** |  |  | Full `AxolotlConfig` with serde |
| **Config Validation** |  |  | Comprehensive validation |
| **Presets** |  |  | llama2-7b, mistral-7b, etc. |
| **CLI Interface** |  |  | `train`, `validate`, `merge` commands |
| **Dataset Config** |  |  | Alpaca, ShareGPT, Completion formats |
| **Training Config** |  |  | Epochs, LR, batch size, etc. |
| **LoRA Settings** |  |  | r, alpha, dropout, target_modules |
| **QLoRA Settings** |  |  | bits, quant_type, double_quant |
| **Optimizer Config** |  |  | AdamW wrapper |
| **LR Schedulers** |  |  | Linear, cosine, constant |
| **Progress Bar** |  |  | indicatif integration |
| **Model Loading** |  |  | LLaMA via candle-transformers |
| **Tokenizer Integration** |  |  | Via candle-transformers |
| **Training Forward/Backward** |  |  | Cross-entropy loss |
| **peft-rs Integration** |  |  | AdapterLayers with LoraLayer |
| **qlora-rs Integration** |  |  | QuantizedLinear support |
| **Adapter Checkpointing** |  |  | safetensors format (HF compatible) |

###  Missing Features (35%)

| Feature | Priority | Effort | Blocker? |
|---------|----------|--------|----------|
| **Adapter Merging** | High | 2 days | No |
| **DPO/ORPO/GRPO** | Medium | 2 weeks | No |
| **Multi-GPU (FSDP)** | High | 3 weeks | Deferred |
| **DeepSpeed Integration** | Medium | 2 weeks | No |
| **Multipacking** | Medium | 1 week | No |
| **Flash Attention** | High | via unsloth-rs | Dependency |
| **Evaluation Loop** | Medium | 3 days | No |
| **HuggingFace Hub** | Low | 1 week | No |

### Current State
```
 AxolotlConfig::from_file() - Working
 Trainer::new() - Creates trainer with adapters
 Dataset::load() - Alpaca format
 load_model() - LLaMA via candle-transformers
 create_adapter_layers() - LoRA/QLoRA layers
 save_adapter_weights() - safetensors format
 adapter_config.json - HuggingFace PEFT compatible
 Trainer::train() - Cross-entropy loss with adapters
🟡 E2E validation - Needs real model testing
```
 load_model() - LLaMA via candle-transformers
 Trainer::train() - Working with cross-entropy loss
🟡 Checkpointing - Needs implementation
```

### Dependency Chain
```
axolotl-rs
├── peft-rs (100% OK) - Adapter management
├── qlora-rs (85% OK) - Quantization + Training
│   └── peft-rs - LoRA adapters
└── unsloth-rs (40% IN PROGRESS) - GPU kernels
    └── CubeCL - GPU backend
```

### Success Criteria for 100%
- [x] Load LLaMA/Mistral models via Candle
- [x] End-to-end training on Alpaca dataset
- [ ] LoRA/QLoRA fine-tuning working with adapters
- [ ] Checkpoint save/resume functional
- [ ] Multi-GPU via Candle distributed (deferred)

---

## 4. unsloth-rs — 40% Parity

**Python Original**: [Unsloth](https://github.com/unslothai/unsloth) (50.7k , 144 contributors)

###  Implemented Features

| Feature | Python Unsloth | unsloth-rs | Notes |
|---------|----------------|------------|-------|
| **Multi-Head Attention** |  |  | CPU reference, GQA support |
| **RoPE Embeddings** |  |  | Pre-computed cos/sin cache |
| **RMSNorm** |  |  | Numerically stable |
| **SwiGLU** |  |  | Gate + up + down projections |
| **Memory Estimation** |  |  | VRAM tracking utilities |
| **Device Dispatch** |  | 🟡 | CPU/CUDA detection, fallback |
| **Flash Attention Config** |  |  | `FusedAttentionConfig` |

###  Missing Features (60%)

| Feature | Priority | Effort | Blocker? |
|---------|----------|--------|----------|
| **Triton → CubeCL Flash Attention** | 🔴 Critical | 4 weeks | Yes |
| **Fused RoPE GPU Kernel** | High | 1 week | Performance |
| **Fused RMSNorm GPU Kernel** | High | 1 week | Performance |
| **Fused SwiGLU GPU Kernel** | High | 1 week | Performance |
| **Gradient Checkpointing** | High | 3 days | Memory |
| **2x Training Speedup** | 🔴 Critical | - | Goal |
| **70% VRAM Reduction** | 🔴 Critical | - | Goal |
| **GRPO/GSPO RL** | Low | 3 weeks | No |
| **TTS Support** | Low | N/A | No |
| **Vision Support** | Low | N/A | No |

### CubeCL Flash Attention Status
```
Phase 1:  Module structure, fallback implementation
Phase 2:  Basic CubeCL kernel (in progress)
Phase 3: ⏳ Tiled algorithm with online softmax
Phase 4: ⏳ Memory optimization and fusion
Phase 5: ⏳ Performance tuning
```

### Current Performance
| Operation | Python Unsloth | unsloth-rs | Gap |
|-----------|----------------|------------|-----|
| Attention | 2x faster | 1x (CPU fallback) |  |
| RoPE | 3x faster | 1x (CPU) |  |
| RMSNorm | 2x faster | 1x (CPU) |  |
| Memory | 70% less | Standard |  |

### Success Criteria for 100%
- [ ] CubeCL Flash Attention matching Triton speed
- [ ] Fused GPU kernels for RoPE/RMSNorm/SwiGLU
- [ ] Achieve 2x training speedup vs naive
- [ ] Achieve 70% VRAM reduction
- [ ] Pass GPU kernel validation tests

---

# Dependency Chain Analysis

```
                    ┌─────────────────┐
                    │   axolotl-rs    │ ← Top-level orchestrator
                    │   (35% parity)  │
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│   peft-rs     │   │  qlora-rs     │   │  unsloth-rs   │
│  (95% parity) │   │ (70% parity)  │   │ (40% parity)  │
│   READY       │   │  INFERENCE    │   │  CPU ONLY     │
└───────────────┘   └───────┬───────┘   └───────┬───────┘
                            │                   │
                            ▼                   ▼
                    ┌───────────────┐   ┌───────────────┐
                    │   peft-rs     │   │    CubeCL     │
                    │ (LoRA layer)  │   │ (GPU backend) │
                    └───────────────┘   └───────────────┘
```

### What Blocks What

1. **qlora-rs training** ← Blocked by lack of paged optimizers
2. **axolotl-rs training** ← Blocked by qlora-rs training + model loading
3. **unsloth-rs performance** ← Blocked by CubeCL Flash Attention
4. **Full stack parity** ← Blocked by unsloth-rs GPU kernels

---

# Prioritized Implementation Plan

## Phase 1: Core Training (4-6 weeks)
1. **qlora-rs: Paged AdamW** - Enable training on consumer GPUs
2. **qlora-rs: Training loop** - Backward pass with gradient accumulation
3. **axolotl-rs: Model loading** - Candle-based LLaMA/Mistral loading

## Phase 2: Integration (4-6 weeks)
4. **axolotl-rs: End-to-end training** - Full pipeline working
5. **axolotl-rs: Checkpointing** - Save/resume functionality
6. **peft-rs: rsLoRA + LoftQ** - Complete adapter coverage

## Phase 3: Performance (8-12 weeks)
7. **unsloth-rs: CubeCL Flash Attention** - Core performance win
8. **unsloth-rs: Fused GPU kernels** - RoPE/RMSNorm/SwiGLU
9. **axolotl-rs: Multi-GPU** - Distributed training via Candle

## Phase 4: Advanced (Future)
10. **axolotl-rs: DPO/ORPO/GRPO** - Preference optimization
11. **unsloth-rs: RL/Vision/TTS** - Extended modalities
12. **Full ecosystem** - HuggingFace Hub integration

---

# Success Metrics

| Milestone | Metric | Target |
|-----------|--------|--------|
| **M1: Training Works** | Train LLaMA-7B LoRA on Alpaca | Loss converges |
| **M2: QLoRA Works** | Train LLaMA-7B 4-bit on 24GB GPU | <24GB VRAM |
| **M3: Performance** | Training throughput | >1.5x vs naive |
| **M4: Memory** | VRAM usage | <50% vs baseline |
| **M5: Full Parity** | Feature coverage | 100% core features |

---

*Analysis generated: January 15, 2026*
*Based on: Source code review + Python repository research*

---

## Quick Reference (Original Status)

## axolotl-rs
- **Overview**: Rust port of [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl), a YAML-driven fine-tuning toolkit for LLMs. Supports full fine-tuning, LoRA/QLoRA, DPO/ORPO/GRPO, multi-GPU, multipacking, Flash Attention.
- **Git Status**: Branch: testing (clean). Branches: Local (bench/config-performance, ci/github-actions, dev, docs/api-doctests, feat/phase-2-core-training, fix/workspace-setup, main, test/trainer-module, testing); Remote (origin/main, origin/testing, etc.). Remotes: origin (https://github.com/tzervas/axolotl-rs). Recent: test fixes, trainer integration, model loading.
- **PRs/Issues**: 0 open PRs, 0 open issues.
- **Parity Status**: ~35% (scaffolding ready, training not functional). Gaps: model loading, training loops, checkpoints, adapters, multi-GPU.

## paste-fork
- **Overview**: Fork of [dtolnay/paste](https://github.com/dtolnay/paste), a Rust macro library for token pasting. Used as dependency in rust-ai (e.g., qlora-rs).
- **Git Status**: Branch: master (clean). Branches: Local (master); Remote (origin/master). Remotes: origin (https://github.com/dtolnay/paste.git). Recent: unmaintained note, lint fixes, releases.
- **PRs/Issues**: N/A (upstream tracking).
- **Parity Status**: Fully functional; no Python original.

## peft-rs
- **Overview**: Rust port of [PEFT](https://github.com/huggingface/peft), modular PEFT adapters for LLM fine-tuning. Supports LoRA, DoRA, AdaLoRA, IA³, LoHa, LoKr, OFT, BOFT, VeRA, Prefix Tuning, Prompt Tuning, multi-adapter, CUDA.
- **Git Status**: Branch: main (clean). Branches: Local (dev, main, testing, working/quality-clippy-fixes); Remote (origin/main, origin/dev, etc.). Remotes: origin (https://github.com/tzervas/peft-rs). Recent: v0.4.0 release, adapter additions.
- **PRs/Issues**: 0 open PRs, 0 open issues.
- **Parity Status**: ~95% functional parity. Missing: rsLoRA, LoftQ init.

## qlora-rs
- **Overview**: Rust implementation of QLoRA from [QLoRA paper](https://github.com/artidoro/qlora), 4-bit quantization for LLMs. Supports NF4, double quantization, QLoRA fine-tuning.
- **Git Status**: Branch: main (clean, untracked: .github/skills/qlora-quantization-impl/# Code Citations.md). Branches: Local (dev, feature/phase1-dual-export-infrastructure, etc.); Remote (matching). Remotes: origin (https://github.com/tzervas/qlora-rs). Recent: updates, security fixes, merges.
- **PRs/Issues**: 0 open PRs, 0 open issues.
- **Parity Status**: ~70% (quantization/inference ready), lacks training. Gaps: training loop, paged optimizers.

## unsloth-rs
- **Overview**: Rust port of [Unsloth](https://github.com/unslothai/unsloth), optimized transformer kernels for LLM fine-tuning. 2x faster training, 70% less VRAM via Triton kernels. Supports full fine-tuning, LoRA/QLoRA, RL, TTS, vision.
- **Git Status**: Branch: feature/gap-resolution-and-ci (clean, untracked: .vscode/). Branches: Local (chore/test-infrastructure, dev, experimental, etc.); Remote (matching). Remotes: origin (https://github.com/tzervas/unsloth-rs). Recent: PR updates, fixes, GPU profiling.
- **PRs/Issues**: 12 open issues (GPU kernel validation).
- **Parity Status**: ~40% (CPU reference implementations). Gaps: CubeCL Flash Attention, GPU kernels, RL/TTS/vision.