# Tritter 100M Training Plan

## Overview

This document outlines the plan for properly training the Tritter 100M model using:
- Real curated datasets (not random tokens)
- Hybrid predictive training with all phases operational
- Proper learning rate schedules and optimization
- Full PREDICT/CORRECT cycles with residual correction

## Current Status

### Working
- [x] Model architecture (100M/500M/1B configs)
- [x] Basic training loop with phases (Warmup, Full)
- [x] GPU acceleration via Candle CUDA backend
- [x] Training monitor TUI with metrics
- [x] Checkpoint saving and HuggingFace upload
- [x] MultiModalTokenizer (text, code, image, audio)
- [x] DataLoader, JsonlDataset, ParquetDataset infrastructure

### Not Working / Critical Gaps
- [ ] PREDICT phase (weight delta prediction returns empty)
- [ ] CORRECT phase (residual store never populated)
- [ ] Real dataset integration (progressive.rs uses random tokens)
- [ ] Dynamics model GRU training (only loss head trained, GRU frozen)
- [ ] Learning rate scheduling

---

## Phase 1: Dataset Infrastructure

### Datasets to Acquire

| Dataset | Size | Purpose | Priority |
|---------|------|---------|----------|
| **FineWeb-Edu 10B sample** | ~28GB, 10B tokens | General knowledge, education | **Primary** |
| **TinyStories** | ~2GB, 500M tokens | Quick validation, testing | Testing |
| **FineMath** | 21B tokens | Math reasoning | Secondary |
| **Stack-Edu** | Code (filtered) | Programming skills | Secondary |
| **SmolLM-Corpus** | Curated mix | Multi-stage training | Extended |

#### Target Data Mix (for your requirements)
Based on user specifications (natural language, code, math, reasoning, AI/ML):

| Domain | Proportion | Sources |
|--------|------------|---------|
| Natural language | 40% | FineWeb-Edu |
| Code (Python, Rust, Triton, TS) | 25% | Stack-Edu, StarCoder data |
| Math & reasoning | 15% | FineMath, MATH dataset |
| AI/ML & data science | 10% | ArXiv ML papers, textbooks |
| Logic & structured | 10% | GSM8K, logic puzzles |

#### Data on Homelab Server
**Note**: Some data already available on homelab data volume. Need to inventory and assess:
- Check what datasets are already downloaded
- Identify gaps requiring additional downloads
- Consider local caching strategy for HuggingFace datasets

### Tokenizer (Already Implemented)

Located in `tritter-model-rs/src/tokenizer.rs`:
- **MultiModalTokenizer** with 65536 vocabulary
- Modalities: Text, Code, Image (VQVAE), Audio (SpeechTokenizer)
- HuggingFace `tokenizers` 0.21 backend for BPE
- Batch encoding/decoding with padding

### Data Loading Pipeline (Already Implemented)

Located in `tritter-model-rs/src/data.rs`:
```
Dataset (HF/local) → JsonlDataset/ParquetDataset → DataLoader → collate_batch → TritterBatch
```

**Integration needed**: Connect to `progressive.rs` training loop (currently uses random tokens).

---

## Phase 2: Training Configuration

### Learning Rate Schedule (Research Complete)

**Recommended for 100M model**:

| Parameter | Value | Notes |
|-----------|-------|-------|
| Base LR | 6e-4 | GPT-2/124M standard |
| Warmup | 1% of steps | ~100-200 steps |
| Schedule | Cosine or WSD | WSD allows flexible training length |
| Min LR | 6e-5 (10% of peak) | Final decay target |

**WSD (Warmup-Stable-Decay) Schedule** - Recommended:
1. Warmup: 1% of steps (linear ramp to peak LR)
2. Stable: 80% of steps (constant at peak LR)
3. Decay: 19% of steps (cosine decay to min LR)

Benefits: Can resume from any stable-phase checkpoint, no commitment to training length upfront.

### AdamW Optimizer Settings

| Parameter | Value | Notes |
|-----------|-------|-------|
| beta1 | 0.9 | Momentum decay |
| beta2 | 0.95 | GPT-3 style (not default 0.999) |
| epsilon | 1e-8 | Numerical stability |
| weight_decay | 0.1 | Apply to weights only, not biases/LayerNorm |

### Batch Configuration

| Setting | Value | Tokens/Step |
|---------|-------|-------------|
| Batch size | 64-128 | Power of 2, multiples of 8 for FP16 |
| Sequence length | 1024-2048 | 2048 requires 4x attention memory |
| Gradient accumulation | 2-4 | For effective larger batches |
| Effective batch | 256-512 sequences | 256K-512K tokens/step |

### Chinchilla Scaling

| Model | Compute-Optimal Tokens | Extended Training |
|-------|------------------------|-------------------|
| 100M | 2B tokens (20x params) | 4-10B tokens |
| 500M | 10B tokens | 20-50B tokens |
| 1B | 20B tokens | 40-100B tokens |

**Target for 100M**: 2-4B tokens minimum, FineWeb-Edu 10B sample for extended training.

---

## Phase 3: Predictive Training Fixes (Critical)

### Architecture Analysis Results

**5 Critical Gaps Identified**:

1. **Weight Delta Prediction Missing** (Priority 1)
   - Location: `hybrid-predict-trainer-rs/src/dynamics.rs:503`
   - Problem: `predict_y_steps()` returns `WeightDelta::empty()`
   - Impact: Core speedup mechanism broken - no actual weight skipping
   - Fix: Implement weight delta prediction from GRU hidden state

2. **GRU Weights Not Updated** (Priority 2)
   - Location: `hybrid-predict-trainer-rs/src/dynamics.rs:572-586`
   - Problem: Only loss head is trained, GRU weights stay at random init
   - Impact: Dynamics model can't learn training patterns
   - Fix: Implement proper backprop through GRU weights

3. **Residual Extraction Not Wired** (Priority 3)
   - Location: `hybrid-predict-trainer-rs/src/lib.rs`
   - Problem: Trainer never extracts prediction residuals to store
   - Impact: Corrector has no data to work with
   - Fix: After predict phase, compute actual - predicted and store

4. **Stochastic Path Unused** (Priority 4)
   - Location: `hybrid-predict-trainer-rs/src/dynamics.rs:308-311`
   - Problem: RSSM stochastic component initialized but never sampled
   - Impact: No uncertainty estimation from stochastic sampling
   - Fix: Implement KL-regularized stochastic sampling

5. **State Encoder Missing** (Priority 5)
   - Location: `hybrid-predict-trainer-rs/src/dynamics.rs:297-315`
   - Problem: Uses raw tanh projection instead of learned encoder
   - Impact: Poor feature representation
   - Fix: Add learned linear projection layer

### PREDICT Phase Requirements

**What's Working**:
- PredictiveExecutor with statistics tracking
- Multi-step prediction via dynamics model (loss trajectory only)
- Confidence-based early termination
- Phase transition logic

**What Needs Fixing**:
- Dynamics model must return non-empty weight deltas
- Deltas should be applied to model weights before forward pass
- Prediction error should trigger correct phase appropriately

### CORRECT Phase Requirements

**What's Working**:
- ResidualCorrector with similarity-based weighting
- Linear correction model (32-dim features)
- 70% historical + 30% learned correction blend

**What Needs Fixing**:
- `weight_correction` field returns `None` (line 265)
- Gradient residuals defined but never used
- Trainer never populates residual store

### Phase Transition Tuning

| Parameter | Current | Recommended |
|-----------|---------|-------------|
| confidence_threshold | 0.30 | 0.50-0.70 (raise after fixes) |
| warmup_steps | 100 | 200-500 (more data for dynamics) |
| full_steps | 20 | 30-50 (more gradient observations) |
| max_predict_steps | 80 | 50-100 (adaptive based on confidence) |

---

## Phase 4: Training Execution

### Training Schedule

1. **Warmup**: 200 steps
   - Baseline statistics collection
   - Initial dynamics model training
   - Loss stabilization

2. **Full**: 30 steps
   - Full forward + backward passes
   - Gradient observations for dynamics model
   - GRU weight updates

3. **Predict**: 50-100 steps (adaptive)
   - Forward pass only (skip backward)
   - Weight deltas applied from dynamics model
   - Confidence monitoring

4. **Correct**: 10-20 steps
   - Validation with actual gradients
   - Residual computation and storage
   - Weight correction application
   - Dynamics model refinement

5. **Repeat**: Full → Predict → Correct cycle

### Expected Loss Trajectory

| Step | Expected Loss | Perplexity | Notes |
|------|---------------|------------|-------|
| 0 | ~10.5 | ~36,000 | Random init (vocab ~65K) |
| 100 | ~4.5 | ~90 | Initial learning |
| 1000 | ~3.5 | ~33 | Rapid descent |
| 5000 | ~2.5 | ~12 | Stabilizing |
| 10000 | ~2.0-2.5 | ~7-12 | Converged |

### Monitoring

- Loss curve tracking (should decrease, not plateau at 11.09)
- Gradient norm monitoring (should be stable, not exploding/vanishing)
- Phase transition logging (PREDICT/CORRECT should activate)
- GPU utilization optimization (target >90%)
- Prediction accuracy tracking

---

## Implementation Tasks

### Priority 1: Dataset Integration
1. [x] Tokenizer crate dependency (already in place: tokenizers 0.21)
2. [ ] Inventory homelab data volume for existing datasets
3. [ ] Download FineWeb-Edu 10B sample if not present
4. [ ] Integrate DataLoader into progressive.rs training loop
5. [ ] Replace random token generation with real data

### Priority 2: Optimizer & LR
6. [ ] Implement WSD learning rate scheduler
7. [ ] Configure AdamW with GPT-3 style betas (0.9, 0.95)
8. [ ] Add weight decay exclusions (biases, LayerNorm)

### Priority 3: Dynamics Model Fixes
9. [ ] Implement weight delta prediction in dynamics.rs
10. [ ] Add GRU weight training (not just loss head)
11. [ ] Wire up residual extraction after predict phase
12. [ ] Populate residual store from trainer

### Priority 4: Phase Transition
13. [ ] Implement proper CORRECT phase weight corrections
14. [ ] Tune confidence threshold with real data
15. [ ] Add adaptive prediction horizon

### Priority 5: Validation
16. [ ] Run full 100M training with FineWeb-Edu 10B
17. [ ] Verify PREDICT/CORRECT phases activate
18. [ ] Evaluate loss convergence (target <3.0)
19. [ ] Generate text samples for qualitative evaluation

---

## Success Criteria

- [ ] Loss converges to <3.0 (not stuck at ~11.09)
- [ ] PREDICT phase activates and skips backward passes
- [ ] CORRECT phase applies meaningful weight corrections
- [ ] GPU utilization >90%
- [ ] Training completes in ~20 hours for 2B tokens on RTX 5080
- [ ] Model produces coherent text samples

---

## Resources

### Download Commands

```bash
# FineWeb-Edu 10B sample (~28GB)
huggingface-cli download HuggingFaceFW/fineweb-edu \
  --include "sample/10BT/*" \
  --local-dir ./fineweb-edu-10B

# TinyStories for testing (~2GB)
huggingface-cli download roneneldan/TinyStories \
  --local-dir ./tinystories

# Check existing data on homelab
ls -la /path/to/homelab/data/volume/
```

### Key Files

| File | Purpose |
|------|---------|
| `tritter-model-rs/src/tokenizer.rs` | MultiModalTokenizer |
| `tritter-model-rs/src/data.rs` | DataLoader, datasets |
| `hybrid-predict-trainer-rs/src/dynamics.rs` | RSSMLite dynamics model |
| `hybrid-predict-trainer-rs/src/corrector.rs` | Residual correction |
| `training-tools/src/progressive.rs` | Training loop |

### References

- [FineWeb-Edu Dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
- [Chinchilla Scaling Laws](https://arxiv.org/abs/2203.15556)
- [WSD Learning Rate Schedule](https://arxiv.org/abs/2508.01483)
- [GPT-2 Reproduction (llm.c)](https://github.com/karpathy/llm.c)
