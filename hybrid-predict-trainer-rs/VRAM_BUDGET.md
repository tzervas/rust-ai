# VRAM Budget & OOM Prevention Guide

**GPU:** NVIDIA RTX 5080 (16GB GDDR7)
**Created:** 2026-02-06
**Purpose:** Prevent OOM errors during development and testing

---

## 🎯 Total Available VRAM

```
Total VRAM:              16,384 MB  (16 GB)
Operating System:        -  800 MB  (OS + drivers)
Desktop Environment:     -  400 MB  (GNOME/KDE/X11)
Browser (open):          -  300 MB  (Chrome/Firefox with tabs)
VS Code:                 -  150 MB  (editor + extensions)
Terminal:                -   50 MB  (multiple terminals)
Misc processes:          -  200 MB  (background services)
────────────────────────────────────────────
Safe Available:          14,484 MB  (~14.1 GB)

Reserved Safety Margin:  -  500 MB  (for OS spikes)
────────────────────────────────────────────
USABLE FOR TRAINING:     13,984 MB  (~13.7 GB)
```

---

## 📊 Current Workload VRAM Requirements

### Phase 1: Development (Current)

**Minimal testing with mock models:**
- Mock model parameters: ~10 MB
- Gradient buffers: ~10 MB
- Training state history: ~50 MB
- **Total:** ~70 MB

✅ **Safe:** 0.5% of available VRAM

### Phase 2-3: Burn Integration Testing

**SimpleMLP (MNIST):**
- Model parameters (784→128→10): ~100 MB
- Gradients: ~100 MB
- Optimizer state (Adam): ~200 MB (momentum + variance)
- Batch data (batch_size=64): ~20 MB
- Intermediate activations: ~50 MB
- RSSM dynamics model: ~150 MB
- Training state & metrics: ~50 MB
- **Total:** ~670 MB

✅ **Safe:** 4.8% of available VRAM

### Phase 4: Small Model Testing

**SmolLM2-135M:**
- Model parameters (FP16): ~270 MB
- Gradients (FP32): ~540 MB
- Optimizer state (Adam): ~1,080 MB
- Batch data (batch_size=8, seq_len=512): ~250 MB
- Activations & KV cache: ~800 MB
- RSSM dynamics (3 ensemble): ~500 MB
- Training state: ~100 MB
- **Total:** ~3,540 MB (~3.5 GB)

✅ **Safe:** 25% of available VRAM

### Phase 5: Medium Model Testing

**TinyLlama-1.1B:**
- Model parameters (FP16): ~2,200 MB
- Gradients (FP32): ~4,400 MB
- Optimizer state (Adam): ~8,800 MB
- Batch data (batch_size=4, seq_len=1024): ~500 MB
- Activations & KV cache: ~2,000 MB
- RSSM dynamics (3 ensemble): ~800 MB
- Training state: ~200 MB
- **Total:** ~18,900 MB (~18.5 GB)

⚠️ **UNSAFE:** Exceeds 13.7 GB budget by ~5 GB!

---

## 🛡️ OOM Prevention Strategies

### Strategy 1: Reduce Batch Size

**TinyLlama with batch_size=2:**
- Model + gradients + optimizer: ~15,400 MB
- Batch data (batch_size=2): ~250 MB
- Activations: ~1,000 MB
- RSSM + training state: ~1,000 MB
- **Total:** ~17,650 MB (~17.3 GB)

⚠️ **Still unsafe** - Need more reduction

**TinyLlama with batch_size=1:**
- Model + gradients + optimizer: ~15,400 MB
- Batch data (batch_size=1): ~125 MB
- Activations: ~500 MB
- RSSM + training state: ~1,000 MB
- **Total:** ~17,025 MB (~16.6 GB)

⚠️ **Still unsafe** - Need quantization or gradient checkpointing

### Strategy 2: Gradient Checkpointing

**TinyLlama with gradient checkpointing + batch_size=2:**
- Model + optimizer: ~15,400 MB
- Gradients (checkpointed): ~1,000 MB (vs ~4,400 MB)
- Batch data: ~250 MB
- Activations (recomputed): ~400 MB (vs ~2,000 MB)
- RSSM + training state: ~1,000 MB
- **Total:** ~18,050 MB (~17.6 GB)

⚠️ **Still marginal** - Close to limit

### Strategy 3: Mixed Precision (FP16/BF16)

**TinyLlama with FP16 + gradient checkpointing + batch_size=2:**
- Model parameters (FP16): ~2,200 MB
- Gradients (FP16, checkpointed): ~550 MB
- Optimizer state (FP32 master copy): ~4,400 MB
- Batch data: ~250 MB
- Activations (FP16, recomputed): ~200 MB
- RSSM + training state: ~1,000 MB
- **Total:** ~8,600 MB (~8.4 GB)

✅ **Safe:** 61% of available VRAM

### Strategy 4: Gradient Accumulation

**TinyLlama with FP16 + accumulation (effective batch_size=8):**
- Model + optimizer: ~6,600 MB
- Gradients (accumulated): ~550 MB
- Batch data (micro_batch=1): ~125 MB
- Activations: ~100 MB
- RSSM + training state: ~1,000 MB
- **Total:** ~8,375 MB (~8.2 GB)

✅ **Safe:** 59% of available VRAM
✅ **Bonus:** Effective batch size of 8 with micro-batches of 1

---

## 📋 Recommended VRAM Budgets by Phase

### Phase 2-3: Burn Integration (SimpleMLP)

```yaml
max_batch_size: 64
model_precision: FP32
gradient_checkpointing: false
expected_vram: 670 MB
safety_margin: 13 GB remaining
status: ✅ SAFE
```

### Phase 4: SmolLM2-135M

```yaml
max_batch_size: 16
model_precision: FP16
gradient_checkpointing: false
expected_vram: 4.2 GB
safety_margin: 9.5 GB remaining
status: ✅ SAFE
```

### Phase 5: TinyLlama-1.1B

```yaml
max_batch_size: 2
model_precision: FP16
gradient_checkpointing: true
gradient_accumulation_steps: 4  # Effective batch_size = 8
expected_vram: 8.4 GB
safety_margin: 5.3 GB remaining
status: ✅ SAFE
```

### Phase 6: Larger Models (3B-7B)

⚠️ **NOT FEASIBLE on RTX 5080 (16GB)**

Would require:
- **LLaMA-3B:** ~22 GB VRAM (with FP16 + checkpointing)
- **LLaMA-7B:** ~40 GB VRAM (with FP16 + checkpointing)

**Recommendation:** Use RTX 3090 Ti (24GB) when available for 3B+ models

---

## 🔍 Monitoring VRAM Usage

### Pre-Training Check

```bash
# Check current VRAM usage before starting training
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits

# Expected output: "800,16384" (800 MB used, 16384 MB total)
```

### During Training Monitor

```bash
# Watch VRAM usage in real-time (updates every 1 second)
watch -n 1 nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits

# Example output:
# 4200,16384,85  (4.2 GB used, 16.4 GB total, 85% GPU utilization)
```

### Log VRAM Usage to File

```bash
# Log VRAM usage during training run
nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader \
    --loop=5 \
    > vram_log_$(date +%Y%m%d_%H%M%S).csv &

# Stop logging: pkill -f "nvidia-smi.*loop"
```

---

## ⚠️ OOM Warning Signs

### Early Warning Indicators

1. **VRAM usage > 12 GB** - Close to limit, monitor closely
2. **VRAM usage > 13 GB** - Danger zone, likely to OOM soon
3. **VRAM usage > 14 GB** - Immediate OOM risk
4. **GPU utilization drops suddenly** - May indicate OOM recovery attempts

### CUDA OOM Error Messages

```
RuntimeError: CUDA out of memory. Tried to allocate X MB
RuntimeError: CUDA error: out of memory
cublas runtime error: out of memory
```

### Recovery Actions

If OOM occurs during training:

1. **Immediate:** Reduce batch_size by 50%
2. **If still failing:** Enable gradient checkpointing
3. **If still failing:** Switch to FP16/BF16 mixed precision
4. **If still failing:** Use gradient accumulation (micro-batches)
5. **Last resort:** Reduce model size or sequence length

---

## 🎯 Testing Protocol

### Before Each Training Run

```bash
# 1. Check available VRAM
nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits

# 2. Verify OS overhead is reasonable (< 2 GB)
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits

# 3. Close unnecessary applications
# - Browser tabs
# - Other GPU applications
# - Electron apps (VS Code, Slack, etc.) if needed

# 4. Start VRAM monitoring in separate terminal
watch -n 1 nvidia-smi
```

### During Training

- Monitor VRAM usage every 10 steps
- If usage > 12 GB, prepare to reduce batch size
- If usage growing linearly, check for memory leaks
- If OOM occurs, log the failure point and configuration

### After Training

```bash
# 1. Check peak VRAM usage from logs
grep "memory.used" vram_log_*.csv | sort -t',' -k2 -n | tail -1

# 2. Verify VRAM was released
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits

# 3. Document actual VRAM usage for future reference
```

---

## 📊 VRAM Budget Table (Quick Reference)

| Model | Precision | Batch Size | Checkpointing | Est. VRAM | Safe? |
|-------|-----------|------------|---------------|-----------|-------|
| SimpleMLP | FP32 | 64 | No | 0.7 GB | ✅ Yes |
| SmolLM2-135M | FP16 | 16 | No | 4.2 GB | ✅ Yes |
| SmolLM2-135M | FP16 | 32 | No | 5.8 GB | ✅ Yes |
| TinyLlama-1.1B | FP32 | 4 | No | 18.5 GB | ❌ OOM |
| TinyLlama-1.1B | FP16 | 4 | No | 12.8 GB | ✅ Yes |
| TinyLlama-1.1B | FP16 | 2 | Yes | 8.4 GB | ✅ Yes |
| TinyLlama-1.1B | FP16 | 1 | Yes | 7.9 GB | ✅ Yes |
| LLaMA-3B | FP16 | 1 | Yes | ~22 GB | ❌ OOM |
| LLaMA-7B | FP16 | 1 | Yes | ~40 GB | ❌ OOM |

---

## 🔧 Configuration Recommendations

### Burn Model Configuration

```rust
// Safe defaults for RTX 5080 (16GB)
let config = HybridTrainerConfig::builder()
    .warmup_steps(100)
    .full_steps(20)
    .max_predict_steps(50)
    // VRAM safety settings
    .max_batch_size(8)  // Conservative for medium models
    .gradient_checkpointing(true)  // Enable for models > 500M params
    .mixed_precision(true)  // Use FP16/BF16 for models > 100M params
    .build();
```

### Model-Specific Configurations

**SimpleMLP (MNIST):**
```rust
batch_size: 64,
precision: FP32,
gradient_checkpointing: false,
// Expected VRAM: ~670 MB
```

**SmolLM2-135M:**
```rust
batch_size: 16,
precision: FP16,
gradient_checkpointing: false,
seq_length: 512,
// Expected VRAM: ~4.2 GB
```

**TinyLlama-1.1B:**
```rust
batch_size: 2,
precision: FP16,
gradient_checkpointing: true,
gradient_accumulation_steps: 4,
seq_length: 1024,
// Expected VRAM: ~8.4 GB
```

---

## 🎓 Best Practices

### 1. Always Start Conservative

- Begin with small batch sizes (1-2)
- Monitor VRAM usage for first 10 steps
- Gradually increase batch size if headroom exists

### 2. Profile Before Scaling

```bash
# Test with minimal config first
cargo run --example burn_mlp_mnist --release -- --batch-size 1

# Check peak VRAM
# If < 8 GB used, try batch_size=2
# If < 6 GB used, try batch_size=4
# etc.
```

### 3. Document Actual Usage

Keep a log of actual VRAM usage per configuration:

```markdown
## VRAM Usage Log

- 2026-02-06: SimpleMLP, batch=64, FP32 → 720 MB peak
- 2026-02-07: SmolLM2, batch=16, FP16 → 4.1 GB peak
- 2026-02-08: TinyLlama, batch=2, FP16+ckpt → 8.2 GB peak
```

### 4. Always Have Fallback Plan

If primary config OOMs:
1. Try batch_size / 2
2. Try gradient_checkpointing = true
3. Try precision = FP16
4. Try gradient_accumulation

---

## 🚨 Critical Safety Rules

### Rule 1: Never Exceed 13.5 GB

Leave at least 2.5 GB for OS + safety margin

### Rule 2: Monitor First 100 Steps

VRAM usage should stabilize by step 100. If still growing, investigate memory leak.

### Rule 3: Test Before Long Runs

Always do a 10-step test run before starting multi-hour training.

### Rule 4: Log Everything

Keep VRAM logs for all training runs to identify optimal configurations.

### Rule 5: When in Doubt, Reduce

Better to train slower with smaller batches than to OOM and lose progress.

---

**Summary:** With careful configuration, RTX 5080 (16GB) can safely handle:
- ✅ SimpleMLP (MNIST) - Easy
- ✅ SmolLM2-135M - Comfortable
- ✅ TinyLlama-1.1B - Requires FP16 + checkpointing
- ❌ LLaMA-3B+ - Need RTX 3090 Ti (24GB)

**Next Steps:** Use this budget when implementing Burn integration in Phase 2!
