# E2E Fine-tuning Validation Research: TinyLLaMA

**Date:** January 15, 2026  
**Status:** Research Complete  
**Purpose:** Document requirements for validating axolotl-rs E2E fine-tuning with small models

---

## Executive Summary

This document outlines the requirements for validating axolotl-rs end-to-end fine-tuning capabilities using small LLaMA-compatible models. The validation will prove the complete training pipeline works before scaling to larger models like LLaMA-7B.

---

## 1. Model Recommendations

### Primary Recommendation: SmolLM2-135M  (Best for Development)

| Property | Value |
|----------|-------|
| **HuggingFace Path** | `HuggingFaceTB/SmolLM2-135M` |
| **Parameters** | 135M |
| **Architecture** | LlamaForCausalLM |
| **Hidden Size** | 576 |
| **Layers** | 30 |
| **Attention Heads** | 9 |
| **KV Heads** | 3 (GQA) |
| **Intermediate Size** | 1,536 |
| **Vocab Size** | 49,152 |
| **Max Position** | 8,192 |
| **RoPE Theta** | 100,000 |
| **Format** | safetensors (BF16) |
| **File Size** | ~270 MB |
| **VRAM (FP16)** | ~300 MB |
| **VRAM (QLoRA 4-bit)** | ~150 MB |
| **License** | Apache 2.0 |

**Why SmolLM2-135M:**
- Smallest viable LLaMA-architecture model
- Same `LlamaForCausalLM` architecture as larger LLaMA models
- Full GQA support (tests KV head handling)
- safetensors format (our default)
- Runs on CPU for development
- Direct support in candle-transformers (same Llama loader)

### Secondary Recommendation: TinyLlama-1.1B (Production Validation)

| Property | Value |
|----------|-------|
| **HuggingFace Path** | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| **Parameters** | 1.1B |
| **Architecture** | LlamaForCausalLM |
| **Hidden Size** | 2,048 |
| **Layers** | 22 |
| **Attention Heads** | 32 |
| **KV Heads** | 4 (GQA) |
| **Intermediate Size** | 5,632 |
| **Vocab Size** | 32,000 |
| **Max Position** | 2,048 |
| **RoPE Theta** | 10,000 |
| **Format** | safetensors (BF16) |
| **File Size** | ~2.2 GB |
| **VRAM (FP16)** | ~2.5 GB |
| **VRAM (QLoRA 4-bit)** | ~1.2 GB |
| **License** | Apache 2.0 |

**Why TinyLlama-1.1B:**
- Standard 32K vocabulary (same as LLaMA-2)
- Well-tested, popular model (1.2M+ monthly downloads)
- RoPE theta=10000 (same as LLaMA-2)
- Good middle-ground before LLaMA-7B

### Other Viable Options

| Model | Params | Hidden | Layers | VRAM (4-bit) |
|-------|--------|--------|--------|--------------|
| SmolLM2-360M | 360M | 960 | 32 | ~250 MB |
| SmolLM2-1.7B | 1.7B | 2,048 | 24 | ~1.5 GB |

---

## 2. candle-transformers Support Verification 

**Status:** Fully Supported

From candle-transformers analysis:
- `candle_transformers::models::llama` - Full LLaMA/LLaMA-2 support
- `candle_transformers::models::quantized_llama` - Quantized GGUF/GGML support
- Explicit TinyLlama support in candle examples (see `candle-examples/examples/llama/main.rs`)

```rust
// candle llama example shows TinyLlama support
Which::TinyLlama1_1BChat => "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
// Single safetensors file (no sharding)
```

The `LlamaConfig` in candle supports all TinyLlama/SmolLM2 parameters:
- Variable KV heads (GQA)
- Different vocab sizes
- Custom RoPE theta values
- BF16/FP16/FP32 dtypes

---

## 3. How to Download/Cache Models

### Method 1: huggingface-cli (Recommended)

```bash
# Install huggingface CLI
pip install huggingface_hub

# Download SmolLM2-135M
huggingface-cli download HuggingFaceTB/SmolLM2-135M --local-dir ~/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-135M

# Download TinyLlama-1.1B
huggingface-cli download TinyLlama/TinyLlama-1.1B-Chat-v1.0 --local-dir ~/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0
```

### Method 2: Python Script

```python
from huggingface_hub import snapshot_download

# SmolLM2-135M
snapshot_download("HuggingFaceTB/SmolLM2-135M")

# TinyLlama-1.1B  
snapshot_download("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
```

### Method 3: Git LFS (Alternative)

```bash
git lfs install
git clone https://huggingface.co/HuggingFaceTB/SmolLM2-135M
```

### Cache Location

axolotl-rs looks for models in:
1. Local path (if absolute path provided)
2. `$HF_HOME/hub/models--{org}--{model}/`
3. `~/.cache/huggingface/hub/models--{org}--{model}/`

### Required Files

| File | Purpose |
|------|---------|
| `config.json` | Model architecture config |
| `model.safetensors` | Model weights (preferred) |
| `tokenizer.json` | Tokenizer vocabulary |
| `tokenizer_config.json` | Tokenizer settings |

---

## 4. Dataset Recommendations

### Primary: tatsu-lab/alpaca (Standard)

| Property | Value |
|----------|-------|
| **HuggingFace Path** | `tatsu-lab/alpaca` |
| **Size** | 52,002 examples |
| **Format** | JSONL with instruction/input/output fields |
| **License** | CC BY-NC 4.0 |
| **Download Size** | ~24 MB |

**Download:**
```bash
# Using datasets library
pip install datasets
python -c "from datasets import load_dataset; load_dataset('tatsu-lab/alpaca')"

# Or direct download
wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001.parquet
```

### Validation Subsets (for Testing)

| Subset Size | Purpose | Command |
|-------------|---------|---------|
| 10 samples | Unit tests | Built into e2e_qlora.rs |
| 100 samples | CI smoke tests | Built into e2e_qlora.rs |
| 1,000 samples | Local validation | First 1K from alpaca |
| 5,000 samples | Loss convergence test | First 5K from alpaca |

### Creating Small Test Dataset

```python
from datasets import load_dataset

# Load and slice
ds = load_dataset("tatsu-lab/alpaca", split="train[:1000]")

# Save as JSONL
ds.to_json("alpaca_1k.jsonl")
```

### Alternative Datasets

| Dataset | Size | Notes |
|---------|------|-------|
| `yahma/alpaca-cleaned` | 51K | Cleaned version of Alpaca |
| `databricks/databricks-dolly-15k` | 15K | Open license (CC BY-SA) |
| `OpenAssistant/oasst1` | 161K | Multi-turn conversations |

---

## 5. VRAM Requirements

### SmolLM2-135M (Development)

| Configuration | VRAM | Notes |
|---------------|------|-------|
| FP32 Inference | ~600 MB | CPU viable |
| FP16 Inference | ~300 MB | CPU viable |
| BF16 Training (full) | ~1 GB | Includes gradients |
| LoRA Training (r=8) | ~400 MB | Trainable params only |
| QLoRA 4-bit (r=8) | ~200 MB | Minimum viable |

### TinyLlama-1.1B (Validation)

| Configuration | VRAM | Notes |
|---------------|------|-------|
| FP16 Inference | ~2.5 GB | RTX 3060+ |
| BF16 Training (full) | ~8 GB | RTX 3080+ |
| LoRA Training (r=8) | ~4 GB | RTX 3070+ |
| QLoRA 4-bit (r=8) | ~1.5 GB | GTX 1660+ |
| QLoRA 4-bit (r=16) | ~2 GB | RTX 3060+ |

### Batch Size Impact (TinyLlama QLoRA)

| Batch Size | Seq Length | VRAM |
|------------|------------|------|
| 1 | 256 | ~1.5 GB |
| 2 | 256 | ~2.0 GB |
| 4 | 256 | ~3.0 GB |
| 1 | 512 | ~2.0 GB |
| 2 | 512 | ~3.0 GB |

---

## 6. Code Changes Required in axolotl-rs

### 6.1 Model Configuration Updates

The current `create_adapter_layers` in [model.rs](src/model.rs#L249-L357) uses hardcoded values:

```rust
// Current (hardcoded)
let hidden_size = 4096; // TODO: get from model config
for layer_idx in 0..32 { // TODO: get num_layers from model config
```

**Required Change:**
```rust
// Read from LlamaConfig
let llama_config = load_llama_config(model_path)?;
let hidden_size = llama_config.hidden_size;
let num_layers = llama_config.num_hidden_layers;
```

### 6.2 Dynamic Layer Configuration

Create a model info struct to pass configuration:

```rust
pub struct ModelInfo {
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_attention_heads: usize,
    pub num_kv_heads: usize,
    pub intermediate_size: usize,
}

impl ModelInfo {
    pub fn from_config(config: &LlamaConfig) -> Self {
        Self {
            hidden_size: config.hidden_size,
            num_layers: config.num_hidden_layers,
            num_attention_heads: config.num_attention_heads,
            num_kv_heads: config.num_key_value_heads.unwrap_or(config.num_attention_heads),
            intermediate_size: config.intermediate_size,
        }
    }
}
```

### 6.3 Target Module Dimension Mapping

Different target modules have different dimensions:

```rust
fn get_target_dims(target: &str, info: &ModelInfo) -> (usize, usize) {
    match target {
        // Attention projections
        "q_proj" => (info.hidden_size, info.hidden_size),
        "k_proj" => (info.hidden_size, info.hidden_size * info.num_kv_heads / info.num_attention_heads),
        "v_proj" => (info.hidden_size, info.hidden_size * info.num_kv_heads / info.num_attention_heads),
        "o_proj" => (info.hidden_size, info.hidden_size),
        // MLP projections
        "gate_proj" | "up_proj" => (info.hidden_size, info.intermediate_size),
        "down_proj" => (info.intermediate_size, info.hidden_size),
        _ => (info.hidden_size, info.hidden_size),
    }
}
```

### 6.4 Model Download Implementation

The `download_model` function needs implementation:

```rust
#[cfg(feature = "download")]
pub async fn download_model(model_id: &str, cache_dir: &str) -> Result<PathBuf> {
    use hf_hub::api::sync::Api;
    
    let api = Api::new()?;
    let repo = api.model(model_id.to_string());
    
    // Download required files
    let config = repo.get("config.json")?;
    let tokenizer = repo.get("tokenizer.json")?;
    let weights = repo.get("model.safetensors")?;
    
    // Return path to downloaded model
    Ok(weights.parent().unwrap().to_path_buf())
}
```

### 6.5 Cargo.toml Updates

Add `hf-hub` for model downloads:

```toml
[dependencies]
hf-hub = { version = "0.3", optional = true }

[features]
download = ["reqwest", "hf-hub"]
```

---

## 7. Validation Test Plan

### Phase 1: Unit Tests (No GPU)

```rust
#[test]
fn test_smollm2_config_parsing() {
    // Verify SmolLM2-135M config.json parses correctly
    let config = load_llama_config("HuggingFaceTB/SmolLM2-135M");
    assert_eq!(config.hidden_size, 576);
    assert_eq!(config.num_hidden_layers, 30);
}

#[test]
fn test_adapter_layer_dimensions() {
    // Verify LoRA layers created with correct dimensions for SmolLM2
    let info = ModelInfo::from_config(&smollm2_config);
    let (in_dim, out_dim) = get_target_dims("q_proj", &info);
    assert_eq!(in_dim, 576);
    assert_eq!(out_dim, 576);
}
```

### Phase 2: CPU Smoke Tests (SmolLM2-135M)

```rust
#[test]
#[ignore] // Run with: cargo test --ignored -- --test-threads=1
fn test_smollm2_forward_pass() {
    // Load SmolLM2-135M and verify forward pass works
    let model = load_model(&smollm2_config, &Device::Cpu)?;
    let input = tokenize("Hello world");
    let output = model.forward(&input)?;
    assert!(output.shape().dims()[0] > 0);
}

#[test]
#[ignore]
fn test_smollm2_lora_training_step() {
    // Verify single training step completes
    let mut trainer = Trainer::new(smollm2_qlora_config)?;
    let loss = trainer.training_step(&mini_batch)?;
    assert!(loss.is_finite());
}
```

### Phase 3: GPU Validation (TinyLlama-1.1B)

```rust
#[test]
#[ignore]
#[cfg(feature = "cuda")]
fn test_tinyllama_qlora_loss_decreases() {
    // Train for 100 steps, verify loss decreases
    let config = AxolotlConfig {
        base_model: "TinyLlama/TinyLlama-1.1B-Chat-v1.0".into(),
        adapter: AdapterType::Qlora,
        // ...
    };
    
    let mut trainer = Trainer::new(config)?;
    let initial_loss = trainer.training_step(&batch)?;
    
    for _ in 0..100 {
        trainer.training_step(&batch)?;
    }
    
    let final_loss = trainer.current_loss();
    assert!(final_loss < initial_loss * 0.9); // At least 10% reduction
}
```

### Phase 4: Full E2E Test

```rust
#[test]
#[ignore]
#[cfg(feature = "cuda")]
fn test_e2e_qlora_finetuning() {
    // Full training run with checkpoint saving
    let config = AxolotlConfig::from_file("test_configs/tinyllama_qlora.yaml")?;
    let mut trainer = Trainer::new(config)?;
    
    trainer.train()?;
    
    // Verify checkpoint saved
    assert!(Path::new("outputs/adapter_model.safetensors").exists());
    assert!(Path::new("outputs/adapter_config.json").exists());
    
    // Verify checkpoint loadable
    let loaded = LoadedModel::load_adapter_weights("outputs/")?;
    assert!(loaded.adapter_layers.is_some());
}
```

---

## 8. Example Configuration Files

### SmolLM2-135M QLoRA Config (Development)

```yaml
# configs/smollm2_qlora_dev.yaml
base_model: "HuggingFaceTB/SmolLM2-135M"
adapter: qlora
output_dir: "./outputs/smollm2-qlora"

lora:
  r: 8
  alpha: 16
  dropout: 0.0
  target_modules:
    - q_proj
    - v_proj

quantization:
  bits: 4
  quant_type: nf4
  double_quant: true
  block_size: 64

dataset:
  path: "./data/alpaca_1k.jsonl"
  type: alpaca
  max_length: 256
  train_split: 0.9

training:
  epochs: 1
  batch_size: 4
  learning_rate: 0.0002
  weight_decay: 0.0
  logging_steps: 10
  save_steps: 100
  warmup_ratio: 0.1

seed: 42
```

### TinyLlama-1.1B QLoRA Config (Validation)

```yaml
# configs/tinyllama_qlora_val.yaml
base_model: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
adapter: qlora
output_dir: "./outputs/tinyllama-qlora"

lora:
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj

quantization:
  bits: 4
  quant_type: nf4
  double_quant: true
  block_size: 64

dataset:
  path: "./data/alpaca_5k.jsonl"
  type: alpaca
  max_length: 512
  train_split: 0.95

training:
  epochs: 3
  batch_size: 2
  learning_rate: 0.0001
  weight_decay: 0.01
  logging_steps: 50
  save_steps: 500
  warmup_ratio: 0.1

seed: 42
```

---

## 9. Success Criteria

### Minimum Viable Validation

1.  Model loads successfully from HF cache
2.  Tokenizer loads and encodes text correctly  
3.  Forward pass produces valid logits
4.  LoRA adapter layers created with correct dimensions
5.  Single training step completes without error
6.  Loss is finite and reasonable (< 10.0 for random init)
7.  Checkpoint saves in safetensors format
8.  Checkpoint loads successfully

### Full Validation

1.  All minimum criteria
2.  Loss decreases over 100+ steps (training signal)
3.  Gradient norms are reasonable (not exploding/vanishing)
4.  Learning rate scheduler works correctly
5.  Multiple epochs complete successfully
6.  Adapter config JSON is HuggingFace compatible
7.  Generated text is coherent (qualitative check)

---

## 10. Next Steps

1. **Implement model config reading** - Remove hardcoded dimensions
2. **Add hf-hub dependency** - Enable model downloads
3. **Create test configs** - SmolLM2 and TinyLlama YAML files
4. **Download test models** - Cache SmolLM2-135M locally
5. **Create test dataset** - 1K Alpaca subset
6. **Run CPU smoke tests** - SmolLM2 forward pass
7. **Run GPU validation** - TinyLlama training loop
8. **Document results** - Update this file with outcomes

---

## References

- [HuggingFace SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M)
- [HuggingFace TinyLlama-1.1B-Chat](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
- [tatsu-lab/alpaca Dataset](https://huggingface.co/datasets/tatsu-lab/alpaca)
- [candle-transformers LLaMA](https://github.com/huggingface/candle/tree/main/candle-transformers/src/models/llama.rs)
- [candle LLaMA example](https://github.com/huggingface/candle/tree/main/candle-examples/examples/llama)
