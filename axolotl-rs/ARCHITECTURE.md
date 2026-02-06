# Axolotl-RS Architecture

**Version:** 1.0.0
**Status:** Initial Release

---

## High-Level Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         axolotl-rs (Main Binary)                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   CLI        │  │   Config     │  │   Dataset    │          │
│  │  (clap)      │──│  (serde)     │──│  (4 formats) │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│          │                │                   │                 │
│          └────────────────┴───────────────────┘                 │
│                           │                                     │
│  ┌────────────────────────┴──────────────────────────┐          │
│  │              Training Coordinator                 │          │
│  │  (Model + Optimizer + Scheduler + Loop)           │          │
│  └────────────────────────────────────────────────────┘         │
│                           │                                     │
└───────────────────────────┼─────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
┌───────▼────────┐  ┌───────▼────────┐  ┌──────▼──────┐
│   peft-rs      │  │   qlora-rs     │  │ unsloth-rs  │
│  (LoRA, DoRA)  │  │ (4-bit quant)  │  │ (optimized) │
│  11 adapters   │  │  NF4, FP4      │  │  kernels    │
└───────┬────────┘  └───────┬────────┘  └──────┬──────┘
        │                   │                   │
        └───────────────────┴───────────────────┘
                            │
                   ┌────────▼────────┐
                   │     candle      │
                   │  (tensor ops)   │
                   └────────┬────────┘
                            │
                ┌───────────┴───────────┐
                │                       │
         ┌──────▼──────┐        ┌──────▼──────┐
         │ CPU Backend │        │ GPU Backend │
         │  (native)   │        │ (CUDA/Metal)│
         └─────────────┘        └─────────────┘
```

---

## Component Architecture

### 1. CLI Layer (`src/cli.rs`, `src/main.rs`)

**Responsibilities:**
- Parse command-line arguments with `clap`
- Route commands to appropriate handlers
- Setup logging/tracing infrastructure
- Handle user-facing errors gracefully

**Commands:**
```rust
pub enum Commands {
    Validate { config: PathBuf },          // Validate YAML config
    Train { config: PathBuf, resume: Option<String> },  // Train model
    Merge { config: PathBuf, output: PathBuf },         // Merge adapter
    Init { output: PathBuf, preset: Option<String> },   // Generate config
}
```

**Data Flow:**
```
CLI args → Command routing → Config loading → Operation execution
```

---

### 2. Configuration System (`src/config.rs`)

**Responsibilities:**
- Parse YAML configurations with `serde_yaml`
- Validate configuration correctness
- Provide preset templates
- Serialize/deserialize for checkpointing

**Key Structures:**

```rust
pub struct AxolotlConfig {
    pub base_model: String,                 // HF model ID
    pub adapter: AdapterType,               // None/LoRA/QLoRA
    pub lora: LoraSettings,                 // r, alpha, dropout, targets
    pub quantization: Option<QuantizationSettings>,  // bits, type
    pub dataset: DatasetConfig,             // path, format, splits
    pub training: TrainingConfig,           // epochs, batch, LR, scheduler
    pub output_dir: String,
    pub seed: u64,
}
```

**Validation Rules:**
- `base_model` must not be empty
- `dataset.path` must be specified
- `lora.r` must be > 0
- QLoRA requires `quantization` config
- File paths must be valid

**Presets:**
- `llama2-7b`: LLaMA-2 7B with QLoRA
- `mistral-7b`: Mistral 7B with QLoRA
- `phi3-mini`: Phi-3 Mini with LoRA (no quantization)

---

### 3. Dataset Pipeline (`src/dataset.rs`)

**Responsibilities:**
- Load datasets from JSONL files
- Support 4 formats: Alpaca, ShareGPT, Completion, Custom
- Split train/validation sets
- Prepare data for tokenization

**Format Specifications:**

**Alpaca:**
```json
{
  "instruction": "Task description",
  "input": "Optional context",
  "output": "Expected response"
}
```

**ShareGPT:**
```json
{
  "conversations": [
    {"from": "human", "value": "Question"},
    {"from": "gpt", "value": "Answer"}
  ]
}
```

**Completion:**
```json
{
  "text": "Raw completion text"
}
```

**Custom:**
```json
{
  "custom_input_field": "...",
  "custom_output_field": "..."
}
```

**Data Flow:**
```
JSONL file → Format detection → Parsing → DatasetExample → Train/Val split
```

**Data Structures:**
```rust
pub struct Dataset {
    pub train: Vec<DatasetExample>,
    pub validation: Vec<DatasetExample>,
}

pub struct DatasetExample {
    pub text: String,      // Full formatted text
    pub input: String,     // Input portion
    pub output: String,    // Expected output
}
```

---

### 4. Training System (`src/trainer.rs`)

**Current State:** Scaffold with stub implementations

**Planned Architecture:**

```rust
pub struct Trainer {
    config: AxolotlConfig,
    model: Model,                    // From candle-transformers
    optimizer: Box<dyn Optimizer>,   // AdamW, SGD, etc.
    scheduler: Box<dyn Scheduler>,   // Cosine, Linear, etc.
    device: Device,                  // CPU/CUDA/Metal
    epoch: usize,
    step: usize,
}

impl Trainer {
    pub fn new(config: AxolotlConfig) -> Result<Self>;
    pub fn train(&mut self) -> Result<()>;
    pub fn resume_from(&mut self, checkpoint: &str) -> Result<()>;
    fn train_step(&mut self, batch: &Batch) -> Result<Loss>;
    fn save_checkpoint(&self, path: &str) -> Result<()>;
}
```

**Training Loop (Planned Phase 2):**

```rust
for epoch in 0..config.epochs {
    for batch in dataloader {
        // Forward pass
        let logits = model.forward(&batch.input_ids)?;
        let loss = cross_entropy(&logits, &batch.labels)?;
        
        // Backward pass
        loss.backward()?;
        
        // Gradient accumulation
        if (step + 1) % grad_accum_steps == 0 {
            // Clip gradients
            clip_grad_norm_(model.parameters(), max_norm)?;
            
            // Optimizer step
            optimizer.step()?;
            scheduler.step()?;
            optimizer.zero_grad()?;
        }
        
        // Logging
        if step % log_steps == 0 {
            log_metrics(loss, perplexity, throughput)?;
        }
        
        // Checkpointing
        if step % save_steps == 0 {
            save_checkpoint(step)?;
        }
        
        step += 1;
    }
    epoch += 1;
}
```

---

### 5. Model Management (`src/model.rs`)

**Current State:** Stub functions

**Planned Architecture (Phase 2):**

```rust
pub struct Model {
    base: Arc<dyn TransformerModel>,   // LLaMA, Mistral, etc.
    adapters: HashMap<String, Box<dyn Adapter>>,
    active_adapter: Option<String>,
    device: Device,
    dtype: DType,
}

impl Model {
    pub fn from_pretrained(
        model_id: &str,
        config: ModelConfig,
    ) -> Result<Self>;
    
    pub fn add_adapter(
        &mut self,
        name: &str,
        adapter: Box<dyn Adapter>,
    ) -> Result<()>;
    
    pub fn set_active_adapter(&mut self, name: &str) -> Result<()>;
    
    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor>;
    
    pub fn save_adapter(&self, path: &str) -> Result<()>;
    
    pub fn merge_adapter(&self) -> Result<Tensor>;
}
```

**Model Loading Flow:**
```
HF model ID → Download safetensors → Load weights → 
Initialize architecture → Apply DType → Move to device → 
Attach adapters → Ready for training
```

---

### 6. Error Handling (`src/error.rs`)

**Design Philosophy:**
- Use `Result<T>` everywhere (aliased to `Result<T, AxolotlError>`)
- Provide context with error messages
- Enable error source chains with `#[from]`

**Error Hierarchy:**

```rust
pub enum AxolotlError {
    Config(String),              // Configuration errors
    ConfigParse(serde_yaml::Error),  // YAML parsing
    Model(String),               // Model loading/operations
    Dataset(String),             // Dataset loading
    Training(String),            // Training loop errors
    Checkpoint(String),          // Save/load checkpoints
    Io(std::io::Error),          // File I/O
    Peft(String),                // PEFT adapter errors
    Qlora(String),               // Quantization errors
    Candle(candle_core::Error),  // Tensor operations
    Tokenizer(tokenizers::Error), // Tokenization
    Template(String),            // Progress bar templates
    Other(String),               // Catch-all
}
```

**Error Propagation:**
```rust
// Low-level error
fs::read_to_string(path)?  // std::io::Error

// Converted via #[from]
↓
AxolotlError::Io(...)

// Propagated with context
.map_err(|e| AxolotlError::Config(format!("Failed to load config: {}", e)))?
```

---

## Integration with Workspace Crates

### peft-rs Integration (Phase 2)

**Current:** Using mock implementations  
**Future:** Direct integration with `peft-rs` crate

```rust
use peft_rs::{LoraConfig, LoraLayer};

// Apply LoRA to model layers
for (name, layer) in model.layers_mut() {
    if config.lora.target_modules.contains(name) {
        let lora = LoraLayer::new_with_zeros(
            layer.in_features(),
            layer.out_features(),
            lora_config.clone(),
            &device,
        )?;
        layer.add_adapter(lora)?;
    }
}
```

**Supported Adapters (via peft-rs):**
- LoRA (Phase 2)
- DoRA (Phase 4)
- AdaLoRA, IA³, LoHa, LoKr, OFT, BOFT, VeRA, Prefix/Prompt Tuning (Phase 4)

---

### qlora-rs Integration (Phase 3)

**Purpose:** 4-bit/8-bit quantization for memory efficiency

```rust
use qlora_rs::{quantize_model, QuantizationConfig, QuantType};

let quant_config = QuantizationConfig {
    bits: 4,
    quant_type: QuantType::Nf4,  // or Fp4
    double_quant: true,
    block_size: 64,
};

let quantized = quantize_model(&model, quant_config)?;
// Model now uses 4-bit weights, ~75% memory reduction
```

---

### unsloth-rs Integration (Phase 3)

**Purpose:** 2x faster inference via optimized kernels

```rust
use unsloth_rs::{optimize_model, OptimizationConfig};

let optimized = optimize_model(&model, OptimizationConfig {
    use_flash_attention: true,
    fuse_rope: true,
    fuse_mlp: true,
})?;
// 2x faster inference, lower latency
```

---

## Data Flow: End-to-End Training

```
1. CLI Input
   └─> axolotl train config.yaml

2. Config Loading
   └─> Parse YAML → Validate → AxolotlConfig

3. Dataset Loading
   └─> Load JSONL → Parse format → Split train/val → Dataset

4. Model Setup (Phase 2)
   └─> Download model → Load weights → Attach adapters → Model

5. Training Initialization
   └─> Create optimizer → Create scheduler → Create dataloader

6. Training Loop
   └─> For each epoch:
       └─> For each batch:
           ├─> Forward pass (Model)
           ├─> Compute loss
           ├─> Backward pass
           ├─> Gradient accumulation
           ├─> Optimizer step
           ├─> Scheduler step
           ├─> Log metrics
           └─> Save checkpoint (every N steps)

7. Final Save
   └─> Save adapter weights → Save config → Done!
```

---

## Memory Management

### Tensor Lifecycle

```rust
// Tensors are reference-counted in candle
let input = Tensor::new(...)?;  // Allocates on device
let output = model.forward(&input)?;  // Reuses memory where possible
// Tensors automatically freed when dropped
```

### Gradient Checkpointing (Phase 2)

```rust
// Save memory by recomputing activations during backward pass
if config.training.gradient_checkpointing {
    model.enable_gradient_checkpointing()?;
}
// Trades compute for memory (2x slower, 50% less memory)
```

### Adapter Memory Overhead

**LoRA (r=64, alpha=16):**
```
Base model: 7B parameters = 14GB (FP16)
LoRA adapters: ~100M parameters = 200MB (FP16)
Overhead: 1.4% of base model
```

---

## Testing Strategy

### Unit Tests (Current: 48 tests)

**Error Handling:** 18 tests
- Error creation for all variants
- Error conversions
- Source chain testing

**Configuration:** 28 tests
- Serialization/deserialization
- Validation logic
- Preset generation
- File I/O roundtrips

**Dataset:** 1+ tests (to be expanded)
- Format parsing
- Train/val splitting
- Edge cases

**Trainer:** 1+ tests (to be expanded)
- Initialization
- Checkpoint management

### Integration Tests (Phase 2)

```rust
#[test]
fn test_end_to_end_training() {
    // 1. Load config
    let config = AxolotlConfig::from_file("tests/fixtures/config.yaml")?;
    
    // 2. Load dataset
    let dataset = Dataset::load(&config.dataset)?;
    
    // 3. Initialize trainer
    let mut trainer = Trainer::new(config)?;
    
    // 4. Train for 1 epoch
    trainer.train()?;
    
    // 5. Verify checkpoint exists
    assert!(Path::new("outputs/checkpoint-1").exists());
    
    // 6. Load adapter and test inference
    let model = Model::load_adapter("outputs/checkpoint-1")?;
    let output = model.generate("Test input")?;
    assert!(!output.is_empty());
}
```

### Benchmarks (Current: 9 benchmarks)

**Config Parsing:**
- Small config (50 lines)
- Large config (500 lines)
- Validation overhead
- Preset generation
- File I/O

**Planned (Phase 2):**
- Dataset loading throughput
- Tokenization speed
- Training step latency
- Memory usage
- Checkpoint save/load time

---

## Performance Considerations

### CPU Optimization

- Multi-threaded tensor operations (via candle)
- SIMD vectorization (auto-optimized by candle)
- Efficient memory allocation (Rust's allocator)
- Zero-copy where possible

### GPU Optimization (Phase 2)

- Minimize host-device transfers
- Batch operations for better utilization
- Use mixed precision (FP16/BF16)
- Gradient checkpointing for large models
- Flash Attention 2 (via unsloth-rs)

### Memory Optimization (Phase 3)

- 4-bit quantization (qlora-rs)
- Gradient accumulation (reduce batch memory)
- Adapter-only training (freeze base model)
- Streaming datasets (avoid loading all data)

---

## Concurrency Model

### Current (Single-threaded)

```
Main thread:
├─> Load config
├─> Load dataset
├─> Training loop (single GPU)
└─> Save checkpoint
```

### Future (Phase 3: Multi-GPU)

```
Main process:
├─> Rank 0 (GPU 0): Coordinator
│   ├─> Load config
│   ├─> Broadcast to other ranks
│   └─> Aggregate gradients
├─> Rank 1-N (GPU 1-N): Workers
│   ├─> Receive config
│   ├─> Load local data shards
│   ├─> Compute local gradients
│   └─> Send to rank 0
└─> All ranks: Synchronize and step
```

---

## Deployment Architecture (Phase 4)

### Training Deployment

```
┌───────────────────────────────────┐
│     Kubernetes Cluster            │
│  ┌──────────────────────────────┐ │
│  │   Training Job Pod           │ │
│  │  ┌────────────────────────┐  │ │
│  │  │  axolotl train         │  │ │
│  │  │  - 8x A100 GPUs        │  │ │
│  │  │  - 1TB RAM             │  │ │
│  │  │  - NVMe storage        │  │ │
│  │  └────────────────────────┘  │ │
│  │  ┌────────────────────────┐  │ │
│  │  │  Persistent Volume     │  │ │
│  │  │  - Checkpoints         │  │ │
│  │  │  - Logs                │  │ │
│  │  └────────────────────────┘  │ │
│  └──────────────────────────────┘ │
└───────────────────────────────────┘
```

### Inference Deployment

```
┌──────────────────────────────────────┐
│     Load Balancer                    │
└────────────┬─────────────────────────┘
             │
    ┌────────┴────────┐
    │                 │
┌───▼──────┐    ┌────▼──────┐
│ Replica 1│    │ Replica 2 │
│ axolotl  │    │ axolotl   │
│  serve   │    │  serve    │
│ (1x GPU) │    │ (1x GPU)  │
└──────────┘    └───────────┘
```

---

## Security Considerations

### Input Validation

- Sanitize file paths (prevent path traversal)
- Validate YAML structure (prevent YAML bombs)
- Limit config file size (<10MB)
- Check dataset format before processing

### Model Downloads

- Verify safetensors checksums
- Use HTTPS for HuggingFace Hub
- Support local model paths (air-gapped deployments)

### Memory Safety

- Rust prevents buffer overflows, use-after-free
- No unsafe code in main codebase (workspace lint)
- GPU memory bounds checking via candle

---

## Future Architecture Enhancements

### Plugin System (Future)

```rust
pub trait ModelAdapter {
    fn load_weights(&mut self, path: &Path) -> Result<()>;
    fn forward(&self, input: &Tensor) -> Result<Tensor>;
}

// Users can implement custom adapters
pub struct MyCustomAdapter { ... }
impl ModelAdapter for MyCustomAdapter { ... }
```

### Distributed Training (Phase 3)

```rust
pub struct DistributedTrainer {
    local_rank: usize,
    world_size: usize,
    backend: CommunicationBackend,  // NCCL, Gloo
}
```

### Model Serving API (Phase 4)

```rust
#[tokio::main]
async fn main() {
    let app = Router::new()
        .route("/v1/completions", post(completions))
        .route("/v1/chat/completions", post(chat_completions));
    
    axum::Server::bind(&"0.0.0.0:8080".parse()?)
        .serve(app.into_make_service())
        .await?;
}
```

---

## References

### Internal Documentation
- [PORTING_PLAN.md](PORTING_PLAN.md) - Detailed porting roadmap
- [CONTRIBUTING.md](CONTRIBUTING.md) - Development workflow
- [TEST_COVERAGE_PLAN.md](TEST_COVERAGE_PLAN.md) - Testing strategy
- [CHANGELOG.md](CHANGELOG.md) - Version history

### External Resources
- [Candle Documentation](https://huggingface.github.io/candle/)
- [Python Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)

---

**Document Version:** 1.0  
**Last Updated:** January 10, 2026  
**Next Update:** Phase 2 kickoff
