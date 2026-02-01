# Aphelion Framework ↔ tritter-accel Integration Handoff

## Overview

This handoff describes integration work between **aphelion-framework-rs** (orchestration) and **tritter-accel** (acceleration). Each crate lives in its own repository.

## Repository Structure

| Repository | Purpose | GitHub |
|------------|---------|--------|
| aphelion-framework-rs | Orchestration framework for AI model construction | https://github.com/tzervas/aphelion-framework-rs |
| tritter-accel | Acceleration for AI training and inference | https://github.com/tzervas/rust-ai (tritter-accel crate) |
| rust-ai-core | Core orchestrator with device management | (separate repo, used by both) |

**Note**: The rust-ai GitHub repo contains tritter-accel as the published crate. Other crates (peft-rs, qlora-rs, etc.) are separate repos that tritter-accel delegates to.

## What Each Project Does

### aphelion-framework-rs
Orchestration and configuration framework (does NOT do tensor operations):
- Deterministic model graph construction with SHA256 hashing
- Type-safe configuration management
- Pipeline orchestration with composable stages
- Backend abstraction for hardware-agnostic execution
- Structured tracing and diagnostics

### tritter-accel
Acceleration primitives for training and inference:
- **Ternary operations**: BitNet b1.58 quantization, packed matmul
- **VSA gradient compression**: 10-100x compression for distributed training
- **Training acceleration**: DeterministicPhaseTrainer
- **Inference acceleration**: TernaryLayer, KVCache
- **Dual API**: Both Rust (`core` module) and Python (PyO3)

## Integration Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    aphelion-framework-rs                        │
│                                                                 │
│  BuildPipeline                                                  │
│  ├── PeftAdapterStage      (→ peft-rs)                         │
│  ├── QuantizationStage     (→ qlora-rs / bitnet-quantize)      │
│  └── AccelerationStage     (→ tritter-accel) ◄── THIS HANDOFF  │
│                                                                 │
│  Backend Trait                                                  │
│  ├── NullBackend (testing)                                     │
│  ├── BurnBackend                                                │
│  └── TriterAccelBackend ◄── NEW                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       tritter-accel                             │
│                                                                 │
│  core::training                                                 │
│  ├── GradientCompressor     10-100x gradient compression       │
│  ├── TernaryCompressedGradient                                  │
│  ├── GradientAccumulator                                        │
│  └── mixed_precision::LossScaler                                │
│                                                                 │
│  core::inference                                                │
│  ├── InferenceEngine        Batched inference                   │
│  ├── TernaryLayer           Pre-quantized layer                 │
│  └── KVCache                                                    │
│                                                                 │
│  core::ternary                                                  │
│  ├── PackedTernary          16x memory compression              │
│  ├── matmul()               Addition-only arithmetic            │
│  └── dot()                                                      │
│                                                                 │
│  core::quantization                                             │
│  ├── quantize_absmean()     BitNet b1.58 style                  │
│  └── quantize_absmax()                                          │
│                                                                 │
│  core::vsa                                                      │
│  ├── VsaOps                 Hyperdimensional computing          │
│  └── bind/unbind/bundle/similarity                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Integration Tasks

### Task 1: Create TriterAccelBackend (Priority: High)

**File**: `aphelion-framework-rs/crates/aphelion-core/src/tritter_backend.rs` (new)

Implement aphelion's `Backend` trait using tritter-accel for acceleration.

```rust
use aphelion_core::{Backend, DeviceCapabilities, AphelionError, BuildGraph};
use tritter_accel::core::{
    inference::InferenceEngine,
    training::GradientCompressor,
};
use candle_core::Device;

pub struct TriterAccelBackend {
    device: Device,
    inference_engine: Option<InferenceEngine>,
    gradient_compressor: Option<GradientCompressor>,
}

impl TriterAccelBackend {
    pub fn new() -> Result<Self, AphelionError> {
        let device = Device::cuda_if_available(0)
            .map_err(|e| AphelionError::backend(format!("device error: {e}")))?;
        Ok(Self {
            device,
            inference_engine: None,
            gradient_compressor: None,
        })
    }

    pub fn with_inference(mut self) -> Result<Self, AphelionError> {
        self.inference_engine = Some(InferenceEngine::new(&self.device)?);
        Ok(self)
    }

    pub fn with_training(mut self, compression_ratio: f32) -> Result<Self, AphelionError> {
        let config = tritter_accel::core::training::TrainingConfig::default()
            .with_compression_ratio(compression_ratio);
        self.gradient_compressor = Some(GradientCompressor::new(config));
        Ok(self)
    }
}

impl Backend for TriterAccelBackend {
    fn name(&self) -> &str {
        "tritter-accel"
    }

    fn capabilities(&self) -> DeviceCapabilities {
        DeviceCapabilities {
            supports_cuda: matches!(self.device, Device::Cuda(_)),
            supports_metal: false,
            supports_vulkan: false,
            max_memory_bytes: None,
        }
    }

    fn execute(&self, graph: &BuildGraph) -> Result<(), AphelionError> {
        // Map graph nodes to tritter-accel operations
        for node in graph.topological_order() {
            self.execute_node(node)?;
        }
        Ok(())
    }
}
```

### Task 2: Create AccelerationStage (Priority: High)

**File**: `aphelion-framework-rs/crates/aphelion-core/src/stages/acceleration.rs` (new)

Pipeline stage that applies tritter-accel optimizations to the build graph.

```rust
use aphelion_core::{PipelineStage, BuildContext, BuildGraph, AphelionError};
use tritter_accel::core::{
    training::{GradientCompressor, TrainingConfig},
    inference::{InferenceEngine, TernaryLayer},
    quantization::{quantize_absmean, QuantizeConfig},
};

pub struct AccelerationStage {
    mode: AccelerationMode,
}

pub enum AccelerationMode {
    /// Training mode: gradient compression, mixed precision
    Training {
        compression_ratio: f32,  // 0.1 = 10x compression
        use_deterministic_trainer: bool,
    },
    /// Inference mode: packed weights, KV caching
    Inference {
        use_ternary_layers: bool,
        batch_size: usize,
    },
}

impl AccelerationStage {
    pub fn training(compression_ratio: f32) -> Self {
        Self {
            mode: AccelerationMode::Training {
                compression_ratio,
                use_deterministic_trainer: true,
            },
        }
    }

    pub fn inference(batch_size: usize) -> Self {
        Self {
            mode: AccelerationMode::Inference {
                use_ternary_layers: true,
                batch_size,
            },
        }
    }
}

impl PipelineStage for AccelerationStage {
    fn name(&self) -> &str {
        "tritter-acceleration"
    }

    fn execute(&self, ctx: &BuildContext, graph: &mut BuildGraph) -> Result<(), AphelionError> {
        match &self.mode {
            AccelerationMode::Training { compression_ratio, use_deterministic_trainer } => {
                // Inject gradient compression hooks
                self.inject_gradient_compression(graph, *compression_ratio)?;

                if *use_deterministic_trainer {
                    // Configure deterministic phase training
                    self.inject_deterministic_trainer(graph)?;
                }
            }
            AccelerationMode::Inference { use_ternary_layers, batch_size } => {
                if *use_ternary_layers {
                    // Convert linear layers to TernaryLayer
                    self.convert_to_ternary_layers(graph)?;
                }

                // Configure batching
                graph.set_metadata("batch_size", batch_size.to_string());
            }
        }
        Ok(())
    }
}
```

### Task 3: Gradient Compression Integration (Priority: High)

**File**: `aphelion-framework-rs/crates/aphelion-core/src/hooks/gradient_compression.rs` (new)

Pre/post hooks for gradient compression in distributed training.

```rust
use aphelion_core::{PipelineHook, HookContext, AphelionError};
use tritter_accel::core::training::{GradientCompressor, TernaryCompressedGradient};

pub struct GradientCompressionHook {
    compressor: GradientCompressor,
    seed: u64,
}

impl GradientCompressionHook {
    pub fn new(compression_ratio: f32, seed: u64) -> Self {
        let config = tritter_accel::core::training::TrainingConfig::default()
            .with_compression_ratio(compression_ratio);
        Self {
            compressor: GradientCompressor::new(config),
            seed,
        }
    }
}

impl PipelineHook for GradientCompressionHook {
    fn name(&self) -> &str {
        "gradient-compression"
    }

    fn pre_execute(&self, ctx: &mut HookContext) -> Result<(), AphelionError> {
        // Called before backward pass
        Ok(())
    }

    fn post_execute(&self, ctx: &mut HookContext) -> Result<(), AphelionError> {
        // Called after backward pass - compress gradients
        if let Some(gradients) = ctx.get_gradients() {
            let compressed = self.compressor.compress(gradients, Some(self.seed))
                .map_err(|e| AphelionError::hook(format!("compression failed: {e}")))?;

            ctx.set_compressed_gradients(compressed);

            // Log compression stats
            tracing::info!(
                "Gradient compression: {}x reduction",
                gradients.len() as f32 / compressed.len() as f32
            );
        }
        Ok(())
    }
}
```

### Task 4: Inference Engine Integration (Priority: Medium)

**File**: `aphelion-framework-rs/crates/aphelion-core/src/execution/inference.rs` (new)

Wrapper for tritter-accel's InferenceEngine.

```rust
use aphelion_core::{BuildGraph, AphelionError};
use tritter_accel::core::inference::{InferenceEngine, TernaryLayer, KVCache};
use candle_core::{Tensor, Device};

pub struct AcceleratedInference {
    engine: InferenceEngine,
    layers: Vec<TernaryLayer>,
    kv_cache: Option<KVCache>,
}

impl AcceleratedInference {
    pub fn from_graph(graph: &BuildGraph, device: &Device) -> Result<Self, AphelionError> {
        let engine = InferenceEngine::new(device)
            .map_err(|e| AphelionError::execution(format!("engine init failed: {e}")))?;

        // Convert graph layers to TernaryLayers
        let layers = graph.nodes()
            .filter(|n| n.op_type() == "linear")
            .map(|n| self.convert_to_ternary_layer(n, device))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            engine,
            layers,
            kv_cache: None,
        })
    }

    pub fn with_kv_cache(mut self, max_seq_len: usize) -> Self {
        self.kv_cache = Some(KVCache::new(max_seq_len));
        self
    }

    pub fn forward(&mut self, input: &Tensor) -> Result<Tensor, AphelionError> {
        let mut x = input.clone();

        for layer in &self.layers {
            x = layer.forward(&x)
                .map_err(|e| AphelionError::execution(format!("layer forward failed: {e}")))?;
        }

        Ok(x)
    }
}
```

### Task 5: Update Cargo.toml (Priority: Low)

**File**: `aphelion-framework-rs/crates/aphelion-core/Cargo.toml`

```toml
[features]
default = []
burn = []
cubecl = []
tritter-accel = ["dep:tritter-accel", "dep:candle-core"]

[dependencies]
# Acceleration
tritter-accel = { version = "0.2", optional = true }
candle-core = { version = "0.9", optional = true }
```

**For local development** (path dependency):
```toml
[dependencies]
tritter-accel = { path = "../../../rust-ai/tritter-accel", optional = true }
```

## API Reference

### tritter-accel Rust API (core module)

```rust
use tritter_accel::core::{
    // Quantization
    quantization::{quantize_absmean, quantize_absmax, QuantizeConfig, QuantizationResult},

    // Ternary operations
    ternary::{PackedTernary, matmul, dot},

    // VSA operations
    vsa::{VsaOps, VsaConfig},

    // Training
    training::{GradientCompressor, TrainingConfig, TernaryCompressedGradient, GradientAccumulator},
    training::mixed_precision::LossScaler,

    // Inference
    inference::{InferenceEngine, TernaryLayer, KVCache},
};
```

### Key Types

| Type | Purpose |
|------|---------|
| `GradientCompressor` | Compresses gradients 10-100x using VSA |
| `TernaryCompressedGradient` | Compressed gradient representation |
| `InferenceEngine` | Batched inference with device management |
| `TernaryLayer` | Pre-quantized layer (16x memory reduction) |
| `KVCache` | Key-value cache for autoregressive inference |
| `PackedTernary` | Packed ternary weights (4 values per byte) |
| `VsaOps` | VSA bind/unbind/bundle/similarity operations |

## Testing Strategy

### Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "tritter-accel")]
    fn test_acceleration_stage_training() {
        let stage = AccelerationStage::training(0.1);
        assert_eq!(stage.name(), "tritter-acceleration");
    }

    #[test]
    #[cfg(feature = "tritter-accel")]
    fn test_gradient_compression_hook() {
        let hook = GradientCompressionHook::new(0.1, 42);
        let gradients: Vec<f32> = vec![0.1, -0.2, 0.3, 0.4];

        // Test compression
        let compressed = hook.compressor.compress(&gradients, Some(42)).unwrap();
        assert!(compressed.len() < gradients.len());
    }
}
```

### Integration Tests

```rust
// tests/tritter_integration.rs
#[test]
#[cfg(feature = "tritter-accel")]
fn test_full_pipeline_with_acceleration() {
    use aphelion_core::{BuildPipeline, TriterAccelBackend};

    let backend = TriterAccelBackend::new()
        .unwrap()
        .with_training(0.1)
        .unwrap();

    let pipeline = BuildPipeline::new()
        .with_stage(AccelerationStage::training(0.1))
        .with_post_hook(GradientCompressionHook::new(0.1, 42));

    let graph = create_test_graph();
    let result = pipeline.execute(&backend, graph);
    assert!(result.is_ok());
}
```

## Success Criteria

1. **TriterAccelBackend**: Implements `Backend` trait, wraps tritter-accel
2. **AccelerationStage**: Configures training/inference acceleration
3. **GradientCompressionHook**: Compresses gradients via VSA
4. **AcceleratedInference**: Wraps InferenceEngine with TernaryLayer support
5. **Feature flag**: Clean `tritter-accel` feature in Cargo.toml
6. **Tests**: Unit + integration tests for all components
7. **Documentation**: All public APIs documented

## Files to Create in aphelion-framework-rs

| File | Description |
|------|-------------|
| `src/tritter_backend.rs` | TriterAccelBackend implementation |
| `src/stages/acceleration.rs` | AccelerationStage |
| `src/hooks/gradient_compression.rs` | GradientCompressionHook |
| `src/execution/inference.rs` | AcceleratedInference wrapper |
| `tests/tritter_integration.rs` | Integration tests |

## Command Reference

```bash
# Build aphelion with tritter-accel
cd aphelion-framework-rs
cargo build --features tritter-accel

# Run tests
cargo test --features tritter-accel

# Build tritter-accel (if modifying)
cd ../rust-ai/tritter-accel
cargo build --release
cargo test
```

## Notes

- tritter-accel delegates to sister crates (trit-vsa, bitnet-quantize, vsa-optim-rs)
- The `core` module provides the Rust API; PyO3 bindings are for Python users
- GPU support requires `cuda` feature flag on tritter-accel
- All operations are deterministic when seeded (reproducibility)
