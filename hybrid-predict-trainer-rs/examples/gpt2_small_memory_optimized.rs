//! GPT-2 Small with HybridTrainer + Memory Optimizations (Phase 2B).
//!
//! This example demonstrates all three Phase 2 memory optimizations:
//! 1. Mixed precision (f16/bf16 activations)
//! 2. Gradient accumulation (virtual batch increase)
//! 3. Predict-aware memory (CPU offload during prediction)
//!
//! Target: Fit 1B params in 14GB VRAM (RTX 5080)
//!
//! ## Usage
//!
//! ```bash
//! # CPU (testing only - mixed precision has no benefit)
//! cargo run --release --example gpt2_small_memory_optimized --features autodiff,ndarray
//!
//! # GPU (required for meaningful results)
//! cargo run --release --example gpt2_small_memory_optimized --features autodiff,cuda
//! ```

use burn::{
    backend::{Autodiff, NdArray},
    module::Module as BurnModule,
    optim::AdamConfig,
    tensor::{backend::Backend, Int, Tensor, TensorData},
};

#[cfg(feature = "cuda")]
use burn::backend::Cuda;
use hybrid_predict_trainer_rs::{
    burn_integration::{BurnBatch, BurnForwardFn, BurnModelWrapper, BurnOptimizerWrapper},
    config::HybridTrainerConfig,
    gradient_accumulation::GradientAccumulationConfig,
    mixed_precision::{MixedPrecisionConfig, Precision},
    models::gpt2::{Gpt2Batch, Gpt2Config, Gpt2Model},
    predict_aware_memory::{MemoryOffloadStrategy, PredictAwareMemoryConfig},
    HybridTrainer, Model, Optimizer,
};
use std::time::Instant;

// Use CUDA backend when available, otherwise NdArray (CPU)
#[cfg(feature = "cuda")]
type MyBackend = Autodiff<Cuda>;

#[cfg(not(feature = "cuda"))]
type MyBackend = Autodiff<NdArray>;

/// Generate a synthetic batch of random tokens.
fn generate_synthetic_batch(
    batch_size: usize,
    seq_len: usize,
    vocab_size: usize,
    device: &<MyBackend as Backend>::Device,
) -> Gpt2Batch<MyBackend> {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    // Generate random input IDs
    let input_data: Vec<i64> = (0..batch_size * seq_len)
        .map(|_| rng.gen_range(0..vocab_size as i64))
        .collect();

    // Generate random targets (in real LM, targets = inputs shifted right)
    let target_data: Vec<i64> = (0..batch_size * seq_len)
        .map(|_| rng.gen_range(0..vocab_size as i64))
        .collect();

    let input_ids = Tensor::from_data(
        TensorData::new(input_data, [batch_size, seq_len]),
        device,
    );
    let targets = Tensor::from_data(
        TensorData::new(target_data, [batch_size, seq_len]),
        device,
    );

    Gpt2Batch { input_ids, targets }
}

/// Measure VRAM usage via nvidia-smi (returns 0 on CPU or if unavailable).
fn measure_vram_mb() -> f32 {
    std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().parse::<f32>().ok())
        .unwrap_or(0.0)
}

/// Forward function for GPT-2.
struct Gpt2Forward;

impl BurnForwardFn<MyBackend, Gpt2Model<MyBackend>, Gpt2Batch<MyBackend>> for Gpt2Forward {
    fn forward(
        &self,
        model: Gpt2Model<MyBackend>,
        batch: &BurnBatch<MyBackend, Gpt2Batch<MyBackend>>,
    ) -> (Gpt2Model<MyBackend>, Tensor<MyBackend, 1>) {
        // Forward pass: [B, S] -> [B, S, V]
        let logits = model.forward(batch.data.input_ids.clone());

        // Cross-entropy loss
        let [b, s, v] = logits.dims();
        let logits_flat = logits.reshape([b * s, v]);
        let targets_flat = batch.data.targets.clone().reshape([b * s]);

        let loss = burn::nn::loss::CrossEntropyLossConfig::new()
            .init(&logits_flat.device())
            .forward(logits_flat, targets_flat);

        (model, loss)
    }
}

fn main() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("🚀 GPT-2 Small Memory-Optimized (Phase 2B)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Configuration
    let config = Gpt2Config::gpt2_small();
    let physical_batch = 2;
    let virtual_batch = 8;
    let seq_len = 64;
    let steps = 50; // Quick validation
    let log_interval = 5;

    println!("Model Configuration:");
    println!("  Model: GPT-2 Small (124M params)");
    println!("  Vocab size: {}", config.vocab_size);
    println!("  Embedding dim: {}", config.n_embd);
    println!("  Layers: {}", config.n_layer);
    println!("  Heads: {}", config.n_head);
    println!("  Physical batch: {}", physical_batch);
    println!("  Virtual batch: {} ({}x accumulation)", virtual_batch, virtual_batch / physical_batch);
    println!("  Sequence length: {}", seq_len);
    println!("  Training steps: {}\n", steps);

    // Memory optimization configurations
    let mixed_precision = MixedPrecisionConfig {
        enabled: true,
        default_precision: Precision::Fp16, // f16 activations (50% memory)
        phase_precisions: std::collections::HashMap::new(),
        auto_recommend: true, // Use recommended precisions per phase
    };

    let grad_accumulation = GradientAccumulationConfig {
        enabled: true,
        accumulation_steps: virtual_batch / physical_batch, // 4x accumulation
        scale_lr: false,            // Keep base LR as-is
        normalize_gradients: true,  // Normalize gradients by accumulation steps
    };

    let predict_memory = PredictAwareMemoryConfig {
        enabled: true,
        offload_strategy: MemoryOffloadStrategy::CpuOffload, // CPU offload
        async_restore: true,            // Async prefetch before full phase
        async_restore_lookahead: 3,     // Start 3 steps early
        discard_activations: true,      // Drop activations in predict phase
        gradient_checkpointing: false,  // No checkpointing (simplicity)
    };

    // HybridTrainer configuration (tuned for quick validation)
    let hybrid_config = HybridTrainerConfig::builder()
        .warmup_steps(5)
        .full_steps(10)
        .max_predict_steps(5)
        .correction_interval(2)
        .divergence_threshold(2.5)
        .confidence_threshold(0.3)  // Lower to trigger Predict phase
        .mixed_precision_config(mixed_precision)
        .gradient_accumulation_config(grad_accumulation)
        .predict_aware_memory_config(predict_memory)
        .build();

    println!("Memory Optimization Configuration:");
    println!("  Mixed Precision:");
    println!("    Enabled: {}", hybrid_config.mixed_precision_config.enabled);
    println!("    Default precision: {:?}", hybrid_config.mixed_precision_config.default_precision);
    println!("    Auto recommend: {}", hybrid_config.mixed_precision_config.auto_recommend);
    println!("  Gradient Accumulation:");
    println!("    Enabled: {}", hybrid_config.gradient_accumulation_config.enabled);
    println!("    Accumulation steps: {}", hybrid_config.gradient_accumulation_config.accumulation_steps);
    println!("    Scale LR: {}", hybrid_config.gradient_accumulation_config.scale_lr);
    println!("  Predict-Aware Memory:");
    println!("    Enabled: {}", hybrid_config.predict_aware_memory_config.enabled);
    println!("    Offload strategy: {:?}", hybrid_config.predict_aware_memory_config.offload_strategy);
    println!("    Async restore: {}", hybrid_config.predict_aware_memory_config.async_restore);
    println!("    Discard activations: {}\n", hybrid_config.predict_aware_memory_config.discard_activations);

    println!("HybridTrainer Configuration:");
    println!("  Warmup steps: {}", hybrid_config.warmup_steps);
    println!("  Full train steps: {}", hybrid_config.full_steps);
    println!("  Max predict steps: {}", hybrid_config.max_predict_steps);
    println!("  Correction interval: {}", hybrid_config.correction_interval);
    println!("  Divergence threshold: {}", hybrid_config.divergence_threshold);
    println!("  Confidence threshold: {}\n", hybrid_config.confidence_threshold);

    // Initialize model and optimizer
    let device = <MyBackend as Backend>::Device::default();
    println!("✓ Creating model...");
    let model = Gpt2Model::new(&config, &device);
    let forward_fn = Gpt2Forward;
    let wrapped_model = BurnModelWrapper::new(model, forward_fn, device.clone());

    println!("✓ Creating optimizer (Adam, lr=6e-4)...");
    let optim = AdamConfig::new()
        .with_epsilon(1e-8)
        .init();
    let wrapped_optimizer = BurnOptimizerWrapper::new(optim, 6e-4);

    // Create HybridTrainer
    println!("✓ Creating HybridTrainer with memory optimizations...");
    let mut trainer = HybridTrainer::new(wrapped_model, wrapped_optimizer, hybrid_config)
        .expect("Failed to create HybridTrainer");

    let initial_vram = measure_vram_mb();
    println!("✓ Initial VRAM: {:.1} MB\n", initial_vram);

    println!("✓ Starting training...\n");
    println!("Step | Phase   | Loss    | Perplexity | VRAM (MB) | Δ VRAM | Time (ms)");
    println!("-----|---------|---------|------------|-----------|--------|----------");

    let start_time = Instant::now();
    let mut total_time_ms = 0.0;
    let mut phase_counts = std::collections::HashMap::new();
    let mut min_vram: f32 = f32::MAX;
    let mut max_vram: f32 = 0.0;

    for step in 0..steps {
        let step_start = Instant::now();

        // Generate batch (physical batch size)
        let batch_data = generate_synthetic_batch(physical_batch, seq_len, config.vocab_size, &device);
        let batch = BurnBatch::new(batch_data, physical_batch);

        // HybridTrainer step
        let result = trainer.step(&batch).expect("Training step failed");

        let step_time = step_start.elapsed().as_secs_f64() * 1000.0;
        total_time_ms += step_time;

        // Track phase distribution
        let phase_str = format!("{:?}", result.phase);
        *phase_counts.entry(phase_str.clone()).or_insert(0) += 1;

        // Track VRAM usage
        let current_vram = measure_vram_mb();
        if current_vram > 0.0 {
            min_vram = min_vram.min(current_vram);
            max_vram = max_vram.max(current_vram);
        }
        let delta_vram = current_vram - initial_vram;

        // Log progress
        if step % log_interval == 0 || step == steps - 1 {
            let perplexity = (result.loss as f64).exp();

            println!(
                "{:4} | {:7} | {:.5} | {:10.1} | {:9.1} | {:6.1} | {:8.1}",
                step, phase_str, result.loss, perplexity, current_vram, delta_vram, step_time
            );
        }
    }

    let total_time_s = start_time.elapsed().as_secs_f64();
    let final_vram = measure_vram_mb();

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("✅ Training Complete");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    println!("Performance:");
    println!("  Total time: {:.1}s", total_time_s);
    println!("  Avg time per step: {:.1}ms", total_time_ms / steps as f64);
    println!("  Effective batch: {} ({} physical × {} accumulation)",
             virtual_batch, physical_batch, virtual_batch / physical_batch);
    println!("  Throughput: {:.1} tokens/sec",
             (virtual_batch * seq_len * steps) as f64 / total_time_s);

    println!("\nMemory Usage:");
    if final_vram > 0.0 {
        println!("  Initial: {:.1} MB", initial_vram);
        println!("  Final: {:.1} MB", final_vram);
        println!("  Peak: {:.1} MB", max_vram);
        println!("  Min: {:.1} MB", min_vram);
        println!("  Delta: {:.1} MB", final_vram - initial_vram);
    } else {
        println!("  (VRAM tracking unavailable - CPU mode)");
    }

    println!("\nPhase Distribution:");
    let mut phases: Vec<_> = phase_counts.iter().collect();
    phases.sort_by_key(|(name, _)| name.as_str());
    for (phase, count) in phases {
        let percentage = (*count as f64 / steps as f64) * 100.0;
        println!("  {}: {} steps ({:.1}%)", phase, count, percentage);
    }

    println!("\n💡 Memory Optimization Benefits:");
    println!("   1. Mixed precision: ~40-50% activation memory reduction");
    println!("   2. Gradient accumulation: {}x larger effective batch", virtual_batch / physical_batch);
    println!("   3. Predict-aware memory: ~60-70% weight offload during predict\n");
}
