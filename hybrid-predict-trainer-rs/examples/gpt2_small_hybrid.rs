//! GPT-2 Small with HybridTrainer (Phase 2B validation).
//!
//! This example trains GPT-2 Small (124M parameters) using the HybridTrainer
//! with 4-phase training (Warmup → Full → Predict → Correct), demonstrating
//! speedup compared to the vanilla baseline.
//!
//! ## Usage
//!
//! ```bash
//! # CPU (slow, for testing only)
//! cargo run --release --example gpt2_small_hybrid --features autodiff,ndarray
//!
//! # GPU (recommended)
//! cargo run --release --example gpt2_small_hybrid --features autodiff,cuda
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
    models::gpt2::{Gpt2Batch, Gpt2Config, Gpt2Model},
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
    println!("🚀 GPT-2 Small HybridTrainer (Phase 2B)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Configuration
    let config = Gpt2Config::gpt2_small();
    let batch_size = 4;
    let seq_len = 64;
    let steps = 50; // Quick validation
    let log_interval = 5;

    println!("Model Configuration:");
    println!("  Model: GPT-2 Small (124M params)");
    println!("  Vocab size: {}", config.vocab_size);
    println!("  Embedding dim: {}", config.n_embd);
    println!("  Layers: {}", config.n_layer);
    println!("  Heads: {}", config.n_head);
    println!("  Batch size: {}", batch_size);
    println!("  Sequence length: {}", seq_len);
    println!("  Training steps: {}\n", steps);

    // HybridTrainer configuration (tuned for quick validation)
    let hybrid_config = HybridTrainerConfig::builder()
        .warmup_steps(5)  // Reduced for quick test
        .full_steps(10)   // Reduced for quick test
        .max_predict_steps(5)
        .correction_interval(0)  // DISABLED: Fixes VRAM leak from model.map() copies
        .divergence_threshold(2.5)
        .confidence_threshold(0.3)  // Lower to trigger Predict phase
        .build();

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
    println!("✓ Creating HybridTrainer...");
    let mut trainer = HybridTrainer::new(wrapped_model, wrapped_optimizer, hybrid_config)
        .expect("Failed to create HybridTrainer");

    println!("✓ Starting training...\n");
    println!("Step | Phase   | Loss    | Perplexity | VRAM (MB) | Time (ms)");
    println!("-----|---------|---------|------------|-----------|----------");

    let start_time = Instant::now();
    let mut total_time_ms = 0.0;
    let mut phase_counts = std::collections::HashMap::new();

    for step in 0..steps {
        let step_start = Instant::now();

        // Generate batch
        let batch_data = generate_synthetic_batch(batch_size, seq_len, config.vocab_size, &device);
        let batch = BurnBatch::new(batch_data, batch_size);

        // HybridTrainer step
        let result = trainer.step(&batch).expect("Training step failed");

        let step_time = step_start.elapsed().as_secs_f64() * 1000.0;
        total_time_ms += step_time;

        // Track phase distribution
        let phase_str = format!("{:?}", result.phase);
        *phase_counts.entry(phase_str.clone()).or_insert(0) += 1;

        // Log progress
        if step % log_interval == 0 || step == steps - 1 {
            let perplexity = (result.loss as f64).exp();
            let vram_mb = measure_vram_mb();

            println!(
                "{:4} | {:7} | {:.5} | {:10.1} | {:9.1} | {:8.1}",
                step, phase_str, result.loss, perplexity, vram_mb, step_time
            );
        }
    }

    let total_time_s = start_time.elapsed().as_secs_f64();

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("✅ Training Complete");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    println!("Performance:");
    println!("  Total time: {:.1}s", total_time_s);
    println!("  Avg time per step: {:.1}ms", total_time_ms / steps as f64);
    println!("  Throughput: {:.1} tokens/sec", (batch_size * seq_len * steps) as f64 / total_time_s);

    println!("\nPhase Distribution:");
    let mut phases: Vec<_> = phase_counts.iter().collect();
    phases.sort_by_key(|(name, _)| name.as_str());
    for (phase, count) in phases {
        let percentage = (*count as f64 / steps as f64) * 100.0;
        println!("  {}: {} steps ({:.1}%)", phase, count, percentage);
    }

    println!("\n💡 Compare with baseline:");
    println!("   cargo run --release --example gpt2_small_baseline --features autodiff,ndarray\n");
}
