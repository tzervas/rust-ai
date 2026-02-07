//! GPT-2 Small baseline training (vanilla Burn).
//!
//! This example trains GPT-2 Small (124M parameters) using standard Burn
//! training without HybridTrainer, establishing a baseline for comparison.
//!
//! ## Usage
//!
//! ```bash
//! # CPU (slow, for testing only)
//! cargo run --release --example gpt2_small_baseline --features autodiff,ndarray
//!
//! # GPU (recommended)
//! cargo run --release --example gpt2_small_baseline --features autodiff,cuda
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
    models::gpt2::{Gpt2Batch, Gpt2Config, Gpt2Model},
    Model, Optimizer, // Import traits for forward/backward/step
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
    println!("🚀 GPT-2 Small Baseline Training (Vanilla Burn)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Configuration
    let config = Gpt2Config::gpt2_small();
    let batch_size = 4;
    let seq_len = 64;
    let steps = 50; // Quick validation
    let log_interval = 10;

    println!("Configuration:");
    println!("  Model: GPT-2 Small (124M params)");
    println!("  Vocab size: {}", config.vocab_size);
    println!("  Embedding dim: {}", config.n_embd);
    println!("  Layers: {}", config.n_layer);
    println!("  Heads: {}", config.n_head);
    println!("  Batch size: {}", batch_size);
    println!("  Sequence length: {}", seq_len);
    println!("  Training steps: {}", steps);
    println!();

    // Initialize model and optimizer
    let device = <MyBackend as Backend>::Device::default();
    println!("✓ Creating model...");
    let model = Gpt2Model::new(&config, &device);
    let forward_fn = Gpt2Forward;
    let mut wrapped_model = BurnModelWrapper::new(model, forward_fn, device.clone());

    println!("✓ Creating optimizer (Adam, lr=6e-4)...");
    let optim = AdamConfig::new()
        .with_epsilon(1e-8)
        .init();
    let mut wrapped_optimizer = BurnOptimizerWrapper::new(optim, 6e-4);

    println!("✓ Starting training...\n");
    println!("Step | Loss    | Perplexity | VRAM (MB) | Time (ms)");
    println!("-----|---------|------------|-----------|----------");

    let start_time = Instant::now();
    let mut total_time_ms = 0.0;

    for step in 0..steps {
        let step_start = Instant::now();

        // Generate batch
        let batch_data = generate_synthetic_batch(batch_size, seq_len, config.vocab_size, &device);
        let batch = BurnBatch::new(batch_data, batch_size);

        // Forward pass
        let loss_value = wrapped_model.forward(&batch).expect("Forward failed");

        // Backward pass
        let grad_info = wrapped_model.backward().expect("Backward failed");

        // Optimizer step
        wrapped_optimizer
            .step(&mut wrapped_model, &grad_info)
            .expect("Optimizer step failed");

        let step_time = step_start.elapsed().as_secs_f64() * 1000.0;
        total_time_ms += step_time;

        // Log progress
        if step % log_interval == 0 || step == steps - 1 {
            let perplexity = (loss_value as f64).exp();
            let vram_mb = measure_vram_mb();

            println!(
                "{:4} | {:.5} | {:10.1} | {:9.1} | {:8.1}",
                step, loss_value, perplexity, vram_mb, step_time
            );
        }
    }

    let total_time_s = start_time.elapsed().as_secs_f64();

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("✅ Training Complete");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    println!("Total time: {:.1}s", total_time_s);
    println!("Avg time per step: {:.1}ms", total_time_ms / steps as f64);
    println!("Throughput: {:.1} tokens/sec", (batch_size * seq_len * steps) as f64 / total_time_s);
    println!("\n💡 Next: Run with HybridTrainer for speedup comparison");
    println!("   cargo run --release --example gpt2_small_hybrid --features autodiff,ndarray\n");
}
