//! Train 500M Tritter model using hybrid predictive training.
//!
//! Usage:
//!   cargo run --example train_500m --release
//!
//! With CUDA:
//!   cargo run --example train_500m --release --features cuda

use std::path::PathBuf;
use std::time::Instant;

use candle_core::Device;
use hybrid_predict_trainer_rs::{phases::Phase, Batch, HybridTrainerConfig};

use tritter_model_rs::{
    data::{DataConfig, DataLoader, JsonlDataset, StreamingDataset},
    trainer::create_trainer_with_config,
    TritterConfig,
};

fn main() -> anyhow::Result<()> {
    println!("=== Tritter 500M Rust Training ===\n");

    // Select device
    #[cfg(feature = "cuda")]
    let device = Device::new_cuda(0)?;
    #[cfg(not(feature = "cuda"))]
    let device = Device::Cpu;

    println!("Device: {:?}", device);

    // Model configuration - 100M fits in GPU memory with gradients
    // 500M requires gradient checkpointing which candle doesn't support
    let mut model_config = TritterConfig::small_100m();
    model_config.use_bitnet = true; // Enable ternary quantization

    println!("Model: 100M parameters (BitNet ternary)");
    println!("  Hidden size: {}", model_config.hidden_size);
    println!("  Layers: {}", model_config.num_layers);
    println!("  Heads: {}", model_config.num_heads);
    println!("  BitNet: {}", model_config.use_bitnet);
    println!(
        "  Estimated memory: {} MB",
        model_config.memory_estimate_bitnet() / (1024 * 1024)
    );
    println!();

    // Training configuration
    let trainer_config = HybridTrainerConfig::builder()
        .warmup_steps(100)
        .full_steps(20)
        .max_predict_steps(80)
        .confidence_threshold(0.85)
        .divergence_threshold(3.0)
        .build();

    println!("Hybrid Training Config:");
    println!("  Warmup steps: {}", trainer_config.warmup_steps);
    println!("  Full steps: {}", trainer_config.full_steps);
    println!("  Max predict steps: {}", trainer_config.max_predict_steps);
    println!("  Confidence threshold: {}", trainer_config.confidence_threshold);
    println!();

    // Data configuration - tuned for GPU (100M model fits with batch 2, seq 256)
    #[cfg(feature = "cuda")]
    let batch_size = 2;
    #[cfg(not(feature = "cuda"))]
    let batch_size = 4;

    #[cfg(feature = "cuda")]
    let seq_length = 256; // Moderate sequences for GPU
    #[cfg(not(feature = "cuda"))]
    let seq_length = 512;

    let data_config = DataConfig::default()
        .with_batch_size(batch_size)
        .with_max_seq_length(seq_length);

    println!("Data Config:");
    println!("  Batch size: {}", data_config.batch_size);
    println!("  Sequence length: {}", data_config.max_seq_length);
    println!();

    // Load training data - use larger dataset if available
    let data_paths = [
        PathBuf::from("/home/kang/data/tritter/processed/curated_data.jsonl"),
        PathBuf::from("/home/kang/Documents/projects/github/python-ai/tritter/data/combined/all_data.jsonl"),
    ];

    let data_path = data_paths.iter().find(|p| p.exists()).cloned();
    let data_path = match data_path {
        Some(p) => p,
        None => anyhow::bail!("No training data found. Tried: {:?}", data_paths),
    };

    let dataset = JsonlDataset::new(&data_path, data_config.max_seq_length)?;
    let dataset_box: Box<dyn StreamingDataset> = Box::new(dataset);

    println!("Dataset loaded from: {:?}", data_path);
    println!();

    // Create data loader
    let data_loader = DataLoader::new(dataset_box, data_config.clone(), device.clone());

    // Create trainer
    let learning_rate = 3e-4;
    let mut trainer = create_trainer_with_config(
        &model_config,
        trainer_config,
        learning_rate,
        &device,
    )?;

    println!("Trainer created successfully.");
    println!();

    // Training loop
    let max_steps = 50000; // Production training
    let log_every = 100;

    println!("Starting training ({} steps)...", max_steps);
    println!("{:-<80}", "");
    println!(
        "{:>6} | {:>8} | {:>10} | {:>8} | {:>8} | {:>10} | {:>8}",
        "Step", "Phase", "Loss", "Fwd", "Bwd%", "Tok/s", "ms/step"
    );
    println!("{:-<80}", "");

    let training_start = Instant::now();
    let mut total_forward = 0u64;
    let mut total_backward = 0u64;
    let mut total_tokens = 0u64;
    let mut last_phase = Phase::Warmup;
    let mut step = 0;

    for batch_result in data_loader {
        if step >= max_steps {
            break;
        }

        let step_start = Instant::now();

        let batch = match batch_result {
            Ok(b) => b,
            Err(e) => {
                eprintln!("Warning: Failed to load batch: {}", e);
                continue;
            }
        };

        let batch_tokens = batch.batch_size() * data_config.max_seq_length;

        // Execute training step
        let result = match trainer.step(&batch) {
            Ok(r) => r,
            Err((e, _)) => {
                eprintln!("Warning: Training step failed: {}", e);
                continue;
            }
        };

        // Track forward/backward passes
        total_forward += 1;
        total_tokens += batch_tokens as u64;
        if matches!(result.phase, Phase::Warmup | Phase::Full | Phase::Correct) {
            total_backward += 1;
        }

        let step_time = step_start.elapsed();

        // Log progress
        if step % log_every == 0 || result.phase != last_phase {
            let phase_str = match result.phase {
                Phase::Warmup => "WARMUP",
                Phase::Full => "FULL",
                Phase::Predict => "PREDICT",
                Phase::Correct => "CORRECT",
            };

            let bwd_pct = if total_forward > 0 {
                100.0 * total_backward as f64 / total_forward as f64
            } else {
                100.0
            };

            let elapsed = training_start.elapsed().as_secs_f64();
            let tokens_per_sec = if elapsed > 0.0 {
                total_tokens as f64 / elapsed
            } else {
                0.0
            };

            println!(
                "{:>6} | {:>8} | {:>10.4} | {:>8} | {:>7.1}% | {:>10.0} | {:>7}",
                step,
                phase_str,
                result.loss,
                total_forward,
                bwd_pct,
                tokens_per_sec,
                step_time.as_millis()
            );
        }

        last_phase = result.phase;
        step += 1;
    }

    println!("{:-<80}", "");

    // Final summary
    let total_time = training_start.elapsed();
    let backward_reduction = 100.0 - (100.0 * total_backward as f64 / total_forward as f64);

    println!("\n=== Training Complete ===");
    println!("Total steps: {}", step);
    println!("Total time: {:.1}s", total_time.as_secs_f64());
    println!("Forward passes: {}", total_forward);
    println!("Backward passes: {}", total_backward);
    println!("Backward reduction: {:.1}%", backward_reduction);
    println!("Total tokens: {}", total_tokens);
    println!(
        "Average tokens/sec: {:.0}",
        total_tokens as f64 / total_time.as_secs_f64()
    );

    Ok(())
}
