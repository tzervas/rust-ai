//! Tritter training example using hybrid predictive training.
//!
//! This example demonstrates:
//! - Creating a Tritter model
//! - Integrating with hybrid-predict-trainer-rs
//! - Phase-based predictive training (WARMUP → FULL → PREDICT → CORRECT)
//!
//! Usage:
//!   cargo run --example train --release
//!
//! With CUDA:
//!   cargo run --example train --release --features cuda

use std::time::Instant;

use candle_core::{Device, Tensor};
use rand::Rng;

use tritter_model_rs::{
    TritterConfig, TritterBatch,
    trainer::create_trainer_with_config,
};
use hybrid_predict_trainer_rs::{HybridTrainerConfig, phases::Phase};

fn main() -> anyhow::Result<()> {
    println!("=== Tritter Rust Training ===\n");

    // Select device
    #[cfg(feature = "cuda")]
    let device = Device::new_cuda(0)?;
    #[cfg(not(feature = "cuda"))]
    let device = Device::Cpu;

    println!("Device: {:?}", device);

    // Model configuration
    let model_size = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "test".to_string());

    let model_config = match model_size.as_str() {
        "100m" => TritterConfig::small_100m(),
        "500m" => TritterConfig::medium_500m(),
        "1b" => TritterConfig::large_1b(),
        _ => TritterConfig::test(),
    };

    println!("Model size: {}", model_size);
    println!("Parameters: ~{:.1}M", model_config.parameter_count() as f64 / 1e6);
    println!("Hidden size: {}", model_config.hidden_size);
    println!("Layers: {}", model_config.num_layers);
    println!("Heads: {}", model_config.num_heads);
    println!();

    // Training configuration
    let trainer_config = HybridTrainerConfig::builder()
        .warmup_steps(50)        // Shorter warmup for demo
        .full_steps(20)          // Standard full training per cycle
        .max_predict_steps(80)   // Up to 80 steps with prediction
        .confidence_threshold(0.85)
        .build();

    println!("Hybrid Training Config:");
    println!("  Warmup steps: {}", trainer_config.warmup_steps);
    println!("  Full steps: {}", trainer_config.full_steps);
    println!("  Max predict steps: {}", trainer_config.max_predict_steps);
    println!();

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

    // Training parameters
    let batch_size = 4;
    let seq_len = 64;
    let max_steps = 500;

    // Training loop
    println!("Starting training...");
    println!("{:-<70}", "");
    println!("{:>6} | {:>8} | {:>10} | {:>8} | {:>8} | {:>8}",
             "Step", "Phase", "Loss", "Fwd", "Bwd", "ms/step");
    println!("{:-<70}", "");

    let mut total_forward = 0;
    let mut total_backward = 0;
    let mut last_phase = Phase::Warmup;

    let mut rng = rand::thread_rng();

    for step in 0..max_steps {
        let step_start = Instant::now();

        // Generate random batch (in production, load from dataset)
        let vocab_size = model_config.vocab_size;
        let random_ids: Vec<u32> = (0..batch_size * seq_len)
            .map(|_| rng.gen_range(0..vocab_size as u32))
            .collect();
        let input_ids = Tensor::from_slice(&random_ids, (batch_size, seq_len), &device)?;

        let batch = TritterBatch::new(input_ids, None);

        // Execute training step
        let result = trainer.step(&batch)
            .map_err(|(e, _)| anyhow::anyhow!("Training step failed: {}", e))?;

        // Track forward/backward passes
        total_forward += 1;
        if matches!(result.phase, Phase::Warmup | Phase::Full | Phase::Correct) {
            total_backward += 1;
        }

        let step_time = step_start.elapsed().as_millis();

        // Log every 10 steps or on phase change
        if step % 10 == 0 || result.phase != last_phase {
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

            println!("{:>6} | {:>8} | {:>10.4} | {:>7} | {:>6.1}% | {:>7}",
                     step, phase_str, result.loss, total_forward, bwd_pct, step_time);
        }

        last_phase = result.phase;
    }

    println!("{:-<70}", "");

    // Summary
    let backward_reduction = 100.0 - (100.0 * total_backward as f64 / total_forward as f64);
    println!("\n=== Training Summary ===");
    println!("Total steps: {}", max_steps);
    println!("Forward passes: {}", total_forward);
    println!("Backward passes: {}", total_backward);
    println!("Backward reduction: {:.1}%", backward_reduction);

    Ok(())
}
