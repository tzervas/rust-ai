//! Train 100M Tritter model with hybrid predictive training.
//!
//! This example demonstrates production training with:
//! - Real tokenization via HuggingFace tokenizers
//! - WSD (Warmup-Stable-Decay) learning rate scheduling
//! - Safetensors checkpointing
//! - JSONL/Parquet data loading
//! - Hybrid predictive training phases
//! - Train-monitor compatible metrics output
//!
//! # Usage
//!
//! ```bash
//! # CPU training (slow, for testing)
//! cargo run -p tritter-model-rs --example train_100m --release
//!
//! # GPU training (recommended)
//! cargo run -p tritter-model-rs --example train_100m --release --features cuda
//!
//! # With custom data path
//! cargo run -p tritter-model-rs --example train_100m --release --features cuda -- \
//!     --data /data/datasets/tritter/iac/combined/iac_all_enriched.jsonl \
//!     --tokenizer tokenizer.json \
//!     --steps 100000
//! ```
//!
//! # Data Paths (Homelab)
//!
//! - IaC Enriched: `/data/datasets/tritter/iac/combined/iac_all_enriched.jsonl` (93MB)
//! - Code: `/data/datasets/tritter/pretrain/code/` (204GB)
//! - Alignment: `/data/datasets/tritter/alignment/` (764MB)
//! - Instruction: `/data/datasets/tritter/instruction/` (13GB)

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use candle_core::Device;
use chrono::Utc;
use tokenizers::Tokenizer;

use hybrid_predict_trainer_rs::{HybridTrainerConfig, Phase};
use training_tools::lr_scheduler::{LRScheduler, WSDScheduler};
use training_tools::{TrainingConfig, TrainingRun, TrainingStatus};
use tritter_model_rs::{
    data::{create_data_loader, DataConfig, JsonlDataset},
    trainer::create_trainer_with_config,
    TritterBatch, TritterConfig,
};

/// Command-line arguments
struct Args {
    /// Path to training data (JSONL or directory)
    data_path: PathBuf,
    /// Path to tokenizer.json
    tokenizer_path: Option<PathBuf>,
    /// Total training steps
    total_steps: u64,
    /// Batch size
    batch_size: usize,
    /// Maximum sequence length
    max_seq_length: usize,
    /// Checkpoint save interval (steps)
    checkpoint_interval: usize,
    /// Output directory for checkpoints
    output_dir: PathBuf,
    /// Peak learning rate
    peak_lr: f32,
    /// Minimum learning rate
    min_lr: f32,
    /// Number of data loading workers
    num_workers: usize,
    /// Model size preset
    model_size: String,
    /// Run name for logging
    run_name: String,
}

/// Convert hybrid-predict-trainer Phase to lowercase string for metrics
fn phase_to_string(phase: Phase) -> &'static str {
    match phase {
        Phase::Warmup => "warmup",
        Phase::Full => "full",
        Phase::Predict => "predict",
        Phase::Correct => "correct",
    }
}

/// Convert Phase to TrainingPhase for run state
fn phase_to_training_phase(phase: Phase) -> training_tools::training_state::TrainingPhase {
    match phase {
        Phase::Warmup => training_tools::training_state::TrainingPhase::Warmup,
        Phase::Full => training_tools::training_state::TrainingPhase::Full,
        Phase::Predict => training_tools::training_state::TrainingPhase::Predict,
        Phase::Correct => training_tools::training_state::TrainingPhase::Correct,
    }
}

impl Default for Args {
    fn default() -> Self {
        Self {
            data_path: PathBuf::from("/data/datasets/tritter/iac/combined/iac_all_enriched.jsonl"),
            tokenizer_path: None,
            total_steps: 100_000,
            batch_size: 8,
            max_seq_length: 2048,
            checkpoint_interval: 1000,
            output_dir: PathBuf::from("./runs"),
            peak_lr: 3e-4,
            min_lr: 3e-6,
            num_workers: 4,
            model_size: "100m".to_string(),
            run_name: "tritter".to_string(),
        }
    }
}

fn parse_args() -> Args {
    let mut args = Args::default();
    let mut argv = std::env::args().skip(1).peekable();

    while let Some(arg) = argv.next() {
        match arg.as_str() {
            "--data" | "-d" => {
                if let Some(path) = argv.next() {
                    args.data_path = PathBuf::from(path);
                }
            }
            "--tokenizer" | "-t" => {
                if let Some(path) = argv.next() {
                    args.tokenizer_path = Some(PathBuf::from(path));
                }
            }
            "--steps" | "-s" => {
                if let Some(s) = argv.next() {
                    args.total_steps = s.parse().unwrap_or(args.total_steps);
                }
            }
            "--batch-size" | "-b" => {
                if let Some(s) = argv.next() {
                    args.batch_size = s.parse().unwrap_or(args.batch_size);
                }
            }
            "--seq-len" => {
                if let Some(s) = argv.next() {
                    args.max_seq_length = s.parse().unwrap_or(args.max_seq_length);
                }
            }
            "--checkpoint-interval" => {
                if let Some(s) = argv.next() {
                    args.checkpoint_interval = s.parse().unwrap_or(args.checkpoint_interval);
                }
            }
            "--output" | "-o" => {
                if let Some(path) = argv.next() {
                    args.output_dir = PathBuf::from(path);
                }
            }
            "--lr" => {
                if let Some(s) = argv.next() {
                    args.peak_lr = s.parse().unwrap_or(args.peak_lr);
                }
            }
            "--min-lr" => {
                if let Some(s) = argv.next() {
                    args.min_lr = s.parse().unwrap_or(args.min_lr);
                }
            }
            "--workers" | "-w" => {
                if let Some(s) = argv.next() {
                    args.num_workers = s.parse().unwrap_or(args.num_workers);
                }
            }
            "--model" | "-m" => {
                if let Some(s) = argv.next() {
                    args.model_size = s;
                }
            }
            "--name" | "-n" => {
                if let Some(s) = argv.next() {
                    args.run_name = s;
                }
            }
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            _ => {
                eprintln!("Unknown argument: {}", arg);
            }
        }
    }

    args
}

fn print_help() {
    println!(
        r#"Train 100M Tritter model with hybrid predictive training

USAGE:
    train_100m [OPTIONS]

OPTIONS:
    -d, --data <PATH>           Path to training data (JSONL or directory)
    -t, --tokenizer <PATH>      Path to tokenizer.json
    -s, --steps <N>             Total training steps [default: 100000]
    -b, --batch-size <N>        Batch size [default: 8]
        --seq-len <N>           Maximum sequence length [default: 2048]
        --checkpoint-interval   Steps between checkpoints [default: 1000]
    -o, --output <PATH>         Output directory for runs [default: ./runs]
        --lr <FLOAT>            Peak learning rate [default: 3e-4]
        --min-lr <FLOAT>        Minimum learning rate [default: 3e-6]
    -w, --workers <N>           Number of data loading workers [default: 4]
    -m, --model <SIZE>          Model size: test, 100m, 500m, 1b [default: 100m]
    -n, --name <NAME>           Run name for logging [default: tritter]
    -h, --help                  Print this help message

EXAMPLES:
    # Train on IaC data
    train_100m --data /data/datasets/tritter/iac/combined/iac_all_enriched.jsonl

    # Train with custom tokenizer
    train_100m --tokenizer gpt2_tokenizer.json --steps 50000

    # Train larger model with custom run name
    train_100m --model 500m --batch-size 4 --seq-len 1024 --name tritter_500m
"#
    );
}

fn main() -> anyhow::Result<()> {
    let args = parse_args();

    println!("=== Tritter 100M Training ===\n");

    // Device selection
    #[cfg(feature = "cuda")]
    let device = Device::new_cuda(0)?;
    #[cfg(not(feature = "cuda"))]
    let device = Device::Cpu;

    println!("Device: {:?}", device);

    // Model configuration
    let model_config = match args.model_size.as_str() {
        "test" => TritterConfig::test(),
        "100m" => TritterConfig::small_100m(),
        "500m" => TritterConfig::medium_500m(),
        "1b" => TritterConfig::large_1b(),
        _ => {
            eprintln!("Unknown model size: {}. Using 100m.", args.model_size);
            TritterConfig::small_100m()
        }
    };

    println!("Model: {}", args.model_size);
    println!(
        "Parameters: ~{:.1}M",
        model_config.parameter_count() as f64 / 1e6
    );
    println!(
        "Hidden: {}, Layers: {}, Heads: {}",
        model_config.hidden_size, model_config.num_layers, model_config.num_heads
    );
    println!();

    // Load tokenizer if provided
    let tokenizer = if let Some(path) = &args.tokenizer_path {
        println!("Loading tokenizer from: {}", path.display());
        Some(Arc::new(Tokenizer::from_file(path).map_err(|e| {
            anyhow::anyhow!("Failed to load tokenizer: {}", e)
        })?))
    } else {
        println!("No tokenizer provided, using byte tokenization (not recommended for production)");
        None
    };

    // Data configuration - use model's max_seq_length to ensure alignment
    let effective_seq_len = args.max_seq_length.min(model_config.max_seq_length);
    let data_config = DataConfig {
        batch_size: args.batch_size,
        max_seq_length: effective_seq_len,
        num_workers: args.num_workers,
        prefetch_factor: 2,
        text_column: None,
        seed: 42,
    };

    println!("Data path: {}", args.data_path.display());
    println!(
        "Batch size: {}, Seq len: {} (model max: {})",
        args.batch_size, effective_seq_len, model_config.max_seq_length
    );
    println!();

    // Create data loader
    let mut loader = if let Some(tok) = tokenizer.clone() {
        // Use JsonlDataset with tokenizer
        let dataset = JsonlDataset::new(&args.data_path, effective_seq_len)?.with_tokenizer(tok);
        tritter_model_rs::data::DataLoader::new(
            Box::new(dataset),
            data_config.clone(),
            device.clone(),
        )
    } else {
        // Use auto-detected loader (JSONL or Parquet)
        create_data_loader(&args.data_path, data_config.clone(), device.clone())?
    };

    // Hybrid training configuration - tuned for ~60% backward reduction
    // Key insight: aggressive prediction early, more validation later
    let trainer_config = HybridTrainerConfig::builder()
        .warmup_steps(50) // Shorter warmup to start predicting faster
        .full_steps(10) // Fewer full steps per cycle (was 20)
        .max_predict_steps(150) // Longer prediction phases (was 80)
        .confidence_threshold(0.70) // More aggressive prediction (was 0.85)
        .divergence_threshold(5.0) // More tolerant of deviation (was 3.0)
        .build();

    println!("Hybrid Training Config:");
    println!("  Warmup: {} steps", trainer_config.warmup_steps);
    println!("  Full: {} steps per cycle", trainer_config.full_steps);
    println!("  Max predict: {} steps", trainer_config.max_predict_steps);
    println!();

    // Create trainer
    let mut trainer =
        create_trainer_with_config(&model_config, trainer_config, args.peak_lr, &device)?;

    // Learning rate scheduler (WSD)
    let lr_scheduler = WSDScheduler::builder()
        .total_steps(args.total_steps)
        .peak_lr(args.peak_lr)
        .min_lr(args.min_lr)
        .warmup_fraction(0.01) // 1% warmup
        .decay_fraction(0.19) // 19% decay, 80% stable
        .build()?;

    println!("LR Schedule (WSD):");
    println!("  Peak LR: {:.2e}", args.peak_lr);
    println!("  Min LR: {:.2e}", args.min_lr);
    println!("  Warmup: {} steps", lr_scheduler.warmup_steps());
    println!("  Stable: {} steps", lr_scheduler.stable_steps());
    println!("  Decay: {} steps", lr_scheduler.decay_steps());
    println!();

    // Create run directory with timestamp
    let timestamp = Utc::now().format("%Y%m%d_%H%M%S");
    let run_dir_name = format!("{}_{}", args.run_name, timestamp);
    let run_dir = args.output_dir.join(&run_dir_name);
    fs::create_dir_all(&run_dir)?;
    println!("Run directory: {}", run_dir.display());

    // Device string for config
    #[cfg(feature = "cuda")]
    let device_str = "cuda:0".to_string();
    #[cfg(not(feature = "cuda"))]
    let device_str = "cpu".to_string();

    // Initialize TrainingRun for monitor compatibility
    let training_config = TrainingConfig {
        config_version: 1,
        config_hash: String::new(),
        model_size: args.model_size.clone(),
        num_parameters: model_config.parameter_count(),
        hidden_size: model_config.hidden_size,
        num_layers: model_config.num_layers,
        num_heads: model_config.num_heads,
        max_seq_length: effective_seq_len,
        batch_size: args.batch_size,
        learning_rate: args.peak_lr,
        max_steps: args.total_steps,
        gradient_checkpointing: false,
        checkpoint_interval: args.checkpoint_interval,
        device: device_str,
    };

    let mut training_run = TrainingRun::new(&args.run_name, training_config, run_dir.clone());
    training_run.status = TrainingStatus::Running;
    training_run.save()?;

    // Create metrics file
    let metrics_file_path = run_dir.join("metrics.jsonl");
    println!("Metrics: {}", metrics_file_path.display());
    println!();

    // Training loop
    println!("Starting training for {} steps...", args.total_steps);
    println!("{:-<80}", "");
    println!(
        "{:>6} | {:>8} | {:>10} | {:>10} | {:>8} | {:>8} | {:>8}",
        "Step", "Phase", "Loss", "LR", "Fwd", "Bwd%", "ms"
    );
    println!("{:-<80}", "");

    let mut total_forward = 0u64;
    let mut total_backward = 0u64;
    let mut last_phase = Phase::Warmup;
    let mut step = 0u64;
    let mut best_loss = f32::INFINITY;
    let training_start = Instant::now();

    // Fail-fast error tracking
    const MAX_CONSECUTIVE_ERRORS: u32 = 5;
    let mut consecutive_errors: u32 = 0;
    let mut consecutive_data_errors: u32 = 0;

    'training: loop {
        // Reset data loader if exhausted (epoch boundary)
        for batch_result in &mut loader {
            let batch = match batch_result {
                Ok(b) => {
                    consecutive_data_errors = 0; // Reset on success
                    b
                }
                Err(e) => {
                    consecutive_data_errors += 1;
                    eprintln!(
                        "[ERROR] Data loading error at step {}: {}. Skipping. ({}/{})",
                        step, e, consecutive_data_errors, MAX_CONSECUTIVE_ERRORS
                    );
                    if consecutive_data_errors >= MAX_CONSECUTIVE_ERRORS {
                        eprintln!(
                            "\n[FATAL] {} consecutive data loading errors. Aborting training.",
                            MAX_CONSECUTIVE_ERRORS
                        );
                        eprintln!("Data path: {}", args.data_path.display());
                        std::process::exit(1);
                    }
                    continue;
                }
            };

            if step >= args.total_steps {
                break 'training;
            }

            let step_start = Instant::now();

            // Update learning rate
            let lr = lr_scheduler.get_lr(step);
            trainer.set_learning_rate::<TritterBatch>(lr);

            // Execute training step
            let result = match trainer.step(&batch) {
                Ok(r) => {
                    consecutive_errors = 0; // Reset on success
                    r
                }
                Err((e, _)) => {
                    consecutive_errors += 1;
                    eprintln!(
                        "[ERROR] Training error at step {}: {} ({}/{})",
                        step, e, consecutive_errors, MAX_CONSECUTIVE_ERRORS
                    );
                    eprintln!("  Batch input_ids shape: {:?}", batch.input_ids.dims());
                    if let Some(ref mask) = batch.attention_mask {
                        eprintln!("  Batch attention_mask shape: {:?}", mask.dims());
                    }
                    if consecutive_errors >= MAX_CONSECUTIVE_ERRORS {
                        eprintln!(
                            "\n[FATAL] {} consecutive forward pass errors. Aborting training.",
                            MAX_CONSECUTIVE_ERRORS
                        );
                        eprintln!("Last error: {}", e);
                        eprintln!("Step: {}, Total forward passes: {}", step, total_forward);
                        std::process::exit(1);
                    }
                    continue;
                }
            };

            // Track forward/backward passes
            total_forward += 1;
            if matches!(result.phase, Phase::Warmup | Phase::Full | Phase::Correct) {
                total_backward += 1;
            }

            // Track best loss
            if result.loss < best_loss && result.loss.is_finite() {
                best_loss = result.loss;
            }

            let step_time_ms = step_start.elapsed().as_millis();

            // Write metrics to JSONL file (after every step)
            let was_predicted = matches!(result.phase, Phase::Predict);
            let prediction_error_str = match result.prediction_error {
                Some(e) => format!("{}", e),
                None => "null".to_string(),
            };
            let metrics_json = format!(
                r#"{{"step":{},"loss":{},"gradient_norm":{},"phase":"{}","was_predicted":{},"prediction_error":{},"step_time_ms":{},"timestamp":"{}"}}"#,
                step,
                if result.loss.is_finite() {
                    result.loss
                } else {
                    0.0
                },
                result.gradient_norm,
                phase_to_string(result.phase),
                was_predicted,
                prediction_error_str,
                step_time_ms,
                Utc::now().to_rfc3339()
            );

            // Append to metrics file
            if let Ok(mut file) = OpenOptions::new()
                .create(true)
                .append(true)
                .open(&metrics_file_path)
            {
                let _ = writeln!(file, "{}", metrics_json);
            }

            // Update run state periodically (every 100 steps) and on phase change
            if step % 100 == 0 || result.phase != last_phase {
                // Track phase transitions
                if result.phase != last_phase && step > 0 {
                    training_run.record_phase_transition(
                        phase_to_training_phase(last_phase),
                        phase_to_training_phase(result.phase),
                    );
                }

                training_run.current_step = step;
                training_run.current_loss = result.loss;
                training_run.current_phase = phase_to_training_phase(result.phase);
                training_run.total_forward = total_forward;
                training_run.total_backward = total_backward;
                if result.loss < training_run.best_loss && result.loss.is_finite() {
                    training_run.best_loss = result.loss;
                    training_run.best_step = step;
                }
                let elapsed_secs = training_start.elapsed().as_secs_f64();
                if elapsed_secs > 0.0 {
                    training_run.steps_per_second = Some(step as f64 / elapsed_secs);
                }
                let _ = training_run.save();
            }

            // Log progress to console
            if step % 100 == 0 || result.phase != last_phase {
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

                println!(
                    "{:>6} | {:>8} | {:>10.4} | {:>10.2e} | {:>7} | {:>6.1}% | {:>7}",
                    step, phase_str, result.loss, lr, total_forward, bwd_pct, step_time_ms
                );
            }

            // Save checkpoint
            if step > 0 && step % args.checkpoint_interval as u64 == 0 {
                let checkpoint_path = run_dir.join(format!("step_{}.safetensors", step));
                match trainer.model().inner().save_safetensors(&checkpoint_path) {
                    Ok(()) => {
                        println!(">>> Checkpoint saved: {}", checkpoint_path.display());
                        training_run.latest_checkpoint = Some(checkpoint_path);
                        let _ = training_run.save();
                    }
                    Err(e) => eprintln!(">>> Checkpoint save failed: {}", e),
                }
            }

            last_phase = result.phase;
            step += 1;
        }

        // If we get here, the data loader is exhausted - reset for next epoch
        if step < args.total_steps {
            println!(">>> Epoch complete, resetting data loader...");
            loader.reset()?;
        } else {
            break 'training;
        }
    }

    println!("{:-<80}", "");

    // Training summary
    let total_time = training_start.elapsed();
    let backward_reduction = 100.0 - (100.0 * total_backward as f64 / total_forward as f64);

    println!("\n=== Training Complete ===");
    println!("Total steps: {}", step);
    println!(
        "Total time: {:.1}s ({:.2} steps/s)",
        total_time.as_secs_f64(),
        step as f64 / total_time.as_secs_f64()
    );
    println!("Forward passes: {}", total_forward);
    println!("Backward passes: {}", total_backward);
    println!("Backward reduction: {:.1}%", backward_reduction);
    println!("Best loss: {:.4}", best_loss);

    // Save final model
    let final_path = run_dir.join("model_final.safetensors");
    trainer.model().inner().save_safetensors(&final_path)?;
    println!("\nFinal model saved: {}", final_path.display());

    // Update final run state
    training_run.status = TrainingStatus::Completed;
    training_run.current_step = step;
    training_run.total_forward = total_forward;
    training_run.total_backward = total_backward;
    training_run.ended_at = Some(Utc::now());
    training_run.latest_checkpoint = Some(final_path);
    if total_time.as_secs_f64() > 0.0 {
        training_run.steps_per_second = Some(step as f64 / total_time.as_secs_f64());
    }
    training_run.save()?;
    println!(
        "Run state saved: {}",
        run_dir.join("run_state.json").display()
    );

    Ok(())
}
