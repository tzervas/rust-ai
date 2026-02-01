//! Progressive training binary.
//!
//! Usage:
//!   train [OPTIONS]
//!
//! Examples:
//!   # Train just the 100M model
//!   train --start 100m --end 100m --steps 10000
//!
//!   # Full progressive training: 100M → 500M → 1B
//!   train --start 100m --end 1b --steps 50000
//!
//!   # With HuggingFace upload
//!   train --start 100m --end 1b --upload --hf-user myusername

use std::path::PathBuf;

use candle_core::Device;
use clap::Parser;
use tracing_subscriber::EnvFilter;

use training_tools::progressive::{
    DatasetBlendStrategy, ProgressiveConfig, ProgressiveTrainer, TrainingStage,
};

#[derive(Parser)]
#[command(name = "train")]
#[command(about = "Progressive parameter expansion training for Tritter models")]
#[command(version)]
struct Args {
    /// Starting model size (100m, 500m, 1b)
    #[arg(short = 's', long, default_value = "100m")]
    start: String,

    /// Ending model size (100m, 500m, 1b)
    #[arg(short = 'e', long, default_value = "1b")]
    end: String,

    /// Training steps per stage
    #[arg(long, default_value = "10000")]
    steps: u64,

    /// Batch size
    #[arg(short = 'b', long, default_value = "4")]
    batch_size: usize,

    /// Sequence length
    #[arg(long, default_value = "512")]
    seq_length: usize,

    /// Learning rate
    #[arg(short = 'l', long, default_value = "0.0003")]
    learning_rate: f32,

    /// Checkpoint every N steps
    #[arg(long, default_value = "1000")]
    checkpoint_every: u64,

    /// Log every N steps
    #[arg(long, default_value = "10")]
    log_every: u64,

    /// Gradient checkpoint interval (layers)
    #[arg(long, default_value = "4")]
    grad_checkpoint_interval: usize,

    /// Directory for training runs
    #[arg(short = 'o', long, default_value = "./runs")]
    runs_dir: PathBuf,

    /// Upload to HuggingFace after each stage
    #[arg(long)]
    upload: bool,

    /// HuggingFace username (required if --upload)
    #[arg(long)]
    hf_user: Option<String>,

    /// Delete local checkpoints after HuggingFace upload
    #[arg(long)]
    delete_after_upload: bool,

    /// Use CUDA (enabled by default, use --no-cuda to disable)
    #[arg(long, default_value = "true", action = clap::ArgAction::Set)]
    cuda: bool,

    /// Disable CUDA and use CPU instead
    #[arg(long)]
    no_cuda: bool,

    /// CUDA device index
    #[arg(long, default_value = "0")]
    cuda_device: usize,

    /// Paths to datasets (JSONL or Parquet files/directories)
    /// Can be specified multiple times for multi-dataset training
    /// If not provided, uses random tokens (testing mode only)
    ///
    /// Example: --dataset /data/code --dataset /data/instruction --dataset /data/alignment
    #[arg(short = 'd', long, action = clap::ArgAction::Append)]
    dataset: Vec<PathBuf>,

    /// Dataset blending strategy when using multiple datasets
    /// - sequential: Process each dataset fully before moving to next
    /// - roundrobin: Cycle through datasets alternating examples
    /// - weighted: Sample proportionally (default weights are equal)
    #[arg(long, default_value = "roundrobin")]
    blend_strategy: String,

    /// Text column name in dataset (auto-detected if not provided)
    #[arg(long)]
    text_column: Option<String>,
}

fn parse_stage(s: &str) -> anyhow::Result<TrainingStage> {
    match s.to_lowercase().as_str() {
        "100m" | "small" => Ok(TrainingStage::Small100M),
        "500m" | "medium" => Ok(TrainingStage::Medium500M),
        "1b" | "large" => Ok(TrainingStage::Large1B),
        _ => anyhow::bail!("Unknown model size: {}. Use 100m, 500m, or 1b", s),
    }
}

fn main() -> anyhow::Result<()> {
    // Parse arguments
    let args = Args::parse();

    // Setup logging
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::from_default_env()
                .add_directive("info".parse()?)
                .add_directive("training_tools=debug".parse()?),
        )
        .init();

    // Parse stages
    let start_stage = parse_stage(&args.start)?;
    let end_stage = parse_stage(&args.end)?;

    // Validate stage ordering
    let stages = [
        TrainingStage::Small100M,
        TrainingStage::Medium500M,
        TrainingStage::Large1B,
    ];
    let start_idx = stages.iter().position(|s| *s == start_stage).unwrap();
    let end_idx = stages.iter().position(|s| *s == end_stage).unwrap();
    if start_idx > end_idx {
        anyhow::bail!(
            "Start stage ({}) must be <= end stage ({})",
            args.start,
            args.end
        );
    }

    // Validate HuggingFace config
    if args.upload && args.hf_user.is_none() {
        anyhow::bail!("--hf-user is required when --upload is specified");
    }

    // Select device - CUDA by default
    let use_cuda = args.cuda && !args.no_cuda;
    let device = if use_cuda {
        #[cfg(feature = "cuda")]
        {
            tracing::info!("Initializing CUDA device {}...", args.cuda_device);
            Device::new_cuda(args.cuda_device)?
        }
        #[cfg(not(feature = "cuda"))]
        {
            tracing::error!("CUDA requested but not compiled with cuda feature!");
            tracing::error!(
                "Rebuild with: cargo build -p training-tools --release --features cuda"
            );
            anyhow::bail!(
                "CUDA feature not enabled. Use --no-cuda to run on CPU (not recommended)."
            );
        }
    } else {
        tracing::warn!("Running on CPU - this will be very slow!");
        tracing::warn!("Use CUDA for reasonable training speed.");
        Device::Cpu
    };

    tracing::info!("=== Rust-AI Progressive Training ===");
    tracing::info!("Device: {:?}", device);
    tracing::info!(
        "Training: {} → {}",
        start_stage.size_str(),
        end_stage.size_str()
    );
    tracing::info!("Steps per stage: {}", args.steps);
    tracing::info!("Batch size: {}", args.batch_size);
    tracing::info!("Sequence length: {}", args.seq_length);
    tracing::info!("Learning rate: {:.2e}", args.learning_rate);
    tracing::info!(
        "Gradient checkpointing: every {} layers",
        args.grad_checkpoint_interval
    );
    if args.upload {
        tracing::info!(
            "HuggingFace upload: enabled (user: {})",
            args.hf_user.as_ref().unwrap()
        );
    }
    if !args.dataset.is_empty() {
        tracing::info!("Datasets ({}):", args.dataset.len());
        for (i, ds) in args.dataset.iter().enumerate() {
            tracing::info!("  [{}] {:?}", i + 1, ds);
        }
        tracing::info!("Blend strategy: {}", args.blend_strategy);
    } else {
        tracing::warn!("No dataset provided - using random tokens (testing mode)");
    }
    tracing::info!("");

    // Create runs directory
    std::fs::create_dir_all(&args.runs_dir)?;

    // Parse blend strategy
    let blend_strategy = match args.blend_strategy.to_lowercase().as_str() {
        "sequential" | "seq" => DatasetBlendStrategy::Sequential,
        "roundrobin" | "rr" | "round-robin" => DatasetBlendStrategy::RoundRobin,
        "weighted" | "w" => DatasetBlendStrategy::Weighted,
        _ => {
            tracing::warn!(
                "Unknown blend strategy '{}', defaulting to RoundRobin",
                args.blend_strategy
            );
            DatasetBlendStrategy::RoundRobin
        }
    };

    // Create progressive config
    let config = ProgressiveConfig {
        start_stage,
        end_stage,
        steps_per_stage: args.steps,
        batch_size: args.batch_size,
        seq_length: args.seq_length,
        learning_rate: args.learning_rate,
        checkpoint_every: args.checkpoint_every,
        log_every: args.log_every,
        gradient_checkpoint_interval: args.grad_checkpoint_interval,
        upload_to_hf: args.upload,
        hf_username: args.hf_user,
        delete_after_upload: args.delete_after_upload,
        dataset_path: None, // Deprecated, use dataset_paths
        dataset_paths: args.dataset,
        dataset_blend_strategy: blend_strategy,
        text_column: args.text_column,
    };

    // Create and run trainer
    let mut trainer = ProgressiveTrainer::new(config, args.runs_dir, device)?;
    trainer.run()?;

    tracing::info!("Training complete!");
    Ok(())
}
