//! Fresh training restart script with proper multi-dataset loading.
//!
//! This script demonstrates training from scratch with the specialist_100m preset,
//! loading data from multiple sources in a structured order:
//! 1. IaC (Infrastructure-as-Code) - smallest, for validation
//! 2. Alignment - instruction following
//! 3. Code - large GitHub corpus
//! 4. Instruction - additional supervised data
//!
//! Run with:
//! ```bash
//! cargo run -p training-tools --example train_fresh --release --features cuda
//! ```

use std::path::PathBuf;

use candle_core::Device;
use training_tools::lr_scheduler::WSDSchedulerBuilder;
use training_tools::TrainingPreset;
use tritter_model_rs::data::{create_data_loader, DataConfig};
use tritter_model_rs::{TritterConfig, TritterModel};

/// Dataset configuration with path and priority.
#[derive(Debug, Clone)]
struct DatasetSource {
    name: &'static str,
    path: PathBuf,
    priority: u8, // Lower = higher priority (train first)
    estimated_size_gb: f64,
}

fn main() -> anyhow::Result<()> {
    // Setup logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("info".parse()?)
                .add_directive("training_tools=debug".parse()?)
                .add_directive("tritter_model_rs=debug".parse()?),
        )
        .init();

    println!("{}", "=".repeat(80));
    println!("TRITTER FRESH TRAINING START");
    println!("{}", "=".repeat(80));
    println!();

    // ========================================================================
    // 1. LOAD SPECIALIST PRESET
    // ========================================================================

    let preset = TrainingPreset::specialist_100m();
    println!("{}", preset.summary());

    // ========================================================================
    // 2. CONFIGURE DATASETS
    // ========================================================================

    let datasets = vec![
        DatasetSource {
            name: "IaC Enriched",
            path: PathBuf::from("/data/datasets/tritter/iac/combined/"),
            priority: 1,
            estimated_size_gb: 0.093, // 93MB
        },
        DatasetSource {
            name: "Alignment",
            path: PathBuf::from("/data/datasets/tritter/alignment/"),
            priority: 2,
            estimated_size_gb: 0.764, // 764MB
        },
        DatasetSource {
            name: "Instruction",
            path: PathBuf::from("/data/datasets/tritter/instruction/"),
            priority: 3,
            estimated_size_gb: 13.0, // 13GB
        },
        DatasetSource {
            name: "Code (GitHub)",
            path: PathBuf::from("/data/datasets/tritter/pretrain/code/"),
            priority: 4,
            estimated_size_gb: 204.0, // 204GB
        },
    ];

    println!("\n{}", "=".repeat(80));
    println!("DATASET CONFIGURATION");
    println!("{}", "=".repeat(80));
    println!();
    println!("| Priority | Name | Path | Size (GB) |");
    println!("|----------|------|------|-----------|");
    for ds in &datasets {
        println!(
            "| {} | {} | {} | {:.2} |",
            ds.priority,
            ds.name,
            ds.path.display(),
            ds.estimated_size_gb
        );
    }
    println!();

    // Verify dataset paths exist
    println!("Verifying dataset paths...");
    let mut verified_datasets = Vec::new();
    for ds in &datasets {
        if ds.path.exists() {
            println!("  ✓ {} found at {:?}", ds.name, ds.path);
            verified_datasets.push(ds.clone());
        } else {
            println!("  ✗ {} NOT FOUND at {:?} (skipping)", ds.name, ds.path);
        }
    }

    if verified_datasets.is_empty() {
        anyhow::bail!("No datasets found! Check your paths.");
    }

    println!("\nProceeding with {} datasets.", verified_datasets.len());
    println!();

    // ========================================================================
    // 3. DEVICE INITIALIZATION
    // ========================================================================

    #[cfg(feature = "cuda")]
    let device = {
        println!("Initializing CUDA device 0...");
        Device::new_cuda(0)?
    };

    #[cfg(not(feature = "cuda"))]
    let device = {
        println!("WARNING: Running on CPU - this will be VERY slow!");
        println!("Rebuild with --features cuda for GPU acceleration.");
        Device::Cpu
    };

    println!("Device: {:?}", device);
    println!();

    // ========================================================================
    // 4. MODEL INITIALIZATION
    // ========================================================================

    println!("{}", "=".repeat(80));
    println!("MODEL INITIALIZATION");
    println!("{}", "=".repeat(80));
    println!();

    let model_config = TritterConfig {
        vocab_size: preset.model.vocab_size,
        hidden_size: preset.model.hidden_size,
        num_layers: preset.model.num_layers,
        num_heads: preset.model.num_heads,
        intermediate_size: preset.model.intermediate_size,
        max_position_embeddings: preset.model.max_seq_length,
        dropout: preset.model.dropout as f64,
        layer_norm_eps: 1e-5,
        gradient_checkpointing: preset.model.gradient_checkpointing,
        checkpoint_every_n_layers: preset.model.checkpoint_every_n_layers,
    };

    println!("Creating model with config:");
    println!("  Hidden size: {}", model_config.hidden_size);
    println!("  Layers: {}", model_config.num_layers);
    println!("  Heads: {}", model_config.num_heads);
    println!("  Vocab size: {}", model_config.vocab_size);
    println!(
        "  Max position embeddings: {}",
        model_config.max_position_embeddings
    );
    println!("  Dropout: {:.3}", model_config.dropout);
    println!(
        "  Gradient checkpointing: {} (every {} layers)",
        model_config.gradient_checkpointing, model_config.checkpoint_every_n_layers
    );
    println!();

    let model = TritterModel::new(&model_config, &device)?;
    println!("✓ Model initialized successfully");
    println!();

    // ========================================================================
    // 5. DATA LOADER SETUP
    // ========================================================================

    println!("{}", "=".repeat(80));
    println!("DATA LOADER CONFIGURATION");
    println!("{}", "=".repeat(80));
    println!();

    // Start with IaC dataset (smallest, priority 1)
    let primary_dataset = verified_datasets
        .iter()
        .min_by_key(|ds| ds.priority)
        .unwrap();

    println!(
        "Primary dataset: {} ({:.2} GB)",
        primary_dataset.name, primary_dataset.estimated_size_gb
    );
    println!("Path: {:?}", primary_dataset.path);
    println!();

    let data_config = DataConfig {
        max_seq_length: preset.data.seq_length,
        batch_size: preset.data.micro_batch_size,
        num_workers: preset.data.num_workers,
        prefetch_factor: 2,
        text_column: None, // Auto-detect
        seed: 42,
    };

    println!("Data configuration:");
    println!("  Max sequence length: {}", data_config.max_seq_length);
    println!("  Micro batch size: {}", data_config.batch_size);
    println!(
        "  Gradient accumulation: {}",
        preset.data.gradient_accumulation_steps
    );
    println!(
        "  Effective batch size: {}",
        preset.data.effective_batch_size()
    );
    println!("  Tokens per step: {}", preset.data.tokens_per_step());
    println!("  Num workers: {}", data_config.num_workers);
    println!();

    let data_loader =
        create_data_loader(&primary_dataset.path, data_config.clone(), device.clone())?;
    println!("✓ Data loader initialized successfully");
    println!();

    // ========================================================================
    // 6. LEARNING RATE SCHEDULER SETUP (WSD)
    // ========================================================================

    println!("{}", "=".repeat(80));
    println!("LEARNING RATE SCHEDULER (WSD)");
    println!("{}", "=".repeat(80));
    println!();

    let total_steps = preset.optimization.total_steps;
    let warmup_steps = (total_steps as f32 * preset.optimization.warmup_fraction) as usize;
    let stable_steps = total_steps as usize - warmup_steps;
    let decay_steps = (total_steps as f32 * preset.optimization.decay_fraction) as usize;

    let scheduler = WSDSchedulerBuilder::new()
        .with_peak_lr(preset.optimization.learning_rate)
        .with_min_lr(preset.optimization.min_learning_rate)
        .with_warmup_steps(warmup_steps)
        .with_stable_steps(stable_steps)
        .with_decay_steps(decay_steps)
        .build();

    println!("WSD Scheduler configuration:");
    println!(
        "  Peak learning rate: {:.2e}",
        preset.optimization.learning_rate
    );
    println!(
        "  Min learning rate: {:.2e}",
        preset.optimization.min_learning_rate
    );
    println!("  Total steps: {}", total_steps);
    println!(
        "  Warmup steps: {} ({:.1}%)",
        warmup_steps,
        preset.optimization.warmup_fraction * 100.0
    );
    println!("  Stable steps: {}", stable_steps);
    println!(
        "  Decay steps: {} ({:.1}%)",
        decay_steps,
        preset.optimization.decay_fraction * 100.0
    );
    println!();

    // ========================================================================
    // 7. TRAINING SUMMARY
    // ========================================================================

    println!("{}", "=".repeat(80));
    println!("TRAINING SUMMARY");
    println!("{}", "=".repeat(80));
    println!();

    println!("Configuration:");
    println!("  Preset: {}", preset.name);
    println!("  Model size: {}", preset.model.size);
    println!("  Total training steps: {}", total_steps);
    println!("  Tokens per step: {}", preset.data.tokens_per_step());
    println!("  Total tokens: {:.2}B", preset.total_tokens() as f64 / 1e9);
    println!();

    println!("Optimization:");
    println!("  Optimizer: AdamW");
    println!("  Beta1: {}", preset.optimization.beta1);
    println!("  Beta2: {}", preset.optimization.beta2);
    println!("  Epsilon: {:.2e}", preset.optimization.epsilon);
    println!("  Weight decay: {}", preset.optimization.weight_decay);
    println!("  Max grad norm: {}", preset.optimization.max_grad_norm);
    println!();

    println!("Hybrid Training:");
    println!("  Warmup steps: {}", preset.hybrid.warmup_steps);
    println!(
        "  Full steps per cycle: {}",
        preset.hybrid.full_steps_per_cycle
    );
    println!("  Max predict steps: {}", preset.hybrid.max_predict_steps);
    println!(
        "  Confidence threshold: {:.2}",
        preset.hybrid.confidence_threshold
    );
    println!(
        "  Divergence threshold: {:.2}",
        preset.hybrid.divergence_threshold
    );
    println!();

    println!("Token Tracking:");
    println!("  Batch size: {}", preset.data.effective_batch_size());
    println!("  Sequence length: {}", preset.data.seq_length);
    println!(
        "  Tokens/step: {} * {} = {}",
        preset.data.effective_batch_size(),
        preset.data.seq_length,
        preset.data.tokens_per_step()
    );
    println!();

    println!("Datasets (training order):");
    for ds in &verified_datasets {
        println!(
            "  {}. {} - {:.2} GB at {:?}",
            ds.priority, ds.name, ds.estimated_size_gb, ds.path
        );
    }
    println!();

    // ========================================================================
    // 8. READY TO START
    // ========================================================================

    println!("{}", "=".repeat(80));
    println!("READY TO START TRAINING");
    println!("{}", "=".repeat(80));
    println!();
    println!("This is a configuration demonstration script.");
    println!("To start training, integrate with hybrid-predict-trainer-rs:");
    println!();
    println!("  use hybrid_predict_trainer_rs::{{create_trainer, TrainerConfig}};");
    println!("  use tritter_model_rs::create_trainer_with_config;");
    println!();
    println!("  let trainer = create_trainer_with_config(");
    println!("      &model_config,");
    println!("      trainer_config,");
    println!("      learning_rate,");
    println!("      &device,");
    println!("  )?;");
    println!();
    println!("  for (step, batch) in data_loader.enumerate() {{");
    println!("      let batch = batch?;");
    println!("      let result = trainer.step(&batch)?;");
    println!("      let lr = scheduler.get_lr(step);");
    println!("      // Update optimizer LR, log metrics, etc.");
    println!("  }}");
    println!();

    println!("{}", "=".repeat(80));
    println!("Configuration verified. All systems ready.");
    println!("{}", "=".repeat(80));

    Ok(())
}
