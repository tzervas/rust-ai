//! HuggingFace upload utility.
//!
//! Usage:
//!   hf-upload --model <path> --size <100m|500m|1b> --user <username>
//!
//! Examples:
//!   hf-upload --model ./runs/tritter_100m/checkpoints/final_model.safetensors \
//!             --size 100m --user myusername

use std::path::PathBuf;

use clap::Parser;
use tracing_subscriber::EnvFilter;

use training_tools::hf::{
    ArchitectureDetails, HuggingFaceUploader, ModelCardData, TrainingDetails,
};
use tritter_model_rs::TritterConfig;

#[derive(Parser)]
#[command(name = "hf-upload")]
#[command(about = "Upload trained Tritter models to HuggingFace Hub")]
struct Args {
    /// Path to model checkpoint (.safetensors)
    #[arg(short, long)]
    model: PathBuf,

    /// Model size (100m, 500m, 1b)
    #[arg(short, long)]
    size: String,

    /// HuggingFace username
    #[arg(short, long)]
    user: String,

    /// Repository name (default: tritter-<size>)
    #[arg(short, long)]
    repo: Option<String>,

    /// Make repository private
    #[arg(long)]
    private: bool,

    /// Cache directory for staging
    #[arg(long, default_value = "./.hf_cache")]
    cache_dir: PathBuf,

    /// Final loss for model card
    #[arg(long)]
    final_loss: Option<f32>,

    /// Best loss for model card
    #[arg(long)]
    best_loss: Option<f32>,

    /// Training steps completed
    #[arg(long)]
    steps: Option<u64>,

    /// Backward reduction percentage (from hybrid training)
    #[arg(long)]
    backward_reduction: Option<f64>,
}

fn get_config(size: &str) -> anyhow::Result<TritterConfig> {
    match size.to_lowercase().as_str() {
        "100m" | "small" => Ok(TritterConfig::small_100m()),
        "500m" | "medium" => Ok(TritterConfig::medium_500m()),
        "1b" | "large" => Ok(TritterConfig::large_1b()),
        _ => anyhow::bail!("Unknown model size: {}. Use 100m, 500m, or 1b", size),
    }
}

fn main() -> anyhow::Result<()> {
    // Parse arguments
    let args = Args::parse();

    // Setup logging
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("info".parse()?))
        .init();

    // Validate model file exists
    if !args.model.exists() {
        anyhow::bail!("Model file not found: {}", args.model.display());
    }

    // Get model config
    let config = get_config(&args.size)?;

    tracing::info!("=== HuggingFace Model Upload ===");
    tracing::info!("Model: {}", args.model.display());
    tracing::info!(
        "Size: {} ({:.1}M params)",
        args.size,
        config.parameter_count() as f64 / 1e6
    );
    tracing::info!("User: {}", args.user);

    // Create uploader
    std::fs::create_dir_all(&args.cache_dir)?;
    let uploader = HuggingFaceUploader::new(&args.user, args.cache_dir.clone());

    // Check authentication
    tracing::info!("Checking HuggingFace authentication...");
    if !uploader.check_auth()? {
        anyhow::bail!("HuggingFace not authenticated. Run: huggingface-cli login");
    }
    tracing::info!("Authentication OK");

    // Create model card
    let size_str = match args.size.to_lowercase().as_str() {
        "100m" | "small" => "100M",
        "500m" | "medium" => "500M",
        "1b" | "large" => "1B",
        _ => &args.size,
    };

    let model_card = ModelCardData {
        model_name: format!("Tritter-{}", size_str),
        model_size: size_str.to_string(),
        num_parameters: config.parameter_count(),
        architecture: ArchitectureDetails {
            hidden_size: config.hidden_size,
            num_layers: config.num_layers,
            num_heads: config.num_heads,
            num_kv_heads: config.num_kv_heads,
            intermediate_size: config.intermediate_size,
            vocab_size: config.vocab_size,
            max_seq_length: config.max_seq_length,
            use_bitnet: config.use_bitnet,
            use_qk_norm: config.use_qk_norm,
            gradient_checkpointing: config.gradient_checkpointing,
        },
        training: TrainingDetails {
            framework: "rust-ai / tritter-model-rs".to_string(),
            learning_rate: 3e-4,
            batch_size: 4,
            total_steps: args.steps.unwrap_or(10000),
            warmup_steps: 100,
            hybrid_training: true,
            backward_reduction: args.backward_reduction.unwrap_or(50.0),
            training_time_hours: 0.0,
        },
        metrics: args
            .final_loss
            .map(|final_loss| training_tools::hf::PerformanceMetrics {
                final_loss,
                best_loss: args.best_loss.unwrap_or(final_loss),
                tokens_per_second: 0.0,
                memory_usage_gb: 0.0,
            }),
        license: "mit".to_string(),
        repo_url: "https://github.com/tzervas/rust-ai".to_string(),
    };

    // Create repository
    let repo_name = args
        .repo
        .unwrap_or_else(|| format!("tritter-{}", size_str.to_lowercase()));
    let repo_id = format!("{}/{}", args.user, repo_name);

    tracing::info!("Creating repository: {}", repo_id);

    let hf_config =
        training_tools::hf::HfRepoConfig::model(&args.user, &repo_name).with_private(args.private);

    uploader.create_repo(&hf_config)?;
    tracing::info!("Repository created/verified");

    // Upload model
    tracing::info!("Uploading model...");
    uploader.upload_file(
        &repo_id,
        &args.model,
        "model.safetensors",
        "Upload model checkpoint",
    )?;

    // Upload model card
    let readme_content = model_card.generate_markdown();
    let readme_path = args.cache_dir.join("README.md");
    std::fs::write(&readme_path, &readme_content)?;

    uploader.upload_file(&repo_id, &readme_path, "README.md", "Upload model card")?;

    // Upload config
    let config_json = serde_json::json!({
        "model_type": "tritter",
        "hidden_size": config.hidden_size,
        "num_layers": config.num_layers,
        "num_heads": config.num_heads,
        "num_kv_heads": config.num_kv_heads,
        "intermediate_size": config.intermediate_size,
        "vocab_size": config.vocab_size,
        "max_seq_length": config.max_seq_length,
        "use_bitnet": config.use_bitnet,
        "use_qk_norm": config.use_qk_norm,
    });
    let config_path = args.cache_dir.join("config.json");
    std::fs::write(&config_path, serde_json::to_string_pretty(&config_json)?)?;

    uploader.upload_file(&repo_id, &config_path, "config.json", "Upload config")?;

    tracing::info!("");
    tracing::info!("Upload complete!");
    tracing::info!("Repository: https://huggingface.co/{}", repo_id);

    // Cleanup cache
    std::fs::remove_file(&readme_path).ok();
    std::fs::remove_file(&config_path).ok();

    Ok(())
}
