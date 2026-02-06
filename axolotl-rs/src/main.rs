//! CLI entry point for axolotl-rs.

use clap::{Parser, Subcommand};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

mod cli;
mod config;
mod dataset;
mod error;
#[cfg(feature = "peft")]
mod llama_common;
#[cfg(feature = "peft")]
mod lora_llama;
mod model;
mod optimizer;
#[cfg(all(feature = "peft", feature = "qlora"))]
mod qlora_llama;
mod scheduler;
mod trainer;

use config::AxolotlConfig;
use error::Result;

#[derive(Parser)]
#[command(name = "axolotl")]
#[command(about = "YAML-driven fine-tuning toolkit for LLMs")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Validate a configuration file
    Validate {
        /// Path to configuration file
        config: String,
    },
    /// Start training
    Train {
        /// Path to configuration file
        config: String,
        /// Resume from checkpoint
        #[arg(long)]
        resume: Option<String>,
    },
    /// Merge adapter weights into base model
    Merge {
        /// Path to configuration file
        #[arg(long)]
        config: String,
        /// Path to adapter checkpoint
        #[arg(long)]
        adapter: Option<String>,
        /// Output directory for merged model
        #[arg(long)]
        output: String,
    },
    /// Generate a sample configuration file
    Init {
        /// Output path for config file
        #[arg(default_value = "config.yaml")]
        output: String,
        /// Model preset (llama2-7b, mistral-7b, phi3-mini)
        #[arg(long, default_value = "llama2-7b")]
        preset: String,
    },
}

fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer())
        .with(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Validate { config } => {
            tracing::info!("Validating configuration: {}", config);
            let config = AxolotlConfig::from_file(&config)?;
            config.validate()?;
            println!("✓ Configuration is valid");
            println!("  Model: {}", config.base_model);
            println!("  Adapter: {:?}", config.adapter);
            println!("  Dataset: {}", config.dataset.path);
        }
        Commands::Train { config, resume } => {
            tracing::info!("Starting training with config: {}", config);
            let config = AxolotlConfig::from_file(&config)?;
            config.validate()?;

            let mut trainer = trainer::Trainer::new(config)?;
            if let Some(checkpoint) = resume {
                trainer.resume_from(&checkpoint)?;
            }
            trainer.train()?;
        }
        Commands::Merge {
            config,
            adapter,
            output,
        } => {
            tracing::info!("Merging adapter to: {}", output);
            let config = AxolotlConfig::from_file(&config)?;
            let adapter_path =
                adapter.unwrap_or_else(|| format!("{}/checkpoint-final", config.output_dir));

            model::merge_adapter(&config, &adapter_path, &output)?;
            println!("✓ Merged model saved to: {output}");
        }
        Commands::Init { output, preset } => {
            tracing::info!("Generating config for preset: {}", preset);
            let config = AxolotlConfig::from_preset(&preset)?;
            config.to_file(&output)?;
            println!("✓ Configuration written to: {output}");
        }
    }

    Ok(())
}
