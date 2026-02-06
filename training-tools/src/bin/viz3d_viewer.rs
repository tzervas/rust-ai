//! 3D Neural Network Viewer
//!
//! A standalone Bevy application for visualizing neural network architectures,
//! attention patterns, and training dynamics in 3D.
//!
//! # Usage
//!
//! ```bash
//! # Run with default transformer configuration
//! cargo run --bin viz3d-viewer --features viz3d
//!
//! # Specify a custom model configuration
//! cargo run --bin viz3d-viewer --features viz3d -- --model transformer --layers 6 --hidden 768 --heads 12
//!
//! # Connect to live training data
//! cargo run --bin viz3d-viewer --features viz3d -- --training-socket /tmp/training.sock
//! ```
//!
//! # Controls
//!
//! - **Right-click + drag**: Rotate camera
//! - **Left-click + drag**: Pan camera
//! - **Scroll**: Zoom in/out
//! - **1-9**: Select layer by number
//! - **J/K**: Navigate layers up/down
//! - **F**: Start forward pass animation
//! - **B**: Start backward pass animation
//! - **T**: Start training loop animation
//! - **Space**: Pause/resume animation
//! - **R**: Reset camera view
//! - **C**: Cycle camera mode (Orbit/Free/Pan/Follow)
//! - **H**: Toggle help panel
//! - **I**: Toggle layer info panel
//! - **S**: Toggle training stats panel
//! - **Escape**: Stop animation

use clap::Parser;
use training_tools::viz3d::{NetworkConfig, Viz3dApp};

/// 3D Neural Network Viewer
#[derive(Parser, Debug)]
#[command(name = "viz3d-viewer")]
#[command(about = "Visualize neural networks in 3D", long_about = None)]
struct Args {
    /// Model architecture type
    #[arg(long, default_value = "transformer")]
    model: String,

    /// Number of layers (for transformer/mlp)
    #[arg(long, default_value = "6")]
    layers: usize,

    /// Hidden dimension size
    #[arg(long, default_value = "768")]
    hidden: usize,

    /// Number of attention heads (for transformer)
    #[arg(long, default_value = "12")]
    heads: usize,

    /// Custom layer sizes (comma-separated, overrides --layers and --hidden)
    #[arg(long)]
    layer_sizes: Option<String>,

    /// Unix socket for live training data
    #[arg(long)]
    training_socket: Option<String>,

    /// Disable connection rendering
    #[arg(long)]
    no_connections: bool,

    /// Animation speed multiplier
    #[arg(long, default_value = "1.0")]
    speed: f32,
}

fn main() {
    let args = Args::parse();

    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("wgpu=warn".parse().unwrap())
                .add_directive("bevy_render=warn".parse().unwrap())
                .add_directive("bevy_ecs=warn".parse().unwrap())
                .add_directive("naga=warn".parse().unwrap()),
        )
        .init();

    // Build network configuration
    let config = if let Some(sizes_str) = args.layer_sizes {
        // Parse custom layer sizes
        let sizes: Vec<usize> = sizes_str
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect();

        if sizes.is_empty() {
            eprintln!("Error: Invalid layer sizes format. Use comma-separated integers.");
            std::process::exit(1);
        }

        let mut config = NetworkConfig::mlp(sizes);
        config.show_connections = !args.no_connections;
        config.animation_speed = args.speed;
        config
    } else {
        match args.model.to_lowercase().as_str() {
            "transformer" => {
                let mut config = NetworkConfig::transformer(args.layers, args.hidden, args.heads);
                config.show_connections = !args.no_connections;
                config.animation_speed = args.speed;
                config
            }
            "mlp" => {
                // Create MLP with expanding then contracting hidden layers
                let mut sizes = vec![args.hidden];
                for i in 0..args.layers {
                    let factor = if i < args.layers / 2 {
                        (i + 2) as f32
                    } else {
                        (args.layers - i + 1) as f32
                    };
                    sizes.push((args.hidden as f32 * factor) as usize);
                }
                sizes.push(args.hidden);

                let mut config = NetworkConfig::mlp(sizes);
                config.show_connections = !args.no_connections;
                config.animation_speed = args.speed;
                config
            }
            "gpt2" => {
                // GPT-2 small configuration
                let mut config = NetworkConfig::transformer(12, 768, 12);
                config.layer_names = create_gpt2_layer_names(12);
                config.show_connections = !args.no_connections;
                config.animation_speed = args.speed;
                config
            }
            "gpt2-medium" => {
                // GPT-2 medium configuration
                let mut config = NetworkConfig::transformer(24, 1024, 16);
                config.layer_names = create_gpt2_layer_names(24);
                config.show_connections = !args.no_connections;
                config.animation_speed = args.speed;
                config
            }
            "llama" => {
                // LLaMA-like configuration
                let mut config = NetworkConfig::transformer(32, 4096, 32);
                config.layer_names = create_llama_layer_names(32);
                config.show_connections = !args.no_connections;
                config.animation_speed = args.speed;
                config
            }
            "bert" => {
                // BERT base configuration
                let mut config = NetworkConfig::transformer(12, 768, 12);
                config.layer_names = create_bert_layer_names(12);
                config.show_connections = !args.no_connections;
                config.animation_speed = args.speed;
                config
            }
            _ => {
                eprintln!(
                    "Unknown model type: {}. Use 'transformer', 'mlp', 'gpt2', 'gpt2-medium', 'llama', or 'bert'.",
                    args.model
                );
                std::process::exit(1);
            }
        }
    };

    // Log configuration
    tracing::info!(
        "Starting 3D viewer with {} layers, {} total parameters (estimated)",
        config.layer_sizes.len(),
        estimate_parameters(&config)
    );

    // Print training socket info if provided
    if let Some(socket) = &args.training_socket {
        tracing::info!("Training data socket: {}", socket);
        // TODO: Connect to training socket for live updates
    }

    // Run the application
    Viz3dApp::run(config);
}

/// Estimate total parameters based on layer sizes.
fn estimate_parameters(config: &NetworkConfig) -> String {
    let total: usize = config
        .layer_sizes
        .windows(2)
        .map(|w| w[0] * w[1])
        .sum::<usize>()
        + config.layer_sizes.iter().sum::<usize>(); // biases

    if total >= 1_000_000_000 {
        format!("{:.1}B", total as f64 / 1_000_000_000.0)
    } else if total >= 1_000_000 {
        format!("{:.1}M", total as f64 / 1_000_000.0)
    } else if total >= 1_000 {
        format!("{:.1}K", total as f64 / 1_000.0)
    } else {
        total.to_string()
    }
}

/// Create layer names for GPT-2 style model.
fn create_gpt2_layer_names(num_layers: usize) -> Vec<String> {
    let mut names = vec![
        "Token Embedding".to_string(),
        "Position Embedding".to_string(),
    ];

    for i in 0..num_layers {
        names.push(format!("Block {} Attn", i));
        names.push(format!("Block {} FFN Up", i));
        names.push(format!("Block {} FFN Down", i));
    }

    names.push("LayerNorm".to_string());
    names.push("LM Head".to_string());
    names
}

/// Create layer names for LLaMA style model.
fn create_llama_layer_names(num_layers: usize) -> Vec<String> {
    let mut names = vec!["Embedding".to_string()];

    for i in 0..num_layers {
        names.push(format!("Layer {} RMSNorm".to_string(), i));
        names.push(format!("Layer {} Attn", i));
        names.push(format!("Layer {} FFN Gate", i));
        names.push(format!("Layer {} FFN Up", i));
        names.push(format!("Layer {} FFN Down", i));
    }

    names.push("Final RMSNorm".to_string());
    names.push("Output".to_string());
    names
}

/// Create layer names for BERT style model.
fn create_bert_layer_names(num_layers: usize) -> Vec<String> {
    let mut names = vec![
        "Token Embedding".to_string(),
        "Position Embedding".to_string(),
        "Type Embedding".to_string(),
        "Embedding LayerNorm".to_string(),
    ];

    for i in 0..num_layers {
        names.push(format!("Layer {} Attn", i));
        names.push(format!("Layer {} Attn Out", i));
        names.push(format!("Layer {} FFN", i));
        names.push(format!("Layer {} Out", i));
    }

    names.push("Pooler".to_string());
    names
}
