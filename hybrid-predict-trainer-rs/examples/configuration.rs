//! Configuration examples.
//!
//! Demonstrates different configuration patterns for the hybrid trainer.

use hybrid_predict_trainer_rs::config::{HybridTrainerConfig, PredictorConfig};

fn main() {
    println!("=== Hybrid Trainer Configuration Examples ===\n");

    // Default configuration
    println!("1. Default configuration:");
    let default_config = HybridTrainerConfig::default();
    print_config(&default_config);

    // Conservative configuration (safer, less speedup)
    println!("\n2. Conservative configuration:");
    let conservative = HybridTrainerConfig::builder()
        .warmup_steps(500)
        .full_steps(50)
        .max_predict_steps(20)
        .confidence_threshold(0.95)
        .build();
    print_config(&conservative);

    // Aggressive configuration (more speedup, higher risk)
    println!("\n3. Aggressive configuration:");
    let aggressive = HybridTrainerConfig::builder()
        .warmup_steps(100)
        .full_steps(10)
        .max_predict_steps(100)
        .confidence_threshold(0.75)
        .build();
    print_config(&aggressive);

    // RSSM predictor configuration
    println!("\n4. RSSM predictor configuration:");
    let rssm_config = HybridTrainerConfig::builder()
        .predictor_config(PredictorConfig::RSSM {
            deterministic_dim: 512,
            stochastic_dim: 64,
            num_categoricals: 32,
            ensemble_size: 5,
        })
        .build();
    println!("  Predictor: {:?}", rssm_config.predictor_config);

    // Custom divergence thresholds
    println!("\n5. Custom divergence thresholds:");
    let custom_divergence = HybridTrainerConfig::builder()
        .divergence_threshold(2.0)
        .build();
    println!(
        "  Divergence threshold: {}",
        custom_divergence.divergence_threshold
    );
}

fn print_config(config: &HybridTrainerConfig) {
    println!("  warmup_steps: {}", config.warmup_steps);
    println!("  full_steps: {}", config.full_steps);
    println!("  max_predict_steps: {}", config.max_predict_steps);
    println!("  confidence_threshold: {}", config.confidence_threshold);
    println!("  divergence_threshold: {}", config.divergence_threshold);
}
