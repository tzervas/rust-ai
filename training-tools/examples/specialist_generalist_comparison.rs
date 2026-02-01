//! Example demonstrating specialist vs generalist training presets.
//!
//! Run with:
//! ```bash
//! cargo run -p training-tools --example specialist_generalist_comparison
//! ```

use training_tools::TrainingPreset;

fn main() {
    println!("{}", "=".repeat(80));
    println!("SPECIALIST VS GENERALIST TRAINING PRESETS");
    println!("{}", "=".repeat(80));

    let specialist = TrainingPreset::specialist_100m();
    let generalist = TrainingPreset::generalist_100m();

    // Print both configurations
    println!("{}", specialist.summary());
    println!("{}", generalist.summary());

    // Print comparison table
    println!("\n{}", "=".repeat(80));
    println!("KEY DIFFERENCES");
    println!("{}", "=".repeat(80));
    println!();
    println!("| Metric | Specialist | Generalist | Reasoning |");
    println!("|--------|-----------|------------|-----------|");

    println!(
        "| Learning Rate | {:.2e} | {:.2e} | Specialist: Conservative for stability |",
        specialist.optimization.learning_rate, generalist.optimization.learning_rate
    );

    println!(
        "| Warmup Phase | {:.1}% | {:.1}% | Specialist: Longer for subtle patterns |",
        specialist.optimization.warmup_fraction * 100.0,
        generalist.optimization.warmup_fraction * 100.0
    );

    println!(
        "| Decay Phase | {:.1}% | {:.1}% | Specialist: Extended refinement |",
        specialist.optimization.decay_fraction * 100.0,
        generalist.optimization.decay_fraction * 100.0
    );

    println!(
        "| Weight Decay | {:.2} | {:.2} | Specialist: Stronger regularization |",
        specialist.optimization.weight_decay, generalist.optimization.weight_decay
    );

    println!(
        "| Dropout | {:.2} | {:.2} | Specialist: Prevent overfitting |",
        specialist.model.dropout, generalist.model.dropout
    );

    println!(
        "| Full Steps/Cycle | {} | {} | Specialist: More precise gradients |",
        specialist.hybrid.full_steps_per_cycle, generalist.hybrid.full_steps_per_cycle
    );

    println!(
        "| Max Predict Steps | {} | {} | Specialist: Quality over speed |",
        specialist.hybrid.max_predict_steps, generalist.hybrid.max_predict_steps
    );

    println!(
        "| Confidence Threshold | {:.2} | {:.2} | Specialist: More conservative |",
        specialist.hybrid.confidence_threshold, generalist.hybrid.confidence_threshold
    );

    println!(
        "| Batch Size | {} | {} | Generalist: Larger for stability |",
        specialist.data.effective_batch_size(),
        generalist.data.effective_batch_size()
    );

    println!(
        "| Total Steps | {} | {} | Generalist: More data coverage |",
        specialist.optimization.total_steps, generalist.optimization.total_steps
    );

    // Print use case recommendations
    println!("\n{}", "=".repeat(80));
    println!("RECOMMENDED USE CASES");
    println!("{}", "=".repeat(80));
    println!();

    println!("SPECIALIST PRESET:");
    println!("  - Medical/legal/scientific domain models");
    println!("  - Code generation for specific frameworks (e.g., React, PyTorch)");
    println!("  - Mathematical theorem proving");
    println!("  - Technical documentation generation");
    println!("  - SQL query generation");
    println!("  - Chemistry/biology sequence modeling");
    println!();

    println!("GENERALIST PRESET:");
    println!("  - General-purpose chatbots");
    println!("  - Multi-task foundation models");
    println!("  - Transfer learning base models");
    println!("  - Broad knowledge assistants");
    println!("  - Multi-domain question answering");
    println!("  - General text completion");
    println!();

    // Print training philosophy summary
    println!("{}", "=".repeat(80));
    println!("TRAINING PHILOSOPHY");
    println!("{}", "=".repeat(80));
    println!();

    println!("SPECIALIST:");
    println!("  Focus: Deep expertise in narrow domain");
    println!("  Priority: Quality and accuracy over speed");
    println!("  Strategy: Conservative learning with strong regularization");
    println!("  Risk: Overfitting to limited domain data");
    println!("  Mitigation: Higher dropout, weight decay, longer warmup/decay");
    println!();

    println!("GENERALIST:");
    println!("  Focus: Broad knowledge across diverse tasks");
    println!("  Priority: Coverage and adaptability");
    println!("  Strategy: Aggressive learning with minimal constraints");
    println!("  Risk: Underfitting on any single domain");
    println!("  Mitigation: Large batch size, more training steps");
    println!();

    println!("{}", "=".repeat(80));
}
