//! Demonstration of BatchTuner for automatic batch size adjustment.
//!
//! Run with: cargo run --example batch_tuner_demo

use training_tools::batch_tuner::BatchTuner;

fn main() {
    println!("=== Batch Tuner Demo ===\n");

    // Create a conservative tuner: start at 4, can go up to 32
    let mut tuner = BatchTuner::conservative(4, 1, 32);
    println!("Initial batch size: {}", tuner.current_batch_size());
    println!("Memory threshold: 75%\n");

    // Simulate training steps
    println!("Simulating successful training steps with low memory usage:");
    for step in 1..=15 {
        let memory_pct = 0.60; // 60% memory usage
        let suggested = tuner.suggest_batch_size(memory_pct);

        println!(
            "Step {}: batch={}, memory={:.1}%, suggested={}",
            step,
            tuner.current_batch_size(),
            memory_pct * 100.0,
            suggested
        );

        tuner.record_step(tuner.current_batch_size(), memory_pct, true);
    }

    let stats = tuner.stats();
    println!("\n{}", stats.format());

    // Simulate memory pressure
    println!("\n\nSimulating memory pressure:");
    for step in 16..=20 {
        let memory_pct = 0.85; // 85% memory usage (above 75% threshold)
        let suggested = tuner.suggest_batch_size(memory_pct);

        println!(
            "Step {}: batch={}, memory={:.1}%, suggested={}",
            step,
            tuner.current_batch_size(),
            memory_pct * 100.0,
            suggested
        );

        tuner.record_step(tuner.current_batch_size(), memory_pct, true);
    }

    let stats = tuner.stats();
    println!("\n{}", stats.format());

    // Simulate OOM scenario
    println!("\n\nSimulating OOM failure:");
    let current_batch = tuner.current_batch_size();
    let memory_pct = 0.98; // Critical memory
    println!(
        "Attempting step with batch={}, memory={:.1}%",
        current_batch,
        memory_pct * 100.0
    );

    tuner.record_step(current_batch, memory_pct, false); // FAILURE

    println!(
        "Emergency reduction: batch reduced to {}",
        tuner.current_batch_size()
    );

    let stats = tuner.stats();
    println!("\n{}", stats.format());

    // Demonstrate aggressive tuner
    println!("\n\n=== Aggressive Tuner ===");
    let mut aggressive = BatchTuner::aggressive(4, 1, 32);
    println!("Memory threshold: 85%");
    println!("Success steps required: 3\n");

    for step in 1..=8 {
        let memory_pct = 0.65;
        tuner.record_step(aggressive.current_batch_size(), memory_pct, true);

        println!(
            "Step {}: batch={}, memory={:.1}%",
            step,
            aggressive.current_batch_size(),
            memory_pct * 100.0
        );
    }

    let stats = aggressive.stats();
    println!("\n{}", stats.format());

    // Demonstrate memory estimation
    println!("\n\n=== Memory Estimation ===");
    let mut estimator = BatchTuner::new(8, 1, 64, 0.8);

    // Record some data points
    estimator.record_step(8, 0.50, true);
    estimator.record_step(16, 0.70, true);
    estimator.record_step(24, 0.85, true);

    // Estimate for different batch sizes
    for target_batch in [32, 40, 48] {
        if let Some(estimated_mem) = estimator.estimate_memory(target_batch) {
            println!(
                "Estimated memory for batch {}: {:.1}%",
                target_batch,
                estimated_mem * 100.0
            );
        }
    }

    println!("\n=== Demo Complete ===");
}
