//! Demonstration of early stopping with patience and delta thresholds.
//!
//! Run with: cargo run --example early_stopping_demo

use training_tools::early_stopping::{EarlyStopping, StoppingDecision, StoppingMode};

fn main() {
    println!("=== Early Stopping Demo ===\n");

    // Example 1: MinLoss mode (stop when loss stops decreasing)
    println!("Example 1: Training with MinLoss mode (patience=5, min_delta=0.01)");
    let mut early_stop = EarlyStopping::new(5, 0.01).with_mode(StoppingMode::MinLoss);

    let loss_sequence = vec![
        1.0, 0.95, 0.92, 0.90, // improving
        0.895, 0.894, 0.893, // small improvements < min_delta
        0.892, 0.891, // still no significant improvement
        0.890, // patience exhausted
    ];

    for (step, loss) in loss_sequence.iter().enumerate() {
        match early_stop.check(*loss, step as u64) {
            StoppingDecision::NewBest => {
                println!(
                    "  Step {}: Loss={:.4} - NEW BEST! (Save checkpoint)",
                    step, loss
                );
            }
            StoppingDecision::NoImprovement { count, remaining } => {
                println!(
                    "  Step {}: Loss={:.4} - No improvement ({}/{}, {} remaining)",
                    step,
                    loss,
                    count,
                    early_stop.counter() + remaining,
                    remaining
                );
            }
            StoppingDecision::Stop => {
                println!(
                    "  Step {}: Loss={:.4} - EARLY STOP! Best was {:.4} at step {}",
                    step,
                    loss,
                    early_stop.best_value(),
                    early_stop.best_step()
                );
                break;
            }
            StoppingDecision::Continue => {
                println!("  Step {}: Loss={:.4} - Continue", step, loss);
            }
        }
    }

    println!("\n---\n");

    // Example 2: MaxMetric mode (stop when accuracy stops increasing)
    println!("Example 2: Training with MaxMetric mode (patience=3, min_delta=0.005)");
    let mut early_stop = EarlyStopping::new(3, 0.005).with_mode(StoppingMode::MaxMetric);

    let accuracy_sequence = vec![
        0.70, 0.75, 0.80, 0.85, // improving
        0.851, 0.852, // small improvements < min_delta
        0.853, // patience exhausted
    ];

    for (step, acc) in accuracy_sequence.iter().enumerate() {
        match early_stop.check(*acc, step as u64) {
            StoppingDecision::NewBest => {
                println!(
                    "  Step {}: Accuracy={:.4} - NEW BEST! (Save checkpoint)",
                    step, acc
                );
            }
            StoppingDecision::NoImprovement { count, remaining } => {
                println!(
                    "  Step {}: Accuracy={:.4} - No improvement ({} steps, {} remaining)",
                    step, acc, count, remaining
                );
            }
            StoppingDecision::Stop => {
                println!(
                    "  Step {}: Accuracy={:.4} - EARLY STOP! Best was {:.4} at step {}",
                    step,
                    acc,
                    early_stop.best_value(),
                    early_stop.best_step()
                );
                break;
            }
            StoppingDecision::Continue => {
                println!("  Step {}: Accuracy={:.4} - Continue", step, acc);
            }
        }
    }

    println!("\n---\n");

    // Example 3: Integration with checkpoint management
    println!("Example 3: Integration with checkpoint saving");
    let mut early_stop = EarlyStopping::new(4, 0.02);

    for step in 0..20 {
        let loss = simulate_training_step(step);

        match early_stop.check(loss, step as u64) {
            StoppingDecision::NewBest => {
                println!("  Step {}: Loss={:.4} - Saving checkpoint...", step, loss);
                // In real code: checkpoint_manager.save(model, step)?;
            }
            StoppingDecision::NoImprovement { remaining, .. } => {
                if remaining == 0 {
                    println!("  Step {}: Loss={:.4} - Final patience step!", step, loss);
                }
            }
            StoppingDecision::Stop => {
                println!(
                    "  Step {}: Loss={:.4} - Stopping and restoring best checkpoint from step {}",
                    step,
                    loss,
                    early_stop.best_step()
                );
                // In real code: checkpoint_manager.load(early_stop.best_step())?;
                break;
            }
            _ => {}
        }
    }

    println!("\n=== Demo Complete ===");
}

// Simulate a training loss curve that plateaus
fn simulate_training_step(step: u64) -> f32 {
    let base = 1.0;
    let decay = (-0.1 * step as f32).exp();
    let plateau = if step > 10 { 0.3 } else { 0.0 };
    base * decay + plateau + (step as f32 * 0.001).sin() * 0.01
}
