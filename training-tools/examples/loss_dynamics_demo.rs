//! Demonstration of loss velocity and acceleration tracking.
//!
//! Run with: cargo run --example loss_dynamics_demo

use training_tools::calculate_loss_dynamics;

fn main() {
    println!("=== Loss Dynamics Demo ===\n");

    // Scenario 1: Healthy training (steadily improving)
    println!("Scenario 1: Healthy Training (Steadily Decreasing Loss)");
    let healthy_losses = vec![
        5.0, 4.8, 4.6, 4.4, 4.2, 4.0, 3.8, 3.6, 3.4, 3.2, 3.0, 2.8, 2.6, 2.4, 2.2, 2.0,
    ];
    let (vel, acc) = calculate_loss_dynamics(&healthy_losses, 20);
    println!("  Velocity:     {:.4} (negative = improving)", vel);
    println!(
        "  Acceleration: {:.4} (negative = slowing improvement)",
        acc
    );
    println!("  Interpretation: Loss is decreasing steadily.\n");

    // Scenario 2: Loss plateau (stuck)
    println!("Scenario 2: Loss Plateau (Stuck)");
    let plateau_losses = vec![
        3.0, 2.98, 3.01, 2.99, 3.02, 2.97, 3.0, 3.01, 2.99, 3.0, 2.98, 3.01,
    ];
    let (vel, acc) = calculate_loss_dynamics(&plateau_losses, 20);
    println!("  Velocity:     {:.4} (near zero = no progress)", vel);
    println!("  Acceleration: {:.4}", acc);
    println!("  Interpretation: Loss is stuck at plateau. Consider increasing LR.\n");

    // Scenario 3: Diverging (loss increasing)
    println!("Scenario 3: Diverging Training (Loss Increasing)");
    let diverging_losses = vec![2.0, 2.1, 2.3, 2.6, 3.0, 3.5, 4.1, 4.8, 5.6, 6.5, 7.5, 8.6];
    let (vel, acc) = calculate_loss_dynamics(&diverging_losses, 20);
    println!("  Velocity:     {:.4} (positive = worsening)", vel);
    println!("  Acceleration: {:.4} (positive = accelerating worse)", acc);
    println!("  Interpretation: Training is diverging. Reduce LR immediately!\n");

    // Scenario 4: Slowing improvement (approaching convergence)
    println!("Scenario 4: Slowing Improvement (Approaching Convergence)");
    let slowing_losses = vec![
        5.0, 4.5, 4.1, 3.8, 3.6, 3.45, 3.35, 3.27, 3.22, 3.18, 3.15, 3.13, 3.12, 3.11, 3.10,
    ];
    let (vel, acc) = calculate_loss_dynamics(&slowing_losses, 20);
    println!(
        "  Velocity:     {:.4} (negative but small = slow improvement)",
        vel
    );
    println!(
        "  Acceleration: {:.4} (positive = improvement slowing)",
        acc
    );
    println!("  Interpretation: Loss still improving but rate is slowing.\n");

    // Scenario 5: Oscillating (unstable)
    println!("Scenario 5: Oscillating Loss (Unstable Training)");
    let oscillating_losses = vec![3.0, 2.5, 3.2, 2.3, 3.5, 2.1, 3.8, 1.9, 4.0, 1.7];
    let (vel, acc) = calculate_loss_dynamics(&oscillating_losses, 20);
    println!("  Velocity:     {:.4}", vel);
    println!("  Acceleration: {:.4}", acc);
    println!("  Interpretation: High oscillation detected. Consider reducing LR.\n");

    // Real-world example: Using with TrainingRun
    println!("=== Real-World Usage ===\n");
    println!("In your training loop:");
    println!("```rust");
    println!("// Collect recent losses");
    println!("let recent_metrics = training_run.read_recent_metrics(30)?;");
    println!("let losses: Vec<f32> = recent_metrics.iter().map(|m| m.loss).collect();");
    println!();
    println!("// Or use the convenience method");
    println!("let (velocity, acceleration) = training_run.loss_dynamics();");
    println!();
    println!("// Make decisions based on dynamics");
    println!("if velocity > 0.01 {{");
    println!("    // Loss increasing - reduce LR");
    println!("    optimizer.set_lr(current_lr * 0.5);");
    println!("}} else if velocity.abs() < 0.001 {{");
    println!("    // Plateau - try increasing LR");
    println!("    optimizer.set_lr(current_lr * 1.2);");
    println!("}}");
    println!("```");
}
