//! Demonstrates the adaptive learning rate controller.
//!
//! Run with: cargo run --example adaptive_lr_demo

use training_tools::lr_controller::AdaptiveLRController;

fn main() {
    println!("Adaptive Learning Rate Controller Demo\n");
    println!("=========================================\n");

    // Create controller with base_lr=1e-4, min_lr=1e-6, max_lr=5e-4
    let mut controller = AdaptiveLRController::new(1e-4, 1e-6, 5e-4);

    println!("Scenario 1: Stable Training (decreasing loss)");
    println!("----------------------------------------------");
    let mut loss = 5.0;
    for step in 0..30 {
        loss *= 0.98; // Gradual decrease
        let gradient_norm = 0.5;
        let lr = controller.update(loss, gradient_norm);
        if step % 5 == 0 {
            println!("Step {}: loss={:.4}, LR={:.6}", step, loss, lr);
        }
    }
    println!("Final LR: {:.6}\n", controller.current_lr());

    // Reset for next scenario
    controller.reset();

    println!("Scenario 2: Oscillating Loss");
    println!("-----------------------------");
    for step in 0..40 {
        let loss = if step % 2 == 0 { 3.0 } else { 1.5 };
        let gradient_norm = 0.5;
        let lr = controller.update(loss, gradient_norm);
        if step % 5 == 0 {
            println!("Step {}: loss={:.4}, LR={:.6}", step, loss, lr);
        }
    }
    println!(
        "Final LR (reduced due to oscillation): {:.6}\n",
        controller.current_lr()
    );

    // Reset for next scenario
    controller.reset();

    println!("Scenario 3: Plateau (constant loss)");
    println!("------------------------------------");
    for step in 0..60 {
        let loss = 2.0; // No improvement
        let gradient_norm = 0.5;
        let lr = controller.update(loss, gradient_norm);
        if step % 10 == 0 {
            println!("Step {}: loss={:.4}, LR={:.6}", step, loss, lr);
        }
    }
    println!(
        "Final LR (increased to escape plateau): {:.6}\n",
        controller.current_lr()
    );

    // Reset for next scenario
    controller.reset();

    println!("Scenario 4: Loss Spike");
    println!("----------------------");
    for step in 0..30 {
        let loss = if step == 15 {
            10.0 // Sudden spike at step 15
        } else {
            2.0
        };
        let gradient_norm = 0.5;
        let lr = controller.update(loss, gradient_norm);
        if step >= 13 && step <= 20 {
            println!("Step {}: loss={:.4}, LR={:.6}", step, loss, lr);
        }
    }
    println!(
        "Final LR (recovered after spike): {:.6}\n",
        controller.current_lr()
    );

    println!("\nDemo completed!");
    println!("\nKey Features:");
    println!("- Oscillation detection: Reduces LR when loss variance is high");
    println!("- Plateau detection: Increases LR when progress stalls");
    println!("- Spike mitigation: Emergency LR reduction on sudden loss spikes");
    println!("- Smooth transitions: Gradual drift back to base LR");
}
