//! Gradient Control Demo
//!
//! Demonstrates the GradientController's automatic threshold adjustment,
//! explosion detection, and vanishing gradient warnings.
//!
//! Run with: cargo run --example gradient_control_demo

use training_tools::{GradientAction, GradientController};

fn main() {
    println!("=== Gradient Control Demo ===\n");

    // Scenario 1: Stable training
    println!("Scenario 1: Stable Training");
    println!("-".repeat(50));
    let mut controller = GradientController::new(1.0);

    for step in 0..20 {
        let grad_norm = 0.5 + (step as f32 * 0.02);
        let action = controller.update(grad_norm);
        println!(
            "Step {}: grad_norm={:.3}, action={:?}",
            step, grad_norm, action
        );
    }

    if let Some(stats) = controller.stats() {
        println!(
            "\nFinal stats: mean={:.3}, std={:.3}, min={:.3}, max={:.3}",
            stats.mean, stats.std_dev, stats.min, stats.max
        );
    }

    // Scenario 2: Gradient explosion
    println!("\n\nScenario 2: Gradient Explosion");
    println!("-".repeat(50));
    let mut controller = GradientController::new(1.0);

    // Start stable
    for _ in 0..15 {
        controller.update(0.5);
    }

    // Sudden explosion
    println!("Stable phase complete, injecting explosion...");
    let action = controller.update(10.0);
    println!("Explosion! grad_norm=10.0, action={:?}", action);
    println!("New threshold: {:.3}", controller.current_threshold());

    // Scenario 3: Vanishing gradients
    println!("\n\nScenario 3: Vanishing Gradients");
    println!("-".repeat(50));
    let mut controller = GradientController::new(1.0);

    // Start stable
    for _ in 0..15 {
        controller.update(0.5);
    }

    // Gradients vanish
    println!("Stable phase complete, injecting vanishing gradient...");
    let action = controller.update(0.0001);
    println!("Vanishing! grad_norm=0.0001, action={:?}", action);

    // Scenario 4: Consecutive high gradients trigger threshold reduction
    println!("\n\nScenario 4: Threshold Auto-Adjustment (High)");
    println!("-".repeat(50));
    let mut controller = GradientController::with_config(
        1.0,   // initial threshold
        0.1,   // min
        10.0,  // max
        40,    // window
        10.0,  // explosion factor
        0.001, // vanishing threshold
        0.2,   // 20% adjustment rate
    );

    // Start with low gradients
    for _ in 0..20 {
        controller.update(0.3);
    }

    let initial_threshold = controller.current_threshold();
    println!("Initial threshold: {:.3}", initial_threshold);

    // Now consistently high (but not explosion)
    for step in 0..15 {
        let action = controller.update(0.9);
        if matches!(action, GradientAction::ReduceThreshold) {
            println!(
                "Step {}: Threshold reduced to {:.3}",
                step,
                controller.current_threshold()
            );
        }
    }

    println!("Final threshold: {:.3}", controller.current_threshold());

    // Scenario 5: Consecutive low gradients trigger threshold increase
    println!("\n\nScenario 5: Threshold Auto-Adjustment (Low)");
    println!("-".repeat(50));
    let mut controller = GradientController::with_config(1.0, 0.1, 10.0, 40, 10.0, 0.001, 0.2);

    // Start mixed
    for i in 0..20 {
        controller.update(if i % 2 == 0 { 0.3 } else { 0.8 });
    }

    let initial_threshold = controller.current_threshold();
    println!("Initial threshold: {:.3}", initial_threshold);

    // Now consistently low
    for step in 0..25 {
        let action = controller.update(0.15);
        if matches!(action, GradientAction::IncreaseThreshold) {
            println!(
                "Step {}: Threshold increased to {:.3}",
                step,
                controller.current_threshold()
            );
        }
    }

    println!("Final threshold: {:.3}", controller.current_threshold());

    // Scenario 6: High variance warning
    println!("\n\nScenario 6: High Variance Detection");
    println!("-".repeat(50));
    let mut controller = GradientController::new(1.0);

    for step in 0..30 {
        let grad_norm = if step % 2 == 0 { 0.1 } else { 2.5 };
        let action = controller.update(grad_norm);

        match action {
            GradientAction::Warning(ref msg) if msg.contains("variance") => {
                println!("Step {}: {}", step, msg);
            }
            _ => {}
        }
    }

    println!("\n=== Demo Complete ===");
}
