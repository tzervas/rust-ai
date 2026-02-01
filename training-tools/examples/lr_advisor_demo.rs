//! Learning Rate Advisor Demo
//!
//! Demonstrates how to use the LR advisor to monitor training and suggest
//! learning rate adjustments based on loss and gradient dynamics.
//!
//! Run with:
//! ```bash
//! cargo run --example lr_advisor_demo
//! ```

use training_tools::lr_advisor::{analyze_lr, LRAdvice, TrainingPhase, Urgency};

fn main() {
    println!("=== Learning Rate Advisor Demo ===\n");

    // Scenario 1: Healthy training - smooth convergence
    println!("Scenario 1: Healthy Training");
    println!("-----------------------------");
    let healthy_losses: Vec<f32> = (0..50)
        .map(|i| 2.5 * (1.0 - 0.2 * i as f32 / 50.0))
        .collect();
    let healthy_grads = vec![0.3; 50];

    match analyze_lr(
        &healthy_losses,
        &healthy_grads,
        1e-4,
        500,
        TrainingPhase::Stable,
    ) {
        Some(advice) => println!("Advice: {}", advice.format()),
        None => println!("✓ Training looks healthy, no LR adjustment needed"),
    }
    println!();

    // Scenario 2: Loss oscillation - LR too high
    println!("Scenario 2: Loss Oscillation (LR too high)");
    println!("-------------------------------------------");
    let oscillating_losses = vec![
        2.0, 1.5, 2.2, 1.4, 2.3, 1.3, 2.4, 1.2, 2.5, 1.1, 2.6, 1.0, 2.5, 1.1, 2.4, 1.2,
    ];
    let oscillating_grads = vec![0.5; 16];

    if let Some(advice) = analyze_lr(
        &oscillating_losses,
        &oscillating_grads,
        1e-4,
        500,
        TrainingPhase::Stable,
    ) {
        print_advice(&advice);
    }
    println!();

    // Scenario 3: Loss plateau - LR too conservative
    println!("Scenario 3: Loss Plateau (LR too low)");
    println!("--------------------------------------");
    let mut plateau_losses = vec![2.0; 60];
    for (i, loss) in plateau_losses.iter_mut().enumerate() {
        *loss += (i as f32 * 0.0001).sin() * 0.001; // Tiny noise
    }
    let plateau_grads = vec![0.01; 60];

    if let Some(advice) = analyze_lr(
        &plateau_losses,
        &plateau_grads,
        1e-4,
        1000,
        TrainingPhase::Stable,
    ) {
        print_advice(&advice);
    }
    println!();

    // Scenario 4: Gradient explosion - critical issue
    println!("Scenario 4: Gradient Explosion (CRITICAL)");
    println!("------------------------------------------");
    let explosion_losses = vec![2.0, 2.1, 2.5, 3.5, 5.0, 8.0, 15.0, 30.0, 60.0, 120.0];
    let explosion_grads = vec![0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.5, 1.5, 1.5, 1.5];

    if let Some(advice) = analyze_lr(
        &explosion_losses,
        &explosion_grads,
        1e-4,
        500,
        TrainingPhase::Stable,
    ) {
        print_advice(&advice);
    }
    println!();

    // Scenario 5: Warmup phase tolerance
    println!("Scenario 5: Warmup Phase (Higher Tolerance)");
    println!("--------------------------------------------");
    let warmup_losses = vec![3.0, 2.5, 3.2, 2.4, 3.3, 2.3, 3.4, 2.2, 3.5, 2.1];
    let warmup_grads = vec![0.6; 10];

    match analyze_lr(
        &warmup_losses,
        &warmup_grads,
        1e-4,
        50,
        TrainingPhase::Warmup,
    ) {
        Some(advice) => {
            println!("Advice during warmup: {}", advice.format());
            println!(
                "  Note: Urgency is {} (lower than it would be in stable phase)",
                advice.urgency
            );
        }
        None => println!("✓ Warmup tolerating oscillation within normal range"),
    }
    println!();

    // Scenario 6: Simulated training loop
    println!("Scenario 6: Simulated Training Loop");
    println!("------------------------------------");
    simulate_training_loop();
}

fn print_advice(advice: &LRAdvice) {
    println!("{}", advice.format());
    println!("  Current LR:  {:.2e}", advice.current_lr);
    println!("  Suggested LR: {:.2e}", advice.suggested_lr);
    println!("  Change:      {:+.1}%", advice.percentage_change());
    println!("  Issue:       {}", advice.issue);

    match advice.urgency {
        Urgency::Critical => println!("  ⚠️  CRITICAL - Apply immediately!"),
        Urgency::High => println!("  ⚠️  HIGH - Apply soon"),
        Urgency::Medium => println!("  ⚡ MEDIUM - Consider applying"),
        Urgency::Low => println!("  ℹ️  LOW - Optional optimization"),
    }
}

fn simulate_training_loop() {
    let mut current_lr = 1e-4_f32;
    let mut loss_history = Vec::new();
    let mut grad_history = Vec::new();

    // Simulate training with initial oscillation, then stabilization
    for step in 0..100 {
        // Simulate loss and gradient
        let (loss, grad_norm) = if step < 20 {
            // Initial phase: oscillating due to high LR
            let osc: f32 = if step % 2 == 0 { 0.3 } else { -0.2 };
            (2.0 + osc, 0.5 + osc.abs())
        } else if step < 60 {
            // After LR reduction: smooth convergence
            let progress = (step - 20) as f32 / 40.0;
            (1.7 - progress * 0.5, 0.3 - progress * 0.1)
        } else {
            // Late stage: plateau
            (1.19 + (step as f32 * 0.001).sin() * 0.001, 0.05)
        };

        loss_history.push(loss);
        grad_history.push(grad_norm);

        // Keep window of last 50 steps
        if loss_history.len() > 50 {
            loss_history.remove(0);
            grad_history.remove(0);
        }

        // Check for advice every 10 steps (after initial 10 steps)
        if step > 0 && step % 10 == 0 && loss_history.len() >= 10 {
            let phase = if step < 100 {
                TrainingPhase::Warmup
            } else {
                TrainingPhase::Stable
            };

            if let Some(advice) = analyze_lr(&loss_history, &grad_history, current_lr, step, phase)
            {
                println!("\nStep {}: {}", step, advice.format());
                println!(
                    "  Applying suggestion: {:.2e} → {:.2e}",
                    advice.current_lr, advice.suggested_lr
                );
                current_lr = advice.suggested_lr;
            }
        }
    }

    println!("\nFinal LR: {:.2e}", current_lr);
    println!(
        "Training completed with {} LR adjustments",
        if current_lr != 1e-4 {
            "automatic"
        } else {
            "no"
        }
    );
}
