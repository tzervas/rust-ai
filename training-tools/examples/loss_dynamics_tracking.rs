//! Example: Loss Dynamics Tracking
//!
//! Demonstrates how to use the LossDynamicsTracker to compute and monitor
//! loss velocity and acceleration during training.
//!
//! Loss velocity = rate of change of loss (first derivative)
//! - Negative: loss is improving
//! - Positive: loss is worsening
//!
//! Loss acceleration = rate of change of velocity (second derivative)
//! - Negative: improvement is accelerating
//! - Positive: improvement is slowing (convergence plateau)

use training_tools::LossDynamicsTracker;

fn main() {
    println!("Loss Dynamics Tracking Example");
    println!("==============================\n");

    // Scenario 1: Ideal training - steady improvement
    println!("Scenario 1: Ideal Training (Steady Improvement)");
    println!("{}", "-".repeat(50));
    scenario_steady_improvement();
    println!();

    // Scenario 2: Converging training - improvement slowing
    println!("Scenario 2: Converging Training (Slowing Improvement)");
    println!("{}", "-".repeat(50));
    scenario_converging();
    println!();

    // Scenario 3: Plateau - stuck at high loss
    println!("Scenario 3: Plateau (Stuck)");
    println!("{}", "-".repeat(50));
    scenario_plateau();
    println!();

    // Scenario 4: Diverging - loss getting worse
    println!("Scenario 4: Diverging (Loss Worsening)");
    println!("{}", "-".repeat(50));
    scenario_diverging();
}

fn scenario_steady_improvement() {
    let mut tracker = LossDynamicsTracker::new(20);

    // Simulate steady loss improvement
    let losses = vec![3.5, 3.2, 3.0, 2.8, 2.6, 2.5, 2.4, 2.3, 2.25, 2.2];

    for (step, loss) in losses.iter().enumerate() {
        let (velocity, acceleration) = tracker.update(*loss);
        println!(
            "Step {:2}: Loss={:.2} | Velocity={:7.4} | Acceleration={:7.4}",
            step, loss, velocity, acceleration
        );
    }

    println!("\nInterpretation:");
    println!("  - Velocity is consistently negative (loss improving)");
    println!("  - Acceleration may vary but overall trend should be positive");
}

fn scenario_converging() {
    let mut tracker = LossDynamicsTracker::new(20);

    // Simulate converging loss - improvement slowing down
    let losses = vec![
        3.5, 3.2, 2.9, 2.6, 2.4, 2.25, 2.15, 2.1, 2.08, 2.07, 2.065, 2.063, 2.062, 2.061, 2.0608,
    ];

    for (step, loss) in losses.iter().enumerate() {
        let (velocity, acceleration) = tracker.update(*loss);
        println!(
            "Step {:2}: Loss={:.4} | Velocity={:8.5} | Acceleration={:8.5}",
            step, loss, velocity, acceleration
        );
    }

    println!("\nInterpretation:");
    println!("  - Velocity is negative but magnitude decreases over time");
    println!("  - Positive acceleration indicates improvement is slowing");
    println!("  - This is a normal convergence pattern");
}

fn scenario_plateau() {
    let mut tracker = LossDynamicsTracker::new(20);

    // Simulate plateau - loss stuck at same value
    let losses = vec![
        2.5, 2.45, 2.42, 2.4, 2.4, 2.4, 2.4, 2.41, 2.4, 2.4, 2.39, 2.4, 2.4, 2.4, 2.4,
    ];

    for (step, loss) in losses.iter().enumerate() {
        let (velocity, acceleration) = tracker.update(*loss);
        println!(
            "Step {:2}: Loss={:.2} | Velocity={:7.4} | Acceleration={:7.4}",
            step, loss, velocity, acceleration
        );
    }

    println!("\nInterpretation:");
    println!("  - Velocity oscillates around zero");
    println!("  - No consistent improvement");
    println!("  - This may indicate learning rate is too small");
}

fn scenario_diverging() {
    let mut tracker = LossDynamicsTracker::new(20);

    // Simulate diverging loss - getting worse
    let losses = vec![2.0, 2.1, 2.25, 2.4, 2.6, 2.8, 3.0, 3.2, 3.5, 3.9];

    for (step, loss) in losses.iter().enumerate() {
        let (velocity, acceleration) = tracker.update(*loss);
        println!(
            "Step {:2}: Loss={:.2} | Velocity={:7.4} | Acceleration={:7.4}",
            step, loss, velocity, acceleration
        );
    }

    println!("\nInterpretation:");
    println!("  - Velocity is positive (loss increasing)");
    println!("  - This indicates training is diverging");
    println!("  - Consider reducing learning rate or checking for NaN/Inf");
}
