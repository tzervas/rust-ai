///! Demonstration of training smoothing utilities.
///!
///! Shows how to use EMA, spike suppression, oscillation damping, and the full pipeline.
use training_tools::smoothing::{
    ExponentialMovingAverage, OscillationDamper, SmoothingConfig, SmoothingPipeline,
    SpikeSuppressor,
};

fn main() {
    println!("=== Training Smoothing Utilities Demo ===\n");

    // Demo 1: Exponential Moving Average
    println!("1. Exponential Moving Average (EMA)");
    println!("   Smooths noisy metrics with exponential weighting");
    let mut ema = ExponentialMovingAverage::new(0.1);
    let noisy_values = vec![1.0, 1.5, 0.8, 1.2, 1.1, 0.9, 1.3];
    print!("   Raw values:      ");
    for &v in &noisy_values {
        print!("{:.2} ", v);
        ema.update(v);
    }
    println!("\n   Smoothed value:  {:.2}\n", ema.value());

    // Demo 2: Spike Suppressor
    println!("2. Spike Suppressor");
    println!("   Detects and mitigates sudden metric spikes");
    let mut suppressor = SpikeSuppressor::new(2.0, 0.3);
    let values_with_spike = vec![1.0, 1.1, 1.05, 10.0, 1.2, 1.15, 1.1];
    println!("   Value  Suppressed  Suppressing?");
    for &v in &values_with_spike {
        let suppressed = suppressor.process(v);
        println!(
            "   {:<6.2} {:<11.2} {}",
            v,
            suppressed,
            if suppressor.is_suppressing() {
                "✓"
            } else {
                " "
            }
        );
    }
    println!();

    // Demo 3: Oscillation Damper
    println!("3. Oscillation Damper");
    println!("   Reduces amplitude of oscillating metrics");
    let mut damper = OscillationDamper::new(10, 0.15, 0.5);
    println!("   Step  Oscillating  Damped  Damping?");
    for i in 0..20 {
        let oscillating = 1.0 + 0.5 * ((i as f32 * 0.5).sin());
        let damped = damper.process(oscillating);
        if i % 3 == 0 {
            println!(
                "   {:<5} {:<12.2} {:<7.2} {}",
                i,
                oscillating,
                damped,
                if damper.is_damping() { "✓" } else { " " }
            );
        }
    }
    println!();

    // Demo 4: Full Smoothing Pipeline
    println!("4. Full Smoothing Pipeline");
    println!("   Combines spike suppression + oscillation damping + EMA");
    let config = SmoothingConfig {
        ema_alpha: 0.1,
        spike_threshold: 2.0,
        spike_recovery_rate: 0.3,
        oscillation_window: 10,
        oscillation_variance_threshold: 0.15,
        oscillation_damping_factor: 0.5,
    };
    let mut pipeline = SmoothingPipeline::new(config);

    // Simulate messy training loss: normal → spike → oscillation → normal
    let messy_loss = vec![
        2.5, 2.4, 2.3, 15.0, 2.2, 2.3, 2.1, 2.5, 1.9, 2.3, 2.0, 2.4, 1.9, 2.3, 2.0, 2.1,
    ];

    println!("   Step  Raw    Smoothed  Spike?  Oscillation?");
    for (i, &loss) in messy_loss.iter().enumerate() {
        let smoothed = pipeline.process(loss);
        let diag = pipeline.diagnostics();
        println!(
            "   {:<5} {:<6.2} {:<9.2} {:<7} {}",
            i,
            loss,
            smoothed,
            if diag.is_suppressing_spike {
                "✓"
            } else {
                " "
            },
            if diag.is_damping_oscillation {
                "✓"
            } else {
                " "
            }
        );
    }

    println!("\n=== Demo Complete ===");
    println!("\nUsage in training loop:");
    println!("  let mut pipeline = SmoothingPipeline::new(config);");
    println!("  for step in 0..max_steps {{");
    println!("      let loss = train_step(...);");
    println!("      let smoothed_loss = pipeline.process(loss);");
    println!("      // Use smoothed_loss for logging/visualization");
    println!("  }}");
}
