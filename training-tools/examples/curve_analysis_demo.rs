//! Demonstration of comprehensive training curve analysis.
//!
//! Run with: cargo run --example curve_analysis_demo

use std::f32::consts::E;

fn main() {
    println!("=== Training Curve Analysis Demo ===\n");

    // Demo 1: Healthy exponential decay
    demo_healthy_decay();
    println!("\n{}\n", "=".repeat(60));

    // Demo 2: Plateau detection
    demo_plateau();
    println!("\n{}\n", "=".repeat(60));

    // Demo 3: Diverging training
    demo_divergence();
    println!("\n{}\n", "=".repeat(60));

    // Demo 4: Oscillating loss
    demo_oscillation();
    println!("\n{}\n", "=".repeat(60));

    // Demo 5: Linear decay (suboptimal)
    demo_linear_decay();
}

fn demo_healthy_decay() {
    use training_tools::CurveAnalyzer;

    println!("Demo 1: Healthy Exponential Decay");
    println!("{}", "-".repeat(40));

    let mut analyzer = CurveAnalyzer::new(100_000_000); // 100M params

    // Generate exponential decay: 5.0 * exp(-0.05*t) + 2.0
    for i in 0..100 {
        let loss = 5.0 * E.powf(-0.05 * i as f32) + 2.0;
        analyzer.add_loss(loss);
    }

    let analysis = analyzer.analyze();
    println!("{}", analysis.report());
}

fn demo_plateau() {
    use training_tools::CurveAnalyzer;

    println!("Demo 2: Plateau Detection");
    println!("{}", "-".repeat(40));

    let mut analyzer = CurveAnalyzer::new(100_000_000);

    // Initial decay, then plateau
    for i in 0..50 {
        let loss = 5.0 * E.powf(-0.1 * i as f32) + 2.5;
        analyzer.add_loss(loss);
    }

    // Plateau with small noise
    for _ in 50..150 {
        let loss = 2.5 + (rand::random::<f32>() - 0.5) * 0.02;
        analyzer.add_loss(loss);
    }

    let analysis = analyzer.analyze();
    println!("{}", analysis.report());
}

fn demo_divergence() {
    use training_tools::CurveAnalyzer;

    println!("Demo 3: Diverging Training (Loss Increasing)");
    println!("{}", "-".repeat(40));

    let mut analyzer = CurveAnalyzer::new(100_000_000);

    // Brief initial decrease, then divergence
    for i in 0..20 {
        let loss = 4.0 - 0.05 * i as f32;
        analyzer.add_loss(loss);
    }

    // Loss starts increasing
    for i in 20..100 {
        let loss = 3.0 + 0.08 * (i - 20) as f32;
        analyzer.add_loss(loss);
    }

    let analysis = analyzer.analyze();
    println!("{}", analysis.report());
}

fn demo_oscillation() {
    use training_tools::CurveAnalyzer;

    println!("Demo 4: Oscillating Loss (High Variance)");
    println!("{}", "-".repeat(40));

    let mut analyzer = CurveAnalyzer::new(100_000_000);

    // High variance oscillations around decreasing trend
    for i in 0..100 {
        let base = 5.0 * E.powf(-0.02 * i as f32) + 2.0;
        let oscillation = (i as f32 * 0.3).sin() * 1.5;
        let loss = base + oscillation;
        analyzer.add_loss(loss);
    }

    let analysis = analyzer.analyze();
    println!("{}", analysis.report());
}

fn demo_linear_decay() {
    use training_tools::CurveAnalyzer;

    println!("Demo 5: Linear Decay (Slower than Expected)");
    println!("{}", "-".repeat(40));

    let mut analyzer = CurveAnalyzer::new(100_000_000);

    // Linear decay instead of exponential
    for i in 0..100 {
        let loss = 5.0 - 0.02 * i as f32;
        analyzer.add_loss(loss);
    }

    let analysis = analyzer.analyze();
    println!("{}", analysis.report());
}
