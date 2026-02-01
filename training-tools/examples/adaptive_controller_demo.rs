//! Demonstration of the adaptive training controller.
//!
//! This example shows how to use the `AdaptiveTrainingController` to
//! dynamically adjust training hyperparameters based on training dynamics.
//!
//! Run with: cargo run --example adaptive_controller_demo

use chrono::Utc;
use training_tools::adaptive::{
    AdaptationStrategy, AdaptiveConfig, AdaptiveLoopHelper, AdaptiveProgressiveConfig,
    AdaptiveTrainingController,
};
use training_tools::{StepMetrics, TrainingPhase};

/// Simulate a training step and return metrics.
fn simulate_step(step: u64, base_loss: f32, noise_level: f32) -> StepMetrics {
    // Simulate loss that generally decreases but has some noise
    let decay = (-0.01 * step as f32).exp();
    let noise = (step as f32 * 0.7).sin() * noise_level;
    let loss = base_loss * decay + 0.5 + noise;

    // Gradient norm correlates with loss
    let grad_norm = loss * 0.3 + noise.abs() * 0.1;

    StepMetrics {
        step,
        loss,
        gradient_norm: grad_norm,
        phase: if step < 50 {
            TrainingPhase::Warmup
        } else {
            TrainingPhase::Full
        },
        was_predicted: false,
        prediction_error: None,
        step_time_ms: 100.0 + noise.abs() * 10.0,
        timestamp: Utc::now(),
        tokens_this_step: 4096,
        total_tokens_trained: step * 4096,
        tokens_remaining: (1000 - step) * 4096,
        confidence: 0.85 + noise * 0.1,
        learning_rate: 1e-4,
        perplexity: loss.exp(),
        train_val_gap: None,
        loss_velocity: 0.0,
        loss_acceleration: 0.0,
        gradient_entropy: None,
        layer_gradients: None,
        layer_gradient_stats: None,
    }
}

fn main() {
    println!("=== Adaptive Training Controller Demo ===\n");

    // Example 1: Basic controller usage
    println!("--- Example 1: Basic Controller Usage ---\n");
    basic_controller_example();

    // Example 2: Using the loop helper
    println!("\n--- Example 2: Loop Helper Integration ---\n");
    loop_helper_example();

    // Example 3: Strategy switching
    println!("\n--- Example 3: Strategy Switching ---\n");
    strategy_example();

    // Example 4: Health monitoring
    println!("\n--- Example 4: Health Monitoring ---\n");
    health_monitoring_example();
}

fn basic_controller_example() {
    // Create controller with default configuration
    let config = AdaptiveConfig {
        initial_lr: 1e-4,
        min_lr: 1e-7,
        max_lr: 1e-2,
        initial_batch_size: 8,
        min_batch_size: 1,
        max_batch_size: 64,
        initial_grad_clip: 1.0,
        min_grad_clip: 0.1,
        max_grad_clip: 10.0,
        initial_warmup_steps: 50,
        initial_momentum_beta1: 0.9,
        model_size_params: 100_000_000,
        memory_threshold: 0.8,
        initial_strategy: AdaptationStrategy::Balanced,
        enable_predictive: true,
        health_check_interval: 50,
    };

    let mut controller = AdaptiveTrainingController::new(config);

    println!("Initial state:");
    println!("  Learning rate: {:.2e}", controller.current_lr());
    println!("  Batch size: {}", controller.current_batch_size());
    println!("  Gradient clip: {:.2}", controller.current_grad_clip());
    println!("  Strategy: {}", controller.current_strategy());

    // Simulate 100 training steps
    let mut adaptation_count = 0;
    for step in 0..100 {
        let metrics = simulate_step(step, 3.0, 0.1);
        let adaptations = controller.update(&metrics);

        if !adaptations.is_empty() {
            adaptation_count += adaptations.len();
            for adaptation in &adaptations {
                println!("  {}", adaptation.format_log());
            }
        }
    }

    println!("\nAfter 100 steps:");
    println!("  Learning rate: {:.2e}", controller.current_lr());
    println!("  Batch size: {}", controller.current_batch_size());
    println!("  Gradient clip: {:.2}", controller.current_grad_clip());
    println!("  Total adaptations: {}", adaptation_count);
}

fn loop_helper_example() {
    // Use the loop helper for easier integration
    let config = AdaptiveProgressiveConfig {
        base_learning_rate: 6e-4,
        batch_size: 4,
        gradient_clip: 1.0,
        adaptive_config: AdaptiveConfig::default(),
        log_adaptations: false,
        health_check_interval: 25,
        min_health_score: 0.2,
    };

    let mut helper = AdaptiveLoopHelper::new(config);

    println!("Training with loop helper...");

    for step in 0..50 {
        let metrics = simulate_step(step, 3.0, 0.2);
        let update = helper.process_step(&metrics);

        if update.has_changes() {
            println!("Step {}: {}", step, update.format_changes());
        }

        if update.should_pause {
            println!("Training paused at step {}", step);
            break;
        }
    }

    let (lr, batch, clip) = helper.current_state();
    println!(
        "\nFinal state: LR={:.2e}, Batch={}, Clip={:.2}",
        lr, batch, clip
    );
}

fn strategy_example() {
    let mut controller = AdaptiveTrainingController::new(AdaptiveConfig::default());

    println!("Testing different strategies:\n");

    for strategy in [
        AdaptationStrategy::Conservative,
        AdaptationStrategy::Balanced,
        AdaptationStrategy::Aggressive,
        AdaptationStrategy::Exploratory,
        AdaptationStrategy::Recovery,
    ] {
        controller.set_strategy(strategy);

        // Reset and run a few steps
        controller.reset();
        for step in 0..30 {
            let metrics = simulate_step(step, 2.5, 0.15);
            controller.update(&metrics);
        }

        println!(
            "Strategy: {:15} | LR: {:.2e} | Cooldown: {:3} steps | Threshold: {:.2}",
            format!("{}", strategy),
            controller.current_lr(),
            strategy.cooldown_steps(),
            strategy.confidence_threshold()
        );
    }
}

fn health_monitoring_example() {
    let mut controller = AdaptiveTrainingController::new(AdaptiveConfig::default());

    // Run some training steps
    for step in 0..100 {
        let metrics = simulate_step(step, 3.0, 0.1);
        controller.update(&metrics);
    }

    // Get health report
    let report = controller.get_health_report();

    println!("Training Health Report");
    println!("======================");
    println!(
        "Overall Health: {} ({:.2}/1.00)",
        report.overall_health, report.health_score
    );
    println!("Loss Trend: {}", report.loss_trend.description());
    println!("Gradient Health: {}", report.gradient_health);
    println!(
        "Steps Since Improvement: {}",
        report.steps_since_improvement
    );

    if !report.recommendations.is_empty() {
        println!("\nRecommendations:");
        for rec in &report.recommendations {
            println!("  - {}", rec);
        }
    }

    if !report.warnings.is_empty() {
        println!("\nWarnings:");
        for warning in &report.warnings {
            println!("  - {}", warning);
        }
    }

    // Get adaptation history summary
    let history = controller.get_adaptation_history();
    let summary = history.summary();
    println!("\n{}", summary.format());
}
