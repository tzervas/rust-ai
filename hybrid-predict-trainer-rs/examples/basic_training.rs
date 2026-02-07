//! Basic hybrid training example with mock model.
//!
//! This example demonstrates the complete usage of the `HybridTrainer`
//! with a mock model and optimizer to show the full training loop,
//! phase transitions, and metrics collection.
//!
//! # Running
//!
//! ```bash
//! cargo run --example basic_training
//! ```

use hybrid_predict_trainer_rs::prelude::*;
use hybrid_predict_trainer_rs::state::WeightDelta;

/// Mock batch implementing the Batch trait.
#[derive(Debug, Clone)]
struct MockBatch {
    /// Simulated input data (unused in mock, but demonstrates real batch structure)
    #[allow(dead_code)]
    data: Vec<f32>,
    /// Batch size
    size: usize,
}

impl MockBatch {
    fn new(size: usize) -> Self {
        Self {
            data: vec![1.0; size * 10], // 10 features per sample
            size,
        }
    }
}

impl Batch for MockBatch {
    fn batch_size(&self) -> usize {
        self.size
    }
}

/// Mock model implementing the Model trait.
///
/// Simulates a simple neural network with gradually decreasing loss
/// to demonstrate convergent training behavior.
struct MockModel {
    /// Model "weights" (just a single value for simplicity)
    weights: Vec<f32>,
    /// Current iteration for loss computation
    iteration: u32,
    /// Last forward pass loss (cached for backward)
    last_loss: f32,
}

impl MockModel {
    fn new() -> Self {
        Self {
            weights: vec![1.0; 1000], // 1000 parameters
            iteration: 0,
            last_loss: 0.0,
        }
    }
}

impl<B: Batch> Model<B> for MockModel {
    fn forward(&mut self, _batch: &B) -> HybridResult<f32> {
        // Simulate realistic loss decay with some noise
        let iter_f32 = self.iteration as f32;
        let base_loss = 3.0 * (-(iter_f32 * 0.002)).exp();
        let noise = (iter_f32 * 0.1).sin() * 0.05;
        let loss = base_loss + noise + 0.1;

        self.last_loss = loss;
        self.iteration += 1;

        Ok(loss)
    }

    fn backward(&mut self) -> HybridResult<GradientInfo> {
        // Simulate gradient computation with realistic gradient norms
        let grad_norm = self.last_loss * 0.5 * (1.0 + (self.iteration as f32 * 0.05).sin() * 0.2);

        Ok(GradientInfo {
            loss: self.last_loss,
            gradient_norm: grad_norm,
            per_param_norms: None,
        })
    }

    fn parameter_count(&self) -> usize {
        self.weights.len()
    }

    fn apply_weight_delta(&mut self, delta: &WeightDelta) -> HybridResult<()> {
        // Apply the weight delta (scaled)
        if let Some(param_delta) = delta.deltas.get("weights") {
            for (w, d) in self.weights.iter_mut().zip(param_delta.iter()) {
                *w += d * delta.scale;
            }
        }
        Ok(())
    }
}

/// Mock optimizer implementing the Optimizer trait.
///
/// Simulates Adam optimizer behavior with momentum and adaptive learning rates.
struct MockOptimizer {
    /// Learning rate
    lr: f32,
    /// First moment (momentum)
    momentum: Vec<f32>,
    /// Second moment (variance)
    variance: Vec<f32>,
    /// Beta1 for momentum
    beta1: f32,
    /// Beta2 for variance
    beta2: f32,
    /// Iteration counter
    t: u64,
}

impl MockOptimizer {
    fn new(lr: f32, param_count: usize) -> Self {
        Self {
            lr,
            momentum: vec![0.0; param_count],
            variance: vec![0.0; param_count],
            beta1: 0.9,
            beta2: 0.999,
            t: 0,
        }
    }
}

impl<M, B> Optimizer<M, B> for MockOptimizer
where
    M: Model<B>,
    B: Batch,
{
    fn step(&mut self, model: &mut M, gradients: &GradientInfo) -> HybridResult<()> {
        self.t += 1;

        // Simulate applying optimizer step
        // In a real implementation, this would update the model parameters
        // For this mock, we just simulate the state updates

        // Simulate momentum and variance updates
        let avg_grad = gradients.gradient_norm / model.parameter_count() as f32;
        for i in 0..model.parameter_count().min(self.momentum.len()) {
            self.momentum[i] = self.beta1 * self.momentum[i] + (1.0 - self.beta1) * avg_grad;
            self.variance[i] =
                self.beta2 * self.variance[i] + (1.0 - self.beta2) * avg_grad * avg_grad;
        }

        Ok(())
    }

    fn learning_rate(&self) -> f32 {
        self.lr
    }

    fn set_learning_rate(&mut self, lr: f32) {
        self.lr = lr;
    }

    fn zero_grad(&mut self) {
        // Clear gradients (mock - no action needed)
    }
}

fn main() -> HybridResult<()> {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║         Hybrid Predictive Training - Full Example            ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    // Build configuration with auto-tuning enabled
    let num_steps = 200;
    let auto_tuning_config = hybrid_predict_trainer_rs::auto_tuning::AutoTuningConfig::default();

    let config = HybridTrainerConfig::builder()
        .warmup_steps(50) // Shorter warmup for demo
        .full_steps(10) // Full training steps per cycle
        .max_predict_steps(30) // Max prediction steps
        .confidence_threshold(0.80) // Lower threshold for demo
        .collect_metrics(true) // Enable metrics
        .auto_tuning(auto_tuning_config) // Enable auto-tuning
        .max_steps(num_steps) // Total training steps for progress calculation
        .build();

    println!("📋 Configuration:");
    println!("   Warmup steps:         {}", config.warmup_steps);
    println!("   Full steps per cycle: {}", config.full_steps);
    println!("   Max predict steps:    {}", config.max_predict_steps);
    println!(
        "   Confidence threshold: {:.2}",
        config.confidence_threshold
    );
    println!(
        "   Auto-tuning:          {}",
        if config.auto_tuning_config.is_some() {
            "Enabled"
        } else {
            "Disabled"
        }
    );
    println!();

    // Validate configuration
    config.validate()?;

    // Create model and optimizer
    let model = MockModel::new();
    let param_count = model.weights.len();
    let optimizer = MockOptimizer::new(0.001, param_count);

    println!("🤖 Model initialized:");
    println!("   Parameters: {}", param_count);
    println!();

    // Create hybrid trainer
    let mut trainer = HybridTrainer::new(model, optimizer, config)?;

    println!("🚀 Starting training loop...\n");
    println!("┌──────┬──────────┬─────────┬───────────┬────────┬───────────┐");
    println!("│ Step │  Phase   │  Loss   │ Predicted │  Conf  │  Time(ms) │");
    println!("├──────┼──────────┼─────────┼───────────┼────────┼───────────┤");
    let mut phase_transitions = Vec::new();
    let mut last_phase = Phase::Warmup;

    // Training loop
    for step in 0..num_steps {
        let batch = MockBatch::new(32);

        match trainer.step(&batch) {
            Ok(result) => {
                // Track phase transitions
                if result.phase != last_phase {
                    phase_transitions.push((step, result.phase));
                    last_phase = result.phase;
                }

                // Print progress every 10 steps or on phase transitions
                if step % 10 == 0 || step < 60 {
                    let predicted_str = if result.was_predicted { "Yes" } else { "No " };
                    println!(
                        "│ {:4} │ {:8?} │ {:7.4} │    {}    │ {:5.2}  │  {:7.2}  │",
                        step,
                        result.phase,
                        result.loss,
                        predicted_str,
                        result.confidence,
                        result.step_time_ms
                    );

                    // Show auto-tuning recommendations every 50 steps
                    if step % 50 == 0 {
                        if let Some(update) = trainer.last_auto_tuning_update() {
                            println!(
                                "│ 🔧 Auto-tuning: Health={:?}, Score={:.2}, Plateau={:?}",
                                update.health, update.health_score, update.plateau_status
                            );
                            if update.should_restart() {
                                println!(
                                    "│    Warmup restart recommended (LR x{:.2})",
                                    update.warmup_restart.unwrap()
                                );
                            }
                            if !update.recommendations.is_empty() {
                                println!("│    Recommendations: {:?}", update.recommendations);
                            }
                        }
                    }
                }
            }
            Err((error, recovery_action)) => {
                println!("\n⚠️  Training error at step {}: {:?}", step, error);
                if let Some(action) = recovery_action {
                    println!("   Recovery action: {:?}", action);
                    // In a real scenario, handle the recovery action
                    if !action.can_continue() {
                        break;
                    }
                }
            }
        }
    }

    println!("└──────┴──────────┴─────────┴───────────┴────────┴───────────┘\n");

    // Print phase transitions
    println!("📊 Phase Transitions:");
    for (step, phase) in phase_transitions.iter() {
        println!("   Step {:3} → {:?}", step, phase);
    }
    println!();

    // Print final statistics
    let stats = trainer.statistics();
    println!("📈 Training Statistics:");
    println!("   Total steps:        {}", stats.total_steps);
    println!(
        "   Predicted steps:    {} ({:.1}%)",
        stats.predict_steps,
        stats.predict_steps as f32 / stats.total_steps.max(1) as f32 * 100.0
    );
    println!("   Full steps:         {}", stats.full_steps);
    println!("   Warmup steps:       {}", stats.warmup_steps);
    println!("   Correct steps:      {}", stats.correct_steps);
    println!("   Final loss:         {:.4}", stats.final_loss);
    println!(
        "   Backward reduction: {:.1}%",
        stats.backward_reduction_pct
    );
    println!("   Average conf:       {:.2}", stats.avg_confidence);

    // Calculate speedup
    let theoretical_speedup = if stats.full_steps > 0 {
        stats.total_steps as f32 / stats.full_steps as f32
    } else {
        1.0
    };

    println!("\n⚡ Performance:");
    println!("   Theoretical speedup: {:.2}x", theoretical_speedup);
    println!("   (Speedup = total steps / full compute steps)");

    println!("\n✅ Training completed successfully!");

    Ok(())
}
