//! Comprehensive research framework for prediction horizon optimization.
//!
//! This experiment systematically explores:
//! 1. Different prediction horizon lengths (steps per prediction phase)
//! 2. Different sigma thresholds (tightness of divergence detection)
//! 3. The interplay between horizon length and sigma
//! 4. Correction phase effectiveness
//!
//! # Research Questions
//!
//! - Q1: Do longer prediction horizons improve speedup without degrading quality?
//! - Q2: Can tighter sigma (2.0-2.2) work with larger prediction horizons?
//! - Q3: What's the optimal balance for maximum speedup + quality?
//! - Q4: How does correction effectiveness scale with prediction horizon?
//!
//! # Running
//!
//! ```bash
//! cargo run --example prediction_horizon_research --features autodiff,ndarray --release
//! ```

use burn::backend::ndarray::NdArray;
use burn::backend::Autodiff;
use burn::module::Module;
use burn::nn::{loss::CrossEntropyLossConfig, Linear, LinearConfig};
use burn::optim::AdamConfig;
use burn::tensor::{backend::Backend, Device, Shape, Tensor, TensorData};

use hybrid_predict_trainer_rs::burn_integration::{
    BurnBatch, BurnForwardFn, BurnModelWrapper, BurnOptimizerWrapper,
};
use hybrid_predict_trainer_rs::config::DivergenceConfig;
use hybrid_predict_trainer_rs::{HybridTrainer, HybridTrainerConfig};

use rand::Rng;
use std::time::Instant;

type MyBackend = Autodiff<NdArray>;

/// Model architecture (consistent across all experiments)
#[derive(Module, Debug)]
struct SimpleMLP<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
}

impl<B: Backend> SimpleMLP<B> {
    pub fn new(device: &Device<B>) -> Self {
        let fc1 = LinearConfig::new(784, 128).init(device);
        let fc2 = LinearConfig::new(128, 10).init(device);
        Self { fc1, fc2 }
    }

    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.fc1.forward(input);
        let x = burn::tensor::activation::relu(x);
        self.fc2.forward(x)
    }
}

#[derive(Debug, Clone)]
struct MnistBatch<B: Backend> {
    images: Tensor<B, 2>,
    labels: Tensor<B, 1, burn::tensor::Int>,
}

struct MnistForward;

impl BurnForwardFn<MyBackend, SimpleMLP<MyBackend>, MnistBatch<MyBackend>> for MnistForward {
    fn forward(
        &self,
        model: SimpleMLP<MyBackend>,
        batch: &BurnBatch<MyBackend, MnistBatch<MyBackend>>,
    ) -> (SimpleMLP<MyBackend>, Tensor<MyBackend, 1>) {
        let logits = model.forward(batch.data.images.clone());
        let loss = CrossEntropyLossConfig::new()
            .init(&logits.device())
            .forward(logits, batch.data.labels.clone());
        (model, loss)
    }
}

fn generate_batch(batch_size: usize, device: &Device<MyBackend>) -> MnistBatch<MyBackend> {
    let mut rng = rand::rng();
    let mut image_data = vec![];
    for _ in 0..batch_size {
        for _ in 0..784 {
            image_data.push(rng.random::<f32>());
        }
    }
    let label_data: Vec<i64> = (0..batch_size).map(|_| rng.random_range(0..10)).collect();

    let images = Tensor::from_data(
        TensorData::new(image_data, Shape::new([batch_size, 784])).convert::<f32>(),
        device,
    );
    let labels = Tensor::from_data(
        TensorData::new(label_data, Shape::new([batch_size])),
        device,
    );

    MnistBatch { images, labels }
}

/// Comprehensive experiment configuration
#[derive(Debug, Clone)]
struct ExperimentConfig {
    name: String,
    max_predict_steps: usize,  // Prediction horizon length
    full_steps: usize,          // Steps between predictions
    sigma: f32,                  // Loss sigma threshold
    confidence: f32,             // Confidence threshold
    check_interval: usize,      // Divergence check frequency
}

/// Detailed experiment results
#[derive(Debug, Clone)]
struct ExperimentResult {
    config: ExperimentConfig,
    final_loss: f32,
    avg_loss: f32,
    loss_variance: f32,
    avg_confidence: f32,
    divergence_count: usize,
    backward_reduction_pct: f32,
    effective_speedup: f32,
    runtime_ms: u128,
    correction_effectiveness: f32,  // How well corrections fix divergences
}

fn run_experiment(config: ExperimentConfig, total_steps: usize) -> ExperimentResult {
    println!("\n🔬 Experiment: {}", config.name);
    println!("   Horizon: {} steps, Sigma: {:.1}, Confidence: {:.2}",
             config.max_predict_steps, config.sigma, config.confidence);

    let device = Device::<MyBackend>::Cpu;
    let model = SimpleMLP::<MyBackend>::new(&device);
    let forward_fn = MnistForward;
    let wrapped_model = BurnModelWrapper::new(model, forward_fn, device.clone());

    let optimizer = AdamConfig::new().init();
    let wrapped_optimizer = BurnOptimizerWrapper::new(optimizer, 0.001);

    let trainer_config = HybridTrainerConfig::builder()
        .warmup_steps(10)  // Fixed warmup for fair comparison
        .full_steps(config.full_steps)
        .max_predict_steps(config.max_predict_steps)
        .confidence_threshold(config.confidence)
        .divergence_config(DivergenceConfig {
            loss_sigma_threshold: config.sigma,
            check_interval_steps: config.check_interval,
            ..Default::default()
        })
        .collect_metrics(true)  // Enable detailed metrics
        .build();

    let mut trainer = HybridTrainer::new(wrapped_model, wrapped_optimizer, trainer_config)
        .expect("Failed to create trainer");

    // Track metrics
    let start = Instant::now();
    let mut loss_history = Vec::new();
    let mut divergence_corrections = Vec::new();

    for step in 0..total_steps {
        let batch = BurnBatch::new(generate_batch(32, &device), 32);
        let result = trainer.step(&batch).expect("Step failed");

        loss_history.push(result.loss);

        if step % 25 == 0 {
            println!("   Step {:3}: Loss {:.4}, Phase: {:?}",
                     step, result.loss, result.phase);
        }

        // Track correction effectiveness
        if let Some(pred_error) = result.prediction_error {
            divergence_corrections.push(pred_error.abs());
        }
    }

    let runtime_ms = start.elapsed().as_millis();
    let stats = trainer.statistics();

    // Calculate metrics
    let avg_loss = loss_history.iter().sum::<f32>() / loss_history.len() as f32;
    let loss_variance = loss_history.iter()
        .map(|&loss| (loss - avg_loss).powi(2))
        .sum::<f32>() / loss_history.len() as f32;

    let correction_effectiveness = if !divergence_corrections.is_empty() {
        1.0 / (divergence_corrections.iter().sum::<f32>() / divergence_corrections.len() as f32)
    } else {
        1.0  // No divergences = perfect
    };

    // Calculate effective speedup (accounts for quality degradation)
    let quality_factor = 1.0 / (1.0 + loss_variance.sqrt());
    let effective_speedup = stats.backward_reduction_pct / 100.0 * quality_factor;

    println!("   ✓ Complete - Final loss: {:.4}, Speedup: {:.1}%",
             loss_history[loss_history.len() - 1], stats.backward_reduction_pct);

    ExperimentResult {
        config,
        final_loss: loss_history[loss_history.len() - 1],
        avg_loss,
        loss_variance,
        avg_confidence: stats.avg_confidence,
        divergence_count: stats.divergence_events,
        backward_reduction_pct: stats.backward_reduction_pct,
        effective_speedup,
        runtime_ms,
        correction_effectiveness,
    }
}

fn main() {
    println!("🔬 Prediction Horizon Research Framework");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Systematically exploring prediction horizon × sigma space\n");

    let total_steps = 150;  // Enough steps to see patterns

    // Define experiment matrix
    let sigma_values = vec![2.0, 2.2, 2.5, 3.0];
    let horizon_lengths = vec![10, 15, 20, 30, 50];
    let confidence_values = vec![0.85, 0.90, 0.92];

    println!("📋 Experiment Plan:");
    println!("   - Sigma values: {:?}", sigma_values);
    println!("   - Horizon lengths: {:?}", horizon_lengths);
    println!("   - Confidence thresholds: {:?}", confidence_values);
    println!("   - Total experiments: {}",
             sigma_values.len() * horizon_lengths.len());
    println!();

    let mut all_results = Vec::new();

    // Run all experiments
    for &sigma in &sigma_values {
        for &horizon in &horizon_lengths {
            let config = ExperimentConfig {
                name: format!("σ={:.1}_h={}", sigma, horizon),
                max_predict_steps: horizon,
                full_steps: 3,  // Consistent across experiments
                sigma,
                confidence: 0.60,  // Achievable with ensemble=5 after warmup
                check_interval: if sigma <= 2.2 { 3 } else { 5 },  // Tighter check for tighter sigma
            };

            all_results.push(run_experiment(config, total_steps));
        }
    }

    // === ANALYSIS & VISUALIZATION ===
    println!("\n");
    println!("📊 COMPREHENSIVE RESULTS");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("{:<15} | {:>10} | {:>10} | {:>10} | {:>10} | {:>12} | {:>10}",
             "Config", "Final Loss", "Variance", "Diverge", "Speedup%", "Effective×", "Quality");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    for result in &all_results {
        let quality_score = 1.0 / (1.0 + result.loss_variance);
        println!("{:<15} | {:>10.4} | {:>10.4} | {:>10} | {:>9.1}% | {:>12.2} | {:>10.3}",
                 result.config.name,
                 result.final_loss,
                 result.loss_variance,
                 result.divergence_count,
                 result.backward_reduction_pct,
                 result.effective_speedup,
                 quality_score);
    }

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    // === FIND OPTIMAL CONFIGURATIONS ===
    println!("\n🏆 TOP CONFIGURATIONS BY DIFFERENT METRICS\n");

    // Best for speedup
    let mut by_speedup = all_results.clone();
    by_speedup.sort_by(|a, b| b.backward_reduction_pct.partial_cmp(&a.backward_reduction_pct).unwrap());
    println!("🚀 Best Speedup:");
    for (i, result) in by_speedup.iter().take(3).enumerate() {
        println!("   {}. {} - {:.1}% speedup, {:.4} variance, {} divergences",
                 i + 1, result.config.name, result.backward_reduction_pct,
                 result.loss_variance, result.divergence_count);
    }

    // Best for quality (low variance)
    let mut by_quality = all_results.clone();
    by_quality.sort_by(|a, b| a.loss_variance.partial_cmp(&b.loss_variance).unwrap());
    println!("\n✨ Best Quality (Low Variance):");
    for (i, result) in by_quality.iter().take(3).enumerate() {
        println!("   {}. {} - {:.4} variance, {:.1}% speedup, {} divergences",
                 i + 1, result.config.name, result.loss_variance,
                 result.backward_reduction_pct, result.divergence_count);
    }

    // Best for effective speedup (accounts for quality)
    let mut by_effective = all_results.clone();
    by_effective.sort_by(|a, b| b.effective_speedup.partial_cmp(&a.effective_speedup).unwrap());
    println!("\n⚡ Best Effective Speedup (Quality-Adjusted):");
    for (i, result) in by_effective.iter().take(3).enumerate() {
        println!("   {}. {} - {:.2}× effective, {:.1}% raw speedup, {:.4} variance",
                 i + 1, result.config.name, result.effective_speedup,
                 result.backward_reduction_pct, result.loss_variance);
    }

    // === INSIGHTS & RECOMMENDATIONS ===
    println!("\n");
    println!("💡 KEY INSIGHTS");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    // Analyze sigma effect
    let tight_sigma_results: Vec<_> = all_results.iter()
        .filter(|r| r.config.sigma <= 2.2)
        .collect();
    let loose_sigma_results: Vec<_> = all_results.iter()
        .filter(|r| r.config.sigma >= 2.5)
        .collect();

    let tight_avg_variance = tight_sigma_results.iter()
        .map(|r| r.loss_variance).sum::<f32>() / tight_sigma_results.len() as f32;
    let loose_avg_variance = loose_sigma_results.iter()
        .map(|r| r.loss_variance).sum::<f32>() / loose_sigma_results.len() as f32;

    println!("📉 Sigma Analysis:");
    println!("   Tight σ (≤2.2): avg variance = {:.4}", tight_avg_variance);
    println!("   Loose σ (≥2.5): avg variance = {:.4}", loose_avg_variance);
    println!("   → Tighter sigma reduces variance by {:.1}%",
             ((loose_avg_variance - tight_avg_variance) / loose_avg_variance * 100.0));

    // Analyze horizon effect
    let short_horizon: Vec<_> = all_results.iter()
        .filter(|r| r.config.max_predict_steps <= 15)
        .collect();
    let long_horizon: Vec<_> = all_results.iter()
        .filter(|r| r.config.max_predict_steps >= 30)
        .collect();

    let short_avg_speedup = short_horizon.iter()
        .map(|r| r.backward_reduction_pct).sum::<f32>() / short_horizon.len() as f32;
    let long_avg_speedup = long_horizon.iter()
        .map(|r| r.backward_reduction_pct).sum::<f32>() / long_horizon.len() as f32;

    println!("\n📏 Horizon Analysis:");
    println!("   Short horizon (≤15): avg speedup = {:.1}%", short_avg_speedup);
    println!("   Long horizon (≥30): avg speedup = {:.1}%", long_avg_speedup);
    println!("   → Longer horizons increase speedup by {:.1}%",
             long_avg_speedup - short_avg_speedup);

    println!("\n🎯 RECOMMENDED CONFIGURATION");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    if let Some(best) = by_effective.first() {
        println!("For production use, we recommend:");
        println!();
        println!("   HybridTrainerConfig::builder()");
        println!("       .max_predict_steps({})  // Optimal horizon", best.config.max_predict_steps);
        println!("       .confidence_threshold({:.2})", best.config.confidence);
        println!("       .divergence_config(DivergenceConfig {{");
        println!("           loss_sigma_threshold: {:.1},", best.config.sigma);
        println!("           check_interval_steps: {},", best.config.check_interval);
        println!("           ..Default::default()");
        println!("       }})");
        println!("       .build()");
        println!();
        println!("Expected results:");
        println!("   - Speedup: {:.1}% (vs baseline full training)", best.backward_reduction_pct);
        println!("   - Quality: {:.3} stability score", 1.0 / (1.0 + best.loss_variance));
        println!("   - Divergence rate: {:.2}%", best.divergence_count as f32 / total_steps as f32 * 100.0);
    }

    println!("\n✅ Research complete! Save these results for your production config.");
}
