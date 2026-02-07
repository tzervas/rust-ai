//! Comprehensive 3D parameter sweep to validate recent optimizations.
//!
//! This experiment systematically explores the interaction between:
//! 1. **Correction Interval** (NEW) - Micro-corrections during predict phase
//! 2. **Prediction Horizon** - Steps per prediction phase
//! 3. **Confidence Threshold** - Minimum confidence for predictions
//!
//! # Research Questions
//!
//! - Q1: Do micro-corrections enable longer prediction horizons?
//! - Q2: What's the optimal correction interval frequency?
//! - Q3: How does confidence threshold interact with micro-corrections?
//! - Q4: What's the new Pareto frontier for speedup vs quality?
//!
//! # Experiment Design
//!
//! - **Total configurations**: 60 (5 × 4 × 3)
//! - **Steps per config**: 200 training steps
//! - **Task**: Synthetic quadratic regression (consistent baseline)
//! - **Metrics**: Loss variance, speedup, divergences, micro-correction count
//!
//! # Running
//!
//! ```bash
//! cargo run --example comprehensive_parameter_sweep --features autodiff,ndarray --release
//! ```

use burn::backend::ndarray::NdArray;
use burn::backend::Autodiff;
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
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

/// Simple linear model for quadratic regression: y = w * x + b
#[derive(Module, Debug)]
struct QuadraticModel<B: Backend> {
    linear: Linear<B>,
}

impl<B: Backend> QuadraticModel<B> {
    pub fn new(device: &Device<B>) -> Self {
        let linear = LinearConfig::new(1, 1).init(device);
        Self { linear }
    }

    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        self.linear.forward(input)
    }
}

#[derive(Debug, Clone)]
struct QuadraticBatch<B: Backend> {
    x: Tensor<B, 2>,
    y: Tensor<B, 2>,
}

struct QuadraticForward;

impl BurnForwardFn<MyBackend, QuadraticModel<MyBackend>, QuadraticBatch<MyBackend>> for QuadraticForward {
    fn forward(
        &self,
        model: QuadraticModel<MyBackend>,
        batch: &BurnBatch<MyBackend, QuadraticBatch<MyBackend>>,
    ) -> (QuadraticModel<MyBackend>, Tensor<MyBackend, 1>) {
        let pred = model.forward(batch.data.x.clone());
        let loss = (pred - batch.data.y.clone()).powf_scalar(2.0).mean();
        (model, loss)
    }
}

/// Generate synthetic quadratic data: y = 2x + noise
fn generate_batch(batch_size: usize, device: &Device<MyBackend>) -> QuadraticBatch<MyBackend> {
    let mut rng = rand::rng();
    let mut x_data = vec![];
    let mut y_data = vec![];

    for _ in 0..batch_size {
        let x = rng.random::<f32>() * 10.0 - 5.0; // [-5, 5]
        let noise = (rng.random::<f32>() - 0.5) * 0.5;
        let y = 2.0 * x + noise;
        x_data.push(x);
        y_data.push(y);
    }

    let x = Tensor::from_data(
        TensorData::new(x_data, Shape::new([batch_size, 1])).convert::<f32>(),
        device,
    );
    let y = Tensor::from_data(
        TensorData::new(y_data, Shape::new([batch_size, 1])).convert::<f32>(),
        device,
    );

    QuadraticBatch { x, y }
}

/// Experiment configuration (3D parameter space)
#[derive(Debug, Clone)]
struct ExperimentConfig {
    name: String,
    correction_interval: usize,  // 0 = disabled, >0 = micro-correction frequency
    max_predict_steps: usize,    // Prediction horizon
    confidence_threshold: f32,    // Confidence threshold
}

/// Comprehensive experiment results
#[derive(Debug, Clone)]
struct ExperimentResult {
    config: ExperimentConfig,
    final_loss: f32,
    avg_loss: f32,
    loss_variance: f32,
    loss_std_dev: f32,
    backward_reduction_pct: f32,
    divergence_count: usize,
    micro_corrections_count: usize,
    quality_score: f32,           // 1 / (1 + variance)
    effective_speedup: f32,       // Quality-adjusted speedup
    runtime_ms: u128,
}

fn run_experiment(config: ExperimentConfig, total_steps: usize) -> ExperimentResult {
    let progress_pct = 0; // Will be set by caller
    println!(
        "\n  🧪 [{:2}%] Testing: {} (interval={}, horizon={}, conf={:.2})",
        progress_pct,
        config.name,
        if config.correction_interval > 0 {
            config.correction_interval.to_string()
        } else {
            "off".to_string()
        },
        config.max_predict_steps,
        config.confidence_threshold
    );

    let device = Device::<MyBackend>::Cpu;
    let model = QuadraticModel::<MyBackend>::new(&device);
    let forward_fn = QuadraticForward;
    let wrapped_model = BurnModelWrapper::new(model, forward_fn, device.clone());

    let optimizer = AdamConfig::new().init();
    let wrapped_optimizer = BurnOptimizerWrapper::new(optimizer, 0.01);

    let trainer_config = HybridTrainerConfig::builder()
        .warmup_steps(20)  // Enough to establish baseline
        .full_steps(3)     // Fixed for fair comparison
        .max_predict_steps(config.max_predict_steps)
        .confidence_threshold(config.confidence_threshold)
        .correction_interval(config.correction_interval)
        .divergence_config(DivergenceConfig {
            loss_sigma_threshold: 2.5,  // Fixed for comparison
            check_interval_steps: 5,
            ..Default::default()
        })
        .collect_metrics(true)
        .build();

    let mut trainer = HybridTrainer::new(wrapped_model, wrapped_optimizer, trainer_config)
        .expect("Failed to create trainer");

    // Track metrics
    let start = Instant::now();
    let mut loss_history = Vec::new();

    for _step in 0..total_steps {
        let batch = BurnBatch::new(generate_batch(32, &device), 32);
        let result = trainer.step(&batch).expect("Step failed");
        loss_history.push(result.loss);
    }

    let runtime_ms = start.elapsed().as_millis();
    let stats = trainer.statistics();

    // Calculate loss statistics
    let avg_loss = loss_history.iter().sum::<f32>() / loss_history.len() as f32;
    let loss_variance = loss_history
        .iter()
        .map(|&loss| (loss - avg_loss).powi(2))
        .sum::<f32>()
        / loss_history.len() as f32;
    let loss_std_dev = loss_variance.sqrt();

    // Quality and speedup metrics
    let quality_score = 1.0 / (1.0 + loss_variance);
    let effective_speedup = (stats.backward_reduction_pct / 100.0) * quality_score;

    ExperimentResult {
        config,
        final_loss: loss_history[loss_history.len() - 1],
        avg_loss,
        loss_variance,
        loss_std_dev,
        backward_reduction_pct: stats.backward_reduction_pct,
        divergence_count: stats.divergence_events,
        micro_corrections_count: stats.micro_corrections_applied,
        quality_score,
        effective_speedup,
        runtime_ms,
    }
}

fn main() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("🔬 COMPREHENSIVE 3D PARAMETER SWEEP");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Validating micro-correction optimizations across parameter space\n");

    let total_steps = 200;  // Enough to see convergence patterns

    // Define parameter space
    let correction_intervals = vec![0, 5, 10, 15, 20];  // 0 = baseline (disabled)
    let prediction_horizons = vec![30, 50, 75, 100];
    let confidence_thresholds = vec![0.55, 0.60, 0.65];

    let total_experiments = correction_intervals.len()
        * prediction_horizons.len()
        * confidence_thresholds.len();

    println!("📋 Experiment Plan:");
    println!("   - Correction intervals: {:?}", correction_intervals);
    println!("   - Prediction horizons: {:?}", prediction_horizons);
    println!("   - Confidence thresholds: {:?}", confidence_thresholds);
    println!("   - Total configurations: {}", total_experiments);
    println!("   - Steps per config: {}", total_steps);
    println!();

    let mut all_results = Vec::new();
    let mut experiment_num = 0;

    let start_time = Instant::now();

    // 3D parameter sweep
    for &interval in &correction_intervals {
        for &horizon in &prediction_horizons {
            for &confidence in &confidence_thresholds {
                experiment_num += 1;
                let progress_pct = (experiment_num * 100) / total_experiments;

                let config = ExperimentConfig {
                    name: format!("I{}_H{}_C{:.2}", interval, horizon, confidence),
                    correction_interval: interval,
                    max_predict_steps: horizon,
                    confidence_threshold: confidence,
                };

                print!(
                    "  🧪 [{:2}%] Testing: {} (interval={}, horizon={}, conf={:.2})",
                    progress_pct,
                    config.name,
                    if interval > 0 {
                        interval.to_string()
                    } else {
                        "off".to_string()
                    },
                    horizon,
                    confidence
                );

                let result = run_experiment(config, total_steps);

                println!(
                    " → Speedup: {:.1}%, Variance: {:.4}, Diverge: {}",
                    result.backward_reduction_pct,
                    result.loss_variance,
                    result.divergence_count
                );

                all_results.push(result);
            }
        }
    }

    let total_runtime = start_time.elapsed().as_secs();

    println!("\n✅ All experiments complete in {}s", total_runtime);

    // === DETAILED RESULTS TABLE ===
    println!("\n");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("📊 COMPREHENSIVE RESULTS (60 CONFIGURATIONS)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!(
        "{:<12} | {:>8} | {:>8} | {:>8} | {:>8} | {:>8} | {:>7} | {:>7}",
        "Config", "Speedup%", "Variance", "StdDev", "Quality", "EffSpd", "Diverg", "MicroC"
    );
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    for result in &all_results {
        println!(
            "{:<12} | {:>7.1}% | {:>8.4} | {:>8.4} | {:>8.3} | {:>8.2} | {:>7} | {:>7}",
            result.config.name,
            result.backward_reduction_pct,
            result.loss_variance,
            result.loss_std_dev,
            result.quality_score,
            result.effective_speedup,
            result.divergence_count,
            result.micro_corrections_count
        );
    }

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    // === TOP CONFIGURATIONS ===
    println!("\n");
    println!("🏆 TOP CONFIGURATIONS BY DIFFERENT METRICS");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Best speedup
    let mut by_speedup = all_results.clone();
    by_speedup.sort_by(|a, b| {
        b.backward_reduction_pct
            .partial_cmp(&a.backward_reduction_pct)
            .unwrap()
    });
    println!("🚀 Best Speedup (Top 5):");
    for (i, result) in by_speedup.iter().take(5).enumerate() {
        println!(
            "   {}. {} - {:.1}% speedup, {:.4} variance, {} divergences, {} micro-corrections",
            i + 1,
            result.config.name,
            result.backward_reduction_pct,
            result.loss_variance,
            result.divergence_count,
            result.micro_corrections_count
        );
    }

    // Best quality
    let mut by_quality = all_results.clone();
    by_quality.sort_by(|a, b| a.loss_variance.partial_cmp(&b.loss_variance).unwrap());
    println!("\n✨ Best Quality / Low Variance (Top 5):");
    for (i, result) in by_quality.iter().take(5).enumerate() {
        println!(
            "   {}. {} - {:.4} variance, {:.1}% speedup, {} divergences, {} micro-corrections",
            i + 1,
            result.config.name,
            result.loss_variance,
            result.backward_reduction_pct,
            result.divergence_count,
            result.micro_corrections_count
        );
    }

    // Best effective speedup (quality-adjusted)
    let mut by_effective = all_results.clone();
    by_effective.sort_by(|a, b| b.effective_speedup.partial_cmp(&a.effective_speedup).unwrap());
    println!("\n⚡ Best Effective Speedup / Quality-Adjusted (Top 5):");
    for (i, result) in by_effective.iter().take(5).enumerate() {
        println!(
            "   {}. {} - {:.2}× effective, {:.1}% raw speedup, {:.4} variance, {} micro-corrections",
            i + 1,
            result.config.name,
            result.effective_speedup,
            result.backward_reduction_pct,
            result.loss_variance,
            result.micro_corrections_count
        );
    }

    // === ANALYSIS SECTIONS ===
    println!("\n");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("💡 KEY INSIGHTS & ANALYSIS");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // 1. Micro-correction impact
    println!("📈 Q1: Do micro-corrections enable longer horizons?\n");

    let baseline_long = all_results
        .iter()
        .filter(|r| r.config.correction_interval == 0 && r.config.max_predict_steps >= 75)
        .collect::<Vec<_>>();
    let corrected_long = all_results
        .iter()
        .filter(|r| r.config.correction_interval > 0 && r.config.max_predict_steps >= 75)
        .collect::<Vec<_>>();

    if !baseline_long.is_empty() && !corrected_long.is_empty() {
        let baseline_avg_variance =
            baseline_long.iter().map(|r| r.loss_variance).sum::<f32>() / baseline_long.len() as f32;
        let corrected_avg_variance = corrected_long.iter().map(|r| r.loss_variance).sum::<f32>()
            / corrected_long.len() as f32;

        println!("   Long horizons (H≥75) WITHOUT micro-corrections:");
        println!("     - Avg variance: {:.4}", baseline_avg_variance);
        println!("     - Configs tested: {}", baseline_long.len());

        println!("\n   Long horizons (H≥75) WITH micro-corrections:");
        println!("     - Avg variance: {:.4}", corrected_avg_variance);
        println!("     - Configs tested: {}", corrected_long.len());
        println!(
            "\n   → Variance reduction: {:.1}%",
            ((baseline_avg_variance - corrected_avg_variance) / baseline_avg_variance * 100.0)
        );
    }

    // 2. Optimal correction interval
    println!("\n📊 Q2: What's the optimal correction interval?\n");

    for &interval in &correction_intervals {
        if interval == 0 {
            continue;
        }
        let interval_results: Vec<_> = all_results
            .iter()
            .filter(|r| r.config.correction_interval == interval)
            .collect();

        let avg_effective = interval_results.iter().map(|r| r.effective_speedup).sum::<f32>()
            / interval_results.len() as f32;
        let avg_variance =
            interval_results.iter().map(|r| r.loss_variance).sum::<f32>() / interval_results.len() as f32;
        let total_corrections: usize = interval_results.iter().map(|r| r.micro_corrections_count).sum();

        println!("   Interval = {} steps:", interval);
        println!("     - Avg effective speedup: {:.2}×", avg_effective);
        println!("     - Avg variance: {:.4}", avg_variance);
        println!("     - Total corrections applied: {}", total_corrections);
    }

    // 3. Confidence threshold interaction
    println!("\n🎯 Q3: How does confidence threshold interact with micro-corrections?\n");

    for &confidence in &confidence_thresholds {
        let no_correction: Vec<_> = all_results
            .iter()
            .filter(|r| r.config.confidence_threshold == confidence && r.config.correction_interval == 0)
            .collect();
        let with_correction: Vec<_> = all_results
            .iter()
            .filter(|r| r.config.confidence_threshold == confidence && r.config.correction_interval > 0)
            .collect();

        let no_corr_speedup =
            no_correction.iter().map(|r| r.backward_reduction_pct).sum::<f32>() / no_correction.len() as f32;
        let with_corr_speedup = with_correction.iter().map(|r| r.backward_reduction_pct).sum::<f32>()
            / with_correction.len() as f32;

        println!("   Confidence = {:.2}:", confidence);
        println!("     - Without corrections: {:.1}% speedup", no_corr_speedup);
        println!("     - With corrections: {:.1}% speedup", with_corr_speedup);
        println!(
            "     → Improvement: +{:.1}%\n",
            with_corr_speedup - no_corr_speedup
        );
    }

    // 4. Pareto frontier
    println!("📐 Q4: What's the new Pareto frontier (speedup vs quality)?\n");

    let mut pareto_candidates = all_results.clone();
    pareto_candidates.sort_by(|a, b| {
        // Sort by speedup descending, then variance ascending
        match b
            .backward_reduction_pct
            .partial_cmp(&a.backward_reduction_pct)
            .unwrap()
        {
            std::cmp::Ordering::Equal => a.loss_variance.partial_cmp(&b.loss_variance).unwrap(),
            other => other,
        }
    });

    // Find true Pareto frontier (non-dominated points)
    let mut pareto_frontier = Vec::new();
    let mut best_variance = f32::INFINITY;

    for result in &pareto_candidates {
        if result.loss_variance < best_variance {
            pareto_frontier.push(result);
            best_variance = result.loss_variance;
        }
    }

    println!("   Pareto-optimal configurations (max speedup for each quality level):");
    for (i, result) in pareto_frontier.iter().take(8).enumerate() {
        println!(
            "   {}. {} - {:.1}% speedup, {:.4} variance, I={}, H={}, C={:.2}",
            i + 1,
            result.config.name,
            result.backward_reduction_pct,
            result.loss_variance,
            result.config.correction_interval,
            result.config.max_predict_steps,
            result.config.confidence_threshold
        );
    }

    // === RECOMMENDATIONS ===
    println!("\n");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("🎯 PRODUCTION RECOMMENDATIONS");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    if let Some(best_effective) = by_effective.first() {
        println!("⭐ RECOMMENDED: Best Overall (Quality-Adjusted Speedup)");
        println!();
        println!("   HybridTrainerConfig::builder()");
        println!(
            "       .max_predict_steps({})  // Prediction horizon",
            best_effective.config.max_predict_steps
        );
        println!(
            "       .confidence_threshold({:.2})",
            best_effective.config.confidence_threshold
        );
        println!(
            "       .correction_interval({})  // Micro-corrections{}",
            best_effective.config.correction_interval,
            if best_effective.config.correction_interval > 0 {
                " enabled"
            } else {
                " disabled"
            }
        );
        println!("       .divergence_config(DivergenceConfig {{");
        println!("           loss_sigma_threshold: 2.5,");
        println!("           check_interval_steps: 5,");
        println!("           ..Default::default()");
        println!("       }})");
        println!("       .build()");
        println!();
        println!("   Expected Performance:");
        println!(
            "     - Backward reduction: {:.1}%",
            best_effective.backward_reduction_pct
        );
        println!(
            "     - Loss variance: {:.4} (σ={:.4})",
            best_effective.loss_variance, best_effective.loss_std_dev
        );
        println!("     - Quality score: {:.3}", best_effective.quality_score);
        println!(
            "     - Effective speedup: {:.2}×",
            best_effective.effective_speedup
        );
        println!(
            "     - Divergences: {} ({:.1}%)",
            best_effective.divergence_count,
            best_effective.divergence_count as f32 / total_steps as f32 * 100.0
        );
        println!(
            "     - Micro-corrections: {}",
            best_effective.micro_corrections_count
        );
    }

    if let Some(best_speedup) = by_speedup.first() {
        println!("\n🚀 AGGRESSIVE: Maximum Speedup (Lower Quality)");
        println!();
        println!("   HybridTrainerConfig::builder()");
        println!(
            "       .max_predict_steps({})  // Longest horizon",
            best_speedup.config.max_predict_steps
        );
        println!(
            "       .confidence_threshold({:.2})",
            best_speedup.config.confidence_threshold
        );
        println!(
            "       .correction_interval({})",
            best_speedup.config.correction_interval
        );
        println!("       .build()");
        println!();
        println!(
            "   Trade-off: {:.1}% speedup but {:.4} variance",
            best_speedup.backward_reduction_pct, best_speedup.loss_variance
        );
    }

    if let Some(best_qual) = by_quality.first() {
        println!("\n✨ CONSERVATIVE: Maximum Quality (Lower Speedup)");
        println!();
        println!("   HybridTrainerConfig::builder()");
        println!(
            "       .max_predict_steps({})  // Shorter horizon",
            best_qual.config.max_predict_steps
        );
        println!(
            "       .confidence_threshold({:.2})",
            best_qual.config.confidence_threshold
        );
        println!(
            "       .correction_interval({})",
            best_qual.config.correction_interval
        );
        println!("       .build()");
        println!();
        println!(
            "   Trade-off: Only {:.1}% speedup but {:.4} variance",
            best_qual.backward_reduction_pct, best_qual.loss_variance
        );
    }

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("✅ Comprehensive parameter sweep complete!");
    println!("   Total runtime: {}s", total_runtime);
    println!("   Configurations tested: {}", total_experiments);
    println!("   Data collected for production tuning and research analysis");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
}
