//! Experiment to compare different confidence/sigma configurations.
//!
//! This example trains the same model with 3 different configurations
//! to demonstrate the tradeoff between speedup and quality.
//!
//! # Running
//!
//! ```bash
//! cargo run --example confidence_tuning_experiment --features autodiff,ndarray --release
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

type MyBackend = Autodiff<NdArray>;

/// Simple MLP for testing
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

struct ExperimentResult {
    name: String,
    final_loss: f32,
    avg_confidence: f32,
    divergence_count: usize,
    backward_reduction_pct: f32,
}

fn run_experiment(
    name: &str,
    config: HybridTrainerConfig,
    steps: usize,
) -> ExperimentResult {
    println!("\n🧪 Running experiment: {}", name);
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let device = Device::<MyBackend>::Cpu;
    let model = SimpleMLP::<MyBackend>::new(&device);
    let forward_fn = MnistForward;
    let wrapped_model = BurnModelWrapper::new(model, forward_fn, device.clone());

    let optimizer = AdamConfig::new().init();
    let wrapped_optimizer = BurnOptimizerWrapper::new(optimizer, 0.001);

    let mut trainer =
        HybridTrainer::new(wrapped_model, wrapped_optimizer, config).expect("Failed to create trainer");

    let mut last_loss = 0.0;
    for step in 0..steps {
        let batch = BurnBatch::new(generate_batch(32, &device), 32);
        let result = trainer.step(&batch).expect("Step failed");
        last_loss = result.loss;

        if step % 20 == 0 {
            println!("  Step {:3}: Loss {:.4}", step, result.loss);
        }
    }

    let stats = trainer.statistics();

    println!("  ✓ Complete - Final loss: {:.4}", last_loss);

    ExperimentResult {
        name: name.to_string(),
        final_loss: last_loss,
        avg_confidence: stats.avg_confidence,
        divergence_count: stats.divergence_events,
        backward_reduction_pct: stats.backward_reduction_pct,
    }
}

fn main() {
    println!("🔬 Confidence & Sigma Tuning Experiment");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Testing 3 configurations over 100 training steps each\n");

    let steps = 100;

    // Configuration 1: Aggressive (default settings)
    let config1 = HybridTrainerConfig::builder()
        .warmup_steps(5)
        .full_steps(3)
        .max_predict_steps(20)
        .confidence_threshold(0.85)
        .divergence_config(DivergenceConfig {
            loss_sigma_threshold: 3.0,
            check_interval_steps: 10,
            ..Default::default()
        })
        .build();

    // Configuration 2: Balanced (recommended)
    let config2 = HybridTrainerConfig::builder()
        .warmup_steps(5)
        .full_steps(3)
        .max_predict_steps(20)
        .confidence_threshold(0.90)
        .divergence_config(DivergenceConfig {
            loss_sigma_threshold: 2.5,
            check_interval_steps: 5,
            ..Default::default()
        })
        .build();

    // Configuration 3: Conservative (high quality)
    let config3 = HybridTrainerConfig::builder()
        .warmup_steps(5)
        .full_steps(3)
        .max_predict_steps(15)
        .confidence_threshold(0.92)
        .divergence_config(DivergenceConfig {
            loss_sigma_threshold: 2.0,
            check_interval_steps: 3,
            ..Default::default()
        })
        .build();

    let result1 = run_experiment("Aggressive (Default)", config1, steps);
    let result2 = run_experiment("Balanced (Recommended)", config2, steps);
    let result3 = run_experiment("Conservative (High Quality)", config3, steps);

    // Print comparison table
    println!("\n");
    println!("📊 RESULTS COMPARISON");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("{:<25} | {:>12} | {:>12} | {:>10} | {:>12}",
             "Configuration", "Final Loss", "Avg Conf", "Divergences", "Speedup %");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    for result in &[result1, result2, result3] {
        println!("{:<25} | {:>12.4} | {:>12.3} | {:>10} | {:>11.1}%",
                 result.name,
                 result.final_loss,
                 result.avg_confidence,
                 result.divergence_count,
                 result.backward_reduction_pct);
    }

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("\n💡 Interpretation:");
    println!("   - Lower final loss = better convergence");
    println!("   - Higher avg confidence = more reliable predictions");
    println!("   - Fewer divergences = smoother training");
    println!("   - Lower speedup % = more conservative (but higher quality)");
    println!("\n✅ Recommended: Use 'Balanced' config for production");
}
