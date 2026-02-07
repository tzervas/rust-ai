//! Burn MLP MNIST training example with HybridTrainer.
//!
//! This example demonstrates how to use `HybridTrainer` with a real Burn model
//! for MNIST-style digit classification using a simple multi-layer perceptron.
//!
//! # Running
//!
//! ```bash
//! # With CPU backend (ndarray)
//! cargo run --example burn_mlp_mnist --features autodiff,ndarray
//!
//! # With GPU backend (CUDA)
//! cargo run --example burn_mlp_mnist --features autodiff,cuda
//! ```
//!
//! Note: This example uses synthetic data for demonstration purposes.
//! For real MNIST training, use burn::data::dataset::vision::MnistDataset.

use burn::backend::ndarray::NdArray;
use burn::backend::Autodiff;
use burn::module::Module;
use burn::nn::{loss::CrossEntropyLossConfig, Linear, LinearConfig};
use burn::optim::AdamConfig;
use burn::tensor::{backend::Backend, Device, Shape, Tensor, TensorData};

use hybrid_predict_trainer_rs::burn_integration::{
    BurnBatch, BurnForwardFn, BurnModelWrapper, BurnOptimizerWrapper,
};
use hybrid_predict_trainer_rs::{HybridTrainer, HybridTrainerConfig};

use rand::Rng;

// Use NdArray backend with autodiff for CPU training
type MyBackend = Autodiff<NdArray>;

/// Simple MLP model for MNIST classification.
///
/// Architecture: 784 (input) -> 128 (hidden) -> 10 (output)
#[derive(Module, Debug)]
struct SimpleMLP<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
}

impl<B: Backend> SimpleMLP<B> {
    /// Creates a new SimpleMLP model.
    pub fn new(device: &Device<B>) -> Self {
        let fc1 = LinearConfig::new(784, 128).init(device);
        let fc2 = LinearConfig::new(128, 10).init(device);

        Self { fc1, fc2 }
    }

    /// Forward pass.
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.fc1.forward(input);
        let x = burn::tensor::activation::relu(x);
        self.fc2.forward(x)
    }
}

/// MNIST batch structure.
#[derive(Debug, Clone)]
struct MnistBatch<B: Backend> {
    /// Input images flattened to [batch_size, 784]
    images: Tensor<B, 2>,
    /// Target labels [batch_size]
    labels: Tensor<B, 1, burn::tensor::Int>,
}

/// Forward function for MNIST training.
struct MnistForward;

impl BurnForwardFn<MyBackend, SimpleMLP<MyBackend>, MnistBatch<MyBackend>> for MnistForward {
    fn forward(
        &self,
        model: SimpleMLP<MyBackend>,
        batch: &BurnBatch<MyBackend, MnistBatch<MyBackend>>,
    ) -> (SimpleMLP<MyBackend>, Tensor<MyBackend, 1>) {
        // Forward pass
        let logits = model.forward(batch.data.images.clone());

        // Compute cross-entropy loss
        let loss = CrossEntropyLossConfig::new()
            .init(&logits.device())
            .forward(logits, batch.data.labels.clone());

        (model, loss)
    }
}

/// Generates synthetic MNIST-like data for demonstration.
///
/// In a real application, use burn::data::dataset::vision::MnistDataset.
fn generate_synthetic_batch(batch_size: usize, device: &Device<MyBackend>) -> MnistBatch<MyBackend> {
    let mut rng = rand::rng();

    // Generate random images (784 pixels each, values 0-1)
    let mut image_data = vec![];
    for _ in 0..batch_size {
        for _ in 0..784 {
            image_data.push(rng.random::<f32>());
        }
    }

    // Generate random labels (0-9)
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

fn main() {
    println!("🚀 Burn MLP MNIST Example with HybridTrainer");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    // Initialize device
    let device = Device::<MyBackend>::Cpu;
    println!("✓ Device: CPU (NdArray backend with Autodiff)");

    // Create model
    let model = SimpleMLP::<MyBackend>::new(&device);
    println!("✓ Model: SimpleMLP (784 → 128 → 10)");

    // Wrap model
    let forward_fn = MnistForward;
    let wrapped_model = BurnModelWrapper::new(model, forward_fn, device.clone());
    println!("✓ Model wrapped with BurnModelWrapper");

    // Create optimizer
    let optim_config = AdamConfig::new();
    let optimizer = optim_config.init();
    let wrapped_optimizer = BurnOptimizerWrapper::new(optimizer, 0.001);
    println!("✓ Optimizer: Adam (lr=0.001)");

    // Create HybridTrainer configuration
    let config = HybridTrainerConfig::builder()
        .warmup_steps(5)
        .full_steps(3)
        .max_predict_steps(10)
        .confidence_threshold(0.60)  // Lowered to achievable threshold after bugfixes
        .collect_metrics(true)  // Enable metrics collection for speedup tracking
        .build();
    println!("✓ HybridTrainer config: warmup=5, full=3, predict≤10");

    // Create trainer
    let mut trainer =
        HybridTrainer::new(wrapped_model, wrapped_optimizer, config).expect("Failed to create trainer");
    println!("✓ HybridTrainer created");
    println!();

    // Training loop
    let batch_size = 32;
    let total_steps = 50;

    println!("Starting training ({} steps)...", total_steps);
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    for step in 0..total_steps {
        // Generate synthetic batch
        let batch_data = generate_synthetic_batch(batch_size, &device);
        let batch = BurnBatch::new(batch_data, batch_size);

        // Training step
        let result = trainer.step(&batch).expect("Training step failed");

        // Log progress
        if step % 5 == 0 {
            let phase_str = format!("{:?}", result.phase);
            let predicted_str = if result.was_predicted { "✓" } else { "-" };
            let conf_str = format!("{:.2}", result.confidence);

            println!(
                "Step {:3} | {:8} | Loss: {:.4} | Pred: {} | Conf: {}",
                trainer.current_step(),
                phase_str,
                result.loss,
                predicted_str,
                conf_str,
            );
        }
    }

    println!();
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Training complete!");
    println!();

    // Print statistics
    let stats = trainer.statistics();
    println!("📊 Training Statistics");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Total steps:           {}", stats.total_steps);
    println!("Backward reduction:    {:.1}%", stats.backward_reduction_pct);
    println!("Average confidence:    {:.3}", stats.avg_confidence);
    println!("Divergence events:     {}", stats.divergence_events);
    println!();

    println!("✅ Example completed successfully!");
    println!();
    println!("💡 Next steps:");
    println!("   - Try with real MNIST: use burn::data::dataset::vision::MnistDataset");
    println!("   - Experiment with config: warmup_steps, full_steps, max_predict_steps");
    println!("   - Monitor training: watch how phases transition during training");
    println!("   - GPU acceleration: run with --features autodiff,cuda");
}
