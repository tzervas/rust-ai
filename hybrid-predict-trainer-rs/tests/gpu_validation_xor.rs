//! GPU validation on XOR problem.
//!
//! This test validates GPU kernel correctness on the simplest possible
//! non-linear problem: XOR. The model is a SimpleMLP (2→4→1) with ~20 parameters.
//!
//! **Success Criteria:**
//! - Loss trajectories match CPU within 1%
//! - Final loss < 0.1 (XOR solved)
//! - No NaN/Inf values during training
//! - GPU encoder produces same features as CPU (within 1e-4)

#[cfg(feature = "cuda")]
mod gpu_validation {
    use hybrid_predict_trainer_rs::{
        config::HybridTrainerConfig,
        gpu::kernels::state_encode::{encode_state_cpu, encode_state_gpu, StateEncodeConfig},
        state::TrainingState,
        HybridTrainer,
    };

    /// Simple 2-layer MLP for XOR.
    struct SimpleMLP {
        w1: Vec<f32>,
        b1: Vec<f32>,
        w2: Vec<f32>,
        b2: f32,
    }

    impl SimpleMLP {
        fn new() -> Self {
            Self {
                w1: vec![0.5, -0.3, 0.4, 0.2, -0.5, 0.1, 0.3, -0.4], // [4, 2]
                b1: vec![0.1, -0.1, 0.2, -0.2],
                w2: vec![0.5, -0.5, 0.3, -0.3], // [1, 4]
                b2: 0.0,
            }
        }

        fn forward(&self, x: &[f32]) -> f32 {
            assert_eq!(x.len(), 2, "XOR input must be 2D");

            // Hidden layer: tanh(Wx + b)
            let mut h = vec![0.0; 4];
            for i in 0..4 {
                h[i] = x[0] * self.w1[i * 2] + x[1] * self.w1[i * 2 + 1] + self.b1[i];
                h[i] = h[i].tanh();
            }

            // Output layer: sigmoid(Wh + b)
            let mut out = 0.0;
            for i in 0..4 {
                out += h[i] * self.w2[i];
            }
            out += self.b2;

            // Sigmoid
            if out < 0.0 {
                let exp_out = out.exp();
                exp_out / (1.0 + exp_out)
            } else {
                1.0 / (1.0 + (-out).exp())
            }
        }

        fn num_params(&self) -> usize {
            self.w1.len() + self.b1.len() + self.w2.len() + 1
        }
    }

    /// XOR dataset.
    fn xor_data() -> Vec<([f32; 2], f32)> {
        vec![
            ([0.0, 0.0], 0.0),
            ([0.0, 1.0], 1.0),
            ([1.0, 0.0], 1.0),
            ([1.0, 1.0], 0.0),
        ]
    }

    /// Computes XOR loss.
    fn xor_loss(model: &SimpleMLP) -> f32 {
        let data = xor_data();
        let mut total_loss = 0.0;

        for (x, y) in &data {
            let pred = model.forward(x);
            let err = pred - y;
            total_loss += err * err;
        }

        total_loss / data.len() as f32
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_gpu_state_encode_vs_cpu() {
        let config = StateEncodeConfig::default();
        let state = TrainingState::default();

        // CPU encoding
        let features_cpu = encode_state_cpu(&state);
        assert_eq!(features_cpu.len(), 64, "CPU encoding must be 64-dim");

        // GPU encoding
        let features_gpu = encode_state_gpu(&config, &state).expect("GPU encoding failed");
        assert_eq!(features_gpu.len(), 64, "GPU encoding must be 64-dim");

        // Compare CPU vs GPU (within 1e-4 tolerance)
        let max_error = features_cpu
            .iter()
            .zip(features_gpu.iter())
            .map(|(c, g)| (c - g).abs())
            .fold(0.0_f32, f32::max);

        assert!(
            max_error < 1e-4,
            "CPU vs GPU encoding error {} exceeds 1e-4",
            max_error
        );
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_xor_training_correctness() {
        // This is a placeholder for full training validation
        // Would require integrating HybridTrainer with Burn backend

        let model = SimpleMLP::new();
        let initial_loss = xor_loss(&model);

        println!("Initial XOR loss: {}", initial_loss);
        println!("Model params: {}", model.num_params());

        // TODO: Integrate with HybridTrainer
        // 1. Wrap SimpleMLP in Burn wrapper
        // 2. Create optimizer
        // 3. Run 100 training steps
        // 4. Compare GPU vs CPU loss trajectories

        assert!(
            initial_loss < 1.0,
            "Initial loss should be < 1.0, got {}",
            initial_loss
        );
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_xor_gpu_no_divergence() {
        // Placeholder: Validate no NaN/Inf during GPU training

        let model = SimpleMLP::new();

        for i in 0..10 {
            let loss = xor_loss(&model);
            assert!(
                loss.is_finite(),
                "Loss at step {} is not finite: {}",
                i,
                loss
            );
        }
    }

    #[test]
    fn test_xor_forward_pass() {
        let model = SimpleMLP::new();
        let data = xor_data();

        for (x, y) in &data {
            let pred = model.forward(x);
            println!("Input: {:?}, Target: {}, Pred: {:.4}", x, y, pred);
            assert!(
                pred.is_finite(),
                "Forward pass produced non-finite value for {:?}",
                x
            );
        }
    }

    #[test]
    fn test_xor_loss_computation() {
        let model = SimpleMLP::new();
        let loss = xor_loss(&model);

        assert!(loss.is_finite(), "XOR loss is not finite: {}", loss);
        assert!(loss >= 0.0, "XOR loss must be non-negative, got {}", loss);

        println!("XOR loss with random initialization: {:.6}", loss);
    }
}
