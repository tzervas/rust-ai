//! Test to verify that autodiff backends work after Sync limitation fix.
//!
//! This test verifies that BurnModelWrapper can be used with autodiff backends
//! now that we've relaxed the Sync requirement and use Mutex instead of RwLock.

#[cfg(feature = "ndarray")]
mod autodiff_tests {
    use burn::backend::ndarray::NdArray;
    use burn::backend::Autodiff;
    use burn::module::Module;
    use burn::nn::{loss::CrossEntropyLossConfig, Linear, LinearConfig};
    use burn::tensor::{backend::Backend, Device, Tensor};
    use hybrid_predict_trainer_rs::burn_integration::{BurnBatch, BurnForwardFn, BurnModelWrapper};
    use hybrid_predict_trainer_rs::Model;

    type TestBackend = Autodiff<NdArray>;

    /// Simple MLP model for testing.
    #[derive(Module, Debug)]
    struct SimpleMLP<B: Backend> {
        fc1: Linear<B>,
        fc2: Linear<B>,
    }

    impl<B: Backend> SimpleMLP<B> {
        pub fn new(
            input_size: usize,
            hidden_size: usize,
            output_size: usize,
            device: &Device<B>,
        ) -> Self {
            let fc1 = LinearConfig::new(input_size, hidden_size).init(device);
            let fc2 = LinearConfig::new(hidden_size, output_size).init(device);
            Self { fc1, fc2 }
        }

        pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
            let x = self.fc1.forward(input);
            let x = burn::tensor::activation::relu(x);
            self.fc2.forward(x)
        }
    }

    /// Batch structure for SimpleMLP
    #[derive(Debug, Clone)]
    struct MLPBatch<B: Backend> {
        inputs: Tensor<B, 2>,
        targets: Tensor<B, 1, burn::tensor::Int>,
    }

    /// Forward function implementing BurnForwardFn trait
    struct MLPForward;

    impl BurnForwardFn<TestBackend, SimpleMLP<TestBackend>, MLPBatch<TestBackend>> for MLPForward {
        fn forward(
            &self,
            model: SimpleMLP<TestBackend>,
            batch: &BurnBatch<TestBackend, MLPBatch<TestBackend>>,
        ) -> (SimpleMLP<TestBackend>, Tensor<TestBackend, 1>) {
            // Forward pass
            let logits = model.forward(batch.data.inputs.clone());

            // Compute cross-entropy loss
            let loss = CrossEntropyLossConfig::new()
                .init(&logits.device())
                .forward(logits, batch.data.targets.clone());

            (model, loss)
        }
    }

    #[test]
    fn test_autodiff_forward_with_sync_fix() {
        let device = Device::<TestBackend>::Cpu;

        // Create model: 4 inputs → 8 hidden → 2 outputs
        let model = SimpleMLP::new(4, 8, 2, &device);
        let forward_fn = MLPForward;

        // Create wrapper - this should work now that we use Mutex instead of RwLock
        let mut wrapper = BurnModelWrapper::new(model, forward_fn, device.clone());

        // Create dummy batch
        let inputs = Tensor::from_data(
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
            ],
            &device,
        );
        let targets = Tensor::from_data([0i64, 1, 0], &device);

        let batch_data = MLPBatch { inputs, targets };
        let batch = BurnBatch::new(batch_data, 3);

        // Test forward pass
        let loss = wrapper.forward(&batch).expect("Forward should succeed");

        // Loss should be positive and finite
        assert!(loss > 0.0, "Loss should be positive");
        assert!(loss.is_finite(), "Loss should be finite");
    }

    #[test]
    fn test_autodiff_backward_with_sync_fix() {
        let device = Device::<TestBackend>::Cpu;

        // Create model
        let model = SimpleMLP::new(4, 8, 2, &device);
        let forward_fn = MLPForward;
        let mut wrapper = BurnModelWrapper::new(model, forward_fn, device.clone());

        // Create batch
        let inputs = Tensor::from_data([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], &device);
        let targets = Tensor::from_data([0i64, 1], &device);
        let batch_data = MLPBatch { inputs, targets };
        let batch = BurnBatch::new(batch_data, 2);

        // Forward pass
        let _loss = wrapper.forward(&batch).expect("Forward should succeed");

        // Backward pass - this should work now with Mutex-based storage
        let grad_info = wrapper.backward().expect("Backward should succeed");

        // Verify gradient info
        assert!(
            grad_info.gradient_norm > 0.0,
            "Gradient norm should be positive"
        );
        assert!(
            grad_info.gradient_norm.is_finite(),
            "Gradient norm should be finite"
        );
    }

    #[test]
    fn test_autodiff_model_is_send() {
        // This test verifies that the wrapper is Send (can be moved between threads)
        // even though it's !Sync (can't be shared between threads)
        let device = Device::<TestBackend>::Cpu;
        let model = SimpleMLP::new(2, 4, 2, &device);
        let forward_fn = MLPForward;
        let wrapper = BurnModelWrapper::new(model, forward_fn, device);

        // This should compile - wrapper is Send
        let handle = std::thread::spawn(move || {
            // wrapper was moved into this thread
            drop(wrapper);
        });

        handle.join().expect("Thread should complete successfully");
    }
}

#[cfg(not(feature = "ndarray"))]
mod no_backend {
    #[test]
    fn test_requires_backend() {
        assert!(true, "Tests require ndarray backend feature");
    }
}
