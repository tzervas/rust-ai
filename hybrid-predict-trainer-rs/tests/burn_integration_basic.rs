//! Basic integration tests for Burn that work around Sync limitations.
//!
//! Note: Burn's autodiff Gradients type is !Sync, which creates challenges
//! for our multi-threaded Model trait. These tests verify core functionality
//! in a single-threaded context.

#[cfg(feature = "ndarray")]
mod basic_tests {
    use burn::backend::ndarray::NdArray;
    use burn::module::Module;
    use burn::nn::{Linear, LinearConfig};
    use burn::tensor::{backend::Backend, Device, Tensor};

    type TestBackend = NdArray;

    /// Simple MLP without autodiff for basic testing
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

    #[test]
    fn test_simple_mlp_creation() {
        let device = Device::<TestBackend>::Cpu;
        let model: SimpleMLP<TestBackend> = SimpleMLP::new(10, 5, 2, &device);

        // Count parameters using ModuleVisitor
        use burn::module::{ModuleVisitor, Param};

        struct ParamCounter {
            count: usize,
        }

        impl<B: Backend> ModuleVisitor<B> for ParamCounter {
            fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
                self.count += param.val().dims().iter().product::<usize>();
            }
        }

        let mut counter = ParamCounter { count: 0 };
        model.visit(&mut counter);

        // fc1: 10*5 + 5 = 55
        // fc2: 5*2 + 2 = 12
        // Total: 67
        assert_eq!(counter.count, 67);
    }

    #[test]
    fn test_simple_mlp_forward_pass() {
        let device = Device::<TestBackend>::Cpu;
        let model: SimpleMLP<TestBackend> = SimpleMLP::new(4, 8, 2, &device);

        // Create input: 3 samples, 4 features
        let input = Tensor::from_data(
            [
                [1.0f32, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
            ],
            &device,
        );

        // Forward pass
        let output = model.forward(input);

        // Output should have shape [3, 2]
        assert_eq!(output.dims(), [3, 2]);

        // Output values should be finite
        let output_data = output.to_data();
        let output_vec = output_data.to_vec::<f32>().unwrap();
        for &val in &output_vec {
            assert!(val.is_finite(), "Output should be finite");
        }
    }

    #[test]
    fn test_module_visitor_pattern() {
        let device = Device::<TestBackend>::Cpu;
        let model: SimpleMLP<TestBackend> = SimpleMLP::new(3, 4, 2, &device);

        use burn::module::{ModuleVisitor, Param};

        struct ParamCollector {
            param_names: Vec<String>,
            param_shapes: Vec<Vec<usize>>,
        }

        impl<B: Backend> ModuleVisitor<B> for ParamCollector {
            fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
                let name = format!("{:?}", param.id);
                let shape = param.val().dims().to_vec();
                self.param_names.push(name);
                self.param_shapes.push(shape);
            }
        }

        let mut collector = ParamCollector {
            param_names: vec![],
            param_shapes: vec![],
        };

        model.visit(&mut collector);

        // Should have 4 parameters (fc1.weight, fc1.bias, fc2.weight, fc2.bias)
        assert_eq!(collector.param_names.len(), 4);
        assert_eq!(collector.param_shapes.len(), 4);
    }

    #[test]
    fn test_module_mapper_pattern() {
        let device = Device::<TestBackend>::Cpu;
        let model: SimpleMLP<TestBackend> = SimpleMLP::new(2, 3, 2, &device);

        use burn::module::{ModuleMapper, Param};

        // Mapper that adds 0.1 to all parameters
        struct DeltaMapper;

        impl<B: Backend> ModuleMapper<B> for DeltaMapper {
            fn map_float<const D: usize>(
                &mut self,
                param: Param<Tensor<B, D>>,
            ) -> Param<Tensor<B, D>> {
                let (id, tensor, mapper) = param.consume();
                let updated = tensor + 0.1;
                Param::from_mapped_value(id, updated, mapper)
            }
        }

        // Get initial parameters
        use burn::module::ModuleVisitor;
        struct ParamExtractor {
            values: Vec<Vec<f32>>,
        }

        impl<B: Backend> ModuleVisitor<B> for ParamExtractor {
            fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
                let vals = param.val().to_data().to_vec::<f32>().unwrap();
                self.values.push(vals);
            }
        }

        let mut extractor = ParamExtractor { values: vec![] };
        model.visit(&mut extractor);
        let initial_values = extractor.values;

        // Apply mapper
        let mut mapper = DeltaMapper;
        let updated_model = model.map(&mut mapper);

        // Get updated parameters
        let mut extractor2 = ParamExtractor { values: vec![] };
        updated_model.visit(&mut extractor2);
        let updated_values = extractor2.values;

        // Verify parameters changed by ~0.1
        assert_eq!(initial_values.len(), updated_values.len());

        for (initial, updated) in initial_values.iter().zip(updated_values.iter()) {
            assert_eq!(initial.len(), updated.len());
            for (&i, &u) in initial.iter().zip(updated.iter()) {
                let diff = (u - i - 0.1).abs();
                assert!(
                    diff < 0.0001,
                    "Parameter should change by 0.1, got diff={}",
                    diff
                );
            }
        }
    }

    #[test]
    fn test_tensor_operations() {
        let device = Device::<TestBackend>::Cpu;

        // Test tensor creation and manipulation
        let tensor1 = Tensor::<TestBackend, 2>::from_data([[1.0f32, 2.0], [3.0, 4.0]], &device);
        let tensor2 = Tensor::from_data([[0.1f32, 0.2], [0.3, 0.4]], &device);

        // Addition
        let sum = tensor1.clone() + tensor2.clone();
        let sum_data = sum.to_data().to_vec::<f32>().unwrap();
        assert!((sum_data[0] - 1.1).abs() < 0.001);
        assert!((sum_data[3] - 4.4).abs() < 0.001);

        // Multiplication
        let product = tensor1 * 2.0;
        let product_data = product.to_data().to_vec::<f32>().unwrap();
        assert!((product_data[0] - 2.0).abs() < 0.001);
        assert!((product_data[3] - 8.0).abs() < 0.001);
    }
}

#[cfg(not(feature = "ndarray"))]
mod no_backend {
    #[test]
    fn test_requires_backend() {
        assert!(true, "Tests require ndarray backend feature");
    }
}
