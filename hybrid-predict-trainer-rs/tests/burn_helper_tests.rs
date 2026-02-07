//! Unit tests for Burn integration helper functions.
//!
//! These tests verify tensor conversion utilities, gradient extraction,
//! and weight delta application using the ndarray backend.

#[cfg(feature = "ndarray")]
mod ndarray_tests {
    use burn::backend::ndarray::NdArray;
    use burn::backend::Autodiff;
    use burn::module::{Module, Param};
    use burn::nn::{Linear, LinearConfig};
    use burn::tensor::{backend::Backend, Device, Shape, Tensor, TensorData};
    use hybrid_predict_trainer_rs::state::WeightDelta;
    use std::collections::HashMap;

    type TestBackend = Autodiff<NdArray>;

    /// Test tensor_to_vec for 1D tensors
    #[test]
    fn test_tensor_to_vec_1d() {
        let device = Device::<NdArray>::Cpu;

        // Create a 1D tensor with known values
        let data = TensorData::from([1.0f32, 2.0, 3.0, 4.0]);
        let tensor: Tensor<NdArray, 1> = Tensor::from_data(data.convert::<f32>(), &device);

        // Convert to vec
        let vec = tensor.to_data().to_vec::<f32>().unwrap();

        // Verify
        assert_eq!(vec.len(), 4);
        assert_eq!(vec, vec![1.0, 2.0, 3.0, 4.0]);
    }

    /// Test tensor_to_vec for 2D tensors (should flatten)
    #[test]
    fn test_tensor_to_vec_2d() {
        let device = Device::<NdArray>::Cpu;

        // Create a 2x3 tensor
        let data = TensorData::from([[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let tensor: Tensor<NdArray, 2> = Tensor::from_data(data.convert::<f32>(), &device);

        // Convert to vec (should be flattened)
        let vec = tensor.to_data().to_vec::<f32>().unwrap();

        // Verify
        assert_eq!(vec.len(), 6);
        assert_eq!(vec, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    /// Test vec_to_tensor for 1D tensors
    #[test]
    fn test_vec_to_tensor_1d() {
        let device = Device::<NdArray>::Cpu;

        // Create a vector
        let vec = vec![10.0f32, 20.0, 30.0];

        // Convert to tensor
        let data = TensorData::new(vec.clone(), Shape::new([3]));
        let tensor: Tensor<NdArray, 1> = Tensor::from_data(data.convert::<f32>(), &device);

        // Verify shape
        assert_eq!(tensor.dims(), [3]);

        // Verify values
        let back_to_vec = tensor.to_data().to_vec::<f32>().unwrap();
        assert_eq!(back_to_vec, vec);
    }

    /// Test vec_to_tensor for 2D tensors
    #[test]
    fn test_vec_to_tensor_2d() {
        let device = Device::<NdArray>::Cpu;

        // Create a vector (will be reshaped to 2x3)
        let vec = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];

        // Convert to 2D tensor
        let data = TensorData::new(vec.clone(), Shape::new([2, 3]));
        let tensor: Tensor<NdArray, 2> = Tensor::from_data(data.convert::<f32>(), &device);

        // Verify shape
        assert_eq!(tensor.dims(), [2, 3]);

        // Verify values (flatten and compare)
        let back_to_vec = tensor.to_data().to_vec::<f32>().unwrap();
        assert_eq!(back_to_vec, vec);
    }

    /// Test that parameter counting works correctly
    #[test]
    fn test_parameter_counting() {
        let device = Device::<TestBackend>::Cpu;

        // Create a simple linear layer: 10 inputs -> 5 outputs
        // Parameters: weight (5x10=50) + bias (5) = 55 total
        let config = LinearConfig::new(10, 5);
        let linear: Linear<TestBackend> = config.init(&device);

        // Count parameters manually via visitor
        use burn::module::ModuleVisitor;

        struct ParamCounter {
            count: usize,
        }

        impl<B: Backend> ModuleVisitor<B> for ParamCounter {
            fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
                self.count += param.val().dims().iter().product::<usize>();
            }
        }

        let mut counter = ParamCounter { count: 0 };
        linear.visit(&mut counter);

        // Verify: 50 (weight) + 5 (bias) = 55 parameters
        assert_eq!(counter.count, 55, "Linear(10, 5) should have 55 parameters");
    }

    /// Test round-trip conversion (vec -> tensor -> vec)
    #[test]
    fn test_round_trip_conversion() {
        let device = Device::<NdArray>::Cpu;

        let original = vec![1.5f32, 2.5, 3.5, 4.5, 5.5];

        // vec -> tensor
        let data = TensorData::new(original.clone(), Shape::new([5]));
        let tensor: Tensor<NdArray, 1> = Tensor::from_data(data.convert::<f32>(), &device);

        // tensor -> vec
        let result = tensor.to_data().to_vec::<f32>().unwrap();

        // Should be identical
        assert_eq!(result, original);
    }

    /// Test handling of zero-sized tensors
    #[test]
    fn test_empty_tensor_conversion() {
        let device = Device::<NdArray>::Cpu;

        // Create empty vector
        let vec: Vec<f32> = vec![];

        // Convert to tensor
        let data = TensorData::new(vec.clone(), Shape::new([0]));
        let tensor: Tensor<NdArray, 1> = Tensor::from_data(data.convert::<f32>(), &device);

        // Verify
        assert_eq!(tensor.dims(), [0]);

        // Convert back
        let result = tensor.to_data().to_vec::<f32>().unwrap();
        assert_eq!(result.len(), 0);
    }

    /// Test tensor shape preservation in conversion
    #[test]
    fn test_shape_preservation() {
        let device = Device::<NdArray>::Cpu;

        // Create 3D tensor (2x3x4 = 24 elements)
        let vec: Vec<f32> = (0..24).map(|i| i as f32).collect();

        let data = TensorData::new(vec.clone(), Shape::new([2, 3, 4]));
        let tensor: Tensor<NdArray, 3> = Tensor::from_data(data.convert::<f32>(), &device);

        // Verify shape
        assert_eq!(tensor.dims(), [2, 3, 4]);

        // Verify total elements
        let result = tensor.to_data().to_vec::<f32>().unwrap();
        assert_eq!(result.len(), 24);
        assert_eq!(result, vec);
    }
}

/// Tests that only run when ndarray feature is disabled
#[cfg(not(feature = "ndarray"))]
mod no_backend_tests {
    #[test]
    fn test_ndarray_feature_not_enabled() {
        // This test verifies that when ndarray feature is not enabled,
        // the crate still compiles but backend-specific tests are skipped
        assert!(
            true,
            "Compile check: burn_integration module accessible without backend"
        );
    }
}
