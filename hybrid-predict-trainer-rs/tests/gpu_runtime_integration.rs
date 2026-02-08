//! GPU runtime integration tests for CubeCL 0.9.
//!
//! These tests validate the full pipeline: buffer upload → kernel launch → result download.
//!
//! **Status**: Tests are ready but marked `#[ignore]` pending CubeCL 0.9 buffer API clarification.
//!
//! **What's Needed**:
//! 1. Proper `client.create()` API for uploading f32 slices to GPU
//! 2. Correct `Handle` extraction for `ArrayArg::from_raw_parts()`
//! 3. Proper `client.read()` API for downloading f32 results
//!
//! Once the buffer API is clarified, these tests will validate:
//! - Correctness: GPU output matches CPU within 1e-4
//! - Performance: GPU is >10× faster at hidden_dim=256
//! - Robustness: No crashes, proper error handling

#[cfg(feature = "cuda")]
mod gpu_runtime_tests {
    use hybrid_predict_trainer_rs::gpu::kernels::gru::{
        gru_forward_cpu, gru_forward_gpu, GpuGruWeights, GruKernelConfig,
    };
    use std::time::Instant;

    fn create_test_weights(hidden_dim: usize, input_dim: usize) -> GpuGruWeights {
        // Initialize with small random-like values
        let mut w_z = Vec::with_capacity(hidden_dim * input_dim);
        for i in 0..hidden_dim * input_dim {
            w_z.push(0.01 * ((i as f32 * 0.1).sin()));
        }

        let mut u_z = Vec::with_capacity(hidden_dim * hidden_dim);
        for i in 0..hidden_dim * hidden_dim {
            u_z.push(0.01 * ((i as f32 * 0.1).cos()));
        }

        GpuGruWeights {
            w_z: w_z.clone(),
            u_z: u_z.clone(),
            b_z: vec![0.0; hidden_dim],
            w_r: w_z.clone(),
            u_r: u_z.clone(),
            b_r: vec![0.0; hidden_dim],
            w_h: w_z,
            u_h: u_z,
            b_h: vec![0.0; hidden_dim],
            hidden_dim,
            input_dim,
        }
    }

    #[test]
    #[ignore] // Requires CubeCL 0.9 buffer API implementation
    fn test_gru_gpu_vs_cpu_correctness_small() {
        let hidden_dim = 64;
        let input_dim = 64;

        let config = GruKernelConfig {
            hidden_dim,
            input_dim,
            batch_size: 1,
        };

        let weights = create_test_weights(hidden_dim, input_dim);
        let hidden = vec![0.5; hidden_dim];
        let input = vec![0.3; input_dim];

        // CPU reference
        let cpu_output = gru_forward_cpu(&weights, &hidden, &input);

        // GPU implementation
        let gpu_output = gru_forward_gpu(&config, &weights, &hidden, &input)
            .expect("GPU forward pass failed");

        // Validate dimensions
        assert_eq!(gpu_output.len(), hidden_dim);
        assert_eq!(cpu_output.len(), hidden_dim);

        // Validate correctness (max error < 1e-4)
        let max_error = cpu_output
            .iter()
            .zip(gpu_output.iter())
            .map(|(c, g)| (c - g).abs())
            .fold(0.0_f32, f32::max);

        println!("Max CPU vs GPU error: {}", max_error);

        assert!(
            max_error < 1e-4,
            "GPU output differs from CPU by {}",
            max_error
        );
    }

    #[test]
    #[ignore] // Requires CubeCL 0.9 buffer API implementation
    fn test_gru_gpu_vs_cpu_correctness_medium() {
        let hidden_dim = 256;
        let input_dim = 64;

        let config = GruKernelConfig {
            hidden_dim,
            input_dim,
            batch_size: 1,
        };

        let weights = create_test_weights(hidden_dim, input_dim);
        let hidden = vec![0.5; hidden_dim];
        let input = vec![0.3; input_dim];

        // CPU reference
        let cpu_output = gru_forward_cpu(&weights, &hidden, &input);

        // GPU implementation
        let gpu_output = gru_forward_gpu(&config, &weights, &hidden, &input)
            .expect("GPU forward pass failed");

        // Validate correctness
        let max_error = cpu_output
            .iter()
            .zip(gpu_output.iter())
            .map(|(c, g)| (c - g).abs())
            .fold(0.0_f32, f32::max);

        assert!(
            max_error < 1e-4,
            "GPU output differs from CPU by {}",
            max_error
        );
    }

    #[test]
    #[ignore] // Requires CubeCL 0.9 buffer API implementation
    fn test_gru_gpu_vs_cpu_correctness_large() {
        let hidden_dim = 1024;
        let input_dim = 64;

        let config = GruKernelConfig {
            hidden_dim,
            input_dim,
            batch_size: 1,
        };

        let weights = create_test_weights(hidden_dim, input_dim);
        let hidden = vec![0.5; hidden_dim];
        let input = vec![0.3; input_dim];

        // CPU reference
        let cpu_output = gru_forward_cpu(&weights, &hidden, &input);

        // GPU implementation
        let gpu_output = gru_forward_gpu(&config, &weights, &hidden, &input)
            .expect("GPU forward pass failed");

        // Validate correctness
        let max_error = cpu_output
            .iter()
            .zip(gpu_output.iter())
            .map(|(c, g)| (c - g).abs())
            .fold(0.0_f32, f32::max);

        assert!(
            max_error < 1e-4,
            "GPU output differs from CPU by {}",
            max_error
        );
    }

    #[test]
    #[ignore] // Requires CubeCL 0.9 buffer API implementation
    fn test_gru_gpu_performance_target() {
        // Target: >10× speedup at hidden_dim=256
        let hidden_dim = 256;
        let input_dim = 64;
        let iterations = 100;

        let config = GruKernelConfig {
            hidden_dim,
            input_dim,
            batch_size: 1,
        };

        let weights = create_test_weights(hidden_dim, input_dim);
        let hidden = vec![0.5; hidden_dim];
        let input = vec![0.3; input_dim];

        // Warm-up GPU
        for _ in 0..10 {
            let _ = gru_forward_gpu(&config, &weights, &hidden, &input);
        }

        // Benchmark CPU
        let cpu_start = Instant::now();
        for _ in 0..iterations {
            let _ = gru_forward_cpu(&weights, &hidden, &input);
        }
        let cpu_time = cpu_start.elapsed();

        // Benchmark GPU
        let gpu_start = Instant::now();
        for _ in 0..iterations {
            let _ = gru_forward_gpu(&config, &weights, &hidden, &input).unwrap();
        }
        let gpu_time = gpu_start.elapsed();

        let speedup = cpu_time.as_secs_f64() / gpu_time.as_secs_f64();

        println!("CPU time: {:?}", cpu_time);
        println!("GPU time: {:?}", gpu_time);
        println!("Speedup: {:.2}×", speedup);

        // Target: >10× speedup
        assert!(
            speedup > 10.0,
            "GPU speedup {:.2}× is below 10× target",
            speedup
        );
    }

    #[test]
    #[ignore] // Requires CubeCL 0.9 buffer API implementation
    fn test_gru_gpu_deterministic() {
        // GPU should produce same output for same input
        let hidden_dim = 256;
        let input_dim = 64;

        let config = GruKernelConfig {
            hidden_dim,
            input_dim,
            batch_size: 1,
        };

        let weights = create_test_weights(hidden_dim, input_dim);
        let hidden = vec![0.5; hidden_dim];
        let input = vec![0.3; input_dim];

        let output1 = gru_forward_gpu(&config, &weights, &hidden, &input)
            .expect("GPU forward pass 1 failed");

        let output2 = gru_forward_gpu(&config, &weights, &hidden, &input)
            .expect("GPU forward pass 2 failed");

        // Should be exactly equal (deterministic)
        for (a, b) in output1.iter().zip(output2.iter()) {
            assert_eq!(a, b, "GPU output is not deterministic");
        }
    }

    #[test]
    #[ignore] // Requires CubeCL 0.9 buffer API implementation
    fn test_gru_gpu_no_nan_inf() {
        // GPU should never produce NaN or Inf
        let hidden_dim = 256;
        let input_dim = 64;

        let config = GruKernelConfig {
            hidden_dim,
            input_dim,
            batch_size: 1,
        };

        let weights = create_test_weights(hidden_dim, input_dim);

        // Test with various inputs (including edge cases)
        let test_cases = vec![
            (vec![0.0; hidden_dim], vec![0.0; input_dim]),
            (vec![1.0; hidden_dim], vec![1.0; input_dim]),
            (vec![-1.0; hidden_dim], vec![-1.0; input_dim]),
            (vec![100.0; hidden_dim], vec![100.0; input_dim]),
        ];

        for (hidden, input) in test_cases {
            let output = gru_forward_gpu(&config, &weights, &hidden, &input)
                .expect("GPU forward pass failed");

            for val in output {
                assert!(val.is_finite(), "GPU produced non-finite value: {}", val);
            }
        }
    }

    #[test]
    #[ignore] // Requires CubeCL 0.9 buffer API implementation
    fn test_gru_gpu_fallback_over_1024() {
        // hidden_dim > 1024 should either fallback to CPU or return error
        let hidden_dim = 2048;
        let input_dim = 64;

        let config = GruKernelConfig {
            hidden_dim,
            input_dim,
            batch_size: 1,
        };

        // Should return error (config validation)
        let result = config.validate();
        assert!(result.is_err(), "Expected validation error for hidden_dim > 1024");
    }
}
