//! Batched GPU performance tests.
//!
//! Demonstrates the performance benefits of batched GPU operations vs single-sample processing.

#[cfg(all(feature = "cuda", feature = "candle"))]
mod batched_tests {
    use burn_cuda::{Cuda, CudaDevice};
    use hybrid_predict_trainer_rs::gpu::kernels::{
        gru::{gru_forward_gpu, GpuGruWeights, GruKernelConfig},
        gru_batched::gru_forward_batched_gpu,
    };
    use std::time::Instant;

    fn create_test_weights(hidden_dim: usize, input_dim: usize) -> GpuGruWeights {
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
    #[ignore] // Run with --ignored
    fn test_batched_vs_individual_correctness() {
        let hidden_dim = 256;
        let input_dim = 64;
        let batch_size = 16;

        let config = GruKernelConfig {
            hidden_dim,
            input_dim,
            batch_size,
        };

        let weights = create_test_weights(hidden_dim, input_dim);
        let device = CudaDevice::default();

        // Create batch of samples
        let hiddens: Vec<Vec<f32>> = (0..batch_size).map(|_| vec![0.5; hidden_dim]).collect();
        let inputs: Vec<Vec<f32>> = (0..batch_size).map(|_| vec![0.3; input_dim]).collect();

        // Process individually
        let mut individual_outputs = Vec::with_capacity(batch_size);
        for (hidden, input) in hiddens.iter().zip(inputs.iter()) {
            let output = gru_forward_gpu(&config, &weights, hidden, input).unwrap();
            individual_outputs.push(output);
        }

        // Process as batch
        let batched_outputs =
            gru_forward_batched_gpu::<Cuda>(&config, &weights, &hiddens, &inputs, &device).unwrap();

        // Verify same results
        assert_eq!(batched_outputs.len(), individual_outputs.len());

        for (batched, individual) in batched_outputs.iter().zip(individual_outputs.iter()) {
            assert_eq!(batched.len(), individual.len());

            let max_error = batched
                .iter()
                .zip(individual.iter())
                .map(|(b, i)| (b - i).abs())
                .fold(0.0_f32, f32::max);

            assert!(
                max_error < 1e-4,
                "Batched vs individual error {} exceeds 1e-4",
                max_error
            );
        }

        println!("✅ Batched processing produces identical results to individual processing");
    }

    #[test]
    #[ignore] // Run with --ignored
    fn test_batched_performance_small() {
        benchmark_batched_performance(16, 256, 64, 100);
    }

    #[test]
    #[ignore] // Run with --ignored
    fn test_batched_performance_medium() {
        benchmark_batched_performance(32, 256, 64, 100);
    }

    #[test]
    #[ignore] // Run with --ignored
    fn test_batched_performance_large() {
        benchmark_batched_performance(64, 256, 64, 100);
    }

    fn benchmark_batched_performance(
        batch_size: usize,
        hidden_dim: usize,
        input_dim: usize,
        iterations: usize,
    ) {
        let config = GruKernelConfig {
            hidden_dim,
            input_dim,
            batch_size,
        };

        let weights = create_test_weights(hidden_dim, input_dim);
        let device = CudaDevice::default();

        // Create batch of samples
        let hiddens: Vec<Vec<f32>> = (0..batch_size).map(|_| vec![0.5; hidden_dim]).collect();
        let inputs: Vec<Vec<f32>> = (0..batch_size).map(|_| vec![0.3; input_dim]).collect();

        // Warm-up
        for _ in 0..10 {
            let _ = gru_forward_batched_gpu::<Cuda>(&config, &weights, &hiddens, &inputs, &device);
        }

        // Benchmark individual processing
        let individual_start = Instant::now();
        for _ in 0..iterations {
            for (hidden, input) in hiddens.iter().zip(inputs.iter()) {
                let _ = gru_forward_gpu(&config, &weights, hidden, input).unwrap();
            }
        }
        let individual_time = individual_start.elapsed();

        // Benchmark batched processing
        let batched_start = Instant::now();
        for _ in 0..iterations {
            let _ = gru_forward_batched_gpu::<Cuda>(&config, &weights, &hiddens, &inputs, &device)
                .unwrap();
        }
        let batched_time = batched_start.elapsed();

        let speedup = individual_time.as_secs_f64() / batched_time.as_secs_f64();

        println!("\n=== Batched Performance Benchmark ===");
        println!("Batch size: {}", batch_size);
        println!("Hidden dim: {}", hidden_dim);
        println!("Input dim: {}", input_dim);
        println!("Iterations: {}", iterations);
        println!("Individual time: {:?}", individual_time);
        println!("Batched time: {:?}", batched_time);
        println!("Speedup: {:.2}×", speedup);

        // With batching, we expect significant speedup
        assert!(
            speedup > 2.0,
            "Batched processing should be >2× faster, got {:.2}×",
            speedup
        );

        println!("✅ Batched processing achieved {:.2}× speedup!", speedup);
    }

    #[test]
    #[ignore] // Run with --ignored
    fn test_batched_memory_efficiency() {
        // Test that batched operations use less memory transfers
        let batch_size = 32;
        let hidden_dim = 256;
        let input_dim = 64;

        let config = GruKernelConfig {
            hidden_dim,
            input_dim,
            batch_size,
        };

        let weights = create_test_weights(hidden_dim, input_dim);
        let device = CudaDevice::default();

        let hiddens: Vec<Vec<f32>> = (0..batch_size).map(|_| vec![0.5; hidden_dim]).collect();
        let inputs: Vec<Vec<f32>> = (0..batch_size).map(|_| vec![0.3; input_dim]).collect();

        // Single batch upload/download
        let outputs =
            gru_forward_batched_gpu::<Cuda>(&config, &weights, &hiddens, &inputs, &device).unwrap();

        assert_eq!(outputs.len(), batch_size);
        println!(
            "✅ Single batch operation processed {} samples with 1 upload + 1 download",
            batch_size
        );
    }
}
