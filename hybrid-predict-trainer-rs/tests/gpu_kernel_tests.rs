// SPDX-License-Identifier: MIT
// Copyright 2026 Tyler Zervas

//! GPU kernel correctness tests.
//!
//! These tests validate GPU kernel implementations against CPU reference
//! implementations. All tests are gated behind the `cuda` feature and
//! marked with `#[ignore]` to prevent failures on CI without GPU.
//!
//! # Running GPU Tests
//!
//! ```bash
//! cargo test --features cuda -- --ignored
//! ```

#![cfg(feature = "cuda")]

use hybrid_predict_trainer_rs::gpu::kernels::gru::{
    gru_forward_cpu, gru_forward_gpu, GpuGruWeights, GruKernelConfig,
};
use hybrid_predict_trainer_rs::gpu::GpuClient;

/// Creates test GRU weights with random values.
fn create_random_weights(hidden_dim: usize, input_dim: usize, seed: u64) -> GpuGruWeights {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    let mut rng = StdRng::seed_from_u64(seed);

    let gen_vec = |size: usize| -> Vec<f32> {
        (0..size).map(|_| rng.gen_range(-0.5..0.5)).collect()
    };

    GpuGruWeights {
        w_z: gen_vec(hidden_dim * input_dim),
        u_z: gen_vec(hidden_dim * hidden_dim),
        b_z: gen_vec(hidden_dim),
        w_r: gen_vec(hidden_dim * input_dim),
        u_r: gen_vec(hidden_dim * hidden_dim),
        b_r: gen_vec(hidden_dim),
        w_h: gen_vec(hidden_dim * input_dim),
        u_h: gen_vec(hidden_dim * hidden_dim),
        b_h: gen_vec(hidden_dim),
        hidden_dim,
        input_dim,
    }
}

/// Creates random input vectors.
fn create_random_vectors(size: usize, seed: u64) -> Vec<f32> {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    let mut rng = StdRng::seed_from_u64(seed);
    (0..size).map(|_| rng.gen_range(-1.0..1.0)).collect()
}

#[test]
#[ignore] // Requires CUDA-enabled GPU
fn test_gpu_infrastructure_smoke() {
    // Test basic GPU client creation
    let client = GpuClient::new(0);
    assert!(client.is_ok(), "Failed to create GPU client");

    let client = client.unwrap();

    // Test buffer allocation
    let test_data = vec![1.0, 2.0, 3.0, 4.0];
    let handle = client.create_buffer(&test_data);

    // Test buffer read
    let read_data = client.read_buffer(&handle, test_data.len());
    for (original, read) in test_data.iter().zip(read_data.iter()) {
        assert!((original - read).abs() < 1e-5, "Buffer roundtrip failed");
    }

    // Test synchronization
    client.sync();
}

#[test]
#[ignore]
fn test_gru_forward_gpu_vs_cpu_small() {
    let hidden_dim = 64;
    let input_dim = 32;

    let weights = create_random_weights(hidden_dim, input_dim, 42);
    let hidden = create_random_vectors(hidden_dim, 123);
    let input = create_random_vectors(input_dim, 456);

    // CPU reference
    let cpu_output = gru_forward_cpu(&weights, &hidden, &input);

    // GPU computation
    let config = GruKernelConfig {
        hidden_dim,
        input_dim,
        batch_size: 1,
    };
    let gpu_output = gru_forward_gpu(&config, &weights, &hidden, &input)
        .expect("GPU forward failed");

    // Compare outputs
    assert_eq!(cpu_output.len(), gpu_output.len());
    let max_error = cpu_output
        .iter()
        .zip(gpu_output.iter())
        .map(|(c, g)| (c - g).abs())
        .fold(0.0, f32::max);

    assert!(
        max_error < 1e-4,
        "GPU output differs from CPU: max error = {}",
        max_error
    );
}

#[test]
#[ignore]
fn test_gru_forward_gpu_vs_cpu_large() {
    let hidden_dim = 256;
    let input_dim = 64;

    let weights = create_random_weights(hidden_dim, input_dim, 42);
    let hidden = create_random_vectors(hidden_dim, 123);
    let input = create_random_vectors(input_dim, 456);

    let cpu_output = gru_forward_cpu(&weights, &hidden, &input);

    let config = GruKernelConfig {
        hidden_dim,
        input_dim,
        batch_size: 1,
    };
    let gpu_output = gru_forward_gpu(&config, &weights, &hidden, &input)
        .expect("GPU forward failed");

    assert_eq!(cpu_output.len(), gpu_output.len());
    let max_error = cpu_output
        .iter()
        .zip(gpu_output.iter())
        .map(|(c, g)| (c - g).abs())
        .fold(0.0, f32::max);

    assert!(
        max_error < 1e-4,
        "GPU output differs from CPU: max error = {}",
        max_error
    );
}

#[test]
#[ignore]
fn test_gru_forward_gpu_deterministic() {
    let hidden_dim = 128;
    let input_dim = 64;

    let weights = create_random_weights(hidden_dim, input_dim, 42);
    let hidden = create_random_vectors(hidden_dim, 123);
    let input = create_random_vectors(input_dim, 456);

    let config = GruKernelConfig {
        hidden_dim,
        input_dim,
        batch_size: 1,
    };

    // Run twice
    let output1 = gru_forward_gpu(&config, &weights, &hidden, &input)
        .expect("GPU forward failed");
    let output2 = gru_forward_gpu(&config, &weights, &hidden, &input)
        .expect("GPU forward failed");

    // Should be identical
    for (a, b) in output1.iter().zip(output2.iter()) {
        assert!(
            (a - b).abs() < 1e-7,
            "GPU forward is non-deterministic"
        );
    }
}

#[test]
#[ignore]
fn test_gru_forward_gpu_dimensions() {
    // Test various dimension combinations
    let test_cases = vec![
        (64, 32),
        (128, 64),
        (256, 128),
        (512, 256),
    ];

    for (hidden_dim, input_dim) in test_cases {
        let weights = create_random_weights(hidden_dim, input_dim, 42);
        let hidden = create_random_vectors(hidden_dim, 123);
        let input = create_random_vectors(input_dim, 456);

        let config = GruKernelConfig {
            hidden_dim,
            input_dim,
            batch_size: 1,
        };

        let result = gru_forward_gpu(&config, &weights, &hidden, &input);
        assert!(
            result.is_ok(),
            "GPU forward failed for dims ({}, {})",
            hidden_dim,
            input_dim
        );

        let output = result.unwrap();
        assert_eq!(
            output.len(),
            hidden_dim,
            "Wrong output size for dims ({}, {})",
            hidden_dim,
            input_dim
        );
    }
}

#[test]
#[ignore]
fn test_gru_forward_gpu_zero_input() {
    let hidden_dim = 128;
    let input_dim = 64;

    let weights = create_random_weights(hidden_dim, input_dim, 42);
    let hidden = vec![0.0; hidden_dim];
    let input = vec![0.0; input_dim];

    let config = GruKernelConfig {
        hidden_dim,
        input_dim,
        batch_size: 1,
    };

    let cpu_output = gru_forward_cpu(&weights, &hidden, &input);
    let gpu_output = gru_forward_gpu(&config, &weights, &hidden, &input)
        .expect("GPU forward failed");

    // With zero inputs, CPU and GPU must agree exactly
    for (c, g) in cpu_output.iter().zip(gpu_output.iter()) {
        assert!(
            (c - g).abs() < 1e-5,
            "Zero-input case failed: CPU={}, GPU={}",
            c,
            g
        );
    }
}

#[test]
#[ignore]
fn test_gru_forward_gpu_numerical_stability() {
    let hidden_dim = 128;
    let input_dim = 64;

    let weights = create_random_weights(hidden_dim, input_dim, 42);

    // Test with extreme values
    let test_cases = vec![
        ("large positive", vec![10.0; hidden_dim], vec![10.0; input_dim]),
        ("large negative", vec![-10.0; hidden_dim], vec![-10.0; input_dim]),
        ("mixed",
         create_random_vectors(hidden_dim, 999),
         create_random_vectors(input_dim, 888)),
    ];

    let config = GruKernelConfig {
        hidden_dim,
        input_dim,
        batch_size: 1,
    };

    for (name, hidden, input) in test_cases {
        let result = gru_forward_gpu(&config, &weights, &hidden, &input);
        assert!(result.is_ok(), "GPU forward failed for case: {}", name);

        let output = result.unwrap();

        // Check for NaN/Inf
        for (i, val) in output.iter().enumerate() {
            assert!(
                val.is_finite(),
                "Non-finite value at index {} for case {}: {}",
                i,
                name,
                val
            );
        }
    }
}
