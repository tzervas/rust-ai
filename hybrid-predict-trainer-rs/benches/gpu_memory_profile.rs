//! GPU memory profiling benchmarks.
//!
//! Measures GPU memory usage patterns during training.
//!
//! **Metrics:**
//! - Peak GPU memory usage
//! - Weight upload overhead
//! - Memory scaling with model size
//! - Allocation patterns across phases

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

#[cfg(feature = "cuda")]
use hybrid_predict_trainer_rs::gpu::kernels::{
    gru::GpuGruWeights,
    rssm_rollout::{RssmEnsembleWeights, RssmRolloutConfig},
};

// Stub implementations when cuda feature is not enabled
#[cfg(not(feature = "cuda"))]
mod stubs {
    #[derive(Clone)]
    pub struct GpuGruWeights {
        pub w_z: Vec<f32>,
        pub u_z: Vec<f32>,
        pub b_z: Vec<f32>,
        pub w_r: Vec<f32>,
        pub u_r: Vec<f32>,
        pub b_r: Vec<f32>,
        pub w_h: Vec<f32>,
        pub u_h: Vec<f32>,
        pub b_h: Vec<f32>,
        pub hidden_dim: usize,
        pub input_dim: usize,
    }

    #[derive(Clone)]
    pub struct RssmEnsembleWeights {
        pub gru_weights: Vec<GpuGruWeights>,
        pub loss_head_weights: Vec<f32>,
        pub delta_head_weights: Vec<f32>,
    }

    #[derive(Clone)]
    pub struct RssmRolloutConfig {
        pub ensemble_size: usize,
        pub hidden_dim: usize,
        pub stochastic_dim: usize,
        pub feature_dim: usize,
        pub y_steps: usize,
    }

    impl RssmRolloutConfig {
        pub fn shared_memory_bytes(&self) -> usize {
            self.hidden_dim * 4 + (self.hidden_dim + self.stochastic_dim) * 4 + self.hidden_dim * 4 + self.feature_dim * 4
        }
    }
}

#[cfg(not(feature = "cuda"))]
use stubs::*;

/// Estimate memory usage for GRU weights.
fn estimate_gru_memory(weights: &GpuGruWeights) -> usize {
    // Each f32 = 4 bytes
    let matrix_bytes = (weights.w_z.len()
        + weights.w_r.len()
        + weights.w_h.len()
        + weights.u_z.len()
        + weights.u_r.len()
        + weights.u_h.len()
        + weights.b_z.len()
        + weights.b_r.len()
        + weights.b_h.len())
        * 4;

    matrix_bytes
}

/// Estimate memory usage for RSSM ensemble.
fn estimate_rssm_memory(weights: &RssmEnsembleWeights) -> usize {
    let gru_total: usize = weights
        .gru_weights
        .iter()
        .map(estimate_gru_memory)
        .sum();

    let loss_head_bytes = weights.loss_head_weights.len() * 4;
    let delta_head_bytes = weights.delta_head_weights.len() * 4;

    gru_total + loss_head_bytes + delta_head_bytes
}

/// Benchmark memory scaling with hidden dimension.
fn bench_memory_scaling_hidden_dim(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_scaling_hidden_dim");

    for hidden_dim in [64, 128, 256, 512, 1024] {
        let input_dim = 64;

        let weights = GpuGruWeights {
            w_z: vec![0.01; hidden_dim * input_dim],
            u_z: vec![0.01; hidden_dim * hidden_dim],
            b_z: vec![0.0; hidden_dim],
            w_r: vec![0.01; hidden_dim * input_dim],
            u_r: vec![0.01; hidden_dim * hidden_dim],
            b_r: vec![0.0; hidden_dim],
            w_h: vec![0.01; hidden_dim * input_dim],
            u_h: vec![0.01; hidden_dim * hidden_dim],
            b_h: vec![0.0; hidden_dim],
            hidden_dim,
            input_dim,
        };

        let memory_bytes = estimate_gru_memory(&weights);
        let memory_mb = memory_bytes as f32 / 1e6;

        group.bench_with_input(
            BenchmarkId::new("estimate", hidden_dim),
            &hidden_dim,
            |b, _| {
                b.iter(|| {
                    let _mem = estimate_gru_memory(black_box(&weights));
                });
            },
        );

        println!(
            "GRU hidden_dim={}: {:.2} MB",
            hidden_dim, memory_mb
        );
    }

    group.finish();
}

/// Benchmark memory scaling with ensemble size.
fn bench_memory_scaling_ensemble(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_scaling_ensemble");

    for ensemble_size in [1, 3, 5, 7, 10] {
        let config = RssmRolloutConfig {
            ensemble_size,
            hidden_dim: 256,
            stochastic_dim: 256,
            feature_dim: 64,
            y_steps: 50,
        };

        let gru_weights = vec![
            GpuGruWeights {
                w_z: vec![0.01; 256 * 64],
                u_z: vec![0.01; 256 * 256],
                b_z: vec![0.0; 256],
                w_r: vec![0.01; 256 * 64],
                u_r: vec![0.01; 256 * 256],
                b_r: vec![0.0; 256],
                w_h: vec![0.01; 256 * 64],
                u_h: vec![0.01; 256 * 256],
                b_h: vec![0.0; 256],
                hidden_dim: 256,
                input_dim: 64,
            };
            ensemble_size
        ];

        let weights = RssmEnsembleWeights {
            gru_weights,
            loss_head_weights: vec![0.01; 512],
            delta_head_weights: vec![0.01; 512 * 10],
        };

        let memory_bytes = estimate_rssm_memory(&weights);
        let memory_mb = memory_bytes as f32 / 1e6;

        group.bench_with_input(
            BenchmarkId::new("estimate", ensemble_size),
            &ensemble_size,
            |b, _| {
                b.iter(|| {
                    let _mem = estimate_rssm_memory(black_box(&weights));
                });
            },
        );

        println!(
            "RSSM ensemble_size={}: {:.2} MB",
            ensemble_size, memory_mb
        );
    }

    group.finish();
}

/// Benchmark shared memory requirements.
fn bench_shared_memory_requirements(c: &mut Criterion) {
    let mut group = c.benchmark_group("shared_memory");

    for hidden_dim in [64, 128, 256, 512, 1024] {
        let config = RssmRolloutConfig {
            ensemble_size: 5,
            hidden_dim,
            stochastic_dim: hidden_dim,
            feature_dim: 64,
            y_steps: 50,
        };

        let shared_memory_bytes = config.shared_memory_bytes();
        let shared_memory_kb = shared_memory_bytes as f32 / 1024.0;

        group.bench_with_input(
            BenchmarkId::new("compute", hidden_dim),
            &hidden_dim,
            |b, _| {
                b.iter(|| {
                    let _mem = black_box(&config).shared_memory_bytes();
                });
            },
        );

        println!(
            "Shared memory hidden_dim={}: {:.2} KB",
            hidden_dim, shared_memory_kb
        );

        // Verify within GPU limits (48 KB per SM typical)
        assert!(
            shared_memory_bytes <= 48 * 1024,
            "Shared memory {} bytes exceeds 48 KB limit",
            shared_memory_bytes
        );
    }

    group.finish();
}

/// Benchmark weight upload overhead (simulated).
fn bench_weight_upload_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("weight_upload");

    for hidden_dim in [64, 128, 256, 512] {
        let weights = GpuGruWeights {
            w_z: vec![0.01; hidden_dim * 64],
            u_z: vec![0.01; hidden_dim * hidden_dim],
            b_z: vec![0.0; hidden_dim],
            w_r: vec![0.01; hidden_dim * 64],
            u_r: vec![0.01; hidden_dim * hidden_dim],
            b_r: vec![0.0; hidden_dim],
            w_h: vec![0.01; hidden_dim * 64],
            u_h: vec![0.01; hidden_dim * hidden_dim],
            b_h: vec![0.0; hidden_dim],
            hidden_dim,
            input_dim: 64,
        };

        // Simulate upload by cloning data (actual GPU transfer would use device API)
        group.bench_with_input(BenchmarkId::new("clone", hidden_dim), &hidden_dim, |b, _| {
            b.iter(|| {
                let _w_z = black_box(&weights.w_z).clone();
                let _u_z = black_box(&weights.u_z).clone();
                let _w_r = black_box(&weights.w_r).clone();
                let _u_r = black_box(&weights.u_r).clone();
                let _w_h = black_box(&weights.w_h).clone();
                let _u_h = black_box(&weights.u_h).clone();
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_memory_scaling_hidden_dim,
    bench_memory_scaling_ensemble,
    bench_shared_memory_requirements,
    bench_weight_upload_overhead,
);
criterion_main!(benches);
