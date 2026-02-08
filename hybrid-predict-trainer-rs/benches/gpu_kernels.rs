//! GPU kernel benchmarks.
//!
//! Measures performance of individual GPU kernels vs CPU implementations.
//!
//! **Targets:**
//! - GRU forward: >10× speedup at hidden_dim=256
//! - RSSM rollout: >5× speedup for 50-step, 5-ensemble
//! - State encoding: >5× speedup for 64-dim with history

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

#[cfg(feature = "cuda")]
use hybrid_predict_trainer_rs::gpu::kernels::{
    gru::{gru_forward_cpu, GpuGruWeights},
    rssm_rollout::{rssm_rollout_cpu, RssmEnsembleWeights, RssmRolloutConfig},
    state_encode::{encode_state_cpu, StateEncodeConfig},
};

#[cfg(feature = "cuda")]
use hybrid_predict_trainer_rs::state::TrainingState;

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

    pub fn gru_forward_cpu(_weights: &GpuGruWeights, _hidden: &[f32], _input: &[f32]) -> Vec<f32> {
        vec![]
    }

    pub fn rssm_rollout_cpu(
        _config: &RssmRolloutConfig,
        _weights: &RssmEnsembleWeights,
        _initial_latents: &[Vec<f32>],
        _features: &[f32],
        _initial_loss: f32,
    ) -> Vec<()> {
        vec![]
    }

    pub fn encode_state_cpu(_state: &TrainingState) -> Vec<f32> {
        vec![0.0; 64]
    }

    pub struct StateEncodeConfig;
    impl StateEncodeConfig {
        pub fn default() -> Self {
            Self
        }
    }

    pub struct TrainingState {
        pub step: usize,
        pub loss: f32,
        pub gradient_norm: f32,
    }

    impl TrainingState {
        pub fn default() -> Self {
            Self {
                step: 0,
                loss: 0.0,
                gradient_norm: 0.0,
            }
        }
        pub fn record_step(&mut self, _loss: f32, _grad: f32) {}
    }
}

#[cfg(not(feature = "cuda"))]
use stubs::*;

/// Benchmark GRU forward pass across dimensions.
fn bench_gru_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("gru_forward");

    for hidden_dim in [64, 128, 256, 512, 1024] {
        let input_dim = 64; // Fixed feature dim

        // Create weights
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

        let hidden = vec![0.5; hidden_dim];
        let input = vec![0.5; input_dim];

        group.throughput(Throughput::Elements(hidden_dim as u64));

        group.bench_with_input(
            BenchmarkId::new("cpu", hidden_dim),
            &hidden_dim,
            |b, _| {
                b.iter(|| {
                    let _output = gru_forward_cpu(
                        black_box(&weights),
                        black_box(&hidden),
                        black_box(&input),
                    );
                });
            },
        );

        // TODO: Add GPU benchmark once runtime integration is complete
        // group.bench_with_input(
        //     BenchmarkId::new("gpu", hidden_dim),
        //     &hidden_dim,
        //     |b, _| {
        //         b.iter(|| {
        //             let _output = gru_forward_gpu(...);
        //         });
        //     },
        // );
    }

    group.finish();
}

/// Benchmark RSSM rollout across horizons.
fn bench_rssm_rollout(c: &mut Criterion) {
    let mut group = c.benchmark_group("rssm_rollout");

    for y_steps in [10, 25, 50, 75, 100] {
        let config = RssmRolloutConfig {
            ensemble_size: 5,
            hidden_dim: 256,
            stochastic_dim: 256,
            feature_dim: 64,
            y_steps,
        };

        // Create dummy weights
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
            5
        ];

        let weights = RssmEnsembleWeights {
            gru_weights,
            loss_head_weights: vec![0.01; 512],
            delta_head_weights: vec![0.01; 512 * 10],
        };

        let initial_latents = vec![vec![0.5; 256]; 5];
        let features = vec![0.5; 64];
        let initial_loss = 1.0;

        group.throughput(Throughput::Elements(y_steps as u64 * 5));

        group.bench_with_input(BenchmarkId::new("cpu", y_steps), &y_steps, |b, _| {
            b.iter(|| {
                let _results = rssm_rollout_cpu(
                    black_box(&config),
                    black_box(&weights),
                    black_box(&initial_latents),
                    black_box(&features),
                    black_box(initial_loss),
                );
            });
        });

        // TODO: Add GPU benchmark
    }

    group.finish();
}

/// Benchmark state encoding.
fn bench_state_encode(c: &mut Criterion) {
    let mut group = c.benchmark_group("state_encode");

    let config = StateEncodeConfig::default();

    // Create state with varying history lengths
    for history_len in [0, 10, 50, 100, 500] {
        let mut state = TrainingState::default();
        state.step = history_len + 10;
        state.loss = 0.5;
        state.gradient_norm = 0.01;

        // Populate history
        for i in 0..history_len {
            state.record_step(0.5 - (i as f32 * 0.001), 0.01 + (i as f32 * 0.0001));
        }

        group.bench_with_input(
            BenchmarkId::new("cpu", history_len),
            &history_len,
            |b, _| {
                b.iter(|| {
                    let _features = encode_state_cpu(black_box(&state));
                });
            },
        );

        // TODO: Add GPU benchmark
        // #[cfg(feature = "cuda")]
        // group.bench_with_input(
        //     BenchmarkId::new("gpu", history_len),
        //     &history_len,
        //     |b, _| {
        //         b.iter(|| {
        //             let _features = encode_state_gpu(&config, &state).unwrap();
        //         });
        //     },
        // );
    }

    group.finish();
}

/// Benchmark end-to-end hybrid trainer step.
fn bench_hybrid_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("hybrid_step");

    // Placeholder: Would benchmark full HybridTrainer::step()
    // across different phases and model sizes

    group.bench_function("dummy", |b| {
        b.iter(|| {
            // TODO: Full integration with Burn backend
            black_box(42);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_gru_forward,
    bench_rssm_rollout,
    bench_state_encode,
    bench_hybrid_step,
);
criterion_main!(benches);
