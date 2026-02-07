//! Comprehensive benchmarks for HybridTrainer
//!
//! Benchmarks key performance-critical paths:
//! - RSSM dynamics model prediction
//! - State encoding
//! - Weight delta application
//! - Phase transitions
//! - Full vs predict step comparison

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use hybrid_predict_trainer_rs::{
    config::{HybridTrainerConfig, PredictorConfig},
    dynamics::{DynamicsModel, RSSMLite},
    state::{TrainingState, WeightDelta},
    GradientInfo,
};
use std::collections::HashMap;

/// Benchmark RSSM prediction for different prediction horizons
fn bench_rssm_prediction(c: &mut Criterion) {
    let mut group = c.benchmark_group("rssm_prediction");

    let config = PredictorConfig::default();
    let mut rssm = RSSMLite::new(&config).expect("Failed to create RSSM");

    let mut state = TrainingState::new();
    state.loss = 2.0;
    state.gradient_norm = 1.5;
    state.optimizer_state_summary.effective_lr = 1e-3;
    state.record_step(2.0, 1.5);
    rssm.initialize_state(&state);

    // Train for a few steps to establish baseline
    let grad_info = GradientInfo {
        loss: 2.0,
        gradient_norm: 1.5,
        per_param_norms: None,
    };
    for _ in 0..10 {
        rssm.observe_gradient(&state, &grad_info);
    }

    // Benchmark different prediction horizons
    for horizon in [1, 5, 10, 15, 25, 50, 75].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(horizon),
            horizon,
            |b, &h| {
                b.iter(|| {
                    let (prediction, _uncertainty) = rssm.predict_y_steps(black_box(&state), black_box(h));
                    black_box(prediction);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark state feature computation
fn bench_state_encoding(c: &mut Criterion) {
    let mut group = c.benchmark_group("state_encoding");

    let mut state = TrainingState::new();
    state.loss = 2.0;
    state.gradient_norm = 1.5;

    // Add history
    for i in 0..100 {
        state.record_step(2.0 + (i as f32) * 0.01, 1.5);
    }

    group.bench_function("compute_features", |b| {
        b.iter(|| {
            let features = state.compute_features();
            black_box(features);
        });
    });

    group.finish();
}

/// Benchmark weight delta creation and scaling
fn bench_weight_delta_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("weight_delta");

    // Create a typical weight delta (10 dimensions)
    let mut deltas = HashMap::new();
    for i in 0..10 {
        deltas.insert(format!("param_{}", i), vec![0.001; 1000]);
    }

    let delta = WeightDelta {
        deltas: deltas.clone(),
        scale: 1.0,
        metadata: Default::default(),
    };

    group.bench_function("clone", |b| {
        b.iter(|| {
            let cloned = delta.clone();
            black_box(cloned);
        });
    });

    group.bench_function("scale", |b| {
        b.iter(|| {
            let mut scaled = delta.clone();
            scaled.scale = black_box(0.5);
            black_box(scaled);
        });
    });

    group.finish();
}

/// Benchmark RSSM gradient observation (training step)
fn bench_rssm_training(c: &mut Criterion) {
    let mut group = c.benchmark_group("rssm_training");

    let config = PredictorConfig::default();
    let mut rssm = RSSMLite::new(&config).expect("Failed to create RSSM");

    let mut state = TrainingState::new();
    state.loss = 2.0;
    state.gradient_norm = 1.5;
    state.record_step(2.0, 1.5);
    rssm.initialize_state(&state);

    let grad_info = GradientInfo {
        loss: 2.0,
        gradient_norm: 1.5,
        per_param_norms: None,
    };

    group.bench_function("observe_gradient", |b| {
        b.iter(|| {
            rssm.observe_gradient(black_box(&state), black_box(&grad_info));
        });
    });

    group.finish();
}

/// Benchmark confidence computation
fn bench_confidence_computation(c: &mut Criterion) {
    let mut group = c.benchmark_group("confidence");

    let config = PredictorConfig::default();
    let mut rssm = RSSMLite::new(&config).expect("Failed to create RSSM");

    let mut state = TrainingState::new();
    state.loss = 2.0;
    state.gradient_norm = 1.5;
    state.record_step(2.0, 1.5);
    rssm.initialize_state(&state);

    // Train for consistency
    let grad_info = GradientInfo {
        loss: 2.0,
        gradient_norm: 1.5,
        per_param_norms: None,
    };
    for _ in 0..20 {
        rssm.observe_gradient(&state, &grad_info);
    }

    group.bench_function("prediction_confidence", |b| {
        b.iter(|| {
            let confidence = rssm.prediction_confidence(black_box(&state));
            black_box(confidence);
        });
    });

    group.finish();
}

/// Benchmark training state history management
fn bench_state_history(c: &mut Criterion) {
    let mut group = c.benchmark_group("state_history");

    let mut state = TrainingState::new();

    group.bench_function("record_step", |b| {
        b.iter(|| {
            state.record_step(black_box(2.0), black_box(1.5));
        });
    });

    // Benchmark with full history
    for i in 0..1000 {
        state.record_step(2.0 + (i as f32) * 0.01, 1.5);
    }

    group.bench_function("record_step_full_history", |b| {
        b.iter(|| {
            state.record_step(black_box(2.0), black_box(1.5));
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_rssm_prediction,
    bench_state_encoding,
    bench_weight_delta_ops,
    bench_rssm_training,
    bench_confidence_computation,
    bench_state_history,
);

criterion_main!(benches);
