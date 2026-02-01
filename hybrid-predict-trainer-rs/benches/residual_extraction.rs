//! Residual extraction benchmarks.
//!
//! Benchmarks residual storage and retrieval performance.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use hybrid_predict_trainer_rs::residuals::{Residual, ResidualStore};
use hybrid_predict_trainer_rs::state::TrainingState;
use hybrid_predict_trainer_rs::Phase;

fn create_test_residual(step: u64) -> Residual {
    Residual {
        step,
        phase: Phase::Full,
        prediction_horizon: 10,
        loss_residual: 0.05 * (step as f32).sin(),
        gradient_residuals: Vec::new(),
        state_features: vec![step as f32 / 100.0; 32],
        prediction_confidence: 0.9,
    }
}

fn benchmark_residual_store_add(c: &mut Criterion) {
    let mut store = ResidualStore::new(1000);

    c.bench_function("residual_store_add", |b| {
        let mut step = 0u64;
        b.iter(|| {
            store.add(black_box(create_test_residual(step)));
            step = step.wrapping_add(1);
        })
    });
}

fn benchmark_residual_store_find_similar(c: &mut Criterion) {
    let mut store = ResidualStore::new(1000);

    // Populate store
    for i in 0..1000 {
        store.add(create_test_residual(i));
    }

    let mut state = TrainingState::new();
    for i in 0..50 {
        state.record_step(2.5 - i as f32 * 0.01, 1.0);
    }

    c.bench_function("residual_find_similar_10", |b| {
        b.iter(|| black_box(store.find_similar(black_box(&state), 10)))
    });

    c.bench_function("residual_find_similar_50", |b| {
        b.iter(|| black_box(store.find_similar(black_box(&state), 50)))
    });
}

criterion_group!(
    residual_benches,
    benchmark_residual_store_add,
    benchmark_residual_store_find_similar,
);
criterion_main!(residual_benches);
