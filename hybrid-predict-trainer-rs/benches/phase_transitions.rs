//! Phase transitions benchmark.
//!
//! Benchmarks the overhead of phase transition logic.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use hybrid_predict_trainer_rs::config::HybridTrainerConfig;
use hybrid_predict_trainer_rs::phases::{DefaultPhaseController, PhaseController};
use hybrid_predict_trainer_rs::state::TrainingState;

fn benchmark_phase_decision(c: &mut Criterion) {
    let config = HybridTrainerConfig::default();
    let controller = DefaultPhaseController::new(&config);
    let mut state = TrainingState::new();

    // Populate state with history
    for i in 0..100 {
        state.record_step(3.0 - i as f32 * 0.01, 1.0);
    }

    c.bench_function("phase_decision", |b| {
        b.iter(|| black_box(controller.decide(black_box(&state), 0.85)))
    });
}

fn benchmark_state_features(c: &mut Criterion) {
    let mut state = TrainingState::new();

    // Populate state with history
    for i in 0..100 {
        state.record_step(3.0 - i as f32 * 0.01, 1.0 + (i as f32 * 0.01).sin() * 0.1);
    }

    c.bench_function("state_compute_features", |b| {
        b.iter(|| black_box(state.compute_features()))
    });
}

criterion_group!(
    phase_benches,
    benchmark_phase_decision,
    benchmark_state_features,
);
criterion_main!(phase_benches);
