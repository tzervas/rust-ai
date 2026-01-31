//! Predictor performance benchmarks.
//!
//! Benchmarks the dynamics model prediction performance.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use hybrid_predict_trainer_rs::config::PredictorConfig;
use hybrid_predict_trainer_rs::dynamics::{DynamicsModel, RSSMLite};
use hybrid_predict_trainer_rs::state::TrainingState;

fn benchmark_rssm_prediction(c: &mut Criterion) {
    let config = PredictorConfig::default();
    let mut rssm = RSSMLite::new(&config).expect("Failed to create RSSM");

    let mut state = TrainingState::new();
    for i in 0..50 {
        state.record_step(3.0 - i as f32 * 0.01, 1.0);
    }

    rssm.initialize(&state);

    c.bench_function("rssm_predict_10_steps", |b| {
        b.iter(|| black_box(rssm.predict_y_steps(black_box(&state), 10)))
    });

    c.bench_function("rssm_predict_50_steps", |b| {
        b.iter(|| black_box(rssm.predict_y_steps(black_box(&state), 50)))
    });
}

fn benchmark_rssm_confidence(c: &mut Criterion) {
    let config = PredictorConfig::default();
    let rssm = RSSMLite::new(&config).expect("Failed to create RSSM");

    let state = TrainingState::new();

    c.bench_function("rssm_confidence", |b| {
        b.iter(|| black_box(rssm.prediction_confidence(black_box(&state))))
    });
}

criterion_group!(
    predictor_benches,
    benchmark_rssm_prediction,
    benchmark_rssm_confidence,
);
criterion_main!(predictor_benches);
