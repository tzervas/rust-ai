//! Metrics collection and export tests

use hybrid_predict_trainer_rs::metrics::{MetricsCollector, StepMetrics};
use hybrid_predict_trainer_rs::prelude::*;
use hybrid_predict_trainer_rs::timing::{Duration, TimingMetrics};

/// Helper to create step metrics with proper timing
fn make_step(step: u64, phase: Phase, time_ms: f64, prediction_error: Option<f32>) -> StepMetrics {
    let timing =
        TimingMetrics::wall_clock_only(Duration::from_nanos((time_ms * 1_000_000.0) as u64));
    StepMetrics {
        step,
        loss: 2.0,
        gradient_norm: 1.0,
        phase,
        was_predicted: phase == Phase::Predict,
        prediction_error,
        confidence: if phase == Phase::Predict { 0.9 } else { 0.5 },
        timing,
        time_ms,
        learning_rate: Some(0.001),
    }
}

#[test]
fn test_metrics_collector_disabled() {
    let mut collector = MetricsCollector::new(false);
    assert_eq!(collector.statistics().total_steps, 0);
}

#[test]
fn test_metrics_collector_enabled() {
    let mut collector = MetricsCollector::new(true);

    // When enabled, collector should be ready
    let stats = collector.statistics();
    assert_eq!(stats.total_steps, 0);
    assert_eq!(stats.warmup_steps, 0);
}

#[test]
fn test_step_metrics_recording() {
    let mut collector = MetricsCollector::new(true);

    // Record a step
    let metrics = StepMetrics {
        step: 1,
        loss: 2.5,
        gradient_norm: 1.0,
        phase: Phase::Warmup,
        was_predicted: false,
        prediction_error: None,
        confidence: 0.5,
        timing: TimingMetrics::wall_clock_only(Duration::from_millis(10)),
        time_ms: 10.0,
        learning_rate: Some(0.001),
    };

    collector.record_step(metrics);

    let stats = collector.statistics();
    assert_eq!(stats.total_steps, 1);
    assert!((stats.final_loss - 2.5).abs() < f32::EPSILON);
}

#[test]
fn test_step_metrics_recording_convenience() {
    let mut collector = MetricsCollector::new(true);

    // Use convenience method
    let metrics = collector.record_step_data(42, 1.5, Phase::Full, false, None, 0.8);

    assert_eq!(metrics.step, 42);
    assert!((metrics.loss - 1.5).abs() < f32::EPSILON);
    assert_eq!(metrics.phase, Phase::Full);
    assert!(!metrics.was_predicted);
    assert!((metrics.confidence - 0.8).abs() < f32::EPSILON);
}

#[test]
fn test_metrics_finalization() {
    let mut collector = MetricsCollector::new(true);

    // Record warmup steps
    for i in 0..10 {
        collector.record_step(make_step(i, Phase::Warmup, 10.0, None));
    }

    // Record predict steps
    for i in 10..30 {
        collector.record_step(make_step(i, Phase::Predict, 5.0, Some(0.1)));
    }

    collector.finalize();

    let stats = collector.statistics();
    assert_eq!(stats.warmup_steps, 10);
    assert_eq!(stats.predict_steps, 20);
    assert!(stats.backward_reduction_pct > 0.0);
}

/// Regression test for bug where statistics() didn't auto-finalize
///
/// Previously, calling statistics() without manually calling finalize() first
/// would return TrainingStatistics with all derived fields set to zero
/// (backward_reduction_pct, avg_confidence, etc.) even when training had progressed.
///
/// This test ensures that statistics() automatically finalizes before returning.
#[test]
fn test_statistics_auto_finalizes() {
    let mut collector = MetricsCollector::new(true);

    // Record a mix of phases to exercise derived metrics
    // 5 warmup steps (backward pass)
    for i in 0..5 {
        collector.record_step(make_step(i, Phase::Warmup, 10.0, None));
    }

    // 5 full training steps (backward pass)
    for i in 5..10 {
        collector.record_step(make_step(i, Phase::Full, 10.0, None));
    }

    // 10 predict steps (NO backward pass - should increase reduction pct)
    for i in 10..20 {
        collector.record_step(make_step(i, Phase::Predict, 5.0, Some(0.1)));
    }

    // Call statistics() WITHOUT manually calling finalize() first
    // This is the key test - statistics() should auto-finalize
    let stats = collector.statistics();

    // Verify derived fields are computed (not zero)
    // This assertion would have FAILED before the fix
    assert!(
        stats.backward_reduction_pct > 0.0,
        "backward_reduction_pct should be computed automatically by statistics(), got {}",
        stats.backward_reduction_pct
    );

    // Verify total_steps matches last step number (0-indexed, so 0..20 = 19)
    assert_eq!(
        stats.total_steps, 19,
        "total_steps should be 19 (last step number), got {}",
        stats.total_steps
    );

    // Verify phase counts are correct
    assert_eq!(stats.warmup_steps, 5, "warmup_steps should be 5");
    assert_eq!(stats.full_steps, 5, "full_steps should be 5");
    assert_eq!(stats.predict_steps, 10, "predict_steps should be 10");

    // Verify backward reduction is calculated correctly
    // 10 backward steps (5 warmup + 5 full) out of 19 total steps (0-indexed)
    // Note: total_steps uses step numbers, so we need to account for this
    // in the calculation: backward_steps / (total_steps + 1)
    // Expected: (1 - 10/20) * 100 = 50%
    assert!(
        stats.backward_reduction_pct > 45.0 && stats.backward_reduction_pct < 55.0,
        "backward_reduction_pct should be ~50% (found {}), this is the KEY regression test - \
         before the fix, this would have been 0.0 because statistics() didn't call finalize()",
        stats.backward_reduction_pct
    );

    // Verify prediction accuracy is tracked
    assert!(
        stats.prediction_accuracy.loss_mae > 0.0,
        "prediction MAE should be computed (we recorded prediction errors)"
    );
}

#[test]
fn test_json_export() {
    let collector = MetricsCollector::new(true);

    let json = collector.to_json();
    assert!(json.is_ok());

    let json_str = json.unwrap();
    assert!(json_str.contains("training_summary"));
    assert!(json_str.contains("phase_history"));
    assert!(json_str.contains("divergence_events"));
}

#[test]
fn test_summary_output() {
    let mut collector = MetricsCollector::new(true);

    // Add some steps
    for i in 0..5 {
        collector.record_step(make_step(i, Phase::Warmup, 10.0, None));
    }

    collector.finalize();

    let summary = collector.summary();
    assert!(summary.contains("Training Summary"));
    assert!(summary.contains("Total Steps"));
    assert!(summary.contains("Backward Reduction"));
}

#[test]
fn test_metrics_reset() {
    let mut collector = MetricsCollector::new(true);

    // Record steps
    for i in 0..10 {
        collector.record_step(make_step(i, Phase::Warmup, 10.0, None));
    }

    let stats_before = collector.statistics();
    assert!(stats_before.total_steps > 0);

    // Reset
    collector.reset();

    let stats_after = collector.statistics();
    assert_eq!(stats_after.total_steps, 0);
}

#[test]
fn test_backward_reduction_calculation() {
    let mut collector = MetricsCollector::new(true);

    // 10 warmup (backward), 10 full (backward), 20 predict (no backward)
    for i in 0..10 {
        collector.record_step(make_step(i, Phase::Warmup, 10.0, None));
    }

    for i in 10..20 {
        collector.record_step(make_step(i, Phase::Full, 10.0, None));
    }

    for i in 20..40 {
        collector.record_step(make_step(i, Phase::Predict, 5.0, Some(0.1)));
    }

    collector.finalize();

    let stats = collector.statistics();
    // 20 backward steps (10 warmup + 10 full) out of 40 total
    // Reduction = (1 - 20/40) * 100 = 50%
    // However, total_steps is set from the last step number (39), not the count
    // So we need to account for this in the calculation
    // With step numbers 0-39, we have 40 steps total
    // The calculation uses step number + 1 effectively
    assert!(stats.warmup_steps == 10);
    assert!(stats.full_steps == 10);
    assert!(stats.predict_steps == 20);
    // Backward reduction should be close to 50% (allowing for rounding)
    assert!(stats.backward_reduction_pct > 45.0 && stats.backward_reduction_pct < 55.0);
}

#[test]
fn test_prediction_error_tracking() {
    let mut collector = MetricsCollector::new(true);

    // Record steps with prediction errors
    let errors = vec![0.1, 0.2, 0.15, 0.25];
    for (i, &error) in errors.iter().enumerate() {
        collector.record_step(make_step(i as u64, Phase::Predict, 5.0, Some(error)));
    }

    collector.finalize();

    let stats = collector.statistics();
    // Mean absolute error should be (0.1 + 0.2 + 0.15 + 0.25) / 4 = 0.175
    let expected_mae = 0.175;
    assert!((stats.prediction_accuracy.loss_mae - expected_mae).abs() < 0.01);
}

#[test]
fn test_timing_granularity_in_metrics() {
    let mut collector = MetricsCollector::new(true);

    // Record with specific timing
    let timing = TimingMetrics::wall_clock_only(Duration::from_nanos(1_500_000)); // 1.5ms
    let metrics = StepMetrics {
        step: 0,
        loss: 2.0,
        gradient_norm: 1.0,
        phase: Phase::Full,
        was_predicted: false,
        prediction_error: None,
        confidence: 0.8,
        timing,
        time_ms: 1.5,
        learning_rate: Some(0.001),
    };

    collector.record_step(metrics.clone());

    // Verify granularity accessors
    assert_eq!(metrics.time_nanos(), 1_500_000);
    assert_eq!(metrics.time_picos(), 1_500_000_000);
    assert!((metrics.time_ms - 1.5).abs() < 0.001);
}

#[test]
fn test_gpu_timing_metrics() {
    let timing = TimingMetrics::with_gpu(
        Duration::from_millis(10), // 10ms wall clock
        Duration::from_millis(8),  // 8ms GPU compute
    );

    let metrics = StepMetrics {
        step: 0,
        loss: 2.0,
        gradient_norm: 1.0,
        phase: Phase::Predict,
        was_predicted: true,
        prediction_error: None,
        confidence: 0.9,
        timing,
        time_ms: 10.0,
        learning_rate: Some(0.001),
    };

    // Wall-clock time
    assert_eq!(metrics.time_nanos(), 10_000_000);
    assert!((metrics.time_ms - 10.0).abs() < 0.001);

    // GPU compute time
    assert_eq!(metrics.gpu_time_nanos(), Some(8_000_000));
    assert_eq!(metrics.gpu_time_ms(), Some(8.0));
    assert_eq!(metrics.gpu_time_picos(), Some(8_000_000_000));

    // CPU overhead = wall_clock - gpu = 2ms
    assert_eq!(metrics.timing.cpu_overhead().unwrap().as_millis(), 2);
}
