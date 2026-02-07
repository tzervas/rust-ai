//! Comprehensive integration tests for training-tools
//!
//! Tests cover:
//! 1. StepMetrics serialization/deserialization roundtrip
//! 2. TrainingRun state transitions
//! 3. LiveMetricsReader file parsing
//! 4. WSDScheduler LR values at key steps
//! 5. CurveQuality detection accuracy (via lr_advisor)
//! 6. GeneralizationHealth detection

use std::fs;
use std::io::Write;
use std::path::PathBuf;

use chrono::Utc;
use training_tools::{
    calculate_loss_dynamics, compute_config_hash, GeneralizationHealth, LRScheduler, StepMetrics,
    TrainingConfig, TrainingPhase, TrainingRun, TrainingStatus, WSDScheduler,
};

// Import lr_advisor for curve quality detection
use training_tools::lr_advisor::TrainingPhase as LRPhase;
use training_tools::lr_advisor::{analyze_lr, Issue, Urgency};

// ============================================================================
// Test 1: StepMetrics Serialization/Deserialization Roundtrip
// ============================================================================

#[test]
fn test_step_metrics_roundtrip_normal() {
    let original = StepMetrics {
        step: 100,
        loss: 2.456,
        gradient_norm: 0.789,
        phase: TrainingPhase::Full,
        was_predicted: false,
        prediction_error: None,
        step_time_ms: 125.5,
        timestamp: Utc::now(),
        tokens_this_step: 4096,
        total_tokens_trained: 409600,
        tokens_remaining: 1_000_000,
        confidence: 0.95,
        learning_rate: 1e-4,
        perplexity: 11.66,
        train_val_gap: Some(0.05),
        loss_velocity: -0.001,
        loss_acceleration: 0.0001,
        gradient_entropy: Some(2.34),
        layer_gradients: None,
        layer_gradient_stats: None,
    };

    // Serialize to JSON
    let json = serde_json::to_string(&original).expect("serialization failed");

    // Deserialize back
    let deserialized: StepMetrics = serde_json::from_str(&json).expect("deserialization failed");

    // Verify all fields match
    assert_eq!(deserialized.step, original.step);
    assert_eq!(deserialized.loss, original.loss);
    assert_eq!(deserialized.gradient_norm, original.gradient_norm);
    assert_eq!(deserialized.phase, original.phase);
    assert_eq!(deserialized.was_predicted, original.was_predicted);
    assert_eq!(deserialized.prediction_error, original.prediction_error);
    assert_eq!(deserialized.step_time_ms, original.step_time_ms);
    assert_eq!(deserialized.tokens_this_step, original.tokens_this_step);
    assert_eq!(
        deserialized.total_tokens_trained,
        original.total_tokens_trained
    );
    assert_eq!(deserialized.tokens_remaining, original.tokens_remaining);
    assert_eq!(deserialized.confidence, original.confidence);
    assert_eq!(deserialized.learning_rate, original.learning_rate);
    assert_eq!(deserialized.perplexity, original.perplexity);
    assert_eq!(deserialized.train_val_gap, original.train_val_gap);
    assert_eq!(deserialized.loss_velocity, original.loss_velocity);
    assert_eq!(deserialized.loss_acceleration, original.loss_acceleration);
    assert_eq!(deserialized.gradient_entropy, original.gradient_entropy);
    assert_eq!(deserialized.layer_gradients, original.layer_gradients);
    assert_eq!(
        deserialized.layer_gradient_stats,
        original.layer_gradient_stats
    );
}

#[test]
fn test_step_metrics_roundtrip_with_nan() {
    let original = StepMetrics {
        step: 50,
        loss: f32::NAN,
        gradient_norm: f32::NAN,
        phase: TrainingPhase::Warmup,
        was_predicted: true,
        prediction_error: Some(f32::NAN),
        step_time_ms: 100.0,
        timestamp: Utc::now(),
        tokens_this_step: 2048,
        total_tokens_trained: 102400,
        tokens_remaining: 500_000,
        confidence: f32::NAN,
        learning_rate: 1e-5,
        perplexity: f32::NAN,
        train_val_gap: Some(f32::NAN),
        loss_velocity: f32::NAN,
        loss_acceleration: f32::NAN,
        gradient_entropy: Some(f32::NAN),
        layer_gradients: None,
        layer_gradient_stats: None,
    };

    // Serialize to JSON (NaN becomes null in JSON)
    let json = serde_json::to_string(&original).expect("serialization failed");

    // Deserialize back
    let deserialized: StepMetrics = serde_json::from_str(&json).expect("deserialization failed");

    // NaN != NaN, so we check with is_nan()
    assert!(deserialized.loss.is_nan());
    assert!(deserialized.gradient_norm.is_nan());
    assert!(deserialized.prediction_error.unwrap().is_nan());
    assert!(deserialized.confidence.is_nan());
    assert!(deserialized.perplexity.is_nan());
    assert!(deserialized.train_val_gap.unwrap().is_nan());
}

#[test]
fn test_step_metrics_roundtrip_with_inf() {
    let original = StepMetrics {
        step: 1,
        loss: f32::INFINITY,
        gradient_norm: f32::NEG_INFINITY,
        phase: TrainingPhase::Correct,
        was_predicted: false,
        prediction_error: Some(f32::INFINITY),
        step_time_ms: 50.0,
        timestamp: Utc::now(),
        tokens_this_step: 512,
        total_tokens_trained: 512,
        tokens_remaining: 10_000_000,
        confidence: 0.0,
        learning_rate: 1e-6,
        perplexity: f32::INFINITY,
        train_val_gap: None,
        loss_velocity: 0.0,
        loss_acceleration: 0.0,
        gradient_entropy: None,
        layer_gradients: None,
        layer_gradient_stats: None,
    };

    let json = serde_json::to_string(&original).expect("serialization failed");
    let deserialized: StepMetrics = serde_json::from_str(&json).expect("deserialization failed");

    assert!(deserialized.loss.is_infinite() && deserialized.loss.is_sign_positive());
    assert!(
        deserialized.gradient_norm.is_infinite() && deserialized.gradient_norm.is_sign_negative()
    );
    assert!(deserialized.prediction_error.unwrap().is_infinite());
}

#[test]
fn test_step_metrics_empty_optional_fields() {
    let original = StepMetrics {
        step: 0,
        loss: 5.0,
        gradient_norm: 1.0,
        phase: TrainingPhase::Warmup,
        was_predicted: false,
        prediction_error: None,
        step_time_ms: 200.0,
        timestamp: Utc::now(),
        tokens_this_step: 0,
        total_tokens_trained: 0,
        tokens_remaining: 0,
        confidence: 0.0,
        learning_rate: 0.0,
        perplexity: 0.0,
        train_val_gap: None,
        loss_velocity: 0.0,
        loss_acceleration: 0.0,
        gradient_entropy: None,
        layer_gradients: None,
        layer_gradient_stats: None,
    };

    let json = serde_json::to_string(&original).expect("serialization failed");
    let deserialized: StepMetrics = serde_json::from_str(&json).expect("deserialization failed");

    assert_eq!(deserialized.prediction_error, None);
    assert_eq!(deserialized.train_val_gap, None);
    assert_eq!(deserialized.gradient_entropy, None);
    assert_eq!(deserialized.layer_gradients, None);
    assert_eq!(deserialized.layer_gradient_stats, None);
}

// ============================================================================
// Test 2: TrainingRun State Transitions
// ============================================================================

#[test]
fn test_training_run_lifecycle() {
    let config = create_test_config("100m");
    let temp_dir = tempfile::tempdir().expect("failed to create temp dir");
    let run_dir = temp_dir.path().join("run_001");
    fs::create_dir_all(&run_dir).expect("failed to create run dir");

    let mut run = TrainingRun::new("test_run", config, run_dir.clone());

    // Initial state
    assert_eq!(run.status, TrainingStatus::Initializing);
    assert_eq!(run.current_step, 0);
    assert_eq!(run.current_loss, f32::INFINITY);
    assert_eq!(run.best_loss, f32::INFINITY);

    // Transition to Running
    run.status = TrainingStatus::Running;
    assert!(run.status.is_active());
    assert!(!run.status.is_finished());

    // Update with metrics
    let metrics = create_test_metric(10, 2.5, TrainingPhase::Warmup, false);
    run.update_step(&metrics);
    assert_eq!(run.current_step, 10);
    assert_eq!(run.current_loss, 2.5);
    assert_eq!(run.best_loss, 2.5);
    assert_eq!(run.best_step, 10);
    assert_eq!(run.total_forward, 1);
    assert_eq!(run.total_backward, 1);

    // Update with predicted step (no backward)
    let metrics2 = create_test_metric(11, 2.4, TrainingPhase::Predict, true);
    run.update_step(&metrics2);
    assert_eq!(run.total_forward, 2);
    assert_eq!(run.total_backward, 1); // No increase
    assert_eq!(run.best_loss, 2.4);

    // Transition to Paused
    run.status = TrainingStatus::Paused;
    assert!(run.status.is_active());

    // Transition to Completed
    run.status = TrainingStatus::Completed;
    run.ended_at = Some(Utc::now());
    assert!(!run.status.is_active());
    assert!(run.status.is_finished());

    // Verify save/load
    run.save().expect("failed to save run");
    let loaded = TrainingRun::load(&run_dir).expect("failed to load run");
    assert_eq!(loaded.run_id, run.run_id);
    assert_eq!(loaded.status, TrainingStatus::Completed);
    assert_eq!(loaded.current_step, 11);
}

#[test]
fn test_training_run_phase_transitions() {
    let config = create_test_config("100m");
    let temp_dir = tempfile::tempdir().expect("failed to create temp dir");
    let run_dir = temp_dir.path().join("run_phase");
    fs::create_dir_all(&run_dir).expect("failed to create run dir");

    let mut run = TrainingRun::new("phase_test", config, run_dir);

    // Record phase transitions
    run.record_phase_transition(TrainingPhase::Warmup, TrainingPhase::Full);
    assert_eq!(run.phase_transitions.len(), 1);
    assert_eq!(run.phase_transitions[0].from_phase, TrainingPhase::Warmup);
    assert_eq!(run.phase_transitions[0].to_phase, TrainingPhase::Full);

    run.current_step = 100;
    run.record_phase_transition(TrainingPhase::Full, TrainingPhase::Predict);
    assert_eq!(run.phase_transitions.len(), 2);
    assert_eq!(run.phase_transitions[1].step, 100);
}

#[test]
fn test_training_run_backward_reduction() {
    let config = create_test_config("100m");
    let temp_dir = tempfile::tempdir().expect("failed to create temp dir");
    let run_dir = temp_dir.path().join("run_reduction");
    fs::create_dir_all(&run_dir).expect("failed to create run dir");

    let mut run = TrainingRun::new("reduction_test", config, run_dir);

    // 10 forward, 5 backward (50% reduction)
    for i in 0..10 {
        let was_predicted = i >= 5;
        let metrics = create_test_metric(i, 2.0, TrainingPhase::Predict, was_predicted);
        run.update_step(&metrics);
    }

    assert_eq!(run.total_forward, 10);
    assert_eq!(run.total_backward, 5);
    assert_eq!(run.backward_reduction(), 50.0);
}

#[test]
fn test_training_run_progress() {
    let config = create_test_config("100m");
    let temp_dir = tempfile::tempdir().expect("failed to create temp dir");
    let run_dir = temp_dir.path().join("run_progress");
    fs::create_dir_all(&run_dir).expect("failed to create run dir");

    let mut run = TrainingRun::new("progress_test", config, run_dir);

    assert_eq!(run.progress(), 0.0);

    run.current_step = 5000; // 50% of 10000
    assert_eq!(run.progress(), 50.0);

    run.current_step = 10000;
    assert_eq!(run.progress(), 100.0);
}

// ============================================================================
// Test 3: LiveMetricsReader File Parsing
// ============================================================================

#[test]
fn test_live_metrics_reader_empty_file() {
    let temp_dir = tempfile::tempdir().expect("failed to create temp dir");
    let metrics_file = temp_dir.path().join("metrics_empty.jsonl");
    fs::write(&metrics_file, "").expect("failed to write empty file");

    // LiveMetricsReader is not public, but we can test via TrainingRun::read_recent_metrics
    let config = create_test_config("100m");
    let run_dir = temp_dir.path().join("run_empty");
    fs::create_dir_all(&run_dir).expect("failed to create run dir");

    let mut run = TrainingRun::new("empty_test", config, run_dir);
    run.metrics_file = metrics_file;

    let metrics = run.read_recent_metrics(10).expect("failed to read metrics");
    assert_eq!(metrics.len(), 0);
}

#[test]
fn test_live_metrics_reader_single_metric() {
    let temp_dir = tempfile::tempdir().expect("failed to create temp dir");
    let metrics_file = temp_dir.path().join("metrics_single.jsonl");

    let metric = create_test_metric(1, 3.0, TrainingPhase::Warmup, false);
    let json = serde_json::to_string(&metric).expect("serialization failed");
    fs::write(&metrics_file, format!("{}\n", json)).expect("failed to write file");

    let config = create_test_config("100m");
    let run_dir = temp_dir.path().join("run_single");
    fs::create_dir_all(&run_dir).expect("failed to create run dir");

    let mut run = TrainingRun::new("single_test", config, run_dir);
    run.metrics_file = metrics_file;

    let metrics = run.read_recent_metrics(10).expect("failed to read metrics");
    assert_eq!(metrics.len(), 1);
    assert_eq!(metrics[0].step, 1);
    assert_eq!(metrics[0].loss, 3.0);
}

#[test]
fn test_live_metrics_reader_multiple_metrics() {
    let temp_dir = tempfile::tempdir().expect("failed to create temp dir");
    let metrics_file = temp_dir.path().join("metrics_multi.jsonl");

    let mut file = fs::File::create(&metrics_file).expect("failed to create file");
    for i in 0..100 {
        let metric = create_test_metric(i, 3.0 - (i as f32 * 0.01), TrainingPhase::Full, false);
        let json = serde_json::to_string(&metric).expect("serialization failed");
        writeln!(file, "{}", json).expect("failed to write line");
    }

    let config = create_test_config("100m");
    let run_dir = temp_dir.path().join("run_multi");
    fs::create_dir_all(&run_dir).expect("failed to create run dir");

    let mut run = TrainingRun::new("multi_test", config, run_dir);
    run.metrics_file = metrics_file;

    // Read last 20
    let metrics = run.read_recent_metrics(20).expect("failed to read metrics");
    assert_eq!(metrics.len(), 20);
    assert_eq!(metrics[0].step, 80);
    assert_eq!(metrics[19].step, 99);

    // Read more than available
    let all_metrics = run
        .read_recent_metrics(200)
        .expect("failed to read metrics");
    assert_eq!(all_metrics.len(), 100);
}

#[test]
fn test_live_metrics_reader_with_invalid_lines() {
    let temp_dir = tempfile::tempdir().expect("failed to create temp dir");
    let metrics_file = temp_dir.path().join("metrics_invalid.jsonl");

    let mut file = fs::File::create(&metrics_file).expect("failed to create file");

    // Valid line
    let metric1 = create_test_metric(1, 2.5, TrainingPhase::Warmup, false);
    writeln!(file, "{}", serde_json::to_string(&metric1).unwrap()).expect("write failed");

    // Invalid JSON
    writeln!(file, "{{invalid json}}").expect("write failed");

    // Valid line
    let metric2 = create_test_metric(2, 2.4, TrainingPhase::Full, false);
    writeln!(file, "{}", serde_json::to_string(&metric2).unwrap()).expect("write failed");

    // Empty line
    writeln!(file).expect("write failed");

    // Valid line
    let metric3 = create_test_metric(3, 2.3, TrainingPhase::Full, false);
    writeln!(file, "{}", serde_json::to_string(&metric3).unwrap()).expect("write failed");

    let config = create_test_config("100m");
    let run_dir = temp_dir.path().join("run_invalid");
    fs::create_dir_all(&run_dir).expect("failed to create run dir");

    let mut run = TrainingRun::new("invalid_test", config, run_dir);
    run.metrics_file = metrics_file;

    let metrics = run.read_recent_metrics(10).expect("failed to read metrics");
    // Should only read valid lines (invalid ones are filtered out)
    assert_eq!(metrics.len(), 3);
    assert_eq!(metrics[0].step, 1);
    assert_eq!(metrics[1].step, 2);
    assert_eq!(metrics[2].step, 3);
}

// ============================================================================
// Test 4: WSDScheduler LR Values at Key Steps
// ============================================================================

#[test]
fn test_wsd_scheduler_warmup_phase() {
    let scheduler = WSDScheduler::builder()
        .total_steps(1000)
        .peak_lr(1e-4)
        .min_lr(1e-6)
        .warmup_fraction(0.1) // 100 steps
        .decay_fraction(0.2) // 200 steps
        .build()
        .expect("failed to build scheduler");

    // Step 0: should be 0
    assert_eq!(scheduler.get_lr(0), 0.0);

    // Step 1: should be 1% of peak
    let lr_1 = scheduler.get_lr(1);
    assert!((lr_1 - 1e-6).abs() < 1e-9);

    // Step 50: should be 50% of peak
    let lr_50 = scheduler.get_lr(50);
    assert!((lr_50 - 5e-5).abs() < 1e-9);

    // Step 99: should be 99% of peak
    let lr_99 = scheduler.get_lr(99);
    assert!((lr_99 - 9.9e-5).abs() < 1e-9);

    // Step 100: should be exactly peak (entering stable)
    assert_eq!(scheduler.get_lr(100), 1e-4);
}

#[test]
fn test_wsd_scheduler_stable_phase() {
    let scheduler = WSDScheduler::builder()
        .total_steps(1000)
        .peak_lr(1e-4)
        .min_lr(1e-6)
        .warmup_fraction(0.1) // 100 steps warmup
        .decay_fraction(0.2) // 200 steps decay
        .build()
        .expect("failed to build scheduler");

    // Stable phase: steps 100-799 (700 steps)
    assert_eq!(scheduler.get_lr(100), 1e-4);
    assert_eq!(scheduler.get_lr(400), 1e-4);
    assert_eq!(scheduler.get_lr(799), 1e-4);
}

#[test]
fn test_wsd_scheduler_decay_phase() {
    let scheduler = WSDScheduler::builder()
        .total_steps(1000)
        .peak_lr(1.0)
        .min_lr(0.0)
        .warmup_fraction(0.0) // No warmup
        .decay_fraction(1.0) // All decay
        .build()
        .expect("failed to build scheduler");

    // Step 0: peak
    assert_eq!(scheduler.get_lr(0), 1.0);

    // Step 500: halfway through cosine (should be 0.5)
    let lr_mid = scheduler.get_lr(500);
    assert!((lr_mid - 0.5).abs() < 0.01);

    // Step 999: should be very close to 0
    let lr_end = scheduler.get_lr(999);
    assert!(lr_end < 0.01);
}

#[test]
fn test_wsd_scheduler_edge_cases() {
    // Zero warmup
    let sched1 = WSDScheduler::builder()
        .total_steps(100)
        .peak_lr(1.0)
        .min_lr(0.0)
        .warmup_fraction(0.0)
        .decay_fraction(0.5)
        .build()
        .expect("failed to build");
    assert_eq!(sched1.get_lr(0), 1.0); // Starts at peak

    // Zero decay
    let sched2 = WSDScheduler::builder()
        .total_steps(100)
        .peak_lr(1.0)
        .min_lr(0.0)
        .warmup_fraction(0.1)
        .decay_fraction(0.0)
        .build()
        .expect("failed to build");
    assert_eq!(sched2.get_lr(99), 1.0); // Ends at peak

    // Single step
    let sched3 = WSDScheduler::builder()
        .total_steps(1)
        .peak_lr(1.0)
        .min_lr(0.0)
        .warmup_fraction(0.0)
        .decay_fraction(0.0)
        .build()
        .expect("failed to build");
    assert_eq!(sched3.get_lr(0), 1.0);
}

#[test]
fn test_wsd_scheduler_phase_boundaries() {
    let scheduler = WSDScheduler::builder()
        .warmup_steps(100)
        .stable_steps(700)
        .decay_steps(200)
        .peak_lr(1.0)
        .min_lr(0.0)
        .build()
        .expect("failed to build");

    // Warmup boundary
    assert_eq!(scheduler.phase_name(99), "warmup");
    assert_eq!(scheduler.phase_name(100), "stable");

    // Stable boundary
    assert_eq!(scheduler.phase_name(799), "stable");
    assert_eq!(scheduler.phase_name(800), "decay");

    // Decay end
    assert_eq!(scheduler.phase_name(999), "decay");
}

// ============================================================================
// Test 5: CurveQuality Detection Accuracy (via lr_advisor)
// ============================================================================

#[test]
fn test_curve_quality_oscillation_detection() {
    // Create oscillating loss curve
    let losses: Vec<f32> = (0..20)
        .map(|i| if i % 2 == 0 { 2.0 } else { 1.0 })
        .collect();
    let gradients = vec![0.5; 20];

    let advice = analyze_lr(&losses, &gradients, 1e-4, 100, LRPhase::Stable);

    assert!(advice.is_some());
    let advice = advice.unwrap();
    assert_eq!(advice.issue, Issue::LossOscillation);
    assert!(matches!(advice.urgency, Urgency::High | Urgency::Critical));
    assert!(advice.suggested_lr < 1e-4); // Should reduce LR
}

#[test]
fn test_curve_quality_plateau_detection() {
    // Create flat loss curve
    let mut losses = vec![2.0; 60];
    // Add tiny noise to avoid exact flatness
    for (i, loss) in losses.iter_mut().enumerate() {
        *loss += (i as f32 * 0.0001).sin() * 0.0001;
    }
    let gradients = vec![0.3; 60];

    let advice = analyze_lr(&losses, &gradients, 1e-4, 1000, LRPhase::Stable);

    assert!(advice.is_some());
    let advice = advice.unwrap();
    assert_eq!(advice.issue, Issue::LossPlateau);
    assert!(advice.suggested_lr > 1e-4); // Should increase LR
}

#[test]
fn test_curve_quality_gradient_explosion() {
    let losses = vec![2.0; 20];
    // Normal baseline, then explosion
    let mut gradients = vec![0.1; 15];
    gradients.extend(vec![2.0, 2.0, 2.0, 2.0, 2.0]); // 20x increase

    let advice = analyze_lr(&losses, &gradients, 1e-4, 100, LRPhase::Stable);

    assert!(advice.is_some());
    let advice = advice.unwrap();
    assert_eq!(advice.issue, Issue::GradientExplosion);
    assert_eq!(advice.urgency, Urgency::Critical);
    assert!(advice.suggested_lr < 1e-4);
}

#[test]
fn test_curve_quality_gradient_vanishing() {
    let losses = vec![2.0; 20];
    // Normal baseline, then vanishing
    let mut gradients = vec![0.1; 15];
    gradients.extend(vec![0.00005, 0.00005, 0.00005, 0.00005, 0.00005]);

    let advice = analyze_lr(&losses, &gradients, 1e-4, 100, LRPhase::Stable);

    assert!(advice.is_some());
    let advice = advice.unwrap();
    assert_eq!(advice.issue, Issue::GradientVanishing);
    assert!(advice.suggested_lr > 1e-4); // Should increase LR
}

#[test]
fn test_curve_quality_healthy_training() {
    // Smooth decreasing loss
    let losses: Vec<f32> = (0..50)
        .map(|i| {
            let progress = i as f32 / 50.0;
            2.5 * (1.0 - 0.2 * progress) + (i as f32 * 0.001).sin() * 0.01
        })
        .collect();
    let gradients = vec![0.3; 50];

    let advice = analyze_lr(&losses, &gradients, 1e-4, 500, LRPhase::Stable);

    // Should return None for healthy training
    assert!(advice.is_none());
}

#[test]
fn test_curve_quality_warmup_tolerance() {
    // Same oscillation, different phases
    let losses = vec![2.0, 1.5, 2.2, 1.4, 2.3, 1.3, 2.4, 1.2, 2.5, 1.1, 2.6, 1.0];
    let gradients = vec![0.3; 12];

    // In warmup: should be more tolerant
    let advice_warmup = analyze_lr(&losses, &gradients, 1e-4, 50, LRPhase::Warmup);

    // In stable: should detect issue
    let advice_stable = analyze_lr(&losses, &gradients, 1e-4, 500, LRPhase::Stable);

    // Warmup should be less urgent or no advice
    if let Some(warmup) = advice_warmup {
        assert!(matches!(warmup.urgency, Urgency::Medium | Urgency::Low));
    }

    // Stable should detect high priority issue
    assert!(advice_stable.is_some());
    if let Some(stable) = advice_stable {
        assert!(matches!(stable.urgency, Urgency::High | Urgency::Critical));
    }
}

// ============================================================================
// Test 6: GeneralizationHealth Detection
// ============================================================================

#[test]
fn test_generalization_health_healthy() {
    let temp_dir = tempfile::tempdir().expect("failed to create temp dir");
    let metrics_file = temp_dir.path().join("metrics_healthy.jsonl");

    // Write healthy training metrics
    let mut file = fs::File::create(&metrics_file).expect("failed to create file");
    for i in 0..20 {
        let mut metric = create_test_metric(i, 3.0 - (i as f32 * 0.05), TrainingPhase::Full, false);
        // Add healthy dynamics
        metric.loss_velocity = -0.05;
        metric.loss_acceleration = 0.0001;
        metric.train_val_gap = Some(0.02); // Small gap
        metric.gradient_entropy = Some(2.5 + (i as f32 * 0.01)); // Stable entropy
        writeln!(file, "{}", serde_json::to_string(&metric).unwrap()).expect("write failed");
    }

    let config = create_test_config("100m");
    let run_dir = temp_dir.path().join("run_healthy");
    fs::create_dir_all(&run_dir).expect("failed to create run dir");

    let mut run = TrainingRun::new("healthy_test", config, run_dir);
    run.metrics_file = metrics_file;

    let health = run.generalization_health();
    assert_eq!(health, GeneralizationHealth::Healthy);
}

#[test]
fn test_generalization_health_overfitting() {
    let temp_dir = tempfile::tempdir().expect("failed to create temp dir");
    let metrics_file = temp_dir.path().join("metrics_overfit.jsonl");

    // Write overfitting metrics (large train-val gap)
    let mut file = fs::File::create(&metrics_file).expect("failed to create file");
    for i in 0..20 {
        let mut metric = create_test_metric(i, 1.5, TrainingPhase::Full, false);
        metric.train_val_gap = Some(0.8); // Large gap
        metric.loss_velocity = -0.001; // Slow improvement
        metric.gradient_entropy = Some(2.5 - (i as f32 * 0.05)); // Decreasing entropy
        writeln!(file, "{}", serde_json::to_string(&metric).unwrap()).expect("write failed");
    }

    let config = create_test_config("100m");
    let run_dir = temp_dir.path().join("run_overfit");
    fs::create_dir_all(&run_dir).expect("failed to create run dir");

    let mut run = TrainingRun::new("overfit_test", config, run_dir);
    run.metrics_file = metrics_file;

    let health = run.generalization_health();
    assert_eq!(health, GeneralizationHealth::Overfitting);
}

#[test]
fn test_generalization_health_underfitting() {
    let temp_dir = tempfile::tempdir().expect("failed to create temp dir");
    let metrics_file = temp_dir.path().join("metrics_underfit.jsonl");

    // Write underfitting metrics (loss increasing)
    let mut file = fs::File::create(&metrics_file).expect("failed to create file");
    for i in 0..20 {
        let mut metric = create_test_metric(i, 3.0 + (i as f32 * 0.01), TrainingPhase::Full, false);
        metric.loss_velocity = 0.01; // Positive velocity
        metric.loss_acceleration = 0.002; // Positive acceleration
        writeln!(file, "{}", serde_json::to_string(&metric).unwrap()).expect("write failed");
    }

    let config = create_test_config("100m");
    let run_dir = temp_dir.path().join("run_underfit");
    fs::create_dir_all(&run_dir).expect("failed to create run dir");

    let mut run = TrainingRun::new("underfit_test", config, run_dir);
    run.metrics_file = metrics_file;

    let health = run.generalization_health();
    assert_eq!(health, GeneralizationHealth::Underfitting);
}

#[test]
fn test_generalization_health_unknown() {
    let temp_dir = tempfile::tempdir().expect("failed to create temp dir");
    let metrics_file = temp_dir.path().join("metrics_unknown.jsonl");

    // Write only 2 metrics (insufficient data)
    let mut file = fs::File::create(&metrics_file).expect("failed to create file");
    for i in 0..2 {
        let metric = create_test_metric(i, 2.5, TrainingPhase::Full, false);
        writeln!(file, "{}", serde_json::to_string(&metric).unwrap()).expect("write failed");
    }

    let config = create_test_config("100m");
    let run_dir = temp_dir.path().join("run_unknown");
    fs::create_dir_all(&run_dir).expect("failed to create run dir");

    let mut run = TrainingRun::new("unknown_test", config, run_dir);
    run.metrics_file = metrics_file;

    let health = run.generalization_health();
    assert_eq!(health, GeneralizationHealth::Unknown);
}

// ============================================================================
// Additional Edge Cases
// ============================================================================

#[test]
fn test_loss_dynamics_empty() {
    let losses: Vec<f32> = vec![];
    let (velocity, acceleration) = calculate_loss_dynamics(&losses, 30);
    assert_eq!(velocity, 0.0);
    assert_eq!(acceleration, 0.0);
}

#[test]
fn test_loss_dynamics_single_value() {
    let losses = vec![2.5];
    let (velocity, acceleration) = calculate_loss_dynamics(&losses, 30);
    assert_eq!(velocity, 0.0);
    assert_eq!(acceleration, 0.0);
}

#[test]
fn test_loss_dynamics_decreasing() {
    let losses: Vec<f32> = (0..50).map(|i| 3.0 - (i as f32 * 0.01)).collect();
    let (velocity, acceleration) = calculate_loss_dynamics(&losses, 30);
    assert!(velocity < 0.0); // Decreasing
}

#[test]
fn test_loss_dynamics_increasing() {
    let losses: Vec<f32> = (0..50).map(|i| 2.0 + (i as f32 * 0.01)).collect();
    let (velocity, acceleration) = calculate_loss_dynamics(&losses, 30);
    assert!(velocity > 0.0); // Increasing
}

#[test]
fn test_config_hash_deterministic() {
    let config1 = create_test_config("100m");
    let config2 = create_test_config("100m");

    let hash1 = compute_config_hash(&config1);
    let hash2 = compute_config_hash(&config2);

    assert_eq!(hash1, hash2);
    assert_eq!(hash1.len(), 12);
}

#[test]
fn test_config_hash_different_configs() {
    let config1 = create_test_config("100m");
    let mut config2 = create_test_config("100m");
    config2.learning_rate = 2e-4; // Different LR

    let hash1 = compute_config_hash(&config1);
    let hash2 = compute_config_hash(&config2);

    assert_ne!(hash1, hash2);
}

#[test]
fn test_training_status_is_active() {
    assert!(TrainingStatus::Initializing.is_active());
    assert!(TrainingStatus::Running.is_active());
    assert!(TrainingStatus::Paused.is_active());
    assert!(!TrainingStatus::Completed.is_active());
    assert!(!TrainingStatus::Failed.is_active());
    assert!(!TrainingStatus::Cancelled.is_active());
}

#[test]
fn test_training_status_is_finished() {
    assert!(!TrainingStatus::Initializing.is_finished());
    assert!(!TrainingStatus::Running.is_finished());
    assert!(!TrainingStatus::Paused.is_finished());
    assert!(TrainingStatus::Completed.is_finished());
    assert!(TrainingStatus::Failed.is_finished());
    assert!(TrainingStatus::Cancelled.is_finished());
}

// ============================================================================
// Test 7: Layer Gradient Stats
// ============================================================================

#[test]
fn test_layer_gradient_stats_from_empty() {
    use std::collections::HashMap;
    use training_tools::LayerGradientStats;

    let empty: HashMap<String, f32> = HashMap::new();
    let stats = LayerGradientStats::from_layer_gradients(&empty);

    // Empty input returns default values (all zeros)
    assert_eq!(stats.max_norm, 0.0);
    assert_eq!(stats.min_norm, 0.0);
    assert_eq!(stats.mean_norm, 0.0);
    assert!(stats.vanishing_layers.is_empty());
    assert!(stats.exploding_layers.is_empty());
    assert!(!stats.has_problems());
}

#[test]
fn test_layer_gradient_stats_normal_gradients() {
    use std::collections::HashMap;
    use training_tools::LayerGradientStats;

    let mut gradients = HashMap::new();
    gradients.insert("layer_0".to_string(), 0.5);
    gradients.insert("layer_1".to_string(), 0.3);
    gradients.insert("layer_2".to_string(), 0.4);

    let stats = LayerGradientStats::from_layer_gradients(&gradients);

    assert!((stats.max_norm - 0.5).abs() < 1e-6);
    assert!((stats.min_norm - 0.3).abs() < 1e-6);
    assert!((stats.mean_norm - 0.4).abs() < 1e-6);
    assert!(stats.vanishing_layers.is_empty());
    assert!(stats.exploding_layers.is_empty());
    assert!(!stats.has_problems());
}

#[test]
fn test_layer_gradient_stats_vanishing_detection() {
    use std::collections::HashMap;
    use training_tools::LayerGradientStats;

    let mut gradients = HashMap::new();
    gradients.insert("layer_0".to_string(), 0.5);
    gradients.insert("layer_1".to_string(), 1e-8); // vanishing
    gradients.insert("layer_2".to_string(), 1e-9); // vanishing

    let stats = LayerGradientStats::from_layer_gradients(&gradients);

    assert_eq!(stats.vanishing_layers.len(), 2);
    assert!(stats.vanishing_layers.contains(&"layer_1".to_string()));
    assert!(stats.vanishing_layers.contains(&"layer_2".to_string()));
    assert!(stats.exploding_layers.is_empty());
    assert!(stats.has_problems());
}

#[test]
fn test_layer_gradient_stats_exploding_detection() {
    use std::collections::HashMap;
    use training_tools::LayerGradientStats;

    let mut gradients = HashMap::new();
    gradients.insert("layer_0".to_string(), 0.5);
    gradients.insert("layer_1".to_string(), 150.0); // exploding
    gradients.insert("layer_2".to_string(), 200.0); // exploding

    let stats = LayerGradientStats::from_layer_gradients(&gradients);

    assert!(stats.vanishing_layers.is_empty());
    assert_eq!(stats.exploding_layers.len(), 2);
    assert!(stats.exploding_layers.contains(&"layer_1".to_string()));
    assert!(stats.exploding_layers.contains(&"layer_2".to_string()));
    assert!(stats.has_problems());
}

#[test]
fn test_layer_gradient_stats_mixed_problems() {
    use std::collections::HashMap;
    use training_tools::LayerGradientStats;

    let mut gradients = HashMap::new();
    gradients.insert("layer_0".to_string(), 1e-8); // vanishing
    gradients.insert("layer_1".to_string(), 0.5); // normal
    gradients.insert("layer_2".to_string(), 150.0); // exploding

    let stats = LayerGradientStats::from_layer_gradients(&gradients);

    assert_eq!(stats.vanishing_layers.len(), 1);
    assert_eq!(stats.exploding_layers.len(), 1);
    assert!(stats.has_problems());
}

#[test]
fn test_layer_gradient_stats_gradient_spread() {
    use std::collections::HashMap;
    use training_tools::LayerGradientStats;

    let mut gradients = HashMap::new();
    gradients.insert("layer_0".to_string(), 0.1);
    gradients.insert("layer_1".to_string(), 1.0);

    let stats = LayerGradientStats::from_layer_gradients(&gradients);

    let spread = stats.gradient_spread().unwrap();
    assert!((spread - 10.0).abs() < 1e-6);

    // Test with zero min
    let mut zero_gradients = HashMap::new();
    zero_gradients.insert("layer_0".to_string(), 0.0);
    zero_gradients.insert("layer_1".to_string(), 1.0);

    let zero_stats = LayerGradientStats::from_layer_gradients(&zero_gradients);
    assert!(zero_stats.gradient_spread().is_none());
}

// ============================================================================
// Test 8: StepMetrics with Layer Gradients Serialization
// ============================================================================

#[test]
fn test_step_metrics_layer_gradients_roundtrip() {
    use std::collections::HashMap;
    use training_tools::LayerGradientStats;

    let mut layer_gradients = HashMap::new();
    layer_gradients.insert("transformer.layer_0.attention".to_string(), 0.5);
    layer_gradients.insert("transformer.layer_1.attention".to_string(), 0.3);
    layer_gradients.insert("transformer.layer_2.ffn".to_string(), 1e-8); // vanishing

    let layer_stats = LayerGradientStats::from_layer_gradients(&layer_gradients);

    let original = StepMetrics {
        step: 100,
        loss: 2.0,
        gradient_norm: 0.4,
        phase: TrainingPhase::Full,
        was_predicted: false,
        prediction_error: None,
        step_time_ms: 100.0,
        timestamp: Utc::now(),
        tokens_this_step: 4096,
        total_tokens_trained: 409600,
        tokens_remaining: 1_000_000,
        confidence: 0.9,
        learning_rate: 1e-4,
        perplexity: 7.39,
        train_val_gap: None,
        loss_velocity: 0.0,
        loss_acceleration: 0.0,
        gradient_entropy: None,
        layer_gradients: Some(layer_gradients.clone()),
        layer_gradient_stats: Some(layer_stats),
    };

    // Serialize
    let json = serde_json::to_string(&original).expect("serialization failed");

    // Deserialize
    let deserialized: StepMetrics = serde_json::from_str(&json).expect("deserialization failed");

    // Verify layer_gradients
    let deser_gradients = deserialized
        .layer_gradients
        .expect("layer_gradients should be present");
    assert_eq!(deser_gradients.len(), 3);
    assert!((deser_gradients["transformer.layer_0.attention"] - 0.5).abs() < 1e-6);

    // Verify layer_gradient_stats
    let deser_stats = deserialized
        .layer_gradient_stats
        .expect("layer_gradient_stats should be present");
    assert_eq!(deser_stats.vanishing_layers.len(), 1);
    assert!(deser_stats
        .vanishing_layers
        .contains(&"transformer.layer_2.ffn".to_string()));
}

#[test]
fn test_step_metrics_backward_compatible_deserialization() {
    // Test that old JSON (without layer_gradients fields) deserializes correctly
    let old_json = r#"{
        "step": 50,
        "loss": 2.5,
        "gradient_norm": 0.5,
        "phase": "full",
        "was_predicted": false,
        "prediction_error": null,
        "step_time_ms": 100.0,
        "timestamp": "2024-01-01T00:00:00Z",
        "tokens_this_step": 4096,
        "total_tokens_trained": 204800,
        "tokens_remaining": 500000,
        "confidence": 0.9,
        "learning_rate": 0.0001,
        "perplexity": 12.18
    }"#;

    let deserialized: StepMetrics =
        serde_json::from_str(old_json).expect("backward compatible deserialization failed");

    assert_eq!(deserialized.step, 50);
    assert!(deserialized.layer_gradients.is_none());
    assert!(deserialized.layer_gradient_stats.is_none());
}

// ============================================================================
// Test 9: TrainingRun Layer Gradient History and Problematic Layers
// ============================================================================

#[test]
fn test_training_run_layer_gradient_history() {
    use std::collections::HashMap;

    let temp_dir = tempfile::tempdir().expect("failed to create temp dir");
    let metrics_file = temp_dir.path().join("metrics_layer_grad.jsonl");

    // Write metrics with layer gradients
    let mut file = fs::File::create(&metrics_file).expect("failed to create file");
    for i in 0..10 {
        let mut layer_gradients = HashMap::new();
        layer_gradients.insert("layer_0".to_string(), 0.5 + (i as f32 * 0.01));
        layer_gradients.insert("layer_1".to_string(), 0.3 + (i as f32 * 0.005));

        let metric =
            create_test_metric_with_layer_gradients(i, 2.5 - (i as f32 * 0.1), layer_gradients);
        let json = serde_json::to_string(&metric).expect("serialization failed");
        writeln!(file, "{}", json).expect("write failed");
    }

    let config = create_test_config("100m");
    let run_dir = temp_dir.path().join("run_layer_grad");
    fs::create_dir_all(&run_dir).expect("failed to create run dir");

    let mut run = TrainingRun::new("layer_grad_test", config, run_dir);
    run.metrics_file = metrics_file;

    // Test layer gradient history
    let layer_0_history = run.layer_gradient_history("layer_0");
    assert_eq!(layer_0_history.len(), 10);
    assert!((layer_0_history[0] - 0.5).abs() < 1e-6);
    assert!((layer_0_history[9] - 0.59).abs() < 1e-6);

    let layer_1_history = run.layer_gradient_history("layer_1");
    assert_eq!(layer_1_history.len(), 10);

    // Non-existent layer should return empty
    let nonexistent_history = run.layer_gradient_history("nonexistent_layer");
    assert!(nonexistent_history.is_empty());
}

#[test]
fn test_training_run_problematic_layers() {
    use std::collections::HashMap;

    let temp_dir = tempfile::tempdir().expect("failed to create temp dir");
    let metrics_file = temp_dir.path().join("metrics_problematic.jsonl");

    // Write metrics with some problematic gradients
    let mut file = fs::File::create(&metrics_file).expect("failed to create file");

    // Step 0: normal gradients
    let mut grads_0 = HashMap::new();
    grads_0.insert("layer_0".to_string(), 0.5);
    grads_0.insert("layer_1".to_string(), 0.4);
    grads_0.insert("layer_2".to_string(), 0.3);
    let metric_0 = create_test_metric_with_layer_gradients(0, 2.5, grads_0);
    writeln!(file, "{}", serde_json::to_string(&metric_0).unwrap()).expect("write failed");

    // Step 1: layer_1 vanishes
    let mut grads_1 = HashMap::new();
    grads_1.insert("layer_0".to_string(), 0.5);
    grads_1.insert("layer_1".to_string(), 1e-8); // vanishing
    grads_1.insert("layer_2".to_string(), 0.3);
    let metric_1 = create_test_metric_with_layer_gradients(1, 2.4, grads_1);
    writeln!(file, "{}", serde_json::to_string(&metric_1).unwrap()).expect("write failed");

    // Step 2: layer_2 explodes
    let mut grads_2 = HashMap::new();
    grads_2.insert("layer_0".to_string(), 0.5);
    grads_2.insert("layer_1".to_string(), 0.4);
    grads_2.insert("layer_2".to_string(), 150.0); // exploding
    let metric_2 = create_test_metric_with_layer_gradients(2, 2.3, grads_2);
    writeln!(file, "{}", serde_json::to_string(&metric_2).unwrap()).expect("write failed");

    let config = create_test_config("100m");
    let run_dir = temp_dir.path().join("run_problematic");
    fs::create_dir_all(&run_dir).expect("failed to create run dir");

    let mut run = TrainingRun::new("problematic_test", config, run_dir);
    run.metrics_file = metrics_file;

    // Test problematic layers detection
    let (vanishing, exploding) = run.problematic_layers();

    assert_eq!(vanishing.len(), 1);
    assert!(vanishing.contains(&"layer_1".to_string()));

    assert_eq!(exploding.len(), 1);
    assert!(exploding.contains(&"layer_2".to_string()));
}

#[test]
fn test_training_run_problematic_layers_empty() {
    let temp_dir = tempfile::tempdir().expect("failed to create temp dir");
    let metrics_file = temp_dir.path().join("metrics_empty_prob.jsonl");

    // Write metrics without layer gradients
    let mut file = fs::File::create(&metrics_file).expect("failed to create file");
    for i in 0..5 {
        let metric = create_test_metric(i, 2.5, TrainingPhase::Full, false);
        writeln!(file, "{}", serde_json::to_string(&metric).unwrap()).expect("write failed");
    }

    let config = create_test_config("100m");
    let run_dir = temp_dir.path().join("run_empty_prob");
    fs::create_dir_all(&run_dir).expect("failed to create run dir");

    let mut run = TrainingRun::new("empty_prob_test", config, run_dir);
    run.metrics_file = metrics_file;

    let (vanishing, exploding) = run.problematic_layers();
    assert!(vanishing.is_empty());
    assert!(exploding.is_empty());
}

#[test]
fn test_training_run_layer_gradient_history_empty_file() {
    let temp_dir = tempfile::tempdir().expect("failed to create temp dir");
    let metrics_file = temp_dir.path().join("metrics_empty_hist.jsonl");
    fs::write(&metrics_file, "").expect("failed to write empty file");

    let config = create_test_config("100m");
    let run_dir = temp_dir.path().join("run_empty_hist");
    fs::create_dir_all(&run_dir).expect("failed to create run dir");

    let mut run = TrainingRun::new("empty_hist_test", config, run_dir);
    run.metrics_file = metrics_file;

    let history = run.layer_gradient_history("any_layer");
    assert!(history.is_empty());
}

// ============================================================================
// Helper Functions
// ============================================================================

fn create_test_config(model_size: &str) -> TrainingConfig {
    let config = TrainingConfig {
        config_version: 1,
        config_hash: String::new(),
        model_size: model_size.to_string(),
        num_parameters: 100_000_000,
        hidden_size: 768,
        num_layers: 12,
        num_heads: 12,
        max_seq_length: 512,
        batch_size: 8,
        learning_rate: 1e-4,
        max_steps: 10000,
        gradient_checkpointing: true,
        checkpoint_interval: 4,
        device: "cpu".to_string(),
    };

    let mut config_with_hash = config.clone();
    config_with_hash.config_hash = compute_config_hash(&config);
    config_with_hash
}

fn create_test_metric(
    step: u64,
    loss: f32,
    phase: TrainingPhase,
    was_predicted: bool,
) -> StepMetrics {
    StepMetrics {
        step,
        loss,
        gradient_norm: 0.5,
        phase,
        was_predicted,
        prediction_error: if was_predicted { Some(0.01) } else { None },
        step_time_ms: 100.0,
        timestamp: Utc::now(),
        tokens_this_step: 4096,
        total_tokens_trained: step * 4096,
        tokens_remaining: (10000 - step) * 4096,
        confidence: 0.9,
        learning_rate: 1e-4,
        perplexity: loss.exp(),
        train_val_gap: None,
        loss_velocity: 0.0,
        loss_acceleration: 0.0,
        gradient_entropy: None,
        layer_gradients: None,
        layer_gradient_stats: None,
    }
}

fn create_test_metric_with_layer_gradients(
    step: u64,
    loss: f32,
    layer_gradients: std::collections::HashMap<String, f32>,
) -> StepMetrics {
    let layer_gradient_stats =
        training_tools::LayerGradientStats::from_layer_gradients(&layer_gradients);
    StepMetrics {
        step,
        loss,
        gradient_norm: 0.5,
        phase: TrainingPhase::Full,
        was_predicted: false,
        prediction_error: None,
        step_time_ms: 100.0,
        timestamp: Utc::now(),
        tokens_this_step: 4096,
        total_tokens_trained: step * 4096,
        tokens_remaining: (10000 - step) * 4096,
        confidence: 0.9,
        learning_rate: 1e-4,
        perplexity: loss.exp(),
        train_val_gap: None,
        loss_velocity: 0.0,
        loss_acceleration: 0.0,
        gradient_entropy: None,
        layer_gradients: Some(layer_gradients),
        layer_gradient_stats: Some(layer_gradient_stats),
    }
}
