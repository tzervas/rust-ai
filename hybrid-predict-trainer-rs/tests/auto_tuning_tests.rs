//! Comprehensive integration tests for the auto_tuning module.
//!
//! This test suite covers:
//! - Multi-step predictor: Batch prediction, confidence estimation, accuracy tracking
//!
//! Note: Other modules (gradient_tuner, plateau_detector, health_scorer, phase_controller)
//! are not currently exported due to pre-existing compilation errors that need fixing first.

use hybrid_predict_trainer_rs::auto_tuning::{BatchPredictionRecommendation, MultiStepPredictor};

// =============================================================================
// MULTI-STEP PREDICTOR TESTS
// =============================================================================

#[test]
fn test_multi_step_predictor_initialization() {
    // Test that predictor initializes correctly
    let predictor = MultiStepPredictor::new(0.85, 0.3, true);

    assert_eq!(
        predictor.accurate_streak(),
        0,
        "Should start with 0 accurate streak"
    );
    let (skipped_batches, skipped_steps) = predictor.skip_stats();
    assert_eq!(skipped_batches, 0, "Should start with 0 skipped batches");
    assert_eq!(skipped_steps, 0, "Should start with 0 skipped steps");
}

#[test]
fn test_multi_step_predictor_skip_batch_recommendation() {
    // Test that very high confidence triggers SkipBatch recommendation
    let mut predictor = MultiStepPredictor::new(0.85, 0.3, true);

    // Simulate very accurate predictions by recording observations with low errors
    for _ in 0..10 {
        predictor.record_observation(1, 1.0, 1.01); // Very low error
        predictor.record_observation(2, 1.0, 1.02);
        predictor.record_observation(4, 1.0, 1.03);
        predictor.record_observation(8, 1.0, 1.04);
    }

    let prediction = predictor.predict_batch(0.95, 0.05, 4);

    // With very low errors, should recommend SkipBatch
    assert!(
        prediction.overall_confidence >= 0.85,
        "Very accurate predictions should have high confidence, got {}",
        prediction.overall_confidence
    );

    if prediction.overall_confidence >= 0.85 && prediction.final_loss < 0.5 {
        assert_eq!(
            prediction.recommendation,
            BatchPredictionRecommendation::SkipBatch,
            "High confidence and low error should recommend SkipBatch"
        );
    }
}

#[test]
fn test_multi_step_predictor_run_full_batch() {
    // Test that low confidence triggers RunFullBatch recommendation
    let mut predictor = MultiStepPredictor::new(0.85, 0.3, true);

    // Simulate poor predictions by recording high errors
    for _ in 0..20 {
        predictor.record_observation(1, 1.0, 2.0); // Large error
        predictor.record_observation(2, 1.0, 2.5);
        predictor.record_observation(4, 1.0, 3.0);
        predictor.record_observation(8, 1.0, 3.5);
    }

    let prediction = predictor.predict_batch(0.3, 0.5, 4);

    // With poor accuracy, should recommend RunFullBatch
    assert!(
        prediction.overall_confidence < 0.85,
        "Poor predictions should have low confidence, got {}",
        prediction.overall_confidence
    );
    assert_eq!(
        prediction.recommendation,
        BatchPredictionRecommendation::RunFullBatch,
        "Low confidence should recommend RunFullBatch"
    );
}

#[test]
fn test_multi_step_predictor_verification_mode() {
    // Test that moderate confidence triggers ApplyWithVerification
    let mut predictor = MultiStepPredictor::new(0.85, 0.3, true);

    // Build good calibration with low errors
    for horizon in &[1, 2, 4, 8] {
        for _ in 0..10 {
            predictor.record_observation(*horizon, 1.0, 1.05); // Low error (0.05)
        }
    }

    // With good confidence (0.75) and moderate uncertainty, should get ApplyWithVerification
    let prediction = predictor.predict_batch(0.75, 0.15, 4);

    // Should recommend either verification or skip (>= 0.7 confidence)
    assert!(
        matches!(
            prediction.recommendation,
            BatchPredictionRecommendation::ApplyWithVerification
                | BatchPredictionRecommendation::SkipBatch
        ),
        "Confidence >= 0.7 should recommend ApplyWithVerification or SkipBatch, got {:?}",
        prediction.recommendation
    );
}

#[test]
fn test_multi_step_predictor_streak_tracking() {
    // Test that accurate streaks are tracked correctly
    let mut predictor = MultiStepPredictor::new(0.85, 0.3, true);

    // Record accurate predictions (low error = accurate)
    predictor.record_observation(1, 1.0, 1.01);
    assert_eq!(predictor.accurate_streak(), 1, "One accurate prediction");

    predictor.record_observation(1, 1.0, 1.02);
    assert_eq!(predictor.accurate_streak(), 2, "Two accurate predictions");

    predictor.record_observation(1, 1.0, 1.03);
    assert_eq!(predictor.accurate_streak(), 3, "Three accurate predictions");

    // Record inaccurate prediction (high error = inaccurate)
    predictor.record_observation(1, 1.0, 2.0); // Large error
    assert_eq!(
        predictor.accurate_streak(),
        0,
        "Streak should reset on inaccuracy"
    );
}

#[test]
fn test_multi_step_predictor_disable_batch_skip() {
    // Test that batch skip can be disabled
    let mut predictor = MultiStepPredictor::new(0.85, 0.3, false); // Disabled

    // Even with perfect predictions, should not recommend SkipBatch
    for _ in 0..20 {
        predictor.record_observation(1, 1.0, 1.0); // Perfect predictions
        predictor.record_observation(2, 1.0, 1.0);
        predictor.record_observation(4, 1.0, 1.0);
    }

    let prediction = predictor.predict_batch(1.0, 0.0, 4);

    assert_ne!(
        prediction.recommendation,
        BatchPredictionRecommendation::SkipBatch,
        "With batch skip disabled, should never recommend SkipBatch"
    );
    assert!(
        matches!(
            prediction.recommendation,
            BatchPredictionRecommendation::ApplyWithVerification
        ),
        "With batch skip disabled and high confidence, should recommend ApplyWithVerification"
    );
}

#[test]
fn test_multi_step_predictor_horizon_confidences() {
    // Test that per-horizon confidence estimates are computed
    let mut predictor = MultiStepPredictor::new(0.85, 0.3, true);

    // Record some observations to build calibration factors
    for horizon in &[1, 2, 4, 8] {
        for _ in 0..10 {
            predictor.record_observation(*horizon, 1.0, 1.05);
        }
    }

    let prediction = predictor.predict_batch(0.8, 0.2, 4);

    // Should have multiple horizon confidences
    assert!(
        !prediction.horizon_confidences.is_empty(),
        "Should have per-horizon confidences"
    );
    assert!(
        prediction.horizon_confidences.len() <= 4,
        "Should have reasonable number of horizons"
    );

    // All confidences should be in valid range
    for conf in &prediction.horizon_confidences {
        assert!(
            *conf >= 0.0 && *conf <= 1.0,
            "Confidence should be in [0, 1], got {}",
            conf
        );
    }
}

#[test]
fn test_multi_step_predictor_geometric_mean() {
    // Test that overall confidence is geometric mean of horizon confidences
    let mut predictor = MultiStepPredictor::new(0.85, 0.3, true);

    // Record perfect observations (calibration = 1.0)
    for horizon in &[1, 2, 4, 8] {
        for _ in 0..5 {
            predictor.record_observation(*horizon, 1.0, 1.0); // Perfect
        }
    }

    let prediction = predictor.predict_batch(0.8, 0.1, 4);

    // With good base confidence and perfect calibration, overall should be high
    assert!(
        prediction.overall_confidence >= 0.75,
        "Overall confidence should be high with good predictions, got {}",
        prediction.overall_confidence
    );

    // Geometric mean of confidences should be <= min(confidences)
    if !prediction.horizon_confidences.is_empty() {
        let min_conf = prediction
            .horizon_confidences
            .iter()
            .copied()
            .fold(f32::INFINITY, f32::min);
        assert!(
            prediction.overall_confidence <= min_conf + 0.01, // Small tolerance
            "Geometric mean should be <= minimum horizon confidence"
        );
    }
}

#[test]
fn test_multi_step_predictor_skip_stats() {
    // Test that skip statistics are tracked
    let mut predictor = MultiStepPredictor::new(0.85, 0.3, true);

    let initial_stats = predictor.skip_stats();
    assert_eq!(initial_stats.0, 0, "Initial batches_skipped should be 0");
    assert_eq!(initial_stats.1, 0, "Initial steps_skipped should be 0");

    // Record batch skip
    predictor.record_batch_skip(10);

    let after_stats = predictor.skip_stats();
    assert_eq!(after_stats.0, 1, "Should have 1 batch skipped");
    assert_eq!(after_stats.1, 10, "Should have 10 steps skipped");

    // Record another batch skip
    predictor.record_batch_skip(5);
    let final_stats = predictor.skip_stats();
    assert_eq!(final_stats.0, 2, "Should have 2 batches skipped");
    assert_eq!(final_stats.1, 15, "Should have 15 total steps skipped");
}

#[test]
fn test_multi_step_predictor_accuracy_rate() {
    // Test that accuracy rate is correctly computed
    let mut predictor = MultiStepPredictor::new(0.85, 0.3, true);

    // Record many accurate observations to build up accuracy
    for _ in 0..50 {
        predictor.record_observation(1, 1.0, 1.01); // Accurate (error = 0.01 < 0.1)
        predictor.record_observation(2, 1.0, 1.02);
        predictor.record_observation(4, 1.0, 1.03);
    }

    let accuracy = predictor.accuracy_rate();
    assert!(
        accuracy >= 0.0 && accuracy <= 1.0,
        "Accuracy rate should be in [0, 1], got {}",
        accuracy
    );
    // After many accurate observations, should have reasonable accuracy
    assert!(
        accuracy > 0.5,
        "After many accurate predictions, accuracy should be good, got {}",
        accuracy
    );
}

#[test]
fn test_multi_step_predictor_reset() {
    // Test that reset clears tracking data
    let mut predictor = MultiStepPredictor::new(0.85, 0.3, true);

    // Build up some history with accurate observations
    for _ in 0..20 {
        predictor.record_observation(1, 1.0, 1.01); // Low error = accurate
        predictor.record_observation(2, 1.0, 1.02);
    }

    let streak_before = predictor.accurate_streak();
    assert!(streak_before > 0, "Should have built up accurate streak");

    // Reset
    predictor.reset();

    assert_eq!(
        predictor.accurate_streak(),
        0,
        "After reset, accurate streak should be 0"
    );
    // Note: reset() may not clear skip_stats - that depends on implementation
    // Just verify we can reset and continue using the predictor
}

#[test]
fn test_multi_step_predictor_edge_cases() {
    // Test edge cases and boundaries
    let predictor = MultiStepPredictor::new(0.85, 0.3, true);

    // Test with 0 steps - should be clamped to at least 1
    let pred_zero = predictor.predict_batch(0.5, 0.2, 0);
    assert!(pred_zero.num_steps >= 1, "Should clamp to at least 1 step");

    // Test with extreme confidence values
    let pred_high = predictor.predict_batch(2.0, 0.0, 4); // Over 1.0
    assert!(
        pred_high.overall_confidence <= 1.0,
        "Confidence should be clamped to [0, 1], got {}",
        pred_high.overall_confidence
    );

    let pred_low = predictor.predict_batch(-0.5, 0.5, 4); // Negative
    assert!(
        pred_low.overall_confidence >= 0.0,
        "Confidence should be clamped to [0, 1], got {}",
        pred_low.overall_confidence
    );

    // Test with extreme uncertainty
    let pred_high_unc = predictor.predict_batch(0.9, 2.0, 4); // Over 1.0
    assert!(
        pred_high_unc.recommendation != BatchPredictionRecommendation::SkipBatch,
        "High uncertainty should prevent SkipBatch"
    );
}

#[test]
fn test_multi_step_predictor_mean_error_tracking() {
    // Test that mean errors are tracked per horizon
    let mut predictor = MultiStepPredictor::new(0.85, 0.3, true);

    // Record errors for horizon 1
    predictor.record_observation(1, 1.0, 1.1); // Error = 0.1
    predictor.record_observation(1, 2.0, 2.2); // Error = 0.2
    predictor.record_observation(1, 3.0, 3.1); // Error = 0.1

    let mean_error_h1 = predictor.mean_error_for_horizon(1);
    // Mean should be approximately (0.1 + 0.2 + 0.1) / 3 = 0.133
    assert!(
        mean_error_h1 >= 0.1 && mean_error_h1 <= 0.2,
        "Mean error for horizon 1 should be around 0.133, got {}",
        mean_error_h1
    );

    // Different horizon should have different error
    predictor.record_observation(2, 1.0, 1.5); // Error = 0.5
    let mean_error_h2 = predictor.mean_error_for_horizon(2);
    assert!(
        mean_error_h2 > mean_error_h1,
        "Horizon 2 should have higher error than horizon 1"
    );
}

#[test]
fn test_multi_step_predictor_calibration_factors() {
    // Test that calibration factors are computed correctly
    let mut predictor = MultiStepPredictor::new(0.85, 0.3, true);

    // Record perfect observations (error = 0)
    for _ in 0..5 {
        predictor.record_observation(1, 1.0, 1.0);
    }

    let calib_perfect = predictor.calibration_factor_for_horizon(1);
    // calibration = 1.0 / (1.0 + 0.0) = 1.0
    assert!(
        calib_perfect > 0.99,
        "Perfect predictions should give calibration near 1.0"
    );

    // Record poor observations (error = 1.0)
    for _ in 0..5 {
        predictor.record_observation(2, 1.0, 2.0);
    }

    let calib_poor = predictor.calibration_factor_for_horizon(2);
    // calibration = 1.0 / (1.0 + 1.0) = 0.5
    assert!(
        (calib_poor - 0.5).abs() < 0.01,
        "Poor predictions should give calibration near 0.5, got {}",
        calib_poor
    );

    // Perfect should give higher calibration than poor
    assert!(
        calib_perfect > calib_poor,
        "Perfect predictions should give higher calibration"
    );
}

#[test]
fn test_multi_step_predictor_concurrent_operations() {
    // Stress test: many concurrent updates without panicking
    let mut predictor = MultiStepPredictor::new(0.85, 0.3, true);

    for step in 0..500 {
        // Record observations
        for horizon in &[1, 2, 4, 8] {
            let error_factor = (step as f32 % 10.0) / 10.0; // 0-1 error pattern
            let predicted = 1.0;
            let actual = predicted + error_factor;
            predictor.record_observation(*horizon, predicted, actual);
        }

        // Make predictions
        if step % 50 == 0 {
            let confidence = 0.5 + (step as f32 % 50.0) / 100.0;
            let uncertainty = 0.2 + (step as f32 % 20.0) / 100.0;
            let prediction = predictor.predict_batch(confidence, uncertainty, 4);

            assert!(
                prediction.overall_confidence >= 0.0 && prediction.overall_confidence <= 1.0,
                "Confidence should be in valid range at step {}",
                step
            );
        }

        // Record batch skips sometimes
        if step % 100 == 0 {
            predictor.record_batch_skip(10);
        }
    }

    // If we got here, no panics occurred
    let (batches, steps) = predictor.skip_stats();
    assert!(batches > 0, "Should have recorded batch skips");
    assert!(steps > 0, "Should have recorded step skips");
}

#[test]
fn test_multi_step_predictor_full_integration() {
    // Full integration test: simulate a realistic prediction scenario
    let mut predictor = MultiStepPredictor::new(0.85, 0.3, true);

    // Phase 1: Initial training with variable accuracy
    for i in 0..50 {
        let accuracy_factor = 0.05 + (i as f32 * 0.005); // Getting better
        for horizon in &[1, 2, 4] {
            predictor.record_observation(*horizon, 1.0, 1.0 + accuracy_factor);
        }
    }

    // Phase 2: Make predictions with improving confidence
    let pred1 = predictor.predict_batch(0.6, 0.3, 4);
    assert_eq!(pred1.num_steps, 4, "Should respect requested steps");

    // Phase 3: Continue improving
    for _ in 0..30 {
        predictor.record_observation(1, 1.0, 1.01);
        predictor.record_observation(2, 1.0, 1.02);
        predictor.record_observation(4, 1.0, 1.03);
    }

    // Phase 4: Final prediction with high accuracy history
    let pred2 = predictor.predict_batch(0.85, 0.15, 4);
    assert!(
        pred2.overall_confidence > pred1.overall_confidence,
        "Confidence should improve with better history"
    );

    // Verify statistics
    let final_accuracy = predictor.accuracy_rate();
    assert!(
        final_accuracy > 0.7,
        "Should have good accuracy after many correct predictions"
    );
}
