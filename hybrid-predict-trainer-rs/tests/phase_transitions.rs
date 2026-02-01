//! Phase transition integration tests

use hybrid_predict_trainer_rs::prelude::*;

#[test]
fn test_phase_ordering() {
    // Warmup -> Full -> Predict -> Correct cycle
    let phases = [Phase::Warmup, Phase::Full, Phase::Predict, Phase::Correct];

    for (i, phase) in phases.iter().enumerate() {
        match i {
            0 => {
                assert!(phase.computes_backward());
                assert!(!phase.uses_predictions());
                assert_eq!(phase.name(), "warmup");
            }
            1 => {
                assert_eq!(*phase, Phase::Full);
                assert!(phase.computes_backward());
                assert_eq!(phase.name(), "full");
            }
            2 => {
                assert_eq!(*phase, Phase::Predict);
                assert!(phase.uses_predictions());
                assert!(!phase.computes_backward());
                assert_eq!(phase.name(), "predict");
            }
            3 => {
                assert_eq!(*phase, Phase::Correct);
                assert!(!phase.computes_backward());
                assert!(!phase.uses_predictions());
                assert_eq!(phase.name(), "correct");
            }
            _ => unreachable!(),
        }
    }
}

#[test]
fn test_phase_properties() {
    // Test Warmup phase
    assert!(Phase::Warmup.computes_backward());
    assert!(!Phase::Warmup.uses_predictions());

    // Test Full phase
    assert!(Phase::Full.computes_backward());
    assert!(!Phase::Full.uses_predictions());

    // Test Predict phase
    assert!(!Phase::Predict.computes_backward());
    assert!(Phase::Predict.uses_predictions());

    // Test Correct phase
    assert!(!Phase::Correct.computes_backward());
    assert!(!Phase::Correct.uses_predictions());
}

#[test]
fn test_training_state_initialization() {
    let state = TrainingState::new();
    assert_eq!(state.step, 0);
    assert_eq!(state.current_phase, Phase::Warmup);
    assert_eq!(state.phase_step, 0);
}

#[test]
fn test_training_state_step_recording() {
    let mut state = TrainingState::new();

    // Record a few steps
    state.record_step(2.5, 1.0);
    assert_eq!(state.step, 1);
    assert!((state.loss - 2.5).abs() < f32::EPSILON);
    assert!((state.gradient_norm - 1.0).abs() < f32::EPSILON);

    state.record_step(2.3, 0.9);
    assert_eq!(state.step, 2);
    assert!((state.loss - 2.3).abs() < f32::EPSILON);
}

#[test]
fn test_phase_transition_tracking() {
    let mut state = TrainingState::new();

    // Start in warmup
    assert_eq!(state.current_phase, Phase::Warmup);
    assert_eq!(state.phase_step, 0);

    // Record some steps
    for _ in 0..5 {
        state.record_step(2.0, 1.0);
    }
    assert_eq!(state.phase_step, 5);

    // Transition to Full phase
    state.enter_phase(Phase::Full);
    assert_eq!(state.current_phase, Phase::Full);
    assert_eq!(state.phase_step, 0);

    // Record more steps
    state.record_step(1.8, 0.9);
    assert_eq!(state.phase_step, 1);
}

#[test]
fn test_phase_decision_types() {
    // Test Warmup decision
    let warmup_decision = PhaseDecision::Warmup { steps: 100 };
    assert_eq!(warmup_decision.phase(), Phase::Warmup);
    assert_eq!(warmup_decision.steps(), 100);

    // Test Full decision
    let full_decision = PhaseDecision::Full { steps: 20 };
    assert_eq!(full_decision.phase(), Phase::Full);
    assert_eq!(full_decision.steps(), 20);

    // Test Predict decision
    let predict_decision = PhaseDecision::Predict {
        steps: 50,
        confidence: 0.9,
        fallback_threshold: 3.0,
    };
    assert_eq!(predict_decision.phase(), Phase::Predict);
    assert_eq!(predict_decision.steps(), 50);

    // Test Correct decision
    let correct_decision = PhaseDecision::Correct {
        validation_samples: 5,
        max_correction_magnitude: 0.1,
    };
    assert_eq!(correct_decision.phase(), Phase::Correct);
    assert_eq!(correct_decision.steps(), 5);
}

#[test]
fn test_loss_statistics() {
    let mut state = TrainingState::new();

    // Record multiple loss values
    let losses = vec![2.5, 2.3, 2.1, 2.0, 1.9];
    for &loss in &losses {
        state.record_step(loss, 1.0);
    }

    // Get statistics
    let stats = state.loss_statistics();
    assert!(stats.mean > 0.0);
    assert!(stats.std >= 0.0);
    assert!(stats.min <= stats.max);
}

#[test]
fn test_gradient_statistics() {
    let mut state = TrainingState::new();

    // Record multiple gradient values
    let grads = vec![1.5, 1.3, 1.1, 1.0, 0.9];
    for &grad in &grads {
        state.record_step(2.0, grad);
    }

    // Get statistics
    let stats = state.gradient_statistics();
    assert!(stats.mean > 0.0);
    assert!(stats.std >= 0.0);
}

#[test]
fn test_loss_within_bounds() {
    let mut state = TrainingState::new();

    // Record several similar losses
    for _ in 0..10 {
        state.record_step(2.0, 1.0);
    }

    // Current loss should be within bounds
    assert!(state.loss_within_bounds(3.0));

    // Set an outlier loss
    state.loss = 10.0;

    // Should now be out of bounds
    assert!(!state.loss_within_bounds(2.0));
}

#[test]
fn test_feature_computation() {
    let mut state = TrainingState::new();

    // Record some training history
    for i in 0..50 {
        state.record_step(2.5 - i as f32 * 0.01, 1.0);
    }

    // Compute features
    let features = state.compute_features();

    // Should return fixed-size feature vector
    assert_eq!(features.len(), 64);

    // Features should be finite
    for &f in &features {
        assert!(f.is_finite());
    }
}
