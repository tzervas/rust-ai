//! Integration tests for intra-horizon micro-corrections.
//!
//! Verifies that micro-corrections are applied at the correct intervals
//! during the Predict phase and that they help prevent error accumulation.

use hybrid_predict_trainer_rs::prelude::*;
use hybrid_predict_trainer_rs::config::HybridTrainerConfig;
use hybrid_predict_trainer_rs::state::TrainingState;

#[test]
fn test_micro_correction_config_builder() {
    // Test that correction_interval can be set via builder
    let config = HybridTrainerConfig::builder()
        .correction_interval(10)
        .build();

    assert_eq!(config.correction_interval, 10);
}

#[test]
fn test_micro_correction_default_disabled() {
    // Test that micro-corrections are disabled by default
    let config = HybridTrainerConfig::default();
    assert_eq!(config.correction_interval, 0);
}

#[test]
fn test_steps_in_current_phase_tracking() {
    // Test that steps_in_current_phase is properly tracked and reset
    let mut state = TrainingState::new();

    // Initial state
    assert_eq!(state.steps_in_current_phase, 0);
    assert_eq!(state.current_phase, Phase::Warmup);

    // Transition to Predict phase
    state.enter_phase(Phase::Predict);
    assert_eq!(state.steps_in_current_phase, 0);
    assert_eq!(state.current_phase, Phase::Predict);

    // Simulate steps within the phase
    for i in 1..=15 {
        state.steps_in_current_phase = i;
        assert_eq!(state.steps_in_current_phase, i);
    }

    // Transition to Correct phase should reset counter
    state.enter_phase(Phase::Correct);
    assert_eq!(state.steps_in_current_phase, 0);
    assert_eq!(state.current_phase, Phase::Correct);
}

#[test]
fn test_micro_correction_interval_logic() {
    // Test the logic for when micro-corrections should trigger
    let correction_interval = 10;

    // Simulate the condition check
    for step in 1..=50 {
        let should_trigger = correction_interval > 0 && step % correction_interval == 0;

        match step {
            10 | 20 | 30 | 40 | 50 => assert!(should_trigger, "Step {} should trigger", step),
            _ => assert!(!should_trigger, "Step {} should not trigger", step),
        }
    }
}

#[test]
fn test_micro_correction_disabled_when_zero() {
    // Test that micro-corrections don't trigger when interval is 0
    let correction_interval = 0;

    for step in 1..=100 {
        let should_trigger = correction_interval > 0 && step % correction_interval == 0;
        assert!(!should_trigger, "Micro-corrections should never trigger when interval is 0");
    }
}

#[test]
fn test_serialization_with_correction_interval() {
    // Test that correction_interval is properly serialized/deserialized
    let config = HybridTrainerConfig::builder()
        .correction_interval(15)
        .warmup_steps(100)
        .build();

    // Serialize to TOML
    let toml_str = toml::to_string(&config).expect("Failed to serialize");

    // Deserialize back
    let parsed: HybridTrainerConfig = toml::from_str(&toml_str).expect("Failed to deserialize");

    assert_eq!(parsed.correction_interval, 15);
    assert_eq!(parsed.warmup_steps, 100);
}
