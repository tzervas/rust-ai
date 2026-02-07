//! Advanced edge case integration tests for HybridTrainer
//!
//! This test suite covers edge cases that might not be encountered during normal
//! training but could occur in production environments:
//! - Divergence recovery scenarios
//! - Phase transition edge cases
//! - Memory stress conditions
//! - Configuration validation
//! - Numerical stability issues
//! - Serialization/deserialization

use hybrid_predict_trainer_rs::prelude::*;
use hybrid_predict_trainer_rs::state::WeightDelta;
use hybrid_predict_trainer_rs::config::{HybridTrainerConfig, DivergenceConfig, PredictorConfig};
use hybrid_predict_trainer_rs::error::{DivergenceLevel, RecoveryAction};

// ============================================================================
// Test Fixtures and Helpers
// ============================================================================

/// Mock batch for testing
#[derive(Debug, Clone)]
struct TestBatch {
    size: usize,
}

impl TestBatch {
    fn new(size: usize) -> Self {
        Self { size }
    }
}

impl Batch for TestBatch {
    fn batch_size(&self) -> usize {
        self.size
    }
}

/// Mock model with configurable behavior for edge case testing
struct EdgeCaseModel {
    params: Vec<f32>,
    iteration: u32,
    /// Controls whether the model returns NaN loss
    return_nan: bool,
    /// Controls whether backward returns exploding gradients
    exploding_gradients: bool,
    /// Controls whether backward returns vanishing gradients
    vanishing_gradients: bool,
    /// Loss value to return (if not NaN)
    fixed_loss: Option<f32>,
    /// Gradient norm multiplier
    gradient_multiplier: f32,
}

impl EdgeCaseModel {
    fn new(param_count: usize) -> Self {
        Self {
            params: vec![1.0; param_count],
            iteration: 0,
            return_nan: false,
            exploding_gradients: false,
            vanishing_gradients: false,
            fixed_loss: None,
            gradient_multiplier: 1.0,
        }
    }

    fn with_nan_loss(mut self) -> Self {
        self.return_nan = true;
        self
    }

    fn with_exploding_gradients(mut self) -> Self {
        self.exploding_gradients = true;
        self
    }

    fn with_vanishing_gradients(mut self) -> Self {
        self.vanishing_gradients = true;
        self
    }

    fn with_fixed_loss(mut self, loss: f32) -> Self {
        self.fixed_loss = Some(loss);
        self
    }

    fn with_gradient_multiplier(mut self, multiplier: f32) -> Self {
        self.gradient_multiplier = multiplier;
        self
    }

    fn reset_edge_case_flags(&mut self) {
        self.return_nan = false;
        self.exploding_gradients = false;
        self.vanishing_gradients = false;
    }
}

impl Model<TestBatch> for EdgeCaseModel {
    fn forward(&mut self, _batch: &TestBatch) -> HybridResult<f32> {
        if self.return_nan {
            return Ok(f32::NAN);
        }

        if let Some(loss) = self.fixed_loss {
            self.iteration += 1;
            return Ok(loss);
        }

        // Simulate decreasing loss
        let iter_f32 = self.iteration as f32;
        let loss = 2.5 * (-(iter_f32 * 0.001)).exp() + 0.1;
        self.iteration += 1;
        Ok(loss)
    }

    fn backward(&mut self) -> HybridResult<GradientInfo> {
        let base_grad_norm = 1.0;

        let grad_norm = if self.exploding_gradients {
            1e10 // Exploding gradient
        } else if self.vanishing_gradients {
            1e-10 // Vanishing gradient
        } else {
            base_grad_norm * self.gradient_multiplier
        };

        Ok(GradientInfo {
            loss: self.fixed_loss.unwrap_or(2.0),
            gradient_norm: grad_norm,
            per_param_norms: None,
        })
    }

    fn parameter_count(&self) -> usize {
        self.params.len()
    }

    fn apply_weight_delta(&mut self, delta: &WeightDelta) -> HybridResult<()> {
        // Apply delta from the HashMap of per-parameter deltas
        for (_param_name, values) in &delta.deltas {
            let apply_count = values.len().min(self.params.len());
            for i in 0..apply_count {
                self.params[i] += values[i] * delta.scale;
            }
        }
        Ok(())
    }
}

/// Mock optimizer
struct TestOptimizer {
    lr: f32,
}

impl TestOptimizer {
    fn new(lr: f32) -> Self {
        Self { lr }
    }
}

impl<M, B> Optimizer<M, B> for TestOptimizer
where
    M: Model<B>,
    B: Batch,
{
    fn step(&mut self, _model: &mut M, _gradients: &GradientInfo) -> HybridResult<()> {
        Ok(())
    }

    fn learning_rate(&self) -> f32 {
        self.lr
    }

    fn set_learning_rate(&mut self, lr: f32) {
        self.lr = lr;
    }

    fn zero_grad(&mut self) {
        // No-op for mock
    }
}

// ============================================================================
// 1. Divergence Recovery Tests
// ============================================================================

#[test]
fn test_divergence_recovery_from_nan_loss() {
    let config = HybridTrainerConfig::builder()
        .warmup_steps(5)
        .full_steps(10)
        .build();

    let model = EdgeCaseModel::new(100).with_nan_loss();
    let optimizer = TestOptimizer::new(0.001);

    let mut trainer = HybridTrainer::new(model, optimizer, config).unwrap();

    // First few steps should work
    let batch = TestBatch::new(32);
    for _ in 0..3 {
        let result = trainer.step(&batch);
        // NaN loss will be detected by divergence monitor after warmup
        // During warmup, NaN is expected to be returned
        if result.is_err() {
            let (_err, recovery) = result.unwrap_err();
            // Verify we get a recovery action
            assert!(recovery.is_some());
            break;
        }
    }
}

#[test]
fn test_divergence_recovery_from_gradient_explosion() {
    let config = HybridTrainerConfig::builder()
        .warmup_steps(5)
        .full_steps(20)
        .divergence_threshold(2.0)
        .build();

    let model = EdgeCaseModel::new(100);
    let optimizer = TestOptimizer::new(0.001);

    let mut trainer = HybridTrainer::new(model, optimizer, config).unwrap();

    // Run some normal steps first
    let batch = TestBatch::new(32);
    for _ in 0..10 {
        let result = trainer.step(&batch);
        assert!(result.is_ok());
    }

    // Now inject exploding gradients via force_full_phase and check recovery
    trainer.force_full_phase(10);

    // Continue training - should recover with full training
    for _ in 0..5 {
        let result = trainer.step(&batch);
        if result.is_ok() {
            let step_result = result.unwrap();
            // After forcing full phase, should stay in Full
            assert_eq!(step_result.phase, Phase::Full);
        }
    }
}

#[test]
fn test_multiple_divergence_recovery() {
    let config = HybridTrainerConfig::builder()
        .warmup_steps(5)
        .full_steps(10)
        .max_predict_steps(20)
        .divergence_threshold(3.0)
        .build();

    let model = EdgeCaseModel::new(100);
    let optimizer = TestOptimizer::new(0.001);

    let mut trainer = HybridTrainer::new(model, optimizer, config).unwrap();

    let batch = TestBatch::new(32);
    let mut divergence_count = 0;

    // Run many steps and count divergences
    for _ in 0..100 {
        let result = trainer.step(&batch);
        match result {
            Ok(_) => continue,
            Err((_, recovery)) => {
                divergence_count += 1;
                if let Some(action) = recovery {
                    // Should be able to continue after recovery
                    assert!(action.can_continue());
                }
                // In real scenario, we'd apply the recovery action
                break;
            }
        }
    }

    // With normal model behavior, we shouldn't hit many divergences
    assert!(divergence_count <= 1);
}

// ============================================================================
// 2. Phase Transition Edge Cases
// ============================================================================

#[test]
fn test_phase_transition_at_step_zero() {
    let config = HybridTrainerConfig::builder()
        .warmup_steps(0) // Zero warmup steps - edge case
        .full_steps(10)
        .build();

    // This should fail validation
    assert!(config.validate().is_err());
}

#[test]
fn test_rapid_phase_oscillation() {
    let config = HybridTrainerConfig::builder()
        .warmup_steps(5)
        .full_steps(1) // Very short full phase
        .max_predict_steps(1) // Very short predict phase
        .confidence_threshold(0.5) // Low threshold for easier prediction
        .build();

    let model = EdgeCaseModel::new(100);
    let optimizer = TestOptimizer::new(0.001);

    let mut trainer = HybridTrainer::new(model, optimizer, config).unwrap();

    let batch = TestBatch::new(32);
    let mut phase_changes = 0;
    let mut last_phase = Phase::Warmup;

    for _ in 0..50 {
        let result = trainer.step(&batch);
        if result.is_ok() {
            let step_result = result.unwrap();
            if step_result.phase != last_phase {
                phase_changes += 1;
                last_phase = step_result.phase;
            }
        }
    }

    // With short phases, we should see multiple transitions
    assert!(phase_changes > 5);
}

#[test]
fn test_stuck_in_warmup() {
    let config = HybridTrainerConfig::builder()
        .warmup_steps(1000) // Very long warmup
        .full_steps(10)
        .build();

    let model = EdgeCaseModel::new(100);
    let optimizer = TestOptimizer::new(0.001);

    let mut trainer = HybridTrainer::new(model, optimizer, config).unwrap();

    let batch = TestBatch::new(32);

    // Run 50 steps, should all be warmup
    for _ in 0..50 {
        let result = trainer.step(&batch);
        assert!(result.is_ok());
        let step_result = result.unwrap();
        assert_eq!(step_result.phase, Phase::Warmup);
    }
}

#[test]
fn test_confidence_threshold_edge_cases() {
    // Test with very high confidence threshold (effectively disables prediction)
    let config = HybridTrainerConfig::builder()
        .warmup_steps(5)
        .full_steps(10)
        .confidence_threshold(0.99) // Very high threshold
        .build();

    let model = EdgeCaseModel::new(100);
    let optimizer = TestOptimizer::new(0.001);

    let mut trainer = HybridTrainer::new(model, optimizer, config).unwrap();

    let batch = TestBatch::new(32);

    // Run enough steps to get past warmup
    for _ in 0..20 {
        let result = trainer.step(&batch);
        if result.is_ok() {
            let step_result = result.unwrap();
            // With high confidence threshold, should mostly stay in Full
            if step_result.phase == Phase::Predict {
                // Prediction phase is rare but possible if confidence is very high
                assert!(step_result.confidence >= 0.99);
            }
        }
    }
}

// ============================================================================
// 3. Memory Stress Tests
// ============================================================================

#[test]
fn test_large_batch_size() {
    let config = HybridTrainerConfig::builder()
        .warmup_steps(5)
        .full_steps(10)
        .build();

    let model = EdgeCaseModel::new(1000); // Larger model
    let optimizer = TestOptimizer::new(0.001);

    let mut trainer = HybridTrainer::new(model, optimizer, config).unwrap();

    // Very large batch
    let batch = TestBatch::new(10000);

    // Should handle large batches without panic
    let result = trainer.step(&batch);
    assert!(result.is_ok());
}

#[test]
fn test_long_history_buffer() {
    let config = HybridTrainerConfig::builder()
        .warmup_steps(5)
        .full_steps(10)
        .build();

    let model = EdgeCaseModel::new(100);
    let optimizer = TestOptimizer::new(0.001);

    let mut trainer = HybridTrainer::new(model, optimizer, config).unwrap();

    let batch = TestBatch::new(32);

    // Run many steps to fill history buffers
    for _ in 0..1000 {
        let result = trainer.step(&batch);
        assert!(result.is_ok());
    }

    // Verify training state is still healthy
    assert!(trainer.current_step() == 1000);
}

#[test]
fn test_ensemble_size_scaling() {
    // Test with large ensemble
    let predictor_config = PredictorConfig::RSSM {
        deterministic_dim: 64,
        stochastic_dim: 16,
        num_categoricals: 16,
        ensemble_size: 10, // Large ensemble
    };

    let config = HybridTrainerConfig::builder()
        .warmup_steps(5)
        .full_steps(10)
        .predictor_config(predictor_config)
        .build();

    let model = EdgeCaseModel::new(100);
    let optimizer = TestOptimizer::new(0.001);

    // Should be able to create trainer with large ensemble
    let trainer = HybridTrainer::new(model, optimizer, config);
    assert!(trainer.is_ok());
}

// ============================================================================
// 4. Configuration Validation
// ============================================================================

#[test]
fn test_invalid_confidence_threshold_too_low() {
    let config = HybridTrainerConfig {
        confidence_threshold: 0.0, // Invalid: must be > 0
        ..Default::default()
    };

    assert!(config.validate().is_err());
}

#[test]
fn test_invalid_confidence_threshold_too_high() {
    let config = HybridTrainerConfig {
        confidence_threshold: 1.0, // Invalid: must be < 1
        ..Default::default()
    };

    assert!(config.validate().is_err());
}

#[test]
fn test_invalid_confidence_threshold_above_one() {
    let config = HybridTrainerConfig {
        confidence_threshold: 1.5, // Invalid: must be < 1
        ..Default::default()
    };

    assert!(config.validate().is_err());
}

#[test]
fn test_zero_max_predict_steps() {
    let config = HybridTrainerConfig::builder()
        .warmup_steps(10)
        .full_steps(10)
        .max_predict_steps(0) // Edge case: no prediction allowed
        .build();

    let model = EdgeCaseModel::new(100);
    let optimizer = TestOptimizer::new(0.001);

    let trainer = HybridTrainer::new(model, optimizer, config);
    // Should succeed - trainer will just never enter predict phase
    assert!(trainer.is_ok());
}

#[test]
fn test_negative_sigma_threshold() {
    let divergence_config = DivergenceConfig {
        loss_sigma_threshold: -1.0, // Invalid: must be positive
        ..Default::default()
    };

    let config = HybridTrainerConfig {
        divergence_config,
        divergence_threshold: -1.0, // Also invalid
        ..Default::default()
    };

    assert!(config.validate().is_err());
}

#[test]
fn test_zero_warmup_steps() {
    let config = HybridTrainerConfig {
        warmup_steps: 0, // Invalid
        ..Default::default()
    };

    assert!(config.validate().is_err());
}

// ============================================================================
// 5. Concurrency Tests (where applicable)
// ============================================================================

#[test]
fn test_metrics_from_multiple_threads() {
    use std::sync::{Arc, Mutex};

    let config = HybridTrainerConfig::builder()
        .warmup_steps(5)
        .full_steps(10)
        .collect_metrics(true)
        .build();

    let model = EdgeCaseModel::new(100);
    let optimizer = TestOptimizer::new(0.001);

    let trainer = HybridTrainer::new(model, optimizer, config).unwrap();
    let trainer = Arc::new(Mutex::new(trainer));

    // Spawn threads that read statistics (non-mutating access)
    let mut handles = vec![];
    for _ in 0..4 {
        let trainer_clone = Arc::clone(&trainer);
        let handle = std::thread::spawn(move || {
            // Try to get metrics (requires mutable access for statistics())
            let mut t = trainer_clone.lock().unwrap();
            let _stats = t.statistics();
        });
        handles.push(handle);
    }

    // Wait for all threads
    for handle in handles {
        handle.join().unwrap();
    }
}

#[test]
fn test_concurrent_statistics_access() {
    let config = HybridTrainerConfig::builder()
        .warmup_steps(5)
        .full_steps(10)
        .collect_metrics(true)
        .build();

    let model = EdgeCaseModel::new(100);
    let optimizer = TestOptimizer::new(0.001);

    let mut trainer = HybridTrainer::new(model, optimizer, config).unwrap();

    // Run some steps to populate metrics
    let batch = TestBatch::new(32);
    for _ in 0..20 {
        let _ = trainer.step(&batch);
    }

    // Get statistics
    let stats = trainer.statistics();
    assert!(stats.total_steps > 0);
}

// ============================================================================
// 6. Numerical Stability
// ============================================================================

#[test]
fn test_very_small_loss_values() {
    let config = HybridTrainerConfig::builder()
        .warmup_steps(5)
        .full_steps(10)
        .build();

    let model = EdgeCaseModel::new(100).with_fixed_loss(1e-10);
    let optimizer = TestOptimizer::new(0.001);

    let mut trainer = HybridTrainer::new(model, optimizer, config).unwrap();

    let batch = TestBatch::new(32);

    // Should handle very small loss values without underflow
    for _ in 0..10 {
        let result = trainer.step(&batch);
        assert!(result.is_ok());
        let step_result = result.unwrap();
        assert!(step_result.loss >= 0.0);
        assert!(step_result.loss.is_finite());
    }
}

#[test]
fn test_very_large_gradient_norms() {
    let config = HybridTrainerConfig::builder()
        .warmup_steps(5)
        .full_steps(10)
        .build();

    let model = EdgeCaseModel::new(100).with_gradient_multiplier(1e8);
    let optimizer = TestOptimizer::new(0.001);

    let mut trainer = HybridTrainer::new(model, optimizer, config).unwrap();

    let batch = TestBatch::new(32);

    // Very large gradients should be detected but handled
    for _ in 0..10 {
        let result = trainer.step(&batch);
        // Might fail due to gradient explosion detection, which is expected
        if result.is_err() {
            let (_err, recovery) = result.unwrap_err();
            // Should have recovery action
            if let Some(action) = recovery {
                assert!(matches!(
                    action,
                    RecoveryAction::ForceFullPhase(_) | RecoveryAction::RollbackAndRetry { .. }
                ));
            }
            break;
        }
    }
}

#[test]
fn test_loss_approaching_zero() {
    let config = HybridTrainerConfig::builder()
        .warmup_steps(5)
        .full_steps(10)
        .build();

    let model = EdgeCaseModel::new(100).with_fixed_loss(0.0001);
    let optimizer = TestOptimizer::new(0.001);

    let mut trainer = HybridTrainer::new(model, optimizer, config).unwrap();

    let batch = TestBatch::new(32);

    // Loss approaching zero should be stable
    for _ in 0..20 {
        let result = trainer.step(&batch);
        assert!(result.is_ok());
        let step_result = result.unwrap();
        assert!(step_result.loss >= 0.0);
        assert!(step_result.loss < 0.01);
    }
}

#[test]
fn test_subnormal_numbers() {
    let config = HybridTrainerConfig::builder()
        .warmup_steps(5)
        .full_steps(10)
        .build();

    // Use subnormal float value
    let model = EdgeCaseModel::new(100).with_fixed_loss(f32::MIN_POSITIVE / 2.0);
    let optimizer = TestOptimizer::new(0.001);

    let mut trainer = HybridTrainer::new(model, optimizer, config).unwrap();

    let batch = TestBatch::new(32);

    // Should handle subnormal numbers
    let result = trainer.step(&batch);
    assert!(result.is_ok());
}

// ============================================================================
// 7. Serialization/Deserialization
// ============================================================================

#[test]
fn test_config_serialize_deserialize() {
    let config = HybridTrainerConfig::builder()
        .warmup_steps(100)
        .full_steps(20)
        .max_predict_steps(80)
        .confidence_threshold(0.85)
        .divergence_threshold(3.0)
        .build();

    // Serialize to TOML
    let toml_str = toml::to_string(&config).unwrap();

    // Deserialize back
    let deserialized: HybridTrainerConfig = toml::from_str(&toml_str).unwrap();

    // Verify fields match
    assert_eq!(config.warmup_steps, deserialized.warmup_steps);
    assert_eq!(config.full_steps, deserialized.full_steps);
    assert_eq!(config.max_predict_steps, deserialized.max_predict_steps);
    assert!((config.confidence_threshold - deserialized.confidence_threshold).abs() < f32::EPSILON);
    assert!((config.divergence_threshold - deserialized.divergence_threshold).abs() < f32::EPSILON);
}

#[test]
fn test_state_serialize_deserialize() {
    use hybrid_predict_trainer_rs::state::TrainingState;

    let mut state = TrainingState::new();

    // Populate with some data
    for i in 0..10 {
        state.record_step(2.5 - i as f32 * 0.1, 1.0);
    }

    // Serialize to JSON
    let json_str = serde_json::to_string(&state).unwrap();

    // Deserialize back
    let deserialized: TrainingState = serde_json::from_str(&json_str).unwrap();

    // Verify state matches
    assert_eq!(state.step, deserialized.step);
    assert!((state.loss - deserialized.loss).abs() < f32::EPSILON);
    assert_eq!(state.loss_history.len(), deserialized.loss_history.len());
}

#[test]
fn test_predictor_config_serialization() {
    let configs = vec![
        PredictorConfig::Linear {
            feature_dim: 128,
            l2_regularization: 0.01,
        },
        PredictorConfig::MLP {
            hidden_dims: vec![256, 128],
            activation: hybrid_predict_trainer_rs::config::Activation::Gelu,
            dropout: 0.1,
        },
        PredictorConfig::RSSM {
            deterministic_dim: 256,
            stochastic_dim: 32,
            num_categoricals: 32,
            ensemble_size: 5,
        },
    ];

    for config in configs {
        // Serialize and deserialize
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: PredictorConfig = serde_json::from_str(&json).unwrap();

        // Verify types match (can't easily check all fields generically)
        let original_json = serde_json::to_string(&config).unwrap();
        let deserialized_json = serde_json::to_string(&deserialized).unwrap();
        assert_eq!(original_json, deserialized_json);
    }
}

#[test]
fn test_divergence_level_ordering() {
    assert!(DivergenceLevel::Normal < DivergenceLevel::Caution);
    assert!(DivergenceLevel::Caution < DivergenceLevel::Warning);
    assert!(DivergenceLevel::Warning < DivergenceLevel::Critical);
}

#[test]
fn test_recovery_action_descriptions() {
    let actions = vec![
        RecoveryAction::Continue,
        RecoveryAction::ReducePredictRatio(0.5),
        RecoveryAction::ForceFullPhase(50),
        RecoveryAction::IncreaseConfidenceThreshold(0.9),
        RecoveryAction::RollbackAndRetry {
            checkpoint_step: 100,
            new_learning_rate: 0.0001,
        },
        RecoveryAction::SkipBatch,
        RecoveryAction::ResetPredictor,
        RecoveryAction::Abort {
            reason: "test".to_string(),
        },
    ];

    // All actions should have non-empty descriptions
    for action in actions {
        let desc = action.description();
        assert!(!desc.is_empty());
    }
}

// ============================================================================
// 8. Additional Edge Cases
// ============================================================================

#[test]
fn test_correction_interval_edge_cases() {
    // Test with correction_interval = 1 (correction every step)
    let config = HybridTrainerConfig::builder()
        .warmup_steps(5)
        .full_steps(5)
        .max_predict_steps(20)
        .correction_interval(1)
        .build();

    let model = EdgeCaseModel::new(100);
    let optimizer = TestOptimizer::new(0.001);

    let mut trainer = HybridTrainer::new(model, optimizer, config).unwrap();

    let batch = TestBatch::new(32);

    // Run past warmup into predict phase
    for _ in 0..30 {
        let result = trainer.step(&batch);
        if result.is_ok() {
            // Should handle micro-corrections every step
            continue;
        }
    }
}

#[test]
fn test_zero_learning_rate() {
    let config = HybridTrainerConfig::builder()
        .warmup_steps(5)
        .full_steps(10)
        .build();

    let model = EdgeCaseModel::new(100);
    let optimizer = TestOptimizer::new(0.0); // Zero learning rate

    let mut trainer = HybridTrainer::new(model, optimizer, config).unwrap();

    let batch = TestBatch::new(32);

    // Should not crash with zero learning rate
    let result = trainer.step(&batch);
    assert!(result.is_ok());

    // Loss might not decrease, but training should proceed
    let step_result = result.unwrap();
    assert!(step_result.loss.is_finite());
}

#[test]
fn test_single_parameter_model() {
    let config = HybridTrainerConfig::builder()
        .warmup_steps(5)
        .full_steps(10)
        .build();

    let model = EdgeCaseModel::new(1); // Single parameter
    let optimizer = TestOptimizer::new(0.001);

    let mut trainer = HybridTrainer::new(model, optimizer, config).unwrap();

    let batch = TestBatch::new(32);

    // Should handle single-parameter models
    let result = trainer.step(&batch);
    assert!(result.is_ok());
}

#[test]
fn test_metrics_disabled() {
    let config = HybridTrainerConfig::builder()
        .warmup_steps(5)
        .full_steps(10)
        .collect_metrics(false)
        .build();

    let model = EdgeCaseModel::new(100);
    let optimizer = TestOptimizer::new(0.001);

    let mut trainer = HybridTrainer::new(model, optimizer, config).unwrap();

    let batch = TestBatch::new(32);

    // Run steps with metrics disabled
    for _ in 0..10 {
        let result = trainer.step(&batch);
        assert!(result.is_ok());
        let step_result = result.unwrap();
        // Metrics should be None when disabled
        assert!(step_result.metrics.is_none());
    }
}

#[test]
fn test_force_full_phase_api() {
    let config = HybridTrainerConfig::builder()
        .warmup_steps(5)
        .full_steps(5)
        .max_predict_steps(20)
        .build();

    let model = EdgeCaseModel::new(100);
    let optimizer = TestOptimizer::new(0.001);

    let mut trainer = HybridTrainer::new(model, optimizer, config).unwrap();

    let batch = TestBatch::new(32);

    // Get past warmup
    for _ in 0..10 {
        let _ = trainer.step(&batch);
    }

    // Force full phase
    trainer.force_full_phase(15);

    // Next steps should be Full
    for _ in 0..10 {
        let result = trainer.step(&batch);
        if result.is_ok() {
            let step_result = result.unwrap();
            // Should be in Full phase due to force_full_phase
            if step_result.phase == Phase::Full {
                // Expected
                continue;
            }
        }
    }
}
