//! End-to-end integration tests for hybrid-predict-trainer-rs

use hybrid_predict_trainer_rs::prelude::*;
use hybrid_predict_trainer_rs::state::WeightDelta;

/// Mock batch implementing the Batch trait for testing.
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

/// Mock model for integration testing.
struct TestModel {
    params: Vec<f32>,
    iteration: u32,
    last_loss: f32,
}

impl TestModel {
    fn new(param_count: usize) -> Self {
        Self {
            params: vec![1.0; param_count],
            iteration: 0,
            last_loss: 0.0,
        }
    }
}

impl Model<TestBatch> for TestModel {
    fn forward(&mut self, _batch: &TestBatch) -> HybridResult<f32> {
        // Simulate decreasing loss
        let iter_f32 = self.iteration as f32;
        let loss = 2.5 * (-(iter_f32 * 0.001)).exp() + 0.1;
        self.last_loss = loss;
        self.iteration += 1;
        Ok(loss)
    }

    fn backward(&mut self) -> HybridResult<GradientInfo> {
        let grad_norm = self.last_loss * 0.5;
        Ok(GradientInfo {
            loss: self.last_loss,
            gradient_norm: grad_norm,
            per_param_norms: None,
        })
    }

    fn parameter_count(&self) -> usize {
        self.params.len()
    }

    fn apply_weight_delta(&mut self, _delta: &WeightDelta) -> HybridResult<()> {
        Ok(())
    }
}

/// Mock optimizer for integration testing.
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

#[test]
fn test_trainer_creation_with_default_config() {
    // Test that HybridTrainerConfig::default() creates valid config
    let config = HybridTrainerConfig::default();
    assert!(config.warmup_steps > 0);
    assert!(config.validate().is_ok());
}

#[test]
fn test_config_builder_validation() {
    // Test config builder pattern
    let config = HybridTrainerConfig::builder()
        .warmup_steps(100)
        .full_steps(50)
        .max_predict_steps(10)
        .build();

    assert!(config.validate().is_ok());
    assert_eq!(config.warmup_steps, 100);
    assert_eq!(config.full_steps, 50);
    assert_eq!(config.max_predict_steps, 10);
}

#[test]
fn test_invalid_config_validation() {
    // Test that invalid configs are rejected
    let config = HybridTrainerConfig {
        warmup_steps: 0,
        ..Default::default()
    };

    assert!(config.validate().is_err());
}

#[test]
fn test_trainer_initialization() {
    // Test creating a trainer with mock model and optimizer
    let config = HybridTrainerConfig::default();
    let model = TestModel::new(100);
    let optimizer = TestOptimizer::new(0.001);

    let trainer = HybridTrainer::new(model, optimizer, config);
    assert!(trainer.is_ok());

    let trainer = trainer.unwrap();
    assert_eq!(trainer.current_step(), 0);
    assert_eq!(trainer.current_phase(), Phase::Warmup);
}

#[test]
fn test_warmup_phase_execution() {
    // Test that trainer starts in warmup and completes it
    let config = HybridTrainerConfig::builder()
        .warmup_steps(10)
        .full_steps(5)
        .build();

    let model = TestModel::new(100);
    let optimizer = TestOptimizer::new(0.001);

    let mut trainer = HybridTrainer::new(model, optimizer, config).unwrap();

    // Execute warmup steps
    for _ in 0..10 {
        let batch = TestBatch::new(32);
        let result = trainer.step(&batch);
        assert!(result.is_ok());

        let step_result = result.unwrap();
        assert_eq!(step_result.phase, Phase::Warmup);
        assert!(!step_result.was_predicted);
    }

    // After warmup, should transition to Full
    let batch = TestBatch::new(32);
    let result = trainer.step(&batch).unwrap();
    assert_eq!(result.phase, Phase::Full);
}

#[test]
fn test_full_training_step() {
    // Test that full training steps execute correctly
    let config = HybridTrainerConfig::builder()
        .warmup_steps(5)
        .full_steps(10)
        .build();

    let model = TestModel::new(100);
    let optimizer = TestOptimizer::new(0.001);

    let mut trainer = HybridTrainer::new(model, optimizer, config).unwrap();

    // Execute a single step
    let batch = TestBatch::new(32);
    let result = trainer.step(&batch);

    assert!(result.is_ok());
    let step_result = result.unwrap();
    assert!(step_result.loss > 0.0);
    assert!(step_result.step_time_ms > 0.0);
}

#[test]
fn test_metrics_collection() {
    // Test that metrics are collected when enabled
    let config = HybridTrainerConfig::builder()
        .warmup_steps(5)
        .collect_metrics(true)
        .build();

    let model = TestModel::new(100);
    let optimizer = TestOptimizer::new(0.001);

    let mut trainer = HybridTrainer::new(model, optimizer, config).unwrap();

    // Execute steps
    for _ in 0..10 {
        let batch = TestBatch::new(32);
        let _ = trainer.step(&batch);
    }

    // Check statistics
    let stats = trainer.statistics();
    assert!(stats.total_steps > 0);
}

#[test]
fn test_step_result_metadata() {
    // Test that StepResult contains expected metadata
    let config = HybridTrainerConfig::default();
    let model = TestModel::new(100);
    let optimizer = TestOptimizer::new(0.001);

    let mut trainer = HybridTrainer::new(model, optimizer, config).unwrap();

    let batch = TestBatch::new(32);
    let result = trainer.step(&batch).unwrap();

    // Verify all fields are populated
    assert!(result.loss >= 0.0);
    assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
    assert!(result.step_time_ms >= 0.0);
}
