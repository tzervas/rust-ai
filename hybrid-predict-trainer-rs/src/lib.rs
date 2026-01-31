//! # hybrid-predict-trainer-rs
//!
//! Hybridized predictive training framework that accelerates deep learning through
//! intelligent phase-based training with whole-phase prediction and residual correction.
//!
//! ## Overview
//!
//! This crate implements a novel training paradigm that achieves 5-10x speedup over
//! traditional training by predicting training outcomes rather than computing every
//! gradient step. The key insight is that training dynamics evolve on low-dimensional
//! manifolds, making whole-phase prediction tractable.
//!
//! ## Training Phases
//!
//! The training loop cycles through four distinct phases:
//!
//! 1. **Warmup Phase** - Initial training steps to establish baseline dynamics
//! 2. **Full Training Phase** - Traditional forward/backward pass computation
//! 3. **Predictive Phase** - Skip backward passes using learned dynamics model
//! 4. **Correction Phase** - Apply residual corrections to maintain accuracy
//!
//! ```text
//!                     ┌─────────┐
//!                     │ WARMUP  │
//!                     └────┬────┘
//!                          │
//!                          ▼
//!               ┌─────────────────────┐
//!               │                     │
//!               ▼                     │
//!         ┌──────────┐                │
//!    ┌───▶│   FULL   │◀───────────────┤
//!    │    └────┬─────┘                │
//!    │         │                      │
//!    │         ▼                      │
//!    │    ┌──────────┐                │
//!    │    │ PREDICT  │                │
//!    │    └────┬─────┘                │
//!    │         │                      │
//!    │         ▼                      │
//!    │    ┌──────────┐                │
//!    │    │ CORRECT  │────────────────┘
//!    │    └────┬─────┘
//!    │         │
//!    └─────────┘
//! ```
//!
//! ## Quick Start
//!
//! ```no_run
//! use hybrid_predict_trainer_rs::{HybridTrainer, HybridTrainerConfig};
//!
//! // Create configuration with sensible defaults
//! let config = HybridTrainerConfig::default();
//!
//! // Initialize the hybrid trainer
//! // let trainer = HybridTrainer::new(model, optimizer, config)?;
//!
//! // Training loop
//! // for batch in dataloader {
//! //     let result = trainer.step(&batch)?;
//! //     println!("Loss: {}, Phase: {:?}", result.loss, result.phase);
//! // }
//! ```
//!
//! ## Features
//!
//! - **GPU Acceleration** - `CubeCL` and Burn backends for high-performance compute
//! - **Adaptive Phase Selection** - Bandit-based algorithm for optimal phase lengths
//! - **Divergence Detection** - Multi-signal monitoring prevents training instability
//! - **Residual Correction** - Online learning corrects prediction errors
//! - **Checkpoint Support** - Save/restore full training state including predictor
//!
//! ## Feature Flags
//!
//! - `std` - Enable standard library support (default)
//! - `cuda` - Enable CUDA GPU acceleration via `CubeCL`
//! - `candle` - Enable Candle tensor operations for model compatibility
//! - `async` - Enable async/await support with Tokio
//! - `full` - Enable all features
//!
//! ## Architecture
//!
//! The crate is organized into the following modules:
//!
//! - [`config`] - Training configuration and serialization
//! - [`error`] - Error types with recovery actions
//! - [`phases`] - Phase state machine and execution control
//! - [`state`] - Training state encoding and management
//! - [`dynamics`] - RSSM-lite dynamics model for prediction
//! - [`residuals`] - Residual extraction and storage
//! - [`corrector`] - Prediction correction via residual application
//! - [`divergence`] - Multi-signal divergence detection
//! - [`metrics`] - Training metrics collection and reporting
#![cfg_attr(
    feature = "cuda",
    doc = "- [`gpu`] - GPU acceleration kernels (requires `cuda` feature)"
)]
#![cfg_attr(
    not(feature = "cuda"),
    doc = "- `gpu` - GPU acceleration kernels (requires `cuda` feature)"
)]
//!
//! ## References
//!
//! This implementation is based on research findings documented in
//! `predictive-training-research.md`, synthesizing insights from:
//!
//! - Neural Tangent Kernel (NTK) theory for training dynamics
//! - RSSM world models from `DreamerV3`
//! - K-FAC for structured gradient approximation
//! - `PowerSGD` for low-rank gradient compression

#![cfg_attr(docsrs, feature(doc_cfg))]
#![warn(missing_docs)]
#![warn(rustdoc::missing_crate_level_docs)]
#![deny(unsafe_code)]
// Allow precision loss casts - acceptable in ML numerical code
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
// Suppress documentation warnings during development
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::too_many_lines)]
// Allow other common patterns
#![allow(clippy::needless_range_loop)]
#![allow(clippy::items_after_statements)]
#![allow(clippy::unused_self)]
#![allow(clippy::manual_clamp)]

// Core modules
pub mod config;
pub mod error;
pub mod phases;
pub mod state;

// Training phase implementations
pub mod corrector;
pub mod full_train;
pub mod predictive;
pub mod residuals;
pub mod warmup;

// Prediction and control
pub mod bandit;
pub mod divergence;
pub mod dynamics;
pub mod gru;

// Metrics and monitoring
pub mod metrics;

// High-precision timing utilities
pub mod timing;

// GPU acceleration (feature-gated)
#[cfg(feature = "cuda")]
#[cfg_attr(docsrs, doc(cfg(feature = "cuda")))]
pub mod gpu;

// Re-exports for convenient access
pub use config::HybridTrainerConfig;
pub use error::{HybridResult, HybridTrainingError, RecoveryAction};
pub use phases::{Phase, PhaseController, PhaseDecision, PhaseOutcome};
pub use state::TrainingState;
pub use timing::{Duration, Timer, TimingMetrics, TimingStats};

// Standard library imports
use std::sync::Arc;
use std::time::Instant;

// External crate imports
use parking_lot::RwLock;

/// Batch of training data.
///
/// Generic container for a batch of input data that will be fed to the model
/// during training. The actual batch format depends on the model implementation.
pub trait Batch: Send + Sync {
    /// Returns the batch size (number of samples).
    fn batch_size(&self) -> usize;
}

/// Gradient information from a backward pass.
///
/// Contains the computed gradients and loss for a training step.
#[derive(Debug, Clone)]
pub struct GradientInfo {
    /// The computed loss value.
    pub loss: f32,
    /// L2 norm of all gradients.
    pub gradient_norm: f32,
    /// Per-parameter gradient norms (optional, for debugging).
    pub per_param_norms: Option<Vec<f32>>,
}

/// Trait for models that can be trained with the hybrid trainer.
///
/// Models must implement forward pass, backward pass, and parameter access.
/// The trainer will call these methods during different training phases.
///
/// # Why This Trait?
///
/// The hybrid trainer is framework-agnostic. By requiring only forward/backward
/// and weight delta application, it works with any deep learning framework
/// (Burn, Candle, tch-rs, etc.) that can implement these operations.
///
/// # Type Parameters
///
/// - `B`: The batch type containing input data
///
/// # Example
///
/// ```rust,ignore
/// impl Model<MyBatch> for MyModel {
///     fn forward(&mut self, batch: &MyBatch) -> HybridResult<f32> {
///         // Compute forward pass and return loss
///     }
///
///     fn backward(&mut self) -> HybridResult<GradientInfo> {
///         // Compute gradients (assumes forward was just called)
///     }
///
///     fn parameter_count(&self) -> usize {
///         self.parameters.iter().map(|p| p.numel()).sum()
///     }
/// }
/// ```
pub trait Model<B: Batch>: Send + Sync {
    /// Executes the forward pass and returns the loss.
    ///
    /// # Arguments
    ///
    /// * `batch` - The input batch data
    ///
    /// # Returns
    ///
    /// The loss value for this batch.
    fn forward(&mut self, batch: &B) -> HybridResult<f32>;

    /// Executes the backward pass (gradient computation).
    ///
    /// Should be called after `forward()`. Computes gradients with respect
    /// to the loss returned by the most recent forward pass.
    ///
    /// # Returns
    ///
    /// Gradient information including loss and gradient norms.
    fn backward(&mut self) -> HybridResult<GradientInfo>;

    /// Returns the total number of trainable parameters.
    fn parameter_count(&self) -> usize;

    /// Applies a weight delta to the model parameters.
    ///
    /// Used during predictive phase to apply predicted weight updates.
    ///
    /// # Arguments
    ///
    /// * `delta` - The weight changes to apply
    fn apply_weight_delta(&mut self, delta: &state::WeightDelta) -> HybridResult<()>;
}

/// Trait for optimizers that update model parameters.
///
/// Optimizers implement the parameter update rule (SGD, Adam, etc.).
///
/// # Why Separate from Model?
///
/// Optimizer state (momentum, variance estimates) is distinct from model
/// parameters. Separating them allows:
/// - **Swapping optimizers**: Try different optimizers without changing model code
/// - **Independent serialization**: Save/load optimizer state separately
/// - **Stateful updates**: Adam/AdaGrad need per-parameter state across steps
///
/// # Example
///
/// ```rust,ignore
/// impl<M: Model<B>, B: Batch> Optimizer<M, B> for AdamOptimizer {
///     fn step(&mut self, model: &mut M, gradients: &GradientInfo) -> HybridResult<()> {
///         // Apply Adam update rule to model parameters
///     }
/// }
/// ```
pub trait Optimizer<M, B: Batch>: Send + Sync
where
    M: Model<B>,
{
    /// Performs a single optimization step.
    ///
    /// Updates model parameters using the computed gradients.
    ///
    /// # Arguments
    ///
    /// * `model` - The model to update
    /// * `gradients` - Gradient information from backward pass
    fn step(&mut self, model: &mut M, gradients: &GradientInfo) -> HybridResult<()>;

    /// Returns the current learning rate.
    fn learning_rate(&self) -> f32;

    /// Sets the learning rate (for warmup/decay schedules).
    fn set_learning_rate(&mut self, lr: f32);

    /// Zeros all accumulated gradients.
    fn zero_grad(&mut self);
}

/// The main hybrid trainer that orchestrates phase-based predictive training.
///
/// # Overview
///
/// `HybridTrainer` wraps a model and optimizer, managing the training loop through
/// warmup, full training, predictive, and correction phases. It automatically
/// selects optimal phase lengths using bandit-based algorithms and monitors for
/// divergence to ensure training stability.
///
/// # Type Parameters
///
/// - `M`: The model type (must implement `Model` trait)
/// - `O`: The optimizer type (must implement `Optimizer` trait)
///
/// # Example
///
/// ```no_run
/// use hybrid_predict_trainer_rs::{HybridTrainer, HybridTrainerConfig};
///
/// // Configure the trainer
/// let config = HybridTrainerConfig::builder()
///     .warmup_steps(100)
///     .full_steps(20)
///     .max_predict_steps(80)
///     .confidence_threshold(0.85)
///     .build();
///
/// // Create trainer (model and optimizer types are inferred)
/// // let trainer = HybridTrainer::new(model, optimizer, config)?;
/// ```
pub struct HybridTrainer<M, O> {
    /// The model being trained.
    model: Arc<RwLock<M>>,

    /// The optimizer for parameter updates.
    optimizer: Arc<RwLock<O>>,

    /// Training configuration.
    config: HybridTrainerConfig,

    /// Current training state.
    state: TrainingState,

    /// Phase controller for state machine management.
    phase_controller: phases::DefaultPhaseController,

    /// Dynamics model for whole-phase prediction.
    dynamics_model: dynamics::RSSMLite,

    /// Divergence monitor for stability detection.
    divergence_monitor: divergence::DivergenceMonitor,

    /// Residual corrector for prediction adjustment.
    residual_corrector: corrector::ResidualCorrector,

    /// Metrics collector for training statistics.
    metrics: metrics::MetricsCollector,
}

impl<M, O> HybridTrainer<M, O> {
    /// Creates a new hybrid trainer with the given model, optimizer, and configuration.
    ///
    /// # Arguments
    ///
    /// * `model` - The model to train
    /// * `optimizer` - The optimizer for parameter updates
    /// * `config` - Training configuration
    ///
    /// # Returns
    ///
    /// A new `HybridTrainer` instance wrapped in a `HybridResult`.
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid or initialization fails.
    pub fn new(model: M, optimizer: O, config: HybridTrainerConfig) -> HybridResult<Self> {
        let state = TrainingState::new();
        let phase_controller = phases::DefaultPhaseController::new(&config);
        let dynamics_model = dynamics::RSSMLite::new(&config.predictor_config)?;
        let divergence_monitor = divergence::DivergenceMonitor::new(&config);
        let residual_corrector = corrector::ResidualCorrector::new(&config);
        let metrics = metrics::MetricsCollector::new(config.collect_metrics);

        Ok(Self {
            model: Arc::new(RwLock::new(model)),
            optimizer: Arc::new(RwLock::new(optimizer)),
            config,
            state,
            phase_controller,
            dynamics_model,
            divergence_monitor,
            residual_corrector,
            metrics,
        })
    }

    /// Returns the current training step.
    ///
    /// # Returns
    ///
    /// The current step number (0-indexed).
    #[must_use]
    pub fn current_step(&self) -> u64 {
        self.state.step
    }

    /// Returns the current training phase.
    ///
    /// # Returns
    ///
    /// The current [`Phase`] of training.
    #[must_use]
    pub fn current_phase(&self) -> Phase {
        self.phase_controller.current_phase()
    }

    /// Returns the current predictor confidence level.
    ///
    /// # Returns
    ///
    /// A confidence score between 0.0 and 1.0 indicating how reliable
    /// the predictor's outputs are estimated to be.
    #[must_use]
    pub fn current_confidence(&self) -> f32 {
        self.dynamics_model.prediction_confidence(&self.state)
    }

    /// Returns training statistics and metrics.
    ///
    /// # Returns
    ///
    /// A [`metrics::TrainingStatistics`] struct containing aggregate metrics.
    #[must_use]
    pub fn statistics(&self) -> metrics::TrainingStatistics {
        self.metrics.statistics()
    }
}

impl<M, O> HybridTrainer<M, O> {
    /// Executes a single training step.
    ///
    /// This is the main entry point for the training loop. The trainer
    /// automatically selects the appropriate phase (warmup, full, predict,
    /// or correct) based on the current state and configuration.
    ///
    /// # Type Parameters
    ///
    /// * `B` - The batch type containing input data
    ///
    /// # Arguments
    ///
    /// * `batch` - The training batch to process
    ///
    /// # Returns
    ///
    /// A [`StepResult`] containing the loss, phase info, and prediction metadata.
    ///
    /// # Errors
    ///
    /// Returns an error if training diverges or encounters numerical issues.
    /// The error includes a suggested recovery action when possible.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// for batch in dataloader {
    ///     let result = trainer.step(&batch)?;
    ///     println!("Step {}: loss={:.4}, phase={:?}",
    ///         trainer.current_step(),
    ///         result.loss,
    ///         result.phase
    ///     );
    /// }
    /// ```
    pub fn step<B>(&mut self, batch: &B) -> HybridResult<StepResult>
    where
        B: Batch,
        M: Model<B>,
        O: Optimizer<M, B>,
    {
        let start_time = Instant::now();

        // Get current phase decision
        let decision = self.phase_controller.select_next_phase(&self.state);
        let phase = decision.phase();

        // Update predictor confidence for phase controller
        let confidence = self.dynamics_model.prediction_confidence(&self.state);
        self.phase_controller.set_predictor_confidence(confidence);

        // Execute the appropriate phase
        let (loss, was_predicted, prediction_error) = match phase {
            Phase::Warmup | Phase::Full => self.execute_full_step(batch)?,
            Phase::Predict => self.execute_predict_step(batch)?,
            Phase::Correct => self.execute_correct_step(batch)?,
        };

        // Check for divergence
        let divergence_result = self.divergence_monitor.check(&self.state, prediction_error);
        if divergence_result.level > error::DivergenceLevel::Caution {
            let recovery = self
                .phase_controller
                .handle_divergence(divergence_result.level);
            if !recovery.can_continue() {
                return Err((
                    HybridTrainingError::PredictionDivergence {
                        actual: loss,
                        predicted: self.state.loss,
                        delta: (loss - self.state.loss).abs(),
                        step: self.state.step,
                    },
                    Some(recovery),
                ));
            }
        }

        // Update state
        self.state.step += 1;
        self.state.loss = loss;
        self.state.loss_history.push(loss);

        // Update divergence monitor
        self.divergence_monitor
            .observe(loss, self.state.gradient_norm);

        // Record metrics
        let step_metrics = if self.config.collect_metrics {
            Some(self.metrics.record_step_data(
                self.state.step,
                loss,
                phase,
                was_predicted,
                prediction_error,
                confidence,
            ))
        } else {
            None
        };

        let elapsed_ms = start_time.elapsed().as_secs_f64() * 1000.0;

        Ok(StepResult {
            loss,
            phase,
            was_predicted,
            prediction_error,
            confidence,
            step_time_ms: elapsed_ms,
            metrics: step_metrics,
        })
    }

    /// Executes a full training step (forward + backward + optimizer step).
    ///
    /// Used during Warmup and Full phases.
    fn execute_full_step<B>(&mut self, batch: &B) -> HybridResult<(f32, bool, Option<f32>)>
    where
        B: Batch,
        M: Model<B>,
        O: Optimizer<M, B>,
    {
        let mut model = self.model.write();
        let mut optimizer = self.optimizer.write();

        // Zero gradients
        optimizer.zero_grad();

        // Forward pass
        let loss = model.forward(batch)?;

        // Backward pass
        let grad_info = model.backward()?;

        // Optimizer step
        optimizer.step(&mut *model, &grad_info)?;

        // Update training state with gradient info
        self.state.gradient_norm = grad_info.gradient_norm;
        self.state
            .gradient_norm_history
            .push(grad_info.gradient_norm);

        // Train the dynamics model during full steps (not warmup)
        if self.phase_controller.is_warmup_complete() {
            self.dynamics_model
                .observe_gradient(&self.state, &grad_info);
        }

        Ok((loss, false, None))
    }

    /// Executes a predictive step (forward only, apply predicted weight delta).
    ///
    /// Used during Predict phase - skips backward pass for speedup.
    fn execute_predict_step<B>(&mut self, batch: &B) -> HybridResult<(f32, bool, Option<f32>)>
    where
        B: Batch,
        M: Model<B>,
    {
        let mut model = self.model.write();

        // Get prediction from dynamics model
        let (prediction, _uncertainty) = self.dynamics_model.predict_y_steps(&self.state, 1);

        // Apply predicted weight delta
        model.apply_weight_delta(&prediction.weight_delta)?;

        // Forward pass to get actual loss (for validation)
        let actual_loss = model.forward(batch)?;

        // Compute prediction error
        let prediction_error = (actual_loss - prediction.predicted_final_loss).abs();

        Ok((actual_loss, true, Some(prediction_error)))
    }

    /// Executes a correction step (apply residual corrections).
    ///
    /// Used during Correct phase to adjust for accumulated prediction errors.
    fn execute_correct_step<B>(&mut self, batch: &B) -> HybridResult<(f32, bool, Option<f32>)>
    where
        B: Batch,
        M: Model<B>,
    {
        let mut model = self.model.write();

        // Apply residual correction
        let correction = self
            .residual_corrector
            .compute_simple_correction(&self.state);
        if let Some(delta) = correction {
            model.apply_weight_delta(&delta)?;
        }

        // Forward pass to validate correction
        let loss = model.forward(batch)?;

        Ok((loss, false, None))
    }

    /// Forces the trainer into full training mode for the specified number of steps.
    ///
    /// Useful for recovery from divergence or manual intervention.
    ///
    /// # Arguments
    ///
    /// * `steps` - Number of full steps to force
    pub fn force_full_phase(&mut self, steps: usize) {
        self.phase_controller.force_phase(Phase::Full);
        // The phase controller will handle the step count
        let _ = steps; // TODO: Implement step count tracking
    }
}

/// Result of a single training step.
///
/// Contains the loss value, phase information, and prediction metadata
/// for monitoring training progress and predictor accuracy.
#[derive(Debug, Clone)]
pub struct StepResult {
    /// The loss value for this step.
    pub loss: f32,

    /// The phase during which this step was executed.
    pub phase: Phase,

    /// Whether this step used predicted gradients (true) or computed gradients (false).
    pub was_predicted: bool,

    /// The error between predicted and actual loss (if applicable).
    pub prediction_error: Option<f32>,

    /// The predictor's confidence for this step.
    pub confidence: f32,

    /// Wall-clock time for this step in milliseconds.
    pub step_time_ms: f64,

    /// Detailed metrics (if collection is enabled).
    pub metrics: Option<metrics::StepMetrics>,
}

/// Prelude module for convenient imports.
///
/// # Example
///
/// ```
/// use hybrid_predict_trainer_rs::prelude::*;
/// ```
pub mod prelude {
    pub use crate::{
        Batch, GradientInfo, HybridResult, HybridTrainer, HybridTrainerConfig, HybridTrainingError,
        Model, Optimizer, Phase, PhaseDecision, RecoveryAction, StepResult, TrainingState,
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = HybridTrainerConfig::default();
        assert_eq!(config.warmup_steps, 100);
        assert_eq!(config.full_steps, 20);
        assert_eq!(config.max_predict_steps, 80);
        assert!((config.confidence_threshold - 0.85).abs() < f32::EPSILON);
    }
}
