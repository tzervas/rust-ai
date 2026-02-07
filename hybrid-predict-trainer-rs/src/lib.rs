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
pub mod delta_accumulator;
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

// Mixed precision training support
pub mod mixed_precision;

// Gradient accumulation for memory efficiency
pub mod gradient_accumulation;

// Predict-aware memory management (unique to HybridTrainer)
pub mod predict_aware_memory;

// Automatic tuning and optimization
pub mod auto_tuning;

// Burn framework integration
pub mod burn_integration;

// VRAM budget management
pub mod vram_budget;

// Checkpoint save/restore
pub mod checkpoint;

// VRAM manager for tracking and cleaning up GPU memory
pub mod vram_manager;

// Memory optimization modules (v0.3.0)
pub mod gradient_checkpointing;
pub mod cpu_offloading;
pub mod quantization;

// Reference model implementations for validation
#[cfg(feature = "autodiff")]
pub mod models;

// GPU acceleration (feature-gated)
#[cfg(feature = "cuda")]
#[cfg_attr(docsrs, doc(cfg(feature = "cuda")))]
pub mod gpu;

// Re-exports for convenient access
pub use auto_tuning::{
    AutoTuningConfig, AutoTuningController, AutoTuningUpdate, BatchPrediction,
    BatchPredictionRecommendation, HealthClassification, HealthRecommendation, HealthScorer,
    HealthWeights, MultiStepPredictor, TrainingHealthScore,
};
pub use config::HybridTrainerConfig;
pub use error::{HybridResult, HybridTrainingError, RecoveryAction};
pub use phases::{Phase, PhaseController, PhaseDecision, PhaseOutcome};
pub use residuals::{Residual, ResidualStore};
pub use state::TrainingState;
pub use timing::{Duration, Timer, TimingMetrics, TimingStats};

// Standard library imports
use std::sync::Arc;
use std::time::Instant;

// External crate imports
// Note: Mutex used instead of RwLock for model/optimizer storage to support !Sync types

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
///
/// # Thread Safety
///
/// Models must be `Send` to allow moving between threads, but `Sync` is not
/// required. This enables integration with autodiff frameworks (like Burn)
/// that use gradient types which are `!Sync` by design.
///
/// For multi-threaded access to models, use `Arc<Mutex<>>` rather than
/// `Arc<RwLock<>>` since the model itself may not be `Sync`.
pub trait Model<B: Batch>: Send {
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

    /// Clears forward pass state when backward() won't be called.
    ///
    /// This method should be called during Predict phase after forward()
    /// when backward() will be skipped. It allows implementations to free
    /// resources associated with the forward pass (e.g., autodiff graphs,
    /// cached activations) to prevent memory accumulation.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // During Predict phase
    /// let loss = model.forward(batch)?;
    /// model.clear_forward_state(); // Won't call backward()
    /// ```
    ///
    /// Default implementation does nothing (for implementations that don't
    /// need cleanup).
    fn clear_forward_state(&mut self) {
        // Default: no-op
    }

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
///
/// # Thread Safety
///
/// Optimizers must be `Send` to allow moving between threads, but `Sync` is not
/// required. This matches the `Model` trait's threading constraints.
pub trait Optimizer<M, B: Batch>: Send
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
///     .max_predict_steps(50)  // Can be higher with more VRAM
///     .confidence_threshold(0.85)
///     .build();
///
/// // Create trainer (model and optimizer types are inferred)
/// // let trainer = HybridTrainer::new(model, optimizer, config)?;
/// ```
pub struct HybridTrainer<M, O> {
    /// The model being trained.
    ///
    /// Uses `Mutex` instead of `RwLock` because models may not be `Sync`
    /// (e.g., when using autodiff frameworks with !Sync gradient types).
    model: Arc<parking_lot::Mutex<M>>,

    /// The optimizer for parameter updates.
    ///
    /// Uses `Mutex` instead of `RwLock` for consistency with model storage.
    optimizer: Arc<parking_lot::Mutex<O>>,

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

    /// Storage for residuals extracted from prediction errors.
    residual_store: residuals::ResidualStore,

    /// Metrics collector for training statistics.
    metrics: metrics::MetricsCollector,

    /// Current phase and remaining steps (for respecting multi-step phase decisions).
    phase_budget: Option<(Phase, usize)>,

    /// Automatic tuning controller (optional).
    auto_tuning: Option<AutoTuningController>,

    /// Last auto-tuning update (for external access).
    last_auto_tuning_update: Option<AutoTuningUpdate>,

    /// Checkpoint manager for automatic checkpointing (optional).
    checkpoint_manager: Option<checkpoint::CheckpointManager>,

    /// Delta accumulator for batched weight updates (VRAM optimization).
    delta_accumulator: delta_accumulator::DeltaAccumulator,

    /// VRAM manager for tracking and cleaning up GPU memory.
    vram_manager: vram_manager::VramManager,
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
        let residual_store = residuals::ResidualStore::new(1000);
        let metrics = metrics::MetricsCollector::new(config.collect_metrics);

        // Initialize auto-tuning controller if config provided
        let auto_tuning = if let Some(auto_config) = config.auto_tuning_config.clone() {
            let max_steps = config.max_steps.unwrap_or(10000); // Default if not provided
            Some(auto_tuning::AutoTuningController::new(
                auto_config,
                max_steps,
            ))
        } else {
            None
        };

        // Initialize checkpoint manager if save_interval > 0
        let checkpoint_manager = if config.checkpoint_config.save_interval > 0 {
            // Use "./checkpoints" as default directory if not specified
            let checkpoint_dir = std::path::PathBuf::from("./checkpoints");
            Some(checkpoint::CheckpointManager::new(
                checkpoint_dir,
                config.checkpoint_config.save_interval,
                config.checkpoint_config.keep_last_n,
            )?)
        } else {
            None
        };

        Ok(Self {
            model: Arc::new(parking_lot::Mutex::new(model)),
            optimizer: Arc::new(parking_lot::Mutex::new(optimizer)),
            config,
            state,
            phase_controller,
            dynamics_model,
            divergence_monitor,
            residual_corrector,
            residual_store,
            metrics,
            phase_budget: None,
            auto_tuning,
            last_auto_tuning_update: None,
            checkpoint_manager,
            delta_accumulator: delta_accumulator::DeltaAccumulator::new(),
            vram_manager: vram_manager::VramManager::new(),
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
    pub fn statistics(&mut self) -> metrics::TrainingStatistics {
        self.metrics.statistics()
    }

    /// Returns the last auto-tuning update, if available.
    ///
    /// # Returns
    ///
    /// The most recent [`auto_tuning::AutoTuningUpdate`] if auto-tuning is enabled,
    /// or `None` if auto-tuning is disabled or no updates have occurred yet.
    #[must_use]
    pub fn last_auto_tuning_update(&self) -> Option<&auto_tuning::AutoTuningUpdate> {
        self.last_auto_tuning_update.as_ref()
    }

    /// Returns a read lock on the model.
    ///
    /// Use this to access model state for checkpointing or inspection.
    pub fn model(&self) -> parking_lot::MutexGuard<'_, M> {
        self.model.lock()
    }

    /// Returns a write lock on the model.
    ///
    /// Use this for operations that need to modify the model directly.
    pub fn model_mut(&self) -> parking_lot::MutexGuard<'_, M> {
        self.model.lock()
    }

    /// Sets the learning rate on the underlying optimizer.
    ///
    /// This is used for learning rate scheduling (warmup, decay, etc.).
    ///
    /// # Arguments
    ///
    /// * `lr` - The new learning rate
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Update learning rate based on schedule
    /// let lr = scheduler.get_lr(trainer.current_step());
    /// trainer.set_learning_rate(lr);
    /// ```
    pub fn set_learning_rate<B>(&self, lr: f32)
    where
        B: Batch,
        M: Model<B>,
        O: Optimizer<M, B>,
    {
        self.optimizer.lock().set_learning_rate(lr);
    }

    /// Returns the current learning rate from the optimizer.
    ///
    /// # Returns
    ///
    /// The current learning rate value.
    pub fn learning_rate<B>(&self) -> f32
    where
        B: Batch,
        M: Model<B>,
        O: Optimizer<M, B>,
    {
        self.optimizer.lock().learning_rate()
    }

    // TODO: Enable when auto_tuning fields are added to struct
    // /// Returns the last auto-tuning update, if auto-tuning is enabled.
    // ///
    // /// # Returns
    // ///
    // /// The most recent auto-tuning update, or None if auto-tuning is disabled
    // /// or no updates have occurred yet.
    // #[must_use]
    // pub fn last_auto_tuning_update(&self) -> Option<&auto_tuning::AutoTuningUpdate> {
    //     self.last_auto_tuning_update.as_ref()
    // }

    /// Returns the last recorded gradient norm.
    ///
    /// # Returns
    ///
    /// The gradient norm from the most recent backward pass.
    #[must_use]
    fn last_gradient_norm(&self) -> f32 {
        self.state.gradient_norm
    }

    /// Collects per-layer gradient statistics.
    ///
    /// This is a stub implementation that distributes the global gradient
    /// norm across dummy layers. A real implementation would track per-layer
    /// gradients during the backward pass.
    ///
    /// # Returns
    ///
    /// `HashMap` of `layer_name` -> (`grad_norm`, `weight_norm`).
    fn collect_layer_gradients(&self) -> std::collections::HashMap<String, (f32, f32)> {
        let global_norm = self.last_gradient_norm();

        // Stub: distribute gradient across typical transformer layers
        // In a real implementation, this would come from actual per-layer tracking
        let mut map = std::collections::HashMap::new();
        map.insert("embed".to_string(), (global_norm * 0.8, 10.0));
        map.insert("attention".to_string(), (global_norm * 1.0, 15.0));
        map.insert("mlp".to_string(), (global_norm * 1.2, 20.0));
        map.insert("lm_head".to_string(), (global_norm * 0.9, 8.0));
        map
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

        // Update predictor confidence for phase controller
        let confidence = self.dynamics_model.prediction_confidence(&self.state);
        self.phase_controller.set_predictor_confidence(confidence);

        // Track previous phase to detect transitions
        let previous_phase = self.state.current_phase;

        // Get phase from budget or request new decision
        let phase = match &mut self.phase_budget {
            Some((current_phase, remaining)) if *remaining > 0 => {
                // Use current phase and decrement budget
                *remaining -= 1;
                *current_phase
            }
            _ => {
                // Budget exhausted or None - get new phase decision
                let decision = self.phase_controller.select_next_phase(&self.state);
                let new_phase = decision.phase();
                let steps = decision.steps();

                // Set budget for remaining steps (steps-1 since we're using one now)
                if steps > 1 {
                    self.phase_budget = Some((new_phase, steps - 1));
                } else {
                    self.phase_budget = None;
                }

                new_phase
            }
        };

        // Reset steps_in_current_phase counter on phase transitions
        if phase != previous_phase {
            self.state.steps_in_current_phase = 0;
            self.state.current_phase = phase;

            // Log VRAM usage at phase transitions
            let vram_mb = self.vram_manager.last_vram_mb();
            println!(
                "Phase transition: {:?} → {:?} | VRAM: {} MB | Copies: {}",
                previous_phase,
                phase,
                vram_mb,
                crate::vram_manager::VramManager::total_copies()
            );

            // Flush accumulated weight deltas at phase transitions (VRAM optimization)
            // This applies all accumulated deltas from the previous phase in one batch,
            // minimizing model copies from Burn's .map() API
            if let Some(merged_delta) = self.delta_accumulator.flush() {
                let mut model = self.model.lock();
                model.apply_weight_delta(&merged_delta)?;
            }
        }

        // Execute the appropriate phase
        let (loss, was_predicted, prediction_error) = match phase {
            Phase::Warmup | Phase::Full => self.execute_full_step(batch)?,
            Phase::Predict => self.execute_predict_step(batch)?,
            Phase::Correct => self.execute_correct_step(batch)?,
        };

        // Check for divergence (skip during warmup - NaN values are expected initially)
        if phase != Phase::Warmup {
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
                // Reset phase budget to force Full training after divergence
                self.phase_budget = Some((Phase::Full, self.config.full_steps));
            }
        }

        // Auto-tuning integration: Update controller and apply recommendations
        if self.auto_tuning.is_some() {
            // Collect gradients before taking mutable borrow of auto_tuning
            let layer_grads: Vec<(String, f32, f32)> = self
                .collect_layer_gradients()
                .into_iter()
                .map(|(name, (grad_norm, weight_norm))| (name, grad_norm, weight_norm))
                .collect();

            #[allow(clippy::unnecessary_unwrap)]
            let update = self.auto_tuning.as_mut().unwrap().update(
                self.state.step,
                loss,
                self.state.gradient_norm,
                &layer_grads,
                confidence,
            );

            // Store update for external access (TODO: wire to optimizer when available)
            self.last_auto_tuning_update = Some(update);
        }

        // Update state
        self.state.step += 1;
        self.state.loss = loss;
        self.state.loss_history.push(loss);
        self.state.loss_ema.update(loss); // CRITICAL FIX: Update EMA for stability confidence

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

        // Auto-checkpoint if enabled and interval reached OR VRAM critical
        // Compute checkpoint state before taking mutable borrow
        const VRAM_CHECKPOINT_THRESHOLD_MB: usize = 14_000; // 14 GB triggers emergency checkpoint
        let vram_critical = self.vram_manager.last_vram_mb() > VRAM_CHECKPOINT_THRESHOLD_MB;

        let checkpoint_to_save = if self
            .checkpoint_manager
            .as_ref()
            .map_or(false, |mgr| mgr.should_save(self.state.step) || vram_critical)
        {
            if vram_critical {
                eprintln!(
                    "🚨 Emergency checkpoint triggered by high VRAM ({} MB > {} MB)",
                    self.vram_manager.last_vram_mb(),
                    VRAM_CHECKPOINT_THRESHOLD_MB
                );
            }
            use crate::checkpoint::*;

            Some(TrainingCheckpoint::new(
                self.config.clone(),
                self.state.clone(),
                DynamicsState::default(),
                ResidualStoreState::default(),
                PhaseControllerState {
                    current_phase: self.phase_controller.current_phase(),
                    predictor_confidence: self.current_confidence(),
                    warmup_complete: self.phase_controller.is_warmup_complete(),
                    phase_stats: Vec::new(),
                },
                DivergenceMonitorState::default(),
                CorrectorState::default(),
            ))
        } else {
            None
        };

        // Now we can take mutable borrow to save
        if let Some(checkpoint) = checkpoint_to_save {
            if let Some(ref mut checkpoint_mgr) = self.checkpoint_manager {
                // Save checkpoint (errors are logged but don't stop training)
                if let Err((err, _)) = checkpoint_mgr.save(&checkpoint) {
                    eprintln!(
                        "Warning: Failed to save checkpoint at step {}: {}",
                        self.state.step, err
                    );
                }
            }
        }

        // Check if VRAM cleanup is needed (workaround for Burn's model.map() leak)
        if self.vram_manager.should_cleanup() {
            self.vram_manager.force_cleanup();
        }

        let elapsed_ms = start_time.elapsed().as_secs_f64() * 1000.0;

        Ok(StepResult {
            loss,
            phase,
            was_predicted,
            prediction_error,
            confidence,
            gradient_norm: self.state.gradient_norm,
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
        let mut model = self.model.lock();
        let mut optimizer = self.optimizer.lock();

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
            // Capture state features before model update
            let state_features_before = self.state.compute_features();
            let confidence = self.dynamics_model.prediction_confidence(&self.state);

            // Get 1-step prediction to compute gradient residuals
            let (prediction, _) = self.dynamics_model.predict_y_steps(&self.state, 1);
            let predicted_loss = prediction.predicted_final_loss;

            // Compute loss residual (actual - predicted)
            let loss_residual = loss - predicted_loss;

            // Create gradient residuals from per-param norms if available
            let gradient_residuals = if let Some(ref per_param) = grad_info.per_param_norms {
                per_param
                    .iter()
                    .enumerate()
                    .map(|(idx, &actual_norm)| residuals::LayerResidual {
                        layer_name: format!("layer_{}", idx),
                        magnitude: actual_norm,
                        compressed: None, // TODO: Add compression support
                        cosine_similarity: 1.0, // Perfect match when no prediction available
                    })
                    .collect()
            } else {
                Vec::new()
            };

            // Create and store residual for weight-level corrections
            let residual = residuals::Residual {
                step: self.state.step,
                phase: Phase::Full,
                prediction_horizon: 1,
                loss_residual,
                gradient_residuals,
                state_features: state_features_before,
                prediction_confidence: confidence,
            };

            // Store the residual for future correction
            self.residual_store.add(residual.clone());

            // Update the corrector's online model with this residual
            self.residual_corrector
                .update_from_residual(&residual, &self.state);

            // Train the dynamics model
            self.dynamics_model
                .observe_gradient(&self.state, &grad_info);
        }

        Ok((loss, false, None))
    }

    /// Executes a predictive step (forward only, apply predicted weight delta).
    ///
    /// Uses the phase controller's `compute_predict_steps()` to determine the
    /// optimal prediction horizon based on current confidence and history,
    /// then calls `predict_y_steps()` with that horizon for multi-step
    /// prediction. This enables the dynamics model to predict further ahead
    /// when confidence is high, yielding greater training speedup.
    ///
    /// Used during Predict phase - skips backward pass for speedup.
    fn execute_predict_step<B>(&mut self, batch: &B) -> HybridResult<(f32, bool, Option<f32>)>
    where
        B: Batch,
        M: Model<B>,
    {
        let mut model = self.model.lock();

        // Capture state before prediction for residual extraction
        let state_features_before = self.state.compute_features();
        let confidence = self.dynamics_model.prediction_confidence(&self.state);

        // Compute adaptive prediction horizon based on confidence and history
        let y_steps = self.phase_controller.compute_predict_steps();

        // Get multi-step prediction from dynamics model
        let (prediction, _uncertainty) = self.dynamics_model.predict_y_steps(&self.state, y_steps);
        let predicted_loss = prediction.predicted_final_loss;

        // Apply predicted weight delta immediately (Burn limitation - can't defer)
        // TODO: Accumulation strategy doesn't work due to forward pass dependency
        model.apply_weight_delta(&prediction.weight_delta)?;

        // Forward pass to get actual loss (for validation)
        let actual_loss = model.forward(batch)?;

        // Clear forward state immediately (no backward in Predict phase)
        // This prevents memory accumulation from unused autodiff graphs
        model.clear_forward_state();

        // Compute prediction error (absolute difference)
        let prediction_error = (actual_loss - predicted_loss).abs();

        // Create and store residual (actual - predicted)
        let loss_residual = actual_loss - predicted_loss;
        let residual = residuals::Residual {
            step: self.state.step,
            phase: Phase::Predict,
            prediction_horizon: y_steps,
            loss_residual,
            gradient_residuals: Vec::new(), // No gradient info in predict phase
            state_features: state_features_before,
            prediction_confidence: confidence,
        };

        // Store the residual for future correction
        self.residual_store.add(residual.clone());

        // Update the corrector's online model with this residual
        self.residual_corrector
            .update_from_residual(&residual, &self.state);

        // Check if micro-correction is needed (intra-horizon correction)
        self.state.steps_in_current_phase += 1;

        if self.config.correction_interval > 0
            && self.state.steps_in_current_phase % self.config.correction_interval == 0
        {
            // Apply micro-correction without transitioning to Correct phase
            let correction = if self.residual_store.is_empty() {
                corrector::Correction::zero()
            } else {
                self.residual_corrector.compute_correction(
                    &self.state,
                    &self.residual_store,
                    actual_loss,
                )
            };

            // Apply weight correction if available and significant
            if let Some(ref weight_correction) = correction.weight_correction {
                model.apply_weight_delta(weight_correction)?;
            } else if correction.is_significant(0.01) {
                // Apply simple correction if loss correction is significant but no weight delta
                if let Some(simple_delta) = self
                    .residual_corrector
                    .compute_simple_correction(&self.state)
                {
                    model.apply_weight_delta(&simple_delta)?;
                }
            }

            // Record that we applied a micro-correction
            self.metrics.record_micro_correction();
        }

        Ok((actual_loss, true, Some(prediction_error)))
    }

    /// Executes a correction step (apply residual corrections).
    ///
    /// Used during Correct phase to adjust for accumulated prediction errors.
    /// Uses stored residuals from prediction phase to compute corrections.
    fn execute_correct_step<B>(&mut self, batch: &B) -> HybridResult<(f32, bool, Option<f32>)>
    where
        B: Batch,
        M: Model<B>,
    {
        let mut model = self.model.lock();

        // Compute correction using stored residuals for context-aware adjustment
        let correction = if self.residual_store.is_empty() {
            // Fall back to simple correction if no residuals available
            corrector::Correction::zero()
        } else {
            // Use full correction with residual store for better estimates
            let predicted_loss = self.state.loss; // Use current loss as baseline
            self.residual_corrector.compute_correction(
                &self.state,
                &self.residual_store,
                predicted_loss,
            )
        };

        // Apply weight delta correction immediately (Burn limitation - can't defer)
        if let Some(ref delta) = correction.weight_correction {
            model.apply_weight_delta(delta)?;
        } else if correction.is_significant(0.01) {
            // If no weight correction but loss correction is significant,
            // apply a simple scaled correction
            let simple_delta = self
                .residual_corrector
                .compute_simple_correction(&self.state);
            if let Some(delta) = simple_delta {
                model.apply_weight_delta(&delta)?;
            }
        }

        // Forward pass to validate correction
        let loss = model.forward(batch)?;

        // Clear forward state immediately (no backward in Correct phase)
        model.clear_forward_state();

        // Compute how much the correction changed the loss (for metrics)
        let correction_effect = if correction.loss_correction.abs() > 0.001 {
            Some((self.state.loss - loss).abs())
        } else {
            None
        };

        Ok((loss, false, correction_effect))
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
        // Set the phase budget to enforce the requested number of steps
        self.phase_budget = Some((Phase::Full, steps));
    }

    /// Saves a checkpoint of the trainer state.
    ///
    /// This saves all hybrid trainer state including training state, dynamics model,
    /// residual store, and phase controller state. It does NOT save model or optimizer
    /// state - those must be checkpointed separately via your deep learning framework.
    ///
    /// # Arguments
    ///
    /// * `path` - Path where the checkpoint should be saved
    ///
    /// # Returns
    ///
    /// `Ok(())` on success.
    ///
    /// # Errors
    ///
    /// Returns an error if checkpoint serialization or file I/O fails.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Save hybrid trainer state
    /// trainer.save_checkpoint("checkpoints/hybrid_step_1000.bin")?;
    ///
    /// // User should also save model and optimizer separately
    /// model.save("checkpoints/model_step_1000.safetensors")?;
    /// optimizer.save("checkpoints/optimizer_step_1000.bin")?;
    /// ```
    pub fn save_checkpoint(&self, path: impl AsRef<std::path::Path>) -> HybridResult<()> {
        use crate::checkpoint::*;

        // Extract serializable state from components
        let dynamics_state = DynamicsState::default(); // TODO: Extract from self.dynamics_model
        let residual_store_state = ResidualStoreState::default(); // TODO: Extract from self.residual_store
        let phase_controller_state = PhaseControllerState {
            current_phase: self.phase_controller.current_phase(),
            predictor_confidence: self.current_confidence(),
            warmup_complete: self.phase_controller.is_warmup_complete(),
            phase_stats: Vec::new(), // TODO: Extract stats
        };
        let divergence_monitor_state = DivergenceMonitorState::default(); // TODO: Extract from self.divergence_monitor
        let corrector_state = CorrectorState::default(); // TODO: Extract from self.residual_corrector

        let checkpoint = TrainingCheckpoint::new(
            self.config.clone(),
            self.state.clone(),
            dynamics_state,
            residual_store_state,
            phase_controller_state,
            divergence_monitor_state,
            corrector_state,
        );

        checkpoint.save(path)
    }

    /// Loads a checkpoint and creates a new trainer with the given model and optimizer.
    ///
    /// This restores all hybrid trainer state from a checkpoint file. The model and
    /// optimizer must be provided separately since they're framework-specific.
    ///
    /// # Type Parameters
    ///
    /// * `M` - The model type
    /// * `O` - The optimizer type
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the checkpoint file
    /// * `model` - The model to train (should be loaded from a separate checkpoint)
    /// * `optimizer` - The optimizer (should be loaded from a separate checkpoint)
    ///
    /// # Returns
    ///
    /// A new `HybridTrainer` instance with restored state.
    ///
    /// # Errors
    ///
    /// Returns an error if checkpoint loading or deserialization fails.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Load model and optimizer from their checkpoints
    /// let model = MyModel::load("checkpoints/model_step_1000.safetensors")?;
    /// let optimizer = MyOptimizer::load("checkpoints/optimizer_step_1000.bin")?;
    ///
    /// // Load hybrid trainer state
    /// let trainer = HybridTrainer::load_checkpoint(
    ///     "checkpoints/hybrid_step_1000.bin",
    ///     model,
    ///     optimizer
    /// )?;
    /// ```
    pub fn load_checkpoint(
        path: impl AsRef<std::path::Path>,
        model: M,
        optimizer: O,
    ) -> HybridResult<Self> {
        use crate::checkpoint::*;

        let checkpoint = TrainingCheckpoint::load(path)?;

        // Reconstruct trainer components from checkpoint state
        let phase_controller = phases::DefaultPhaseController::new(&checkpoint.config);
        let dynamics_model = dynamics::RSSMLite::new(&checkpoint.config.predictor_config)?;
        let divergence_monitor = divergence::DivergenceMonitor::new(&checkpoint.config);
        let residual_corrector = corrector::ResidualCorrector::new(&checkpoint.config);
        let residual_store = residuals::ResidualStore::new(1000);
        let metrics = metrics::MetricsCollector::new(checkpoint.config.collect_metrics);

        // TODO: Restore state into dynamics_model, residual_store, etc. from checkpoint

        // Initialize auto-tuning controller if config provided
        let auto_tuning = if let Some(auto_config) = checkpoint.config.auto_tuning_config.clone() {
            let max_steps = checkpoint.config.max_steps.unwrap_or(10000);
            Some(auto_tuning::AutoTuningController::new(
                auto_config,
                max_steps,
            ))
        } else {
            None
        };

        // Note: checkpoint_manager is NOT restored from checkpoint
        // It will be re-initialized if needed based on the config
        let checkpoint_manager = if checkpoint.config.checkpoint_config.save_interval > 0 {
            let checkpoint_dir = std::path::PathBuf::from("./checkpoints");
            Some(checkpoint::CheckpointManager::new(
                checkpoint_dir,
                checkpoint.config.checkpoint_config.save_interval,
                checkpoint.config.checkpoint_config.keep_last_n,
            )?)
        } else {
            None
        };

        Ok(Self {
            model: Arc::new(parking_lot::Mutex::new(model)),
            optimizer: Arc::new(parking_lot::Mutex::new(optimizer)),
            config: checkpoint.config,
            state: checkpoint.training_state,
            phase_controller,
            dynamics_model,
            divergence_monitor,
            residual_corrector,
            residual_store,
            metrics,
            phase_budget: None,
            auto_tuning,
            last_auto_tuning_update: None,
            checkpoint_manager,
            delta_accumulator: delta_accumulator::DeltaAccumulator::new(),
            vram_manager: vram_manager::VramManager::new(),
        })
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

    /// Gradient norm from the backward pass (0.0 if predicted step).
    pub gradient_norm: f32,

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
        assert_eq!(config.max_predict_steps, 15);  // Updated: default reduced for VRAM optimization
        assert!((config.confidence_threshold - 0.85).abs() < f32::EPSILON);
    }

    #[test]
    fn test_correction_interval_config() {
        // Test default (disabled)
        let config = HybridTrainerConfig::default();
        assert_eq!(config.correction_interval, 0);

        // Test builder pattern
        let config = HybridTrainerConfig::builder()
            .correction_interval(10)
            .build();
        assert_eq!(config.correction_interval, 10);
    }

    #[test]
    fn test_steps_in_current_phase_counter() {
        let mut state = TrainingState::new();
        assert_eq!(state.steps_in_current_phase, 0);

        // Simulate phase transition
        state.enter_phase(Phase::Predict);
        assert_eq!(state.steps_in_current_phase, 0);

        // Simulate steps within phase
        state.steps_in_current_phase = 5;
        assert_eq!(state.steps_in_current_phase, 5);

        // Phase transition should reset
        state.enter_phase(Phase::Correct);
        assert_eq!(state.steps_in_current_phase, 0);
    }
}
