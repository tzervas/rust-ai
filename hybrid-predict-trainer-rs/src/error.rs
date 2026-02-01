//! Error types and recovery actions for hybrid predictive training.
//!
//! This module defines the comprehensive error hierarchy and recovery mechanisms
//! used throughout the hybrid trainer. Errors are designed to be actionable,
//! providing both diagnostic information and suggested recovery actions.
//!
//! # Why Recoverable Errors?
//!
//! Training runs are expensive. Rather than failing on the first anomaly, the
//! hybrid trainer attempts automatic recovery:
//! - **Prediction divergence**: Fall back to full training temporarily
//! - **Gradient explosion**: Reduce learning rate or rollback
//! - **Numerical instability**: Skip batch and continue
//!
//! This design philosophy prioritizes training completion over perfection,
//! enabling unsupervised training runs with self-healing behavior.
//!
//! # Error Categories
//!
//! - **Prediction Errors**: Divergence between predicted and actual outcomes
//! - **Numerical Errors**: NaN, infinity, or gradient explosion/vanishing
//! - **Configuration Errors**: Invalid parameters or incompatible settings
//! - **I/O Errors**: Checkpoint save/load failures
//! - **Integration Errors**: Issues with external crate dependencies
//!
//! # Recovery Actions
//!
//! Many errors include a suggested [`RecoveryAction`] that the trainer can
//! automatically execute to continue training. This enables robust training
//! that self-heals from transient issues.
//!
//! # Example
//!
//! ```rust
//! use hybrid_predict_trainer_rs::error::{HybridTrainingError, RecoveryAction};
//!
//! fn handle_error(error: HybridTrainingError, action: Option<RecoveryAction>) {
//!     match action {
//!         Some(RecoveryAction::ForceFullPhase(steps)) => {
//!             println!("Recovering by forcing {} full steps", steps);
//!         }
//!         Some(RecoveryAction::RollbackAndRetry { checkpoint_step, .. }) => {
//!             println!("Rolling back to step {}", checkpoint_step);
//!         }
//!         None => {
//!             println!("Unrecoverable error: {}", error);
//!         }
//!         _ => {}
//!     }
//! }
//! ```

use thiserror::Error;

/// The main error type for hybrid predictive training.
///
/// Each variant includes relevant context for debugging and, where applicable,
/// suggests a recovery action that can restore training stability.
#[derive(Debug, Error)]
pub enum HybridTrainingError {
    /// Prediction diverged significantly from actual training outcome.
    ///
    /// This occurs when the predicted loss or weight changes deviate
    /// substantially from what was actually observed during validation.
    #[error("Prediction divergence at step {step}: actual loss {actual:.4} vs predicted {predicted:.4} (delta: {delta:.4})")]
    PredictionDivergence {
        /// The actual observed loss.
        actual: f32,
        /// The predicted loss.
        predicted: f32,
        /// The absolute difference.
        delta: f32,
        /// The training step where divergence was detected.
        step: u64,
    },

    /// Numerical instability detected (NaN, infinity, or extreme values).
    ///
    /// Indicates potential issues with learning rate, gradient scaling,
    /// or data preprocessing.
    #[error("Numerical instability at step {step}: {detail}")]
    NumericalInstability {
        /// Description of the instability.
        detail: String,
        /// The training step where instability was detected.
        step: u64,
    },

    /// Gradient explosion detected.
    ///
    /// Gradient norms exceeded safe thresholds, risking training divergence.
    #[error(
        "Gradient explosion at step {step}: norm {norm:.2e} exceeds threshold {threshold:.2e}"
    )]
    GradientExplosion {
        /// The observed gradient norm.
        norm: f32,
        /// The threshold that was exceeded.
        threshold: f32,
        /// The training step.
        step: u64,
    },

    /// Gradient vanishing detected.
    ///
    /// Gradient norms dropped below minimum thresholds, indicating
    /// potential dead neurons or loss saturation.
    #[error("Gradient vanishing at step {step}: norm {norm:.2e} below threshold {threshold:.2e}")]
    GradientVanishing {
        /// The observed gradient norm.
        norm: f32,
        /// The threshold that was not met.
        threshold: f32,
        /// The training step.
        step: u64,
    },

    /// Predictor model training or inference failed.
    ///
    /// The dynamics predictor encountered an internal error.
    #[error("Predictor error: {reason}")]
    PredictorError {
        /// Description of the predictor failure.
        reason: String,
    },

    /// Configuration error (invalid parameters or incompatible settings).
    #[error("Configuration error: {detail}")]
    ConfigError {
        /// Description of the configuration issue.
        detail: String,
    },

    /// Checkpoint save or load failed.
    #[error("Checkpoint error: {reason}")]
    CheckpointError {
        /// Description of the checkpoint failure.
        reason: String,
    },

    /// Integration error with external crate.
    #[error("Integration error with {crate_name}: {detail}")]
    IntegrationError {
        /// Name of the external crate.
        crate_name: String,
        /// Description of the integration issue.
        detail: String,
    },

    /// Phase transition error (invalid state machine transition).
    #[error("Invalid phase transition from {from:?} to {to:?}: {reason}")]
    InvalidPhaseTransition {
        /// The source phase.
        from: crate::Phase,
        /// The attempted destination phase.
        to: crate::Phase,
        /// Why the transition is invalid.
        reason: String,
    },

    /// State encoding/decoding error.
    #[error("State encoding error: {detail}")]
    StateEncodingError {
        /// Description of the encoding failure.
        detail: String,
    },

    /// Memory allocation error.
    #[error("Memory error: {detail}")]
    MemoryError {
        /// Description of the memory issue.
        detail: String,
    },

    /// GPU/CUDA error.
    #[cfg(feature = "cuda")]
    #[error("GPU error: {detail}")]
    GpuError {
        /// Description of the GPU error.
        detail: String,
    },
}

/// Recovery actions that can be taken in response to training errors.
///
/// These actions enable the trainer to self-heal from transient issues
/// without manual intervention.
#[derive(Debug, Clone)]
pub enum RecoveryAction {
    /// Continue training normally (false alarm).
    ///
    /// The error was detected but determined to be non-critical.
    Continue,

    /// Reduce the ratio of predicted steps to full steps.
    ///
    /// Makes training more conservative by computing more gradients.
    ReducePredictRatio(f32),

    /// Force a full training phase for the specified number of steps.
    ///
    /// Temporarily abandons prediction to re-establish stable dynamics.
    ForceFullPhase(usize),

    /// Increase the minimum confidence threshold for predictions.
    ///
    /// Requires higher predictor confidence before using predictions.
    IncreaseConfidenceThreshold(f32),

    /// Rollback to a checkpoint and retry with adjusted parameters.
    ///
    /// Restores training state from a previous checkpoint and optionally
    /// adjusts learning rate or other parameters.
    RollbackAndRetry {
        /// The step to rollback to.
        checkpoint_step: u64,
        /// New learning rate to use (if changed).
        new_learning_rate: f32,
    },

    /// Skip the current batch and continue.
    ///
    /// Useful for data-related issues that don't indicate systemic problems.
    SkipBatch,

    /// Reset the predictor and retrain from scratch.
    ///
    /// Clears predictor state and initiates a new warmup phase.
    ResetPredictor,

    /// Abort training (unrecoverable error).
    ///
    /// The error is too severe to continue; training must be stopped.
    Abort {
        /// Reason for aborting.
        reason: String,
    },
}

impl RecoveryAction {
    /// Returns whether this action allows training to continue.
    ///
    /// # Returns
    ///
    /// `true` if training can proceed after this action, `false` if it must stop.
    #[must_use]
    pub fn can_continue(&self) -> bool {
        !matches!(self, RecoveryAction::Abort { .. })
    }

    /// Returns a human-readable description of the recovery action.
    #[must_use]
    pub fn description(&self) -> String {
        match self {
            Self::Continue => "Continue training normally".to_string(),
            Self::ReducePredictRatio(ratio) => {
                format!("Reduce prediction ratio to {:.1}%", ratio * 100.0)
            }
            Self::ForceFullPhase(steps) => {
                format!("Force {steps} full training steps")
            }
            Self::IncreaseConfidenceThreshold(thresh) => {
                format!("Increase confidence threshold to {thresh:.2}")
            }
            Self::RollbackAndRetry {
                checkpoint_step,
                new_learning_rate,
            } => {
                format!(
                    "Rollback to step {checkpoint_step} with learning rate {new_learning_rate:.2e}"
                )
            }
            Self::SkipBatch => "Skip current batch".to_string(),
            Self::ResetPredictor => "Reset predictor and restart warmup".to_string(),
            Self::Abort { reason } => format!("Abort training: {reason}"),
        }
    }
}

/// Severity levels for divergence events.
///
/// Used by the divergence monitor to categorize the severity of
/// detected anomalies and determine appropriate responses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DivergenceLevel {
    /// Training metrics are within normal ranges.
    Normal,

    /// Metrics are trending toward unusual values.
    ///
    /// Increased monitoring recommended, but no action required yet.
    Caution,

    /// Significant deviation detected.
    ///
    /// Consider falling back to full training or reducing prediction length.
    Warning,

    /// Imminent failure detected (NaN, explosion, etc.).
    ///
    /// Immediate intervention required to prevent training failure.
    Critical,
}

impl DivergenceLevel {
    /// Returns the recommended recovery action for this severity level.
    ///
    /// # Arguments
    ///
    /// * `current_step` - The current training step
    /// * `last_checkpoint` - The most recent checkpoint step
    ///
    /// # Returns
    ///
    /// A suggested recovery action based on severity.
    #[must_use]
    pub fn suggested_action(&self, current_step: u64, last_checkpoint: u64) -> RecoveryAction {
        match self {
            Self::Normal => RecoveryAction::Continue,
            Self::Caution => RecoveryAction::ReducePredictRatio(0.5),
            Self::Warning => RecoveryAction::ForceFullPhase(50),
            Self::Critical => {
                if current_step > last_checkpoint {
                    RecoveryAction::RollbackAndRetry {
                        checkpoint_step: last_checkpoint,
                        new_learning_rate: 0.0, // Caller should determine actual LR
                    }
                } else {
                    RecoveryAction::Abort {
                        reason: "Critical divergence with no valid checkpoint".to_string(),
                    }
                }
            }
        }
    }
}

/// Result type that includes both the error and a suggested recovery action.
///
/// This allows error handlers to both diagnose the issue and take corrective
/// action without additional computation.
pub type HybridResult<T> = Result<T, (HybridTrainingError, Option<RecoveryAction>)>;

/// Extension trait for converting standard Results to `HybridResults`.
pub trait IntoHybridResult<T> {
    /// Converts this result into a `HybridResult` with no recovery action.
    fn into_hybrid(self) -> HybridResult<T>;

    /// Converts this result into a `HybridResult` with the specified recovery action.
    fn with_recovery(self, action: RecoveryAction) -> HybridResult<T>;
}

impl<T, E: Into<HybridTrainingError>> IntoHybridResult<T> for Result<T, E> {
    fn into_hybrid(self) -> HybridResult<T> {
        self.map_err(|e| (e.into(), None))
    }

    fn with_recovery(self, action: RecoveryAction) -> HybridResult<T> {
        self.map_err(|e| (e.into(), Some(action)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_divergence_level_ordering() {
        assert!(DivergenceLevel::Normal < DivergenceLevel::Caution);
        assert!(DivergenceLevel::Caution < DivergenceLevel::Warning);
        assert!(DivergenceLevel::Warning < DivergenceLevel::Critical);
    }

    #[test]
    fn test_recovery_action_can_continue() {
        assert!(RecoveryAction::Continue.can_continue());
        assert!(RecoveryAction::ForceFullPhase(10).can_continue());
        assert!(!RecoveryAction::Abort {
            reason: "test".to_string()
        }
        .can_continue());
    }

    #[test]
    fn test_suggested_action_for_critical() {
        let action = DivergenceLevel::Critical.suggested_action(100, 50);
        matches!(
            action,
            RecoveryAction::RollbackAndRetry {
                checkpoint_step: 50,
                ..
            }
        );
    }
}
