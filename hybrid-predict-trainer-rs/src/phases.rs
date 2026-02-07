//! Phase state machine and execution control.
//!
//! This module implements the state machine that governs transitions between
//! training phases (Warmup → Full → Predict → Correct) and provides the
//! controller interface for making phase decisions.
//!
//! # Why a State Machine?
//!
//! Training dynamics evolve through distinct regimes that require different
//! handling. A formal state machine ensures:
//! - **Correctness**: Invalid transitions are rejected at compile/runtime
//! - **Predictability**: Phase behavior is deterministic and testable
//! - **Extensibility**: New phases can be added without breaking existing logic
//!
//! # Phase Descriptions
//!
//! - **Warmup**: Initial training to establish baseline dynamics and train predictor
//! - **Full**: Traditional forward/backward computation for ground truth gradients
//! - **Predict**: Skip backward pass using learned dynamics model
//! - **Correct**: Apply residual corrections to maintain prediction accuracy
//!
//! # State Machine
//!
//! ```text
//! WARMUP ──(warmup_complete)──▶ FULL
//!                                │
//!            ┌───────────────────┘
//!            │
//!            ▼
//!   ┌─────────────────────┐
//!   │                     │
//!   ▼                     │
//! FULL ──(confidence_ok)──▶ PREDICT
//!   │                        │
//!   │◀─(low_confidence)──────┤
//!   │                        │
//!   │◀─(divergence)──────────┤
//!   │                        │
//!   │                        ▼
//!   │                     CORRECT
//!   │                        │
//!   │◀───────────────────────┘
//!   │
//!   └──(repeat)──▶ FULL
//! ```

use serde::{Deserialize, Serialize};

use crate::config::HybridTrainerConfig;
use crate::error::{DivergenceLevel, HybridResult, HybridTrainingError, RecoveryAction};
use crate::state::TrainingState;

/// Training phases in the hybrid predictive training loop.
///
/// Each phase has distinct characteristics and purposes within the
/// overall training strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Phase {
    /// Warmup phase: establish baseline dynamics.
    ///
    /// During warmup, only full training is performed. The predictor
    /// observes these steps to learn the initial training dynamics.
    /// No predictions are made during this phase.
    Warmup,

    /// Full training phase: compute actual gradients.
    ///
    /// Traditional forward and backward passes are executed. The
    /// results are used to:
    /// 1. Update model weights
    /// 2. Train the dynamics predictor
    /// 3. Collect residuals for correction
    Full,

    /// Predictive phase: use learned dynamics.
    ///
    /// The backward pass is skipped. Instead, the dynamics model
    /// predicts weight updates based on the current state. This
    /// provides significant speedup at the cost of some accuracy.
    Predict,

    /// Correction phase: apply residual corrections.
    ///
    /// After prediction, residuals from previous full phases are
    /// applied to correct accumulated prediction errors. This
    /// maintains training quality without full gradient computation.
    Correct,
}

impl Phase {
    /// Returns a human-readable name for the phase.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Phase::Warmup => "warmup",
            Phase::Full => "full",
            Phase::Predict => "predict",
            Phase::Correct => "correct",
        }
    }

    /// Returns whether backward passes are computed in this phase.
    #[must_use]
    pub fn computes_backward(&self) -> bool {
        matches!(self, Phase::Warmup | Phase::Full)
    }

    /// Returns whether predictions are used in this phase.
    #[must_use]
    pub fn uses_predictions(&self) -> bool {
        matches!(self, Phase::Predict)
    }
}

/// Decision about which phase to execute next.
///
/// Returned by the phase controller to indicate what the trainer
/// should do, including phase type and duration.
#[derive(Debug, Clone)]
pub enum PhaseDecision {
    /// Execute warmup phase.
    Warmup {
        /// Number of warmup steps to execute.
        steps: usize,
    },

    /// Execute full training phase.
    Full {
        /// Number of full steps to execute.
        steps: usize,
    },

    /// Execute predictive phase.
    Predict {
        /// Maximum number of prediction steps.
        steps: usize,
        /// Current predictor confidence.
        confidence: f32,
        /// Loss threshold at which to abort prediction.
        fallback_threshold: f32,
    },

    /// Execute correction phase.
    Correct {
        /// Number of validation samples for correction.
        validation_samples: usize,
        /// Maximum magnitude of correction to apply.
        max_correction_magnitude: f32,
    },
}

impl PhaseDecision {
    /// Returns the phase type for this decision.
    #[must_use]
    pub fn phase(&self) -> Phase {
        match self {
            PhaseDecision::Warmup { .. } => Phase::Warmup,
            PhaseDecision::Full { .. } => Phase::Full,
            PhaseDecision::Predict { .. } => Phase::Predict,
            PhaseDecision::Correct { .. } => Phase::Correct,
        }
    }

    /// Returns the number of steps for this phase.
    ///
    /// Each phase decision variant stores its step count differently, but this
    /// method provides a uniform interface for callers who just need the count.
    #[must_use]
    #[allow(clippy::match_same_arms)] // Keep arms separate for clarity on each variant
    pub fn steps(&self) -> usize {
        match self {
            PhaseDecision::Warmup { steps } => *steps,
            PhaseDecision::Full { steps } => *steps,
            PhaseDecision::Predict { steps, .. } => *steps,
            PhaseDecision::Correct {
                validation_samples, ..
            } => *validation_samples,
        }
    }
}

/// Outcome of a completed phase.
///
/// Contains aggregate statistics about what happened during the phase,
/// used by the controller to inform future phase decisions.
#[derive(Debug, Clone)]
pub struct PhaseOutcome {
    /// The phase that was executed.
    pub phase: Phase,

    /// Number of steps actually executed.
    pub steps_executed: usize,

    /// Average loss during the phase.
    pub average_loss: f32,

    /// Final loss at end of phase.
    pub final_loss: f32,

    /// Whether the phase completed normally.
    pub completed_normally: bool,

    /// Early termination reason (if any).
    pub early_termination_reason: Option<String>,

    /// Prediction error (for Predict phase only).
    pub prediction_error: Option<f32>,

    /// Time taken in milliseconds.
    pub duration_ms: f64,
}

/// Trait for phase controllers that manage state machine transitions.
///
/// Implementations decide which phase to execute next based on training
/// state, history, and configuration.
pub trait PhaseController: Send + Sync {
    /// Determines the next phase based on current state and history.
    ///
    /// # Arguments
    ///
    /// * `state` - Current training state
    ///
    /// # Returns
    ///
    /// A decision about which phase to execute and for how long.
    fn select_next_phase(&mut self, state: &TrainingState) -> PhaseDecision;

    /// Updates the controller with the outcome of a completed phase.
    ///
    /// # Arguments
    ///
    /// * `phase` - The phase that completed
    /// * `outcome` - Statistics about the phase execution
    fn observe_outcome(&mut self, phase: Phase, outcome: &PhaseOutcome);

    /// Handles divergence by selecting an appropriate recovery action.
    ///
    /// # Arguments
    ///
    /// * `severity` - The detected divergence severity
    ///
    /// # Returns
    ///
    /// A recovery action to take.
    fn handle_divergence(&mut self, severity: DivergenceLevel) -> RecoveryAction;

    /// Returns the current phase.
    fn current_phase(&self) -> Phase;

    /// Forces a transition to a specific phase.
    ///
    /// Used for recovery actions or manual intervention.
    fn force_phase(&mut self, phase: Phase);
}

/// Default implementation of the phase controller.
///
/// Uses a simple state machine with configurable phase lengths and
/// confidence-based prediction gating.
pub struct DefaultPhaseController {
    /// Current phase.
    current_phase: Phase,

    /// Configuration parameters.
    config: PhaseControllerConfig,

    /// Steps remaining in current phase.
    _steps_remaining: usize,

    /// Whether warmup has completed.
    warmup_complete: bool,

    /// Whether at least one Full cycle has completed after warmup.
    /// This ensures we collect real gradient data before allowing predictions.
    had_full_after_warmup: bool,

    /// Current predictor confidence (updated externally).
    predictor_confidence: f32,

    /// Count of consecutive prediction phases.
    consecutive_predict_phases: usize,

    /// Last checkpoint step for rollback recovery.
    last_checkpoint_step: u64,

    /// History of phase outcomes for adaptive decisions.
    outcome_history: Vec<PhaseOutcome>,
}

/// Configuration for the default phase controller.
#[derive(Debug, Clone)]
pub struct PhaseControllerConfig {
    /// Number of warmup steps.
    pub warmup_steps: usize,
    /// Number of full steps per cycle.
    pub full_steps: usize,
    /// Maximum prediction steps.
    pub max_predict_steps: usize,
    /// Minimum confidence for predictions.
    pub confidence_threshold: f32,
    /// Maximum consecutive predict phases before forced full.
    pub max_consecutive_predicts: usize,
}

impl From<&HybridTrainerConfig> for PhaseControllerConfig {
    fn from(config: &HybridTrainerConfig) -> Self {
        Self {
            warmup_steps: config.warmup_steps,
            full_steps: config.full_steps,
            max_predict_steps: config.max_predict_steps,
            confidence_threshold: config.confidence_threshold,
            max_consecutive_predicts: 5,
        }
    }
}

impl DefaultPhaseController {
    /// Creates a new phase controller with the given configuration.
    #[must_use]
    pub fn new(config: &HybridTrainerConfig) -> Self {
        let ctrl_config = PhaseControllerConfig::from(config);
        Self {
            current_phase: Phase::Warmup,
            _steps_remaining: ctrl_config.warmup_steps,
            config: ctrl_config,
            warmup_complete: false,
            had_full_after_warmup: false,
            predictor_confidence: 0.0,
            consecutive_predict_phases: 0,
            last_checkpoint_step: 0,
            outcome_history: Vec::with_capacity(100),
        }
    }

    /// Updates the predictor confidence used for phase decisions.
    pub fn set_predictor_confidence(&mut self, confidence: f32) {
        self.predictor_confidence = confidence;
    }

    /// Updates the last checkpoint step for recovery decisions.
    pub fn set_last_checkpoint(&mut self, step: u64) {
        self.last_checkpoint_step = step;
    }

    /// Returns whether warmup has completed.
    #[must_use]
    pub fn is_warmup_complete(&self) -> bool {
        self.warmup_complete
    }

    /// Computes adaptive prediction length based on confidence and history.
    ///
    /// The prediction horizon scales quadratically with confidence (higher
    /// confidence = more steps) and is penalized by consecutive prediction
    /// phases to prevent runaway prediction without validation.
    ///
    /// # Returns
    ///
    /// The number of prediction steps to use, clamped between 10 and
    /// `max_predict_steps`.
    #[must_use]
    pub fn compute_predict_steps(&self) -> usize {
        // Scale prediction length by confidence
        let base_steps = self.config.max_predict_steps;
        let confidence_factor = self.predictor_confidence.powf(2.0);
        let adaptive_steps = (base_steps as f32 * confidence_factor) as usize;

        // Reduce if we've had many consecutive predictions
        let consecutive_penalty = 1.0 - (self.consecutive_predict_phases as f32 * 0.1).min(0.5);
        let penalized_steps = (adaptive_steps as f32 * consecutive_penalty) as usize;

        penalized_steps.max(10).min(self.config.max_predict_steps)
    }
}

impl PhaseController for DefaultPhaseController {
    fn select_next_phase(&mut self, state: &TrainingState) -> PhaseDecision {
        // During warmup, stay in warmup until complete
        if !self.warmup_complete {
            if state.step < self.config.warmup_steps as u64 {
                return PhaseDecision::Warmup {
                    steps: (self.config.warmup_steps as u64 - state.step) as usize,
                };
            }
            self.warmup_complete = true;
            self.current_phase = Phase::Full;
        }

        // After warmup, cycle through Full → Predict → Correct → Full
        match self.current_phase {
            Phase::Warmup => {
                // Transition to Full after warmup
                self.current_phase = Phase::Full;
                PhaseDecision::Full {
                    steps: self.config.full_steps,
                }
            }

            Phase::Full | Phase::Correct => {
                // Mark that we've completed at least one Full cycle after warmup
                if !self.had_full_after_warmup {
                    self.had_full_after_warmup = true;
                    // First Full after warmup - stay in Full to collect gradient data
                    self.current_phase = Phase::Full;
                    return PhaseDecision::Full {
                        steps: self.config.full_steps,
                    };
                }

                // After Full or Correct, decide between Predict and Full
                if self.predictor_confidence >= self.config.confidence_threshold
                    && self.consecutive_predict_phases < self.config.max_consecutive_predicts
                {
                    self.current_phase = Phase::Predict;
                    self.consecutive_predict_phases += 1;

                    let steps = self.compute_predict_steps();
                    let loss_stats = state.loss_statistics();
                    let fallback_threshold = loss_stats.mean as f32 + 3.0 * loss_stats.std as f32;

                    PhaseDecision::Predict {
                        steps,
                        confidence: self.predictor_confidence,
                        fallback_threshold,
                    }
                } else {
                    self.current_phase = Phase::Full;
                    self.consecutive_predict_phases = 0;
                    PhaseDecision::Full {
                        steps: self.config.full_steps,
                    }
                }
            }

            Phase::Predict => {
                // After Predict, go to Correct then Full
                self.current_phase = Phase::Correct;
                PhaseDecision::Correct {
                    validation_samples: 5,
                    max_correction_magnitude: 0.1,
                }
            }
        }
    }

    fn observe_outcome(&mut self, _phase: Phase, outcome: &PhaseOutcome) {
        // Store outcome in history
        self.outcome_history.push(outcome.clone());
        if self.outcome_history.len() > 100 {
            self.outcome_history.remove(0);
        }

        // Adjust based on outcome
        if !outcome.completed_normally {
            // Phase failed, be more conservative
            self.consecutive_predict_phases = self.config.max_consecutive_predicts;
        }
    }

    fn handle_divergence(&mut self, severity: DivergenceLevel) -> RecoveryAction {
        match severity {
            DivergenceLevel::Normal => RecoveryAction::Continue,

            DivergenceLevel::Caution => {
                // Be more conservative with predictions
                RecoveryAction::ReducePredictRatio(0.5)
            }

            DivergenceLevel::Warning => {
                // Force full training for a while
                self.current_phase = Phase::Full;
                self.consecutive_predict_phases = self.config.max_consecutive_predicts;
                RecoveryAction::ForceFullPhase(self.config.full_steps * 2)
            }

            DivergenceLevel::Critical => {
                // Rollback if possible
                RecoveryAction::RollbackAndRetry {
                    checkpoint_step: self.last_checkpoint_step,
                    new_learning_rate: 0.0, // Caller determines actual LR
                }
            }
        }
    }

    fn current_phase(&self) -> Phase {
        self.current_phase
    }

    fn force_phase(&mut self, phase: Phase) {
        self.current_phase = phase;
        if phase == Phase::Full {
            self.consecutive_predict_phases = 0;
        }
    }
}

/// Validates that a phase transition is legal.
///
/// Each allowed transition is documented explicitly for clarity about the
/// state machine rules, even though they share the same `true` return value.
///
/// # Arguments
///
/// * `from` - Source phase
/// * `to` - Destination phase
///
/// # Errors
///
/// Returns an error if the transition is not allowed by the state machine.
#[allow(clippy::match_same_arms)] // Each transition is documented separately for clarity
pub fn validate_transition(from: Phase, to: Phase) -> HybridResult<()> {
    let valid = match (from, to) {
        // Warmup can only go to Full
        (Phase::Warmup, Phase::Full) => true,
        (Phase::Warmup, _) => false,

        // Full can go to Predict or stay in Full
        (Phase::Full, Phase::Predict) => true,
        (Phase::Full, Phase::Full) => true,
        (Phase::Full, Phase::Correct) => true, // After forced full

        // Predict goes to Correct
        (Phase::Predict, Phase::Correct) => true,
        (Phase::Predict, Phase::Full) => true, // Early termination

        // Correct goes to Full or Predict
        (Phase::Correct, Phase::Full) => true,
        (Phase::Correct, Phase::Predict) => true,

        // Invalid transitions
        _ => false,
    };

    if valid {
        Ok(())
    } else {
        Err((
            HybridTrainingError::InvalidPhaseTransition {
                from,
                to,
                reason: format!("Transition from {from:?} to {to:?} is not allowed"),
            },
            Some(RecoveryAction::ForceFullPhase(10)),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phase_properties() {
        assert!(Phase::Warmup.computes_backward());
        assert!(Phase::Full.computes_backward());
        assert!(!Phase::Predict.computes_backward());
        assert!(!Phase::Correct.computes_backward());

        assert!(Phase::Predict.uses_predictions());
        assert!(!Phase::Full.uses_predictions());
    }

    #[test]
    fn test_valid_transitions() {
        assert!(validate_transition(Phase::Warmup, Phase::Full).is_ok());
        assert!(validate_transition(Phase::Full, Phase::Predict).is_ok());
        assert!(validate_transition(Phase::Predict, Phase::Correct).is_ok());
        assert!(validate_transition(Phase::Correct, Phase::Full).is_ok());
    }

    #[test]
    fn test_invalid_transitions() {
        assert!(validate_transition(Phase::Warmup, Phase::Predict).is_err());
        assert!(validate_transition(Phase::Warmup, Phase::Correct).is_err());
    }

    #[test]
    fn test_default_controller_warmup() {
        let config = HybridTrainerConfig::default();
        let mut controller = DefaultPhaseController::new(&config);
        let state = TrainingState::new();

        let decision = controller.select_next_phase(&state);
        assert!(matches!(decision, PhaseDecision::Warmup { .. }));
    }
}
