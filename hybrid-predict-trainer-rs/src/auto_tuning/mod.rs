//! Automatic tuning and health monitoring for hybrid predictive training.
//!
//! This module provides automatic tuning capabilities that monitor training health
//! and generate recommendations for parameter adjustments to optimize the training
//! process for speedup while maintaining quality.
//!
//! # Components
//!
//! - [`GradientRangeTuner`]: Per-layer adaptive gradient clipping with phase-aware thresholds
//! - [`PlateauDetector`]: Loss dynamics monitoring and plateau detection with warmup restart
//! - [`HealthScorer`]: Unified training health scoring combining multiple signals
//! - [`MultiStepPredictor`]: Multi-horizon prediction with calibrated confidence
//! - [`AdaptivePhaseController`]: Health-aware phase recommendations
//! - [`AutoTuningController`]: Orchestrates all components into unified recommendations

pub mod controller;
pub mod gradient_tuner;
pub mod health_scorer;
pub mod multi_step_predictor;
pub mod phase_controller;
pub mod plateau_detector;

// Re-export public types for convenient access
pub use controller::{AutoTuningConfig, AutoTuningController, AutoTuningUpdate};
pub use gradient_tuner::{
    GradientRangeTuner, LayerGradientStats, PhaseGradientThresholds, TrainingProgressPhase,
};
pub use health_scorer::{
    HealthClassification, HealthRecommendation, HealthScorer, HealthWeights, TrainingHealthScore,
};
pub use multi_step_predictor::{
    BatchPrediction, BatchPredictionRecommendation, MultiStepPredictor,
};
pub use phase_controller::AdaptivePhaseController;
pub use plateau_detector::{LossDynamics, PlateauDetector, PlateauStatus};
