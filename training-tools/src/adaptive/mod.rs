//! Adaptive Training Controller
//!
//! Comprehensive adaptive training control that coordinates:
//! - Learning rate adjustment (via `AdaptiveLRController`)
//! - Batch size tuning (via `BatchTuner`)
//! - Gradient clipping control (via `GradientController`)
//! - Curve analysis for predictive adjustments (via `CurveAnalyzer`)
//!
//! # Overview
//!
//! The `AdaptiveTrainingController` provides a unified interface for dynamically
//! adjusting training hyperparameters based on real-time training dynamics.
//! It supports multiple adaptation strategies ranging from conservative to
//! aggressive, with automatic recovery mode when issues are detected.
//!
//! # Example
//!
//! ```rust
//! use training_tools::adaptive::{
//!     AdaptiveTrainingController, AdaptiveConfig, AdaptationStrategy
//! };
//! use training_tools::StepMetrics;
//!
//! // Create controller with default configuration
//! let mut controller = AdaptiveTrainingController::new(AdaptiveConfig::default());
//!
//! // In your training loop
//! for step in 0..10000 {
//!     // ... forward/backward pass ...
//!     let metrics = /* your step metrics */;
//!
//!     // Get adaptations to apply
//!     let adaptations = controller.update(&metrics);
//!
//!     for adaptation in &adaptations {
//!         println!("Step {}: {} {:?} from {} to {} - {}",
//!             adaptation.step,
//!             if adaptation.new_value > adaptation.old_value { "increased" } else { "decreased" },
//!             adaptation.param,
//!             adaptation.old_value,
//!             adaptation.new_value,
//!             adaptation.reason
//!         );
//!     }
//!
//!     // Apply adaptations to your optimizer/training loop
//!     let lr = controller.current_lr();
//!     let batch_size = controller.current_batch_size();
//!     let grad_clip = controller.current_grad_clip();
//! }
//!
//! // Get health report
//! let report = controller.get_health_report();
//! println!("{}", report.format());
//! ```

mod controller;
mod health;
mod history;
mod integration;
mod strategy;

pub use controller::{Adaptation, AdaptedParam, AdaptiveConfig, AdaptiveTrainingController};
pub use health::{GradientHealth, Health, TrainingHealthReport};
pub use history::{AdaptationHistory, AdaptationSummary, StrategyChange};
pub use integration::{
    AdaptiveLoopHelper, AdaptiveProgressiveConfig, AdaptiveProgressiveTrainer, AdaptiveUpdate,
    ApplyAdaptations, StepResult,
};
pub use strategy::AdaptationStrategy;
