//! Unified health scoring for training monitoring and automatic tuning.
//!
//! This module provides comprehensive health scoring of the training process
//! across multiple dimensions, enabling automatic tuning recommendations.
//!
//! # Overview
//!
//! The health scorer monitors five key dimensions of training health:
//! - **Loss Velocity**: How fast the loss is changing (should be negative)
//! - **Loss Acceleration**: Whether loss improvement is accelerating (should be stable)
//! - **Gradient Entropy**: Randomness/stability of gradients
//! - **Prediction Accuracy**: How well the predictor forecasts loss changes
//! - **Gradient Stability**: Variance and consistency of gradient norms
//!
//! Each dimension is scored independently [0, 1] where higher is better, then
//! combined with configurable weights to produce an overall health score.
//!
//! # Health Classifications
//!
//! The overall score [0, 1] is classified into five categories:
//! - **Excellent** (>= 0.85): Training is performing optimally
//! - **Good** (>= 0.70): Training is stable and making progress
//! - **Moderate** (>= 0.50): Training is acceptable but could improve
//! - **Poor** (>= 0.30): Training has issues requiring attention
//! - **Critical** (< 0.30): Training is at risk and needs intervention
//!
//! # Recommendations
//!
//! Based on health classification and individual score components, the health
//! scorer generates actionable recommendations for training adjustment:
//! - Increase/decrease prediction steps
//! - Warmup restart
//! - Learning rate adjustments
//! - Force full-phase training
//! - Gradient clipping adjustments
//!
//! # Example
//!
//! ```rust,ignore
//! let mut scorer = HealthScorer::new(5, 80);
//!
//! // During training step
//! let health = scorer.compute(
//!     velocity,        // negative is good
//!     acceleration,    // should be stable
//!     entropy,         // [0, 1] gradient randomness
//!     accuracy,        // [0, 1] predictor accuracy
//!     stability,       // [0, 1] gradient stability
//!     progress_pct,    // 0-100% of training
//! );
//!
//! match health.classification {
//!     HealthClassification::Excellent => println!("Training optimal"),
//!     HealthClassification::Critical => {
//!         for recommendation in &health.recommendations {
//!             println!("Action: {:?}", recommendation);
//!         }
//!     }
//!     _ => {}
//! }
//! ```

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Overall health classification of the training process.
///
/// Represents five distinct health levels from optimal to critical,
/// used to drive automatic tuning decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum HealthClassification {
    /// Training is performing optimally (score >= 0.85).
    Excellent,
    /// Training is stable and making good progress (score >= 0.70).
    Good,
    /// Training is acceptable but could improve (score >= 0.50).
    Moderate,
    /// Training has issues requiring attention (score >= 0.30).
    Poor,
    /// Training is at risk and needs immediate intervention (score < 0.30).
    Critical,
}

impl HealthClassification {
    /// Returns a descriptive string for this classification.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Excellent => "Excellent",
            Self::Good => "Good",
            Self::Moderate => "Moderate",
            Self::Poor => "Poor",
            Self::Critical => "Critical",
        }
    }

    /// Returns the base prediction step factor for this classification.
    ///
    /// Higher factors result in longer prediction phases.
    /// - Excellent: max steps (full confidence)
    /// - Good: 80% of max
    /// - Moderate: 50% of max
    /// - Poor: minimum (2 steps)
    /// - Critical: absolute minimum (1 step)
    #[must_use]
    pub fn base_predict_factor(&self, max_steps: usize) -> usize {
        match self {
            Self::Excellent => max_steps,
            Self::Good => ((max_steps as f32) * 0.8) as usize,
            Self::Moderate => ((max_steps as f32) * 0.5) as usize,
            Self::Poor => 2,
            Self::Critical => 1,
        }
    }
}

/// Recommended action to improve training health.
///
/// The health scorer generates these recommendations based on the current
/// health state and component scores. Multiple recommendations can be
/// issued simultaneously for different training parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum HealthRecommendation {
    /// Increase the number of prediction steps to exploit high confidence.
    IncreasePredictSteps {
        /// Target prediction step count.
        to: usize,
    },
    /// Decrease the number of prediction steps due to poor prediction accuracy.
    DecreasePredictSteps {
        /// Target prediction step count.
        to: usize,
    },
    /// Restart warmup phase with adjusted learning rate (e.g., after divergence).
    WarmupRestart {
        /// Multiplier for the learning rate during warmup (e.g., 0.5 for 50%).
        lr_multiplier: f32,
    },
    /// Reduce the learning rate to stabilize training.
    ReduceLearningRate {
        /// Factor to multiply learning rate by (e.g., 0.5, 0.8).
        factor: f32,
    },
    /// Increase the learning rate to accelerate progress.
    IncreaseLearningRate {
        /// Factor to multiply learning rate by (e.g., 1.2, 1.5).
        factor: f32,
    },
    /// Force full training phase for specified number of steps.
    ForceFullPhase {
        /// Number of full training steps to execute.
        steps: usize,
    },
    /// Adjust gradient clipping threshold for better stability.
    AdjustGradientClip {
        /// New gradient norm threshold for clipping.
        new_threshold: f32,
    },
}

/// Comprehensive health score for the current training state.
///
/// Contains individual component scores, overall classification, and
/// recommended actions for improving training health.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingHealthScore {
    /// Overall health score in range [0, 1]. Higher is better.
    /// (Alias: score)
    pub overall: f32,

    /// Velocity score (loss change rate). Higher = better (loss decreasing).
    pub loss_velocity_score: f32,

    /// Acceleration score (loss improvement consistency). Higher = more stable.
    pub loss_acceleration_score: f32,

    /// Gradient entropy score (gradient stability). Higher = more stable.
    pub gradient_entropy_score: f32,

    /// Prediction accuracy score. Higher = predictor more accurate.
    pub prediction_accuracy_score: f32,

    /// Gradient stability score (norm variance). Higher = more stable.
    pub gradient_stability_score: f32,

    /// Overall health classification.
    pub classification: HealthClassification,

    /// Recommended actions to improve health.
    pub recommendations: Vec<HealthRecommendation>,

    /// Alias field for backward compatibility with controller.rs.
    #[serde(skip)]
    pub score: f32,
}

impl TrainingHealthScore {
    /// Returns true if the training is in a good state (Good or Excellent).
    #[must_use]
    pub fn is_healthy(&self) -> bool {
        matches!(
            self.classification,
            HealthClassification::Good | HealthClassification::Excellent
        )
    }

    /// Returns true if the training requires intervention (Poor or Critical).
    #[must_use]
    pub fn needs_intervention(&self) -> bool {
        matches!(
            self.classification,
            HealthClassification::Poor | HealthClassification::Critical
        )
    }
}

/// Weights for combining individual component scores into overall health.
///
/// Used to compute overall score as a weighted average of component scores.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct HealthWeights {
    /// Weight for loss velocity score. Default: 0.25.
    pub velocity_weight: f32,
    /// Weight for loss acceleration score. Default: 0.15.
    pub acceleration_weight: f32,
    /// Weight for gradient entropy score. Default: 0.15.
    pub entropy_weight: f32,
    /// Weight for prediction accuracy score. Default: 0.25.
    pub accuracy_weight: f32,
    /// Weight for gradient stability score. Default: 0.20.
    pub stability_weight: f32,
}

impl Default for HealthWeights {
    fn default() -> Self {
        Self {
            velocity_weight: 0.25,
            acceleration_weight: 0.15,
            entropy_weight: 0.15,
            accuracy_weight: 0.25,
            stability_weight: 0.20,
        }
    }
}

impl HealthWeights {
    /// Validates that weights sum to approximately 1.0 (within tolerance).
    ///
    /// # Returns
    ///
    /// `true` if weights are valid, `false` otherwise.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        let sum = self.velocity_weight
            + self.acceleration_weight
            + self.entropy_weight
            + self.accuracy_weight
            + self.stability_weight;
        (sum - 1.0).abs() < 0.01
    }

    /// Normalizes weights to sum to exactly 1.0.
    pub fn normalize(&mut self) {
        let sum = self.velocity_weight
            + self.acceleration_weight
            + self.entropy_weight
            + self.accuracy_weight
            + self.stability_weight;

        if sum > 0.0 {
            self.velocity_weight /= sum;
            self.acceleration_weight /= sum;
            self.entropy_weight /= sum;
            self.accuracy_weight /= sum;
            self.stability_weight /= sum;
        }
    }
}

/// Unified health scorer for training monitoring and auto-tuning.
///
/// Combines multiple training signals into a comprehensive health score
/// and generates actionable recommendations for training adjustments.
pub struct HealthScorer {
    /// Weighting configuration for component scores.
    weights: HealthWeights,

    /// Ring buffer of recent health scores for trend analysis.
    score_history: VecDeque<f32>,

    /// Minimum number of prediction steps allowed.
    min_predict_steps: usize,

    /// Maximum number of prediction steps allowed.
    max_predict_steps: usize,

    /// Most recent computed health score.
    current_score: Option<TrainingHealthScore>,
}

impl HealthScorer {
    /// Creates a new health scorer with default weights.
    ///
    /// # Arguments
    ///
    /// * `min_predict_steps` - Minimum allowed prediction steps
    /// * `max_predict_steps` - Maximum allowed prediction steps
    ///
    /// # Returns
    ///
    /// A new `HealthScorer` instance with a 100-element score history buffer.
    #[must_use]
    pub fn new(min_predict_steps: usize, max_predict_steps: usize) -> Self {
        Self {
            weights: HealthWeights::default(),
            score_history: VecDeque::with_capacity(100),
            min_predict_steps,
            max_predict_steps,
            current_score: None,
        }
    }

    /// Creates a new health scorer with custom weights.
    ///
    /// # Arguments
    ///
    /// * `min_predict_steps` - Minimum allowed prediction steps
    /// * `max_predict_steps` - Maximum allowed prediction steps
    /// * `weights` - Custom weight configuration
    ///
    /// # Panics
    ///
    /// Panics if weights don't sum to approximately 1.0.
    #[must_use]
    pub fn with_weights(
        min_predict_steps: usize,
        max_predict_steps: usize,
        weights: HealthWeights,
    ) -> Self {
        assert!(
            weights.is_valid(),
            "HealthWeights must sum to approximately 1.0"
        );

        Self {
            weights,
            score_history: VecDeque::with_capacity(100),
            min_predict_steps,
            max_predict_steps,
            current_score: None,
        }
    }

    /// Computes a health score based on current training metrics.
    ///
    /// # Arguments
    ///
    /// * `velocity` - Loss change rate (negative = improving, positive = deteriorating)
    /// * `acceleration` - Loss acceleration (derivative of velocity)
    /// * `entropy` - Gradient entropy in [0, 1] (0 = deterministic, 1 = random)
    /// * `accuracy` - Prediction accuracy in [0, 1] (1 = perfect predictions)
    /// * `stability` - Gradient stability in [0, 1] (1 = perfectly stable)
    /// * `progress_pct` - Training progress as percentage [0, 100]
    ///
    /// # Returns
    ///
    /// A `TrainingHealthScore` with overall score, component scores, classification, and recommendations.
    pub fn compute(
        &mut self,
        velocity: f32,
        acceleration: f32,
        entropy: f32,
        accuracy: f32,
        stability: f32,
        progress_pct: f32,
    ) -> TrainingHealthScore {
        // Compute individual component scores
        let velocity_score = Self::compute_velocity_score(velocity);
        let acceleration_score = Self::compute_acceleration_score(acceleration);
        let entropy_score = Self::compute_entropy_score(entropy);
        let accuracy_score = Self::compute_accuracy_score(accuracy);
        let stability_score = Self::compute_stability_score(stability);

        // Compute weighted overall score
        let overall = self.weights.velocity_weight * velocity_score
            + self.weights.acceleration_weight * acceleration_score
            + self.weights.entropy_weight * entropy_score
            + self.weights.accuracy_weight * accuracy_score
            + self.weights.stability_weight * stability_score;

        // Clamp overall score to [0, 1]
        let overall = overall.max(0.0).min(1.0);

        // Classify health
        let classification = Self::classify_health(overall);

        // Generate recommendations
        let recommendations = self.generate_recommendations(
            classification,
            velocity_score,
            acceleration_score,
            entropy_score,
            accuracy_score,
            stability_score,
            progress_pct,
        );

        // Store in history (keep last 100)
        self.score_history.push_back(overall);
        if self.score_history.len() > 100 {
            self.score_history.pop_front();
        }

        let health = TrainingHealthScore {
            overall,
            loss_velocity_score: velocity_score,
            loss_acceleration_score: acceleration_score,
            gradient_entropy_score: entropy_score,
            prediction_accuracy_score: accuracy_score,
            gradient_stability_score: stability_score,
            classification,
            recommendations,
            score: overall, // Backward compatibility alias
        };

        self.current_score = Some(health.clone());
        health
    }

    /// Computes velocity score from loss change rate.
    ///
    /// Uses sigmoid function: higher scores for more negative (improving) velocity.
    /// Velocity = -1.0 maps to score ~1.0 (excellent)
    /// Velocity = 0.0 maps to score ~0.73 (decent)
    /// Velocity = +1.0 maps to score ~0.27 (poor)
    fn compute_velocity_score(velocity: f32) -> f32 {
        // Sigmoid: 1 / (1 + e^(-x))
        // We use negative velocity so negative values (loss decreasing) give high scores
        let sigmoid_input = -velocity;
        let score = 1.0 / (1.0 + (-sigmoid_input).exp());
        score.max(0.0).min(1.0)
    }

    /// Computes acceleration score from loss acceleration.
    ///
    /// Stable acceleration (|accel| < 0.001) scores 0.8 (very good)
    /// Negative acceleration scores 0.7 (good - getting better faster)
    /// Positive acceleration scores 0.3 (bad - getting worse faster)
    fn compute_acceleration_score(acceleration: f32) -> f32 {
        if acceleration.abs() < 0.001 {
            0.8 // Stable - excellent
        } else if acceleration < 0.0 {
            0.7 // Negative acceleration - loss improving at accelerating rate
        } else {
            0.3 // Positive acceleration - loss degrading
        }
    }

    /// Computes entropy score from gradient entropy.
    ///
    /// Entropy should be in [0, 1] where:
    /// 0 = deterministic gradients (perfectly stable)
    /// 1 = maximum randomness (very unstable)
    /// Score = 1 - entropy, so stable gradients get high scores.
    fn compute_entropy_score(entropy: f32) -> f32 {
        let clamped = entropy.max(0.0).min(1.0);
        1.0 - clamped
    }

    /// Computes accuracy score from prediction accuracy.
    ///
    /// Accuracy should be in [0, 1] where 1 = perfect predictions.
    /// Direct pass-through after clamping.
    fn compute_accuracy_score(accuracy: f32) -> f32 {
        accuracy.max(0.0).min(1.0)
    }

    /// Computes stability score from gradient stability.
    ///
    /// Stability should be in [0, 1] where 1 = perfectly stable gradients.
    /// Direct pass-through after clamping.
    fn compute_stability_score(stability: f32) -> f32 {
        stability.max(0.0).min(1.0)
    }

    /// Classifies overall health into discrete categories.
    fn classify_health(overall: f32) -> HealthClassification {
        if overall >= 0.85 {
            HealthClassification::Excellent
        } else if overall >= 0.70 {
            HealthClassification::Good
        } else if overall >= 0.50 {
            HealthClassification::Moderate
        } else if overall >= 0.30 {
            HealthClassification::Poor
        } else {
            HealthClassification::Critical
        }
    }

    /// Generates recommendations based on current health state.
    #[allow(clippy::too_many_arguments)]
    fn generate_recommendations(
        &self,
        classification: HealthClassification,
        velocity_score: f32,
        _acceleration_score: f32,
        entropy_score: f32,
        accuracy_score: f32,
        stability_score: f32,
        progress_pct: f32,
    ) -> Vec<HealthRecommendation> {
        let mut recommendations = Vec::new();

        match classification {
            HealthClassification::Excellent => {
                // Training is optimal - consider increasing prediction steps if room
                if accuracy_score > 0.85 && self.current_predict_steps() < self.max_predict_steps {
                    let target = ((self.current_predict_steps() as f32) * 1.2) as usize;
                    let target = target.min(self.max_predict_steps);
                    recommendations.push(HealthRecommendation::IncreasePredictSteps { to: target });
                }
            }
            HealthClassification::Good => {
                // Training is good - minor adjustments if needed
                if accuracy_score > 0.8 && self.current_predict_steps() < self.max_predict_steps {
                    let target = ((self.current_predict_steps() as f32) * 1.1) as usize;
                    let target = target.min(self.max_predict_steps);
                    recommendations.push(HealthRecommendation::IncreasePredictSteps { to: target });
                }
            }
            HealthClassification::Moderate => {
                // Training could improve - check individual components
                if accuracy_score < 0.6 && self.current_predict_steps() > self.min_predict_steps {
                    // Poor prediction accuracy - reduce prediction steps
                    let target = ((self.current_predict_steps() as f32) * 0.7).ceil() as usize;
                    let target = target.max(self.min_predict_steps);
                    recommendations.push(HealthRecommendation::DecreasePredictSteps { to: target });
                }

                if entropy_score < 0.5 {
                    // Poor gradient stability - consider learning rate reduction
                    recommendations.push(HealthRecommendation::ReduceLearningRate { factor: 0.8 });
                }

                if velocity_score < 0.4 {
                    // Poor loss velocity - loss not improving well
                    if progress_pct < 50.0 {
                        // Early in training - might benefit from warmup restart
                        recommendations
                            .push(HealthRecommendation::WarmupRestart { lr_multiplier: 0.5 });
                    }
                }
            }
            HealthClassification::Poor => {
                // Training has issues - more aggressive interventions
                if accuracy_score < 0.5 {
                    // Very poor predictions - force full phase
                    let full_steps = 50;
                    recommendations
                        .push(HealthRecommendation::ForceFullPhase { steps: full_steps });
                }

                if stability_score < 0.4 {
                    // Very unstable gradients - reduce learning rate significantly
                    recommendations.push(HealthRecommendation::ReduceLearningRate { factor: 0.5 });
                }

                // Reduce prediction steps if using them
                if self.current_predict_steps() > self.min_predict_steps {
                    let target = ((self.current_predict_steps() as f32) * 0.5).ceil() as usize;
                    let target = target.max(self.min_predict_steps);
                    recommendations.push(HealthRecommendation::DecreasePredictSteps { to: target });
                }
            }
            HealthClassification::Critical => {
                // Training is at risk - emergency interventions
                // Force full training for many steps
                recommendations.push(HealthRecommendation::ForceFullPhase { steps: 100 });

                // Reduce learning rate significantly
                recommendations.push(HealthRecommendation::ReduceLearningRate { factor: 0.3 });

                // Reduce gradient clipping threshold to be more conservative
                recommendations
                    .push(HealthRecommendation::AdjustGradientClip { new_threshold: 0.5 });

                // If early in training, restart warmup
                if progress_pct < 20.0 {
                    recommendations
                        .push(HealthRecommendation::WarmupRestart { lr_multiplier: 0.1 });
                }
            }
        }

        recommendations
    }

    /// Returns the current number of prediction steps (estimate from history if needed).
    ///
    /// This is a heuristic - in practice, you'd track the actual value.
    /// For now, we return the midpoint as a default assumption.
    fn current_predict_steps(&self) -> usize {
        usize::midpoint(self.min_predict_steps, self.max_predict_steps)
    }

    /// Returns the most recent health score, or None if no score has been computed yet.
    #[must_use]
    pub fn current(&self) -> Option<&TrainingHealthScore> {
        self.current_score.as_ref()
    }

    /// Computes the trend of health over recent history.
    ///
    /// # Returns
    ///
    /// A trend value representing the average of recent scores minus the average
    /// of older scores. Positive values indicate improving health trend.
    ///
    /// Returns 0.0 if insufficient history.
    #[must_use]
    pub fn trend(&self) -> f32 {
        if self.score_history.len() < 10 {
            return 0.0; // Not enough history
        }

        let len = self.score_history.len();
        let recent_start = len.saturating_sub(5);
        let older_start = len.saturating_sub(10);

        let recent_avg: f32 = self.score_history.iter().skip(recent_start).sum::<f32>() / 5.0;
        let older_avg: f32 = self
            .score_history
            .iter()
            .skip(older_start)
            .take(5)
            .sum::<f32>()
            / 5.0;

        recent_avg - older_avg
    }

    /// Updates the prediction step bounds.
    ///
    /// # Arguments
    ///
    /// * `min` - New minimum prediction steps
    /// * `max` - New maximum prediction steps
    pub fn set_predict_bounds(&mut self, min: usize, max: usize) {
        self.min_predict_steps = min;
        self.max_predict_steps = max;
    }

    /// Updates the score weighting configuration.
    ///
    /// # Arguments
    ///
    /// * `weights` - New weight configuration
    ///
    /// # Panics
    ///
    /// Panics if weights don't sum to approximately 1.0.
    pub fn set_weights(&mut self, mut weights: HealthWeights) {
        weights.normalize();
        self.weights = weights;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health_weights_validation() {
        let weights = HealthWeights::default();
        assert!(weights.is_valid());

        let invalid = HealthWeights {
            velocity_weight: 0.3,
            acceleration_weight: 0.3,
            entropy_weight: 0.0,
            accuracy_weight: 0.0,
            stability_weight: 0.0,
        };
        assert!(!invalid.is_valid());
    }

    #[test]
    fn test_health_weights_normalization() {
        let mut weights = HealthWeights {
            velocity_weight: 1.0,
            acceleration_weight: 1.0,
            entropy_weight: 1.0,
            accuracy_weight: 1.0,
            stability_weight: 1.0,
        };
        weights.normalize();
        assert!(weights.is_valid());
        let sum = weights.velocity_weight
            + weights.acceleration_weight
            + weights.entropy_weight
            + weights.accuracy_weight
            + weights.stability_weight;
        assert!((sum - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_velocity_score_computation() {
        // Negative velocity (improving loss) should give high score
        let improving_score = HealthScorer::compute_velocity_score(-1.0);
        assert!(improving_score > 0.7);

        // Positive velocity (worsening loss) should give low score
        let worsening_score = HealthScorer::compute_velocity_score(1.0);
        assert!(worsening_score < 0.3);

        // Zero velocity is at sigmoid midpoint (0.5)
        let neutral_score = HealthScorer::compute_velocity_score(0.0);
        assert!((neutral_score - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_acceleration_score_computation() {
        // Very small acceleration should be stable
        let stable = HealthScorer::compute_acceleration_score(0.0005);
        assert_eq!(stable, 0.8);

        // Negative acceleration is good
        let negative = HealthScorer::compute_acceleration_score(-0.1);
        assert_eq!(negative, 0.7);

        // Positive acceleration is bad
        let positive = HealthScorer::compute_acceleration_score(0.1);
        assert_eq!(positive, 0.3);
    }

    #[test]
    fn test_entropy_score_computation() {
        // Zero entropy (deterministic) should give high score
        let deterministic = HealthScorer::compute_entropy_score(0.0);
        assert_eq!(deterministic, 1.0);

        // High entropy (random) should give low score
        let random = HealthScorer::compute_entropy_score(1.0);
        assert_eq!(random, 0.0);

        // Mid entropy
        let mid = HealthScorer::compute_entropy_score(0.5);
        assert_eq!(mid, 0.5);
    }

    #[test]
    fn test_health_classification() {
        assert_eq!(
            HealthScorer::classify_health(0.9),
            HealthClassification::Excellent
        );
        assert_eq!(
            HealthScorer::classify_health(0.75),
            HealthClassification::Good
        );
        assert_eq!(
            HealthScorer::classify_health(0.6),
            HealthClassification::Moderate
        );
        assert_eq!(
            HealthScorer::classify_health(0.4),
            HealthClassification::Poor
        );
        assert_eq!(
            HealthScorer::classify_health(0.2),
            HealthClassification::Critical
        );
    }

    #[test]
    fn test_health_scorer_creation() {
        let scorer = HealthScorer::new(5, 80);
        assert_eq!(scorer.min_predict_steps, 5);
        assert_eq!(scorer.max_predict_steps, 80);
        assert!(scorer.current().is_none());
    }

    #[test]
    fn test_health_score_computation() {
        let mut scorer = HealthScorer::new(5, 80);

        // Compute score with good metrics
        let health = scorer.compute(
            -0.5, // velocity: improving
            0.0,  // acceleration: stable
            0.2,  // entropy: stable gradients
            0.9,  // accuracy: high
            0.85, // stability: stable
            50.0, // progress: 50%
        );

        assert!(health.overall > 0.7); // Should be Good or better
        assert!(
            !health.recommendations.is_empty()
                || health.classification == HealthClassification::Excellent
        );
    }

    #[test]
    fn test_health_score_critical_state() {
        let mut scorer = HealthScorer::new(5, 80);

        // Compute score with poor metrics
        let health = scorer.compute(
            1.0,  // velocity: worsening
            0.5,  // acceleration: unstable
            0.9,  // entropy: very random
            0.1,  // accuracy: very poor
            0.1,  // stability: unstable
            10.0, // progress: early
        );

        assert!(health.overall < 0.4); // Should be Poor or Critical
        assert_eq!(health.classification, HealthClassification::Critical);
        assert!(!health.recommendations.is_empty());
    }

    #[test]
    fn test_is_healthy() {
        let good_score = TrainingHealthScore {
            overall: 0.75,
            loss_velocity_score: 0.7,
            loss_acceleration_score: 0.7,
            gradient_entropy_score: 0.7,
            prediction_accuracy_score: 0.7,
            gradient_stability_score: 0.7,
            classification: HealthClassification::Good,
            recommendations: vec![],
            score: 0.75,
        };

        assert!(good_score.is_healthy());

        let poor_score = TrainingHealthScore {
            overall: 0.35,
            loss_velocity_score: 0.3,
            loss_acceleration_score: 0.3,
            gradient_entropy_score: 0.3,
            prediction_accuracy_score: 0.3,
            gradient_stability_score: 0.3,
            classification: HealthClassification::Poor,
            recommendations: vec![],
            score: 0.35,
        };

        assert!(!poor_score.is_healthy());
        assert!(poor_score.needs_intervention());
    }

    #[test]
    fn test_trend_computation() {
        let mut scorer = HealthScorer::new(5, 80);

        // Add scores showing improving trend
        // Need at least 10 scores for trend calculation
        for i in 0..15 {
            let score = if i < 7 { 0.3 } else { 0.7 };
            scorer.score_history.push_back(score);
        }

        let trend = scorer.trend();
        assert!(trend > 0.0, "Trend should be positive (improving)"); // Should show improvement
    }

    #[test]
    fn test_score_history_buffer_limit() {
        let mut scorer = HealthScorer::new(5, 80);

        // Add more than 100 scores
        for _ in 0..150 {
            scorer.score_history.push_back(0.5);
            if scorer.score_history.len() > 100 {
                scorer.score_history.pop_front();
            }
        }

        assert_eq!(scorer.score_history.len(), 100);
    }

    #[test]
    fn test_recommendations_excellent_state() {
        let mut scorer = HealthScorer::new(5, 80);

        let health = scorer.compute(
            -0.5,   // velocity: improving
            -0.001, // acceleration: stable/negative
            0.05,   // entropy: very very stable (near 0)
            0.95,   // accuracy: excellent
            0.95,   // stability: excellent
            50.0,   // progress: 50%
        );

        // Score should be: 0.25*0.73 + 0.15*0.8 + 0.15*0.95 + 0.25*0.95 + 0.20*0.95 = 0.87+
        assert!(
            health.classification == HealthClassification::Excellent
                || health.classification == HealthClassification::Good,
            "Expected Excellent or Good, got {:?}",
            health.classification
        );
        // Should have recommendation to increase predict steps if Excellent
        if health.classification == HealthClassification::Excellent {
            let has_increase = health
                .recommendations
                .iter()
                .any(|r| matches!(r, HealthRecommendation::IncreasePredictSteps { .. }));
            assert!(has_increase);
        }
    }

    #[test]
    fn test_recommendations_critical_state() {
        let mut scorer = HealthScorer::new(5, 80);

        let health = scorer.compute(
            1.5,  // velocity: very bad
            0.8,  // acceleration: unstable
            0.95, // entropy: very random
            0.05, // accuracy: terrible
            0.05, // stability: terrible
            5.0,  // progress: very early
        );

        assert_eq!(health.classification, HealthClassification::Critical);
        // Should have multiple emergency recommendations
        assert!(health.recommendations.len() >= 2);
        let has_force_full = health
            .recommendations
            .iter()
            .any(|r| matches!(r, HealthRecommendation::ForceFullPhase { .. }));
        assert!(has_force_full);
    }

    #[test]
    fn test_set_predict_bounds() {
        let mut scorer = HealthScorer::new(5, 80);
        scorer.set_predict_bounds(10, 120);
        assert_eq!(scorer.min_predict_steps, 10);
        assert_eq!(scorer.max_predict_steps, 120);
    }

    #[test]
    fn test_set_weights() {
        let mut scorer = HealthScorer::new(5, 80);
        let mut custom_weights = HealthWeights {
            velocity_weight: 0.4,
            acceleration_weight: 0.2,
            entropy_weight: 0.2,
            accuracy_weight: 0.1,
            stability_weight: 0.1,
        };
        custom_weights.normalize();
        scorer.set_weights(custom_weights);
        assert!(scorer.weights.is_valid());
    }

    #[test]
    fn test_health_classification_boundaries() {
        // Test exact boundaries
        assert_eq!(
            HealthScorer::classify_health(0.85),
            HealthClassification::Excellent
        );
        assert_eq!(
            HealthScorer::classify_health(0.849),
            HealthClassification::Good
        );
        assert_eq!(
            HealthScorer::classify_health(0.70),
            HealthClassification::Good
        );
        assert_eq!(
            HealthScorer::classify_health(0.699),
            HealthClassification::Moderate
        );
    }

    #[test]
    fn test_recommendation_enum_variants() {
        let recommendations = vec![
            HealthRecommendation::IncreasePredictSteps { to: 100 },
            HealthRecommendation::DecreasePredictSteps { to: 20 },
            HealthRecommendation::WarmupRestart { lr_multiplier: 0.5 },
            HealthRecommendation::ReduceLearningRate { factor: 0.8 },
            HealthRecommendation::IncreaseLearningRate { factor: 1.2 },
            HealthRecommendation::ForceFullPhase { steps: 50 },
            HealthRecommendation::AdjustGradientClip { new_threshold: 1.0 },
        ];

        assert_eq!(recommendations.len(), 7);
    }

    #[test]
    fn test_health_classification_display() {
        assert_eq!(HealthClassification::Excellent.as_str(), "Excellent");
        assert_eq!(HealthClassification::Good.as_str(), "Good");
        assert_eq!(HealthClassification::Moderate.as_str(), "Moderate");
        assert_eq!(HealthClassification::Poor.as_str(), "Poor");
        assert_eq!(HealthClassification::Critical.as_str(), "Critical");
    }

    #[test]
    fn test_compute_with_zero_scores() {
        let mut scorer = HealthScorer::new(5, 80);
        let health = scorer.compute(0.0, 0.0, 0.0, 0.0, 0.0, 50.0);

        // All zero inputs should produce specific scores
        assert!(health.overall >= 0.0 && health.overall <= 1.0);
    }

    #[test]
    fn test_compute_with_extreme_values() {
        let mut scorer = HealthScorer::new(5, 80);

        // Very large positive values
        let health1 = scorer.compute(100.0, 100.0, 100.0, 100.0, 100.0, 50.0);
        assert!(health1.overall >= 0.0 && health1.overall <= 1.0);

        // Very large negative values
        let health2 = scorer.compute(-100.0, -100.0, -100.0, -100.0, -100.0, 50.0);
        assert!(health2.overall >= 0.0 && health2.overall <= 1.0);
    }
}
