//! Adaptation strategies for training control.
//!
//! Defines different strategies for adapting hyperparameters during training,
//! ranging from conservative (stable but slow) to aggressive (fast but riskier).

use serde::{Deserialize, Serialize};

/// Strategy for adapting training hyperparameters.
///
/// The strategy determines how aggressively the controller adjusts parameters
/// in response to training dynamics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AdaptationStrategy {
    /// Conservative: Small adjustments, prioritize stable training.
    ///
    /// - LR adjustments: +/- 10% per adaptation
    /// - Batch size changes: +/- 1 per adaptation
    /// - Longer cooldown between adjustments
    /// - Higher confidence threshold for changes
    Conservative,

    /// Balanced: Moderate adjustments, balance stability and speed.
    ///
    /// - LR adjustments: +/- 20-30% per adaptation
    /// - Batch size changes: +/- 25% per adaptation
    /// - Standard cooldown periods
    /// - Default behavior
    Balanced,

    /// Aggressive: Larger adjustments, faster convergence but higher risk.
    ///
    /// - LR adjustments: +/- 50% per adaptation
    /// - Batch size changes: +/- 50% per adaptation
    /// - Shorter cooldown periods
    /// - Lower confidence threshold for changes
    Aggressive,

    /// Exploratory: Try different settings to find optimum.
    ///
    /// - Periodically tests different LR values
    /// - Uses cosine annealing with restarts
    /// - Records performance at different settings
    /// - Useful for hyperparameter search
    Exploratory,

    /// Recovery: Careful recovery after detecting training issues.
    ///
    /// - Very conservative adjustments
    /// - Focus on stabilizing training
    /// - Gradually increase parameters back to normal
    /// - Exits to Conservative when stable
    Recovery,
}

impl AdaptationStrategy {
    /// Get the learning rate adjustment factor for increases.
    pub fn lr_increase_factor(&self) -> f32 {
        match self {
            Self::Conservative => 1.10, // +10%
            Self::Balanced => 1.25,     // +25%
            Self::Aggressive => 1.50,   // +50%
            Self::Exploratory => 1.30,  // +30%
            Self::Recovery => 1.05,     // +5%
        }
    }

    /// Get the learning rate adjustment factor for decreases.
    pub fn lr_decrease_factor(&self) -> f32 {
        match self {
            Self::Conservative => 0.90, // -10%
            Self::Balanced => 0.75,     // -25%
            Self::Aggressive => 0.50,   // -50%
            Self::Exploratory => 0.70,  // -30%
            Self::Recovery => 0.80,     // -20%
        }
    }

    /// Get the batch size adjustment ratio for increases.
    pub fn batch_increase_ratio(&self) -> f32 {
        match self {
            Self::Conservative => 1.10, // +10%
            Self::Balanced => 1.25,     // +25%
            Self::Aggressive => 1.50,   // +50%
            Self::Exploratory => 1.20,  // +20%
            Self::Recovery => 1.05,     // +5%
        }
    }

    /// Get the batch size adjustment ratio for decreases.
    pub fn batch_decrease_ratio(&self) -> f32 {
        match self {
            Self::Conservative => 0.90, // -10%
            Self::Balanced => 0.75,     // -25%
            Self::Aggressive => 0.67,   // -33%
            Self::Exploratory => 0.80,  // -20%
            Self::Recovery => 0.85,     // -15%
        }
    }

    /// Get the gradient clip adjustment factor.
    pub fn grad_clip_factor(&self) -> f32 {
        match self {
            Self::Conservative => 0.95, // -5%
            Self::Balanced => 0.85,     // -15%
            Self::Aggressive => 0.70,   // -30%
            Self::Exploratory => 0.80,  // -20%
            Self::Recovery => 0.90,     // -10%
        }
    }

    /// Get the minimum steps between adaptations (cooldown).
    pub fn cooldown_steps(&self) -> u64 {
        match self {
            Self::Conservative => 50,
            Self::Balanced => 25,
            Self::Aggressive => 10,
            Self::Exploratory => 20,
            Self::Recovery => 100,
        }
    }

    /// Get the confidence threshold for making changes.
    ///
    /// Higher values mean more evidence is needed before adapting.
    pub fn confidence_threshold(&self) -> f32 {
        match self {
            Self::Conservative => 0.9,
            Self::Balanced => 0.7,
            Self::Aggressive => 0.5,
            Self::Exploratory => 0.6,
            Self::Recovery => 0.95,
        }
    }

    /// Whether this strategy allows exploratory probing.
    pub fn allows_exploration(&self) -> bool {
        matches!(self, Self::Exploratory)
    }

    /// Whether this strategy prioritizes stability over speed.
    pub fn prioritizes_stability(&self) -> bool {
        matches!(self, Self::Conservative | Self::Recovery)
    }

    /// Human-readable description of the strategy.
    pub fn description(&self) -> &'static str {
        match self {
            Self::Conservative => "Conservative: Small adjustments, stable training",
            Self::Balanced => "Balanced: Moderate adjustments, balance stability and speed",
            Self::Aggressive => "Aggressive: Larger adjustments, faster convergence",
            Self::Exploratory => "Exploratory: Testing different settings to find optimum",
            Self::Recovery => "Recovery: Careful recovery after training issues",
        }
    }

    /// Get the recommended strategy based on training health.
    pub fn recommended_for_health(health_score: f32, loss_trend_is_improving: bool) -> Self {
        if health_score < 0.3 {
            // Very poor health - switch to recovery
            Self::Recovery
        } else if health_score < 0.5 {
            // Poor health - be conservative
            Self::Conservative
        } else if health_score > 0.8 && loss_trend_is_improving {
            // Excellent health and improving - can be more aggressive
            Self::Aggressive
        } else {
            // Normal conditions
            Self::Balanced
        }
    }
}

impl Default for AdaptationStrategy {
    fn default() -> Self {
        Self::Balanced
    }
}

impl std::fmt::Display for AdaptationStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Conservative => write!(f, "Conservative"),
            Self::Balanced => write!(f, "Balanced"),
            Self::Aggressive => write!(f, "Aggressive"),
            Self::Exploratory => write!(f, "Exploratory"),
            Self::Recovery => write!(f, "Recovery"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strategy_factors() {
        let conservative = AdaptationStrategy::Conservative;
        let aggressive = AdaptationStrategy::Aggressive;

        // Conservative should have smaller adjustments
        assert!(conservative.lr_increase_factor() < aggressive.lr_increase_factor());
        assert!(conservative.lr_decrease_factor() > aggressive.lr_decrease_factor());

        // Conservative should have longer cooldowns
        assert!(conservative.cooldown_steps() > aggressive.cooldown_steps());

        // Conservative should have higher confidence threshold
        assert!(conservative.confidence_threshold() > aggressive.confidence_threshold());
    }

    #[test]
    fn test_recommended_strategy() {
        // Poor health should recommend Recovery
        assert_eq!(
            AdaptationStrategy::recommended_for_health(0.2, false),
            AdaptationStrategy::Recovery
        );

        // Moderate health should recommend Conservative
        assert_eq!(
            AdaptationStrategy::recommended_for_health(0.4, true),
            AdaptationStrategy::Conservative
        );

        // Good health with improvement should recommend Aggressive
        assert_eq!(
            AdaptationStrategy::recommended_for_health(0.85, true),
            AdaptationStrategy::Aggressive
        );

        // Normal health should recommend Balanced
        assert_eq!(
            AdaptationStrategy::recommended_for_health(0.6, true),
            AdaptationStrategy::Balanced
        );
    }

    #[test]
    fn test_strategy_properties() {
        assert!(AdaptationStrategy::Exploratory.allows_exploration());
        assert!(!AdaptationStrategy::Balanced.allows_exploration());

        assert!(AdaptationStrategy::Conservative.prioritizes_stability());
        assert!(AdaptationStrategy::Recovery.prioritizes_stability());
        assert!(!AdaptationStrategy::Aggressive.prioritizes_stability());
    }

    #[test]
    fn test_default_strategy() {
        assert_eq!(AdaptationStrategy::default(), AdaptationStrategy::Balanced);
    }
}
