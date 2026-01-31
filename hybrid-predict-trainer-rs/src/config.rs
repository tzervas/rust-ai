//! Configuration types for hybrid predictive training.
//!
//! This module provides the configuration structures and builders for customizing
//! the hybrid trainer behavior, including phase lengths, confidence thresholds,
//! predictor architecture, and divergence detection parameters.
//!
//! # Overview
//!
//! The configuration system is designed to be:
//! - **Serializable** - Load/save configurations from TOML or JSON files
//! - **Validated** - Invalid configurations are rejected at construction time
//! - **Defaulted** - Sensible defaults work well for most use cases
//! - **Documented** - Each parameter has clear documentation and valid ranges
//!
//! # Example
//!
//! ```rust
//! use hybrid_predict_trainer_rs::config::{HybridTrainerConfig, PredictorConfig};
//!
//! // Using defaults
//! let config = HybridTrainerConfig::default();
//!
//! // Using builder pattern
//! let config = HybridTrainerConfig::builder()
//!     .warmup_steps(200)
//!     .full_steps(30)
//!     .max_predict_steps(100)
//!     .confidence_threshold(0.90)
//!     .build();
//!
//! // Loading from file
//! // let config = HybridTrainerConfig::from_file("config.toml")?;
//! ```

use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::error::HybridResult;

/// Main configuration for the hybrid trainer.
///
/// Controls all aspects of the training loop including phase lengths,
/// prediction thresholds, and divergence detection parameters.
///
/// # Defaults
///
/// The default configuration is tuned for a balanced tradeoff between
/// speedup and training quality:
///
/// | Parameter | Default | Description |
/// |-----------|---------|-------------|
/// | `warmup_steps` | 100 | Steps before prediction begins |
/// | `full_steps` | 20 | Full-compute steps per cycle |
/// | `max_predict_steps` | 80 | Maximum prediction phase length |
/// | `confidence_threshold` | 0.85 | Minimum confidence for predictions |
/// | `divergence_threshold` | 3.0 | Loss deviation threshold (σ) |
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridTrainerConfig {
    /// Number of warmup steps before prediction begins.
    ///
    /// During warmup, only full training is performed to establish baseline
    /// dynamics and train the initial predictor. Recommended range: 50-500.
    #[serde(default = "default_warmup_steps")]
    pub warmup_steps: usize,

    /// Number of full-compute steps per training cycle.
    ///
    /// After warmup, training alternates between full and predict phases.
    /// More full steps = higher quality but lower speedup. Range: 10-100.
    #[serde(default = "default_full_steps")]
    pub full_steps: usize,

    /// Maximum number of prediction steps per phase.
    ///
    /// The actual prediction length is adaptively determined based on
    /// confidence and past performance. This is the upper bound. Range: 20-200.
    #[serde(default = "default_max_predict_steps")]
    pub max_predict_steps: usize,

    /// Minimum confidence threshold for using predictions.
    ///
    /// When predictor confidence drops below this threshold, the trainer
    /// falls back to full training. Range: 0.5-0.99.
    #[serde(default = "default_confidence_threshold")]
    pub confidence_threshold: f32,

    /// Loss deviation threshold for divergence detection.
    ///
    /// If actual loss deviates from predicted by more than this many
    /// standard deviations, divergence handling is triggered. Range: 2.0-5.0.
    #[serde(default = "default_divergence_threshold")]
    pub divergence_threshold: f32,

    /// Predictor architecture configuration.
    #[serde(default)]
    pub predictor_config: PredictorConfig,

    /// Memory budget for the predictor in bytes.
    ///
    /// The predictor will be sized to fit within this budget.
    /// Default: 50MB (sufficient for 7B parameter models).
    #[serde(default = "default_predictor_memory_budget")]
    pub predictor_memory_budget: usize,

    /// Whether to collect detailed metrics during training.
    ///
    /// Enables step-level metrics collection for debugging and analysis.
    /// Has minimal performance overhead (~0.1%).
    #[serde(default = "default_collect_metrics")]
    pub collect_metrics: bool,

    /// Divergence detection configuration.
    #[serde(default)]
    pub divergence_config: DivergenceConfig,

    /// Checkpoint configuration.
    #[serde(default)]
    pub checkpoint_config: CheckpointConfig,
}

// Default value functions for serde
fn default_warmup_steps() -> usize {
    100
}
fn default_full_steps() -> usize {
    20
}
fn default_max_predict_steps() -> usize {
    80
}
fn default_confidence_threshold() -> f32 {
    0.85
}
fn default_divergence_threshold() -> f32 {
    3.0
}
fn default_predictor_memory_budget() -> usize {
    50 * 1024 * 1024
} // 50MB
fn default_collect_metrics() -> bool {
    true
}

impl Default for HybridTrainerConfig {
    fn default() -> Self {
        Self {
            warmup_steps: default_warmup_steps(),
            full_steps: default_full_steps(),
            max_predict_steps: default_max_predict_steps(),
            confidence_threshold: default_confidence_threshold(),
            divergence_threshold: default_divergence_threshold(),
            predictor_config: PredictorConfig::default(),
            predictor_memory_budget: default_predictor_memory_budget(),
            collect_metrics: default_collect_metrics(),
            divergence_config: DivergenceConfig::default(),
            checkpoint_config: CheckpointConfig::default(),
        }
    }
}

impl HybridTrainerConfig {
    /// Creates a new configuration builder.
    ///
    /// # Example
    ///
    /// ```rust
    /// use hybrid_predict_trainer_rs::config::HybridTrainerConfig;
    ///
    /// let config = HybridTrainerConfig::builder()
    ///     .warmup_steps(200)
    ///     .confidence_threshold(0.9)
    ///     .build();
    /// ```
    #[must_use]
    pub fn builder() -> HybridTrainerConfigBuilder {
        HybridTrainerConfigBuilder::default()
    }

    /// Loads configuration from a TOML file.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the TOML configuration file
    ///
    /// # Returns
    ///
    /// The parsed configuration or an error if loading/parsing fails.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or parsed.
    pub fn from_file<P: AsRef<Path>>(path: P) -> HybridResult<Self> {
        let content = std::fs::read_to_string(path.as_ref()).map_err(|e| {
            (
                crate::error::HybridTrainingError::ConfigError {
                    detail: format!("Failed to read config file: {e}"),
                },
                None,
            )
        })?;

        toml::from_str(&content).map_err(|e| {
            (
                crate::error::HybridTrainingError::ConfigError {
                    detail: format!("Failed to parse config: {e}"),
                },
                None,
            )
        })
    }

    /// Saves configuration to a TOML file.
    ///
    /// # Arguments
    ///
    /// * `path` - Path where the configuration should be saved
    ///
    /// # Errors
    ///
    /// Returns an error if serialization or writing fails.
    pub fn to_file<P: AsRef<Path>>(&self, path: P) -> HybridResult<()> {
        let content = toml::to_string_pretty(self).map_err(|e| {
            (
                crate::error::HybridTrainingError::ConfigError {
                    detail: format!("Failed to serialize config: {e}"),
                },
                None,
            )
        })?;

        std::fs::write(path.as_ref(), content).map_err(|e| {
            (
                crate::error::HybridTrainingError::ConfigError {
                    detail: format!("Failed to write config file: {e}"),
                },
                None,
            )
        })
    }

    /// Validates the configuration.
    ///
    /// Checks that all parameters are within valid ranges and consistent
    /// with each other.
    ///
    /// # Returns
    ///
    /// `Ok(())` if valid, or an error describing the validation failure.
    pub fn validate(&self) -> HybridResult<()> {
        if self.warmup_steps == 0 {
            return Err((
                crate::error::HybridTrainingError::ConfigError {
                    detail: "warmup_steps must be > 0".to_string(),
                },
                None,
            ));
        }

        if self.confidence_threshold <= 0.0 || self.confidence_threshold >= 1.0 {
            return Err((
                crate::error::HybridTrainingError::ConfigError {
                    detail: "confidence_threshold must be in (0, 1)".to_string(),
                },
                None,
            ));
        }

        if self.divergence_threshold <= 0.0 {
            return Err((
                crate::error::HybridTrainingError::ConfigError {
                    detail: "divergence_threshold must be > 0".to_string(),
                },
                None,
            ));
        }

        Ok(())
    }
}

/// Builder for `HybridTrainerConfig`.
#[derive(Debug, Default)]
pub struct HybridTrainerConfigBuilder {
    warmup_steps: Option<usize>,
    full_steps: Option<usize>,
    max_predict_steps: Option<usize>,
    confidence_threshold: Option<f32>,
    divergence_threshold: Option<f32>,
    predictor_config: Option<PredictorConfig>,
    predictor_memory_budget: Option<usize>,
    collect_metrics: Option<bool>,
}

impl HybridTrainerConfigBuilder {
    /// Sets the number of warmup steps.
    #[must_use]
    pub fn warmup_steps(mut self, steps: usize) -> Self {
        self.warmup_steps = Some(steps);
        self
    }

    /// Sets the number of full training steps per cycle.
    #[must_use]
    pub fn full_steps(mut self, steps: usize) -> Self {
        self.full_steps = Some(steps);
        self
    }

    /// Sets the maximum prediction steps.
    #[must_use]
    pub fn max_predict_steps(mut self, steps: usize) -> Self {
        self.max_predict_steps = Some(steps);
        self
    }

    /// Sets the confidence threshold.
    #[must_use]
    pub fn confidence_threshold(mut self, threshold: f32) -> Self {
        self.confidence_threshold = Some(threshold);
        self
    }

    /// Sets the divergence threshold.
    #[must_use]
    pub fn divergence_threshold(mut self, threshold: f32) -> Self {
        self.divergence_threshold = Some(threshold);
        self
    }

    /// Sets the predictor configuration.
    #[must_use]
    pub fn predictor_config(mut self, config: PredictorConfig) -> Self {
        self.predictor_config = Some(config);
        self
    }

    /// Sets the predictor memory budget.
    #[must_use]
    pub fn predictor_memory_budget(mut self, budget: usize) -> Self {
        self.predictor_memory_budget = Some(budget);
        self
    }

    /// Sets whether to collect metrics.
    #[must_use]
    pub fn collect_metrics(mut self, collect: bool) -> Self {
        self.collect_metrics = Some(collect);
        self
    }

    /// Builds the configuration with defaults for unset values.
    pub fn build(self) -> HybridTrainerConfig {
        HybridTrainerConfig {
            warmup_steps: self.warmup_steps.unwrap_or_else(default_warmup_steps),
            full_steps: self.full_steps.unwrap_or_else(default_full_steps),
            max_predict_steps: self
                .max_predict_steps
                .unwrap_or_else(default_max_predict_steps),
            confidence_threshold: self
                .confidence_threshold
                .unwrap_or_else(default_confidence_threshold),
            divergence_threshold: self
                .divergence_threshold
                .unwrap_or_else(default_divergence_threshold),
            predictor_config: self.predictor_config.unwrap_or_default(),
            predictor_memory_budget: self
                .predictor_memory_budget
                .unwrap_or_else(default_predictor_memory_budget),
            collect_metrics: self.collect_metrics.unwrap_or_else(default_collect_metrics),
            divergence_config: DivergenceConfig::default(),
            checkpoint_config: CheckpointConfig::default(),
        }
    }
}

/// Predictor architecture configuration.
///
/// Supports three predictor types with different accuracy/overhead tradeoffs:
///
/// - **Linear**: Lowest overhead, suitable for stable training dynamics
/// - **MLP**: Balanced accuracy and overhead, good default choice
/// - **RSSM**: Highest accuracy, best for complex dynamics
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum PredictorConfig {
    /// Linear predictor with minimal overhead.
    ///
    /// Uses a simple linear transformation from encoded state to predictions.
    /// Best for: stable fine-tuning with predictable dynamics.
    Linear {
        /// Dimension of the feature encoding.
        #[serde(default = "default_linear_feature_dim")]
        feature_dim: usize,

        /// L2 regularization strength.
        #[serde(default = "default_l2_regularization")]
        l2_regularization: f32,
    },

    /// Multi-layer perceptron predictor.
    ///
    /// Uses a feedforward network for non-linear prediction.
    /// Best for: general use cases with moderate complexity.
    MLP {
        /// Hidden layer dimensions.
        #[serde(default = "default_mlp_hidden_dims")]
        hidden_dims: Vec<usize>,

        /// Activation function.
        #[serde(default = "default_activation")]
        activation: Activation,

        /// Dropout rate.
        #[serde(default = "default_dropout")]
        dropout: f32,
    },

    /// RSSM-style recurrent predictor.
    ///
    /// Uses a recurrent state-space model with deterministic and stochastic
    /// components for maximum prediction accuracy.
    /// Best for: complex dynamics, longer prediction horizons.
    RSSM {
        /// Dimension of deterministic (GRU) state.
        #[serde(default = "default_deterministic_dim")]
        deterministic_dim: usize,

        /// Dimension of stochastic latent variables.
        #[serde(default = "default_stochastic_dim")]
        stochastic_dim: usize,

        /// Number of categorical distributions for stochastic state.
        #[serde(default = "default_num_categoricals")]
        num_categoricals: usize,

        /// Number of ensemble members for uncertainty estimation.
        #[serde(default = "default_ensemble_size")]
        ensemble_size: usize,
    },
}

// Default value functions for PredictorConfig
fn default_linear_feature_dim() -> usize {
    128
}
fn default_l2_regularization() -> f32 {
    0.01
}
fn default_mlp_hidden_dims() -> Vec<usize> {
    vec![256, 128]
}
fn default_activation() -> Activation {
    Activation::Gelu
}
fn default_dropout() -> f32 {
    0.1
}
fn default_deterministic_dim() -> usize {
    256
}
fn default_stochastic_dim() -> usize {
    32
}
fn default_num_categoricals() -> usize {
    32
}
fn default_ensemble_size() -> usize {
    3
}

impl Default for PredictorConfig {
    fn default() -> Self {
        Self::RSSM {
            deterministic_dim: default_deterministic_dim(),
            stochastic_dim: default_stochastic_dim(),
            num_categoricals: default_num_categoricals(),
            ensemble_size: default_ensemble_size(),
        }
    }
}

/// Activation function options for MLP predictor.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Activation {
    /// Rectified Linear Unit.
    Relu,

    /// Gaussian Error Linear Unit (default).
    #[default]
    Gelu,

    /// Sigmoid Linear Unit.
    Silu,

    /// Hyperbolic tangent.
    Tanh,
}

/// Configuration for divergence detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DivergenceConfig {
    /// Gradient norm multiplier threshold for explosion detection.
    ///
    /// If gradient norm exceeds baseline × this value, it's flagged as potential explosion.
    #[serde(default = "default_gradient_norm_multiplier")]
    pub gradient_norm_multiplier: f32,

    /// Loss deviation threshold in standard deviations.
    #[serde(default = "default_loss_sigma_threshold")]
    pub loss_sigma_threshold: f32,

    /// Threshold below which gradients are considered vanishing.
    #[serde(default = "default_vanishing_gradient_threshold")]
    pub vanishing_gradient_threshold: f32,

    /// How often to check for divergence (in steps).
    #[serde(default = "default_check_interval")]
    pub check_interval_steps: usize,
}

fn default_gradient_norm_multiplier() -> f32 {
    100.0
}
fn default_loss_sigma_threshold() -> f32 {
    3.0
}
fn default_vanishing_gradient_threshold() -> f32 {
    0.01
}
fn default_check_interval() -> usize {
    10
}

impl Default for DivergenceConfig {
    fn default() -> Self {
        Self {
            gradient_norm_multiplier: default_gradient_norm_multiplier(),
            loss_sigma_threshold: default_loss_sigma_threshold(),
            vanishing_gradient_threshold: default_vanishing_gradient_threshold(),
            check_interval_steps: default_check_interval(),
        }
    }
}

/// Configuration for checkpointing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointConfig {
    /// Save checkpoint every N steps.
    #[serde(default = "default_save_interval")]
    pub save_interval: usize,

    /// Number of checkpoints to keep.
    #[serde(default = "default_keep_last_n")]
    pub keep_last_n: usize,

    /// Whether to include predictor state in checkpoints.
    #[serde(default = "default_include_predictor")]
    pub include_predictor: bool,
}

fn default_save_interval() -> usize {
    1000
}
fn default_keep_last_n() -> usize {
    3
}
fn default_include_predictor() -> bool {
    true
}

impl Default for CheckpointConfig {
    fn default() -> Self {
        Self {
            save_interval: default_save_interval(),
            keep_last_n: default_keep_last_n(),
            include_predictor: default_include_predictor(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_is_valid() {
        let config = HybridTrainerConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_builder_pattern() {
        let config = HybridTrainerConfig::builder()
            .warmup_steps(200)
            .full_steps(30)
            .confidence_threshold(0.9)
            .build();

        assert_eq!(config.warmup_steps, 200);
        assert_eq!(config.full_steps, 30);
        assert!((config.confidence_threshold - 0.9).abs() < f32::EPSILON);
    }

    #[test]
    fn test_serialization_roundtrip() {
        let config = HybridTrainerConfig::default();
        let toml_str = toml::to_string(&config).unwrap();
        let parsed: HybridTrainerConfig = toml::from_str(&toml_str).unwrap();

        assert_eq!(config.warmup_steps, parsed.warmup_steps);
        assert_eq!(config.full_steps, parsed.full_steps);
    }

    #[test]
    fn test_invalid_confidence_threshold() {
        let config = HybridTrainerConfig {
            confidence_threshold: 1.5,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }
}
