//! Configuration for BitNet quantization and layers.

use serde::{Deserialize, Serialize};

/// Configuration for BitNet b1.58 quantization.
///
/// BitNet uses:
/// - Ternary weights: {-1, 0, +1} via AbsMean quantization
/// - INT8 activations: Per-token AbsMax scaling
///
/// # Reference
///
/// "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits"
/// <https://arxiv.org/abs/2402.17764>
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BitNetConfig {
    /// Group size for weight quantization.
    /// Weights are quantized in groups, each with its own scale.
    /// Typical values: 64, 128, 256.
    pub group_size: usize,

    /// Number of bits for activation quantization.
    /// BitNet b1.58 uses 8 bits (INT8).
    pub activation_bits: u8,

    /// Whether to use per-token activation scaling.
    /// If true, each token gets its own scale factor.
    /// If false, uses per-tensor scaling.
    pub per_token_activation: bool,

    /// Whether to apply RMS normalization before quantization.
    pub use_rms_norm: bool,

    /// Epsilon for numerical stability in normalization.
    pub eps: f32,

    /// Whether to enable Straight-Through Estimator for training.
    pub enable_ste: bool,
}

impl Default for BitNetConfig {
    fn default() -> Self {
        Self {
            group_size: 64,
            activation_bits: 8,
            per_token_activation: true,
            use_rms_norm: true,
            eps: 1e-5,
            enable_ste: true,
        }
    }
}

impl BitNetConfig {
    /// Create a new configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create configuration optimized for inference.
    ///
    /// Disables training-specific features like STE.
    #[must_use]
    pub fn inference() -> Self {
        Self {
            enable_ste: false,
            ..Default::default()
        }
    }

    /// Create configuration for training.
    ///
    /// Enables STE for gradient estimation through quantization.
    #[must_use]
    pub fn training() -> Self {
        Self {
            enable_ste: true,
            ..Default::default()
        }
    }

    /// Set the group size for weight quantization.
    #[must_use]
    pub const fn with_group_size(mut self, group_size: usize) -> Self {
        self.group_size = group_size;
        self
    }

    /// Set the activation bit width.
    #[must_use]
    pub const fn with_activation_bits(mut self, bits: u8) -> Self {
        self.activation_bits = bits;
        self
    }

    /// Enable or disable per-token activation scaling.
    #[must_use]
    pub const fn with_per_token_activation(mut self, enabled: bool) -> Self {
        self.per_token_activation = enabled;
        self
    }

    /// Enable or disable RMS normalization.
    #[must_use]
    pub const fn with_rms_norm(mut self, enabled: bool) -> Self {
        self.use_rms_norm = enabled;
        self
    }

    /// Enable or disable Straight-Through Estimator.
    #[must_use]
    pub const fn with_ste(mut self, enabled: bool) -> Self {
        self.enable_ste = enabled;
        self
    }

    /// Validate the configuration.
    ///
    /// # Errors
    ///
    /// Returns error if configuration is invalid.
    pub fn validate(&self) -> crate::Result<()> {
        if self.group_size == 0 {
            return Err(crate::BitNetError::InvalidConfig(
                "group_size must be > 0".to_string(),
            ));
        }

        if !self.group_size.is_power_of_two() {
            return Err(crate::BitNetError::InvalidConfig(
                "group_size must be a power of 2".to_string(),
            ));
        }

        if self.activation_bits == 0 || self.activation_bits > 16 {
            return Err(crate::BitNetError::InvalidConfig(
                "activation_bits must be 1-16".to_string(),
            ));
        }

        if self.eps <= 0.0 {
            return Err(crate::BitNetError::InvalidConfig(
                "eps must be > 0".to_string(),
            ));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = BitNetConfig::default();
        assert_eq!(config.group_size, 64);
        assert_eq!(config.activation_bits, 8);
        assert!(config.per_token_activation);
        assert!(config.use_rms_norm);
        assert!(config.enable_ste);
    }

    #[test]
    fn test_inference_config() {
        let config = BitNetConfig::inference();
        assert!(!config.enable_ste);
    }

    #[test]
    fn test_training_config() {
        let config = BitNetConfig::training();
        assert!(config.enable_ste);
    }

    #[test]
    fn test_builder_pattern() {
        let config = BitNetConfig::new()
            .with_group_size(128)
            .with_activation_bits(4)
            .with_per_token_activation(false)
            .with_ste(false);

        assert_eq!(config.group_size, 128);
        assert_eq!(config.activation_bits, 4);
        assert!(!config.per_token_activation);
        assert!(!config.enable_ste);
    }

    #[test]
    fn test_validation() {
        let valid = BitNetConfig::default();
        assert!(valid.validate().is_ok());

        let invalid_group = BitNetConfig {
            group_size: 0,
            ..Default::default()
        };
        assert!(invalid_group.validate().is_err());

        let invalid_bits = BitNetConfig {
            activation_bits: 0,
            ..Default::default()
        };
        assert!(invalid_bits.validate().is_err());

        let non_power_of_two = BitNetConfig {
            group_size: 65,
            ..Default::default()
        };
        assert!(non_power_of_two.validate().is_err());
    }
}
