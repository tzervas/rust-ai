//! peft-rs Adapter integration for BitNet.
//!
//! This module provides `BitNetAdapter` which implements the peft-rs `Adapter` trait,
//! enabling BitNet quantization to be used within the PEFT fine-tuning framework.

#[cfg(feature = "peft")]
use candle_core::Tensor;
#[cfg(feature = "peft")]
use candle_nn::VarMap;

use crate::config::BitNetConfig;
use crate::layer::BitLinear;

#[cfg(feature = "peft")]
use crate::error::Result;

/// BitNet adapter configuration for peft-rs integration.
#[derive(Debug, Clone)]
pub struct BitNetAdapterConfig {
    /// BitNet quantization configuration.
    pub bitnet: BitNetConfig,

    /// Target modules to apply BitNet quantization to.
    pub target_modules: Vec<String>,
}

impl Default for BitNetAdapterConfig {
    fn default() -> Self {
        Self {
            bitnet: BitNetConfig::default(),
            target_modules: vec![
                "q_proj".to_string(),
                "k_proj".to_string(),
                "v_proj".to_string(),
                "o_proj".to_string(),
                "gate_proj".to_string(),
                "up_proj".to_string(),
                "down_proj".to_string(),
            ],
        }
    }
}

impl BitNetAdapterConfig {
    /// Create a new adapter configuration.
    #[must_use]
    pub fn new(bitnet: BitNetConfig) -> Self {
        Self {
            bitnet,
            ..Default::default()
        }
    }

    /// Set target modules.
    #[must_use]
    pub fn with_target_modules(mut self, modules: Vec<String>) -> Self {
        self.target_modules = modules;
        self
    }
}

/// BitNet adapter for peft-rs integration.
///
/// This adapter wraps a BitLinear layer and implements the peft-rs Adapter trait
/// (when the `peft` feature is enabled).
#[derive(Debug)]
pub struct BitNetAdapter {
    /// The underlying BitLinear layer.
    layer: BitLinear,

    /// Adapter configuration.
    config: BitNetAdapterConfig,

    /// Whether the adapter is frozen.
    frozen: bool,
}

impl BitNetAdapter {
    /// Create a new BitNet adapter from a BitLinear layer.
    #[must_use]
    pub fn new(layer: BitLinear, config: BitNetAdapterConfig) -> Self {
        Self {
            layer,
            config,
            frozen: false,
        }
    }

    /// Get reference to the underlying layer.
    #[must_use]
    pub const fn layer(&self) -> &BitLinear {
        &self.layer
    }

    /// Get mutable reference to the underlying layer.
    pub fn layer_mut(&mut self) -> &mut BitLinear {
        &mut self.layer
    }

    /// Get reference to the configuration.
    #[must_use]
    pub const fn config(&self) -> &BitNetAdapterConfig {
        &self.config
    }

    /// Check if the adapter is frozen.
    #[must_use]
    pub const fn is_frozen(&self) -> bool {
        self.frozen
    }

    /// Freeze the adapter (disable gradient computation).
    pub fn freeze(&mut self) {
        self.frozen = true;
    }

    /// Unfreeze the adapter (enable gradient computation).
    pub fn unfreeze(&mut self) {
        self.frozen = false;
    }

    /// Get the number of quantized parameters.
    #[must_use]
    pub fn num_parameters(&self) -> usize {
        self.layer.in_features() * self.layer.out_features()
    }

    /// Get the compression ratio.
    #[must_use]
    pub fn compression_ratio(&self) -> f32 {
        self.layer.compression_ratio()
    }
}

#[cfg(feature = "peft")]
impl peft_rs::AdapterConfig for BitNetAdapterConfig {
    fn validate(&self) -> peft_rs::Result<()> {
        self.bitnet
            .validate()
            .map_err(|e| peft_rs::Error::Config(e.to_string()))
    }
}

#[cfg(feature = "peft")]
impl peft_rs::Adapter for BitNetAdapter {
    type Config = BitNetAdapterConfig;

    fn forward(&self, input: &Tensor, _base_output: Option<&Tensor>) -> peft_rs::Result<Tensor> {
        use candle_nn::Module;
        self.layer
            .forward(input)
            .map_err(|e| peft_rs::Error::Forward(e.to_string()))
    }

    fn num_parameters(&self) -> usize {
        self.num_parameters()
    }

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

#[cfg(feature = "peft")]
impl peft_rs::Trainable for BitNetAdapter {
    fn register_parameters(&self, _var_map: &mut VarMap, _prefix: &str) -> peft_rs::Result<()> {
        // BitNet weights are quantized and typically not trained directly
        // The bias (if present) could be registered here
        Ok(())
    }

    fn freeze(&mut self) {
        self.frozen = true;
    }

    fn unfreeze(&mut self) {
        self.frozen = false;
    }

    fn is_frozen(&self) -> bool {
        self.frozen
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use candle_core::Tensor;

    #[test]
    fn test_adapter_creation() {
        let device = Device::Cpu;
        let bitnet_config = BitNetConfig::default().with_group_size(64);
        let adapter_config = BitNetAdapterConfig::new(bitnet_config);

        let weight = Tensor::randn(0.0f32, 1.0, (64, 128), &device).unwrap();
        let layer = BitLinear::from_weight(&weight, None, &adapter_config.bitnet).unwrap();

        let adapter = BitNetAdapter::new(layer, adapter_config);

        assert_eq!(adapter.num_parameters(), 64 * 128);
        assert!(!adapter.is_frozen());
    }

    #[test]
    fn test_adapter_freeze_unfreeze() {
        let device = Device::Cpu;
        let bitnet_config = BitNetConfig::default().with_group_size(64);
        let adapter_config = BitNetAdapterConfig::new(bitnet_config);

        let weight = Tensor::randn(0.0f32, 1.0, (64, 128), &device).unwrap();
        let layer = BitLinear::from_weight(&weight, None, &adapter_config.bitnet).unwrap();

        let mut adapter = BitNetAdapter::new(layer, adapter_config);

        adapter.freeze();
        assert!(adapter.is_frozen());

        adapter.unfreeze();
        assert!(!adapter.is_frozen());
    }

    #[test]
    fn test_adapter_config_default() {
        let config = BitNetAdapterConfig::default();

        assert!(!config.target_modules.is_empty());
        assert!(config.target_modules.contains(&"q_proj".to_string()));
    }
}
