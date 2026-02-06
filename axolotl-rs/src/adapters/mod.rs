//! Adapter integration layer.
//!
//! This module provides unified access to PEFT adapters (LoRA, QLoRA, etc.)
//! using either the real peft-rs/qlora-rs crates or mock implementations.

#[cfg(feature = "peft")]
use std::collections::HashMap;
#[cfg(feature = "peft")]
use std::path::Path;

use candle_core::Device;
use candle_nn::VarMap;

#[cfg(feature = "qlora")]
use crate::config::QuantizationSettings;
use crate::config::{AdapterType, AxolotlConfig, LoraSettings};
use crate::error::{AxolotlError, Result};

// Re-export based on features
#[cfg(feature = "peft")]
pub use peft_rs::{
    Adapter, AdapterConfig, LoraConfig as PeftLoraConfig, LoraLayer, Mergeable, PeftModel,
    SaveLoad, Trainable,
};

#[cfg(feature = "qlora")]
pub use qlora_rs::{QLoraConfig, QLoraLayer, QuantizationConfig, QuantizedLinear, QuantizedTensor};

/// Unified adapter wrapper that works with both real and mock implementations.
pub struct AdapterWrapper {
    /// The type of adapter being used
    pub adapter_type: AdapterType,
    /// Whether quantization is enabled
    pub quantized: bool,
    /// Trainable parameters (LoRA weights)
    pub trainable_params: VarMap,
    /// Device where adapter is loaded
    pub device: Device,
}

impl AdapterWrapper {
    /// Create a new adapter based on configuration.
    ///
    /// # Arguments
    /// * `config` - The axolotl configuration
    /// * `device` - Device to create adapter on
    ///
    /// # Errors
    /// Returns an error if the adapter cannot be created.
    pub fn new(config: &AxolotlConfig, device: &Device) -> Result<Self> {
        let trainable_params = VarMap::new();

        match config.adapter {
            AdapterType::None => Ok(Self {
                adapter_type: AdapterType::None,
                quantized: false,
                trainable_params,
                device: device.clone(),
            }),
            AdapterType::Lora => {
                tracing::info!(
                    "Creating LoRA adapter with r={}, alpha={}",
                    config.lora.r,
                    config.lora.alpha
                );
                Ok(Self {
                    adapter_type: AdapterType::Lora,
                    quantized: false,
                    trainable_params,
                    device: device.clone(),
                })
            }
            AdapterType::Qlora => {
                if config.quantization.is_none() {
                    return Err(AxolotlError::Config(
                        "QLoRA requires quantization settings".into(),
                    ));
                }
                tracing::info!(
                    "Creating QLoRA adapter with r={}, alpha={}, quantization enabled",
                    config.lora.r,
                    config.lora.alpha
                );
                Ok(Self {
                    adapter_type: AdapterType::Qlora,
                    quantized: true,
                    trainable_params,
                    device: device.clone(),
                })
            }
        }
    }

    /// Convert axolotl LoRA settings to peft-rs config.
    #[cfg(feature = "peft")]
    pub fn to_peft_lora_config(settings: &LoraSettings) -> PeftLoraConfig {
        PeftLoraConfig {
            r: settings.r,
            alpha: settings.alpha,
            dropout: settings.dropout,
            target_modules: settings.target_modules.clone(),
            ..Default::default()
        }
    }

    /// Convert axolotl quantization settings to qlora-rs config.
    #[cfg(feature = "qlora")]
    pub fn to_qlora_config(
        lora: &LoraSettings,
        quant: &QuantizationSettings,
    ) -> Result<QLoraConfig> {
        let quant_config = QuantizationConfig {
            block_size: quant.block_size,
            double_quant: quant.double_quant,
            ..Default::default()
        };

        let lora_config = Self::to_peft_lora_config(lora);

        Ok(QLoraConfig {
            lora: lora_config,
            quantization: quant_config,
            target_modules: lora.target_modules.clone(),
            cache_dequantized: false, // On-the-fly dequant for training
        })
    }

    /// Get the number of trainable parameters.
    pub fn trainable_param_count(&self) -> usize {
        self.trainable_params
            .all_vars()
            .iter()
            .map(|v| v.elem_count())
            .sum()
    }

    /// Apply adapter to a linear layer, returning a wrapped layer.
    #[cfg(feature = "peft")]
    pub fn wrap_linear(
        &self,
        in_features: usize,
        out_features: usize,
        lora_config: &PeftLoraConfig,
        vb: candle_nn::VarBuilder,
    ) -> Result<LoraLayer> {
        LoraLayer::new(in_features, out_features, lora_config.clone(), vb)
            .map_err(|e| AxolotlError::Model(format!("Failed to create LoRA layer: {}", e)))
    }

    /// Save adapter weights to a directory.
    ///
    /// # Arguments
    /// * `path` - Directory to save adapter files to
    /// * `lora_config` - LoRA configuration to save
    /// * `layers` - Map of layer names to LoraLayer instances
    ///
    /// # Errors
    /// Returns error if saving fails.
    #[cfg(feature = "peft")]
    pub fn save_adapter<P: AsRef<Path>>(
        &self,
        path: P,
        lora_config: &PeftLoraConfig,
        layers: &HashMap<String, LoraLayer>,
    ) -> Result<()> {
        use candle_core::Tensor;
        // Import SaveLoad trait to enable state_dict() method
        use crate::adapters::SaveLoad;

        let dir = path.as_ref();
        std::fs::create_dir_all(dir)?;

        // Collect all adapter weights into a single state dict
        let mut all_tensors: Vec<(String, Tensor)> = Vec::new();

        for (name, layer) in layers {
            let state = layer.state_dict().map_err(|e| {
                AxolotlError::Model(format!("Failed to get state dict for {}: {}", name, e))
            })?;

            for (key, tensor) in state {
                all_tensors.push((format!("{}.{}", name, key), tensor));
            }
        }

        // Save weights to safetensors
        let weights_path = dir.join("adapter_model.safetensors");
        let tensors_ref: Vec<(&str, Tensor)> = all_tensors
            .iter()
            .map(|(name, tensor)| (name.as_str(), tensor.clone()))
            .collect();

        safetensors::tensor::serialize_to_file(tensors_ref, &None, &weights_path).map_err(|e| {
            AxolotlError::Checkpoint(format!("Failed to save adapter weights: {}", e))
        })?;

        // Save config to JSON
        let config_path = dir.join("adapter_config.json");
        let config_json = serde_json::to_string_pretty(lora_config).map_err(|e| {
            AxolotlError::Checkpoint(format!("Failed to serialize adapter config: {}", e))
        })?;
        std::fs::write(&config_path, config_json)?;

        tracing::info!("Saved adapter with {} layers to {:?}", layers.len(), dir);
        Ok(())
    }

    /// Load adapter weights from a directory.
    ///
    /// # Arguments
    /// * `path` - Directory containing adapter files
    ///
    /// # Returns
    /// Tuple of (LoraConfig, HashMap of tensors)
    ///
    /// # Errors
    /// Returns error if loading fails.
    #[cfg(feature = "peft")]
    pub fn load_adapter<P: AsRef<Path>>(
        &self,
        path: P,
    ) -> Result<(PeftLoraConfig, HashMap<String, candle_core::Tensor>)> {
        let dir = path.as_ref();

        // Load config
        let config_path = dir.join("adapter_config.json");
        let config_json = std::fs::read_to_string(&config_path).map_err(|e| {
            AxolotlError::Checkpoint(format!("Failed to read adapter config: {}", e))
        })?;
        let config: PeftLoraConfig = serde_json::from_str(&config_json).map_err(|e| {
            AxolotlError::Checkpoint(format!("Failed to parse adapter config: {}", e))
        })?;

        // Load weights
        let weights_path = dir.join("adapter_model.safetensors");
        let tensors = candle_core::safetensors::load(&weights_path, &self.device).map_err(|e| {
            AxolotlError::Checkpoint(format!("Failed to load adapter weights: {}", e))
        })?;

        tracing::info!(
            "Loaded adapter with {} tensors from {:?}",
            tensors.len(),
            dir
        );
        Ok((config, tensors))
    }
}

/// Configuration for applying adapters to a model.
#[derive(Debug, Clone)]
pub struct AdapterApplicationConfig {
    /// Target module patterns (e.g., "q_proj", "v_proj")
    pub target_modules: Vec<String>,
    /// LoRA rank
    pub r: usize,
    /// LoRA alpha scaling
    pub alpha: usize,
    /// LoRA dropout
    pub dropout: f32,
}

impl From<&LoraSettings> for AdapterApplicationConfig {
    fn from(settings: &LoraSettings) -> Self {
        Self {
            target_modules: settings.target_modules.clone(),
            r: settings.r,
            alpha: settings.alpha,
            dropout: settings.dropout as f32,
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adapter_wrapper_creation() {
        let mut config = AxolotlConfig::from_preset("llama2-7b").unwrap();
        // Override to use LoRA (default preset uses QLoRA)
        config.adapter = AdapterType::Lora;
        config.quantization = None;
        let device = Device::Cpu;

        let wrapper = AdapterWrapper::new(&config, &device).unwrap();
        assert_eq!(wrapper.adapter_type, AdapterType::Lora);
        assert!(!wrapper.quantized);
    }

    #[test]
    fn test_adapter_wrapper_creation_qlora() {
        let config = AxolotlConfig::from_preset("llama2-7b").unwrap();
        let device = Device::Cpu;

        let wrapper = AdapterWrapper::new(&config, &device).unwrap();
        assert_eq!(wrapper.adapter_type, AdapterType::Qlora);
        assert!(wrapper.quantized);
    }

    #[test]
    fn test_adapter_application_config_from_lora_settings() {
        let lora_settings = LoraSettings {
            r: 16,
            alpha: 32,
            dropout: 0.1,
            target_modules: vec!["q_proj".into(), "v_proj".into()],
        };

        let app_config: AdapterApplicationConfig = (&lora_settings).into();
        assert_eq!(app_config.r, 16);
        assert_eq!(app_config.alpha, 32);
        assert!((app_config.dropout - 0.1).abs() < 0.001);
    }

    #[cfg(feature = "peft")]
    #[test]
    fn test_to_peft_lora_config() {
        let lora_settings = LoraSettings {
            r: 8,
            alpha: 16,
            dropout: 0.05,
            target_modules: vec!["q_proj".into()],
        };

        let peft_config = AdapterWrapper::to_peft_lora_config(&lora_settings);
        assert_eq!(peft_config.r, 8);
        assert_eq!(peft_config.alpha, 16);
        assert!((peft_config.dropout - 0.05).abs() < 0.001);
    }

    #[cfg(feature = "peft")]
    #[test]
    fn test_adapter_save_and_load() {
        use candle_nn::VarBuilder;
        use tempfile::TempDir;

        let temp_dir = TempDir::new().unwrap();
        let device = Device::Cpu;

        // Create adapter wrapper
        let config = AxolotlConfig::from_preset("llama2-7b").unwrap();
        let wrapper = AdapterWrapper::new(&config, &device).unwrap();

        // Create a test LoRA layer
        let lora_config = PeftLoraConfig {
            r: 8,
            alpha: 16,
            dropout: 0.0,
            target_modules: vec!["q_proj".into()],
            ..Default::default()
        };

        let vb = VarBuilder::zeros(candle_core::DType::F32, &device);
        let layer = LoraLayer::new(768, 768, lora_config.clone(), vb).unwrap();

        // Save adapter
        let mut layers = HashMap::new();
        layers.insert("model.layers.0.self_attn.q_proj".to_string(), layer);

        wrapper
            .save_adapter(temp_dir.path(), &lora_config, &layers)
            .unwrap();

        // Verify files exist
        assert!(temp_dir.path().join("adapter_model.safetensors").exists());
        assert!(temp_dir.path().join("adapter_config.json").exists());

        // Load adapter
        let (loaded_config, loaded_tensors) = wrapper.load_adapter(temp_dir.path()).unwrap();
        assert_eq!(loaded_config.r, 8);
        assert_eq!(loaded_config.alpha, 16);
        assert!(!loaded_tensors.is_empty());
    }
}
