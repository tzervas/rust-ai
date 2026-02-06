//! Mock PEFT (Parameter-Efficient Fine-Tuning) implementation

use serde::{Deserialize, Serialize};

/// Mock `LoRA` configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoraConfig {
    /// Rank of the `LoRA` matrices
    pub r: usize,
    /// Alpha parameter for scaling
    pub lora_alpha: f32,
    /// Dropout probability
    pub lora_dropout: f32,
    /// Target modules for `LoRA` adaptation
    pub target_modules: Vec<String>,
}

impl Default for LoraConfig {
    fn default() -> Self {
        Self {
            r: 8,
            lora_alpha: 16.0,
            lora_dropout: 0.1,
            target_modules: vec!["q_proj".to_string(), "v_proj".to_string()],
        }
    }
}

/// Mock adapter trait
pub trait Adapter {
    /// Apply the adapter to a model
    ///
    /// # Errors
    ///
    /// Returns an error if the adapter cannot be applied.
    fn apply(&self) -> Result<(), String>;

    /// Get adapter configuration
    fn config(&self) -> &LoraConfig;
}

/// Mock `LoRA` adapter
pub struct LoraAdapter {
    config: LoraConfig,
}

impl LoraAdapter {
    /// Create a new `LoRA` adapter
    #[must_use]
    pub fn new(config: LoraConfig) -> Self {
        Self { config }
    }
}

impl Adapter for LoraAdapter {
    fn apply(&self) -> Result<(), String> {
        Ok(())
    }

    fn config(&self) -> &LoraConfig {
        &self.config
    }
}
