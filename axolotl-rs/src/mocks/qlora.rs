//! Mock `QLoRA` (Quantized `LoRA`) implementation

use serde::{Deserialize, Serialize};

/// Mock quantization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationConfig {
    /// Number of bits for quantization
    pub bits: u8,
    /// Use double quantization
    pub double_quant: bool,
    /// Quantization type
    pub quant_type: String,
}

impl Default for QuantizationConfig {
    fn default() -> Self {
        Self {
            bits: 4,
            double_quant: true,
            quant_type: "nf4".to_string(),
        }
    }
}

/// Mock quantize function
///
/// # Errors
///
/// Returns an error if the quantization configuration is invalid.
pub fn quantize(config: &QuantizationConfig) -> Result<(), String> {
    if config.bits != 4 && config.bits != 8 {
        return Err("Only 4-bit and 8-bit quantization supported".to_string());
    }
    Ok(())
}

/// Mock quantized model wrapper
pub struct QuantizedModel {
    config: QuantizationConfig,
}

impl QuantizedModel {
    /// Create a new quantized model
    ///
    /// # Errors
    ///
    /// Returns an error if the quantization configuration is invalid.
    pub fn new(config: QuantizationConfig) -> Result<Self, String> {
        quantize(&config)?;
        Ok(Self { config })
    }

    /// Get quantization config
    #[must_use]
    pub fn config(&self) -> &QuantizationConfig {
        &self.config
    }
}
