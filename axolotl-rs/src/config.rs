//! Configuration parsing and validation.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::{AxolotlError, Result};

/// Main configuration for Axolotl training.
///
/// # Example
///
/// ```rust
/// use axolotl_rs::AxolotlConfig;
///
/// # fn main() -> axolotl_rs::Result<()> {
/// // Load from a YAML file
/// let config = AxolotlConfig::from_file("examples/configs/llama2-7b-qlora.yaml")?;
///
/// // Or create from a preset
/// let config = AxolotlConfig::from_preset("llama2-7b")?;
///
/// // Validate configuration
/// config.validate()?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AxolotlConfig {
    /// Base model identifier (`HuggingFace` model ID or local path).
    pub base_model: String,

    /// Adapter type.
    #[serde(default)]
    pub adapter: AdapterType,

    /// LoRA configuration (if using LoRA/QLoRA).
    #[serde(default)]
    pub lora: LoraSettings,

    /// Quantization configuration (if using QLoRA).
    #[serde(default)]
    pub quantization: Option<QuantizationSettings>,

    /// Dataset configuration.
    pub dataset: DatasetConfig,

    /// Training hyperparameters.
    #[serde(default)]
    pub training: TrainingConfig,

    /// Output directory.
    #[serde(default = "default_output_dir")]
    pub output_dir: String,

    /// Random seed.
    #[serde(default = "default_seed")]
    pub seed: u64,
}

fn default_output_dir() -> String {
    "./outputs".into()
}

fn default_seed() -> u64 {
    42
}

/// Adapter type for fine-tuning.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AdapterType {
    /// No adapter (full fine-tuning).
    None,
    /// Standard `LoRA`.
    #[default]
    Lora,
    /// 4-bit quantized `LoRA`.
    Qlora,
}

/// LoRA-specific settings.
///
/// # Example
///
/// ```rust
/// use axolotl_rs::config::LoraSettings;
///
/// let lora = LoraSettings {
///     r: 64,
///     alpha: 16,
///     dropout: 0.05,
///     target_modules: vec![
///         "q_proj".to_string(),
///         "k_proj".to_string(),
///         "v_proj".to_string(),
///         "o_proj".to_string(),
///     ],
/// };
///
/// assert_eq!(lora.r, 64);
/// assert_eq!(lora.alpha, 16);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoraSettings {
    /// Rank of low-rank decomposition.
    #[serde(default = "default_lora_r")]
    pub r: usize,

    /// Scaling factor.
    #[serde(default = "default_lora_alpha")]
    pub alpha: usize,

    /// Dropout probability.
    #[serde(default)]
    pub dropout: f64,

    /// Target modules for LoRA.
    #[serde(default = "default_target_modules")]
    pub target_modules: Vec<String>,
}

fn default_lora_r() -> usize {
    64
}
fn default_lora_alpha() -> usize {
    16
}
fn default_target_modules() -> Vec<String> {
    vec![
        "q_proj".into(),
        "k_proj".into(),
        "v_proj".into(),
        "o_proj".into(),
    ]
}

impl Default for LoraSettings {
    fn default() -> Self {
        Self {
            r: default_lora_r(),
            alpha: default_lora_alpha(),
            dropout: 0.05,
            target_modules: default_target_modules(),
        }
    }
}

/// Quantization settings for `QLoRA`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationSettings {
    /// Number of bits (4 for `QLoRA`).
    #[serde(default = "default_bits")]
    pub bits: u8,

    /// Quantization type.
    #[serde(default)]
    pub quant_type: QuantType,

    /// Use double quantization.
    #[serde(default = "default_true")]
    pub double_quant: bool,

    /// Block size for quantization.
    #[serde(default = "default_block_size")]
    pub block_size: usize,
}

fn default_bits() -> u8 {
    4
}
fn default_true() -> bool {
    true
}
fn default_block_size() -> usize {
    64
}

impl Default for QuantizationSettings {
    fn default() -> Self {
        Self {
            bits: 4,
            quant_type: QuantType::Nf4,
            double_quant: true,
            block_size: 64,
        }
    }
}

/// Quantization type.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum QuantType {
    /// 4-bit `NormalFloat`.
    #[default]
    Nf4,
    /// 4-bit float point.
    Fp4,
}

/// Dataset configuration.
///
/// # Example
///
/// ```rust
/// use axolotl_rs::config::{DatasetConfig, DatasetFormat};
///
/// let dataset_config = DatasetConfig {
///     path: "./data/train.jsonl".to_string(),
///     format: DatasetFormat::Alpaca,
///     max_length: 2048,
///     val_split: 0.05,
///     ..Default::default()
/// };
///
/// assert_eq!(dataset_config.max_length, 2048);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetConfig {
    /// Path to dataset (local file or `HuggingFace` dataset ID).
    pub path: String,

    /// Dataset format type.
    #[serde(default)]
    pub format: DatasetFormat,

    /// Field containing input text.
    #[serde(default = "default_input_field")]
    pub input_field: String,

    /// Field containing output text.
    #[serde(default = "default_output_field")]
    pub output_field: String,

    /// Maximum sequence length.
    #[serde(default = "default_max_length")]
    pub max_length: usize,

    /// Validation split ratio.
    #[serde(default = "default_val_split")]
    pub val_split: f32,
}

fn default_input_field() -> String {
    "instruction".into()
}
fn default_output_field() -> String {
    "output".into()
}
fn default_max_length() -> usize {
    2048
}
fn default_val_split() -> f32 {
    0.05
}

impl Default for DatasetConfig {
    fn default() -> Self {
        Self {
            path: String::new(),
            format: DatasetFormat::Alpaca,
            input_field: default_input_field(),
            output_field: default_output_field(),
            max_length: default_max_length(),
            val_split: default_val_split(),
        }
    }
}

/// Dataset format.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DatasetFormat {
    /// Alpaca format: instruction, input, output.
    #[default]
    Alpaca,
    /// `ShareGPT` format: conversations array.
    Sharegpt,
    /// Simple completion: just text.
    Completion,
    /// Custom format with specified fields.
    Custom,
}

/// Training hyperparameters.
///
/// # Example
///
/// ```rust
/// use axolotl_rs::TrainingConfig;
/// use axolotl_rs::config::LrScheduler;
///
/// let training = TrainingConfig {
///     epochs: 3,
///     batch_size: 4,
///     learning_rate: 2e-4,
///     lr_scheduler: LrScheduler::Cosine,
///     warmup_ratio: 0.03,
///     gradient_accumulation_steps: 4,
///     ..Default::default()
/// };
///
/// assert_eq!(training.epochs, 3);
/// assert_eq!(training.batch_size, 4);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Number of training epochs.
    #[serde(default = "default_epochs")]
    pub epochs: usize,

    /// Batch size per device.
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,

    /// Gradient accumulation steps.
    #[serde(default = "default_grad_accum")]
    pub gradient_accumulation_steps: usize,

    /// Learning rate.
    #[serde(default = "default_lr")]
    pub learning_rate: f64,

    /// Learning rate scheduler.
    #[serde(default)]
    pub lr_scheduler: LrScheduler,

    /// Warmup ratio.
    #[serde(default = "default_warmup")]
    pub warmup_ratio: f32,

    /// Weight decay.
    #[serde(default)]
    pub weight_decay: f64,

    /// Maximum gradient norm for clipping.
    #[serde(default = "default_grad_norm")]
    pub max_grad_norm: f32,

    /// Save checkpoint every N steps.
    #[serde(default = "default_save_steps")]
    pub save_steps: usize,

    /// Log every N steps.
    #[serde(default = "default_log_steps")]
    pub logging_steps: usize,

    /// Use gradient checkpointing.
    #[serde(default)]
    pub gradient_checkpointing: bool,

    /// Use mixed precision training.
    #[serde(default = "default_true")]
    pub mixed_precision: bool,
}

fn default_epochs() -> usize {
    3
}
fn default_batch_size() -> usize {
    4
}
fn default_grad_accum() -> usize {
    4
}
fn default_lr() -> f64 {
    2e-4
}
fn default_warmup() -> f32 {
    0.03
}
fn default_grad_norm() -> f32 {
    1.0
}
fn default_save_steps() -> usize {
    500
}
fn default_log_steps() -> usize {
    10
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            epochs: default_epochs(),
            batch_size: default_batch_size(),
            gradient_accumulation_steps: default_grad_accum(),
            learning_rate: default_lr(),
            lr_scheduler: LrScheduler::Cosine,
            warmup_ratio: default_warmup(),
            weight_decay: 0.0,
            max_grad_norm: default_grad_norm(),
            save_steps: default_save_steps(),
            logging_steps: default_log_steps(),
            gradient_checkpointing: false,
            mixed_precision: true,
        }
    }
}

/// Learning rate scheduler.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LrScheduler {
    /// Cosine annealing.
    #[default]
    Cosine,
    /// Linear decay.
    Linear,
    /// Constant learning rate.
    Constant,
}

impl AxolotlConfig {
    /// Load configuration from a YAML file.
    ///
    /// # Example
    ///
    /// ```rust
    /// use axolotl_rs::AxolotlConfig;
    ///
    /// # fn main() -> axolotl_rs::Result<()> {
    /// let config = AxolotlConfig::from_file("examples/configs/llama2-7b-qlora.yaml")?;
    /// assert_eq!(config.base_model, "meta-llama/Llama-2-7b-hf");
    /// # Ok(())
    /// # }
    /// ```
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = serde_yaml::from_str(&content)?;
        Ok(config)
    }

    /// Save configuration to a YAML file.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use axolotl_rs::AxolotlConfig;
    ///
    /// # fn main() -> axolotl_rs::Result<()> {
    /// let config = AxolotlConfig::from_preset("llama2-7b")?;
    ///
    /// // Save to a file
    /// config.to_file("my-config.yaml")?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let content = serde_yaml::to_string(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }

    /// Create a configuration from a preset.
    ///
    /// Available presets:
    /// - `"llama2-7b"` - LLaMA 2 7B with QLoRA
    /// - `"mistral-7b"` - Mistral 7B with QLoRA
    /// - `"phi3-mini"` - Phi-3 Mini with LoRA
    ///
    /// # Example
    ///
    /// ```rust
    /// use axolotl_rs::AxolotlConfig;
    ///
    /// # fn main() -> axolotl_rs::Result<()> {
    /// // Create a preset configuration
    /// let config = AxolotlConfig::from_preset("llama2-7b")?;
    /// assert_eq!(config.base_model, "meta-llama/Llama-2-7b-hf");
    ///
    /// // Mistral preset
    /// let config = AxolotlConfig::from_preset("mistral-7b")?;
    /// assert_eq!(config.base_model, "mistralai/Mistral-7B-v0.1");
    ///
    /// // Phi-3 preset
    /// let config = AxolotlConfig::from_preset("phi3-mini")?;
    /// assert_eq!(config.base_model, "microsoft/phi-3-mini-4k-instruct");
    /// # Ok(())
    /// # }
    /// ```
    pub fn from_preset(preset: &str) -> Result<Self> {
        match preset {
            "llama2-7b" => Ok(Self::llama2_7b_preset()),
            "mistral-7b" => Ok(Self::mistral_7b_preset()),
            "phi3-mini" => Ok(Self::phi3_mini_preset()),
            _ => Err(AxolotlError::Config(format!("Unknown preset: {preset}"))),
        }
    }

    /// Creates a configuration preset for LLaMA-2 7B model with QLoRA.
    #[must_use]
    pub fn llama2_7b_preset() -> Self {
        Self {
            base_model: "meta-llama/Llama-2-7b-hf".into(),
            adapter: AdapterType::Qlora,
            lora: LoraSettings {
                r: 64,
                alpha: 16,
                ..Default::default()
            },
            quantization: Some(QuantizationSettings::default()),
            dataset: DatasetConfig {
                path: "./data/train.jsonl".into(),
                ..Default::default()
            },
            training: TrainingConfig {
                learning_rate: 2e-4,
                ..Default::default()
            },
            output_dir: "./outputs/llama2-7b-qlora".into(),
            seed: 42,
        }
    }

    /// Creates a configuration preset for Mistral 7B model with QLoRA.
    #[must_use]
    pub fn mistral_7b_preset() -> Self {
        Self {
            base_model: "mistralai/Mistral-7B-v0.1".into(),
            adapter: AdapterType::Qlora,
            lora: LoraSettings {
                r: 64,
                alpha: 16,
                target_modules: vec![
                    "q_proj".into(),
                    "k_proj".into(),
                    "v_proj".into(),
                    "o_proj".into(),
                    "gate_proj".into(),
                    "up_proj".into(),
                    "down_proj".into(),
                ],
                ..Default::default()
            },
            quantization: Some(QuantizationSettings::default()),
            dataset: DatasetConfig {
                path: "./data/train.jsonl".into(),
                ..Default::default()
            },
            training: TrainingConfig::default(),
            output_dir: "./outputs/mistral-7b-qlora".into(),
            seed: 42,
        }
    }

    /// Creates a configuration preset for Phi-3 Mini model with LoRA.
    #[must_use]
    pub fn phi3_mini_preset() -> Self {
        Self {
            base_model: "microsoft/phi-3-mini-4k-instruct".into(),
            adapter: AdapterType::Lora,
            lora: LoraSettings {
                r: 32,
                alpha: 16,
                ..Default::default()
            },
            quantization: None,
            dataset: DatasetConfig {
                path: "./data/train.jsonl".into(),
                max_length: 4096,
                ..Default::default()
            },
            training: TrainingConfig {
                learning_rate: 1e-4,
                ..Default::default()
            },
            output_dir: "./outputs/phi3-mini-lora".into(),
            seed: 42,
        }
    }

    /// Validate the configuration.
    ///
    /// Checks for:
    /// - Required fields are set (base_model, dataset path)
    /// - LoRA rank is valid
    /// - QLoRA has quantization config
    ///
    /// # Example
    ///
    /// ```rust
    /// use axolotl_rs::AxolotlConfig;
    ///
    /// # fn main() -> axolotl_rs::Result<()> {
    /// let config = AxolotlConfig::from_preset("llama2-7b")?;
    ///
    /// // Validate configuration
    /// config.validate()?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// ```rust
    /// use axolotl_rs::AxolotlConfig;
    /// use axolotl_rs::config::AdapterType;
    ///
    /// # fn main() {
    /// let mut config = AxolotlConfig::from_preset("llama2-7b").unwrap();
    ///
    /// // This will fail validation
    /// config.base_model = String::new();
    /// assert!(config.validate().is_err());
    /// # }
    /// ```
    pub fn validate(&self) -> Result<()> {
        if self.base_model.is_empty() {
            return Err(AxolotlError::Config("base_model is required".into()));
        }

        if self.dataset.path.is_empty() {
            return Err(AxolotlError::Config("dataset.path is required".into()));
        }

        if self.lora.r == 0 {
            return Err(AxolotlError::Config("lora.r must be > 0".into()));
        }

        if matches!(self.adapter, AdapterType::Qlora) && self.quantization.is_none() {
            return Err(AxolotlError::Config(
                "quantization config required for QLoRA".into(),
            ));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_config_serialization() {
        let config = AxolotlConfig::llama2_7b_preset();
        let yaml = serde_yaml::to_string(&config).unwrap();
        let restored: AxolotlConfig = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(config.base_model, restored.base_model);
    }

    #[test]
    fn test_config_validation() {
        let mut config = AxolotlConfig::llama2_7b_preset();
        assert!(config.validate().is_ok());

        config.base_model = String::new();
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_presets() {
        assert!(AxolotlConfig::from_preset("llama2-7b").is_ok());
        assert!(AxolotlConfig::from_preset("mistral-7b").is_ok());
        assert!(AxolotlConfig::from_preset("phi3-mini").is_ok());
        assert!(AxolotlConfig::from_preset("invalid").is_err());
    }

    // LoraSettings tests
    #[test]
    fn test_lora_settings_valid() {
        let lora = LoraSettings {
            r: 64,
            alpha: 16,
            dropout: 0.05,
            target_modules: vec!["q_proj".into(), "v_proj".into()],
        };
        assert_eq!(lora.r, 64);
        assert_eq!(lora.alpha, 16);
        assert_eq!(lora.dropout, 0.05);
        assert_eq!(lora.target_modules.len(), 2);
    }

    #[test]
    fn test_lora_settings_invalid_r() {
        let mut config = AxolotlConfig::llama2_7b_preset();

        // r = 0 should fail validation
        config.lora.r = 0;
        assert!(config.validate().is_err());

        // Valid r values
        config.lora.r = 8;
        assert!(config.validate().is_ok());

        config.lora.r = 256;
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_lora_settings_default_target_modules() {
        let lora = LoraSettings::default();
        assert_eq!(lora.target_modules.len(), 4);
        assert!(lora.target_modules.contains(&"q_proj".to_string()));
        assert!(lora.target_modules.contains(&"k_proj".to_string()));
        assert!(lora.target_modules.contains(&"v_proj".to_string()));
        assert!(lora.target_modules.contains(&"o_proj".to_string()));
        assert_eq!(lora.r, 64);
        assert_eq!(lora.alpha, 16);
    }

    #[test]
    fn test_lora_settings_empty_target_modules() {
        let lora = LoraSettings {
            r: 64,
            alpha: 16,
            dropout: 0.05,
            target_modules: vec![],
        };
        // Empty target modules is allowed but should be handled by application logic
        assert_eq!(lora.target_modules.len(), 0);
    }

    // QuantizationSettings tests
    #[test]
    fn test_quantization_4bit() {
        let quant = QuantizationSettings {
            bits: 4,
            quant_type: QuantType::Nf4,
            double_quant: true,
            block_size: 64,
        };
        assert_eq!(quant.bits, 4);
        assert!(matches!(quant.quant_type, QuantType::Nf4));
        assert!(quant.double_quant);
        assert_eq!(quant.block_size, 64);
    }

    #[test]
    fn test_quantization_8bit() {
        let quant = QuantizationSettings {
            bits: 8,
            quant_type: QuantType::Fp4,
            double_quant: false,
            block_size: 128,
        };
        assert_eq!(quant.bits, 8);
        assert!(matches!(quant.quant_type, QuantType::Fp4));
        assert!(!quant.double_quant);
    }

    #[test]
    fn test_quantization_none() {
        let mut config = AxolotlConfig::llama2_7b_preset();
        config.adapter = AdapterType::Lora;
        config.quantization = None;
        // LoRA without quantization should be valid
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_quantization_compute_dtype() {
        let quant = QuantizationSettings::default();
        assert_eq!(quant.bits, 4);
        assert!(quant.double_quant);
        assert!(matches!(quant.quant_type, QuantType::Nf4));
    }

    // DatasetConfig tests
    #[test]
    fn test_dataset_config_all_formats() {
        // Test Alpaca format
        let alpaca = DatasetConfig {
            path: "./data.jsonl".into(),
            format: DatasetFormat::Alpaca,
            ..Default::default()
        };
        assert!(matches!(alpaca.format, DatasetFormat::Alpaca));

        // Test ShareGPT format
        let sharegpt = DatasetConfig {
            path: "./data.jsonl".into(),
            format: DatasetFormat::Sharegpt,
            ..Default::default()
        };
        assert!(matches!(sharegpt.format, DatasetFormat::Sharegpt));

        // Test Completion format
        let completion = DatasetConfig {
            path: "./data.jsonl".into(),
            format: DatasetFormat::Completion,
            ..Default::default()
        };
        assert!(matches!(completion.format, DatasetFormat::Completion));

        // Test Custom format
        let custom = DatasetConfig {
            path: "./data.jsonl".into(),
            format: DatasetFormat::Custom,
            ..Default::default()
        };
        assert!(matches!(custom.format, DatasetFormat::Custom));
    }

    #[test]
    fn test_dataset_config_split_ratios() {
        // Valid split ratio: 0.0
        let dataset1 = DatasetConfig {
            path: "./data.jsonl".into(),
            val_split: 0.0,
            ..Default::default()
        };
        assert_eq!(dataset1.val_split, 0.0);

        // Valid split ratio: 0.5
        let dataset2 = DatasetConfig {
            path: "./data.jsonl".into(),
            val_split: 0.5,
            ..Default::default()
        };
        assert_eq!(dataset2.val_split, 0.5);

        // Valid split ratio: 1.0
        let dataset3 = DatasetConfig {
            path: "./data.jsonl".into(),
            val_split: 1.0,
            ..Default::default()
        };
        assert_eq!(dataset3.val_split, 1.0);

        // Invalid split ratio: > 1.0 (currently not validated, but we test the value)
        let dataset4 = DatasetConfig {
            path: "./data.jsonl".into(),
            val_split: 1.5,
            ..Default::default()
        };
        assert_eq!(dataset4.val_split, 1.5);
    }

    #[test]
    fn test_dataset_config_missing_path() {
        let mut config = AxolotlConfig::llama2_7b_preset();
        config.dataset.path = String::new();
        assert!(config.validate().is_err());
    }

    // TrainingConfig tests
    #[test]
    fn test_training_config_batch_size_validation() {
        let training = TrainingConfig {
            batch_size: 1,
            gradient_accumulation_steps: 1,
            ..Default::default()
        };
        assert_eq!(training.batch_size, 1);

        let training2 = TrainingConfig {
            batch_size: 32,
            gradient_accumulation_steps: 4,
            ..Default::default()
        };
        assert_eq!(training2.batch_size, 32);
        assert_eq!(training2.gradient_accumulation_steps, 4);
    }

    #[test]
    fn test_training_config_learning_rate() {
        let training = TrainingConfig {
            learning_rate: 1e-5,
            ..Default::default()
        };
        assert_eq!(training.learning_rate, 1e-5);

        let training2 = TrainingConfig {
            learning_rate: 5e-4,
            ..Default::default()
        };
        assert_eq!(training2.learning_rate, 5e-4);
    }

    #[test]
    fn test_training_config_epochs() {
        let training = TrainingConfig {
            epochs: 1,
            ..Default::default()
        };
        assert_eq!(training.epochs, 1);

        let training2 = TrainingConfig {
            epochs: 10,
            ..Default::default()
        };
        assert_eq!(training2.epochs, 10);
    }

    // File I/O tests
    #[test]
    fn test_load_config_missing_file() {
        let result = AxolotlConfig::from_file("/nonexistent/path/config.yaml");
        assert!(result.is_err());
    }

    #[test]
    fn test_load_config_malformed_yaml() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "invalid: yaml: content: [[[").unwrap();

        let result = AxolotlConfig::from_file(temp_file.path());
        assert!(result.is_err());
    }

    #[test]
    fn test_save_config_roundtrip() {
        let config = AxolotlConfig::llama2_7b_preset();
        let temp_file = NamedTempFile::new().unwrap();

        // Save config
        config.to_file(temp_file.path()).unwrap();

        // Load it back
        let loaded = AxolotlConfig::from_file(temp_file.path()).unwrap();

        // Verify key fields match
        assert_eq!(config.base_model, loaded.base_model);
        assert_eq!(config.lora.r, loaded.lora.r);
        assert_eq!(config.lora.alpha, loaded.lora.alpha);
        assert_eq!(config.training.learning_rate, loaded.training.learning_rate);
    }

    #[test]
    fn test_save_config_invalid_path() {
        let config = AxolotlConfig::llama2_7b_preset();
        let result = config.to_file("/nonexistent/directory/config.yaml");
        assert!(result.is_err());
    }

    // Preset tests
    #[test]
    fn test_preset_mistral_7b() {
        let config = AxolotlConfig::from_preset("mistral-7b").unwrap();
        assert_eq!(config.base_model, "mistralai/Mistral-7B-v0.1");
        assert!(matches!(config.adapter, AdapterType::Qlora));
        assert_eq!(config.lora.r, 64);
        assert_eq!(config.lora.alpha, 16);
        // Mistral has 7 target modules
        assert_eq!(config.lora.target_modules.len(), 7);
        assert!(config.quantization.is_some());
    }

    #[test]
    fn test_preset_phi3_mini() {
        let config = AxolotlConfig::from_preset("phi3-mini").unwrap();
        assert_eq!(config.base_model, "microsoft/phi-3-mini-4k-instruct");
        assert!(matches!(config.adapter, AdapterType::Lora));
        assert_eq!(config.lora.r, 32);
        assert_eq!(config.lora.alpha, 16);
        assert!(config.quantization.is_none());
        assert_eq!(config.dataset.max_length, 4096);
        assert_eq!(config.training.learning_rate, 1e-4);
    }

    #[test]
    fn test_preset_unknown() {
        let result = AxolotlConfig::from_preset("unknown-model");
        assert!(result.is_err());

        let result2 = AxolotlConfig::from_preset("gpt-4");
        assert!(result2.is_err());
    }

    // Additional validation tests
    #[test]
    fn test_validation_qlora_requires_quantization() {
        let mut config = AxolotlConfig::llama2_7b_preset();
        config.adapter = AdapterType::Qlora;
        config.quantization = None;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validation_empty_base_model() {
        let mut config = AxolotlConfig::llama2_7b_preset();
        config.base_model = String::new();
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_lr_schedulers() {
        let cosine = TrainingConfig {
            lr_scheduler: LrScheduler::Cosine,
            ..Default::default()
        };
        assert!(matches!(cosine.lr_scheduler, LrScheduler::Cosine));

        let linear = TrainingConfig {
            lr_scheduler: LrScheduler::Linear,
            ..Default::default()
        };
        assert!(matches!(linear.lr_scheduler, LrScheduler::Linear));

        let constant = TrainingConfig {
            lr_scheduler: LrScheduler::Constant,
            ..Default::default()
        };
        assert!(matches!(constant.lr_scheduler, LrScheduler::Constant));
    }

    #[test]
    fn test_adapter_types() {
        let none_adapter = AxolotlConfig {
            adapter: AdapterType::None,
            ..AxolotlConfig::llama2_7b_preset()
        };
        assert!(matches!(none_adapter.adapter, AdapterType::None));

        let lora_adapter = AxolotlConfig {
            adapter: AdapterType::Lora,
            quantization: None,
            ..AxolotlConfig::llama2_7b_preset()
        };
        assert!(matches!(lora_adapter.adapter, AdapterType::Lora));
        assert!(lora_adapter.validate().is_ok());

        let qlora_adapter = AxolotlConfig::llama2_7b_preset();
        assert!(matches!(qlora_adapter.adapter, AdapterType::Qlora));
    }
}
