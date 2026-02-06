//! Dataset loading and preprocessing.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::config::{DatasetConfig, DatasetFormat};
use crate::error::{AxolotlError, Result};

/// A single training example.
///
/// # Example
///
/// ```rust
/// use axolotl_rs::dataset::Example;
///
/// let example = Example {
///     input: "What is the capital of France?".to_string(),
///     output: "The capital of France is Paris.".to_string(),
///     text: "### Instruction:\nWhat is the capital of France?\n\n### Response:\nThe capital of France is Paris.".to_string(),
/// };
///
/// assert_eq!(example.output, "The capital of France is Paris.");
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Example {
    /// Input text (instruction/prompt).
    pub input: String,
    /// Target output.
    pub output: String,
    /// Full formatted text for training.
    pub text: String,
}

/// Dataset for training.
///
/// # Example
///
/// ```no_run
/// use axolotl_rs::dataset::Dataset;
/// use axolotl_rs::config::{DatasetConfig, DatasetFormat};
///
/// # fn main() -> axolotl_rs::Result<()> {
/// let config = DatasetConfig {
///     path: "./data/train.jsonl".to_string(),
///     format: DatasetFormat::Alpaca,
///     max_length: 2048,
///     val_split: 0.05,
///     ..Default::default()
/// };
///
/// // Load dataset
/// let dataset = Dataset::load(&config)?;
/// println!("Loaded {} training examples", dataset.len());
/// println!("Validation set size: {}", dataset.validation.len());
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct Dataset {
    /// Training examples.
    pub train: Vec<Example>,
    /// Validation examples.
    #[allow(dead_code)]
    pub validation: Vec<Example>,
    /// Configuration used.
    #[allow(dead_code)]
    pub config: DatasetConfig,
}

impl Dataset {
    /// Load dataset from configuration.
    ///
    /// Supports multiple formats:
    /// - Alpaca: `{"instruction": "", "input": "", "output": ""}`
    /// - ShareGPT: `{"conversations": [{"from": "human", "value": ""}, ...]}`
    /// - Completion: `{"text": ""}`
    /// - Custom: Configurable field names
    ///
    /// # Example - Alpaca Format
    ///
    /// ```no_run
    /// use axolotl_rs::dataset::Dataset;
    /// use axolotl_rs::config::{DatasetConfig, DatasetFormat};
    ///
    /// # fn main() -> axolotl_rs::Result<()> {
    /// let config = DatasetConfig {
    ///     path: "./data/alpaca.jsonl".to_string(),
    ///     format: DatasetFormat::Alpaca,
    ///     ..Default::default()
    /// };
    ///
    /// let dataset = Dataset::load(&config)?;
    /// assert!(!dataset.is_empty());
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Example - ShareGPT Format
    ///
    /// ```no_run
    /// use axolotl_rs::dataset::Dataset;
    /// use axolotl_rs::config::{DatasetConfig, DatasetFormat};
    ///
    /// # fn main() -> axolotl_rs::Result<()> {
    /// let config = DatasetConfig {
    ///     path: "./data/sharegpt.jsonl".to_string(),
    ///     format: DatasetFormat::Sharegpt,
    ///     ..Default::default()
    /// };
    ///
    /// let dataset = Dataset::load(&config)?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Example - Custom Format
    ///
    /// ```no_run
    /// use axolotl_rs::dataset::Dataset;
    /// use axolotl_rs::config::{DatasetConfig, DatasetFormat};
    ///
    /// # fn main() -> axolotl_rs::Result<()> {
    /// let config = DatasetConfig {
    ///     path: "./data/custom.jsonl".to_string(),
    ///     format: DatasetFormat::Custom,
    ///     input_field: "question".to_string(),
    ///     output_field: "answer".to_string(),
    ///     ..Default::default()
    /// };
    ///
    /// let dataset = Dataset::load(&config)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn load(config: &DatasetConfig) -> Result<Self> {
        let path = Path::new(&config.path);

        if !path.exists() {
            return Err(AxolotlError::Dataset(format!(
                "Dataset not found: {}",
                config.path
            )));
        }

        let examples = match config.format {
            DatasetFormat::Alpaca => load_alpaca(path, config)?,
            DatasetFormat::Sharegpt => load_sharegpt(path, config)?,
            DatasetFormat::Completion => load_completion(path, config)?,
            DatasetFormat::Custom => load_custom(path, config)?,
        };

        // Split into train/validation
        #[allow(clippy::cast_precision_loss)]
        let len_f32 = examples.len() as f32;
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            clippy::cast_precision_loss
        )]
        let split_idx = ((1.0 - config.val_split) * len_f32)
            .round()
            .max(0.0)
            .min(examples.len() as f32) as usize;
        let (train, validation) = examples.split_at(split_idx);

        Ok(Self {
            train: train.iter().cloned().collect(),
            validation: validation.iter().cloned().collect(),
            config: config.clone(),
        })
    }

    /// Get number of training examples.
    #[must_use]
    pub fn len(&self) -> usize {
        self.train.len()
    }

    /// Check if dataset is empty.
    #[must_use]
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.train.is_empty()
    }
}

/// Alpaca format: {"instruction": "", "input": "", "output": ""}
#[derive(Deserialize)]
struct AlpacaExample {
    instruction: String,
    #[serde(default)]
    input: String,
    output: String,
}

fn load_alpaca(path: &Path, _config: &DatasetConfig) -> Result<Vec<Example>> {
    let content = std::fs::read_to_string(path)?;
    let mut examples = Vec::new();

    for line in content.lines() {
        if line.trim().is_empty() {
            continue;
        }

        let alpaca: AlpacaExample = serde_json::from_str(line)
            .map_err(|e| AxolotlError::Dataset(format!("Failed to parse line: {e}")))?;

        let input = if alpaca.input.is_empty() {
            alpaca.instruction.clone()
        } else {
            format!("{}\n\n{}", alpaca.instruction, alpaca.input)
        };

        let text = format!(
            "### Instruction:\n{}\n\n### Response:\n{}",
            input, alpaca.output
        );

        examples.push(Example {
            input,
            output: alpaca.output,
            text,
        });
    }

    Ok(examples)
}

/// `ShareGPT` format: {"conversations": [{"from": "human", "value": ""}, {"from": "gpt", "value": ""}]}
#[derive(Deserialize)]
struct ShareGptExample {
    conversations: Vec<ShareGptMessage>,
}

#[derive(Deserialize)]
struct ShareGptMessage {
    from: String,
    value: String,
}

fn load_sharegpt(path: &Path, _config: &DatasetConfig) -> Result<Vec<Example>> {
    let content = std::fs::read_to_string(path)?;
    let mut examples = Vec::new();

    for line in content.lines() {
        if line.trim().is_empty() {
            continue;
        }

        let sharegpt: ShareGptExample = serde_json::from_str(line)
            .map_err(|e| AxolotlError::Dataset(format!("Failed to parse line: {e}")))?;

        let mut text = String::new();
        let mut input = String::new();
        let mut output = String::new();

        for msg in &sharegpt.conversations {
            match msg.from.as_str() {
                "human" | "user" => {
                    use std::fmt::Write;
                    write!(text, "### Human:\n{}\n\n", msg.value).unwrap();
                    input.clone_from(&msg.value);
                }
                "gpt" | "assistant" => {
                    use std::fmt::Write;
                    write!(text, "### Assistant:\n{}\n\n", msg.value).unwrap();
                    output.clone_from(&msg.value);
                }
                _ => {}
            }
        }

        if !output.is_empty() {
            examples.push(Example {
                input,
                output,
                text,
            });
        }
    }

    Ok(examples)
}

/// Completion format: {"text": ""}
fn load_completion(path: &Path, _config: &DatasetConfig) -> Result<Vec<Example>> {
    #[derive(Deserialize)]
    struct CompletionExample {
        text: String,
    }

    let content = std::fs::read_to_string(path)?;
    let mut examples = Vec::new();

    for line in content.lines() {
        if line.trim().is_empty() {
            continue;
        }

        let completion: CompletionExample = serde_json::from_str(line)
            .map_err(|e| AxolotlError::Dataset(format!("Failed to parse line: {e}")))?;

        examples.push(Example {
            input: String::new(),
            output: completion.text.clone(),
            text: completion.text,
        });
    }

    Ok(examples)
}

/// Custom format with configurable fields.
fn load_custom(path: &Path, config: &DatasetConfig) -> Result<Vec<Example>> {
    let content = std::fs::read_to_string(path)?;
    let mut examples = Vec::new();

    for line in content.lines() {
        if line.trim().is_empty() {
            continue;
        }

        let obj: serde_json::Value = serde_json::from_str(line)
            .map_err(|e| AxolotlError::Dataset(format!("Failed to parse line: {e}")))?;

        let input = obj
            .get(&config.input_field)
            .and_then(|v: &serde_json::Value| v.as_str())
            .unwrap_or("")
            .to_string();

        let output = obj
            .get(&config.output_field)
            .and_then(|v: &serde_json::Value| v.as_str())
            .unwrap_or("")
            .to_string();

        let text = format!("### Input:\n{input}\n\n### Output:\n{output}");

        examples.push(Example {
            input,
            output,
            text,
        });
    }

    Ok(examples)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    // ========== Alpaca Format Tests ==========

    #[test]
    fn test_load_alpaca_valid() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(
            file,
            r#"{{"instruction": "Test", "input": "", "output": "Response"}}"#
        )
        .unwrap();

        let config = DatasetConfig {
            path: file.path().to_string_lossy().into(),
            format: DatasetFormat::Alpaca,
            val_split: 0.0, // No validation split for this test
            ..Default::default()
        };

        let dataset = Dataset::load(&config).unwrap();
        assert_eq!(dataset.train.len(), 1);
        assert_eq!(dataset.train[0].output, "Response");
    }

    #[test]
    fn test_load_alpaca_empty_array() {
        let file = NamedTempFile::new().unwrap();
        // Empty file - no lines

        let config = DatasetConfig {
            path: file.path().to_string_lossy().into(),
            format: DatasetFormat::Alpaca,
            ..Default::default()
        };

        let dataset = Dataset::load(&config).unwrap();
        assert_eq!(dataset.train.len(), 0);
        assert!(dataset.is_empty());
    }

    #[test]
    fn test_load_alpaca_malformed_json() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, r#"{{"instruction": "Test", "output": "Response""#).unwrap(); // Missing closing brace

        let config = DatasetConfig {
            path: file.path().to_string_lossy().into(),
            format: DatasetFormat::Alpaca,
            ..Default::default()
        };

        let result = Dataset::load(&config);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Failed to parse line"));
    }

    #[test]
    fn test_load_alpaca_missing_required_fields() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, r#"{{"instruction": "Test only"}}"#).unwrap(); // Missing "output"

        let config = DatasetConfig {
            path: file.path().to_string_lossy().into(),
            format: DatasetFormat::Alpaca,
            ..Default::default()
        };

        let result = Dataset::load(&config);
        assert!(result.is_err());
    }

    #[test]
    fn test_load_alpaca_large_dataset() {
        let mut file = NamedTempFile::new().unwrap();
        // Create a larger dataset with 1000 examples
        for i in 0..1000 {
            writeln!(
                file,
                r#"{{"instruction": "Task {}", "input": "", "output": "Result {}"}}"#,
                i, i
            )
            .unwrap();
        }

        let config = DatasetConfig {
            path: file.path().to_string_lossy().into(),
            format: DatasetFormat::Alpaca,
            val_split: 0.1,
            ..Default::default()
        };

        let dataset = Dataset::load(&config).unwrap();
        assert_eq!(dataset.train.len(), 900);
        assert_eq!(dataset.validation.len(), 100);
    }

    #[test]
    fn test_load_alpaca_with_input_field() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(
            file,
            r#"{{"instruction": "Translate", "input": "Hello", "output": "Hola"}}"#
        )
        .unwrap();

        let config = DatasetConfig {
            path: file.path().to_string_lossy().into(),
            format: DatasetFormat::Alpaca,
            val_split: 0.0,
            ..Default::default()
        };

        let dataset = Dataset::load(&config).unwrap();
        assert_eq!(dataset.train.len(), 1);
        assert!(dataset.train[0].input.contains("Translate"));
        assert!(dataset.train[0].input.contains("Hello"));
    }

    // ========== ShareGPT Format Tests ==========

    #[test]
    fn test_load_sharegpt_valid() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(
            file,
            r#"{{"conversations": [{{"from": "human", "value": "Hello"}}, {{"from": "gpt", "value": "Hi there"}}]}}"#
        ).unwrap();

        let config = DatasetConfig {
            path: file.path().to_string_lossy().into(),
            format: DatasetFormat::Sharegpt,
            val_split: 0.0,
            ..Default::default()
        };

        let dataset = Dataset::load(&config).unwrap();
        assert_eq!(dataset.train.len(), 1);
        assert_eq!(dataset.train[0].input, "Hello");
        assert_eq!(dataset.train[0].output, "Hi there");
    }

    #[test]
    fn test_load_sharegpt_multi_turn() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(
            file,
            r#"{{"conversations": [{{"from": "human", "value": "Q1"}}, {{"from": "gpt", "value": "A1"}}, {{"from": "human", "value": "Q2"}}, {{"from": "gpt", "value": "A2"}}]}}"#
        ).unwrap();

        let config = DatasetConfig {
            path: file.path().to_string_lossy().into(),
            format: DatasetFormat::Sharegpt,
            val_split: 0.0,
            ..Default::default()
        };

        let dataset = Dataset::load(&config).unwrap();
        assert_eq!(dataset.train.len(), 1);
        // Should use last turn
        assert!(dataset.train[0].text.contains("Q2"));
        assert!(dataset.train[0].text.contains("A2"));
    }

    #[test]
    fn test_load_sharegpt_user_assistant_roles() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(
            file,
            r#"{{"conversations": [{{"from": "user", "value": "Help me"}}, {{"from": "assistant", "value": "Sure"}}]}}"#
        ).unwrap();

        let config = DatasetConfig {
            path: file.path().to_string_lossy().into(),
            format: DatasetFormat::Sharegpt,
            val_split: 0.0,
            ..Default::default()
        };

        let dataset = Dataset::load(&config).unwrap();
        assert_eq!(dataset.train.len(), 1);
        assert_eq!(dataset.train[0].input, "Help me");
        assert_eq!(dataset.train[0].output, "Sure");
    }

    #[test]
    fn test_load_sharegpt_empty_conversations() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, r#"{{"conversations": []}}"#).unwrap();

        let config = DatasetConfig {
            path: file.path().to_string_lossy().into(),
            format: DatasetFormat::Sharegpt,
            ..Default::default()
        };

        let dataset = Dataset::load(&config).unwrap();
        // Empty conversations should not produce examples
        assert_eq!(dataset.train.len(), 0);
    }

    #[test]
    fn test_load_sharegpt_missing_role_field() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, r#"{{"conversations": [{{"value": "Hello"}}]}}"#).unwrap(); // Missing "from" field

        let config = DatasetConfig {
            path: file.path().to_string_lossy().into(),
            format: DatasetFormat::Sharegpt,
            ..Default::default()
        };

        let result = Dataset::load(&config);
        assert!(result.is_err());
    }

    // ========== Completion Format Tests ==========

    #[test]
    fn test_load_completion_valid() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, r#"{{"text": "Sample completion text"}}"#).unwrap();

        let config = DatasetConfig {
            path: file.path().to_string_lossy().into(),
            format: DatasetFormat::Completion,
            val_split: 0.0,
            ..Default::default()
        };

        let dataset = Dataset::load(&config).unwrap();
        assert_eq!(dataset.train.len(), 1);
        assert_eq!(dataset.train[0].text, "Sample completion text");
        assert_eq!(dataset.train[0].output, "Sample completion text");
    }

    #[test]
    fn test_load_completion_missing_field() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, r#"{{"content": "Wrong field"}}"#).unwrap();

        let config = DatasetConfig {
            path: file.path().to_string_lossy().into(),
            format: DatasetFormat::Completion,
            ..Default::default()
        };

        let result = Dataset::load(&config);
        assert!(result.is_err());
    }

    #[test]
    fn test_load_completion_empty_string() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, r#"{{"text": ""}}"#).unwrap();

        let config = DatasetConfig {
            path: file.path().to_string_lossy().into(),
            format: DatasetFormat::Completion,
            val_split: 0.0,
            ..Default::default()
        };

        let dataset = Dataset::load(&config).unwrap();
        assert_eq!(dataset.train.len(), 1);
        assert_eq!(dataset.train[0].text, "");
    }

    // ========== Custom Format Tests ==========

    #[test]
    fn test_load_custom_valid() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(
            file,
            r#"{{"prompt": "Custom prompt", "response": "Custom response"}}"#
        )
        .unwrap();

        let config = DatasetConfig {
            path: file.path().to_string_lossy().into(),
            format: DatasetFormat::Custom,
            input_field: "prompt".into(),
            output_field: "response".into(),
            val_split: 0.0,
            ..Default::default()
        };

        let dataset = Dataset::load(&config).unwrap();
        assert_eq!(dataset.train.len(), 1);
        assert_eq!(dataset.train[0].input, "Custom prompt");
        assert_eq!(dataset.train[0].output, "Custom response");
    }

    #[test]
    fn test_load_custom_missing_field() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, r#"{{"prompt": "Only prompt"}}"#).unwrap();

        let config = DatasetConfig {
            path: file.path().to_string_lossy().into(),
            format: DatasetFormat::Custom,
            input_field: "prompt".into(),
            output_field: "response".into(),
            val_split: 0.0,
            ..Default::default()
        };

        let dataset = Dataset::load(&config).unwrap();
        assert_eq!(dataset.train.len(), 1);
        // Missing field should result in empty string
        assert_eq!(dataset.train[0].output, "");
    }

    // ========== Dataset Struct Tests ==========

    #[test]
    fn test_dataset_len_and_is_empty() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(
            file,
            r#"{{"instruction": "Test", "input": "", "output": "Response"}}"#
        )
        .unwrap();
        writeln!(
            file,
            r#"{{"instruction": "Test2", "input": "", "output": "Response2"}}"#
        )
        .unwrap();

        let config = DatasetConfig {
            path: file.path().to_string_lossy().into(),
            format: DatasetFormat::Alpaca,
            val_split: 0.0,
            ..Default::default()
        };

        let dataset = Dataset::load(&config).unwrap();
        assert_eq!(dataset.len(), 2);
        assert!(!dataset.is_empty());
    }

    #[test]
    fn test_dataset_split_ratios() {
        let mut file = NamedTempFile::new().unwrap();
        for i in 0..100 {
            writeln!(
                file,
                r#"{{"instruction": "Task {}", "input": "", "output": "Result {}"}}"#,
                i, i
            )
            .unwrap();
        }

        let config = DatasetConfig {
            path: file.path().to_string_lossy().into(),
            format: DatasetFormat::Alpaca,
            val_split: 0.2,
            ..Default::default()
        };

        let dataset = Dataset::load(&config).unwrap();
        assert_eq!(dataset.train.len(), 80);
        assert_eq!(dataset.validation.len(), 20);
    }

    #[test]
    fn test_dataset_split_zero() {
        let mut file = NamedTempFile::new().unwrap();
        for i in 0..10 {
            writeln!(
                file,
                r#"{{"instruction": "Task {}", "input": "", "output": "Result {}"}}"#,
                i, i
            )
            .unwrap();
        }

        let config = DatasetConfig {
            path: file.path().to_string_lossy().into(),
            format: DatasetFormat::Alpaca,
            val_split: 0.0,
            ..Default::default()
        };

        let dataset = Dataset::load(&config).unwrap();
        assert_eq!(dataset.train.len(), 10);
        assert_eq!(dataset.validation.len(), 0);
    }

    #[test]
    fn test_dataset_split_one() {
        let mut file = NamedTempFile::new().unwrap();
        for i in 0..10 {
            writeln!(
                file,
                r#"{{"instruction": "Task {}", "input": "", "output": "Result {}"}}"#,
                i, i
            )
            .unwrap();
        }

        let config = DatasetConfig {
            path: file.path().to_string_lossy().into(),
            format: DatasetFormat::Alpaca,
            val_split: 1.0,
            ..Default::default()
        };

        let dataset = Dataset::load(&config).unwrap();
        assert_eq!(dataset.train.len(), 0);
        assert_eq!(dataset.validation.len(), 10);
    }

    #[test]
    fn test_empty_dataset_split() {
        let file = NamedTempFile::new().unwrap();

        let config = DatasetConfig {
            path: file.path().to_string_lossy().into(),
            format: DatasetFormat::Alpaca,
            val_split: 0.2,
            ..Default::default()
        };

        let dataset = Dataset::load(&config).unwrap();
        assert_eq!(dataset.train.len(), 0);
        assert_eq!(dataset.validation.len(), 0);
        assert!(dataset.is_empty());
    }

    // ========== File I/O Error Tests ==========

    #[test]
    fn test_nonexistent_file() {
        let config = DatasetConfig {
            path: "/nonexistent/path/to/dataset.jsonl".into(),
            format: DatasetFormat::Alpaca,
            ..Default::default()
        };

        let result = Dataset::load(&config);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Dataset not found"));
    }

    #[test]
    fn test_invalid_utf8_content() {
        use std::io::Write;

        let mut file = NamedTempFile::new().unwrap();
        // Write invalid UTF-8 bytes
        file.write_all(&[0xFF, 0xFE, 0xFD]).unwrap();
        file.flush().unwrap();

        let config = DatasetConfig {
            path: file.path().to_string_lossy().into(),
            format: DatasetFormat::Alpaca,
            ..Default::default()
        };

        let result = Dataset::load(&config);
        assert!(result.is_err());
    }

    // ========== Dataset Format Router Test ==========

    #[test]
    fn test_dataset_format_routing() {
        // Test that Dataset::load correctly routes to different format loaders
        let mut alpaca_file = NamedTempFile::new().unwrap();
        writeln!(
            alpaca_file,
            r#"{{"instruction": "Test", "input": "", "output": "Response"}}"#
        )
        .unwrap();

        let mut sharegpt_file = NamedTempFile::new().unwrap();
        writeln!(
            sharegpt_file,
            r#"{{"conversations": [{{"from": "human", "value": "Q"}}, {{"from": "gpt", "value": "A"}}]}}"#
        ).unwrap();

        let mut completion_file = NamedTempFile::new().unwrap();
        writeln!(completion_file, r#"{{"text": "Completion"}}"#).unwrap();

        let mut custom_file = NamedTempFile::new().unwrap();
        writeln!(custom_file, r#"{{"question": "Q", "answer": "A"}}"#).unwrap();

        // Test Alpaca format
        let alpaca_config = DatasetConfig {
            path: alpaca_file.path().to_string_lossy().into(),
            format: DatasetFormat::Alpaca,
            val_split: 0.0,
            ..Default::default()
        };
        let alpaca_dataset = Dataset::load(&alpaca_config).unwrap();
        assert_eq!(alpaca_dataset.train.len(), 1);

        // Test ShareGPT format
        let sharegpt_config = DatasetConfig {
            path: sharegpt_file.path().to_string_lossy().into(),
            format: DatasetFormat::Sharegpt,
            val_split: 0.0,
            ..Default::default()
        };
        let sharegpt_dataset = Dataset::load(&sharegpt_config).unwrap();
        assert_eq!(sharegpt_dataset.train.len(), 1);

        // Test Completion format
        let completion_config = DatasetConfig {
            path: completion_file.path().to_string_lossy().into(),
            format: DatasetFormat::Completion,
            val_split: 0.0,
            ..Default::default()
        };
        let completion_dataset = Dataset::load(&completion_config).unwrap();
        assert_eq!(completion_dataset.train.len(), 1);

        // Test Custom format
        let custom_config = DatasetConfig {
            path: custom_file.path().to_string_lossy().into(),
            format: DatasetFormat::Custom,
            input_field: "question".into(),
            output_field: "answer".into(),
            val_split: 0.0,
            ..Default::default()
        };
        let custom_dataset = Dataset::load(&custom_config).unwrap();
        assert_eq!(custom_dataset.train.len(), 1);
    }
}
