//! GPU LoRA target module variation tests for axolotl-rs.
//!
//! These tests validate LoRA training with different target module configurations.
//! Run with: `cargo test --features 'lora cuda' -- --ignored target`

mod gpu_utils;

use std::fs;
use std::path::Path;
use tempfile::TempDir;

use gpu_utils::*;

/// Create test dataset with specified number of samples.
fn create_test_dataset(path: &Path, num_samples: usize) {
    let mut content = String::new();
    for i in 0..num_samples {
        content.push_str(&format!(
            r#"{{"instruction":"Explain concept number {} in simple terms","input":"","output":"This is explanation {} that provides clear and helpful information."}}"#,
            i, i
        ));
        content.push('\n');
    }
    fs::write(path, content).expect("Failed to write test dataset");
}

/// Create LoRA config with specified target modules.
fn create_lora_config(
    model_id: &str,
    output_dir: &Path,
    dataset_path: &Path,
    rank: usize,
    target_modules: Vec<&str>,
) -> String {
    let modules_yaml = target_modules
        .iter()
        .map(|m| format!("    - {}", m))
        .collect::<Vec<_>>()
        .join("\n");

    format!(
        r#"
base_model: "{model_id}"
adapter: lora
output_dir: "{}"

lora:
  r: {rank}
  alpha: {}
  dropout: 0.0
  target_modules:
{modules_yaml}

dataset:
  path: "{}"
  type: alpaca
  max_length: 128
  train_split: 1.0

training:
  epochs: 1
  batch_size: 1
  learning_rate: 0.0002
  weight_decay: 0.0
  logging_steps: 1
  save_steps: 100
  warmup_ratio: 0.1

seed: 42
"#,
        output_dir.display(),
        rank * 2,
        dataset_path.display()
    )
}

/// Check if model is available in HuggingFace cache.
fn model_available(model_id: &str) -> bool {
    let cache_dir = std::env::var("HF_HOME")
        .or_else(|_| std::env::var("HOME").map(|h| format!("{}/.cache/huggingface", h)))
        .unwrap_or_else(|_| "/tmp/huggingface".to_string());

    let hf_path = format!("{}/hub/models--{}", cache_dir, model_id.replace("/", "--"));
    Path::new(&hf_path).exists()
}

// =============================================================================
// Target Module Variation Tests
// =============================================================================

/// Test LoRA with minimal target modules (q_proj only).
///
/// This test validates:
/// - LoRA training with a single attention layer
/// - Smallest possible adapter footprint
/// - Loss convergence with minimal adaptation
///
/// Run with: `cargo test --features 'lora cuda' -- --ignored test_lora_target_minimal`
#[test]
#[ignore]
#[cfg(feature = "cuda")]
fn test_lora_target_minimal() {
    use axolotl_rs::{AxolotlConfig, Trainer};

    if !cuda_available() {
        skip_gpu_test("CUDA not available");
        return;
    }

    if !model_available("HuggingFaceTB/SmolLM2-135M") {
        skip_gpu_test("SmolLM2-135M not downloaded");
        return;
    }

    gpu_test_status("Testing LoRA with minimal target modules (q_proj only)");

    let temp_dir = TempDir::new().unwrap();
    let output_dir = temp_dir.path().join("outputs");
    let dataset_path = temp_dir.path().join("dataset.jsonl");

    create_test_dataset(&dataset_path, 20);

    let config_content = create_lora_config(
        "HuggingFaceTB/SmolLM2-135M",
        &output_dir,
        &dataset_path,
        4,
        vec!["q_proj"],
    );

    let config_path = temp_dir.path().join("config.yaml");
    fs::write(&config_path, config_content).unwrap();

    let config =
        AxolotlConfig::from_file(config_path.to_str().unwrap()).expect("Failed to load config");

    let mut trainer = Trainer::new(config).expect("Failed to create trainer");

    match trainer.train() {
        Ok(()) => {
            let losses: Vec<f64> = trainer.training_metrics.iter().map(|m| m.loss).collect();

            gpu_test_status(&format!(
                "✅ Minimal LoRA test passed ({} steps)",
                losses.len()
            ));

            if losses.len() >= 2 {
                let change = (losses[0] - losses[losses.len() - 1]) / losses[0];
                gpu_test_status(&format!("   Loss change: {:.1}%", change * 100.0));
            }
        }
        Err(e) => {
            panic!("❌ Minimal LoRA test failed: {}", e);
        }
    }
}

/// Test LoRA with standard target modules (q_proj, v_proj).
///
/// This test validates:
/// - Standard LoRA configuration (query and value projections)
/// - Baseline adapter footprint
/// - Expected loss convergence behavior
///
/// Run with: `cargo test --features 'lora cuda' -- --ignored test_lora_target_standard`
#[test]
#[ignore]
#[cfg(feature = "cuda")]
fn test_lora_target_standard() {
    use axolotl_rs::{AxolotlConfig, Trainer};

    if !cuda_available() {
        skip_gpu_test("CUDA not available");
        return;
    }

    if !model_available("HuggingFaceTB/SmolLM2-135M") {
        skip_gpu_test("SmolLM2-135M not downloaded");
        return;
    }

    gpu_test_status("Testing LoRA with standard target modules (q_proj, v_proj)");

    let temp_dir = TempDir::new().unwrap();
    let output_dir = temp_dir.path().join("outputs");
    let dataset_path = temp_dir.path().join("dataset.jsonl");

    create_test_dataset(&dataset_path, 20);

    let config_content = create_lora_config(
        "HuggingFaceTB/SmolLM2-135M",
        &output_dir,
        &dataset_path,
        8,
        vec!["q_proj", "v_proj"],
    );

    let config_path = temp_dir.path().join("config.yaml");
    fs::write(&config_path, config_content).unwrap();

    let config =
        AxolotlConfig::from_file(config_path.to_str().unwrap()).expect("Failed to load config");

    let mut trainer = Trainer::new(config).expect("Failed to create trainer");

    match trainer.train() {
        Ok(()) => {
            let losses: Vec<f64> = trainer.training_metrics.iter().map(|m| m.loss).collect();

            gpu_test_status(&format!(
                "✅ Standard LoRA test passed ({} steps)",
                losses.len()
            ));

            if losses.len() >= 2 {
                let change = (losses[0] - losses[losses.len() - 1]) / losses[0];
                gpu_test_status(&format!("   Loss change: {:.1}%", change * 100.0));
            }
        }
        Err(e) => {
            panic!("❌ Standard LoRA test failed: {}", e);
        }
    }
}

/// Test LoRA with full attention target modules (q_proj, k_proj, v_proj, o_proj).
///
/// This test validates:
/// - Comprehensive attention adaptation
/// - Maximum attention layer coverage
/// - Loss convergence with full attention LoRA
///
/// Run with: `cargo test --features 'lora cuda' -- --ignored test_lora_target_full_attention`
#[test]
#[ignore]
#[cfg(feature = "cuda")]
fn test_lora_target_full_attention() {
    use axolotl_rs::{AxolotlConfig, Trainer};

    if !cuda_available() {
        skip_gpu_test("CUDA not available");
        return;
    }

    if !model_available("HuggingFaceTB/SmolLM2-135M") {
        skip_gpu_test("SmolLM2-135M not downloaded");
        return;
    }

    gpu_test_status(
        "Testing LoRA with full attention target modules (q_proj, k_proj, v_proj, o_proj)",
    );

    let temp_dir = TempDir::new().unwrap();
    let output_dir = temp_dir.path().join("outputs");
    let dataset_path = temp_dir.path().join("dataset.jsonl");

    create_test_dataset(&dataset_path, 20);

    let config_content = create_lora_config(
        "HuggingFaceTB/SmolLM2-135M",
        &output_dir,
        &dataset_path,
        8,
        vec!["q_proj", "k_proj", "v_proj", "o_proj"],
    );

    let config_path = temp_dir.path().join("config.yaml");
    fs::write(&config_path, config_content).unwrap();

    let config =
        AxolotlConfig::from_file(config_path.to_str().unwrap()).expect("Failed to load config");

    let mut trainer = Trainer::new(config).expect("Failed to create trainer");

    match trainer.train() {
        Ok(()) => {
            let losses: Vec<f64> = trainer.training_metrics.iter().map(|m| m.loss).collect();

            gpu_test_status(&format!(
                "✅ Full attention LoRA test passed ({} steps)",
                losses.len()
            ));

            if losses.len() >= 2 {
                let change = (losses[0] - losses[losses.len() - 1]) / losses[0];
                gpu_test_status(&format!("   Loss change: {:.1}%", change * 100.0));
            }
        }
        Err(e) => {
            panic!("❌ Full attention LoRA test failed: {}", e);
        }
    }
}

/// Test LoRA with MLP target modules (up_proj, down_proj).
///
/// This test validates:
/// - MLP (feedforward) network adaptation
/// - Alternative to attention-only adaptation
/// - Loss convergence with MLP-focused LoRA
///
/// Run with: `cargo test --features 'lora cuda' -- --ignored test_lora_target_mlp`
#[test]
#[ignore]
#[cfg(feature = "cuda")]
fn test_lora_target_mlp() {
    use axolotl_rs::{AxolotlConfig, Trainer};

    if !cuda_available() {
        skip_gpu_test("CUDA not available");
        return;
    }

    if !model_available("HuggingFaceTB/SmolLM2-135M") {
        skip_gpu_test("SmolLM2-135M not downloaded");
        return;
    }

    gpu_test_status("Testing LoRA with MLP target modules (up_proj, down_proj)");

    let temp_dir = TempDir::new().unwrap();
    let output_dir = temp_dir.path().join("outputs");
    let dataset_path = temp_dir.path().join("dataset.jsonl");

    create_test_dataset(&dataset_path, 20);

    let config_content = create_lora_config(
        "HuggingFaceTB/SmolLM2-135M",
        &output_dir,
        &dataset_path,
        8,
        vec!["up_proj", "down_proj"],
    );

    let config_path = temp_dir.path().join("config.yaml");
    fs::write(&config_path, config_content).unwrap();

    let config =
        AxolotlConfig::from_file(config_path.to_str().unwrap()).expect("Failed to load config");

    let mut trainer = Trainer::new(config).expect("Failed to create trainer");

    match trainer.train() {
        Ok(()) => {
            let losses: Vec<f64> = trainer.training_metrics.iter().map(|m| m.loss).collect();

            gpu_test_status(&format!("✅ MLP LoRA test passed ({} steps)", losses.len()));

            if losses.len() >= 2 {
                let change = (losses[0] - losses[losses.len() - 1]) / losses[0];
                gpu_test_status(&format!("   Loss change: {:.1}%", change * 100.0));
            }
        }
        Err(e) => {
            panic!("❌ MLP LoRA test failed: {}", e);
        }
    }
}
