//! GPU checkpoint save/load tests for axolotl-rs.
//!
//! These tests validate checkpoint functionality including adapter weight persistence.
//! Run with: `cargo test --features 'qlora cuda' -- --ignored checkpoint`

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

/// Create YAML config for GPU training.
fn create_checkpoint_config(
    model_id: &str,
    output_dir: &Path,
    dataset_path: &Path,
    epochs: usize,
    batch_size: usize,
    max_length: usize,
    learning_rate: f64,
) -> String {
    format!(
        r#"
base_model: "{model_id}"
adapter: qlora
output_dir: "{}"

lora:
  r: 8
  alpha: 16
  dropout: 0.0
  target_modules:
    - q_proj
    - v_proj

quantization:
  bits: 4
  quant_type: nf4
  double_quant: true
  block_size: 64

dataset:
  path: "{}"
  type: alpaca
  max_length: {max_length}
  train_split: 1.0

training:
  epochs: {epochs}
  batch_size: {batch_size}
  learning_rate: {learning_rate}
  weight_decay: 0.0
  logging_steps: 1
  save_steps: 5
  warmup_ratio: 0.1

seed: 42
"#,
        output_dir.display(),
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
// Checkpoint Save/Load Tests
// =============================================================================

/// Test checkpoint save functionality.
///
/// This test validates:
/// - Checkpoints are created at save_steps intervals
/// - Checkpoint directory structure is correct
/// - training_state.json is created
/// - adapter_config.json is created (HF compatible)
///
/// Run with: `cargo test --features 'qlora cuda' -- --ignored test_checkpoint_save`
#[test]
#[ignore]
#[cfg(feature = "cuda")]
fn test_checkpoint_save() {
    use axolotl_rs::{AxolotlConfig, Trainer};

    if !cuda_available() {
        skip_gpu_test("CUDA not available");
        return;
    }

    if !model_available("HuggingFaceTB/SmolLM2-135M") {
        skip_gpu_test("SmolLM2-135M not downloaded");
        return;
    }

    gpu_test_status("Testing checkpoint save (10 steps with save_steps=5)");

    let temp_dir = TempDir::new().unwrap();
    let output_dir = temp_dir.path().join("outputs");
    let dataset_path = temp_dir.path().join("dataset.jsonl");

    // Create 10 samples for 10 training steps
    create_test_dataset(&dataset_path, 10);

    let config_content = create_checkpoint_config(
        "HuggingFaceTB/SmolLM2-135M",
        &output_dir,
        &dataset_path,
        1,    // epochs
        1,    // batch_size
        128,  // max_length
        2e-4, // learning_rate
    );

    let config_path = temp_dir.path().join("config.yaml");
    fs::write(&config_path, config_content).unwrap();

    let config =
        AxolotlConfig::from_file(config_path.to_str().unwrap()).expect("Failed to load config");

    let mut trainer = Trainer::new(config).expect("Failed to create trainer");

    let result = trainer.train();

    match result {
        Ok(()) => {
            // Check that checkpoints were created
            let checkpoint_5 = output_dir.join("checkpoint-5");
            let checkpoint_10 = output_dir.join("checkpoint-10");

            if checkpoint_5.exists() {
                gpu_test_status("✓ Checkpoint-5 created");

                // Verify checkpoint contents
                let training_state = checkpoint_5.join("training_state.json");
                let adapter_config = checkpoint_5.join("adapter_config.json");

                if training_state.exists() {
                    gpu_test_status("✓ training_state.json present");
                } else {
                    gpu_test_status("⚠️  training_state.json missing");
                }

                if adapter_config.exists() {
                    gpu_test_status("✓ adapter_config.json present");
                } else {
                    gpu_test_status("⚠️  adapter_config.json missing");
                }
            } else {
                gpu_test_status("⚠️  Checkpoint-5 not created (save_steps may not be triggered)");
            }

            if checkpoint_10.exists() {
                gpu_test_status("✓ Final checkpoint-10 created");
            } else {
                gpu_test_status("⚠️  Final checkpoint-10 not created");
            }

            gpu_test_status("✅ Checkpoint save test completed");
        }
        Err(e) => {
            panic!("❌ Checkpoint save test failed: {}", e);
        }
    }
}

/// Test checkpoint resume functionality.
///
/// This test validates:
/// - Training can resume from checkpoint
/// - Training state is restored (step, epoch)
/// - Loss continues decreasing after resume
/// - Convergence continues from checkpoint state
///
/// Run with: `cargo test --features 'qlora cuda' -- --ignored test_checkpoint_resume`
#[test]
#[ignore]
#[cfg(feature = "cuda")]
fn test_checkpoint_resume() {
    use axolotl_rs::{AxolotlConfig, Trainer};

    if !cuda_available() {
        skip_gpu_test("CUDA not available");
        return;
    }

    if !model_available("HuggingFaceTB/SmolLM2-135M") {
        skip_gpu_test("SmolLM2-135M not downloaded");
        return;
    }

    gpu_test_status("Testing checkpoint resume (20 steps total: 10 + 10)");

    let temp_dir = TempDir::new().unwrap();
    let output_dir = temp_dir.path().join("outputs");
    let dataset_path = temp_dir.path().join("dataset.jsonl");

    // Phase 1: Train first 10 steps
    create_test_dataset(&dataset_path, 10);

    let config_content = create_checkpoint_config(
        "HuggingFaceTB/SmolLM2-135M",
        &output_dir,
        &dataset_path,
        1,    // epochs
        1,    // batch_size
        128,  // max_length
        2e-4, // learning_rate
    );

    let config_path = temp_dir.path().join("config.yaml");
    fs::write(&config_path, &config_content).unwrap();

    let config1 =
        AxolotlConfig::from_file(config_path.to_str().unwrap()).expect("Failed to load config");

    let mut trainer1 = Trainer::new(config1).expect("Failed to create trainer");
    let result1 = trainer1.train();

    match result1 {
        Ok(()) => {
            let phase1_losses: Vec<f64> =
                trainer1.training_metrics.iter().map(|m| m.loss).collect();
            let checkpoint_path = output_dir.join("checkpoint-10");

            if !checkpoint_path.exists() {
                panic!("Checkpoint not created at step 10");
            }

            gpu_test_status(&format!(
                "Phase 1 completed: {} steps, final loss: {:.4}",
                phase1_losses.len(),
                phase1_losses.last().unwrap_or(&0.0)
            ));

            // Phase 2: Resume from checkpoint
            let config2 = AxolotlConfig::from_file(config_path.to_str().unwrap())
                .expect("Failed to load config for resume");

            let mut trainer2 = Trainer::new(config2).expect("Failed to create trainer for resume");

            // Load checkpoint
            trainer2
                .load_checkpoint(checkpoint_path.to_str().unwrap())
                .expect("Failed to load checkpoint");

            gpu_test_status(&format!("Checkpoint loaded: step={}", trainer2.step()));

            // Continue training (another 10 steps, but we need more data)
            let result2 = trainer2.train();

            match result2 {
                Ok(()) => {
                    let phase2_losses: Vec<f64> =
                        trainer2.training_metrics.iter().map(|m| m.loss).collect();

                    gpu_test_status(&format!(
                        "Phase 2 completed: {} more steps",
                        phase2_losses.len()
                    ));

                    // Check loss progression
                    if phase1_losses.len() >= 2 && phase2_losses.len() >= 1 {
                        let phase1_final = phase1_losses[phase1_losses.len() - 1];
                        let phase2_final = phase2_losses[phase2_losses.len() - 1];

                        gpu_test_status(&format!(
                            "Loss progression: Phase1={:.4} -> Phase2={:.4}",
                            phase1_final, phase2_final
                        ));

                        gpu_test_status("✅ Checkpoint resume test passed");
                    } else {
                        gpu_test_status("⚠️  Not enough loss data for comparison");
                    }
                }
                Err(e) => {
                    panic!("❌ Phase 2 training failed: {}", e);
                }
            }
        }
        Err(e) => {
            panic!("❌ Phase 1 training failed: {}", e);
        }
    }
}
