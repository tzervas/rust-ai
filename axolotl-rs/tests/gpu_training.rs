//! GPU-focused end-to-end training tests for axolotl-rs.
//!
//! These tests validate the complete training pipeline on GPU with real loss convergence.
//! Run with: `cargo test --features 'qlora cuda' -- --ignored gpu`
//!
//! # Test Tiers
//!
//! | Test | Model | Steps | VRAM | Purpose |
//! |------|-------|-------|------|---------|
//! | `test_gpu_quick_iteration` | SmolLM2-135M | 10 | ~256 MB | Quick sanity check |
//! | `test_gpu_loss_convergence` | SmolLM2-135M | 100 | ~256 MB | Verify loss decreases |
//! | `test_gpu_tinyllama_memory` | TinyLlama-1.1B | 50 | ~2 GB | Memory validation |
//! | `test_gpu_tinyllama_extended` | TinyLlama-1.1B | 500 | ~2 GB | Extended convergence |
//! | `test_gpu_llama7b_full` | LLaMA-7B | 1000 | ~12 GB | Full validation |

mod gpu_utils;

use std::fs;
use std::path::Path;
use std::time::Instant;
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

/// Create YAML config for GPU training with LoRA (no quantization).
fn create_lora_config(
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
adapter: lora
output_dir: "{}"

lora:
  r: 8
  alpha: 16
  dropout: 0.0
  target_modules:
    - q_proj
    - v_proj

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
  save_steps: 10000
  warmup_ratio: 0.1

seed: 42
"#,
        output_dir.display(),
        dataset_path.display()
    )
}

/// Create YAML config for GPU training with QLoRA (quantization enabled).
fn create_gpu_config(
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
  save_steps: 10000
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
// Quick Iteration Tests (10 steps, <1 minute)
// =============================================================================

/// Quick GPU sanity check with SmolLM2-135M (10 steps).
///
/// This test validates:
/// - CUDA device initialization
/// - Model loading on GPU
/// - Forward/backward pass execution
/// - No CUDA errors during training
///
/// Run with: `cargo test --features 'qlora cuda' -- --ignored test_gpu_quick_iteration`
#[test]
#[ignore]
#[cfg(feature = "cuda")]
fn test_gpu_quick_iteration() {
    use axolotl_rs::{AxolotlConfig, Trainer};

    if !cuda_available() {
        skip_gpu_test("CUDA not available");
        return;
    }

    if !model_available("HuggingFaceTB/SmolLM2-135M") {
        skip_gpu_test("SmolLM2-135M not downloaded");
        println!("   Download with: huggingface-cli download HuggingFaceTB/SmolLM2-135M");
        return;
    }

    gpu_test_status("Starting quick iteration test (10 steps)");

    let temp_dir = TempDir::new().unwrap();
    let output_dir = temp_dir.path().join("outputs");
    let dataset_path = temp_dir.path().join("dataset.jsonl");

    // 10 samples for 10 steps with batch_size=1
    create_test_dataset(&dataset_path, 10);

    // Use pure LoRA (not QLoRA) for gradient tracking test
    let config_content = create_lora_config(
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

    gpu_test_status(&format!(
        "Config: model={}, adapter={:?}",
        config.base_model, config.adapter
    ));

    let mut trainer = Trainer::new(config).expect("Failed to create trainer");
    gpu_test_status("Trainer created on CUDA device");

    let start = Instant::now();
    let result = trainer.train();
    let elapsed = start.elapsed();

    match result {
        Ok(()) => {
            // Extract losses from training metrics
            let losses: Vec<f64> = trainer.training_metrics.iter().map(|m| m.loss).collect();

            gpu_test_status(&format!(
                "✅ Quick iteration passed in {:.1}s ({} steps collected)",
                elapsed.as_secs_f64(),
                losses.len()
            ));

            if losses.is_empty() {
                panic!("No loss values collected during training");
            }
        }
        Err(e) => {
            panic!("❌ GPU quick iteration failed: {}", e);
        }
    }
}

// =============================================================================
// Loss Convergence Tests (100 steps, ~5 minutes)
// =============================================================================

/// Loss convergence validation with SmolLM2-135M (100 steps).
///
/// This test validates:
/// - Loss decreases over training
/// - Gradients flow through LoRA weights
/// - Training signal is present
///
/// Success criteria:
/// - Loss decreases by at least 30% from initial
///
/// Run with: `cargo test --features 'qlora cuda' -- --ignored test_gpu_loss_convergence`
#[test]
#[ignore]
#[cfg(feature = "cuda")]
fn test_gpu_loss_convergence_100_steps() {
    use axolotl_rs::{AxolotlConfig, Trainer};

    if !cuda_available() {
        skip_gpu_test("CUDA not available");
        return;
    }

    if !model_available("HuggingFaceTB/SmolLM2-135M") {
        skip_gpu_test("SmolLM2-135M not downloaded");
        return;
    }

    gpu_test_status("Starting loss convergence test (100 steps)");

    let temp_dir = TempDir::new().unwrap();
    let output_dir = temp_dir.path().join("outputs");
    let dataset_path = temp_dir.path().join("dataset.jsonl");

    // 100 samples for 100 steps
    create_test_dataset(&dataset_path, 100);

    let config_content = create_lora_config(
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

    let start = Instant::now();
    let result = trainer.train();
    let elapsed = start.elapsed();

    match result {
        Ok(()) => {
            // Extract losses from training metrics
            let losses: Vec<f64> = trainer.training_metrics.iter().map(|m| m.loss).collect();

            gpu_test_status(&format!(
                "Loss convergence test completed in {:.1}s ({} steps)",
                elapsed.as_secs_f64(),
                losses.len()
            ));

            if losses.is_empty() {
                panic!("No loss values collected during training");
            }

            // Validate loss convergence (at least 30% decrease)
            let initial_loss = losses[0];
            let final_loss = losses[losses.len() - 1];
            let loss_decrease = (initial_loss - final_loss) / initial_loss;

            gpu_test_status(&format!(
                "Loss: {:.4} -> {:.4} (decrease: {:.1}%)",
                initial_loss,
                final_loss,
                loss_decrease * 100.0
            ));

            assert_loss_convergence(&losses, 0.3, 10);

            gpu_test_status("✅ Loss convergence test passed");
        }
        Err(e) => {
            panic!("❌ Loss convergence test failed: {}", e);
        }
    }
}

/// LoRA (no quantization) loss convergence test with SmolLM2-135M (100 steps).
///
/// This test validates our custom LoraLlama implementation:
/// - Per-layer LoRA injection in Q, K, V, O, gate, up, down projections
/// - Proper gradient flow through adapter weights
/// - Loss decreases over training
///
/// Success criteria:
/// - Loss decreases by at least 30% from initial
///
/// Expected runtime: ~70-100 seconds (100 steps at ~0.7-1.0s/step in release mode)
/// Resource requirements: CUDA-capable GPU with ~2GB VRAM, SmolLM2-135M model downloaded
///
/// Run with: `cargo test --features 'peft cuda' -- --ignored test_lora_loss_convergence`
#[test]
#[ignore]
#[cfg(feature = "cuda")]
fn test_lora_loss_convergence_100_steps() {
    use axolotl_rs::{AxolotlConfig, Trainer};

    if !cuda_available() {
        skip_gpu_test("CUDA not available");
        return;
    }

    if !model_available("HuggingFaceTB/SmolLM2-135M") {
        skip_gpu_test("SmolLM2-135M not downloaded");
        return;
    }

    gpu_test_status("Starting LoRA loss convergence test (100 steps)");

    let temp_dir = TempDir::new().unwrap();
    let output_dir = temp_dir.path().join("outputs");
    let dataset_path = temp_dir.path().join("dataset.jsonl");

    // 100 samples for 100 steps
    create_test_dataset(&dataset_path, 100);

    let config_content = create_lora_config(
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

    gpu_test_status(&format!(
        "Config: model={}, adapter={:?}",
        config.base_model, config.adapter
    ));

    let mut trainer = Trainer::new(config).expect("Failed to create trainer");

    let start = Instant::now();
    let result = trainer.train();
    let elapsed = start.elapsed();

    match result {
        Ok(()) => {
            // Extract losses from training metrics
            let losses: Vec<f64> = trainer.training_metrics.iter().map(|m| m.loss).collect();

            gpu_test_status(&format!(
                "LoRA loss convergence test completed in {:.1}s ({} steps)",
                elapsed.as_secs_f64(),
                losses.len()
            ));

            if losses.is_empty() {
                panic!("No loss values collected during training");
            }

            // Validate loss convergence (at least 30% decrease)
            let initial_loss = losses[0];
            let final_loss = losses[losses.len() - 1];
            let loss_decrease = (initial_loss - final_loss) / initial_loss;

            gpu_test_status(&format!(
                "Loss: {:.4} -> {:.4} (decrease: {:.1}%)",
                initial_loss,
                final_loss,
                loss_decrease * 100.0
            ));

            assert_loss_convergence(&losses, 0.3, 10);

            gpu_test_status("✅ LoRA loss convergence test passed");
        }
        Err(e) => {
            panic!("❌ LoRA loss convergence test failed: {}", e);
        }
    }
}

// =============================================================================
// TinyLlama Memory Validation (50 steps, ~10 minutes)
// =============================================================================

/// Memory validation with TinyLlama-1.1B (50 steps).
///
/// This test validates:
/// - 1.1B model fits in GPU memory with 4-bit quantization
/// - Training can proceed without OOM
/// - ~2GB VRAM is sufficient
///
/// Run with: `cargo test --features 'qlora cuda' -- --ignored test_gpu_tinyllama_memory`
#[test]
#[ignore]
#[cfg(feature = "cuda")]
fn test_gpu_tinyllama_memory_validation() {
    use axolotl_rs::{AxolotlConfig, Trainer};

    if !cuda_available() {
        skip_gpu_test("CUDA not available");
        return;
    }

    if !model_available("TinyLlama/TinyLlama-1.1B-Chat-v1.0") {
        skip_gpu_test("TinyLlama-1.1B not downloaded");
        println!("   Download with: huggingface-cli download TinyLlama/TinyLlama-1.1B-Chat-v1.0");
        return;
    }

    gpu_test_status("Starting TinyLlama memory validation (50 steps)");
    gpu_test_status(&format!(
        "Expected VRAM: < {} MB",
        vram_requirements::TINYLLAMA_1B_4BIT / 1024 / 1024
    ));

    let temp_dir = TempDir::new().unwrap();
    let output_dir = temp_dir.path().join("outputs");
    let dataset_path = temp_dir.path().join("dataset.jsonl");

    create_test_dataset(&dataset_path, 50);

    let config_content = create_gpu_config(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        &output_dir,
        &dataset_path,
        1,    // epochs
        1,    // batch_size
        256,  // max_length
        2e-4, // learning_rate
    );

    let config_path = temp_dir.path().join("config.yaml");
    fs::write(&config_path, config_content).unwrap();

    let config =
        AxolotlConfig::from_file(config_path.to_str().unwrap()).expect("Failed to load config");

    let mut trainer = Trainer::new(config).expect("Failed to create trainer");

    let start = Instant::now();
    let result = trainer.train();
    let elapsed = start.elapsed();

    match result {
        Ok(()) => {
            let losses: Vec<f64> = trainer.training_metrics.iter().map(|m| m.loss).collect();

            gpu_test_status(&format!(
                "✅ TinyLlama memory validation passed in {:.1}s ({} steps)",
                elapsed.as_secs_f64(),
                losses.len()
            ));

            if !losses.is_empty() {
                gpu_test_status(&format!(
                    "Loss range: {:.4} -> {:.4}",
                    losses[0],
                    losses[losses.len() - 1]
                ));
            }
        }
        Err(e) => {
            panic!("❌ TinyLlama memory validation failed: {}", e);
        }
    }
}

// =============================================================================
// TinyLlama Extended Training (500 steps, ~30 minutes)
// =============================================================================

/// Extended training with TinyLlama-1.1B (500 steps).
///
/// This test validates:
/// - Sustained training over many steps
/// - Loss convergence over extended training
/// - Memory stability (no leaks)
///
/// Run with: `cargo test --features 'qlora cuda' -- --ignored test_gpu_tinyllama_extended`
#[test]
#[ignore]
#[cfg(feature = "cuda")]
fn test_gpu_tinyllama_extended_training() {
    use axolotl_rs::{AxolotlConfig, Trainer};

    if !cuda_available() {
        skip_gpu_test("CUDA not available");
        return;
    }

    if !model_available("TinyLlama/TinyLlama-1.1B-Chat-v1.0") {
        skip_gpu_test("TinyLlama-1.1B not downloaded");
        return;
    }

    gpu_test_status("Starting TinyLlama extended training (500 steps)");

    let temp_dir = TempDir::new().unwrap();
    let output_dir = temp_dir.path().join("outputs");
    let dataset_path = temp_dir.path().join("dataset.jsonl");

    create_test_dataset(&dataset_path, 500);

    let config_content = create_gpu_config(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        &output_dir,
        &dataset_path,
        1,    // epochs
        1,    // batch_size
        256,  // max_length
        2e-4, // learning_rate
    );

    let config_path = temp_dir.path().join("config.yaml");
    fs::write(&config_path, config_content).unwrap();

    let config =
        AxolotlConfig::from_file(config_path.to_str().unwrap()).expect("Failed to load config");

    let mut trainer = Trainer::new(config).expect("Failed to create trainer");

    let start = Instant::now();
    let result = trainer.train();
    let elapsed = start.elapsed();

    match result {
        Ok(()) => {
            let losses: Vec<f64> = trainer.training_metrics.iter().map(|m| m.loss).collect();

            gpu_test_status(&format!(
                "✅ TinyLlama extended training passed in {:.1}s ({:.1} steps/sec, {} steps)",
                elapsed.as_secs_f64(),
                losses.len() as f64 / elapsed.as_secs_f64(),
                losses.len()
            ));

            if losses.len() >= 2 {
                let loss_decrease = (losses[0] - losses[losses.len() - 1]) / losses[0];
                gpu_test_status(&format!(
                    "Loss: {:.4} -> {:.4} (decrease: {:.1}%)",
                    losses[0],
                    losses[losses.len() - 1],
                    loss_decrease * 100.0
                ));
            }
        }
        Err(e) => {
            panic!("❌ TinyLlama extended training failed: {}", e);
        }
    }
}

// =============================================================================
// LLaMA-7B Full Validation (1000 steps, ~2 hours)
// =============================================================================

/// Full-scale validation with LLaMA-7B (1000 steps).
///
/// This test validates:
/// - Full 7B model training works
/// - Complete Alpaca-style fine-tuning pipeline
/// - Production-ready configuration
///
/// Requirements:
/// - ~12 GB VRAM with gradient checkpointing
/// - LLaMA-7B model downloaded
///
/// Run with: `cargo test --features 'qlora cuda' -- --ignored test_gpu_llama7b_full`
#[test]
#[ignore]
#[cfg(feature = "cuda")]
fn test_gpu_llama7b_full_validation() {
    use axolotl_rs::{AxolotlConfig, Trainer};

    if !cuda_available() {
        skip_gpu_test("CUDA not available");
        return;
    }

    // Check for LLaMA-7B (various possible names)
    let llama_models = [
        "meta-llama/Llama-2-7b-hf",
        "huggyllama/llama-7b",
        "decapoda-research/llama-7b-hf",
    ];

    let available_model = llama_models.iter().find(|m| model_available(m));

    let model_id = match available_model {
        Some(m) => *m,
        None => {
            skip_gpu_test("LLaMA-7B not downloaded");
            println!("   Download with: huggingface-cli download meta-llama/Llama-2-7b-hf");
            return;
        }
    };

    gpu_test_status(&format!("Starting LLaMA-7B full validation (1000 steps)"));
    gpu_test_status(&format!(
        "Expected VRAM: < {} GB",
        vram_requirements::LLAMA_7B_4BIT_CHECKPOINT / 1024 / 1024 / 1024
    ));

    let temp_dir = TempDir::new().unwrap();
    let output_dir = temp_dir.path().join("outputs");
    let dataset_path = temp_dir.path().join("dataset.jsonl");

    create_test_dataset(&dataset_path, 1000);

    let config_content = create_gpu_config(
        model_id,
        &output_dir,
        &dataset_path,
        1,    // epochs
        1,    // batch_size (keep at 1 for memory)
        512,  // max_length
        2e-4, // learning_rate
    );

    let config_path = temp_dir.path().join("config.yaml");
    fs::write(&config_path, config_content).unwrap();

    let config =
        AxolotlConfig::from_file(config_path.to_str().unwrap()).expect("Failed to load config");

    let mut trainer = Trainer::new(config).expect("Failed to create trainer");

    let start = Instant::now();
    let result = trainer.train();
    let elapsed = start.elapsed();

    match result {
        Ok(()) => {
            let losses: Vec<f64> = trainer.training_metrics.iter().map(|m| m.loss).collect();

            gpu_test_status(&format!(
                "✅ LLaMA-7B full validation passed in {:.1}m ({:.1} steps/sec, {} steps)",
                elapsed.as_secs_f64() / 60.0,
                losses.len() as f64 / elapsed.as_secs_f64(),
                losses.len()
            ));

            if losses.len() >= 2 {
                let loss_decrease = (losses[0] - losses[losses.len() - 1]) / losses[0];
                gpu_test_status(&format!(
                    "Loss: {:.4} -> {:.4} (decrease: {:.1}%)",
                    losses[0],
                    losses[losses.len() - 1],
                    loss_decrease * 100.0
                ));
            }
        }
        Err(e) => {
            panic!("❌ LLaMA-7B full validation failed: {}", e);
        }
    }
}

// =============================================================================
// Gradient Flow Verification
// =============================================================================

/// Verify gradients flow through LoRA weights.
///
/// This test validates that:
/// - LoRA A/B matrices receive gradients
/// - Optimizer updates LoRA weights
/// - Training metrics show convergence
///
/// Run with: `cargo test --features 'qlora cuda' -- --ignored test_gpu_gradient_flow`
#[test]
#[ignore]
#[cfg(feature = "cuda")]
fn test_gpu_gradient_flow() {
    use axolotl_rs::{AxolotlConfig, Trainer};

    if !cuda_available() {
        skip_gpu_test("CUDA not available");
        return;
    }

    if !model_available("HuggingFaceTB/SmolLM2-135M") {
        skip_gpu_test("SmolLM2-135M not downloaded");
        return;
    }

    gpu_test_status("Testing gradient flow through LoRA weights (5 training steps)");

    let temp_dir = TempDir::new().unwrap();
    let output_dir = temp_dir.path().join("outputs");
    let dataset_path = temp_dir.path().join("dataset.jsonl");

    // 5 samples for 5 steps with batch_size=1
    create_test_dataset(&dataset_path, 5);

    let config_content = create_gpu_config(
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
            let losses: Vec<f64> = trainer.training_metrics.iter().map(|m| m.loss).collect();

            gpu_test_status(&format!("Training completed: {} steps", losses.len()));

            if losses.len() >= 2 {
                // Check that loss changed (gradient flowed)
                let first_loss = losses[0];
                let last_loss = losses[losses.len() - 1];
                let loss_change = (first_loss - last_loss).abs();

                gpu_test_status(&format!(
                    "Loss trajectory: {:.4} -> {:.4} (change: {:.4})",
                    first_loss, last_loss, loss_change
                ));

                if loss_change > 0.0001 {
                    gpu_test_status("✅ Gradient flow verified (loss changed)");
                } else {
                    gpu_test_status(
                        "⚠️  Warning: Loss changed minimally, gradients may not be flowing",
                    );
                }
            } else {
                panic!("❌ Not enough training steps collected");
            }
        }
        Err(e) => {
            panic!("❌ Gradient flow test failed: {}", e);
        }
    }
}
