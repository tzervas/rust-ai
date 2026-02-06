//! Integration tests for the axolotl CLI.

use assert_cmd::Command;
use std::fs;
use std::path::PathBuf;
use tempfile::TempDir;

/// Helper function to create a test configuration file.
///
/// # Arguments
/// * `path` - Path where the config file should be created
/// * `content` - YAML content for the configuration file
///
/// # Returns
/// Path to the created configuration file
fn create_test_config(path: &PathBuf, content: &str) -> PathBuf {
    let config_path = path.join("config.yaml");
    fs::write(&config_path, content).expect("Failed to write test config");
    config_path
}

/// Helper function to run the axolotl CLI with given arguments.
///
/// # Arguments
/// * `args` - Command line arguments to pass to the CLI
///
/// # Returns
/// An assert_cmd::Command ready to be executed
fn run_cli(args: &[&str]) -> Command {
    let mut cmd = Command::cargo_bin("axolotl").expect("Failed to find axolotl binary");
    cmd.args(args);
    cmd
}

/// Helper function to create a valid test configuration.
///
/// Returns a minimal valid YAML configuration for testing.
fn valid_config_yaml() -> &'static str {
    r#"
base_model: "meta-llama/Llama-2-7b-hf"
model_type: "llama"
output_dir: "./outputs"

dataset:
  path: "data/train.jsonl"
  type: "alpaca"

training:
  num_epochs: 3
  batch_size: 4
  learning_rate: 0.0002
  gradient_accumulation_steps: 1

adapter: "lora"

lora:
  r: 8
  alpha: 16
  target_modules:
    - "q_proj"
    - "v_proj"
"#
}

/// Helper function to create an invalid test configuration (missing required fields).
fn invalid_config_yaml() -> &'static str {
    r#"
base_model: "meta-llama/Llama-2-7b-hf"
# Missing required fields like dataset and training
"#
}

#[test]
fn test_validate_command_valid_config() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let config_path = create_test_config(&temp_dir.path().to_path_buf(), valid_config_yaml());

    let mut cmd = run_cli(&["validate", config_path.to_str().unwrap()]);

    cmd.assert()
        .success()
        .stdout(predicates::str::contains("Configuration is valid"));
}

#[test]
fn test_validate_command_invalid_config() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let config_path = create_test_config(&temp_dir.path().to_path_buf(), invalid_config_yaml());

    let mut cmd = run_cli(&["validate", config_path.to_str().unwrap()]);

    // Should fail with invalid configuration
    cmd.assert().failure();
}

#[test]
fn test_validate_command_missing_file() {
    let mut cmd = run_cli(&["validate", "/nonexistent/config.yaml"]);

    // Should fail when config file doesn't exist
    cmd.assert().failure();
}

#[test]
fn test_train_command_help() {
    let mut cmd = run_cli(&["train", "--help"]);

    cmd.assert()
        .success()
        .stdout(predicates::str::contains("Start training"))
        .stdout(predicates::str::contains("CONFIG"))
        .stdout(predicates::str::contains("--resume"));
}

#[test]
fn test_merge_command_help() {
    let mut cmd = run_cli(&["merge", "--help"]);

    cmd.assert()
        .success()
        .stdout(predicates::str::contains("Merge adapter weights"))
        .stdout(predicates::str::contains("--config"))
        .stdout(predicates::str::contains("--adapter"))
        .stdout(predicates::str::contains("--output"));
}

#[test]
fn test_init_command_help() {
    let mut cmd = run_cli(&["init", "--help"]);

    cmd.assert()
        .success()
        .stdout(predicates::str::contains("Generate a sample configuration"))
        .stdout(predicates::str::contains("--preset"));
}

#[test]
fn test_init_command_creates_config() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let output_path = temp_dir.path().join("test_config.yaml");

    let mut cmd = run_cli(&[
        "init",
        output_path.to_str().unwrap(),
        "--preset",
        "llama2-7b",
    ]);

    cmd.assert().success();

    // Verify the config file was created
    assert!(output_path.exists(), "Config file should be created");

    // Verify it contains expected content
    let content = fs::read_to_string(&output_path).expect("Failed to read generated config");
    assert!(content.contains("base_model"));
    assert!(content.contains("dataset"));
}

#[test]
fn test_cli_version() {
    let mut cmd = Command::cargo_bin("axolotl").expect("Failed to find axolotl binary");
    cmd.arg("--version");

    cmd.assert()
        .success()
        .stdout(predicates::str::contains("axolotl"));
}

#[test]
fn test_cli_help() {
    let mut cmd = Command::cargo_bin("axolotl").expect("Failed to find axolotl binary");
    cmd.arg("--help");

    cmd.assert()
        .success()
        .stdout(predicates::str::contains("YAML-driven fine-tuning"))
        .stdout(predicates::str::contains("validate"))
        .stdout(predicates::str::contains("train"))
        .stdout(predicates::str::contains("merge"))
        .stdout(predicates::str::contains("init"));
}
