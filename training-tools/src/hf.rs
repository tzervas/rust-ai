//! HuggingFace Hub integration for model uploads.
//!
//! Provides utilities for:
//! - Creating and managing HuggingFace repositories
//! - Uploading model checkpoints
//! - Generating model cards with proper documentation
//! - Managing LFS files for large tensors

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde::{Deserialize, Serialize};

use crate::training_state::TrainingRun;

/// HuggingFace repository configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HfRepoConfig {
    /// Repository ID (user/repo-name)
    pub repo_id: String,
    /// Repository type (model, dataset, space)
    pub repo_type: String,
    /// Whether the repo is private
    pub private: bool,
    /// License
    pub license: String,
    /// Tags for the model
    pub tags: Vec<String>,
}

impl HfRepoConfig {
    /// Create config for a model repository.
    pub fn model(user: &str, model_name: &str) -> Self {
        Self {
            repo_id: format!("{}/{}", user, model_name),
            repo_type: "model".to_string(),
            private: false,
            license: "mit".to_string(),
            tags: vec![
                "tritter".to_string(),
                "rust".to_string(),
                "bitnet".to_string(),
                "transformer".to_string(),
            ],
        }
    }

    /// Set as private repository.
    pub fn with_private(mut self, private: bool) -> Self {
        self.private = private;
        self
    }

    /// Add additional tags.
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags.extend(tags);
        self
    }
}

/// Model card template data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCardData {
    /// Model name
    pub model_name: String,
    /// Model size (100M, 500M, 1B)
    pub model_size: String,
    /// Number of parameters
    pub num_parameters: usize,
    /// Architecture details
    pub architecture: ArchitectureDetails,
    /// Training details
    pub training: TrainingDetails,
    /// Performance metrics
    pub metrics: Option<PerformanceMetrics>,
    /// License
    pub license: String,
    /// Repository URL
    pub repo_url: String,
}

/// Model architecture details.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchitectureDetails {
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: Option<usize>,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub max_seq_length: usize,
    pub use_bitnet: bool,
    pub use_qk_norm: bool,
    pub gradient_checkpointing: bool,
}

/// Training details.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingDetails {
    pub framework: String,
    pub learning_rate: f32,
    pub batch_size: usize,
    pub total_steps: u64,
    pub warmup_steps: u64,
    pub hybrid_training: bool,
    pub backward_reduction: f64,
    pub training_time_hours: f64,
}

/// Performance metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub final_loss: f32,
    pub best_loss: f32,
    pub tokens_per_second: f64,
    pub memory_usage_gb: f64,
}

impl ModelCardData {
    /// Generate the model card markdown content.
    pub fn generate_markdown(&self) -> String {
        let mut md = String::new();

        // YAML frontmatter
        md.push_str("---\n");
        md.push_str(&format!("license: {}\n", self.license));
        md.push_str("language:\n- en\n");
        md.push_str("tags:\n");
        md.push_str("- tritter\n");
        md.push_str("- rust\n");
        md.push_str("- transformer\n");
        if self.architecture.use_bitnet {
            md.push_str("- bitnet\n");
            md.push_str("- 1.58-bit\n");
        }
        md.push_str("library_name: candle\n");
        md.push_str("---\n\n");

        // Title
        md.push_str(&format!("# {}\n\n", self.model_name));

        // Model description
        md.push_str(&format!(
            "A {} parameter Tritter transformer model trained with hybrid predictive training.\n\n",
            self.model_size
        ));

        // Key features
        md.push_str("## Key Features\n\n");
        md.push_str(&format!(
            "- **Parameters**: {:.1}M\n",
            self.num_parameters as f64 / 1e6
        ));
        md.push_str(&format!(
            "- **Architecture**: {} layers, {} heads, {} hidden dim\n",
            self.architecture.num_layers,
            self.architecture.num_heads,
            self.architecture.hidden_size
        ));
        if self.architecture.use_bitnet {
            md.push_str("- **Quantization**: BitNet 1.58-bit ternary weights\n");
        }
        md.push_str(&format!(
            "- **Context Length**: {} tokens\n",
            self.architecture.max_seq_length
        ));
        if self.training.hybrid_training {
            md.push_str(&format!(
                "- **Training Efficiency**: {:.1}% backward reduction via hybrid prediction\n",
                self.training.backward_reduction
            ));
        }
        md.push_str("\n");

        // Architecture
        md.push_str("## Architecture\n\n");
        md.push_str("| Parameter | Value |\n");
        md.push_str("|-----------|-------|\n");
        md.push_str(&format!(
            "| Hidden Size | {} |\n",
            self.architecture.hidden_size
        ));
        md.push_str(&format!("| Layers | {} |\n", self.architecture.num_layers));
        md.push_str(&format!(
            "| Attention Heads | {} |\n",
            self.architecture.num_heads
        ));
        if let Some(kv) = self.architecture.num_kv_heads {
            md.push_str(&format!("| KV Heads | {} |\n", kv));
        }
        md.push_str(&format!(
            "| FFN Size | {} |\n",
            self.architecture.intermediate_size
        ));
        md.push_str(&format!(
            "| Vocab Size | {} |\n",
            self.architecture.vocab_size
        ));
        md.push_str(&format!(
            "| Max Sequence | {} |\n",
            self.architecture.max_seq_length
        ));
        md.push_str(&format!(
            "| QK-Norm | {} |\n",
            if self.architecture.use_qk_norm {
                "Yes"
            } else {
                "No"
            }
        ));
        md.push_str(&format!(
            "| BitNet | {} |\n",
            if self.architecture.use_bitnet {
                "Yes (1.58-bit)"
            } else {
                "No"
            }
        ));
        md.push_str("\n");

        // Training
        md.push_str("## Training\n\n");
        md.push_str(&format!(
            "This model was trained using `{}` with the following configuration:\n\n",
            self.training.framework
        ));
        md.push_str(&format!(
            "- **Learning Rate**: {:.2e}\n",
            self.training.learning_rate
        ));
        md.push_str(&format!("- **Batch Size**: {}\n", self.training.batch_size));
        md.push_str(&format!(
            "- **Total Steps**: {}\n",
            self.training.total_steps
        ));
        md.push_str(&format!(
            "- **Warmup Steps**: {}\n",
            self.training.warmup_steps
        ));
        if self.training.hybrid_training {
            md.push_str(&format!(
                "- **Hybrid Training**: Yes ({:.1}% backward reduction)\n",
                self.training.backward_reduction
            ));
        }
        md.push_str(&format!(
            "- **Training Time**: {:.1} hours\n",
            self.training.training_time_hours
        ));
        md.push_str("\n");

        // Metrics
        if let Some(ref metrics) = self.metrics {
            md.push_str("## Performance\n\n");
            md.push_str(&format!("- **Final Loss**: {:.4}\n", metrics.final_loss));
            md.push_str(&format!("- **Best Loss**: {:.4}\n", metrics.best_loss));
            md.push_str(&format!(
                "- **Throughput**: {:.0} tokens/sec\n",
                metrics.tokens_per_second
            ));
            md.push_str(&format!(
                "- **Memory**: {:.1} GB\n",
                metrics.memory_usage_gb
            ));
            md.push_str("\n");
        }

        // Usage
        md.push_str("## Usage\n\n");
        md.push_str("### Rust (recommended)\n\n");
        md.push_str("```rust\n");
        md.push_str("use tritter_model_rs::{TritterConfig, TritterModel};\n");
        md.push_str("use candle_core::Device;\n\n");
        md.push_str("// Load model\n");
        md.push_str(&format!(
            "let config = TritterConfig::{}();\n",
            match self.model_size.as_str() {
                "100M" => "small_100m",
                "500M" => "medium_500m",
                "1B" => "large_1b",
                _ => "small_100m",
            }
        ));
        md.push_str("let device = Device::Cpu; // or Device::new_cuda(0)?\n");
        md.push_str("let model = TritterModel::load(&config, \"model.safetensors\", &device)?;\n");
        md.push_str("```\n\n");

        // Limitations
        md.push_str("## Limitations\n\n");
        md.push_str("- This is a base model trained on synthetic data for demonstration\n");
        md.push_str("- Not suitable for production use without further training\n");
        md.push_str("- Primarily serves as a checkpoint for progressive parameter expansion\n\n");

        // License
        md.push_str("## License\n\n");
        md.push_str(&format!(
            "This model is released under the {} license.\n\n",
            self.license.to_uppercase()
        ));

        // Citation
        md.push_str("## Citation\n\n");
        md.push_str("```bibtex\n");
        md.push_str("@misc{tritter-rust-ai,\n");
        md.push_str("  title={Tritter: Rust AI Training Stack},\n");
        md.push_str("  author={Tyler Zervas},\n");
        md.push_str("  year={2025},\n");
        md.push_str(&format!("  url={{{}}}\n", self.repo_url));
        md.push_str("}\n");
        md.push_str("```\n");

        md
    }
}

/// HuggingFace Hub uploader.
pub struct HuggingFaceUploader {
    /// HuggingFace username
    username: String,
    /// Cache directory for temporary files
    cache_dir: PathBuf,
}

impl HuggingFaceUploader {
    /// Create a new uploader.
    pub fn new(username: &str, cache_dir: PathBuf) -> Self {
        Self {
            username: username.to_string(),
            cache_dir,
        }
    }

    /// Check if huggingface-cli is installed and authenticated.
    /// Check if huggingface-cli is available and authenticated.
    /// Returns Ok(false) if CLI is not installed or not authenticated.
    pub fn check_auth(&self) -> anyhow::Result<bool> {
        match Command::new("huggingface-cli").args(["whoami"]).output() {
            Ok(output) => Ok(output.status.success()),
            Err(e) => {
                if e.kind() == std::io::ErrorKind::NotFound {
                    tracing::warn!(
                        "huggingface-cli not found - install with: pip install huggingface_hub"
                    );
                    Ok(false)
                } else {
                    Err(e.into())
                }
            }
        }
    }

    /// Create a new repository on HuggingFace Hub.
    pub fn create_repo(&self, config: &HfRepoConfig) -> anyhow::Result<()> {
        let mut args = vec![
            "repo".to_string(),
            "create".to_string(),
            config.repo_id.clone(),
            "--type".to_string(),
            config.repo_type.clone(),
        ];

        if config.private {
            args.push("--private".to_string());
        }

        let output = Command::new("huggingface-cli").args(&args).output()?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            // Ignore "already exists" errors
            if !stderr.contains("already exists") {
                anyhow::bail!("Failed to create repo: {}", stderr);
            }
        }

        Ok(())
    }

    /// Upload a file to the repository.
    pub fn upload_file(
        &self,
        repo_id: &str,
        local_path: &Path,
        remote_path: &str,
        commit_message: &str,
    ) -> anyhow::Result<()> {
        let output = Command::new("huggingface-cli")
            .args([
                "upload",
                repo_id,
                local_path.to_str().unwrap(),
                remote_path,
                "--commit-message",
                commit_message,
            ])
            .output()?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("Failed to upload file: {}", stderr);
        }

        Ok(())
    }

    /// Upload a directory to the repository.
    pub fn upload_directory(
        &self,
        repo_id: &str,
        local_dir: &Path,
        commit_message: &str,
    ) -> anyhow::Result<()> {
        let output = Command::new("huggingface-cli")
            .args([
                "upload",
                repo_id,
                local_dir.to_str().unwrap(),
                ".",
                "--commit-message",
                commit_message,
            ])
            .output()?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("Failed to upload directory: {}", stderr);
        }

        Ok(())
    }

    /// Upload a model checkpoint to HuggingFace.
    pub fn upload_checkpoint(
        &self,
        run: &TrainingRun,
        checkpoint_path: &Path,
        model_card: &ModelCardData,
    ) -> anyhow::Result<String> {
        let repo_name = format!(
            "tritter-{}-step{}",
            run.config.model_size.to_lowercase(),
            run.current_step
        );
        let repo_id = format!("{}/{}", self.username, repo_name);

        // Create repository
        let config = HfRepoConfig::model(&self.username, &repo_name);
        self.create_repo(&config)?;

        // Create staging directory
        let staging_dir = self.cache_dir.join(&repo_name);
        fs::create_dir_all(&staging_dir)?;

        // Copy checkpoint
        let model_file = staging_dir.join("model.safetensors");
        fs::copy(checkpoint_path, &model_file)?;

        // Write model card
        let readme_path = staging_dir.join("README.md");
        fs::write(&readme_path, model_card.generate_markdown())?;

        // Write config
        let config_path = staging_dir.join("config.json");
        let config_json = serde_json::json!({
            "model_type": "tritter",
            "hidden_size": run.config.hidden_size,
            "num_layers": run.config.num_layers,
            "num_heads": run.config.num_heads,
            "vocab_size": 65536,
            "max_seq_length": run.config.max_seq_length,
            "use_bitnet": true,
        });
        fs::write(&config_path, serde_json::to_string_pretty(&config_json)?)?;

        // Upload
        self.upload_directory(
            &repo_id,
            &staging_dir,
            &format!("Upload checkpoint at step {}", run.current_step),
        )?;

        // Cleanup staging
        fs::remove_dir_all(&staging_dir)?;

        Ok(repo_id)
    }

    /// Upload final trained model.
    pub fn upload_final_model(
        &self,
        run: &TrainingRun,
        model_path: &Path,
        model_card: &ModelCardData,
    ) -> anyhow::Result<String> {
        let repo_name = format!("tritter-{}", run.config.model_size.to_lowercase());
        let repo_id = format!("{}/{}", self.username, repo_name);

        // Create repository
        let config = HfRepoConfig::model(&self.username, &repo_name);
        self.create_repo(&config)?;

        // Create staging directory
        let staging_dir = self.cache_dir.join(&repo_name);
        fs::create_dir_all(&staging_dir)?;

        // Copy model
        let model_file = staging_dir.join("model.safetensors");
        fs::copy(model_path, &model_file)?;

        // Write model card
        let readme_path = staging_dir.join("README.md");
        fs::write(&readme_path, model_card.generate_markdown())?;

        // Write config
        let config_path = staging_dir.join("config.json");
        let config_json = serde_json::json!({
            "model_type": "tritter",
            "hidden_size": run.config.hidden_size,
            "num_layers": run.config.num_layers,
            "num_heads": run.config.num_heads,
            "vocab_size": 65536,
            "max_seq_length": run.config.max_seq_length,
            "use_bitnet": true,
        });
        fs::write(&config_path, serde_json::to_string_pretty(&config_json)?)?;

        // Upload
        self.upload_directory(&repo_id, &staging_dir, "Upload final trained model")?;

        // Cleanup staging
        fs::remove_dir_all(&staging_dir)?;

        Ok(repo_id)
    }
}
