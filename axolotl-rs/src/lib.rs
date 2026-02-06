//! # axolotl-rs
//!
//! YAML-driven configurable fine-tuning toolkit for LLMs.
//!
//! This crate provides a user-friendly interface for fine-tuning language models,
//! similar to the Python Axolotl project but in pure Rust.
//!
//! ## Features
//!
//! - **YAML Configuration** - Define entire training runs in simple config files
//! - **Multiple Adapters** - Support for `LoRA`, `QLoRA`, full fine-tuning
//! - **Dataset Handling** - Automatic loading and preprocessing
//! - **Multi-GPU** - Distributed training support (planned)
//!
//! ## Quick Start (CLI)
//!
//! ```bash
//! # Validate configuration
//! axolotl validate config.yaml
//!
//! # Start training
//! axolotl train config.yaml
//!
//! # Merge adapters
//! axolotl merge --config config.yaml --output ./merged-model
//! ```
//!
//! ## Quick Start (Library)
//!
//! ```no_run
//! use axolotl_rs::{AxolotlConfig, Trainer};
//!
//! # fn main() -> axolotl_rs::Result<()> {
//! // Load configuration from YAML file
//! let config = AxolotlConfig::from_file("config.yaml")?;
//!
//! // Create trainer and start training
//! let mut trainer = Trainer::new(config)?;
//! trainer.train()?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Using Presets
//!
//! ```rust
//! use axolotl_rs::AxolotlConfig;
//!
//! # fn main() -> axolotl_rs::Result<()> {
//! // Create mutable config from preset
//! let mut config = AxolotlConfig::from_preset("llama2-7b")?;
//!
//! // Customize as needed
//! config.training.epochs = 5;
//! config.training.learning_rate = 1e-4;
//! # Ok(())
//! # }
//! ```
//!
//! ## Building Custom Configurations
//!
//! ```rust
//! use axolotl_rs::{AxolotlConfig, TrainingConfig};
//! use axolotl_rs::config::{AdapterType, LoraSettings, DatasetConfig};
//!
//! let config = AxolotlConfig {
//!     base_model: "meta-llama/Llama-2-7b-hf".to_string(),
//!     adapter: AdapterType::Lora,
//!     lora: LoraSettings {
//!         r: 64,
//!         alpha: 16,
//!         ..Default::default()
//!     },
//!     quantization: None,
//!     dataset: DatasetConfig {
//!         path: "./data/train.jsonl".to_string(),
//!         ..Default::default()
//!     },
//!     training: TrainingConfig {
//!         epochs: 3,
//!         learning_rate: 2e-4,
//!         ..Default::default()
//!     },
//!     output_dir: "./outputs".to_string(),
//!     seed: 42,
//! };
//! ```

#![warn(missing_docs)]
#![warn(clippy::pedantic)]

pub mod adapters;
pub mod cli;
pub mod config;
pub mod dataset;
pub mod error;
#[cfg(feature = "peft")]
pub mod llama_common;
#[cfg(feature = "peft")]
pub mod lora_llama;
pub mod model;
pub mod normalization;
pub mod optimizer;
#[cfg(all(feature = "peft", feature = "qlora"))]
pub mod qlora_llama;
pub mod scheduler;
pub mod trainer;
#[cfg(feature = "vsa-optim")]
pub mod vsa_accel;

// Mock modules for testing without external dependencies
#[cfg(any(
    feature = "mock-peft",
    feature = "mock-qlora",
    feature = "mock-unsloth"
))]
pub mod mocks;

pub use config::{AxolotlConfig, TrainingConfig};
pub use error::{AxolotlError, Result};
pub use trainer::Trainer;

#[cfg(feature = "vsa-optim")]
pub use vsa_accel::{VSAAccelerator, VSAAcceleratorConfig, VSAStats, VSAStepInfo};
