//! Pure Rust Tritter Transformer Model with BitNet Quantization
//!
//! This crate implements the Tritter model architecture in Rust, providing:
//! - Transformer with QK-Norm attention
//! - BitNet 1.58-bit ternary weight quantization
//! - Squared ReLU (x * ReLU(x)) activation for ternary stability
//! - Integration with hybrid-predict-trainer-rs for predictive training
//! - Gradient checkpointing for memory-efficient training of large models
//!
//! # Example
//!
//! ```no_run
//! use tritter_model_rs::{TritterConfig, TritterModel};
//! use candle_core::Device;
//!
//! let config = TritterConfig::small_100m();
//! let device = Device::Cpu;
//! let model = TritterModel::new(&config, &device).unwrap();
//! ```
//!
//! # Gradient Checkpointing
//!
//! For training large models (500M+) on limited GPU memory, enable gradient
//! checkpointing to trade compute for memory:
//!
//! ```no_run
//! use tritter_model_rs::{TritterConfig, TritterModel};
//! use candle_core::Device;
//!
//! let mut config = TritterConfig::medium_500m();
//! config.gradient_checkpointing = true;
//! config.checkpoint_every_n_layers = 4;
//!
//! let device = Device::Cpu;
//! let mut model = TritterModel::new(&config, &device).unwrap();
//!
//! // Memory savings: ~75% of activation memory
//! // Compute overhead: ~33% additional forward passes
//! ```

pub mod attention;
pub mod bitnet;
pub mod checkpoint;
pub mod config;
pub mod data;
pub mod error;
pub mod layer;
pub mod memory;
pub mod mlp;
pub mod model;
pub mod norm;
pub mod tokenizer;
pub mod trainer;

pub use bitnet::TritterLinear;
pub use checkpoint::{CheckpointStore, GradientCheckpointConfig};
pub use config::{TrainingMemoryEstimate, TritterConfig};
pub use error::{TritterError, TritterResult};
pub use memory::{format_bytes, CheckpointConfig, MemoryTracker};
pub use model::TritterModel;
pub use tokenizer::{ModalityType, MultiModalTokenizer};
pub use trainer::{
    create_trainer, create_trainer_with_checkpointing, create_trainer_with_config, TritterBatch,
    TritterModelWrapper, TritterOptimizer, TritterTrainer,
};

/// Prelude for convenient imports
pub mod prelude {
    pub use crate::checkpoint::{CheckpointStore, GradientCheckpointConfig};
    pub use crate::config::TritterConfig;
    pub use crate::error::{TritterError, TritterResult};
    pub use crate::model::TritterModel;
    pub use crate::tokenizer::{ModalityType, MultiModalTokenizer};
    pub use crate::trainer::{
        create_trainer, create_trainer_with_checkpointing, TritterBatch, TritterModelWrapper,
        TritterOptimizer, TritterTrainer,
    };
}
