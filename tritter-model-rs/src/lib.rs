//! Pure Rust Tritter Transformer Model with BitNet Quantization
//!
//! This crate implements the Tritter model architecture in Rust, providing:
//! - Transformer with QK-Norm attention
//! - BitNet 1.58-bit ternary weight quantization
//! - Squared ReLU (x * ReLU(x)) activation for ternary stability
//! - Integration with hybrid-predict-trainer-rs for predictive training
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

pub mod attention;
pub mod bitnet;
pub mod config;
pub mod data;
pub mod error;
pub mod layer;
pub mod memory;
pub mod mlp;
pub mod model;
pub mod tokenizer;
pub mod trainer;

pub use bitnet::TritterLinear;
pub use config::{TrainingMemoryEstimate, TritterConfig};
pub use error::{TritterError, TritterResult};
pub use memory::{format_bytes, CheckpointConfig, MemoryTracker};
pub use model::TritterModel;
pub use tokenizer::{ModalityType, MultiModalTokenizer};
pub use trainer::{TritterBatch, TritterOptimizer, TritterTrainer};

/// Prelude for convenient imports
pub mod prelude {
    pub use crate::config::TritterConfig;
    pub use crate::error::{TritterError, TritterResult};
    pub use crate::model::TritterModel;
    pub use crate::tokenizer::{ModalityType, MultiModalTokenizer};
    pub use crate::trainer::{TritterBatch, TritterOptimizer, TritterTrainer};
}
