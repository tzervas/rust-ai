//! Training Tools for rust-ai
//!
//! This crate provides:
//! - Real-time training monitoring with TUI
//! - HuggingFace Hub integration for model uploads
//! - Checkpoint management with compression and cleanup
//! - Progressive parameter expansion training (100M → 500M → 1B)
//! - GPU memory management and monitoring
//!
//! # Binaries
//!
//! - `train` - Main training orchestrator with progressive expansion
//! - `train-monitor` - Real-time TUI monitor for active training
//! - `hf-upload` - Upload models and checkpoints to HuggingFace Hub

pub mod checkpoint_manager;
pub mod gpu_stats;
pub mod hf;
pub mod live_monitor;
pub mod lr_scheduler;
pub mod memory;
pub mod monitor;
pub mod progressive;
pub mod training_state;

pub use checkpoint_manager::CheckpointManager;
pub use gpu_stats::{query_gpu_stats, GpuStats, GpuStatsMonitor};
pub use hf::HuggingFaceUploader;
pub use lr_scheduler::{LRScheduler, SchedulerError, WSDScheduler, WSDSchedulerBuilder};
pub use memory::{
    find_optimal_params, query_gpu_memory, GpuMemoryInfo, MemoryBudget, MemoryMonitor,
    OptimalTrainingParams,
};
pub use monitor::{TrainingMonitor, ViewMode};
pub use progressive::ProgressiveTrainer;
pub use training_state::{CheckpointEvent, PhaseTransition, TrainingRun, TrainingStatus};
