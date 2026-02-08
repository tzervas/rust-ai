//! Rust AI ecosystem integration.
//!
//! This module provides integration with the rust-ai workspace ecosystem
//! via the `GpuDispatchable` trait from `rust-ai-core`.

#[cfg(feature = "ecosystem")]
use rust_ai_core::traits::GpuDispatchable;
#[cfg(feature = "ecosystem")]
use rust_ai_core::error::{CoreError, Result};
#[cfg(feature = "ecosystem")]
use candle_core::{Device, Tensor};

use crate::error::{HybridTrainingError};
use crate::state::TrainingState;

/// GPU operation dispatcher for hybrid training.
///
/// This struct implements the `GpuDispatchable` trait from `rust-ai-core`,
/// enabling compatibility with the rust-ai workspace ecosystem.
///
/// # Example
///
/// ```rust,ignore
/// use hybrid_predict_trainer_rs::ecosystem::HybridDispatcher;
/// use rust_ai_core::traits::GpuDispatchable;
/// use candle_core::Device;
///
/// let dispatcher = HybridDispatcher::new();
/// let device = Device::cuda_if_available(0)?;
/// let output = dispatcher.dispatch(&input_state, &device)?;
/// ```
#[derive(Debug, Clone)]
pub struct HybridDispatcher {
    /// Whether GPU acceleration is enabled.
    gpu_enabled: bool,
}

impl HybridDispatcher {
    /// Create a new dispatcher with GPU acceleration enabled.
    #[must_use]
    pub fn new() -> Self {
        Self { gpu_enabled: true }
    }

    /// Create a new dispatcher with GPU acceleration disabled.
    #[must_use]
    pub fn cpu_only() -> Self {
        Self { gpu_enabled: false }
    }
}

impl Default for HybridDispatcher {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "ecosystem")]
impl GpuDispatchable for HybridDispatcher {
    /// Input: Training state features as tensor.
    type Input = Tensor;

    /// Output: Encoded state tensor.
    type Output = Tensor;

    fn dispatch_gpu(&self, input: &Self::Input, device: &Device) -> Result<Self::Output> {
        // Validate device is CUDA
        if !matches!(device, Device::Cuda(_)) {
            return Err(CoreError::KernelError {
                message: "dispatch_gpu called with non-CUDA device".to_string(),
            });
        }

        // TODO: Launch CubeCL kernels for state encoding
        // For now, use Candle operations as fallback
        tracing::warn!(
            "GPU dispatch called but CubeCL integration not complete. Using Candle fallback."
        );

        // Simple linear transformation as placeholder
        // Real implementation will use Flash Attention kernel
        input.to_device(device).map_err(|e| CoreError::KernelError {
            message: format!("Device transfer failed: {}", e),
        })
    }

    fn dispatch_cpu(&self, input: &Self::Input, device: &Device) -> Result<Self::Output> {
        // Validate device is CPU
        if !matches!(device, Device::Cpu) {
            return Err(CoreError::KernelError {
                message: "dispatch_cpu called with non-CPU device".to_string(),
            });
        }

        tracing::debug!("Using CPU fallback for state encoding");

        // Transfer to CPU and process
        input.to_device(device).map_err(|e| CoreError::KernelError {
            message: format!("Device transfer failed: {}", e),
        })
    }
}

/// Convert HybridTrainingError to rust-ai-core CoreError.
#[cfg(feature = "ecosystem")]
impl From<HybridTrainingError> for CoreError {
    fn from(err: HybridTrainingError) -> Self {
        match err {
            #[cfg(feature = "cuda")]
            HybridTrainingError::GpuError { detail } => CoreError::KernelError {
                message: detail,
            },
            HybridTrainingError::ConfigError { detail } => CoreError::InvalidConfig(detail),
            HybridTrainingError::MemoryError { detail } => CoreError::OutOfMemory {
                message: detail,
            },
            HybridTrainingError::StateEncodingError { detail } => CoreError::KernelError {
                message: detail,
            },
            _ => CoreError::KernelError {
                message: format!("Hybrid training error: {}", err),
            },
        }
    }
}

/// Convert rust-ai-core CoreError to HybridTrainingError.
#[cfg(feature = "ecosystem")]
impl From<CoreError> for HybridTrainingError {
    fn from(err: CoreError) -> Self {
        // CoreError is an opaque type, convert to string
        HybridTrainingError::IntegrationError {
            crate_name: "rust-ai-core".to_string(),
            detail: format!("{}", err),
        }
    }
}

/// Helper to convert TrainingState to Tensor for ecosystem integration.
impl TrainingState {
    /// Convert training state to Candle tensor for ecosystem integration.
    ///
    /// # Errors
    ///
    /// Returns error if tensor creation fails.
    #[cfg(feature = "ecosystem")]
    pub fn to_tensor(&self, device: &Device) -> Result<Tensor> {
        use candle_core::Tensor as CandleTensor;

        // Extract features as f32 vector
        let features = self.compute_features();

        // Create tensor from features
        CandleTensor::new(&features[..], device).map_err(|e| CoreError::KernelError {
            message: format!("Tensor creation failed: {}", e),
        })
    }

    /// Create training state from Candle tensor.
    ///
    /// # Errors
    ///
    /// Returns error if tensor conversion fails.
    #[cfg(feature = "ecosystem")]
    pub fn from_tensor(_tensor: &Tensor) -> Result<Self> {
        // TODO: Implement proper deserialization from tensor
        // For now, return a dummy state
        Err(CoreError::NotImplemented {
            feature: "TrainingState::from_tensor".to_string(),
        })
    }
}

#[cfg(test)]
#[cfg(feature = "ecosystem")]
mod tests {
    use super::*;

    #[test]
    fn test_hybrid_dispatcher_creation() {
        let dispatcher = HybridDispatcher::new();
        assert!(dispatcher.gpu_enabled);

        let cpu_dispatcher = HybridDispatcher::cpu_only();
        assert!(!cpu_dispatcher.gpu_enabled);
    }

    #[test]
    fn test_hybrid_dispatcher_default() {
        let dispatcher = HybridDispatcher::default();
        assert!(dispatcher.gpu_enabled);
    }
}
