// SPDX-License-Identifier: MIT
// Copyright 2026 Tyler Zervas

//! GPU dispatch traits and utilities for trit-vsa.
//!
//! This module provides the core abstractions for GPU/CPU kernel dispatch,
//! following a CUDA-first design pattern.

use candle_core::Device;
use std::sync::Once;
use thiserror::Error;

/// Result type alias for GPU operations.
pub type GpuResult<T> = std::result::Result<T, GpuError>;

/// Errors that can occur during GPU operations.
#[derive(Debug, Error)]
pub enum GpuError {
    /// Invalid configuration parameter.
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),

    /// Dimension mismatch between tensors/vectors.
    #[error("dimension mismatch: {0}")]
    DimensionMismatch(String),

    /// GPU kernel execution error.
    #[error("kernel error: {0}")]
    KernelError(String),

    /// Device not available.
    #[error("device not available: {0}")]
    DeviceNotAvailable(String),

    /// Underlying Candle error.
    #[error("candle error: {0}")]
    Candle(#[from] candle_core::Error),
}

impl GpuError {
    /// Create an invalid configuration error.
    pub fn invalid_config(msg: impl Into<String>) -> Self {
        Self::InvalidConfig(msg.into())
    }

    /// Create a dimension mismatch error.
    pub fn dim_mismatch(msg: impl Into<String>) -> Self {
        Self::DimensionMismatch(msg.into())
    }

    /// Create a kernel error.
    pub fn kernel(msg: impl Into<String>) -> Self {
        Self::KernelError(msg.into())
    }

    /// Create a device not available error.
    pub fn device_not_available(msg: impl Into<String>) -> Self {
        Self::DeviceNotAvailable(msg.into())
    }
}

/// GPU/CPU dispatch trait for operations with both implementations.
///
/// This trait enables the CUDA-first pattern: operations that have both
/// GPU (CubeCL) and CPU implementations should implement this trait to
/// automatically route to the appropriate backend.
///
/// # Design Pattern
///
/// ```rust,ignore
/// use trit_vsa::gpu::traits::{GpuDispatchable, warn_if_cpu};
///
/// struct MyOperation;
///
/// impl GpuDispatchable for MyOperation {
///     type Input = (PackedTritVec, PackedTritVec);
///     type Output = PackedTritVec;
///
///     fn dispatch_gpu(&self, input: &Self::Input, device: &Device) -> GpuResult<Self::Output> {
///         // CubeCL kernel implementation
///     }
///
///     fn dispatch_cpu(&self, input: &Self::Input, device: &Device) -> GpuResult<Self::Output> {
///         warn_if_cpu(device, "trit-vsa");
///         // CPU fallback implementation
///     }
/// }
/// ```
pub trait GpuDispatchable: Send + Sync {
    /// Input type for the operation.
    type Input;

    /// Output type for the operation.
    type Output;

    /// Execute operation on GPU using CubeCL kernels.
    ///
    /// # Arguments
    ///
    /// * `input` - Operation input
    /// * `device` - Must be a CUDA device
    ///
    /// # Errors
    ///
    /// Returns `GpuError::KernelError` if kernel execution fails.
    fn dispatch_gpu(&self, input: &Self::Input, device: &Device) -> GpuResult<Self::Output>;

    /// Execute operation on CPU (fallback).
    ///
    /// This should emit a warning via `warn_if_cpu()` before execution.
    ///
    /// # Arguments
    ///
    /// * `input` - Operation input
    /// * `device` - CPU device
    ///
    /// # Errors
    ///
    /// Returns appropriate error if operation fails.
    fn dispatch_cpu(&self, input: &Self::Input, device: &Device) -> GpuResult<Self::Output>;

    /// Automatically dispatch to GPU or CPU based on device.
    ///
    /// This is the primary entry point. It checks the device type and
    /// routes to the appropriate implementation.
    ///
    /// # Arguments
    ///
    /// * `input` - Operation input
    /// * `device` - Target device (CUDA or CPU)
    ///
    /// # Returns
    ///
    /// Operation result from GPU or CPU path.
    ///
    /// # Errors
    ///
    /// Returns error if the operation fails or if Metal device is used (not supported).
    fn dispatch(&self, input: &Self::Input, device: &Device) -> GpuResult<Self::Output> {
        match device {
            Device::Cuda(_) => self.dispatch_gpu(input, device),
            Device::Cpu => self.dispatch_cpu(input, device),
            Device::Metal(_) => Err(GpuError::device_not_available("Metal device not supported")),
        }
    }

    /// Check if GPU dispatch is available for this operation.
    ///
    /// Default implementation checks if CUDA feature is enabled and
    /// a CUDA device is available.
    fn gpu_available(&self) -> bool {
        matches!(Device::cuda_if_available(0), Ok(Device::Cuda(_)))
    }
}

/// Emit a warning when running on CPU instead of GPU.
///
/// This function should be called at the start of any CPU fallback implementation
/// to alert users that they're not getting optimal performance.
///
/// The warning is only emitted once per process to avoid log spam.
///
/// # Arguments
///
/// * `device` - The device being used
/// * `crate_name` - Name of the crate for the warning message
///
/// # Example
///
/// ```rust,ignore
/// fn dispatch_cpu(&self, input: &Self::Input, device: &Device) -> GpuResult<Self::Output> {
///     warn_if_cpu(device, "trit-vsa");
///     // ... perform operation
/// }
/// ```
pub fn warn_if_cpu(device: &Device, crate_name: &str) {
    static WARN_ONCE: Once = Once::new();

    if matches!(device, Device::Cpu) {
        WARN_ONCE.call_once(|| {
            tracing::warn!(
                crate_name = crate_name,
                "Running on CPU - GPU acceleration unavailable. \
                 For optimal performance, use a CUDA-enabled device."
            );
        });
    }
}
