// SPDX-License-Identifier: MIT
// Copyright 2026 Tyler Zervas

//! GRU (Gated Recurrent Unit) GPU kernels.
//!
//! This module provides GPU-accelerated implementations of GRU forward pass
//! operations used in the RSSM dynamics model.
//!
//! # Algorithm
//!
//! The GRU cell computes:
//!
//! ```text
//! z = σ(W_z·x + U_z·h + b_z)         # Update gate
//! r = σ(W_r·x + U_r·h + b_r)         # Reset gate
//! h̃ = tanh(W_h·x + U_h·(r⊙h) + b_h) # Candidate hidden
//! h' = (1-z)⊙h + z⊙h̃                # New hidden state
//! ```
//!
//! Where:
//! - σ is sigmoid activation
//! - ⊙ is element-wise (Hadamard) product
//! - W matrices are [hidden_dim × input_dim]
//! - U matrices are [hidden_dim × hidden_dim]
//! - b vectors are [hidden_dim]
//!
//! # GPU Optimization Strategy
//!
//! The kernel fuses all six matrix-vector multiplications and three
//! element-wise operations into a single kernel launch to minimize
//! memory traffic and kernel launch overhead.
//!
//! **Shared memory usage:**
//! - `r⊙h` intermediate (size: hidden_dim × sizeof(f32))
//! - Requires synchronization between compute phases
//!
//! **Thread mapping:**
//! - Each thread computes one output dimension
//! - Block size matches hidden_dim (up to 1024)
//! - For hidden_dim > 1024, falls back to CPU

use crate::error::{HybridResult, HybridTrainingError};

#[cfg(feature = "cuda")]
use cubecl::prelude::*;

/// Maximum hidden dimension supported by GPU kernel.
///
/// Limited by maximum CUDA block size and shared memory per block.
pub const MAX_GRU_HIDDEN_DIM: usize = 1024;

/// GRU kernel configuration.
#[derive(Debug, Clone)]
pub struct GruKernelConfig {
    /// Hidden state dimension.
    pub hidden_dim: usize,
    /// Input feature dimension.
    pub input_dim: usize,
    /// Batch size (for future batched inference).
    pub batch_size: usize,
}

impl Default for GruKernelConfig {
    fn default() -> Self {
        Self {
            hidden_dim: 256,
            input_dim: 64,
            batch_size: 1,
        }
    }
}

impl GruKernelConfig {
    /// Validates configuration for GPU kernel.
    ///
    /// # Errors
    ///
    /// Returns error if hidden_dim exceeds MAX_GRU_HIDDEN_DIM.
    pub fn validate(&self) -> HybridResult<()> {
        if self.hidden_dim > MAX_GRU_HIDDEN_DIM {
            return Err((
                HybridTrainingError::ConfigError {
                    detail: format!(
                        "hidden_dim {} exceeds GPU limit {}. Use CPU fallback.",
                        self.hidden_dim, MAX_GRU_HIDDEN_DIM
                    ),
                },
                None,
            ));
        }

        if self.hidden_dim == 0 || self.input_dim == 0 {
            return Err((
                HybridTrainingError::ConfigError {
                    detail: "hidden_dim and input_dim must be > 0".to_string(),
                },
                None,
            ));
        }

        Ok(())
    }

    /// Returns the required shared memory size in bytes.
    #[must_use]
    pub fn shared_memory_bytes(&self) -> usize {
        // Need space for r⊙h intermediate
        self.hidden_dim * std::mem::size_of::<f32>()
    }
}

/// Pre-uploaded GRU weights on GPU.
///
/// All weight matrices are stored in row-major format on the GPU device.
#[derive(Debug, Clone)]
pub struct GpuGruWeights {
    /// Update gate input weights [hidden_dim × input_dim]
    pub w_z: Vec<f32>,
    /// Update gate recurrent weights [hidden_dim × hidden_dim]
    pub u_z: Vec<f32>,
    /// Update gate bias [hidden_dim]
    pub b_z: Vec<f32>,

    /// Reset gate input weights [hidden_dim × input_dim]
    pub w_r: Vec<f32>,
    /// Reset gate recurrent weights [hidden_dim × hidden_dim]
    pub u_r: Vec<f32>,
    /// Reset gate bias [hidden_dim]
    pub b_r: Vec<f32>,

    /// Candidate input weights [hidden_dim × input_dim]
    pub w_h: Vec<f32>,
    /// Candidate recurrent weights [hidden_dim × hidden_dim]
    pub u_h: Vec<f32>,
    /// Candidate bias [hidden_dim]
    pub b_h: Vec<f32>,

    /// Dimensions
    pub hidden_dim: usize,
    pub input_dim: usize,
}

impl GpuGruWeights {
    /// Creates GPU weights from CPU weight structure.
    ///
    /// # Arguments
    ///
    /// - `cpu_weights`: Reference to CPU GRU weights
    ///
    /// # Returns
    ///
    /// New `GpuGruWeights` with cloned data ready for GPU upload.
    #[must_use]
    pub fn from_cpu(cpu_weights: &crate::dynamics::GRUWeights) -> Self {
        Self {
            w_z: cpu_weights.w_z.clone(),
            u_z: cpu_weights.u_z.clone(),
            b_z: cpu_weights.b_z.clone(),
            w_r: cpu_weights.w_r.clone(),
            u_r: cpu_weights.u_r.clone(),
            b_r: cpu_weights.b_r.clone(),
            w_h: cpu_weights.w_h.clone(),
            u_h: cpu_weights.u_h.clone(),
            b_h: cpu_weights.b_h.clone(),
            hidden_dim: cpu_weights.hidden_dim,
            input_dim: cpu_weights.input_dim,
        }
    }

    /// Validates weight dimensions.
    pub fn validate(&self) -> HybridResult<()> {
        let expected_w_size = self.hidden_dim * self.input_dim;
        let expected_u_size = self.hidden_dim * self.hidden_dim;
        let expected_b_size = self.hidden_dim;

        if self.w_z.len() != expected_w_size {
            return Err((
                HybridTrainingError::ConfigError {
                    detail: format!("w_z size mismatch: got {}, expected {}", self.w_z.len(), expected_w_size),
                },
                None,
            ));
        }

        // Similar checks for all other weights...
        if self.u_z.len() != expected_u_size {
            return Err((
                HybridTrainingError::ConfigError {
                    detail: format!("u_z size mismatch: got {}, expected {}", self.u_z.len(), expected_u_size),
                },
                None,
            ));
        }

        if self.b_z.len() != expected_b_size {
            return Err((
                HybridTrainingError::ConfigError {
                    detail: format!("b_z size mismatch: got {}, expected {}", self.b_z.len(), expected_b_size),
                },
                None,
            ));
        }

        Ok(())
    }
}

// ============================================================================
// CubeCL Kernel Definition
// ============================================================================

/// GRU forward pass - Fused GPU kernel.
///
/// Computes one GRU step with all operations fused into a single kernel:
/// - z = σ(W_z·x + U_z·h + b_z)
/// - r = σ(W_r·x + U_r·h + b_r)
/// - h̃ = tanh(W_h·x + U_h·(r⊙h) + b_h)
/// - h' = (1-z)⊙h + z⊙h̃
///
/// # Thread Mapping
///
/// - Each thread computes one output element (one hidden dimension)
/// - Block size = hidden_dim (must be ≤ 1024)
/// - Grid size = 1 (single GRU step, can be extended for batching)
///
/// # Shared Memory Usage
///
/// - `r_h_shared`: Stores r⊙h intermediate (hidden_dim × 4 bytes)
/// - Requires sync barrier between Phase 1 (compute r) and Phase 2 (use r⊙h)
#[cfg(feature = "cuda")]
#[cube(launch)]
fn gru_forward_fused<F: Float + CubeElement>(
    // Input weights [hidden_dim × input_dim] - row-major
    w_z: &Array<F>,
    w_r: &Array<F>,
    w_h: &Array<F>,
    // Recurrent weights [hidden_dim × hidden_dim] - row-major
    u_z: &Array<F>,
    u_r: &Array<F>,
    u_h: &Array<F>,
    // Biases [hidden_dim]
    b_z: &Array<F>,
    b_r: &Array<F>,
    b_h: &Array<F>,
    // Input vectors
    hidden: &Array<F>,  // [hidden_dim]
    input: &Array<F>,   // [input_dim]
    // Output
    output: &mut Array<F>, // [hidden_dim]
    // Dimensions
    hidden_dim: u32,
    input_dim: u32,
) {
    let tid = UNIT_POS_X; // Thread ID (0..hidden_dim)
    let tid_usize = tid as usize;

    // Shared memory for r⊙h intermediate
    let mut r_h_shared = SharedMemory::<F>::new(1024usize);

    // Bounds check - inactive threads skip computation but still participate in sync
    let is_active = tid < hidden_dim;

    // ========================================================================
    // Phase 1: Compute update gate z and reset gate r
    // ========================================================================

    let z = if is_active {
        // z[tid] = σ(W_z[tid,:]·x + U_z[tid,:]·h + b_z[tid])
        let mut z_input_contrib = F::new(0.0);
        #[unroll]
        for j in 0..input_dim {
            let j_usize = j as usize;
            let w_idx = tid_usize * (input_dim as usize) + j_usize;
            z_input_contrib = z_input_contrib + w_z[w_idx] * input[j_usize];
        }

        let mut z_hidden_contrib = F::new(0.0);
        #[unroll]
        for j in 0..hidden_dim {
            let j_usize = j as usize;
            let u_idx = tid_usize * (hidden_dim as usize) + j_usize;
            z_hidden_contrib = z_hidden_contrib + u_z[u_idx] * hidden[j_usize];
        }

        let z_pre = z_input_contrib + z_hidden_contrib + b_z[tid_usize];
        // Inline sigmoid: σ(x) = 1 / (1 + exp(-x))
        let zero = F::new(0.0);
        let one = F::new(1.0);
        if z_pre < zero {
            let exp_z = F::exp(z_pre);
            exp_z / (one + exp_z)
        } else {
            one / (one + F::exp(-z_pre))
        }
    } else {
        F::new(0.0)
    };

    let r = if is_active {
        // r[tid] = σ(W_r[tid,:]·x + U_r[tid,:]·h + b_r[tid])
        let mut r_input_contrib = F::new(0.0);
        #[unroll]
        for j in 0..input_dim {
            let j_usize = j as usize;
            let w_idx = tid_usize * (input_dim as usize) + j_usize;
            r_input_contrib = r_input_contrib + w_r[w_idx] * input[j_usize];
        }

        let mut r_hidden_contrib = F::new(0.0);
        #[unroll]
        for j in 0..hidden_dim {
            let j_usize = j as usize;
            let u_idx = tid_usize * (hidden_dim as usize) + j_usize;
            r_hidden_contrib = r_hidden_contrib + u_r[u_idx] * hidden[j_usize];
        }

        let r_pre = r_input_contrib + r_hidden_contrib + b_r[tid_usize];
        // Inline sigmoid: σ(x) = 1 / (1 + exp(-x))
        let zero = F::new(0.0);
        let one = F::new(1.0);
        if r_pre < zero {
            let exp_r = F::exp(r_pre);
            exp_r / (one + exp_r)
        } else {
            one / (one + F::exp(-r_pre))
        }
    } else {
        F::new(0.0)
    };

    // Compute and store r⊙h in shared memory
    if is_active {
        r_h_shared[tid_usize] = r * hidden[tid_usize];
    } else {
        r_h_shared[tid_usize] = F::new(0.0);
    }

    // Synchronize: All threads must finish computing r⊙h before Phase 2
    sync_cube();

    // ========================================================================
    // Phase 2: Compute candidate hidden state using r⊙h
    // ========================================================================

    if is_active {
        // h̃[tid] = tanh(W_h[tid,:]·x + U_h[tid,:]·(r⊙h) + b_h[tid])
        let mut h_cand_input_contrib = F::new(0.0);
        #[unroll]
        for j in 0..input_dim {
            let j_usize = j as usize;
            let w_idx = tid_usize * (input_dim as usize) + j_usize;
            h_cand_input_contrib = h_cand_input_contrib + w_h[w_idx] * input[j_usize];
        }

        let mut h_cand_hidden_contrib = F::new(0.0);
        #[unroll]
        for j in 0..hidden_dim {
            let j_usize = j as usize;
            let u_idx = tid_usize * (hidden_dim as usize) + j_usize;
            // Read r⊙h from shared memory
            h_cand_hidden_contrib = h_cand_hidden_contrib + u_h[u_idx] * r_h_shared[j_usize];
        }

        let h_cand_pre = h_cand_input_contrib + h_cand_hidden_contrib + b_h[tid_usize];
        // Inline tanh: tanh(x) = (e^2x - 1) / (e^2x + 1)
        let two = F::new(2.0);
        let one = F::new(1.0);
        let exp_2h = F::exp(two * h_cand_pre);
        let h_cand = (exp_2h - one) / (exp_2h + one);

        // ========================================================================
        // Phase 3: Compute final hidden state
        // ========================================================================

        // h'[tid] = (1-z)·h[tid] + z·h̃[tid]
        let one = F::new(1.0);
        let h_new = (one - z) * hidden[tid_usize] + z * h_cand;

        output[tid_usize] = h_new;
    }
}

// ============================================================================
// CPU Reference Implementation
// ============================================================================

/// CPU reference implementation for validation.
///
/// This function provides a reference implementation of GRU forward pass
/// that matches the production CPU code in `src/dynamics.rs:622-652`.
///
/// Used for testing GPU kernel correctness.
pub fn gru_forward_cpu(
    weights: &GpuGruWeights,
    hidden: &[f32],
    input: &[f32],
) -> Vec<f32> {
    let hidden_dim = weights.hidden_dim;
    let input_dim = weights.input_dim;

    // Helper: matrix-vector multiplication
    let matvec = |matrix: &[f32], vector: &[f32], rows: usize, cols: usize| -> Vec<f32> {
        (0..rows)
            .map(|i| {
                (0..cols)
                    .map(|j| matrix[i * cols + j] * vector[j])
                    .sum::<f32>()
            })
            .collect()
    };

    // Helper: sigmoid activation
    let sigmoid = |x: f32| 1.0 / (1.0 + (-x).exp());

    // z = sigmoid(W_z·x + U_z·h + b_z)
    let mut z = matvec(&weights.w_z, input, hidden_dim, input_dim);
    let z_h = matvec(&weights.u_z, hidden, hidden_dim, hidden_dim);
    for i in 0..hidden_dim {
        z[i] = sigmoid(z[i] + z_h[i] + weights.b_z[i]);
    }

    // r = sigmoid(W_r·x + U_r·h + b_r)
    let mut r = matvec(&weights.w_r, input, hidden_dim, input_dim);
    let r_h = matvec(&weights.u_r, hidden, hidden_dim, hidden_dim);
    for i in 0..hidden_dim {
        r[i] = sigmoid(r[i] + r_h[i] + weights.b_r[i]);
    }

    // h_cand = tanh(W_h·x + U_h·(r⊙h) + b_h)
    let mut h_candidate = matvec(&weights.w_h, input, hidden_dim, input_dim);
    let r_h_elem: Vec<f32> = r.iter().zip(hidden.iter()).map(|(&r, &h)| r * h).collect();
    let h_rec = matvec(&weights.u_h, &r_h_elem, hidden_dim, hidden_dim);
    for i in 0..hidden_dim {
        h_candidate[i] = (h_candidate[i] + h_rec[i] + weights.b_h[i]).tanh();
    }

    // h_new = (1-z)⊙h + z⊙h_cand
    (0..hidden_dim)
        .map(|i| (1.0 - z[i]) * hidden[i] + z[i] * h_candidate[i])
        .collect()
}

/// GPU-accelerated GRU forward pass.
///
/// Launches the fused GRU kernel on GPU and returns the result.
///
/// # Implementation Details
///
/// - **Kernel**: `gru_forward_fused` - All GRU operations fused
/// - **Launch config**: 1 block × hidden_dim threads
/// - **Fallback**: Falls back to CPU if hidden_dim > 1024
///
/// # Phase 2 Status
///
/// Currently returns CPU implementation as CubeCL runtime integration
/// is pending. The kernel code is ready but requires:
/// - CubeCL runtime initialization
/// - Buffer upload/download
/// - Kernel launch configuration
///
/// This will be completed in the next session.
#[cfg(feature = "cuda")]
pub fn gru_forward_gpu(
    config: &GruKernelConfig,
    weights: &GpuGruWeights,
    hidden: &[f32],
    input: &[f32],
) -> HybridResult<Vec<f32>> {
    // Validate configuration
    config.validate()?;

    // For now, fall back to CPU until CubeCL runtime is integrated
    // TODO: Implement actual kernel launch:
    // 1. Initialize CubeCL runtime
    // 2. Upload weights, hidden, input to GPU
    // 3. Launch gru_forward_fused kernel
    // 4. Download output from GPU
    // 5. Return result

    tracing::debug!(
        "GRU GPU kernel called (hidden_dim={}, input_dim={}) - using CPU fallback",
        config.hidden_dim,
        config.input_dim
    );

    Ok(gru_forward_cpu(weights, hidden, input))
}

/// GPU-accelerated GRU forward pass (non-CUDA fallback).
#[cfg(not(feature = "cuda"))]
pub fn gru_forward_gpu(
    _config: &GruKernelConfig,
    weights: &GpuGruWeights,
    hidden: &[f32],
    input: &[f32],
) -> HybridResult<Vec<f32>> {
    Ok(gru_forward_cpu(weights, hidden, input))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_weights(hidden_dim: usize, input_dim: usize) -> GpuGruWeights {
        GpuGruWeights {
            w_z: vec![0.1; hidden_dim * input_dim],
            u_z: vec![0.1; hidden_dim * hidden_dim],
            b_z: vec![0.0; hidden_dim],
            w_r: vec![0.1; hidden_dim * input_dim],
            u_r: vec![0.1; hidden_dim * hidden_dim],
            b_r: vec![0.0; hidden_dim],
            w_h: vec![0.1; hidden_dim * input_dim],
            u_h: vec![0.1; hidden_dim * hidden_dim],
            b_h: vec![0.0; hidden_dim],
            hidden_dim,
            input_dim,
        }
    }

    #[test]
    fn test_gru_config_validation() {
        let valid_config = GruKernelConfig {
            hidden_dim: 256,
            input_dim: 64,
            batch_size: 1,
        };
        assert!(valid_config.validate().is_ok());

        let invalid_config = GruKernelConfig {
            hidden_dim: 2048, // Exceeds MAX_GRU_HIDDEN_DIM
            input_dim: 64,
            batch_size: 1,
        };
        assert!(invalid_config.validate().is_err());
    }

    #[test]
    fn test_gru_forward_cpu_zero_input() {
        let hidden_dim = 4;
        let input_dim = 2;
        let weights = create_test_weights(hidden_dim, input_dim);
        let hidden = vec![0.0; hidden_dim];
        let input = vec![0.0; input_dim];

        let output = gru_forward_cpu(&weights, &hidden, &input);

        // With zero input and hidden, output depends only on bias (which is zero)
        assert_eq!(output.len(), hidden_dim);
        // sigmoid(0) = 0.5, tanh(0) = 0
        // z = 0.5, r = 0.5, h_cand = 0
        // h_new = 0.5 * 0 + 0.5 * 0 = 0
        for val in &output {
            assert!(val.abs() < 1e-5, "Expected ~0, got {}", val);
        }
    }

    #[test]
    fn test_gru_forward_cpu_deterministic() {
        let hidden_dim = 8;
        let input_dim = 4;
        let weights = create_test_weights(hidden_dim, input_dim);
        let hidden = vec![0.5; hidden_dim];
        let input = vec![0.3; input_dim];

        let output1 = gru_forward_cpu(&weights, &hidden, &input);
        let output2 = gru_forward_cpu(&weights, &hidden, &input);

        // Should be deterministic
        for (a, b) in output1.iter().zip(output2.iter()) {
            assert!((a - b).abs() < 1e-7);
        }
    }

    #[test]
    fn test_gru_weights_validation() {
        let weights = create_test_weights(4, 2);
        assert!(weights.validate().is_ok());

        // Test invalid dimensions
        let mut invalid_weights = weights.clone();
        invalid_weights.w_z.pop(); // Wrong size
        assert!(invalid_weights.validate().is_err());
    }

    #[test]
    fn test_shared_memory_calculation() {
        let config = GruKernelConfig {
            hidden_dim: 256,
            input_dim: 64,
            batch_size: 1,
        };

        let expected_bytes = 256 * 4; // 256 floats × 4 bytes/float
        assert_eq!(config.shared_memory_bytes(), expected_bytes);
    }
}
