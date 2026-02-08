// SPDX-License-Identifier: MIT
// Copyright 2026 Tyler Zervas
// Adapted from unsloth-rs Flash Attention kernel

//! Flash Attention kernel for RSSM GRU predictions.
//!
//! This module provides GPU-accelerated attention mechanisms adapted from
//! unsloth-rs for use in the RSSM dynamics model's GRU forward pass.
//!
//! ## Algorithm Overview
//!
//! Flash Attention processes attention in tiles with online softmax:
//!
//! ```text
//! For each Q tile (i = 0..Tr):
//!     Load Q_i into shared memory
//!     Initialize accumulators: O_i = 0, m_i = -∞, l_i = 0
//!
//!     For each KV tile (j = 0..Tc):
//!         Load K_j, V_j into shared memory
//!         S_ij = Q_i @ K_j^T / sqrt(d)
//!
//!         # Online softmax update
//!         m_ij = max(m_i, rowmax(S_ij))
//!         P_ij = exp(S_ij - m_ij)
//!         l_ij = exp(m_i - m_ij) * l_i + rowsum(P_ij)
//!
//!         # Output update with correction
//!         O_i = (l_i * exp(m_i - m_ij) * O_i + P_ij @ V_j) / l_ij
//!
//!         m_i = m_ij
//!         l_i = l_ij
//!
//!     Store O_i to global memory
//! ```
//!
//! ## Source Attribution
//!
//! Kernel implementation extracted from:
//! - **Source:** unsloth-rs/src/kernels/cubecl/kernel.rs
//! - **Author:** Tyler Zervas
//! - **License:** MIT
//! - **Adaptations:** Modified for RSSM dynamics model integration

use crate::error::{HybridResult, HybridTrainingError};

// Burn imports for tensor operations
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{Shape, Tensor};

// CubeCL imports for kernel implementation
use cubecl::prelude::*;
use cubecl_cuda::CudaRuntime;

/// Maximum block size for kernel launches
pub const MAX_BLOCK_SIZE: u32 = 1024;

/// Warp size for NVIDIA GPUs
pub const WARP_SIZE: u32 = 32;

// ============================================================================
// CubeCL Kernel Definition
// ============================================================================

/// Flash Attention forward kernel - Production implementation.
///
/// This kernel computes attention for a single Q row using online softmax.
/// Each block handles one (batch, head, q_row) combination with threads
/// cooperatively processing head_dim elements.
///
/// Memory layout: [batch, heads, seq_len, head_dim] stored contiguously.
#[cube(launch)]
fn flash_attention_tile<F: Float + CubeElement>(
    q: &Array<F>,       // Query [batch * heads * seq_len * head_dim]
    k: &Array<F>,       // Key [batch * heads * seq_len * head_dim]
    v: &Array<F>,       // Value [batch * heads * seq_len * head_dim]
    out: &mut Array<F>, // Output [batch * heads * seq_len * head_dim]
    scale: F,           // 1/sqrt(head_dim)
    // Runtime parameters for dimensions
    seq_len_val: u32,
    head_dim_val: u32,
    block_size_val: u32, // Actual block size being used
) {
    // Thread/block indices
    let batch_head_idx = CUBE_POS_X; // Which (batch, head) pair (u32)
    let q_row_idx = CUBE_POS_Y; // Which Q row within this batch-head (u32)
    let tid = UNIT_POS_X; // Thread within block (u32)
    let tid_usize = tid as usize; // Cast for array indexing

    // Strides for [batch*heads, seq_len, head_dim] layout
    let head_stride = (seq_len_val as usize) * (head_dim_val as usize);

    // Base offset for this batch-head
    let base_offset = (batch_head_idx as usize) * head_stride;

    // Bounds check: threads beyond head_dim don't participate in main computation
    // but still participate in synchronization for correctness
    let is_active = tid_usize < (head_dim_val as usize);

    // Initialize running statistics for online softmax
    let mut running_max = F::new(-1e30); // Running max of attention scores
    let mut running_sum = F::new(0.0); // Running sum for normalization
    let mut running_out = F::new(0.0); // Running output accumulator

    // Get Q value for this thread's position (only if active)
    let q_val = if is_active {
        let q_offset = base_offset + ((q_row_idx as usize) * (head_dim_val as usize) + tid_usize);
        q[q_offset]
    } else {
        F::new(0.0)
    };

    // Shared memory for reduction - sized to block_size (power of 2, max 1024)
    let mut score_tile = SharedMemory::<F>::new(1024usize);

    // Iterate over all K/V positions
    for kv_idx in 0u32..(seq_len_val) {
        let kv_idx_usize = kv_idx as usize;
        // Compute dot product contribution for this thread
        let score_contrib = if is_active {
            let k_offset = base_offset + (kv_idx_usize * (head_dim_val as usize) + tid_usize);
            let k_val = k[k_offset];
            q_val * k_val
        } else {
            F::new(0.0)
        };

        // Store contribution in shared memory
        score_tile[tid_usize] = score_contrib;
        sync_cube();

        // Tree reduction for sum - handles non-power-of-2 head_dim
        // by padding with zeros (inactive threads contribute 0)
        let mut stride = (block_size_val / 2) as usize;
        while stride > 0 {
            if tid_usize < stride {
                // Only add if the partner thread has valid data
                let partner_idx = tid_usize + stride;
                if partner_idx < (block_size_val as usize) {
                    score_tile[tid_usize] = score_tile[tid_usize] + score_tile[partner_idx];
                }
            }
            sync_cube();
            stride = stride / 2;
        }

        // Thread 0 has the full dot product, apply scale
        let score = score_tile[0] * scale;

        // Broadcast score to all threads via shared memory
        if tid == 0 {
            score_tile[0] = score;
        }
        sync_cube();
        let attn_score = score_tile[0];

        // Online softmax update (all threads, even inactive, for synchronization)
        let new_max = F::max(running_max, attn_score);
        let exp_old = F::exp(running_max - new_max);
        let exp_new = F::exp(attn_score - new_max);

        // Update running sum
        let new_sum = exp_old * running_sum + exp_new;

        // Update output: scale old output and add new contribution
        if is_active {
            let v_offset = base_offset + (kv_idx_usize * (head_dim_val as usize) + tid_usize);
            let v_val = v[v_offset];
            running_out = (exp_old * running_sum * running_out + exp_new * v_val) / new_sum;
        }

        // Update statistics
        running_max = new_max;
        running_sum = new_sum;
    }

    // Write output (only active threads)
    if is_active {
        let out_offset = base_offset + ((q_row_idx as usize) * (head_dim_val as usize) + tid_usize);
        out[out_offset] = running_out;
    }
}

/// Flash Attention with causal masking for autoregressive predictions.
///
/// This kernel extends the basic flash attention with upper triangular masking
/// for autoregressive (causal) attention patterns used in RSSM GRU rollouts.
#[cube(launch)]
fn flash_attention_causal<F: Float + CubeElement>(
    q: &Array<F>,
    k: &Array<F>,
    v: &Array<F>,
    out: &mut Array<F>,
    scale: F,
    seq_len_val: u32,
    head_dim_val: u32,
    block_size_val: u32,
) {
    let batch_head_idx = CUBE_POS_X;
    let q_row_idx = CUBE_POS_Y;
    let tid = UNIT_POS_X;
    let tid_usize = tid as usize;

    let head_stride = (seq_len_val as usize) * (head_dim_val as usize);
    let base_offset = (batch_head_idx as usize) * head_stride;
    let is_active = tid_usize < (head_dim_val as usize);

    let mut running_max = F::new(-1e30);
    let mut running_sum = F::new(0.0);
    let mut running_out = F::new(0.0);

    let q_val = if is_active {
        let q_offset = base_offset + ((q_row_idx as usize) * (head_dim_val as usize) + tid_usize);
        q[q_offset]
    } else {
        F::new(0.0)
    };

    let mut score_tile = SharedMemory::<F>::new(1024usize);

    // Causal masking: only attend to positions <= current position
    // kv_idx goes from 0 to q_row_idx (inclusive)
    let max_kv_idx = q_row_idx + 1;

    for kv_idx in 0u32..(max_kv_idx) {
        let kv_idx_usize = kv_idx as usize;
        let score_contrib = if is_active {
            let k_offset = base_offset + (kv_idx_usize * (head_dim_val as usize) + tid_usize);
            let k_val = k[k_offset];
            q_val * k_val
        } else {
            F::new(0.0)
        };

        score_tile[tid_usize] = score_contrib;
        sync_cube();

        let mut stride = (block_size_val / 2) as usize;
        while stride > 0 {
            if tid_usize < stride {
                let partner_idx = tid_usize + stride;
                if partner_idx < (block_size_val as usize) {
                    score_tile[tid_usize] = score_tile[tid_usize] + score_tile[partner_idx];
                }
            }
            sync_cube();
            stride = stride / 2;
        }

        let score = score_tile[0] * scale;

        if tid == 0 {
            score_tile[0] = score;
        }
        sync_cube();
        let attn_score = score_tile[0];

        let new_max = F::max(running_max, attn_score);
        let exp_old = F::exp(running_max - new_max);
        let exp_new = F::exp(attn_score - new_max);
        let new_sum = exp_old * running_sum + exp_new;

        if is_active {
            let v_offset = base_offset + (kv_idx_usize * (head_dim_val as usize) + tid_usize);
            let v_val = v[v_offset];
            running_out = (exp_old * running_sum * running_out + exp_new * v_val) / new_sum;
        }

        running_max = new_max;
        running_sum = new_sum;
    }

    if is_active {
        let out_offset = base_offset + ((q_row_idx as usize) * (head_dim_val as usize) + tid_usize);
        out[out_offset] = running_out;
    }
}

// ============================================================================
// Public API
// ============================================================================

/// Configuration for Flash Attention kernel.
#[derive(Debug, Clone)]
pub struct FlashAttentionConfig {
    /// Whether to apply causal masking (upper triangular mask)
    pub causal: bool,

    /// Tile size for tiled attention (future optimization)
    pub tile_size: Option<usize>,

    /// Device ID (0 for default GPU)
    pub device_id: usize,
}

impl Default for FlashAttentionConfig {
    fn default() -> Self {
        Self {
            causal: false,
            tile_size: None,
            device_id: 0,
        }
    }
}

impl FlashAttentionConfig {
    /// Create a new configuration with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable causal masking for autoregressive attention.
    pub fn with_causal_mask(mut self) -> Self {
        self.causal = true;
        self
    }

    /// Set the device ID for kernel execution.
    pub fn with_device(mut self, device_id: usize) -> Self {
        self.device_id = device_id;
        self
    }
}

/// Launch Flash Attention kernel on GPU.
///
/// # Arguments
///
/// * `_q_data` - Query tensor data [batch, heads, seq_len, head_dim]
/// * `_k_data` - Key tensor data [batch, heads, seq_len, head_dim]
/// * `_v_data` - Value tensor data [batch, heads, seq_len, head_dim]
/// * `_scale` - Attention scale factor (typically 1/sqrt(head_dim))
/// * `_config` - Kernel configuration (causal masking, etc.)
///
/// # Returns
///
/// Output tensor data [batch, heads, seq_len, head_dim] with attention applied
///
/// # Errors
///
/// Returns error if:
/// - Input tensors have incompatible shapes
/// - head_dim exceeds MAX_BLOCK_SIZE (1024)
/// - CUDA runtime initialization fails
///
/// # Note
///
/// This is currently a placeholder. Full CubeCL integration requires:
/// 1. Burn -> CubeCL tensor conversion
/// 2. CubeCL runtime initialization
/// 3. Kernel launch with proper grid/block configuration
/// 4. CubeCL -> Burn tensor conversion
pub fn flash_attention_rssm(
    _q_data: &[f32],
    _k_data: &[f32],
    _v_data: &[f32],
    _scale: f64,
    _config: &FlashAttentionConfig,
    _dims: (usize, usize, usize, usize), // (batch, heads, seq_len, head_dim)
) -> HybridResult<Vec<f32>> {
    // Placeholder implementation
    // Will be replaced with actual CubeCL kernel launch in Phase 2

    tracing::warn!(
        "Flash Attention kernel called but CubeCL integration not yet complete. \
         Returning empty output placeholder."
    );

    Ok(Vec::new())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flash_attention_config_default() {
        let config = FlashAttentionConfig::default();
        assert!(!config.causal);
        assert_eq!(config.device_id, 0);
        assert!(config.tile_size.is_none());
    }

    #[test]
    fn test_flash_attention_config_builder() {
        let config = FlashAttentionConfig::new()
            .with_causal_mask()
            .with_device(1);

        assert!(config.causal);
        assert_eq!(config.device_id, 1);
    }

    // TODO: Add integration tests with actual tensor inputs once
    // CubeCL integration is complete
}
