// SPDX-License-Identifier: MIT
// Copyright 2026 Tyler Zervas

//! RSSM multi-step rollout GPU kernel.
//!
//! This module implements GPU-accelerated multi-step prediction for the
//! RSSM (Recurrent State Space Model) dynamics model used in hybrid training.
//!
//! # Algorithm Overview
//!
//! For each ensemble member (parallelized across blocks):
//! ```text
//! for step in 0..y_steps:
//!     h' = GRU(h, features)           # Deterministic update
//!     s' = sample_stochastic(h')      # Stochastic sampling
//!     combined = concat(h', s')       # State combination
//!     loss = loss_head(combined)      # Loss prediction
//!     features[0] = loss              # Feature evolution
//! ```
//!
//! # Parallelization Strategy
//!
//! - **Grid X (blocks)**: ensemble_size (typically 5)
//! - **Block X (threads)**: hidden_dim (max 1024)
//! - **Sequential loop**: y_steps with sync barriers
//!
//! # Memory Budget
//!
//! Per-block shared memory (~4.4 KB for hidden_dim=256):
//! - GRU r⊙h intermediate: 256 × 4 = 1 KB
//! - Combined state: 512 × 4 = 2 KB
//! - Reduction buffer: 256 × 4 = 1 KB
//! - Features evolution: 64 × 4 = 256 bytes
//! - **Total**: ~4.3 KB (well within 48 KB per-SM limit)

use crate::error::{HybridResult, HybridTrainingError};

/// RSSM rollout kernel configuration.
#[derive(Debug, Clone)]
pub struct RssmRolloutConfig {
    /// Number of ensemble members.
    pub ensemble_size: usize,
    /// Hidden state dimension.
    pub hidden_dim: usize,
    /// Stochastic dimension.
    pub stochastic_dim: usize,
    /// Input feature dimension.
    pub feature_dim: usize,
    /// Number of prediction steps.
    pub y_steps: usize,
}

impl Default for RssmRolloutConfig {
    fn default() -> Self {
        Self {
            ensemble_size: 5,
            hidden_dim: 256,
            stochastic_dim: 256,
            feature_dim: 64,
            y_steps: 50,
        }
    }
}

impl RssmRolloutConfig {
    /// Validates configuration for GPU kernel.
    pub fn validate(&self) -> HybridResult<()> {
        if self.hidden_dim > 1024 {
            return Err((
                HybridTrainingError::ConfigError {
                    detail: format!(
                        "hidden_dim {} exceeds GPU limit 1024",
                        self.hidden_dim
                    ),
                },
                None,
            ));
        }

        if self.ensemble_size == 0 || self.y_steps == 0 {
            return Err((
                HybridTrainingError::ConfigError {
                    detail: "ensemble_size and y_steps must be > 0".to_string(),
                },
                None,
            ));
        }

        Ok(())
    }

    /// Returns combined state dimension (deterministic + stochastic).
    #[must_use]
    pub fn combined_dim(&self) -> usize {
        self.hidden_dim + self.stochastic_dim
    }

    /// Returns shared memory size in bytes.
    #[must_use]
    pub fn shared_memory_bytes(&self) -> usize {
        // r⊙h + combined state + reduction buffer + features
        self.hidden_dim * 4 + self.combined_dim() * 4 + self.hidden_dim * 4 + self.feature_dim * 4
    }
}

/// GPU weights for RSSM ensemble.
#[derive(Debug, Clone)]
pub struct RssmEnsembleWeights {
    /// GRU weights for each ensemble member.
    pub gru_weights: Vec<crate::gpu::kernels::gru::GpuGruWeights>,
    /// Loss head weights [combined_dim] - shared across ensemble.
    pub loss_head_weights: Vec<f32>,
    /// Weight delta head weights [combined_dim × 10] - shared across ensemble.
    pub delta_head_weights: Vec<f32>,
}

/// Result of RSSM rollout for one ensemble member.
#[derive(Debug, Clone)]
pub struct RolloutResult {
    /// Loss trajectory [y_steps + 1] (includes initial loss).
    pub trajectory: Vec<f32>,
    /// Final combined state [combined_dim].
    pub final_combined: Vec<f32>,
    /// Total entropy accumulated during rollout.
    pub total_entropy: f32,
    /// Metrics for research analysis.
    pub metrics: RolloutMetrics,
}

/// Detailed metrics for RSSM rollout analysis.
#[derive(Debug, Clone)]
pub struct RolloutMetrics {
    /// Per-step prediction deltas (|loss[t+1] - loss[t]|).
    pub step_deltas: Vec<f32>,
    /// Per-step hidden state norm (L2).
    pub hidden_norms: Vec<f32>,
    /// Final loss variance estimate.
    pub loss_variance: f32,
    /// Trajectory smoothness (avg absolute second derivative).
    pub trajectory_smoothness: f32,
}

impl Default for RolloutMetrics {
    fn default() -> Self {
        Self {
            step_deltas: Vec::new(),
            hidden_norms: Vec::new(),
            loss_variance: 0.0,
            trajectory_smoothness: 0.0,
        }
    }
}

// ============================================================================
// CubeCL Kernel Definition
// ============================================================================

/// RSSM multi-step rollout kernel - Production implementation.
///
/// Each block handles one ensemble member, rolling out y_steps predictions.
///
/// # Current Status
///
/// **Phase 3 - Placeholder**: Kernel structure defined but CubeCL implementation
/// pending. This will be completed after Phase 2 GRU kernel is validated.
///
/// # Implementation Plan
///
/// 1. Grid/block configuration: ensemble_size blocks × hidden_dim threads
/// 2. Load GRU weights to shared memory (if they fit, else global reads)
/// 3. Sequential loop over y_steps:
///    - Call GRU forward (reuse gru_forward_fused logic)
///    - Stochastic sampling via softmax reduction
///    - Decode loss via linear projection
///    - Update features[0] with predicted loss
/// 4. Write trajectory and final state to global memory
#[cfg(feature = "cuda")]
pub fn rssm_rollout_gpu(
    _config: &RssmRolloutConfig,
    _weights: &RssmEnsembleWeights,
    _initial_latents: &[Vec<f32>],
    _features: &[f32],
) -> HybridResult<Vec<RolloutResult>> {
    // TODO: Phase 3 - Implement actual kernel
    // For now, return error indicating not implemented
    Err((
        HybridTrainingError::GpuError {
            detail: "RSSM rollout GPU kernel not yet implemented (Phase 3)".to_string(),
        },
        None,
    ))
}

/// CPU reference implementation for validation.
///
/// This matches the logic in `RSSMLite::predict_y_steps` for testing.
pub fn rssm_rollout_cpu(
    config: &RssmRolloutConfig,
    weights: &RssmEnsembleWeights,
    initial_latents: &[Vec<f32>],
    features: &[f32],
    initial_loss: f32,
) -> Vec<RolloutResult> {
    let _span = tracing::debug_span!(
        "rssm_rollout_cpu",
        ensemble_size = config.ensemble_size,
        y_steps = config.y_steps,
        hidden_dim = config.hidden_dim
    )
    .entered();

    let mut results = Vec::with_capacity(config.ensemble_size);

    for ensemble_idx in 0..config.ensemble_size {
        let _member_span = tracing::trace_span!(
            "ensemble_member",
            idx = ensemble_idx
        )
        .entered();

        let gru_weights = &weights.gru_weights[ensemble_idx];
        let loss_head_w = &weights.loss_head_weights;

        // Clone initial latent state
        let mut deterministic = initial_latents[ensemble_idx].clone();
        let mut trajectory = Vec::with_capacity(config.y_steps + 1);
        trajectory.push(initial_loss);

        // Evolving features (update loss prediction each step)
        let mut evolving_features = features.to_vec();

        let mut total_entropy = 0.0_f32;

        // Metrics collection for research
        let mut step_deltas = Vec::with_capacity(config.y_steps);
        let mut hidden_norms = Vec::with_capacity(config.y_steps);

        for step_idx in 0..config.y_steps {
            let _step_span = tracing::trace_span!(
                "rollout_step",
                step = step_idx
            )
            .entered();

            // GRU forward pass
            deterministic = crate::gpu::kernels::gru::gru_forward_cpu(
                gru_weights,
                &deterministic,
                &evolving_features,
            );

            // Compute hidden state L2 norm for metrics
            let hidden_norm: f32 = deterministic.iter().map(|x| x * x).sum::<f32>().sqrt();
            hidden_norms.push(hidden_norm);

            // Simplified stochastic sampling (placeholder - real version uses softmax)
            // For CPU reference, just copy deterministic to stochastic
            let stochastic = deterministic.clone();

            // Combined state
            let mut combined = Vec::with_capacity(config.combined_dim());
            combined.extend_from_slice(&deterministic);
            combined.extend_from_slice(&stochastic);

            // Decode loss via linear projection (no bias)
            let loss_pred: f32 = combined
                .iter()
                .zip(loss_head_w.iter())
                .map(|(c, w)| c * w)
                .sum();

            // Track step delta for metrics
            let prev_loss = trajectory.last().copied().unwrap_or(initial_loss);
            step_deltas.push((loss_pred - prev_loss).abs());

            trajectory.push(loss_pred);

            tracing::trace!(
                step = step_idx,
                loss_pred = %loss_pred,
                hidden_norm = %hidden_norm,
                delta = %(loss_pred - prev_loss).abs(),
                "Rollout step complete"
            );

            // Update features[0] for next step (if not last step)
            if step_idx + 1 < config.y_steps {
                evolving_features[0] = loss_pred;
            }

            // Placeholder entropy (real version computes from softmax)
            total_entropy += 0.01;
        }

        // Final combined state after all steps
        let stochastic = deterministic.clone();
        let mut final_combined = Vec::with_capacity(config.combined_dim());
        final_combined.extend_from_slice(&deterministic);
        final_combined.extend_from_slice(&stochastic);

        // Compute trajectory metrics for research
        let loss_variance = if trajectory.len() > 1 {
            let mean = trajectory.iter().sum::<f32>() / trajectory.len() as f32;
            trajectory.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / trajectory.len() as f32
        } else {
            0.0
        };

        // Trajectory smoothness: average absolute second derivative
        let trajectory_smoothness = if trajectory.len() > 2 {
            let mut second_derivs = Vec::new();
            for i in 1..trajectory.len() - 1 {
                let d2 = trajectory[i+1] - 2.0 * trajectory[i] + trajectory[i-1];
                second_derivs.push(d2.abs());
            }
            second_derivs.iter().sum::<f32>() / second_derivs.len() as f32
        } else {
            0.0
        };

        let metrics = RolloutMetrics {
            step_deltas,
            hidden_norms,
            loss_variance,
            trajectory_smoothness,
        };

        tracing::debug!(
            ensemble_idx,
            final_loss = %trajectory.last().copied().unwrap_or(0.0),
            total_entropy = %total_entropy,
            loss_variance = %loss_variance,
            smoothness = %trajectory_smoothness,
            "Ensemble member rollout complete"
        );

        results.push(RolloutResult {
            trajectory,
            final_combined,
            total_entropy,
            metrics,
        });
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::kernels::gru;

    #[test]
    fn test_rssm_config_validation() {
        let valid = RssmRolloutConfig::default();
        assert!(valid.validate().is_ok());

        let invalid = RssmRolloutConfig {
            hidden_dim: 2048, // Exceeds limit
            ..Default::default()
        };
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn test_rssm_config_combined_dim() {
        let config = RssmRolloutConfig {
            hidden_dim: 256,
            stochastic_dim: 256,
            ..Default::default()
        };
        assert_eq!(config.combined_dim(), 512);
    }

    #[test]
    fn test_rssm_rollout_cpu_basic() {
        let config = RssmRolloutConfig {
            ensemble_size: 2,
            hidden_dim: 4,
            stochastic_dim: 4,
            feature_dim: 2,
            y_steps: 3,
        };

        // Create dummy weights
        let gru_weights = vec![
            crate::gpu::kernels::gru::GpuGruWeights {
                w_z: vec![0.1; 4 * 2],
                u_z: vec![0.1; 4 * 4],
                b_z: vec![0.0; 4],
                w_r: vec![0.1; 4 * 2],
                u_r: vec![0.1; 4 * 4],
                b_r: vec![0.0; 4],
                w_h: vec![0.1; 4 * 2],
                u_h: vec![0.1; 4 * 4],
                b_h: vec![0.0; 4],
                hidden_dim: 4,
                input_dim: 2,
            };
            2
        ];

        let weights = RssmEnsembleWeights {
            gru_weights,
            loss_head_weights: vec![0.1; 8],
            delta_head_weights: vec![0.1; 8 * 10],
        };

        let initial_latents = vec![vec![0.5; 4]; 2];
        let features = vec![1.0, 0.5];
        let initial_loss = 1.0;

        let results = rssm_rollout_cpu(&config, &weights, &initial_latents, &features, initial_loss);

        assert_eq!(results.len(), 2); // 2 ensemble members
        assert_eq!(results[0].trajectory.len(), 4); // y_steps + 1
        assert_eq!(results[0].final_combined.len(), 8); // combined_dim
    }

    #[test]
    fn test_rssm_rollout_cpu_deterministic() {
        let config = RssmRolloutConfig {
            ensemble_size: 1,
            hidden_dim: 4,
            stochastic_dim: 4,
            feature_dim: 2,
            y_steps: 5,
        };

        let gru_weights = vec![crate::gpu::kernels::gru::GpuGruWeights {
            w_z: vec![0.1; 4 * 2],
            u_z: vec![0.1; 4 * 4],
            b_z: vec![0.0; 4],
            w_r: vec![0.1; 4 * 2],
            u_r: vec![0.1; 4 * 4],
            b_r: vec![0.0; 4],
            w_h: vec![0.1; 4 * 2],
            u_h: vec![0.1; 4 * 4],
            b_h: vec![0.0; 4],
            hidden_dim: 4,
            input_dim: 2,
        }];

        let weights = RssmEnsembleWeights {
            gru_weights,
            loss_head_weights: vec![0.1; 8],
            delta_head_weights: vec![0.1; 8 * 10],
        };

        let initial_latents = vec![vec![0.5; 4]];
        let features = vec![1.0, 0.5];

        // Run twice with same inputs
        let results1 = rssm_rollout_cpu(&config, &weights, &initial_latents, &features, 1.0);
        let results2 = rssm_rollout_cpu(&config, &weights, &initial_latents, &features, 1.0);

        // Should be deterministic
        for (r1, r2) in results1[0].trajectory.iter().zip(results2[0].trajectory.iter()) {
            assert!((r1 - r2).abs() < 1e-6);
        }
    }
}
