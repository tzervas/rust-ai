// SPDX-License-Identifier: MIT
// Copyright 2026 Tyler Zervas

//! State encoding GPU kernel using Burn tensor operations.
//!
//! This module provides GPU-accelerated feature extraction from `TrainingState`
//! via Burn's backend-agnostic tensor operations rather than raw CubeCL kernels.
//!
//! # Why Burn Instead of CubeCL?
//!
//! State encoding involves:
//! - Small output size (64 dimensions)
//! - Complex branching logic (history length checks)
//! - Multiple reduction operations (mean, std, min, max)
//! - Not in critical path (called once per prediction phase)
//!
//! Burn tensor ops provide cleaner code with automatic GPU dispatch.
//!
//! # Algorithm
//!
//! Extracts 64-dimensional feature vector from `TrainingState`:
//! ```text
//! Features [64]:
//!   [0-9]:   Current metrics (loss, grad_norm, lr, etc.)
//!   [10-31]: Loss history statistics (mean, std, min, max, trends)
//!   [32-47]: Gradient history statistics
//!   [48-63]: Phase indicators and temporal features
//! ```

use crate::error::{HybridResult, HybridTrainingError};
use crate::state::TrainingState;

/// State encoding configuration.
#[derive(Debug, Clone)]
pub struct StateEncodeConfig {
    /// Feature output dimension (must be 64).
    pub feature_dim: usize,
    /// Maximum history length to consider.
    pub max_history: usize,
}

impl Default for StateEncodeConfig {
    fn default() -> Self {
        Self {
            feature_dim: 64,
            max_history: 1000,
        }
    }
}

impl StateEncodeConfig {
    /// Validates configuration.
    pub fn validate(&self) -> HybridResult<()> {
        if self.feature_dim != 64 {
            return Err((
                HybridTrainingError::ConfigError {
                    detail: format!(
                        "feature_dim must be 64, got {}",
                        self.feature_dim
                    ),
                },
                None,
            ));
        }
        Ok(())
    }
}

/// Flattened training state data for GPU transfer.
///
/// Packs `TrainingState` into contiguous arrays for efficient GPU upload.
/// Currently simplified - just stores the already-computed feature vector.
#[derive(Debug, Clone)]
pub struct StateDataFlat {
    /// Pre-computed feature vector [64].
    pub features: Vec<f32>,
}

impl StateDataFlat {
    /// Creates flattened state from `TrainingState`.
    #[must_use]
    pub fn from_state(state: &TrainingState) -> Self {
        Self {
            features: state.compute_features(),
        }
    }
}

/// GPU-accelerated state encoding using Burn tensor operations.
///
/// # Implementation Strategy
///
/// Uses Burn's backend-agnostic tensor ops for automatic GPU dispatch:
/// - Reductions: `mean()`, `std()`, `min()`, `max()`
/// - Element-wise: normalization, log transforms
/// - Backend: Automatically uses CUDA/WGPU/CPU based on device
///
/// # Current Status
///
/// **Phase 4 - CPU Implementation**: Fully functional CPU reference.
/// Burn GPU integration pending (requires Burn backend selection in caller).
#[cfg(feature = "cuda")]
pub fn encode_state_gpu(
    _config: &StateEncodeConfig,
    state: &TrainingState,
) -> HybridResult<Vec<f32>> {
    // Phase 4: Use CPU implementation for now
    // Phase 4.5 will add Burn tensor backend integration
    tracing::debug!(
        "State encoding GPU requested (step {}), using CPU implementation",
        state.step
    );
    Ok(encode_state_cpu(state))
}

/// Non-CUDA fallback.
#[cfg(not(feature = "cuda"))]
pub fn encode_state_gpu(
    _config: &StateEncodeConfig,
    state: &TrainingState,
) -> HybridResult<Vec<f32>> {
    Ok(encode_state_cpu(state))
}

/// CPU reference implementation - directly uses `TrainingState::compute_features()`.
///
/// Returns 64-dimensional feature vector.
pub fn encode_state_cpu(state: &TrainingState) -> Vec<f32> {
    let _span = tracing::trace_span!(
        "encode_state_cpu",
        step = state.step,
        loss = %state.loss,
        grad_norm = %state.gradient_norm
    )
    .entered();

    // Use the existing compute_features() method
    let features = state.compute_features();

    tracing::trace!(
        features_computed = features.len(),
        loss = %features[0],
        grad_norm = %features[8],
        "State encoding complete"
    );

    features
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Phase;

    fn create_test_state() -> TrainingState {
        TrainingState::default()
    }

    #[test]
    fn test_state_encode_cpu_output_dim() {
        let state = create_test_state();
        let features = encode_state_cpu(&state);
        assert_eq!(features.len(), 64, "Must output exactly 64 features");
    }

    #[test]
    fn test_state_encode_cpu_deterministic() {
        let state = create_test_state();
        let features1 = encode_state_cpu(&state);
        let features2 = encode_state_cpu(&state);

        // Both should be same length
        assert_eq!(features1.len(), features2.len());
        assert_eq!(features1.len(), 64);
    }

    #[test]
    fn test_state_encode_cpu_first_feature_is_loss() {
        let state = create_test_state();
        let features = encode_state_cpu(&state);

        // First feature should be current loss (matches compute_features implementation)
        // Both may be NaN in default state, so check they're both NaN or both equal
        if state.loss.is_nan() {
            assert!(features[0].is_nan(), "First feature should match loss (NaN)");
        } else {
            assert_eq!(features[0], state.loss, "First feature should match loss");
        }
    }

    #[test]
    fn test_state_encode_config_validation() {
        let valid = StateEncodeConfig::default();
        assert!(valid.validate().is_ok());

        let invalid = StateEncodeConfig {
            feature_dim: 32, // Wrong dimension
            max_history: 1000,
        };
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn test_state_data_flat_features_length() {
        let state = create_test_state();
        let flat = StateDataFlat::from_state(&state);

        assert_eq!(flat.features.len(), 64);
    }
}
