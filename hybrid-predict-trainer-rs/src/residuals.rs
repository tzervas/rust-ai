//! Residual extraction and storage for prediction correction.
//!
//! Residuals are the differences between predicted and actual training
//! outcomes. They are extracted during full training phases and used
//! during correction phases to improve prediction accuracy.
//!
//! # Why Store Residuals?
//!
//! Residuals encode the predictor's systematic errors. By storing them
//! alongside the training state context, we can:
//! - **Learn error patterns**: Similar states produce similar errors
//! - **Improve predictions online**: Apply learned corrections to future predictions
//! - **Diagnose predictor weaknesses**: Analyze residual distributions
//!
//! # Residual Types
//!
//! - **Loss residuals**: Difference between predicted and actual loss
//! - **Gradient residuals**: Difference between predicted and actual gradients
//! - **Weight residuals**: Accumulated error in weight predictions
//!
//! # Storage Strategy
//!
//! Residuals can be stored in:
//! - Memory (fast, limited capacity)
//! - Disk (slower, unlimited capacity)
//! - Hybrid (recent in memory, older on disk)
//!
//! # Compression
//!
//! Gradient residuals can be compressed using low-rank approximation
//! (similar to `PowerSGD`) to reduce memory footprint while preserving
//! the most important correction information.

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

use crate::state::TrainingState;

/// A single residual observation from comparing prediction to reality.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Residual {
    /// Training step where residual was computed.
    pub step: u64,

    /// Phase during which this was collected.
    pub phase: crate::Phase,

    /// Number of prediction steps this residual covers.
    pub prediction_horizon: usize,

    /// Loss residual (actual - predicted).
    pub loss_residual: f32,

    /// Per-layer gradient residuals (compressed).
    pub gradient_residuals: Vec<LayerResidual>,

    /// Training state features at time of prediction.
    pub state_features: Vec<f32>,

    /// Prediction confidence when this residual was generated.
    pub prediction_confidence: f32,
}

/// Residual information for a single layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerResidual {
    /// Layer name/identifier.
    pub layer_name: String,

    /// Residual magnitude (L2 norm).
    pub magnitude: f32,

    /// Compressed residual (low-rank factors).
    pub compressed: Option<CompressedResidual>,

    /// Cosine similarity between predicted and actual gradient.
    pub cosine_similarity: f32,
}

/// Low-rank compressed representation of gradient residual.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressedResidual {
    /// Left factor of low-rank decomposition.
    pub left_factor: Vec<f32>,

    /// Right factor of low-rank decomposition.
    pub right_factor: Vec<f32>,

    /// Singular values.
    pub singular_values: Vec<f32>,

    /// Rank of the compression.
    pub rank: usize,

    /// Original tensor shape (rows, cols).
    pub original_shape: (usize, usize),

    /// Compression ratio achieved.
    pub compression_ratio: f32,
}

impl CompressedResidual {
    /// Creates a new compressed residual from factors.
    #[must_use]
    pub fn new(
        left: Vec<f32>,
        right: Vec<f32>,
        singular_values: Vec<f32>,
        shape: (usize, usize),
    ) -> Self {
        let rank = singular_values.len();
        let original_size = shape.0 * shape.1;
        let compressed_size = left.len() + right.len() + rank;
        let compression_ratio = compressed_size as f32 / original_size as f32;

        Self {
            left_factor: left,
            right_factor: right,
            singular_values,
            rank,
            original_shape: shape,
            compression_ratio,
        }
    }

    /// Reconstructs the full residual tensor.
    ///
    /// Returns a vector in row-major order.
    #[must_use]
    pub fn reconstruct(&self) -> Vec<f32> {
        let (rows, cols) = self.original_shape;
        let mut result = vec![0.0; rows * cols];

        for r in 0..self.rank {
            let sigma = self.singular_values[r];
            for i in 0..rows {
                for j in 0..cols {
                    result[i * cols + j] += sigma
                        * self.left_factor[i * self.rank + r]
                        * self.right_factor[r * cols + j];
                }
            }
        }

        result
    }

    /// Returns the memory size in bytes.
    #[must_use]
    pub fn memory_size(&self) -> usize {
        (self.left_factor.len() + self.right_factor.len() + self.singular_values.len())
            * std::mem::size_of::<f32>()
    }
}

/// Storage for residuals with configurable capacity and eviction.
pub struct ResidualStore {
    /// In-memory residuals (recent).
    memory_store: VecDeque<Residual>,

    /// Maximum number of residuals to keep in memory.
    max_memory_residuals: usize,

    /// Total residuals collected (including evicted).
    total_collected: usize,

    /// Running statistics about residuals.
    statistics: ResidualStatistics,
}

/// Statistics about collected residuals.
#[derive(Debug, Clone, Default)]
pub struct ResidualStatistics {
    /// Total residuals collected.
    pub total_collected: usize,

    /// Mean loss residual magnitude.
    pub mean_loss_residual: f64,

    /// Variance of loss residuals.
    pub loss_residual_variance: f64,

    /// Mean gradient residual magnitude.
    pub mean_gradient_residual: f64,

    /// Mean prediction confidence when residuals were collected.
    pub mean_confidence: f64,

    /// Correlation between confidence and residual magnitude.
    pub confidence_residual_correlation: f64,
}

impl Default for ResidualStore {
    fn default() -> Self {
        Self::new(1000)
    }
}

impl ResidualStore {
    /// Creates a new residual store with the specified capacity.
    #[must_use]
    pub fn new(max_memory_residuals: usize) -> Self {
        Self {
            memory_store: VecDeque::with_capacity(max_memory_residuals),
            max_memory_residuals,
            total_collected: 0,
            statistics: ResidualStatistics::default(),
        }
    }

    /// Adds a residual to the store.
    pub fn add(&mut self, residual: Residual) {
        self.update_statistics(&residual);

        if self.memory_store.len() >= self.max_memory_residuals {
            self.memory_store.pop_front();
        }

        self.memory_store.push_back(residual);
        self.total_collected += 1;
    }

    /// Returns the most recent residuals.
    #[must_use]
    pub fn recent(&self, n: usize) -> Vec<&Residual> {
        self.memory_store.iter().rev().take(n).collect()
    }

    /// Returns all stored residuals.
    pub fn all(&self) -> impl Iterator<Item = &Residual> {
        self.memory_store.iter()
    }

    /// Returns the number of stored residuals.
    #[must_use]
    pub fn len(&self) -> usize {
        self.memory_store.len()
    }

    /// Returns whether the store is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.memory_store.is_empty()
    }

    /// Returns residual statistics.
    #[must_use]
    pub fn statistics(&self) -> &ResidualStatistics {
        &self.statistics
    }

    /// Finds residuals similar to the given state for correction.
    ///
    /// # Arguments
    ///
    /// * `state` - Current training state
    /// * `n` - Maximum number of similar residuals to return
    ///
    /// # Returns
    ///
    /// Residuals ordered by similarity (most similar first).
    #[must_use]
    pub fn find_similar(&self, state: &TrainingState, n: usize) -> Vec<&Residual> {
        let current_features = state.compute_features();

        let mut scored: Vec<_> = self
            .memory_store
            .iter()
            .map(|r| {
                let similarity = cosine_similarity(&current_features, &r.state_features);
                (similarity, r)
            })
            .collect();

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        scored.into_iter().take(n).map(|(_, r)| r).collect()
    }

    /// Clears all stored residuals.
    pub fn clear(&mut self) {
        self.memory_store.clear();
    }

    /// Updates running statistics with a new residual.
    fn update_statistics(&mut self, residual: &Residual) {
        let n = self.statistics.total_collected as f64;
        let n1 = n + 1.0;

        // Update mean loss residual
        let loss_abs = f64::from(residual.loss_residual.abs());
        self.statistics.mean_loss_residual =
            (self.statistics.mean_loss_residual * n + loss_abs) / n1;

        // Update mean confidence
        let conf = f64::from(residual.prediction_confidence);
        self.statistics.mean_confidence = (self.statistics.mean_confidence * n + conf) / n1;

        // Update mean gradient residual
        if !residual.gradient_residuals.is_empty() {
            let grad_mag: f64 = residual
                .gradient_residuals
                .iter()
                .map(|g| f64::from(g.magnitude))
                .sum::<f64>()
                / residual.gradient_residuals.len() as f64;
            self.statistics.mean_gradient_residual =
                (self.statistics.mean_gradient_residual * n + grad_mag) / n1;
        }

        self.statistics.total_collected += 1;
    }
}

/// Computes cosine similarity between two feature vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a < 1e-8 || norm_b < 1e-8 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

/// Trait for residual extractors.
///
/// Implementations extract residuals from the difference between
/// predicted and actual training outcomes.
pub trait ResidualExtractor: Send + Sync {
    /// Extracts a residual from prediction vs reality.
    ///
    /// # Arguments
    ///
    /// * `state_before` - Training state before prediction
    /// * `state_after` - Actual training state after full training
    /// * `predicted_loss` - The loss that was predicted
    /// * `prediction_horizon` - Number of steps the prediction covered
    /// * `confidence` - Predictor confidence at time of prediction
    ///
    /// # Returns
    ///
    /// The extracted residual.
    fn extract(
        &self,
        state_before: &TrainingState,
        state_after: &TrainingState,
        predicted_loss: f32,
        prediction_horizon: usize,
        confidence: f32,
    ) -> Residual;

    /// Returns the compression rank used for gradient residuals.
    fn compression_rank(&self) -> usize;
}

/// Default residual extractor with configurable compression.
pub struct DefaultResidualExtractor {
    /// Rank for low-rank compression.
    compression_rank: usize,

    /// Whether to store full residuals (no compression).
    _store_full: bool,
}

impl Default for DefaultResidualExtractor {
    fn default() -> Self {
        Self::new(4)
    }
}

impl DefaultResidualExtractor {
    /// Creates a new extractor with the specified compression rank.
    #[must_use]
    pub fn new(compression_rank: usize) -> Self {
        Self {
            compression_rank,
            _store_full: false,
        }
    }

    /// Creates an extractor that stores full (uncompressed) residuals.
    #[must_use]
    pub fn full_precision() -> Self {
        Self {
            compression_rank: 0,
            _store_full: true,
        }
    }
}

impl ResidualExtractor for DefaultResidualExtractor {
    fn extract(
        &self,
        state_before: &TrainingState,
        state_after: &TrainingState,
        predicted_loss: f32,
        prediction_horizon: usize,
        confidence: f32,
    ) -> Residual {
        let loss_residual = state_after.loss - predicted_loss;

        Residual {
            step: state_after.step,
            phase: state_before.current_phase,
            prediction_horizon,
            loss_residual,
            gradient_residuals: Vec::new(), // Would be populated with actual gradient info
            state_features: state_before.compute_features(),
            prediction_confidence: confidence,
        }
    }

    fn compression_rank(&self) -> usize {
        self.compression_rank
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_residual_store_capacity() {
        let mut store = ResidualStore::new(3);

        for i in 0..5 {
            store.add(Residual {
                step: i,
                phase: crate::Phase::Full,
                prediction_horizon: 10,
                loss_residual: i as f32 * 0.1,
                gradient_residuals: Vec::new(),
                state_features: vec![i as f32],
                prediction_confidence: 0.9,
            });
        }

        // Should only have 3 most recent
        assert_eq!(store.len(), 3);
        let recent: Vec<_> = store.recent(3);
        assert_eq!(recent[0].step, 4);
        assert_eq!(recent[1].step, 3);
        assert_eq!(recent[2].step, 2);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.001);

        let c = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &c).abs() < 0.001);
    }

    #[test]
    fn test_compressed_residual_reconstruct() {
        // Simple 2x2 matrix with rank-1 approximation
        let compressed = CompressedResidual::new(
            vec![1.0, 0.5, 2.0, 1.0], // Left factor (2x2 in row-major, but rank=2 so 2x2)
            vec![1.0, 0.5, 0.5, 0.25], // Right factor
            vec![1.0, 0.5],           // Singular values
            (2, 2),
        );

        let reconstructed = compressed.reconstruct();
        assert_eq!(reconstructed.len(), 4);
    }

    /// Helper: create a Residual with given step, loss_residual, confidence, and state features.
    fn make_residual(
        step: u64,
        loss_residual: f32,
        confidence: f32,
        state_features: Vec<f32>,
    ) -> Residual {
        Residual {
            step,
            phase: crate::Phase::Predict,
            prediction_horizon: 10,
            loss_residual,
            gradient_residuals: Vec::new(),
            state_features,
            prediction_confidence: confidence,
        }
    }

    /// Helper: create a TrainingState with recorded steps for realistic features.
    fn make_state(step: u64, loss: f32, grad_norm: f32) -> TrainingState {
        let mut state = TrainingState::new();
        for i in 0..step {
            state.record_step(loss + 0.01 * i as f32, grad_norm);
        }
        state
    }

    #[test]
    fn test_find_similar_returns_most_similar() {
        let mut store = ResidualStore::new(100);

        // Create a query state and get its features
        let query_state = make_state(10, 2.0, 1.0);
        let query_features = query_state.compute_features();

        // Add a residual with features identical to the query (most similar)
        store.add(make_residual(1, 0.1, 0.9, query_features.clone()));

        // Add a residual with a very different feature vector (least similar)
        let mut far_features = vec![0.0; 64];
        far_features[0] = -100.0;
        far_features[1] = 50.0;
        far_features[2] = -200.0;
        store.add(make_residual(2, 0.2, 0.9, far_features));

        // Add a residual with partially similar features (medium similarity)
        let mut medium_features = query_features.clone();
        for f in medium_features.iter_mut().take(32) {
            *f *= 0.5; // Modify half the features
        }
        store.add(make_residual(3, 0.3, 0.9, medium_features));

        let similar = store.find_similar(&query_state, 3);

        assert_eq!(similar.len(), 3);
        // The first result should be the most similar (identical features at step 1)
        assert_eq!(
            similar[0].step, 1,
            "Most similar residual should be the one with identical features (step 1)"
        );
        // The last result should be the least similar (far features at step 2)
        assert_eq!(
            similar[2].step, 2,
            "Least similar residual should be the one with far features (step 2)"
        );
    }

    #[test]
    fn test_find_similar_empty_store() {
        let store = ResidualStore::new(100);
        let state = make_state(10, 2.0, 1.0);

        let similar = store.find_similar(&state, 5);

        assert!(
            similar.is_empty(),
            "find_similar on empty store should return empty vec"
        );
    }

    #[test]
    fn test_statistics_update_correctness() {
        let mut store = ResidualStore::new(100);

        // Add residuals with known values
        let loss_residuals = [0.1_f32, 0.2, 0.3, 0.4, 0.5];
        let confidences = [0.8_f32, 0.85, 0.9, 0.95, 1.0];

        for (i, (&lr, &conf)) in loss_residuals.iter().zip(confidences.iter()).enumerate() {
            store.add(make_residual(i as u64, lr, conf, vec![0.0; 64]));
        }

        let stats = store.statistics();

        // Verify total_collected
        assert_eq!(stats.total_collected, 5);

        // Verify mean_loss_residual (mean of absolute values)
        // |0.1| + |0.2| + |0.3| + |0.4| + |0.5| = 1.5, mean = 0.3
        // But the mean is computed incrementally: (running_mean * n + new) / (n+1)
        // After 5 additions: mean of [0.1, 0.2, 0.3, 0.4, 0.5] = 0.3
        assert!(
            (stats.mean_loss_residual - 0.3).abs() < 1e-6,
            "Mean loss residual should be 0.3, got {}",
            stats.mean_loss_residual
        );

        // Verify mean_confidence
        // (0.8 + 0.85 + 0.9 + 0.95 + 1.0) / 5 = 4.5 / 5 = 0.9
        assert!(
            (stats.mean_confidence - 0.9).abs() < 1e-6,
            "Mean confidence should be 0.9, got {}",
            stats.mean_confidence
        );
    }

    #[test]
    fn test_compressed_residual_memory_size() {
        // left: 4 elements, right: 6 elements, singular_values: 2 elements
        // Total = (4 + 6 + 2) * 4 bytes = 48 bytes
        let compressed = CompressedResidual::new(
            vec![1.0, 2.0, 3.0, 4.0],            // left: 4 elements
            vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0], // right: 6 elements
            vec![1.5, 0.5],                      // singular_values: 2 elements
            (2, 3),
        );

        let expected_size = (4 + 6 + 2) * std::mem::size_of::<f32>();
        assert_eq!(
            compressed.memory_size(),
            expected_size,
            "Memory size should be {} bytes, got {}",
            expected_size,
            compressed.memory_size()
        );
        assert_eq!(compressed.memory_size(), 48);
    }

    #[test]
    fn test_compressed_residual_rank1_exact() {
        // Create a rank-1 matrix: M = sigma * u * v^T
        // where u = [1, 2], v = [3, 4, 5], sigma = 2.0
        // M = 2.0 * [1, 2]^T * [3, 4, 5]
        //   = [[6, 8, 10],
        //      [12, 16, 20]]
        let rows = 2;
        let cols = 3;
        let rank = 1;

        // Left factor: u stored as rows x rank = 2x1, so [1.0, 2.0]
        let left = vec![1.0, 2.0];
        // Right factor: v stored as rank x cols = 1x3, so [3.0, 4.0, 5.0]
        let right = vec![3.0, 4.0, 5.0];
        let singular_values = vec![2.0];

        let compressed = CompressedResidual::new(left, right, singular_values, (rows, cols));

        assert_eq!(compressed.rank, rank);

        let reconstructed = compressed.reconstruct();
        assert_eq!(reconstructed.len(), rows * cols);

        // Expected matrix in row-major: [6, 8, 10, 12, 16, 20]
        let expected = vec![6.0, 8.0, 10.0, 12.0, 16.0, 20.0];

        for (i, (&actual, &exp)) in reconstructed.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - exp).abs() < 1e-5,
                "Element [{i}] mismatch: expected {exp}, got {actual}"
            );
        }
    }

    #[test]
    fn test_store_eviction_preserves_recent() {
        let capacity = 5;
        let mut store = ResidualStore::new(capacity);

        // Add 10 residuals (more than capacity)
        for i in 0..10 {
            store.add(make_residual(
                i as u64,
                i as f32 * 0.1,
                0.9,
                vec![i as f32; 64],
            ));
        }

        // Store should only have `capacity` residuals
        assert_eq!(store.len(), capacity);

        // All stored residuals should be from the most recent additions (steps 5-9)
        let all_steps: Vec<u64> = store.all().map(|r| r.step).collect();
        for &step in &all_steps {
            assert!(
                step >= 5,
                "Eviction should remove old residuals. Found step {step}, expected >= 5"
            );
        }

        // Verify that the most recent residuals are specifically steps 5, 6, 7, 8, 9
        let mut sorted_steps = all_steps.clone();
        sorted_steps.sort();
        assert_eq!(
            sorted_steps,
            vec![5, 6, 7, 8, 9],
            "After eviction, expected steps [5, 6, 7, 8, 9], got {sorted_steps:?}"
        );

        // Verify total_collected tracks all additions (not just stored)
        assert_eq!(store.statistics().total_collected, 10);
    }
}
