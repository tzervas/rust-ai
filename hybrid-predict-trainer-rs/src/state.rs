//! Training state management and encoding.
//!
//! This module provides the [`TrainingState`] struct that captures the complete
//! state of training at any point in time, along with utilities for encoding
//! state into compact representations suitable for the dynamics predictor.
//!
//! # Why Track Training State?
//!
//! Accurate prediction of training dynamics requires rich context about the
//! current training regime. By tracking loss and gradient histories along with
//! optimizer statistics, the predictor can identify patterns like:
//! - Learning rate warmup/decay phases
//! - Loss plateau regions indicating convergence
//! - Gradient magnitude trends signaling instability
//!
//! # Overview
//!
//! Training state encompasses:
//! - Current step and loss values
//! - Historical loss and gradient norm trajectories
//! - Optimizer state summaries (momentum, variance estimates)
//! - Optional K-FAC factors for structured gradient approximation
//!
//! The state encoder compresses this information into a fixed-size latent
//! representation that captures the essential training dynamics while
//! remaining small enough for efficient prediction.
//!
//! # Example
//!
//! ```rust
//! use hybrid_predict_trainer_rs::state::{TrainingState, StateEncoder};
//!
//! // Create a new training state
//! let mut state = TrainingState::new();
//!
//! // Update state after each step
//! state.record_step(2.5, 1.2); // loss, gradient_norm
//! state.record_step(2.3, 1.1);
//! state.record_step(2.1, 1.0);
//!
//! // Encode for prediction
//! // let encoder = StateEncoder::new(128);
//! // let encoded = encoder.encode(&state);
//! ```

use serde::{Deserialize, Serialize};

/// Exponential Moving Average tracker for multiple timescales.
///
/// Tracks fast (4-step), medium (16-step), and slow (64-step) EMAs to capture
/// training dynamics at different temporal resolutions. The spread between
/// fast and slow EMAs indicates momentum direction and magnitude.
///
/// # Why Multiple Timescales?
///
/// Training dynamics exhibit patterns at different frequencies. Short-term
/// fluctuations (batch noise) are captured by the fast EMA, while longer-term
/// trends (convergence, learning rate effects) appear in the slow EMA. The
/// predictor uses these signals to distinguish transient noise from systematic
/// changes in training behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiScaleEMA {
    fast: f32,
    medium: f32,
    slow: f32,
    count: usize,
}

impl Default for MultiScaleEMA {
    fn default() -> Self {
        Self::new()
    }
}

impl MultiScaleEMA {
    /// Creates a new multi-scale EMA with zeroed initial values.
    #[must_use]
    pub fn new() -> Self {
        Self {
            fast: 0.0,
            medium: 0.0,
            slow: 0.0,
            count: 0,
        }
    }
    /// Updates all EMA scales with a new value.
    pub fn update(&mut self, value: f32) {
        self.count += 1;
        if self.count == 1 {
            self.fast = value;
            self.medium = value;
            self.slow = value;
        } else {
            self.fast = 0.75 * self.fast + 0.25 * value;
            self.medium = 0.9375 * self.medium + 0.0625 * value;
            self.slow = 0.984_375 * self.slow + 0.015_625 * value;
        }
    }
    /// Returns the fast EMA value.
    #[must_use]
    pub fn fast(&self) -> f32 {
        self.fast
    }
    /// Returns the medium EMA value.
    #[must_use]
    pub fn medium(&self) -> f32 {
        self.medium
    }
    /// Returns the slow EMA value.
    #[must_use]
    pub fn slow(&self) -> f32 {
        self.slow
    }
    /// Returns the spread between fast and slow EMAs.
    #[must_use]
    pub fn spread(&self) -> f32 {
        self.fast - self.slow
    }
    /// Returns whether the EMA has sufficient samples to be reliable.
    #[must_use]
    pub fn is_warm(&self) -> bool {
        self.count >= 16
    }
}

/// Running statistics for online feature normalization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunningStats {
    mean: f64,
    variance: f64,
    count: usize,
}

impl Default for RunningStats {
    fn default() -> Self {
        Self::new()
    }
}

impl RunningStats {
    /// Creates a new running statistics tracker.
    #[must_use]
    pub fn new() -> Self {
        Self {
            mean: 0.0,
            variance: 0.0,
            count: 0,
        }
    }
    /// Updates the statistics with a new value.
    pub fn update(&mut self, value: f64) {
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = value - self.mean;
        self.variance += delta * delta2;
    }
    /// Normalizes a value using the running mean and standard deviation.
    #[must_use]
    pub fn normalize(&self, value: f64) -> f32 {
        if self.count < 2 {
            return 0.0;
        }
        let std_dev = (self.variance / self.count as f64).sqrt();
        if std_dev < 1e-8 {
            return 0.0;
        }
        let normalized = (value - self.mean) / std_dev;
        normalized.clamp(-10.0, 10.0) as f32
    }
    /// Returns the current running mean.
    #[must_use]
    pub fn mean(&self) -> f64 {
        self.mean
    }
    /// Returns the current standard deviation.
    #[must_use]
    pub fn std_dev(&self) -> f64 {
        if self.count < 2 {
            return 0.0;
        }
        (self.variance / self.count as f64).sqrt()
    }
}

/// Fixed-size ring buffer for history tracking.
///
/// Provides O(1) insertion and maintains the most recent N values.
/// Used for loss and gradient norm history.
///
/// # Why a Ring Buffer?
///
/// Training history needs bounded memory regardless of training duration.
/// A ring buffer automatically evicts old values while preserving the most
/// recent N observations, which are most relevant for dynamics prediction.
/// The O(1) operations ensure minimal overhead per training step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RingBuffer<T, const N: usize> {
    buffer: Vec<T>,
    head: usize,
    len: usize,
}

impl<T: Clone + Default, const N: usize> Default for RingBuffer<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Clone + Default, const N: usize> RingBuffer<T, N> {
    /// Creates a new empty ring buffer.
    #[must_use]
    pub fn new() -> Self {
        Self {
            buffer: vec![T::default(); N],
            head: 0,
            len: 0,
        }
    }

    /// Pushes a value into the buffer, overwriting the oldest if full.
    pub fn push(&mut self, value: T) {
        self.buffer[self.head] = value;
        self.head = (self.head + 1) % N;
        if self.len < N {
            self.len += 1;
        }
    }

    /// Returns the number of values in the buffer.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns whether the buffer is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns an iterator over values in chronological order (oldest first).
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        let start = if self.len < N { 0 } else { self.head };
        let len = self.len;
        let buffer = &self.buffer;
        (0..len).map(move |i| &buffer[(start + i) % N])
    }

    /// Returns an iterator over values in reverse chronological order (newest first).
    pub fn iter_rev(&self) -> impl Iterator<Item = &T> {
        let len = self.len;
        let buffer = &self.buffer;
        let head = self.head;
        (0..len).map(move |i| {
            let idx = if head == 0 {
                N - 1 - i
            } else {
                (head + N - 1 - i) % N
            };
            &buffer[idx]
        })
    }

    /// Returns the most recent value, if any.
    #[must_use]
    pub fn last(&self) -> Option<&T> {
        if self.len == 0 {
            None
        } else {
            let idx = if self.head == 0 { N - 1 } else { self.head - 1 };
            Some(&self.buffer[idx])
        }
    }

    /// Returns statistics (mean, std, min, max) for numeric buffers.
    pub fn statistics(&self) -> BufferStatistics
    where
        T: Into<f64> + Copy,
    {
        if self.is_empty() {
            return BufferStatistics::default();
        }

        let values: Vec<f64> = self.iter().map(|&v| v.into()).collect();
        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
        let std = variance.sqrt();
        let min = values.iter().copied().fold(f64::INFINITY, f64::min);
        let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        BufferStatistics {
            mean,
            std,
            min,
            max,
        }
    }
}

/// Statistics computed from a ring buffer.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BufferStatistics {
    /// Mean of values.
    pub mean: f64,
    /// Standard deviation of values.
    pub std: f64,
    /// Minimum value.
    pub min: f64,
    /// Maximum value.
    pub max: f64,
}

/// Summary of optimizer state for encoding.
///
/// Captures the essential characteristics of the optimizer's internal
/// state (e.g., Adam momentum and variance estimates) without storing
/// the full tensors.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OptimizerStateSummary {
    /// Mean of first moment estimates (momentum).
    pub momentum_mean: f32,
    /// Standard deviation of first moment estimates.
    pub momentum_std: f32,
    /// Mean of second moment estimates (variance).
    pub variance_mean: f32,
    /// Standard deviation of second moment estimates.
    pub variance_std: f32,
    /// Current effective learning rate (after scheduling).
    pub effective_lr: f32,
    /// Beta1^t decay factor for bias correction.
    pub beta1_power: f32,
    /// Beta2^t decay factor for bias correction.
    pub beta2_power: f32,
}

/// K-FAC (Kronecker-Factored Approximate Curvature) factors.
///
/// These capture structured gradient information through factorized
/// approximations of the Fisher information matrix.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct KFACFactors {
    /// Per-layer input activation covariance (A factors).
    pub activation_covariances: Vec<LayerKFACFactor>,
    /// Per-layer backpropagated gradient covariance (G factors).
    pub gradient_covariances: Vec<LayerKFACFactor>,
}

/// K-FAC factor for a single layer.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LayerKFACFactor {
    /// Layer name/identifier.
    pub layer_name: String,
    /// Eigenvalues of the factor (top-k for efficiency).
    pub eigenvalues: Vec<f32>,
    /// Trace of the full factor matrix.
    pub trace: f32,
    /// Exponential moving average decay used.
    pub ema_decay: f32,
}

/// Complete training state at a point in time.
///
/// Captures all information needed to:
/// 1. Resume training from this point
/// 2. Encode state for dynamics prediction
/// 3. Detect divergence and anomalies
///
/// # Memory Footprint
///
/// With default history sizes (256 loss, 64 gradient):
/// - Core fields: ~50 bytes
/// - Loss history: ~1 KB
/// - Gradient history: ~256 bytes
/// - Optimizer summary: ~32 bytes
/// - K-FAC factors (optional): Varies by model size
///
/// Total: ~1.5 KB without K-FAC, plus ~100 KB per 1B parameters with K-FAC.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingState {
    /// Current training step (0-indexed).
    pub step: u64,

    /// Current loss value.
    pub loss: f32,

    /// History of recent loss values.
    pub loss_history: RingBuffer<f32, 256>,

    /// Multi-scale exponential moving averages of loss.
    pub loss_ema: MultiScaleEMA,

    /// Running statistics for feature normalization.
    pub feature_normalizer: RunningStats,

    /// Current gradient norm.
    pub gradient_norm: f32,

    /// History of recent gradient norms.
    pub gradient_norm_history: RingBuffer<f32, 64>,

    /// Checksum for detecting unexpected weight changes.
    ///
    /// Computed as a hash of sampled weight values for efficiency.
    pub weight_checksum: u64,

    /// Summary of optimizer internal state.
    pub optimizer_state_summary: OptimizerStateSummary,

    /// K-FAC factors for structured prediction (optional).
    ///
    /// Only populated if K-FAC encoding is enabled in configuration.
    pub kfac_factors: Option<KFACFactors>,

    /// Current phase within the training cycle.
    pub current_phase: crate::Phase,

    /// Step within the current phase.
    pub phase_step: usize,

    /// Steps completed in current prediction phase (reset on phase transition).
    ///
    /// Used to track when to apply intra-horizon micro-corrections during
    /// long prediction phases.
    pub steps_in_current_phase: usize,

    /// Random state for reproducibility.
    pub random_seed: u64,
}

impl Default for TrainingState {
    fn default() -> Self {
        Self::new()
    }
}

impl TrainingState {
    /// Creates a new training state at step 0.
    #[must_use]
    pub fn new() -> Self {
        Self {
            step: 0,
            loss: f32::NAN,
            loss_history: RingBuffer::new(),
            loss_ema: MultiScaleEMA::new(),
            feature_normalizer: RunningStats::new(),
            gradient_norm: f32::NAN,
            gradient_norm_history: RingBuffer::new(),
            weight_checksum: 0,
            optimizer_state_summary: OptimizerStateSummary::default(),
            kfac_factors: None,
            current_phase: crate::Phase::Warmup,
            phase_step: 0,
            steps_in_current_phase: 0,
            random_seed: 0,
        }
    }

    /// Records metrics from a training step.
    ///
    /// # Arguments
    ///
    /// * `loss` - The loss value for this step
    /// * `gradient_norm` - The global gradient norm for this step
    pub fn record_step(&mut self, loss: f32, gradient_norm: f32) {
        self.step += 1;
        self.loss = loss;
        self.gradient_norm = gradient_norm;
        self.loss_history.push(loss);
        self.loss_ema.update(loss);
        self.gradient_norm_history.push(gradient_norm);
        self.phase_step += 1;
    }

    /// Transitions to a new phase.
    ///
    /// # Arguments
    ///
    /// * `phase` - The new phase to enter
    pub fn enter_phase(&mut self, phase: crate::Phase) {
        self.current_phase = phase;
        self.phase_step = 0;
        self.steps_in_current_phase = 0;
    }

    /// Returns statistics about recent loss values.
    #[must_use]
    pub fn loss_statistics(&self) -> BufferStatistics {
        self.loss_history.statistics()
    }

    /// Returns statistics about recent gradient norms.
    #[must_use]
    pub fn gradient_statistics(&self) -> BufferStatistics {
        self.gradient_norm_history.statistics()
    }

    /// Checks if loss is within expected bounds.
    ///
    /// # Arguments
    ///
    /// * `sigma_threshold` - Number of standard deviations for bounds
    ///
    /// # Returns
    ///
    /// `true` if loss is within `mean ± sigma_threshold * std`.
    #[must_use]
    pub fn loss_within_bounds(&self, sigma_threshold: f32) -> bool {
        let stats = self.loss_statistics();
        let bound = stats.std * f64::from(sigma_threshold);
        let deviation = (f64::from(self.loss) - stats.mean).abs();
        deviation <= bound
    }

    /// Computes a feature vector for state encoding.
    ///
    /// Returns a fixed-size vector of features derived from the training
    /// state, suitable for input to the dynamics predictor.
    ///
    /// # Returns
    ///
    /// A vector of f32 features with length determined by the feature set.
    #[must_use]
    pub fn compute_features(&self) -> Vec<f32> {
        let mut features = Vec::with_capacity(64);

        // Loss features (8)
        let loss_stats = self.loss_statistics();
        features.push(self.loss);
        features.push(loss_stats.mean as f32);
        features.push(loss_stats.std as f32);
        features.push(loss_stats.min as f32);
        features.push(loss_stats.max as f32);
        // Loss trend (recent vs older)
        if self.loss_history.len() >= 32 {
            // Collect to Vec first since RingBuffer iter doesn't implement DoubleEndedIterator
            let all_losses: Vec<f32> = self.loss_history.iter().copied().collect();
            let recent: Vec<f32> = all_losses.iter().rev().take(16).copied().collect();
            let older: Vec<f32> = all_losses.iter().rev().skip(16).take(16).copied().collect();
            let recent_mean: f32 = recent.iter().sum::<f32>() / recent.len() as f32;
            let older_mean: f32 = older.iter().sum::<f32>() / older.len() as f32;
            features.push(recent_mean - older_mean); // Trend
            features.push(recent_mean / older_mean.max(1e-8)); // Ratio
        } else {
            features.push(0.0);
            features.push(1.0);
        }
        features.push(self.step as f32 / 10000.0); // Normalized step

        // Gradient features (8)
        let grad_stats = self.gradient_statistics();
        features.push(self.gradient_norm);
        features.push(grad_stats.mean as f32);
        features.push(grad_stats.std as f32);
        features.push(grad_stats.min as f32);
        features.push(grad_stats.max as f32);
        features.push(self.gradient_norm.log10().max(-10.0)); // Log scale
        features.push((self.gradient_norm / grad_stats.mean.max(1e-8) as f32).min(10.0)); // Relative
        features.push(grad_stats.std as f32 / grad_stats.mean.max(1e-8) as f32); // CV

        // Optimizer features (8)
        features.push(self.optimizer_state_summary.momentum_mean);
        features.push(self.optimizer_state_summary.momentum_std);
        features.push(self.optimizer_state_summary.variance_mean);
        features.push(self.optimizer_state_summary.variance_std);
        features.push(self.optimizer_state_summary.effective_lr.log10().max(-10.0));
        features.push(self.optimizer_state_summary.beta1_power);
        features.push(self.optimizer_state_summary.beta2_power);
        features.push(0.0); // Reserved

        // Phase features (8)
        features.push(match self.current_phase {
            crate::Phase::Warmup => 0.0,
            crate::Phase::Full => 1.0,
            crate::Phase::Predict => 2.0,
            crate::Phase::Correct => 3.0,
        });
        features.push(self.phase_step as f32 / 100.0);
        // Pad to 64 features
        while features.len() < 64 {
            features.push(0.0);
        }

        features
    }
}

/// Trait for encoding training state into compact representations.
///
/// Implementations transform the full training state into a fixed-size
/// latent vector suitable for the dynamics predictor.
pub trait StateEncoder: Send + Sync {
    /// The encoded state type.
    type EncodedState: Clone + Send;

    /// Encodes the training state into a compact representation.
    ///
    /// # Arguments
    ///
    /// * `state` - The training state to encode
    ///
    /// # Returns
    ///
    /// The encoded state representation.
    fn encode(&self, state: &TrainingState) -> Self::EncodedState;

    /// Decodes a predicted state delta back to weight changes.
    ///
    /// # Arguments
    ///
    /// * `encoded_delta` - The predicted change in encoded state
    ///
    /// # Returns
    ///
    /// The decoded weight delta.
    fn decode_delta(&self, encoded_delta: &Self::EncodedState) -> WeightDelta;

    /// Returns the dimension of the encoded state.
    fn encoding_dim(&self) -> usize;
}

/// Represents a change in model weights.
///
/// Can be applied to model parameters to update them according to
/// a predicted or computed gradient step.
#[derive(Debug, Clone)]
pub struct WeightDelta {
    /// Per-parameter deltas, keyed by parameter name.
    pub deltas: std::collections::HashMap<String, Vec<f32>>,

    /// Global scale factor for all deltas.
    pub scale: f32,

    /// Metadata about the delta computation.
    pub metadata: WeightDeltaMetadata,
}

/// Metadata about a weight delta.
#[derive(Debug, Clone, Default)]
pub struct WeightDeltaMetadata {
    /// Whether this delta was predicted (vs computed).
    pub is_predicted: bool,
    /// Confidence in the prediction (if predicted).
    pub confidence: Option<f32>,
    /// Source phase.
    pub source_phase: Option<crate::Phase>,
    /// Number of steps this delta represents.
    pub num_steps: usize,
}

impl WeightDelta {
    /// Creates a new empty weight delta.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            deltas: std::collections::HashMap::new(),
            scale: 1.0,
            metadata: WeightDeltaMetadata::default(),
        }
    }

    /// Creates a weight delta with the given per-parameter deltas.
    #[must_use]
    pub fn new(deltas: std::collections::HashMap<String, Vec<f32>>) -> Self {
        Self {
            deltas,
            scale: 1.0,
            metadata: WeightDeltaMetadata::default(),
        }
    }

    /// Scales all deltas by the given factor.
    pub fn scale_by(&mut self, factor: f32) {
        self.scale *= factor;
    }

    /// Creates a scaled identity-like delta.
    ///
    /// This creates an empty delta with a specific scale factor,
    /// useful for placeholder corrections.
    ///
    /// # Arguments
    ///
    /// * `scale` - The scale factor for the delta
    #[must_use]
    pub fn scaled_identity(scale: f32) -> Self {
        Self {
            deltas: std::collections::HashMap::new(),
            scale,
            metadata: WeightDeltaMetadata {
                is_predicted: true,
                confidence: Some(1.0),
                ..Default::default()
            },
        }
    }
}

/// Default state encoder using linear projection.
///
/// Projects the feature vector to a fixed-size latent space using
/// a learned linear transformation.
pub struct LinearStateEncoder {
    _feature_dim: usize,
    latent_dim: usize,
    // Projection matrix would be stored here
}

impl LinearStateEncoder {
    /// Creates a new linear state encoder.
    ///
    /// # Arguments
    ///
    /// * `latent_dim` - Dimension of the encoded state
    #[must_use]
    pub fn new(latent_dim: usize) -> Self {
        Self {
            _feature_dim: 32, // From TrainingState::compute_features
            latent_dim,
        }
    }
}

impl StateEncoder for LinearStateEncoder {
    type EncodedState = Vec<f32>;

    fn encode(&self, state: &TrainingState) -> Self::EncodedState {
        // Placeholder: just return features padded/truncated to latent_dim
        let features = state.compute_features();
        let mut encoded = vec![0.0; self.latent_dim];
        for (i, &f) in features.iter().enumerate() {
            if i < self.latent_dim {
                encoded[i] = f;
            }
        }
        encoded
    }

    fn decode_delta(&self, _encoded_delta: &Self::EncodedState) -> WeightDelta {
        // Placeholder implementation
        WeightDelta::empty()
    }

    fn encoding_dim(&self) -> usize {
        self.latent_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ring_buffer_push_and_iter() {
        let mut buf: RingBuffer<f32, 4> = RingBuffer::new();
        buf.push(1.0);
        buf.push(2.0);
        buf.push(3.0);

        let values: Vec<f32> = buf.iter().cloned().collect();
        assert_eq!(values, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_ring_buffer_overflow() {
        let mut buf: RingBuffer<i32, 3> = RingBuffer::new();
        buf.push(1);
        buf.push(2);
        buf.push(3);
        buf.push(4); // Overwrites 1

        let values: Vec<i32> = buf.iter().cloned().collect();
        assert_eq!(values, vec![2, 3, 4]);
    }

    #[test]
    fn test_training_state_record_step() {
        let mut state = TrainingState::new();
        state.record_step(2.5, 1.0);
        state.record_step(2.3, 0.9);

        assert_eq!(state.step, 2);
        assert!((state.loss - 2.3).abs() < f32::EPSILON);
        assert_eq!(state.loss_history.len(), 2);
    }

    #[test]
    fn test_compute_features() {
        let mut state = TrainingState::new();
        state.record_step(2.5, 1.0);
        state.record_step(2.3, 0.9);

        let features = state.compute_features();
        assert!(features.len() >= 32);
    }

    #[test]
    fn test_multi_scale_ema() {
        let mut ema = MultiScaleEMA::new();
        ema.update(2.0);
        assert_eq!(ema.fast(), 2.0);
        for _ in 0..20 {
            ema.update(1.0);
        }
        assert!(ema.is_warm());
    }

    #[test]
    fn test_running_stats() {
        let mut stats = RunningStats::new();
        for i in 1..=5 {
            stats.update(i as f64);
        }
        assert!((stats.mean() - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_enhanced_features_64() {
        let mut state = TrainingState::new();
        for i in 0..50 {
            state.record_step(2.0 - i as f32 * 0.01, 1.0);
        }
        let features = state.compute_features();
        assert_eq!(features.len(), 64);
    }

    #[test]
    fn test_loss_ema_integration() {
        let mut state = TrainingState::new();
        state.record_step(3.0, 1.0);
        assert_eq!(state.loss_ema.fast(), 3.0);
        state.record_step(2.0, 1.0);
        assert!(state.loss_ema.fast() < state.loss_ema.slow());
    }
}
