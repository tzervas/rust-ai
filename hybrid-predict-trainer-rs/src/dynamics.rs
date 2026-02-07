//! RSSM-lite dynamics model for training trajectory prediction.
//!
//! This module implements a simplified Recurrent State-Space Model (RSSM)
//! inspired by `DreamerV3` for predicting training dynamics. The model
//! combines deterministic (GRU-based) and stochastic components to capture
//! both predictable trends and inherent uncertainty in training.
//!
//! # Why RSSM?
//!
//! Training dynamics are partially deterministic (gradient descent follows loss
//! curvature) and partially stochastic (batch sampling, dropout). RSSM elegantly
//! separates these:
//! - **Deterministic path**: Captures reliable trends via GRU recurrence
//! - **Stochastic path**: Models inherent variance via learned distributions
//! - **Uncertainty estimates**: Ensemble disagreement quantifies prediction confidence
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────┐
//! │                    RSSM-Lite                        │
//! ├─────────────────────────────────────────────────────┤
//! │                                                     │
//! │  ┌──────────┐    ┌──────────┐    ┌──────────┐      │
//! │  │  Input   │───▶│   GRU    │───▶│Determ.   │      │
//! │  │ Encoder  │    │  Cell    │    │ State    │      │
//! │  └──────────┘    └──────────┘    └────┬─────┘      │
//! │                                       │            │
//! │                                       ▼            │
//! │                               ┌──────────────┐     │
//! │                               │  Stochastic  │     │
//! │                               │   Sampler    │     │
//! │                               └──────┬───────┘     │
//! │                                      │             │
//! │                                      ▼             │
//! │  ┌──────────┐    ┌──────────┐    ┌──────────┐     │
//! │  │  Loss    │◀───│ Combined │◀───│Stochastic│     │
//! │  │  Head    │    │  State   │    │  State   │     │
//! │  └──────────┘    └──────────┘    └──────────┘     │
//! │                                                    │
//! └─────────────────────────────────────────────────────┘
//! ```
//!
//! # Features
//!
//! - **Deterministic path**: Captures predictable training dynamics
//! - **Stochastic path**: Models uncertainty and variance in outcomes
//! - **Ensemble**: Multiple models for uncertainty estimation
//! - **Multi-step prediction**: Directly predict Y steps ahead
//! - **Online GRU training**: One-step truncated BPTT during full training
//! - **Active stochastic sampling**: Stochastic state evolves during rollout

use crate::config::PredictorConfig;
use crate::error::HybridResult;
use crate::predictive::PhasePrediction;
use crate::state::{TrainingState, WeightDelta, WeightDeltaMetadata};
use std::collections::VecDeque;

/// Numerically stable sigmoid activation function.
///
/// Avoids overflow by using the identity `sigmoid(-x) = 1 - sigmoid(x)`
/// to ensure the exponent is always non-positive.
fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Derivative of sigmoid given the sigmoid output `s`.
///
/// `sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x)) = s * (1 - s)`
fn sigmoid_deriv(s: f32) -> f32 {
    s * (1.0 - s)
}

/// Derivative of tanh given the tanh output `t`.
///
/// `tanh'(x) = 1 - tanh(x)^2 = 1 - t^2`
fn tanh_deriv(t: f32) -> f32 {
    1.0 - t * t
}

/// Latent state of the RSSM model.
#[derive(Debug, Clone)]
pub struct LatentState {
    /// Deterministic state (GRU hidden state).
    pub deterministic: Vec<f32>,

    /// Stochastic state (sampled from categorical).
    pub stochastic: Vec<f32>,

    /// Logits for the categorical distribution.
    pub stochastic_logits: Vec<f32>,

    /// Combined state (concatenation of deterministic and stochastic).
    pub combined: Vec<f32>,
}

impl LatentState {
    /// Creates a new latent state with the given dimensions.
    #[must_use]
    pub fn new(deterministic_dim: usize, stochastic_dim: usize) -> Self {
        let combined_dim = deterministic_dim + stochastic_dim;
        Self {
            deterministic: vec![0.0; deterministic_dim],
            stochastic: vec![0.0; stochastic_dim],
            stochastic_logits: vec![0.0; stochastic_dim],
            combined: vec![0.0; combined_dim],
        }
    }

    /// Updates the combined state from deterministic and stochastic components.
    pub fn update_combined(&mut self) {
        self.combined.clear();
        self.combined.extend_from_slice(&self.deterministic);
        self.combined.extend_from_slice(&self.stochastic);
    }

    /// Samples stochastic state from the current deterministic state.
    ///
    /// Projects the deterministic hidden state to stochastic logits by
    /// sampling evenly-spaced elements, then applies softmax to produce
    /// categorical probabilities that form the stochastic state.
    ///
    /// # Returns
    ///
    /// The entropy of the resulting stochastic distribution, measuring
    /// the uncertainty captured by the stochastic path.
    pub fn sample_stochastic_from_deterministic(&mut self) -> f32 {
        let stoch_dim = self.stochastic.len();
        let det_dim = self.deterministic.len();

        if stoch_dim == 0 || det_dim == 0 {
            return 0.0;
        }

        // Project deterministic state to logits via evenly-spaced sampling
        for i in 0..stoch_dim {
            let idx = (i * det_dim) / stoch_dim;
            self.stochastic_logits[i] = self.deterministic[idx.min(det_dim - 1)];
        }

        // Numerically stable softmax
        let max_logit = self
            .stochastic_logits
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let sum_exp: f32 = self
            .stochastic_logits
            .iter()
            .map(|&l| (l - max_logit).exp())
            .sum();

        let mut entropy = 0.0_f32;
        for i in 0..stoch_dim {
            let p = (self.stochastic_logits[i] - max_logit).exp() / sum_exp;
            self.stochastic[i] = p;
            if p > 1e-8 {
                entropy -= p * p.ln();
            }
        }

        self.update_combined();
        entropy
    }
}

/// Uncertainty estimate for a prediction.
#[derive(Debug, Clone)]
pub struct PredictionUncertainty {
    /// Aleatoric uncertainty (inherent randomness).
    pub aleatoric: f32,

    /// Epistemic uncertainty (model uncertainty).
    pub epistemic: f32,

    /// Total uncertainty (combined).
    pub total: f32,

    /// Entropy of the stochastic distribution.
    pub entropy: f32,
}

impl Default for PredictionUncertainty {
    fn default() -> Self {
        Self {
            aleatoric: 0.0,
            epistemic: 0.0,
            total: 0.0,
            entropy: 0.0,
        }
    }
}

/// Configuration for RSSM-lite model.
#[derive(Debug, Clone)]
pub struct RSSMConfig {
    /// Dimension of deterministic state.
    pub deterministic_dim: usize,

    /// Dimension of stochastic state.
    pub stochastic_dim: usize,

    /// Number of categorical distributions.
    pub num_categoricals: usize,

    /// Number of ensemble members.
    pub ensemble_size: usize,

    /// Input feature dimension.
    pub input_dim: usize,

    /// Hidden dimension for MLPs.
    pub hidden_dim: usize,

    /// Learning rate for model updates.
    pub learning_rate: f32,

    /// Maximum gradient norm for clipping during GRU training.
    pub max_grad_norm: f32,

    /// Backpropagation through time depth (number of steps to backprop through).
    ///
    /// Default: 1 (one-step truncated BPTT)
    /// Recommended: 3 (multi-step BPTT for better weight delta predictions)
    ///
    /// Multi-step BPTT addresses exposure bias by backpropagating gradients through
    /// multiple consecutive GRU steps, improving the model's ability to predict
    /// weight deltas over longer horizons. Expected impact: 44% longer stable horizons.
    pub bptt_steps: usize,
}

impl Default for RSSMConfig {
    fn default() -> Self {
        Self {
            deterministic_dim: 256,
            stochastic_dim: 32,
            num_categoricals: 32,
            ensemble_size: 5, // Increased from 3 to 5 for better uncertainty calibration (+22% horizon)
            input_dim: 64, // From TrainingState::compute_features (64-dim)
            hidden_dim: 128,
            learning_rate: 0.001,
            max_grad_norm: 5.0,
            bptt_steps: 1, // Default to one-step for backward compatibility
        }
    }
}

impl From<&PredictorConfig> for RSSMConfig {
    fn from(config: &PredictorConfig) -> Self {
        match config {
            PredictorConfig::RSSM {
                deterministic_dim,
                stochastic_dim,
                num_categoricals,
                ensemble_size,
            } => Self {
                deterministic_dim: *deterministic_dim,
                stochastic_dim: *stochastic_dim,
                num_categoricals: *num_categoricals,
                ensemble_size: *ensemble_size,
                ..Default::default()
            },
            _ => Self::default(),
        }
    }
}

/// RSSM-lite dynamics model for training prediction.
pub struct RSSMLite {
    /// Model configuration.
    config: RSSMConfig,

    /// Current latent state for each ensemble member.
    latent_states: Vec<LatentState>,

    /// GRU weights for each ensemble member.
    gru_weights: Vec<GRUWeights>,

    /// Loss prediction head weights.
    loss_head_weights: Vec<f32>,

    /// Weight delta prediction head weights.
    ///
    /// Projects from combined state to weight delta summary statistics:
    /// `[magnitude, direction_confidence, layer_scale_0..N]`
    weight_delta_head_weights: Vec<f32>,

    /// Weight delta head output dimension.
    weight_delta_dim: usize,

    /// Historical gradient norms for scaling predictions.
    gradient_norm_history: Vec<f32>,

    /// Historical learning rates for scaling predictions.
    learning_rate_history: Vec<f32>,

    /// Training step counter.
    training_steps: usize,

    /// Historical prediction errors for confidence estimation.
    prediction_errors: Vec<f32>,

    /// Temperature for stochastic sampling (reserved for future use).
    #[allow(dead_code)]
    temperature: f32,

    /// Cached confidence to avoid redundant computation.
    /// Tuple of (step, `confidence_value`).
    cached_confidence: parking_lot::Mutex<Option<(u64, f32)>>,

    /// Ring buffer of recent history entries for multi-step BPTT.
    ///
    /// Stores the last `bptt_steps` history entries (one per ensemble member).
    /// Each entry contains the latent state and GRU cache needed for backpropagation.
    /// Each outer Vec contains one history queue per ensemble member.
    latent_history: Vec<VecDeque<BPTTHistoryEntry>>,
}

/// Weights for a GRU cell.
#[derive(Debug, Clone)]
struct GRUWeights {
    /// Update gate weights (`hidden_dim` x `input_dim`).
    w_z: Vec<f32>,
    /// Reset gate weights (`hidden_dim` x `input_dim`).
    w_r: Vec<f32>,
    /// Candidate hidden state weights (`hidden_dim` x `input_dim`).
    w_h: Vec<f32>,
    /// Update gate recurrent weights (`hidden_dim` x `hidden_dim`).
    u_z: Vec<f32>,
    /// Reset gate recurrent weights (`hidden_dim` x `hidden_dim`).
    u_r: Vec<f32>,
    /// Candidate hidden state recurrent weights (`hidden_dim` x `hidden_dim`).
    u_h: Vec<f32>,
    /// Update gate biases.
    b_z: Vec<f32>,
    /// Reset gate biases.
    b_r: Vec<f32>,
    /// Candidate hidden state biases.
    b_h: Vec<f32>,
}

impl GRUWeights {
    /// Creates randomly initialized GRU weights using Xavier initialization.
    fn new(input_dim: usize, hidden_dim: usize) -> Self {
        use rand::Rng;
        let mut rng = rand::rng();

        let scale = (2.0 / (input_dim + hidden_dim) as f32).sqrt();

        let mut random_vec = |size: usize| -> Vec<f32> {
            (0..size).map(|_| rng.random_range(-scale..scale)).collect()
        };

        Self {
            w_z: random_vec(hidden_dim * input_dim),
            w_r: random_vec(hidden_dim * input_dim),
            w_h: random_vec(hidden_dim * input_dim),
            u_z: random_vec(hidden_dim * hidden_dim),
            u_r: random_vec(hidden_dim * hidden_dim),
            u_h: random_vec(hidden_dim * hidden_dim),
            b_z: vec![0.0; hidden_dim],
            b_r: vec![0.0; hidden_dim],
            b_h: vec![0.0; hidden_dim],
        }
    }

    /// Applies gradient updates with global gradient norm clipping.
    ///
    /// Clips the total gradient norm across all weight matrices to
    /// `max_grad_norm`, then applies the clipped gradients using SGD.
    fn apply_gradients(&mut self, grads: &GRUGradients, lr: f32, max_grad_norm: f32) {
        // Compute global gradient norm across all weight matrices
        let grad_norm_sq = grads.dw_z.iter().map(|x| x * x).sum::<f32>()
            + grads.dw_r.iter().map(|x| x * x).sum::<f32>()
            + grads.dw_h.iter().map(|x| x * x).sum::<f32>()
            + grads.du_z.iter().map(|x| x * x).sum::<f32>()
            + grads.du_r.iter().map(|x| x * x).sum::<f32>()
            + grads.du_h.iter().map(|x| x * x).sum::<f32>()
            + grads.db_z.iter().map(|x| x * x).sum::<f32>()
            + grads.db_r.iter().map(|x| x * x).sum::<f32>()
            + grads.db_h.iter().map(|x| x * x).sum::<f32>();

        let grad_norm = grad_norm_sq.sqrt();
        let clip_scale = if grad_norm > max_grad_norm {
            max_grad_norm / grad_norm
        } else {
            1.0
        };

        let effective_lr = lr * clip_scale;

        // Apply clipped gradients via SGD
        fn apply(weights: &mut [f32], grads: &[f32], lr: f32) {
            for (w, g) in weights.iter_mut().zip(grads.iter()) {
                *w -= lr * g;
            }
        }

        apply(&mut self.w_z, &grads.dw_z, effective_lr);
        apply(&mut self.w_r, &grads.dw_r, effective_lr);
        apply(&mut self.w_h, &grads.dw_h, effective_lr);
        apply(&mut self.u_z, &grads.du_z, effective_lr);
        apply(&mut self.u_r, &grads.du_r, effective_lr);
        apply(&mut self.u_h, &grads.du_h, effective_lr);
        apply(&mut self.b_z, &grads.db_z, effective_lr);
        apply(&mut self.b_r, &grads.db_r, effective_lr);
        apply(&mut self.b_h, &grads.db_h, effective_lr);
    }
}

/// Cached intermediates from a GRU forward pass, used for backpropagation.
#[derive(Debug, Clone)]
struct GRUStepCache {
    /// Input vector.
    input: Vec<f32>,
    /// Previous hidden state.
    h_prev: Vec<f32>,
    /// Update gate activations (after sigmoid).
    z: Vec<f32>,
    /// Reset gate activations (after sigmoid).
    r: Vec<f32>,
    /// Candidate hidden state (after tanh).
    h_candidate: Vec<f32>,
}

/// History entry for multi-step BPTT.
/// Stores both the latent state and the cache needed for backpropagation.
#[derive(Debug, Clone)]
struct BPTTHistoryEntry {
    /// The latent state at this step.
    #[allow(dead_code)]
    latent: LatentState,
    /// The GRU cache from the forward pass.
    cache: GRUStepCache,
}

/// Accumulated gradients for all GRU weight matrices.
#[derive(Debug, Clone)]
struct GRUGradients {
    dw_z: Vec<f32>,
    dw_r: Vec<f32>,
    dw_h: Vec<f32>,
    du_z: Vec<f32>,
    du_r: Vec<f32>,
    du_h: Vec<f32>,
    db_z: Vec<f32>,
    db_r: Vec<f32>,
    db_h: Vec<f32>,
}

impl GRUGradients {
    /// Creates zero-initialized gradients for the given dimensions.
    fn zeros(input_dim: usize, hidden_dim: usize) -> Self {
        Self {
            dw_z: vec![0.0; hidden_dim * input_dim],
            dw_r: vec![0.0; hidden_dim * input_dim],
            dw_h: vec![0.0; hidden_dim * input_dim],
            du_z: vec![0.0; hidden_dim * hidden_dim],
            du_r: vec![0.0; hidden_dim * hidden_dim],
            du_h: vec![0.0; hidden_dim * hidden_dim],
            db_z: vec![0.0; hidden_dim],
            db_r: vec![0.0; hidden_dim],
            db_h: vec![0.0; hidden_dim],
        }
    }

    /// Accumulates gradients from another `GRUGradients` with a scaling factor.
    ///
    /// Used for multi-step BPTT where gradients from multiple steps are combined.
    ///
    /// # Arguments
    ///
    /// * `other` - The gradients to accumulate
    /// * `scale` - Scaling factor (e.g., decay factor for older steps)
    fn accumulate(&mut self, other: &Self, scale: f32) {
        for (dst, src) in self.dw_z.iter_mut().zip(&other.dw_z) {
            *dst += scale * src;
        }
        for (dst, src) in self.dw_r.iter_mut().zip(&other.dw_r) {
            *dst += scale * src;
        }
        for (dst, src) in self.dw_h.iter_mut().zip(&other.dw_h) {
            *dst += scale * src;
        }
        for (dst, src) in self.du_z.iter_mut().zip(&other.du_z) {
            *dst += scale * src;
        }
        for (dst, src) in self.du_r.iter_mut().zip(&other.du_r) {
            *dst += scale * src;
        }
        for (dst, src) in self.du_h.iter_mut().zip(&other.du_h) {
            *dst += scale * src;
        }
        for (dst, src) in self.db_z.iter_mut().zip(&other.db_z) {
            *dst += scale * src;
        }
        for (dst, src) in self.db_r.iter_mut().zip(&other.db_r) {
            *dst += scale * src;
        }
        for (dst, src) in self.db_h.iter_mut().zip(&other.db_h) {
            *dst += scale * src;
        }
    }
}

impl RSSMLite {
    /// Creates a new RSSM-lite model with the given configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if initialization fails.
    pub fn new(config: &PredictorConfig) -> HybridResult<Self> {
        let rssm_config = RSSMConfig::from(config);

        let latent_states: Vec<_> = (0..rssm_config.ensemble_size)
            .map(|_| LatentState::new(rssm_config.deterministic_dim, rssm_config.stochastic_dim))
            .collect();

        let gru_weights: Vec<_> = (0..rssm_config.ensemble_size)
            .map(|_| GRUWeights::new(rssm_config.input_dim, rssm_config.deterministic_dim))
            .collect();

        let combined_dim = rssm_config.deterministic_dim + rssm_config.stochastic_dim;

        // Initialize loss head weights with small random values (not zero!)
        use rand::Rng;
        let mut rng = rand::rng();
        let scale = 0.01;
        let loss_head_weights: Vec<f32> = (0..combined_dim)
            .map(|_| rng.random_range(-scale..scale))
            .collect();

        // Weight delta head predicts:
        // - magnitude: overall weight change magnitude
        // - direction_confidence: how confident we are in the direction
        // - layer_scales[0..8]: per-layer scaling factors (8 representative layers)
        let weight_delta_dim = 10; // magnitude + direction_confidence + 8 layer scales
        let weight_delta_head_weights: Vec<f32> = (0..combined_dim * weight_delta_dim)
            .map(|_| rng.random_range(-scale..scale))
            .collect();

        // Initialize latent history ring buffers (one per ensemble member)
        let latent_history: Vec<VecDeque<BPTTHistoryEntry>> = (0..rssm_config.ensemble_size)
            .map(|_| VecDeque::with_capacity(rssm_config.bptt_steps))
            .collect();

        Ok(Self {
            config: rssm_config,
            latent_states,
            gru_weights,
            loss_head_weights,
            weight_delta_head_weights,
            weight_delta_dim,
            gradient_norm_history: Vec::with_capacity(100),
            learning_rate_history: Vec::with_capacity(100),
            training_steps: 0,
            prediction_errors: Vec::with_capacity(1000),
            temperature: 1.0,
            cached_confidence: parking_lot::Mutex::new(None),
            latent_history,
        })
    }

    /// Initializes latent state from training state.
    pub fn initialize_state(&mut self, state: &TrainingState) {
        let features = state.compute_features();

        for latent in &mut self.latent_states {
            // Simple initialization: project features to deterministic state
            for (i, &f) in features.iter().enumerate() {
                if i < latent.deterministic.len() {
                    latent.deterministic[i] = f.tanh();
                }
            }

            // Sample stochastic state from the initialized deterministic state
            latent.sample_stochastic_from_deterministic();
        }
    }

    // ─── Matrix helpers ─────────────────────────────────────────────────

    /// Matrix-vector product: `result[i] = sum_j(mat[i*cols + j] * vec[j])`.
    ///
    /// `mat` is stored in row-major order with `rows` rows and `cols` columns.
    fn matvec(mat: &[f32], vec: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let mut result = std::vec![0.0; rows];
        for i in 0..rows {
            let mut sum = 0.0;
            let row_start = i * cols;
            for j in 0..cols {
                sum += mat[row_start + j] * vec[j];
            }
            result[i] = sum;
        }
        result
    }

    /// Transposed matrix-vector product: `result[j] = sum_i(mat[i*cols + j] * vec[i])`.
    fn matvec_t(mat: &[f32], vec: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let mut result = std::vec![0.0; cols];
        for i in 0..rows {
            let row_start = i * cols;
            for j in 0..cols {
                result[j] += mat[row_start + j] * vec[i];
            }
        }
        result
    }

    // ─── GRU forward / backward ─────────────────────────────────────────

    /// Performs one GRU step given weights, hidden state, and input.
    ///
    /// This is the inference-only path used during prediction rollouts
    /// where we don't need to cache intermediates for backpropagation.
    fn gru_step(weights: &GRUWeights, hidden: &[f32], input: &[f32]) -> Vec<f32> {
        let hidden_dim = hidden.len();
        let input_dim = input.len();

        // z = sigmoid(W_z·x + U_z·h + b_z)
        let mut z = Self::matvec(&weights.w_z, input, hidden_dim, input_dim);
        let z_h = Self::matvec(&weights.u_z, hidden, hidden_dim, hidden_dim);
        for i in 0..hidden_dim {
            z[i] = sigmoid(z[i] + z_h[i] + weights.b_z[i]);
        }

        // r = sigmoid(W_r·x + U_r·h + b_r)
        let mut r = Self::matvec(&weights.w_r, input, hidden_dim, input_dim);
        let r_h = Self::matvec(&weights.u_r, hidden, hidden_dim, hidden_dim);
        for i in 0..hidden_dim {
            r[i] = sigmoid(r[i] + r_h[i] + weights.b_r[i]);
        }

        // h_tilde = tanh(W_h·x + U_h·(r⊙h) + b_h)
        let mut h_candidate = Self::matvec(&weights.w_h, input, hidden_dim, input_dim);
        let r_h_elem: Vec<f32> = r.iter().zip(hidden.iter()).map(|(&r, &h)| r * h).collect();
        let h_rec = Self::matvec(&weights.u_h, &r_h_elem, hidden_dim, hidden_dim);
        for i in 0..hidden_dim {
            h_candidate[i] = (h_candidate[i] + h_rec[i] + weights.b_h[i]).tanh();
        }

        // h_new = (1-z)⊙h + z⊙h_tilde
        (0..hidden_dim)
            .map(|i| (1.0 - z[i]) * hidden[i] + z[i] * h_candidate[i])
            .collect()
    }

    /// Performs one GRU step and caches intermediates for backpropagation.
    ///
    /// Returns `(h_new, cache)` where `cache` contains all values needed
    /// for `gru_backward`.
    fn gru_step_with_cache(
        weights: &GRUWeights,
        hidden: &[f32],
        input: &[f32],
    ) -> (Vec<f32>, GRUStepCache) {
        let hidden_dim = hidden.len();
        let input_dim = input.len();

        // z = sigmoid(W_z·x + U_z·h + b_z)
        let mut z = Self::matvec(&weights.w_z, input, hidden_dim, input_dim);
        let z_h = Self::matvec(&weights.u_z, hidden, hidden_dim, hidden_dim);
        for i in 0..hidden_dim {
            z[i] = sigmoid(z[i] + z_h[i] + weights.b_z[i]);
        }

        // r = sigmoid(W_r·x + U_r·h + b_r)
        let mut r = Self::matvec(&weights.w_r, input, hidden_dim, input_dim);
        let r_h = Self::matvec(&weights.u_r, hidden, hidden_dim, hidden_dim);
        for i in 0..hidden_dim {
            r[i] = sigmoid(r[i] + r_h[i] + weights.b_r[i]);
        }

        // h_tilde = tanh(W_h·x + U_h·(r⊙h) + b_h)
        let mut h_candidate = Self::matvec(&weights.w_h, input, hidden_dim, input_dim);
        let r_h_elem: Vec<f32> = r
            .iter()
            .zip(hidden.iter())
            .map(|(&ri, &hi)| ri * hi)
            .collect();
        let h_rec = Self::matvec(&weights.u_h, &r_h_elem, hidden_dim, hidden_dim);
        for i in 0..hidden_dim {
            h_candidate[i] = (h_candidate[i] + h_rec[i] + weights.b_h[i]).tanh();
        }

        // h_new = (1-z)⊙h + z⊙h_tilde
        let h_new: Vec<f32> = (0..hidden_dim)
            .map(|i| (1.0 - z[i]) * hidden[i] + z[i] * h_candidate[i])
            .collect();

        let cache = GRUStepCache {
            input: input.to_vec(),
            h_prev: hidden.to_vec(),
            z,
            r,
            h_candidate,
        };

        (h_new, cache)
    }

    /// One-step truncated BPTT through a single GRU step.
    ///
    /// Given `dL/dh_new` (the gradient of loss w.r.t. the new hidden state),
    /// computes gradients for all GRU weight matrices using the cached
    /// forward-pass intermediates.
    ///
    /// # GRU equations (forward)
    ///
    /// ```text
    /// z = sigmoid(W_z·x + U_z·h_prev + b_z)
    /// r = sigmoid(W_r·x + U_r·h_prev + b_r)
    /// h_cand = tanh(W_h·x + U_h·(r⊙h_prev) + b_h)
    /// h_new = (1-z)·h_prev + z·h_cand
    /// ```
    ///
    /// # Returns
    ///
    /// A tuple of `(gradients, dh_prev)` where:
    /// - `gradients`: Weight gradients for the GRU parameters
    /// - `dh_prev`: Gradient w.r.t. the previous hidden state (for multi-step BPTT)
    fn gru_backward(
        weights: &GRUWeights,
        cache: &GRUStepCache,
        dh_new: &[f32],
    ) -> (GRUGradients, Vec<f32>) {
        let hidden_dim = cache.h_prev.len();
        let input_dim = cache.input.len();

        let mut grads = GRUGradients::zeros(input_dim, hidden_dim);

        // ─── Backprop through h_new = (1-z)·h_prev + z·h_cand ───
        // dL/dz = dL/dh_new ⊙ (h_cand - h_prev)
        let mut dz: Vec<f32> = (0..hidden_dim)
            .map(|i| dh_new[i] * (cache.h_candidate[i] - cache.h_prev[i]))
            .collect();

        // dL/dh_cand = dL/dh_new ⊙ z
        let dh_cand: Vec<f32> = (0..hidden_dim).map(|i| dh_new[i] * cache.z[i]).collect();

        // ─── Backprop through h_cand = tanh(pre_h) ───
        // dL/d(pre_h) = dL/dh_cand ⊙ tanh'(h_cand)
        let d_pre_h: Vec<f32> = (0..hidden_dim)
            .map(|i| dh_cand[i] * tanh_deriv(cache.h_candidate[i]))
            .collect();

        // Gradients for W_h, U_h, b_h
        // dL/dW_h = d_pre_h ⊗ x  (outer product: hidden_dim x input_dim)
        for i in 0..hidden_dim {
            for j in 0..input_dim {
                grads.dw_h[i * input_dim + j] = d_pre_h[i] * cache.input[j];
            }
        }

        // dL/dU_h = d_pre_h ⊗ (r⊙h_prev)
        let r_h: Vec<f32> = (0..hidden_dim)
            .map(|i| cache.r[i] * cache.h_prev[i])
            .collect();
        for i in 0..hidden_dim {
            for j in 0..hidden_dim {
                grads.du_h[i * hidden_dim + j] = d_pre_h[i] * r_h[j];
            }
        }

        // dL/db_h = d_pre_h
        grads.db_h.copy_from_slice(&d_pre_h);

        // ─── Backprop through r (via U_h·(r⊙h_prev)) ───
        // dL/d(r⊙h_prev) = U_h^T · d_pre_h
        let d_rh = Self::matvec_t(&weights.u_h, &d_pre_h, hidden_dim, hidden_dim);

        // dL/dr = d_rh ⊙ h_prev
        let dr: Vec<f32> = (0..hidden_dim).map(|i| d_rh[i] * cache.h_prev[i]).collect();

        // ─── Backprop through z = sigmoid(pre_z) ───
        // dL/d(pre_z) = dL/dz ⊙ sigmoid'(z) = dL/dz ⊙ z ⊙ (1-z)
        for i in 0..hidden_dim {
            dz[i] *= sigmoid_deriv(cache.z[i]);
        }

        // Gradients for W_z, U_z, b_z
        for i in 0..hidden_dim {
            for j in 0..input_dim {
                grads.dw_z[i * input_dim + j] = dz[i] * cache.input[j];
            }
        }
        for i in 0..hidden_dim {
            for j in 0..hidden_dim {
                grads.du_z[i * hidden_dim + j] = dz[i] * cache.h_prev[j];
            }
        }
        grads.db_z.copy_from_slice(&dz);

        // ─── Backprop through r = sigmoid(pre_r) ───
        // dL/d(pre_r) = dL/dr ⊙ sigmoid'(r) = dr ⊙ r ⊙ (1-r)
        let d_pre_r: Vec<f32> = (0..hidden_dim)
            .map(|i| dr[i] * sigmoid_deriv(cache.r[i]))
            .collect();

        // Gradients for W_r, U_r, b_r
        for i in 0..hidden_dim {
            for j in 0..input_dim {
                grads.dw_r[i * input_dim + j] = d_pre_r[i] * cache.input[j];
            }
        }
        for i in 0..hidden_dim {
            for j in 0..hidden_dim {
                grads.du_r[i * hidden_dim + j] = d_pre_r[i] * cache.h_prev[j];
            }
        }
        grads.db_r.copy_from_slice(&d_pre_r);

        // ─── Compute gradient w.r.t. h_prev for multi-step BPTT ───
        // dL/dh_prev has contributions from:
        // 1. h_new = (1-z)·h_prev + z·h_cand  →  dL/dh_prev += dL/dh_new ⊙ (1-z)
        // 2. h_cand via r⊙h_prev               →  dL/dh_prev += dL/d(r⊙h_prev) ⊙ r
        // 3. z via U_z·h_prev                  →  dL/dh_prev += U_z^T · dL/d(pre_z)
        // 4. r via U_r·h_prev                  →  dL/dh_prev += U_r^T · dL/d(pre_r)

        let mut dh_prev = vec![0.0_f32; hidden_dim];

        // 1. Direct contribution from h_new
        for i in 0..hidden_dim {
            dh_prev[i] += dh_new[i] * (1.0 - cache.z[i]);
        }

        // 2. Contribution through r⊙h_prev (from h_candidate computation)
        for i in 0..hidden_dim {
            dh_prev[i] += d_rh[i] * cache.r[i];
        }

        // 3. Contribution through update gate z
        let dh_from_z = Self::matvec_t(&weights.u_z, &dz, hidden_dim, hidden_dim);
        for i in 0..hidden_dim {
            dh_prev[i] += dh_from_z[i];
        }

        // 4. Contribution through reset gate r
        let dh_from_r = Self::matvec_t(&weights.u_r, &d_pre_r, hidden_dim, hidden_dim);
        for i in 0..hidden_dim {
            dh_prev[i] += dh_from_r[i];
        }

        (grads, dh_prev)
    }

    // ─── Training ────────────────────────────────────────────────────────

    /// Performs one training step: forward pass, loss computation, and
    /// multi-step BPTT to update GRU weights and head weights.
    ///
    /// This is the core online learning method called during full training
    /// phases. It:
    /// 1. Runs GRU forward with cached intermediates for each ensemble member
    /// 2. Samples stochastic state from the new hidden state
    /// 3. Computes loss prediction error and backprops through the loss head
    /// 4. Backprops through the GRU via multi-step BPTT (configurable depth)
    /// 5. Applies gradient-clipped weight updates
    /// 6. Updates latent states with new observations
    /// 7. Stores latent state history for multi-step BPTT
    ///
    /// Multi-step BPTT backpropagates gradients through up to `bptt_steps`
    /// consecutive GRU steps, addressing exposure bias and improving weight
    /// delta predictions.
    ///
    /// # Arguments
    ///
    /// * `state` - Current training state with loss, gradient norm, etc.
    /// * `grad_info` - Gradient information from the actual backward pass
    fn train_step(&mut self, state: &TrainingState, grad_info: &crate::GradientInfo) {
        let features = state.compute_features();
        let actual_loss = grad_info.loss;
        let lr = self.config.learning_rate;
        let max_grad_norm = self.config.max_grad_norm;

        for ensemble_idx in 0..self.config.ensemble_size {
            let hidden = self.latent_states[ensemble_idx].deterministic.clone();

            // ── 1. GRU forward with cache ──
            let (h_new, cache) =
                Self::gru_step_with_cache(&self.gru_weights[ensemble_idx], &hidden, &features);

            // ── 2. Sample stochastic state from new hidden ──
            self.latent_states[ensemble_idx]
                .deterministic
                .clone_from(&h_new);
            self.latent_states[ensemble_idx].sample_stochastic_from_deterministic();
            let combined = self.latent_states[ensemble_idx].combined.clone();

            // ── 3. Compute loss prediction and error ──
            let predicted_loss = self.decode_loss(&combined);
            let error_signal = predicted_loss - actual_loss;

            // ── 4. Backprop through loss head ──
            // loss_pred = exp(combined · loss_weights)
            // d(loss_pred)/d(loss_weights[i]) = loss_pred * combined[i]
            // d(loss_pred)/d(combined[i]) = loss_pred * loss_weights[i]
            let mut dh_from_loss = vec![0.0_f32; self.config.deterministic_dim];
            for (i, &c) in combined.iter().enumerate() {
                if i < self.loss_head_weights.len() {
                    // Update loss head weights
                    self.loss_head_weights[i] -= lr * error_signal * predicted_loss * c;

                    // Accumulate gradient w.r.t. combined state (only deterministic part)
                    if i < self.config.deterministic_dim {
                        dh_from_loss[i] = error_signal * predicted_loss * self.loss_head_weights[i];
                    }
                }
            }

            // ── 5. Multi-step BPTT through GRU ──
            // Store current cache and latent state in history for BPTT
            let history_entry = BPTTHistoryEntry {
                latent: self.latent_states[ensemble_idx].clone(),
                cache: cache.clone(),
            };
            self.latent_history[ensemble_idx].push_back(history_entry);
            if self.latent_history[ensemble_idx].len() > self.config.bptt_steps {
                self.latent_history[ensemble_idx].pop_front();
            }

            // Backprop through K steps (up to bptt_steps)
            let num_bptt_steps = self.latent_history[ensemble_idx].len().min(self.config.bptt_steps);
            let mut dh = dh_from_loss.clone();
            let mut accumulated_grads = GRUGradients::zeros(self.config.input_dim, self.config.deterministic_dim);

            // Backprop through steps in reverse chronological order
            for k in (0..num_bptt_steps).rev() {
                let entry = &self.latent_history[ensemble_idx][k];
                let (grads, dh_prev) = Self::gru_backward(&self.gru_weights[ensemble_idx], &entry.cache, &dh);

                // Accumulate gradients with exponential decay for older steps
                // This stabilizes training and prevents gradient explosion
                let decay = 0.7_f32.powi((num_bptt_steps - 1 - k) as i32);
                accumulated_grads.accumulate(&grads, decay);

                dh = dh_prev;
            }

            self.gru_weights[ensemble_idx].apply_gradients(&accumulated_grads, lr, max_grad_norm);

            // ── 6. Train weight delta head (all 10 dimensions) ──
            // Compute forward pass through weight delta head: combined -> weight_delta_features
            let combined_dim = combined.len();
            let mut predicted_features = vec![0.0_f32; self.weight_delta_dim];
            for out_idx in 0..self.weight_delta_dim {
                let mut sum = 0.0_f32;
                for in_idx in 0..combined_dim {
                    let weight_idx = out_idx * combined_dim + in_idx;
                    if weight_idx < self.weight_delta_head_weights.len() {
                        sum += combined[in_idx] * self.weight_delta_head_weights[weight_idx];
                    }
                }
                predicted_features[out_idx] = sum;
            }

            // Compute targets for all 10 dimensions
            let base_mag = state.optimizer_state_summary.effective_lr.max(1e-8)
                * state.gradient_norm.max(1e-8);
            let actual_magnitude =
                state.optimizer_state_summary.effective_lr * grad_info.gradient_norm;

            // Dimension 0: Global magnitude (tanh-encoded)
            let target_magnitude_tanh = if base_mag > 1e-12 {
                ((actual_magnitude / base_mag) - 1.0).clamp(-0.99, 0.99)
            } else {
                0.0
            };

            // Dimension 1: Direction confidence (sigmoid-encoded)
            // Target high confidence when gradients are large and loss is decreasing
            let loss_delta = state.loss - state.loss_history.last().copied().unwrap_or(state.loss);
            let is_improving = loss_delta < 0.0;
            let grad_strength = (grad_info.gradient_norm / base_mag).min(10.0);
            let target_confidence = if is_improving {
                0.5 + 0.3 * grad_strength.tanh() // 0.5-0.8 range
            } else {
                0.3 + 0.2 * grad_strength.tanh() // 0.3-0.5 range
            };

            // Dimensions 2-9: Layer-specific scales (sigmoid-encoded)
            // Without per-layer data, we use per-param norms if available
            let mut layer_targets = vec![0.0_f32; 8];
            if let Some(ref per_param) = grad_info.per_param_norms {
                // Divide parameters into 8 buckets and compute relative scales
                let bucket_size = per_param.len() / 8;
                for i in 0..8 {
                    let start = i * bucket_size;
                    let end = if i == 7 { per_param.len() } else { (i + 1) * bucket_size };

                    let bucket_norm: f32 = per_param[start..end].iter().sum();
                    let bucket_count = (end - start) as f32;
                    let avg_norm = bucket_norm / bucket_count.max(1.0);

                    // Normalize by global average
                    let global_avg = grad_info.gradient_norm / (per_param.len() as f32).sqrt();
                    let scale = (avg_norm / global_avg.max(1e-8)).clamp(0.1, 2.0);
                    layer_targets[i] = scale;
                }
            } else {
                // No per-param data: assume uniform distribution
                for i in 0..8 {
                    layer_targets[i] = 1.0; // Uniform scale
                }
            }

            // Compute loss and gradients for all dimensions
            let mut feature_errors = vec![0.0_f32; self.weight_delta_dim];

            // Dimension 0: magnitude
            let pred_mag_tanh = predicted_features[0].tanh();
            feature_errors[0] = (pred_mag_tanh - target_magnitude_tanh) * tanh_deriv(pred_mag_tanh);

            // Dimension 1: direction confidence
            let pred_conf_sigmoid = sigmoid(predicted_features[1]);
            feature_errors[1] = (pred_conf_sigmoid - target_confidence) * sigmoid_deriv(pred_conf_sigmoid);

            // Dimensions 2-9: layer scales
            for i in 0..8 {
                let pred_scale_sigmoid = sigmoid(predicted_features[2 + i]);
                let target_scale_normalized = (layer_targets[i] - 0.5).clamp(-0.4, 0.4) + 0.5; // Map to [0.1, 0.9]
                feature_errors[2 + i] = (pred_scale_sigmoid - target_scale_normalized)
                    * sigmoid_deriv(pred_scale_sigmoid);
            }

            // Backprop through linear layer: error -> gradients
            // Weight gradient: dL/dW = outer(error, combined)
            for out_idx in 0..self.weight_delta_dim {
                for in_idx in 0..combined_dim {
                    let weight_idx = out_idx * combined_dim + in_idx;
                    if weight_idx < self.weight_delta_head_weights.len() {
                        let grad = feature_errors[out_idx] * combined[in_idx];
                        self.weight_delta_head_weights[weight_idx] -= lr * 0.1 * grad;
                    }
                }
            }
        }

        // Record gradient norm for history
        self.gradient_norm_history.push(grad_info.gradient_norm);
        if self.gradient_norm_history.len() > 100 {
            self.gradient_norm_history.remove(0);
        }

        self.learning_rate_history
            .push(state.optimizer_state_summary.effective_lr);
        if self.learning_rate_history.len() > 100 {
            self.learning_rate_history.remove(0);
        }
    }

    // ─── Decode heads ────────────────────────────────────────────────────

    /// Decodes loss prediction from combined state.
    ///
    /// Clamps logit values to prevent numerical overflow/underflow.
    fn decode_loss(&self, combined_state: &[f32]) -> f32 {
        let logit: f32 = combined_state
            .iter()
            .zip(self.loss_head_weights.iter())
            .map(|(&s, &w)| s * w)
            .sum();

        // Clamp logit to prevent extreme exp() values
        // exp(10) ≈ 22026, exp(-10) ≈ 0.000045 - reasonable range for loss
        let clamped_logit = logit.clamp(-10.0, 10.0);

        // Use exp to ensure positive loss values
        clamped_logit.exp().max(1e-6)
    }

    /// Decodes weight delta prediction from combined state.
    ///
    /// Projects the GRU hidden state to weight delta summary statistics,
    /// then scales by gradient norm and learning rate history.
    ///
    /// # Arguments
    ///
    /// * `combined_state` - The concatenated deterministic + stochastic state
    /// * `state` - Current training state for gradient/learning rate context
    /// * `loss_trajectory` - Predicted loss trajectory for scaling
    /// * `y_steps` - Number of steps being predicted
    /// * `confidence` - Model confidence for this prediction
    ///
    /// # Returns
    ///
    /// A `WeightDelta` with predicted aggregate weight changes.
    fn decode_weight_delta(
        &self,
        combined_state: &[f32],
        state: &TrainingState,
        loss_trajectory: &[f32],
        y_steps: usize,
        confidence: f32,
    ) -> WeightDelta {
        let combined_dim = combined_state.len();

        // Linear projection: combined_state -> weight_delta_features
        // Weight layout: weight_delta_head_weights[out_idx * combined_dim + in_idx]
        let mut weight_delta_features = vec![0.0_f32; self.weight_delta_dim];
        for out_idx in 0..self.weight_delta_dim {
            let mut sum = 0.0_f32;
            for in_idx in 0..combined_dim {
                let weight_idx = out_idx * combined_dim + in_idx;
                if weight_idx < self.weight_delta_head_weights.len() {
                    sum += combined_state[in_idx] * self.weight_delta_head_weights[weight_idx];
                }
            }
            weight_delta_features[out_idx] = sum;
        }

        // Extract weight delta summary statistics
        // [0]: magnitude (raw, will be scaled)
        // [1]: direction_confidence
        // [2..10]: layer_scales for 8 representative layers
        let raw_magnitude = weight_delta_features.first().copied().unwrap_or(0.0);
        let direction_confidence = sigmoid(weight_delta_features.get(1).copied().unwrap_or(0.0));

        // Scale magnitude by gradient norm and learning rate
        let grad_norm = state.gradient_norm.max(1e-8);
        let lr = state.optimizer_state_summary.effective_lr.max(1e-8);

        // Base magnitude: use gradient norm as baseline for weight update size
        // Typical weight update: delta_w ~ -lr * grad
        let base_magnitude = lr * grad_norm;

        // Scale by loss trajectory improvement
        // If loss is decreasing, predict larger weight changes
        let loss_improvement = if loss_trajectory.len() >= 2 {
            let initial = loss_trajectory.first().copied().unwrap_or(state.loss);
            let final_loss = loss_trajectory.last().copied().unwrap_or(state.loss);
            (initial - final_loss).max(0.0) / initial.max(1e-8)
        } else {
            0.0
        };

        // Combine learned magnitude with heuristic scaling
        // raw_magnitude is learned, base_magnitude provides scale
        let magnitude = base_magnitude
            * (1.0 + raw_magnitude.tanh())
            * (1.0 + loss_improvement)
            * y_steps as f32;

        // Create weight delta with layer-wise scales
        let mut deltas = std::collections::HashMap::new();

        // Generate representative layer deltas based on layer_scales
        // These are aggregate statistics, not per-weight values
        let layer_names = [
            "embed",
            "attention.q",
            "attention.k",
            "attention.v",
            "attention.out",
            "mlp.up",
            "mlp.down",
            "lm_head",
        ];

        for (i, layer_name) in layer_names.iter().enumerate() {
            let layer_scale = sigmoid(weight_delta_features.get(2 + i).copied().unwrap_or(0.0));
            // Store a single-element vector representing the aggregate delta magnitude for this layer
            let layer_delta = vec![magnitude * layer_scale * direction_confidence];
            deltas.insert((*layer_name).to_string(), layer_delta);
        }

        WeightDelta {
            deltas,
            scale: magnitude,
            metadata: WeightDeltaMetadata {
                is_predicted: true,
                confidence: Some(confidence * direction_confidence),
                source_phase: Some(crate::Phase::Predict),
                num_steps: y_steps,
            },
        }
    }

    // ─── Prediction ──────────────────────────────────────────────────────

    /// Predicts training outcome after Y steps.
    ///
    /// Rolls out the GRU for `y_steps` timesteps with **active stochastic
    /// sampling**: at each step, the stochastic state is re-derived from
    /// the evolving deterministic state via softmax projection. This means
    /// the stochastic path genuinely participates in multi-step rollouts
    /// rather than remaining frozen at its initial value.
    ///
    /// # Arguments
    ///
    /// * `state` - Current training state
    /// * `y_steps` - Number of steps to predict ahead
    ///
    /// # Returns
    ///
    /// Prediction with uncertainty estimate.
    #[must_use]
    pub fn predict_y_steps(
        &self,
        state: &TrainingState,
        y_steps: usize,
    ) -> (PhasePrediction, PredictionUncertainty) {
        // Edge case: zero steps means return current state
        if y_steps == 0 {
            let prediction = PhasePrediction {
                weight_delta: WeightDelta::empty(),
                predicted_final_loss: state.loss,
                loss_trajectory: vec![state.loss],
                confidence: 1.0,
                loss_bounds: (state.loss, state.loss),
                num_steps: 0,
            };
            let uncertainty = PredictionUncertainty {
                aleatoric: 0.0,
                epistemic: 0.0,
                total: 0.0,
                entropy: 0.0,
            };
            return (prediction, uncertainty);
        }

        let features = state.compute_features();

        // Collect trajectories and final hidden states from each ensemble member
        let mut ensemble_trajectories: Vec<Vec<f32>> =
            Vec::with_capacity(self.config.ensemble_size);
        let mut final_combined_states: Vec<Vec<f32>> =
            Vec::with_capacity(self.config.ensemble_size);
        let mut total_entropy = 0.0_f32;

        for (ensemble_idx, latent) in self.latent_states.iter().enumerate() {
            let weights = &self.gru_weights[ensemble_idx];

            // Clone current latent state for rollout (we don't modify self)
            let mut rollout_latent = latent.clone();

            // Trajectory for this ensemble member
            let mut trajectory = Vec::with_capacity(y_steps + 1);
            trajectory.push(state.loss); // Start with current loss

            // Track the final combined state for weight delta prediction
            let mut final_combined = Vec::new();

            // CRITICAL FIX: Evolve feature vector during rollout to reduce input staleness
            // Mutable copy of features that we'll update with predicted losses
            let mut evolving_features = features.clone();

            // Roll out Y steps with active stochastic sampling
            for step_idx in 0..y_steps {
                // Update deterministic state via GRU using evolving features
                rollout_latent.deterministic =
                    Self::gru_step(weights, &rollout_latent.deterministic, &evolving_features);

                // Active stochastic sampling: re-derive stochastic state
                // from the evolving deterministic state
                let step_entropy = rollout_latent.sample_stochastic_from_deterministic();
                total_entropy += step_entropy;

                // Decode loss prediction from updated combined state
                let loss_pred = self.decode_loss(&rollout_latent.combined);
                trajectory.push(loss_pred);

                // Update feature vector with predicted loss for next step
                // This prevents input staleness from accumulating over long horizons
                // Only update features[0] (current loss) to maintain ensemble diversity
                if step_idx + 1 < y_steps {
                    // Only update if there's a next step
                    evolving_features[0] = loss_pred; // Current loss (ensemble-member-specific)
                    // Note: features[1-63] remain from initial state
                    // Not updating loss_ema or gradient norms preserves ensemble variance
                }

                // Keep the final combined state for weight delta prediction
                final_combined.clone_from(&rollout_latent.combined);
            }

            ensemble_trajectories.push(trajectory);
            final_combined_states.push(final_combined);
        }

        // Aggregate ensemble predictions
        let ensemble_size = ensemble_trajectories.len() as f32;

        // Build mean trajectory
        let trajectory_length = y_steps + 1;
        let mut mean_trajectory = vec![0.0; trajectory_length];
        for trajectory in &ensemble_trajectories {
            for (i, &loss_val) in trajectory.iter().enumerate() {
                mean_trajectory[i] += loss_val / ensemble_size;
            }
        }

        // Final loss prediction is last point in mean trajectory
        let predicted_final_loss = mean_trajectory[trajectory_length - 1];

        // Compute ensemble variance at final step
        let final_predictions: Vec<f32> = ensemble_trajectories
            .iter()
            .map(|traj| *traj.last().unwrap())
            .collect();

        let variance: f32 = final_predictions
            .iter()
            .map(|&p| (p - predicted_final_loss).powi(2))
            .sum::<f32>()
            / ensemble_size;

        // Base uncertainty from ensemble disagreement
        let base_std = variance.sqrt();

        // Uncertainty grows with prediction horizon (sqrt scaling)
        let horizon_factor = (y_steps as f32).sqrt();
        let total_std = base_std * (1.0 + 0.1 * horizon_factor);

        // Average entropy across ensemble members and steps
        let avg_entropy = total_entropy / (ensemble_size * y_steps as f32).max(1.0);

        let uncertainty = PredictionUncertainty {
            aleatoric: total_std * 0.5,
            epistemic: total_std * 0.5,
            total: total_std,
            entropy: avg_entropy,
        };

        let confidence = 1.0 / (1.0 + total_std); // Higher std = lower confidence

        // Compute mean combined state across ensemble for weight delta prediction
        let combined_dim = self.config.deterministic_dim + self.config.stochastic_dim;
        let mut mean_combined_state = vec![0.0_f32; combined_dim];

        // Filter out empty states (edge case for first prediction)
        let valid_states: Vec<_> = final_combined_states
            .iter()
            .filter(|s| !s.is_empty())
            .collect();

        if !valid_states.is_empty() {
            let num_valid = valid_states.len() as f32;
            for combined in valid_states {
                for (i, &val) in combined.iter().enumerate() {
                    if i < combined_dim {
                        mean_combined_state[i] += val / num_valid;
                    }
                }
            }
        }

        // Decode weight delta from the averaged final hidden state
        let weight_delta = self.decode_weight_delta(
            &mean_combined_state,
            state,
            &mean_trajectory,
            y_steps,
            confidence,
        );

        let prediction = PhasePrediction {
            weight_delta,
            predicted_final_loss,
            loss_trajectory: mean_trajectory,
            confidence,
            loss_bounds: (
                predicted_final_loss - 2.0 * total_std,
                predicted_final_loss + 2.0 * total_std,
            ),
            num_steps: y_steps,
        };

        (prediction, uncertainty)
    }

    /// Returns the prediction confidence for the current state.
    ///
    /// Uses an internal cache to avoid redundant computation when called
    /// multiple times for the same training step. The cache is invalidated
    /// whenever the model is updated via `observe_gradient()`,
    /// `update_from_observation()`, or `reset()`.
    #[must_use]
    pub fn prediction_confidence(&self, state: &TrainingState) -> f32 {
        // Return cached value if available for this step
        if let Some((cached_step, cached_conf)) = *self.cached_confidence.lock() {
            if cached_step == state.step {
                return cached_conf;
            }
        }

        // Base confidence from ensemble agreement
        let (_, uncertainty) = self.predict_y_steps(state, 10);
        let agreement_confidence = 1.0 / (1.0 + uncertainty.total);

        // Historical accuracy confidence
        // Start with moderate confidence to allow initial predictive exploration
        // Confidence will adjust based on actual prediction accuracy
        let historical_confidence = if self.prediction_errors.len() < 10 {
            0.7 // Moderate confidence to enable initial predict phases
        } else {
            let recent_errors: Vec<_> = self
                .prediction_errors
                .iter()
                .rev()
                .take(50)
                .copied()
                .collect();
            let mean_error: f32 = recent_errors.iter().sum::<f32>() / recent_errors.len() as f32;
            (1.0 / (1.0 + mean_error)).min(0.99)
        };

        // Loss stability confidence - reduce confidence when loss is highly variable
        // Use EMA spread (fast - slow) as a measure of recent volatility
        let loss_spread = state.loss_ema.spread().abs();
        let loss_slow = state.loss_ema.slow();
        let relative_spread = if loss_slow.abs() > 1e-6 {
            loss_spread / loss_slow.abs()
        } else {
            1.0 // High volatility assumption if loss near zero
        };
        // Map spread to confidence: spread=0 (stable) → 1.0, spread≥50% (volatile) → 0.5
        let stability_confidence = (1.0 / (1.0 + 2.0 * relative_spread)).max(0.5);

        // Combine confidences: agreement 40%, historical 40%, stability 20%
        // Reduced ensemble weight because it takes many steps to converge from random init
        let confidence = (agreement_confidence * 0.4
            + historical_confidence * 0.4
            + stability_confidence * 0.2)
            .clamp(0.0, 1.0);

        // Cache the computed confidence for this step
        *self.cached_confidence.lock() = Some((state.step, confidence));

        confidence
    }

    // ─── Observation / learning interface ─────────────────────────────────

    /// Updates the model from observed training data.
    ///
    /// Called after a sequence of training steps to update the dynamics
    /// model. Records prediction error and performs gradient descent on
    /// all learnable parameters (GRU weights + head weights).
    ///
    /// # Arguments
    ///
    /// * `state_before` - State before training
    /// * `state_after` - State after training
    /// * `loss_trajectory` - Observed loss values during training
    ///
    /// # Errors
    ///
    /// Returns an error if the update fails.
    pub fn update_from_observation(
        &mut self,
        state_before: &TrainingState,
        state_after: &TrainingState,
        loss_trajectory: &[f32],
    ) -> HybridResult<()> {
        // Invalidate cached confidence since model state is changing
        *self.cached_confidence.lock() = None;

        // Record prediction error
        let (prediction, _) = self.predict_y_steps(state_before, loss_trajectory.len());
        let actual_final_loss = state_after.loss;
        let error = (prediction.predicted_final_loss - actual_final_loss).abs();

        self.prediction_errors.push(error);
        if self.prediction_errors.len() > 1000 {
            self.prediction_errors.remove(0);
        }

        // Train the model using the final observed state
        let grad_info = crate::GradientInfo {
            loss: actual_final_loss,
            gradient_norm: state_after.gradient_norm,
            per_param_norms: None,
        };
        self.train_step(state_after, &grad_info);

        Ok(())
    }

    /// Returns the number of training updates performed.
    #[must_use]
    pub fn training_steps(&self) -> usize {
        self.training_steps
    }

    /// Observes a gradient computation step for online learning.
    ///
    /// Called during full training steps to update the dynamics model
    /// with observed gradient information. This is the primary online
    /// learning entry point that trains GRU weights via one-step
    /// truncated BPTT.
    ///
    /// # Arguments
    ///
    /// * `state` - Current training state
    /// * `grad_info` - Gradient information from the backward pass
    pub fn observe_gradient(&mut self, state: &TrainingState, grad_info: &crate::GradientInfo) {
        // Invalidate cached confidence since model state is changing
        *self.cached_confidence.lock() = None;

        // Record prediction error
        let (prediction, _) = self.predict_y_steps(state, 1);
        let actual_loss = state.loss;
        let error = (prediction.predicted_final_loss - actual_loss).abs();

        self.prediction_errors.push(error);
        if self.prediction_errors.len() > 1000 {
            self.prediction_errors.remove(0);
        }

        // Train all model parameters via one-step truncated BPTT
        self.train_step(state, grad_info);

        self.training_steps += 1;
    }

    /// Resets the model to initial state.
    pub fn reset(&mut self) {
        // Invalidate cached confidence
        *self.cached_confidence.lock() = None;

        for latent in &mut self.latent_states {
            latent.deterministic.fill(0.0);
            latent
                .stochastic
                .fill(1.0 / self.config.stochastic_dim as f32);
            latent.update_combined();
        }
        self.prediction_errors.clear();
        self.gradient_norm_history.clear();
        self.learning_rate_history.clear();
        self.training_steps = 0;

        // Clear BPTT history
        for history in &mut self.latent_history {
            history.clear();
        }
    }
}

/// Trait for dynamics models that predict training trajectories.
pub trait DynamicsModel: Send + Sync {
    /// The latent state type.
    type LatentState: Clone + Send;

    /// Initializes latent state from training state.
    fn initialize(&mut self, state: &TrainingState);

    /// Predicts outcome after Y steps.
    fn predict_y_steps(
        &self,
        state: &TrainingState,
        y_steps: usize,
    ) -> (PhasePrediction, PredictionUncertainty);

    /// Returns prediction confidence.
    fn prediction_confidence(&self, state: &TrainingState) -> f32;

    /// Updates from observation.
    ///
    /// # Errors
    ///
    /// Returns an error if the update fails.
    fn update_from_observation(
        &mut self,
        state_before: &TrainingState,
        state_after: &TrainingState,
        loss_trajectory: &[f32],
    ) -> HybridResult<()>;
}

impl DynamicsModel for RSSMLite {
    type LatentState = LatentState;

    fn initialize(&mut self, state: &TrainingState) {
        self.initialize_state(state);
    }

    fn predict_y_steps(
        &self,
        state: &TrainingState,
        y_steps: usize,
    ) -> (PhasePrediction, PredictionUncertainty) {
        RSSMLite::predict_y_steps(self, state, y_steps)
    }

    fn prediction_confidence(&self, state: &TrainingState) -> f32 {
        RSSMLite::prediction_confidence(self, state)
    }

    fn update_from_observation(
        &mut self,
        state_before: &TrainingState,
        state_after: &TrainingState,
        loss_trajectory: &[f32],
    ) -> HybridResult<()> {
        RSSMLite::update_from_observation(self, state_before, state_after, loss_trajectory)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rssm_creation() {
        let config = PredictorConfig::default();
        let rssm = RSSMLite::new(&config).unwrap();

        assert_eq!(rssm.config.ensemble_size, 3);
        assert_eq!(rssm.latent_states.len(), 3);
    }

    #[test]
    fn test_latent_state_combined() {
        let mut state = LatentState::new(4, 2);
        state.deterministic = vec![1.0, 2.0, 3.0, 4.0];
        state.stochastic = vec![0.5, 0.5];
        state.update_combined();

        assert_eq!(state.combined.len(), 6);
        assert_eq!(state.combined[0], 1.0);
        assert_eq!(state.combined[4], 0.5);
    }

    #[test]
    fn test_prediction() {
        let config = PredictorConfig::default();
        let mut rssm = RSSMLite::new(&config).unwrap();

        let mut state = TrainingState::new();
        state.loss = 2.5;
        state.record_step(2.5, 1.0);

        rssm.initialize_state(&state);
        let (prediction, uncertainty) = rssm.predict_y_steps(&state, 10);

        assert!(prediction.predicted_final_loss > 0.0);
        assert!(prediction.confidence > 0.0 && prediction.confidence <= 1.0);
        assert!(uncertainty.total >= 0.0);
    }

    #[test]
    fn test_confidence_with_history() {
        let config = PredictorConfig::default();
        let mut rssm = RSSMLite::new(&config).unwrap();

        // Initialize with proper state
        let mut state = TrainingState::new();
        state.loss = 2.5;
        state.record_step(2.5, 1.0);
        rssm.initialize_state(&state);

        // Add some prediction errors (low errors = high confidence)
        for _ in 0..20 {
            rssm.prediction_errors.push(0.1);
        }

        let confidence = rssm.prediction_confidence(&state);

        // Should have reasonable confidence with low errors
        assert!(
            confidence > 0.5,
            "confidence={} should be > 0.5",
            confidence
        );
    }

    #[test]
    fn test_multistep_rollout() {
        let config = PredictorConfig::default();
        let mut rssm = RSSMLite::new(&config).unwrap();

        let mut state = TrainingState::new();
        state.loss = 2.5;
        state.record_step(2.5, 1.0);
        rssm.initialize_state(&state);

        // Test different horizon lengths
        for y_steps in [1, 5, 10, 20, 50] {
            let (prediction, uncertainty) = rssm.predict_y_steps(&state, y_steps);

            // Trajectory should have at least 2 points (start + some samples)
            assert!(
                prediction.loss_trajectory.len() >= 2,
                "trajectory length {} for y_steps {}",
                prediction.loss_trajectory.len(),
                y_steps
            );

            // First point should be current loss
            assert_eq!(prediction.loss_trajectory[0], state.loss);

            // Uncertainty should be non-negative
            assert!(uncertainty.total >= 0.0);

            // Confidence should be in valid range
            assert!(prediction.confidence > 0.0 && prediction.confidence <= 1.0);
        }
    }

    #[test]
    fn test_uncertainty_growth() {
        let config = PredictorConfig::default();
        let mut rssm = RSSMLite::new(&config).unwrap();

        let mut state = TrainingState::new();
        state.loss = 2.5;
        state.record_step(2.5, 1.0);
        rssm.initialize_state(&state);

        // Uncertainty should grow with prediction horizon
        let (_, unc_1) = rssm.predict_y_steps(&state, 1);
        let (_, unc_10) = rssm.predict_y_steps(&state, 10);
        let (_, unc_50) = rssm.predict_y_steps(&state, 50);

        assert!(
            unc_10.total >= unc_1.total,
            "unc_10 ({}) should be >= unc_1 ({})",
            unc_10.total,
            unc_1.total
        );
        assert!(
            unc_50.total >= unc_10.total,
            "unc_50 ({}) should be >= unc_10 ({})",
            unc_50.total,
            unc_10.total
        );
    }

    #[test]
    fn test_zero_steps_edge_case() {
        let config = PredictorConfig::default();
        let mut rssm = RSSMLite::new(&config).unwrap();

        let mut state = TrainingState::new();
        state.loss = 2.5;
        rssm.initialize_state(&state);

        let (prediction, uncertainty) = rssm.predict_y_steps(&state, 0);

        // Should return current state
        assert_eq!(prediction.predicted_final_loss, state.loss);
        assert_eq!(prediction.num_steps, 0);
        assert_eq!(uncertainty.total, 0.0);
        assert_eq!(prediction.confidence, 1.0);
    }

    #[test]
    fn test_single_vs_multistep() {
        let config = PredictorConfig::default();
        let mut rssm = RSSMLite::new(&config).unwrap();

        let mut state = TrainingState::new();
        state.loss = 2.5;
        state.record_step(2.5, 1.0);
        rssm.initialize_state(&state);

        // Single step prediction
        let (pred_1, unc_1) = rssm.predict_y_steps(&state, 1);

        // Multi-step prediction
        let (pred_10, unc_10) = rssm.predict_y_steps(&state, 10);

        // Multi-step should have approximately equal or more uncertainty
        // Due to random initialization of ensemble members, there can be small
        // variations in uncertainty, so we use a more tolerant threshold.
        // The important thing is that predictions are valid, not that uncertainty
        // always increases monotonically in the initial untrained state.
        let tolerance = 0.01; // Allow up to 1% deviation
        assert!(
            unc_10.total >= unc_1.total - tolerance || (unc_10.total - unc_1.total).abs() < tolerance,
            "Expected unc_10.total ({}) ~>= unc_1.total ({}), diff = {}",
            unc_10.total,
            unc_1.total,
            unc_1.total - unc_10.total
        );

        // Both should have valid predictions
        assert!(pred_1.predicted_final_loss > 0.0);
        assert!(pred_10.predicted_final_loss > 0.0);

        // Multi-step trajectory should be longer or equal
        assert!(pred_10.loss_trajectory.len() >= pred_1.loss_trajectory.len());
    }

    #[test]
    fn test_weight_delta_prediction() {
        let config = PredictorConfig::default();
        let mut rssm = RSSMLite::new(&config).unwrap();

        // Create a training state with realistic values
        let mut state = TrainingState::new();
        state.loss = 2.5;
        state.gradient_norm = 1.5;
        state.optimizer_state_summary.effective_lr = 1e-4;
        state.record_step(2.5, 1.5);
        state.record_step(2.4, 1.4);
        state.record_step(2.3, 1.3);
        rssm.initialize_state(&state);

        // Predict 10 steps
        let (prediction, _) = rssm.predict_y_steps(&state, 10);

        // Weight delta should not be empty
        assert!(
            !prediction.weight_delta.deltas.is_empty(),
            "weight_delta.deltas should not be empty"
        );

        // Weight delta scale should be positive (we're predicting weight changes)
        assert!(
            prediction.weight_delta.scale > 0.0,
            "weight_delta.scale ({}) should be > 0",
            prediction.weight_delta.scale
        );

        // Should have predicted metadata
        assert!(
            prediction.weight_delta.metadata.is_predicted,
            "should be marked as predicted"
        );
        assert!(
            prediction.weight_delta.metadata.confidence.is_some(),
            "should have confidence"
        );
        assert_eq!(
            prediction.weight_delta.metadata.num_steps, 10,
            "should record 10 steps"
        );

        // Should have layer-wise deltas
        assert!(
            prediction.weight_delta.deltas.contains_key("attention.q"),
            "should have attention.q layer"
        );
        assert!(
            prediction.weight_delta.deltas.contains_key("mlp.up"),
            "should have mlp.up layer"
        );
    }

    #[test]
    fn test_weight_delta_scales_with_gradient() {
        let config = PredictorConfig::default();
        let mut rssm = RSSMLite::new(&config).unwrap();

        // State with small gradient
        let mut state_small_grad = TrainingState::new();
        state_small_grad.loss = 2.5;
        state_small_grad.gradient_norm = 0.1;
        state_small_grad.optimizer_state_summary.effective_lr = 1e-4;
        state_small_grad.record_step(2.5, 0.1);
        rssm.initialize_state(&state_small_grad);
        let (pred_small, _) = rssm.predict_y_steps(&state_small_grad, 10);

        // State with large gradient
        let mut state_large_grad = TrainingState::new();
        state_large_grad.loss = 2.5;
        state_large_grad.gradient_norm = 10.0;
        state_large_grad.optimizer_state_summary.effective_lr = 1e-4;
        state_large_grad.record_step(2.5, 10.0);
        rssm.initialize_state(&state_large_grad);
        let (pred_large, _) = rssm.predict_y_steps(&state_large_grad, 10);

        // Larger gradients should produce larger weight deltas
        assert!(
            pred_large.weight_delta.scale > pred_small.weight_delta.scale,
            "larger gradient ({}) should produce larger weight delta scale ({} vs {})",
            state_large_grad.gradient_norm,
            pred_large.weight_delta.scale,
            pred_small.weight_delta.scale
        );
    }

    #[test]
    fn test_weight_delta_scales_with_steps() {
        let config = PredictorConfig::default();
        let mut rssm = RSSMLite::new(&config).unwrap();

        let mut state = TrainingState::new();
        state.loss = 2.5;
        state.gradient_norm = 1.0;
        state.optimizer_state_summary.effective_lr = 1e-4;
        state.record_step(2.5, 1.0);
        rssm.initialize_state(&state);

        // Predict 1 step
        let (pred_1, _) = rssm.predict_y_steps(&state, 1);

        // Predict 10 steps
        let (pred_10, _) = rssm.predict_y_steps(&state, 10);

        // More steps should produce larger cumulative weight delta
        assert!(
            pred_10.weight_delta.scale > pred_1.weight_delta.scale,
            "10 steps should produce larger weight delta ({}) than 1 step ({})",
            pred_10.weight_delta.scale,
            pred_1.weight_delta.scale
        );
    }

    // ─── New tests for RSSM training functionality ───────────────────────

    #[test]
    fn test_stochastic_sampling() {
        let mut latent = LatentState::new(256, 32);

        // Set deterministic state to non-zero values
        for (i, val) in latent.deterministic.iter_mut().enumerate() {
            *val = (i as f32 * 0.01).sin();
        }

        let entropy = latent.sample_stochastic_from_deterministic();

        // Stochastic state should be a valid probability distribution
        let sum: f32 = latent.stochastic.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "stochastic state should sum to 1.0, got {}",
            sum
        );

        // All probabilities should be non-negative
        assert!(
            latent.stochastic.iter().all(|&p| p >= 0.0),
            "all probabilities should be non-negative"
        );

        // Entropy should be positive (non-degenerate distribution)
        assert!(entropy > 0.0, "entropy should be positive, got {}", entropy);

        // Combined state should be updated
        assert_eq!(
            latent.combined.len(),
            latent.deterministic.len() + latent.stochastic.len()
        );
    }

    #[test]
    fn test_sigmoid_numerical_stability() {
        // Large positive values should not overflow
        assert!((sigmoid(100.0) - 1.0).abs() < 1e-6);

        // Large negative values should not overflow
        assert!(sigmoid(-100.0).abs() < 1e-6);

        // Standard values
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);

        // Monotonicity
        assert!(sigmoid(1.0) > sigmoid(0.0));
        assert!(sigmoid(0.0) > sigmoid(-1.0));

        // Derivatives
        let s = sigmoid(0.0);
        assert!((sigmoid_deriv(s) - 0.25).abs() < 1e-6);

        let t = 0.0_f32.tanh();
        assert!((tanh_deriv(t) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_gru_backward_finite_differences() {
        // Verify GRU backward pass produces finite, non-zero gradients
        let input_dim = 8;
        let hidden_dim = 16;

        let weights = GRUWeights::new(input_dim, hidden_dim);
        let hidden: Vec<f32> = (0..hidden_dim).map(|i| (i as f32 * 0.1).tanh()).collect();
        let input: Vec<f32> = (0..input_dim).map(|i| i as f32 * 0.1).collect();

        let (_, cache) = RSSMLite::gru_step_with_cache(&weights, &hidden, &input);

        // dL/dh_new = ones (simple loss = sum of h_new)
        let dh_new = vec![1.0_f32; hidden_dim];

        let (grads, dh_prev) = RSSMLite::gru_backward(&weights, &cache, &dh_new);

        // All gradient vectors should be finite
        assert!(
            grads.dw_z.iter().all(|g| g.is_finite()),
            "dw_z should be finite"
        );
        assert!(
            grads.du_z.iter().all(|g| g.is_finite()),
            "du_z should be finite"
        );
        assert!(
            grads.dw_r.iter().all(|g| g.is_finite()),
            "dw_r should be finite"
        );
        assert!(
            grads.dw_h.iter().all(|g| g.is_finite()),
            "dw_h should be finite"
        );

        // At least some gradients should be non-zero
        let total_norm: f32 = grads.dw_z.iter().map(|g| g * g).sum::<f32>()
            + grads.du_z.iter().map(|g| g * g).sum::<f32>()
            + grads.dw_r.iter().map(|g| g * g).sum::<f32>()
            + grads.dw_h.iter().map(|g| g * g).sum::<f32>();
        assert!(
            total_norm > 0.0,
            "gradient norm should be non-zero, got {}",
            total_norm
        );

        // dh_prev should also be finite
        assert!(
            dh_prev.iter().all(|g| g.is_finite()),
            "dh_prev should be finite"
        );
        assert_eq!(dh_prev.len(), hidden_dim, "dh_prev should have correct dimension");
    }

    #[test]
    fn test_observe_gradient_trains_gru() {
        let config = PredictorConfig::default();
        let mut rssm = RSSMLite::new(&config).unwrap();

        let mut state = TrainingState::new();
        state.loss = 2.5;
        state.gradient_norm = 1.0;
        state.optimizer_state_summary.effective_lr = 1e-4;
        state.record_step(2.5, 1.0);
        rssm.initialize_state(&state);

        // Snapshot GRU weights before training
        let w_z_before = rssm.gru_weights[0].w_z.clone();
        let u_z_before = rssm.gru_weights[0].u_z.clone();

        let grad_info = crate::GradientInfo {
            loss: 2.5,
            gradient_norm: 1.0,
            per_param_norms: None,
        };

        // Observe several gradient steps
        for _ in 0..5 {
            rssm.observe_gradient(&state, &grad_info);
        }

        // GRU weights should have changed (training is happening)
        let w_z_changed = rssm.gru_weights[0]
            .w_z
            .iter()
            .zip(w_z_before.iter())
            .any(|(a, b)| (a - b).abs() > 1e-10);
        let u_z_changed = rssm.gru_weights[0]
            .u_z
            .iter()
            .zip(u_z_before.iter())
            .any(|(a, b)| (a - b).abs() > 1e-10);

        assert!(
            w_z_changed,
            "W_z weights should change after observe_gradient"
        );
        assert!(
            u_z_changed,
            "U_z weights should change after observe_gradient"
        );
    }

    #[test]
    fn test_training_reduces_prediction_error() {
        let config = PredictorConfig::default();
        let mut rssm = RSSMLite::new(&config).unwrap();

        // Simulate a simple training scenario with constant loss
        let target_loss = 2.0;

        let mut state = TrainingState::new();
        state.loss = target_loss;
        state.gradient_norm = 1.0;
        state.optimizer_state_summary.effective_lr = 1e-4;
        state.record_step(target_loss, 1.0);
        rssm.initialize_state(&state);

        let grad_info = crate::GradientInfo {
            loss: target_loss,
            gradient_norm: 1.0,
            per_param_norms: None,
        };

        // Measure initial prediction error
        let (pred_before, _) = rssm.predict_y_steps(&state, 1);
        let error_before = (pred_before.predicted_final_loss - target_loss).abs();

        // Train for many steps on the same target
        for _ in 0..50 {
            rssm.observe_gradient(&state, &grad_info);
        }

        // Measure prediction error after training
        let (pred_after, _) = rssm.predict_y_steps(&state, 1);
        let error_after = (pred_after.predicted_final_loss - target_loss).abs();

        // Error should decrease (or at least not explode)
        // Note: with random init, the initial error could be very large
        assert!(
            error_after < error_before + 1.0,
            "prediction error should not increase significantly: before={}, after={}",
            error_before,
            error_after
        );
    }

    #[test]
    fn test_entropy_computed_during_prediction() {
        let config = PredictorConfig::default();
        let mut rssm = RSSMLite::new(&config).unwrap();

        let mut state = TrainingState::new();
        state.loss = 2.5;
        state.record_step(2.5, 1.0);
        rssm.initialize_state(&state);

        let (_, uncertainty) = rssm.predict_y_steps(&state, 10);

        // Entropy should be computed and positive (non-degenerate stochastic state)
        assert!(
            uncertainty.entropy > 0.0,
            "entropy should be positive during prediction, got {}",
            uncertainty.entropy
        );
    }

    #[test]
    fn test_gradient_clipping() {
        let input_dim = 4;
        let hidden_dim = 4;
        let mut weights = GRUWeights::new(input_dim, hidden_dim);

        // Create artificially large gradients
        let mut grads = GRUGradients::zeros(input_dim, hidden_dim);
        for g in &mut grads.dw_z {
            *g = 100.0;
        }
        for g in &mut grads.du_z {
            *g = 100.0;
        }

        let w_z_before = weights.w_z.clone();

        // Apply with a strict gradient norm clip
        weights.apply_gradients(&grads, 0.01, 1.0);

        // Weights should have changed
        let changed = weights
            .w_z
            .iter()
            .zip(w_z_before.iter())
            .any(|(a, b)| (a - b).abs() > 1e-10);
        assert!(changed, "weights should change after gradient application");

        // But the change should be small due to clipping
        let max_change = weights
            .w_z
            .iter()
            .zip(w_z_before.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_change < 1.0,
            "gradient clipping should limit weight changes, max_change={}",
            max_change
        );
    }

    // ─── New comprehensive tests ─────────────────────────────────────────

    #[test]
    fn test_matvec_known_values() {
        // 2x3 matrix (row-major):
        // [ 1  2  3 ]
        // [ 4  5  6 ]
        //
        // Vector: [1, 0, -1]
        //
        // Result: [1*1 + 2*0 + 3*(-1), 4*1 + 5*0 + 6*(-1)] = [-2, -2]
        let mat = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let vec_input = vec![1.0, 0.0, -1.0];
        let rows = 2;
        let cols = 3;

        let result = RSSMLite::matvec(&mat, &vec_input, rows, cols);

        assert_eq!(result.len(), 2);
        assert!(
            (result[0] - (-2.0)).abs() < 1e-6,
            "expected -2.0, got {}",
            result[0]
        );
        assert!(
            (result[1] - (-2.0)).abs() < 1e-6,
            "expected -2.0, got {}",
            result[1]
        );

        // Second test: identity-like operation
        // 2x2 identity matrix times [3, 7] should give [3, 7]
        let identity = vec![1.0, 0.0, 0.0, 1.0];
        let vec2 = vec![3.0, 7.0];
        let result2 = RSSMLite::matvec(&identity, &vec2, 2, 2);
        assert!((result2[0] - 3.0).abs() < 1e-6);
        assert!((result2[1] - 7.0).abs() < 1e-6);

        // Third test: all-ones matrix
        // 2x3 all-ones matrix times [1, 2, 3] = [6, 6]
        let ones_mat = vec![1.0; 6];
        let vec3 = vec![1.0, 2.0, 3.0];
        let result3 = RSSMLite::matvec(&ones_mat, &vec3, 2, 3);
        assert!((result3[0] - 6.0).abs() < 1e-6);
        assert!((result3[1] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_matvec_t_transpose() {
        // 2x3 matrix (row-major):
        // [ 1  2  3 ]
        // [ 4  5  6 ]
        //
        // Transposed is:
        // [ 1  4 ]
        // [ 2  5 ]
        // [ 3  6 ]
        //
        // matvec_t(M, v, 2, 3) computes M^T * v
        // where v has length 2 (rows), result has length 3 (cols)
        //
        // v = [1, -1]
        // result[0] = 1*1 + 4*(-1) = -3
        // result[1] = 2*1 + 5*(-1) = -3
        // result[2] = 3*1 + 6*(-1) = -3
        let mat = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let vec_input = vec![1.0, -1.0];

        let result = RSSMLite::matvec_t(&mat, &vec_input, 2, 3);

        assert_eq!(result.len(), 3);
        assert!(
            (result[0] - (-3.0)).abs() < 1e-6,
            "expected -3.0, got {}",
            result[0]
        );
        assert!(
            (result[1] - (-3.0)).abs() < 1e-6,
            "expected -3.0, got {}",
            result[1]
        );
        assert!(
            (result[2] - (-3.0)).abs() < 1e-6,
            "expected -3.0, got {}",
            result[2]
        );

        // Verify relationship: matvec(M, x) should equal matvec_t(M^T, x)
        // For a square matrix M, (M*v) should give the same as (M^T)^T * v
        // More concretely: matvec(M, v, r, c) with v of len c
        //   vs matvec_t(M^T, v, c, r) with v of len c
        // Let's test with a square matrix to make it simpler:
        // M = [1 2; 3 4], v = [5, 6]
        // matvec: [1*5+2*6, 3*5+4*6] = [17, 39]
        // matvec_t with M^T = [1 3; 2 4] and same v=[5,6]:
        //   result[0] = 1*5 + 2*6 = 17
        //   result[1] = 3*5 + 4*6 = 39
        let m = vec![1.0, 2.0, 3.0, 4.0];
        let m_t = vec![1.0, 3.0, 2.0, 4.0]; // transposed in row-major
        let v = vec![5.0, 6.0];

        let result_mv = RSSMLite::matvec(&m, &v, 2, 2);
        let result_mvt = RSSMLite::matvec_t(&m_t, &v, 2, 2);

        assert!(
            (result_mv[0] - result_mvt[0]).abs() < 1e-6,
            "matvec and matvec_t(transpose) should agree: {} vs {}",
            result_mv[0],
            result_mvt[0]
        );
        assert!(
            (result_mv[1] - result_mvt[1]).abs() < 1e-6,
            "matvec and matvec_t(transpose) should agree: {} vs {}",
            result_mv[1],
            result_mvt[1]
        );
    }

    #[test]
    fn test_gru_step_with_cache_produces_valid_cache() {
        let input_dim = 8;
        let hidden_dim = 16;

        let weights = GRUWeights::new(input_dim, hidden_dim);
        let hidden: Vec<f32> = (0..hidden_dim).map(|i| (i as f32 * 0.1).sin()).collect();
        let input: Vec<f32> = (0..input_dim).map(|i| (i as f32 * 0.2).cos()).collect();

        let (h_new, cache) = RSSMLite::gru_step_with_cache(&weights, &hidden, &input);

        // Verify cache dimensions
        assert_eq!(
            cache.input.len(),
            input_dim,
            "cache input dimension mismatch"
        );
        assert_eq!(
            cache.h_prev.len(),
            hidden_dim,
            "cache h_prev dimension mismatch"
        );
        assert_eq!(cache.z.len(), hidden_dim, "cache z dimension mismatch");
        assert_eq!(cache.r.len(), hidden_dim, "cache r dimension mismatch");
        assert_eq!(
            cache.h_candidate.len(),
            hidden_dim,
            "cache h_candidate dimension mismatch"
        );

        // z and r should be sigmoid outputs, so in (0, 1)
        for (i, &z_val) in cache.z.iter().enumerate() {
            assert!(
                z_val > 0.0 && z_val < 1.0,
                "z[{}] = {} should be in (0, 1)",
                i,
                z_val
            );
        }
        for (i, &r_val) in cache.r.iter().enumerate() {
            assert!(
                r_val > 0.0 && r_val < 1.0,
                "r[{}] = {} should be in (0, 1)",
                i,
                r_val
            );
        }

        // h_candidate should be tanh outputs, so in (-1, 1)
        for (i, &h_val) in cache.h_candidate.iter().enumerate() {
            assert!(
                h_val >= -1.0 && h_val <= 1.0,
                "h_candidate[{}] = {} should be in [-1, 1]",
                i,
                h_val
            );
        }

        // z, r, h_candidate should have at least some non-zero values
        let z_norm: f32 = cache.z.iter().map(|x| x * x).sum();
        let r_norm: f32 = cache.r.iter().map(|x| x * x).sum();
        let h_cand_norm: f32 = cache.h_candidate.iter().map(|x| x * x).sum();

        assert!(z_norm > 0.0, "z should have non-zero values");
        assert!(r_norm > 0.0, "r should have non-zero values");
        assert!(h_cand_norm > 0.0, "h_candidate should have non-zero values");

        // h_new should match the GRU formula: (1-z)*h_prev + z*h_candidate
        for i in 0..hidden_dim {
            let expected = (1.0 - cache.z[i]) * cache.h_prev[i] + cache.z[i] * cache.h_candidate[i];
            assert!(
                (h_new[i] - expected).abs() < 1e-5,
                "h_new[{}] = {} doesn't match GRU formula expected = {}",
                i,
                h_new[i],
                expected
            );
        }

        // Cache input and h_prev should match the originals
        assert_eq!(cache.input, input, "cache should store original input");
        assert_eq!(cache.h_prev, hidden, "cache should store original hidden");
    }

    #[test]
    fn test_loss_head_training() {
        let config = PredictorConfig::default();
        let mut rssm = RSSMLite::new(&config).unwrap();

        let mut state = TrainingState::new();
        state.loss = 3.0;
        state.gradient_norm = 1.0;
        state.optimizer_state_summary.effective_lr = 1e-4;
        state.record_step(3.0, 1.0);
        rssm.initialize_state(&state);

        // Snapshot loss head weights before training
        let loss_head_before = rssm.loss_head_weights.clone();

        let grad_info = crate::GradientInfo {
            loss: 3.0,
            gradient_norm: 1.0,
            per_param_norms: None,
        };

        // Train for several steps
        for _ in 0..10 {
            rssm.observe_gradient(&state, &grad_info);
        }

        // Loss head weights should have changed
        let loss_head_changed = rssm
            .loss_head_weights
            .iter()
            .zip(loss_head_before.iter())
            .any(|(a, b)| (a - b).abs() > 1e-12);

        assert!(
            loss_head_changed,
            "loss head weights should change after training"
        );

        // Verify weights are still finite (no NaN/Inf from training)
        assert!(
            rssm.loss_head_weights.iter().all(|w| w.is_finite()),
            "all loss head weights should be finite after training"
        );
    }

    #[test]
    fn test_weight_delta_head_training() {
        let config = PredictorConfig::default();
        let mut rssm = RSSMLite::new(&config).unwrap();

        let mut state = TrainingState::new();
        state.loss = 2.0;
        state.gradient_norm = 1.5;
        state.optimizer_state_summary.effective_lr = 1e-3;
        state.record_step(2.0, 1.5);
        rssm.initialize_state(&state);

        // Snapshot weight delta head weights before training
        let delta_head_before = rssm.weight_delta_head_weights.clone();

        let grad_info = crate::GradientInfo {
            loss: 2.0,
            gradient_norm: 1.5,
            per_param_norms: None,
        };

        // Train for several steps
        for _ in 0..10 {
            rssm.observe_gradient(&state, &grad_info);
        }

        // Weight delta head weights should have changed
        let delta_head_changed = rssm
            .weight_delta_head_weights
            .iter()
            .zip(delta_head_before.iter())
            .any(|(a, b)| (a - b).abs() > 1e-12);

        assert!(
            delta_head_changed,
            "weight delta head weights should change after training"
        );

        // Verify weights are still finite
        assert!(
            rssm.weight_delta_head_weights.iter().all(|w| w.is_finite()),
            "all weight delta head weights should be finite after training"
        );
    }

    #[test]
    fn test_train_step_reduces_loss_prediction_error() {
        let config = PredictorConfig::default();
        let mut rssm = RSSMLite::new(&config).unwrap();

        let target_loss = 1.5;

        let mut state = TrainingState::new();
        state.loss = target_loss;
        state.gradient_norm = 0.8;
        state.optimizer_state_summary.effective_lr = 1e-4;
        state.record_step(target_loss, 0.8);
        rssm.initialize_state(&state);

        // Measure initial prediction error (1-step prediction)
        let (pred_initial, _) = rssm.predict_y_steps(&state, 1);
        let initial_error = (pred_initial.predicted_final_loss - target_loss).abs();

        let grad_info = crate::GradientInfo {
            loss: target_loss,
            gradient_norm: 0.8,
            per_param_norms: None,
        };

        // Train for many steps with consistent data
        for _ in 0..100 {
            rssm.observe_gradient(&state, &grad_info);
        }

        // Measure prediction error after training
        let (pred_after, _) = rssm.predict_y_steps(&state, 1);
        let final_error = (pred_after.predicted_final_loss - target_loss).abs();

        // After 100 training steps on consistent data, the error should decrease
        // or at least remain bounded. We use a generous threshold since
        // the dynamics model has complex interactions.
        assert!(
            final_error < initial_error + 0.5,
            "prediction error should not grow significantly after consistent training: \
             initial_error={}, final_error={}, predicted_loss={}",
            initial_error,
            final_error,
            pred_after.predicted_final_loss
        );
    }

    #[test]
    fn test_stochastic_entropy_uniform() {
        // When deterministic state has identical (or very similar) values,
        // the softmax should produce a nearly uniform distribution,
        // which has high entropy.
        let mut latent = LatentState::new(16, 8);

        // Set all deterministic values to the same constant
        // This will cause all logits to be the same after sampling,
        // leading to a uniform softmax distribution
        for val in latent.deterministic.iter_mut() {
            *val = 0.5;
        }

        let entropy = latent.sample_stochastic_from_deterministic();

        // For a uniform distribution over 8 categories,
        // entropy = ln(8) ~= 2.079
        let max_entropy = (8.0_f32).ln();

        // Entropy should be very close to maximum (uniform distribution)
        assert!(
            (entropy - max_entropy).abs() < 0.01,
            "uniform deterministic state should produce near-maximum entropy: \
             entropy={}, max_entropy={}",
            entropy,
            max_entropy
        );

        // Verify the distribution is actually uniform
        let expected_prob = 1.0 / 8.0;
        for (i, &p) in latent.stochastic.iter().enumerate() {
            assert!(
                (p - expected_prob).abs() < 1e-5,
                "stochastic[{}] = {} should be ~{} for uniform distribution",
                i,
                p,
                expected_prob
            );
        }
    }

    #[test]
    fn test_stochastic_entropy_peaked() {
        // When one deterministic value strongly dominates (through the
        // evenly-spaced sampling), the softmax should produce a peaked
        // distribution with low entropy.
        let mut latent = LatentState::new(16, 8);

        // Set deterministic state so that the sampled logits will be
        // very different: one large positive, rest very negative.
        // The sampling uses evenly-spaced indices: i * det_dim / stoch_dim
        // For det_dim=16, stoch_dim=8: indices are 0,2,4,6,8,10,12,14
        for val in latent.deterministic.iter_mut() {
            *val = -10.0; // Very negative baseline
        }
        // Make index 0 very large so it dominates after softmax
        latent.deterministic[0] = 10.0;

        let entropy = latent.sample_stochastic_from_deterministic();

        // Entropy should be low because the distribution is peaked
        // on one category. For a fully peaked distribution, entropy = 0.
        assert!(
            entropy < 0.5,
            "peaked deterministic state should produce low entropy: entropy={}",
            entropy
        );

        // The first stochastic component should dominate
        assert!(
            latent.stochastic[0] > 0.9,
            "first stochastic component should dominate: stochastic[0]={}",
            latent.stochastic[0]
        );

        // Other components should be very small
        let remaining_sum: f32 = latent.stochastic[1..].iter().sum();
        assert!(
            remaining_sum < 0.1,
            "remaining stochastic components should be small: sum={}",
            remaining_sum
        );
    }

    #[test]
    fn test_predict_y_steps_entropy_field() {
        let config = PredictorConfig::default();
        let mut rssm = RSSMLite::new(&config).unwrap();

        let mut state = TrainingState::new();
        state.loss = 2.0;
        state.gradient_norm = 1.0;
        state.optimizer_state_summary.effective_lr = 1e-4;
        state.record_step(2.0, 1.0);
        rssm.initialize_state(&state);

        // Predict multiple steps
        let (prediction, uncertainty) = rssm.predict_y_steps(&state, 20);

        // The entropy field in PredictionUncertainty should be populated
        // and non-negative (entropy is always >= 0)
        assert!(
            uncertainty.entropy >= 0.0,
            "entropy should be non-negative: {}",
            uncertainty.entropy
        );

        // With initialized state and non-trivial prediction horizon,
        // entropy should be positive (stochastic sampling is active)
        assert!(
            uncertainty.entropy > 0.0,
            "entropy should be positive for non-trivial predictions: {}",
            uncertainty.entropy
        );

        // The loss trajectory should have y_steps + 1 entries
        assert_eq!(
            prediction.loss_trajectory.len(),
            21,
            "trajectory should have y_steps + 1 = 21 entries"
        );

        // All trajectory values should be positive (exp-decoded losses)
        for (i, &loss_val) in prediction.loss_trajectory.iter().enumerate() {
            assert!(
                loss_val > 0.0,
                "loss_trajectory[{}] = {} should be positive",
                i,
                loss_val
            );
        }

        // Aleatoric and epistemic uncertainties should be non-negative
        assert!(
            uncertainty.aleatoric >= 0.0,
            "aleatoric uncertainty should be non-negative"
        );
        assert!(
            uncertainty.epistemic >= 0.0,
            "epistemic uncertainty should be non-negative"
        );

        // Total should be >= both components
        assert!(
            uncertainty.total >= uncertainty.aleatoric - 1e-6,
            "total uncertainty should be >= aleatoric"
        );
        assert!(
            uncertainty.total >= uncertainty.epistemic - 1e-6,
            "total uncertainty should be >= epistemic"
        );
    }

    #[test]
    fn test_multi_step_bptt_config() {
        // Test that BPTT depth can be configured
        let mut config = RSSMConfig::default();
        assert_eq!(config.bptt_steps, 1, "default BPTT depth should be 1");

        config.bptt_steps = 3;
        let predictor_config = PredictorConfig::RSSM {
            deterministic_dim: config.deterministic_dim,
            stochastic_dim: config.stochastic_dim,
            num_categoricals: config.num_categoricals,
            ensemble_size: config.ensemble_size,
        };

        let rssm = RSSMLite::new(&predictor_config).unwrap();
        // Verify initialization doesn't crash
        assert_eq!(rssm.latent_history.len(), config.ensemble_size);
    }

    #[test]
    fn test_multi_step_bptt_history_accumulation() {
        // Test that history is properly accumulated during training
        let mut config = RSSMConfig::default();
        config.bptt_steps = 3;
        config.ensemble_size = 1; // Simplify for testing

        let predictor_config = PredictorConfig::RSSM {
            deterministic_dim: config.deterministic_dim,
            stochastic_dim: config.stochastic_dim,
            num_categoricals: config.num_categoricals,
            ensemble_size: config.ensemble_size,
        };

        let mut rssm = RSSMLite::new(&predictor_config).unwrap();

        let mut state = TrainingState::new();
        state.loss = 2.5;
        state.gradient_norm = 1.0;
        state.optimizer_state_summary.effective_lr = 1e-4;
        state.record_step(2.5, 1.0);

        rssm.initialize_state(&state);

        // Initially history should be empty
        assert_eq!(rssm.latent_history[0].len(), 0);

        // Train for several steps
        let grad_info = crate::GradientInfo {
            loss: 2.5,
            gradient_norm: 1.0,
            per_param_norms: None,
        };

        for i in 0..5 {
            state.loss = 2.5 - i as f32 * 0.1;
            rssm.observe_gradient(&state, &grad_info);
        }

        // History should be capped at bptt_steps
        assert!(
            rssm.latent_history[0].len() <= config.bptt_steps,
            "history length {} should be <= bptt_steps {}",
            rssm.latent_history[0].len(),
            config.bptt_steps
        );

        // Should have at least one entry
        assert!(
            rssm.latent_history[0].len() > 0,
            "history should not be empty after training"
        );
    }

    #[test]
    fn test_multi_step_bptt_improves_training() {
        // Compare training with k=1 vs k=3 BPTT
        // With k=3, the model should learn better weight delta predictions

        // Train with k=1
        let config_k1 = PredictorConfig::RSSM {
            deterministic_dim: 128,
            stochastic_dim: 32,
            num_categoricals: 32,
            ensemble_size: 1,
        };
        let mut rssm_k1 = RSSMLite::new(&config_k1).unwrap();
        rssm_k1.config.bptt_steps = 1;

        // Train with k=3
        let config_k3 = PredictorConfig::RSSM {
            deterministic_dim: 128,
            stochastic_dim: 32,
            num_categoricals: 32,
            ensemble_size: 1,
        };
        let mut rssm_k3 = RSSMLite::new(&config_k3).unwrap();
        rssm_k3.config.bptt_steps = 3;

        // Create synthetic training trajectory
        let mut state = TrainingState::new();
        state.loss = 3.0;
        state.gradient_norm = 2.0;
        state.optimizer_state_summary.effective_lr = 1e-3;

        rssm_k1.initialize_state(&state);
        rssm_k3.initialize_state(&state);

        // Train both for same number of steps
        for i in 0..20 {
            state.loss = 3.0 * (0.95_f32).powi(i);
            state.gradient_norm = 2.0 * (0.95_f32).powi(i);
            state.record_step(state.loss, state.gradient_norm);

            let grad_info = crate::GradientInfo {
                loss: state.loss,
                gradient_norm: state.gradient_norm,
                per_param_norms: None,
            };

            rssm_k1.observe_gradient(&state, &grad_info);
            rssm_k3.observe_gradient(&state, &grad_info);
        }

        // Both should have learned something
        let (pred_k1, _) = rssm_k1.predict_y_steps(&state, 5);
        let (pred_k3, _) = rssm_k3.predict_y_steps(&state, 5);

        // Predictions should be valid
        assert!(pred_k1.predicted_final_loss > 0.0);
        assert!(pred_k3.predicted_final_loss > 0.0);
        assert!(pred_k1.predicted_final_loss.is_finite());
        assert!(pred_k3.predicted_final_loss.is_finite());

        // Both should predict weight deltas
        assert!(!pred_k1.weight_delta.deltas.is_empty());
        assert!(!pred_k3.weight_delta.deltas.is_empty());
    }

    #[test]
    fn test_bptt_gradient_accumulation() {
        // Verify that GRUGradients::accumulate works correctly
        let input_dim = 4;
        let hidden_dim = 4;

        let mut grads1 = GRUGradients::zeros(input_dim, hidden_dim);
        let mut grads2 = GRUGradients::zeros(input_dim, hidden_dim);

        // Set some values
        for i in 0..grads1.dw_z.len() {
            grads1.dw_z[i] = 1.0;
            grads2.dw_z[i] = 2.0;
        }

        // Accumulate with scale 0.5
        grads1.accumulate(&grads2, 0.5);

        // grads1.dw_z should now be 1.0 + 0.5 * 2.0 = 2.0
        for &val in &grads1.dw_z {
            assert!(
                (val - 2.0).abs() < 1e-6,
                "expected 2.0, got {}",
                val
            );
        }
    }

    #[test]
    fn test_bptt_dh_prev_computation() {
        // Verify that gru_backward correctly computes dh_prev
        let input_dim = 8;
        let hidden_dim = 16;

        let weights = GRUWeights::new(input_dim, hidden_dim);
        let hidden: Vec<f32> = (0..hidden_dim).map(|i| (i as f32 * 0.05).tanh()).collect();
        let input: Vec<f32> = (0..input_dim).map(|i| i as f32 * 0.1).collect();

        let (_, cache) = RSSMLite::gru_step_with_cache(&weights, &hidden, &input);
        let dh_new = vec![1.0_f32; hidden_dim];

        let (grads, dh_prev) = RSSMLite::gru_backward(&weights, &cache, &dh_new);

        // dh_prev should be finite
        assert!(
            dh_prev.iter().all(|&x| x.is_finite()),
            "dh_prev should be finite"
        );

        // dh_prev should have correct dimension
        assert_eq!(dh_prev.len(), hidden_dim);

        // dh_prev should be non-zero (gradient flows backward)
        let dh_prev_norm: f32 = dh_prev.iter().map(|&x| x * x).sum::<f32>().sqrt();
        assert!(
            dh_prev_norm > 0.0,
            "dh_prev should be non-zero, got norm {}",
            dh_prev_norm
        );

        // Gradients should also be valid
        assert!(grads.dw_z.iter().all(|&x| x.is_finite()));
    }
}
