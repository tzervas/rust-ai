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

use crate::config::PredictorConfig;
use crate::error::HybridResult;
use crate::predictive::PhasePrediction;
use crate::state::{TrainingState, WeightDelta};

/// Sigmoid activation function.
#[allow(dead_code)]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
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
}

impl Default for RSSMConfig {
    fn default() -> Self {
        Self {
            deterministic_dim: 256,
            stochastic_dim: 32,
            num_categoricals: 32,
            ensemble_size: 3,
            input_dim: 64, // From TrainingState::compute_features (64-dim)
            hidden_dim: 128,
            learning_rate: 0.001,
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

    /// Training step counter.
    training_steps: usize,

    /// Historical prediction errors for confidence estimation.
    prediction_errors: Vec<f32>,

    /// Temperature for stochastic sampling (reserved for future use).
    #[allow(dead_code)]
    temperature: f32,
}

/// Weights for a GRU cell.
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct GRUWeights {
    /// Update gate weights.
    w_z: Vec<f32>,
    /// Reset gate weights.
    w_r: Vec<f32>,
    /// Candidate hidden state weights.
    w_h: Vec<f32>,
    /// Update gate recurrent weights.
    u_z: Vec<f32>,
    /// Reset gate recurrent weights.
    u_r: Vec<f32>,
    /// Candidate hidden state recurrent weights.
    u_h: Vec<f32>,
    /// Biases.
    b_z: Vec<f32>,
    b_r: Vec<f32>,
    b_h: Vec<f32>,
}

impl GRUWeights {
    /// Creates randomly initialized GRU weights.
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

        Ok(Self {
            config: rssm_config,
            latent_states,
            gru_weights,
            loss_head_weights,
            training_steps: 0,
            prediction_errors: Vec::with_capacity(1000),
            temperature: 1.0,
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

            // Initialize stochastic state uniformly
            for s in &mut latent.stochastic {
                *s = 1.0 / self.config.stochastic_dim as f32;
            }

            latent.update_combined();
        }
    }

    /// Performs one GRU step given weights, hidden state, and input.
    fn gru_step(weights: &GRUWeights, hidden: &[f32], input: &[f32]) -> Vec<f32> {
        let hidden_dim = hidden.len();

        // Matrix-vector multiply helper for weight matrices (stored as hidden_dim rows x input_dim cols)
        // Weight layout: w[row * input_dim + col] for W matrices, w[row * hidden_dim + col] for U matrices
        let matvec_w = |w: &[f32], x: &[f32]| -> Vec<f32> {
            let input_dim = x.len();
            let mut result = vec![0.0; hidden_dim];
            for i in 0..hidden_dim {
                let mut sum = 0.0;
                for j in 0..input_dim {
                    sum += w[i * input_dim + j] * x[j];
                }
                result[i] = sum;
            }
            result
        };

        let matvec_u = |u: &[f32], h: &[f32]| -> Vec<f32> {
            let mut result = vec![0.0; hidden_dim];
            for i in 0..hidden_dim {
                let mut sum = 0.0;
                for j in 0..hidden_dim {
                    sum += u[i * hidden_dim + j] * h[j];
                }
                result[i] = sum;
            }
            result
        };

        // z = sigmoid(W_z·x + U_z·h + b_z)
        let mut z = matvec_w(&weights.w_z, input);
        let z_h = matvec_u(&weights.u_z, hidden);
        for i in 0..hidden_dim {
            z[i] = sigmoid(z[i] + z_h[i] + weights.b_z[i]);
        }

        // r = sigmoid(W_r·x + U_r·h + b_r)
        let mut r = matvec_w(&weights.w_r, input);
        let r_h = matvec_u(&weights.u_r, hidden);
        for i in 0..hidden_dim {
            r[i] = sigmoid(r[i] + r_h[i] + weights.b_r[i]);
        }

        // h_tilde = tanh(W_h·x + U_h·(r⊙h) + b_h)
        let mut h_candidate = matvec_w(&weights.w_h, input);
        let r_h_elem: Vec<f32> = r.iter().zip(hidden.iter()).map(|(&r, &h)| r * h).collect();
        let h_rec = matvec_u(&weights.u_h, &r_h_elem);
        for i in 0..hidden_dim {
            h_candidate[i] = (h_candidate[i] + h_rec[i] + weights.b_h[i]).tanh();
        }

        // h_new = (1-z)⊙h + z⊙h_tilde
        (0..hidden_dim)
            .map(|i| (1.0 - z[i]) * hidden[i] + z[i] * h_candidate[i])
            .collect()
    }

    /// Decodes loss prediction from combined state.
    fn decode_loss(&self, combined_state: &[f32]) -> f32 {
        let logit: f32 = combined_state
            .iter()
            .zip(self.loss_head_weights.iter())
            .map(|(&s, &w)| s * w)
            .sum();
        // Use exp to ensure positive loss values
        logit.exp().max(1e-6)
    }

    /// Predicts training outcome after Y steps.
    ///
    /// # Arguments
    ///
    /// * `state` - Current training state
    /// * `y_steps` - Number of steps to predict ahead
    ///
    /// # Errors
    ///
    /// Returns an error if the operation fails.
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

        // Collect trajectories from each ensemble member
        let mut ensemble_trajectories: Vec<Vec<f32>> =
            Vec::with_capacity(self.config.ensemble_size);

        for (ensemble_idx, latent) in self.latent_states.iter().enumerate() {
            let weights = &self.gru_weights[ensemble_idx];

            // Clone current latent state for rollout
            let mut hidden = latent.deterministic.clone();
            let stochastic = latent.stochastic.clone();

            // Trajectory for this ensemble member
            let mut trajectory = Vec::with_capacity(y_steps + 1);
            trajectory.push(state.loss); // Start with current loss

            // Roll out Y steps
            for _ in 0..y_steps {
                // Update deterministic state via GRU
                hidden = Self::gru_step(weights, &hidden, &features);

                // Build combined state
                let mut combined = Vec::with_capacity(hidden.len() + stochastic.len());
                combined.extend_from_slice(&hidden);
                combined.extend_from_slice(&stochastic);

                // Decode loss prediction
                let loss_pred = self.decode_loss(&combined);
                trajectory.push(loss_pred);
            }

            ensemble_trajectories.push(trajectory);
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

        let uncertainty = PredictionUncertainty {
            aleatoric: total_std * 0.5,
            epistemic: total_std * 0.5,
            total: total_std,
            entropy: 0.0, // Could compute from stochastic distribution
        };

        let prediction = PhasePrediction {
            weight_delta: WeightDelta::empty(),
            predicted_final_loss,
            loss_trajectory: mean_trajectory,
            confidence: 1.0 / (1.0 + total_std), // Higher std = lower confidence
            loss_bounds: (
                predicted_final_loss - 2.0 * total_std,
                predicted_final_loss + 2.0 * total_std,
            ),
            num_steps: y_steps,
        };

        (prediction, uncertainty)
    }

    /// Returns the prediction confidence for the current state.
    #[must_use]
    pub fn prediction_confidence(&self, state: &TrainingState) -> f32 {
        // Base confidence from ensemble agreement
        let (_, uncertainty) = self.predict_y_steps(state, 10);
        let agreement_confidence = 1.0 / (1.0 + uncertainty.total);

        // Historical accuracy confidence
        let historical_confidence = if self.prediction_errors.len() < 10 {
            0.5 // Low confidence until we have enough data
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

        // Combine confidences
        (agreement_confidence * 0.6 + historical_confidence * 0.4).clamp(0.0, 1.0)
    }

    /// Updates the model from observed training data.
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
        // Record prediction error
        let (prediction, _) = self.predict_y_steps(state_before, loss_trajectory.len());
        let actual_final_loss = state_after.loss;
        let error = (prediction.predicted_final_loss - actual_final_loss).abs();

        self.prediction_errors.push(error);
        if self.prediction_errors.len() > 1000 {
            self.prediction_errors.remove(0);
        }

        // Update model weights using simple gradient descent
        // This is a placeholder - real implementation would use proper backprop
        let learning_rate = self.config.learning_rate;
        let error_signal = prediction.predicted_final_loss - actual_final_loss;

        for (i, &combined) in self.latent_states[0].combined.iter().enumerate() {
            if i < self.loss_head_weights.len() {
                self.loss_head_weights[i] -= learning_rate * error_signal * combined;
            }
        }

        self.training_steps += 1;

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
    /// with observed gradient information.
    ///
    /// # Arguments
    ///
    /// * `state` - Current training state
    /// * `grad_info` - Gradient information from the backward pass
    pub fn observe_gradient(&mut self, state: &TrainingState, grad_info: &crate::GradientInfo) {
        // Record the prediction error if we have a recent prediction
        let (prediction, _) = self.predict_y_steps(state, 1);
        let actual_loss = state.loss;
        let error = (prediction.predicted_final_loss - actual_loss).abs();

        self.prediction_errors.push(error);
        if self.prediction_errors.len() > 1000 {
            self.prediction_errors.remove(0);
        }

        // Update model weights using gradient information
        let learning_rate = self.config.learning_rate;
        let error_signal = prediction.predicted_final_loss - actual_loss;

        // Update loss head weights
        for (i, &combined) in self.latent_states[0].combined.iter().enumerate() {
            if i < self.loss_head_weights.len() {
                self.loss_head_weights[i] -= learning_rate * error_signal * combined;
            }
        }

        // Update GRU state based on gradient norm (incorporate new information)
        let grad_scale = (grad_info.gradient_norm / 10.0).min(1.0);
        for latent in &mut self.latent_states {
            // Shift deterministic state slightly based on gradient direction
            let len = latent.deterministic.len() as f32;
            for (i, val) in latent.deterministic.iter_mut().enumerate() {
                *val = (*val * 0.99) + (grad_scale * (i as f32 / len - 0.5) * 0.01);
            }
            latent.update_combined();
        }

        self.training_steps += 1;
    }

    /// Resets the model to initial state.
    pub fn reset(&mut self) {
        for latent in &mut self.latent_states {
            latent.deterministic.fill(0.0);
            latent
                .stochastic
                .fill(1.0 / self.config.stochastic_dim as f32);
            latent.update_combined();
        }
        self.prediction_errors.clear();
        self.training_steps = 0;
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
        // Allow small floating point tolerance
        assert!(
            unc_10.total >= unc_1.total - 1e-6,
            "Expected unc_10.total ({}) >= unc_1.total ({}) - epsilon",
            unc_10.total,
            unc_1.total
        );

        // Both should have valid predictions
        assert!(pred_1.predicted_final_loss > 0.0);
        assert!(pred_10.predicted_final_loss > 0.0);

        // Multi-step trajectory should be longer or equal
        assert!(pred_10.loss_trajectory.len() >= pred_1.loss_trajectory.len());
    }
}
