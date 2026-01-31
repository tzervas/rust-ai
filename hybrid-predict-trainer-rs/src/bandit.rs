//! Bandit-based adaptive phase selection.
//!
//! Uses multi-armed bandit algorithms to select optimal phase lengths
//! and configurations based on observed training performance. This
//! enables the trainer to automatically adapt to different training
//! dynamics without manual tuning.
//!
//! # Why Bandits for Phase Selection?
//!
//! Optimal prediction phase length depends on factors that change during
//! training: model curvature, learning rate, loss landscape smoothness.
//! Static schedules cannot adapt. Bandit algorithms provide:
//! - **Online adaptation**: Learns from actual training outcomes
//! - **Exploration/exploitation balance**: Tries new lengths while favoring proven ones
//! - **Regret minimization**: Converges to optimal choices over time
//!
//! # Algorithm: `LinUCB`
//!
//! We use Linear Upper Confidence Bound (`LinUCB`) which:
//! - Models reward as a linear function of context features
//! - Balances exploration (trying different configurations) with
//!   exploitation (using known good configurations)
//! - Adapts online as more training data is collected
//!
//! # Arms
//!
//! Each "arm" represents a configuration choice:
//! - Prediction phase lengths (10, 20, 50, 80 steps, etc.)
//! - Confidence thresholds
//! - Correction intensity
//!
//! # Reward
//!
//! Reward is computed from:
//! - Speedup achieved (higher is better)
//! - Loss quality maintained (closer to baseline is better)
//! - Training stability (fewer divergences is better)

use crate::state::TrainingState;
use rand::Rng;

/// A bandit arm representing a configuration choice.
#[derive(Debug, Clone)]
pub struct Arm {
    /// Unique identifier for this arm.
    pub id: usize,

    /// Human-readable name.
    pub name: String,

    /// Configuration value (interpretation depends on arm type).
    pub value: f32,

    /// Number of times this arm has been selected.
    pub selection_count: usize,

    /// Total reward accumulated.
    pub total_reward: f64,

    /// Running mean reward.
    pub mean_reward: f64,

    /// Running variance of rewards.
    pub reward_variance: f64,
}

impl Arm {
    /// Creates a new arm with the given id and value.
    #[must_use]
    pub fn new(id: usize, name: String, value: f32) -> Self {
        Self {
            id,
            name,
            value,
            selection_count: 0,
            total_reward: 0.0,
            mean_reward: 0.0,
            reward_variance: 0.0,
        }
    }

    /// Updates the arm statistics with a new reward observation.
    pub fn update(&mut self, reward: f64) {
        self.selection_count += 1;
        self.total_reward += reward;

        let n = self.selection_count as f64;
        let delta = reward - self.mean_reward;
        self.mean_reward += delta / n;
        let delta2 = reward - self.mean_reward;
        self.reward_variance += delta * delta2;
    }

    /// Returns the standard deviation of rewards.
    #[must_use]
    pub fn reward_std(&self) -> f64 {
        if self.selection_count < 2 {
            f64::INFINITY
        } else {
            (self.reward_variance / (self.selection_count - 1) as f64).sqrt()
        }
    }

    /// Computes the UCB score for this arm.
    ///
    /// UCB = `mean_reward` + sqrt(2 * `ln(total_selections)` / `arm_selections`)
    #[must_use]
    pub fn ucb_score(&self, total_selections: usize, exploration_factor: f64) -> f64 {
        if self.selection_count == 0 {
            f64::INFINITY // Always explore unselected arms
        } else {
            let exploitation = self.mean_reward;
            let exploration = exploration_factor
                * (2.0 * (total_selections as f64).ln() / self.selection_count as f64).sqrt();
            exploitation + exploration
        }
    }
}

/// Configuration for the bandit selector.
#[derive(Debug, Clone)]
pub struct BanditConfig {
    /// Exploration factor for UCB (higher = more exploration).
    pub exploration_factor: f64,

    /// Minimum exploration probability (epsilon-greedy fallback).
    pub min_exploration_prob: f64,

    /// Whether to use contextual features.
    pub use_context: bool,

    /// L2 regularization for `LinUCB`.
    pub l2_regularization: f64,
}

impl Default for BanditConfig {
    fn default() -> Self {
        Self {
            exploration_factor: 1.0,
            min_exploration_prob: 0.05,
            use_context: true,
            l2_regularization: 1.0,
        }
    }
}

/// Bandit selector for adaptive phase configuration.
pub struct BanditSelector {
    /// Configuration.
    config: BanditConfig,

    /// Available arms.
    arms: Vec<Arm>,

    /// Total number of selections made.
    total_selections: usize,

    /// Last selected arm index.
    last_selection: Option<usize>,

    /// Feature dimension for contextual bandit.
    _feature_dim: usize,

    /// `LinUCB` A matrices (one per arm).
    a_matrices: Vec<Vec<f64>>,

    /// `LinUCB` b vectors (one per arm).
    b_vectors: Vec<Vec<f64>>,
}

impl BanditSelector {
    /// Creates a new bandit selector with the given arms.
    #[must_use]
    pub fn new(arm_values: &[(String, f32)], config: BanditConfig) -> Self {
        let arms: Vec<_> = arm_values
            .iter()
            .enumerate()
            .map(|(id, (name, value))| Arm::new(id, name.clone(), *value))
            .collect();

        let num_arms = arms.len();
        let feature_dim = 32; // From TrainingState::compute_features

        // Initialize LinUCB matrices
        // A = I (identity matrix, stored as diagonal)
        let a_matrices: Vec<_> = (0..num_arms)
            .map(|_| vec![config.l2_regularization; feature_dim])
            .collect();

        let b_vectors: Vec<_> = (0..num_arms).map(|_| vec![0.0; feature_dim]).collect();

        Self {
            config,
            arms,
            total_selections: 0,
            last_selection: None,
            _feature_dim: feature_dim,
            a_matrices,
            b_vectors,
        }
    }

    /// Creates a selector for prediction phase length selection.
    #[must_use]
    pub fn for_predict_length() -> Self {
        let arms = vec![
            ("predict_10".to_string(), 10.0),
            ("predict_20".to_string(), 20.0),
            ("predict_40".to_string(), 40.0),
            ("predict_60".to_string(), 60.0),
            ("predict_80".to_string(), 80.0),
        ];
        Self::new(&arms, BanditConfig::default())
    }

    /// Selects an arm based on current context.
    ///
    /// # Arguments
    ///
    /// * `state` - Current training state (for contextual features)
    ///
    /// # Returns
    ///
    /// The selected arm index and value.
    pub fn select(&mut self, state: &TrainingState) -> (usize, f32) {
        let mut rng = rand::rng();

        // Epsilon-greedy exploration
        if rng.random::<f64>() < self.config.min_exploration_prob {
            let idx = rng.random_range(0..self.arms.len());
            self.last_selection = Some(idx);
            self.total_selections += 1;
            return (idx, self.arms[idx].value);
        }

        // Compute UCB scores
        let features = state.compute_features();
        let scores: Vec<f64> = if self.config.use_context {
            self.compute_linucb_scores(&features)
        } else {
            self.arms
                .iter()
                .map(|arm| arm.ucb_score(self.total_selections, self.config.exploration_factor))
                .collect()
        };

        // Select arm with highest score
        let best_idx = scores
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(idx, _)| idx);

        self.last_selection = Some(best_idx);
        self.total_selections += 1;

        (best_idx, self.arms[best_idx].value)
    }

    /// Updates the bandit with observed reward.
    ///
    /// # Arguments
    ///
    /// * `arm_idx` - Index of the arm that was used
    /// * `reward` - Observed reward (higher is better)
    /// * `state` - Training state when arm was selected (for context)
    pub fn update(&mut self, arm_idx: usize, reward: f64, state: &TrainingState) {
        if arm_idx >= self.arms.len() {
            return;
        }

        // Update simple statistics
        self.arms[arm_idx].update(reward);

        // Update LinUCB model
        if self.config.use_context {
            let features = state.compute_features();

            // A_a = A_a + x * x^T (diagonal approximation: A_a += x^2)
            for (i, &f) in features.iter().enumerate() {
                if i < self.a_matrices[arm_idx].len() {
                    self.a_matrices[arm_idx][i] += f64::from(f * f);
                }
            }

            // b_a = b_a + r * x
            for (i, &f) in features.iter().enumerate() {
                if i < self.b_vectors[arm_idx].len() {
                    self.b_vectors[arm_idx][i] += reward * f64::from(f);
                }
            }
        }
    }

    /// Computes `LinUCB` scores for each arm.
    fn compute_linucb_scores(&self, features: &[f32]) -> Vec<f64> {
        self.arms
            .iter()
            .enumerate()
            .map(|(idx, _)| {
                // theta_a = A_a^{-1} * b_a (diagonal approx: theta = b / A)
                let theta: Vec<f64> = self.b_vectors[idx]
                    .iter()
                    .zip(self.a_matrices[idx].iter())
                    .map(|(&b, &a)| b / a.max(1e-6))
                    .collect();

                // p_a = theta^T * x
                let exploitation: f64 = theta
                    .iter()
                    .zip(features.iter())
                    .map(|(&t, &f)| t * f64::from(f))
                    .sum();

                // UCB term: alpha * sqrt(x^T * A^{-1} * x)
                let variance: f64 = features
                    .iter()
                    .zip(self.a_matrices[idx].iter())
                    .map(|(&f, &a)| f64::from(f).powi(2) / a.max(1e-6))
                    .sum();
                let exploration = self.config.exploration_factor * variance.sqrt();

                exploitation + exploration
            })
            .collect()
    }

    /// Returns the current best arm based on mean reward.
    #[must_use]
    pub fn best_arm(&self) -> Option<&Arm> {
        self.arms
            .iter()
            .filter(|a| a.selection_count > 0)
            .max_by(|a, b| {
                a.mean_reward
                    .partial_cmp(&b.mean_reward)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// Returns statistics about arm selections.
    #[must_use]
    pub fn statistics(&self) -> BanditStatistics {
        let selection_counts: Vec<_> = self.arms.iter().map(|a| a.selection_count).collect();
        let mean_rewards: Vec<_> = self.arms.iter().map(|a| a.mean_reward).collect();

        BanditStatistics {
            total_selections: self.total_selections,
            selection_counts,
            mean_rewards,
            best_arm_idx: self.best_arm().map(|a| a.id),
        }
    }

    /// Resets the bandit to initial state.
    pub fn reset(&mut self) {
        for arm in &mut self.arms {
            arm.selection_count = 0;
            arm.total_reward = 0.0;
            arm.mean_reward = 0.0;
            arm.reward_variance = 0.0;
        }

        for a in &mut self.a_matrices {
            for v in a.iter_mut() {
                *v = self.config.l2_regularization;
            }
        }

        for b in &mut self.b_vectors {
            for v in b.iter_mut() {
                *v = 0.0;
            }
        }

        self.total_selections = 0;
        self.last_selection = None;
    }
}

/// Statistics about bandit performance.
#[derive(Debug, Clone)]
pub struct BanditStatistics {
    /// Total number of selections.
    pub total_selections: usize,

    /// Selection count per arm.
    pub selection_counts: Vec<usize>,

    /// Mean reward per arm.
    pub mean_rewards: Vec<f64>,

    /// Index of best arm (if determined).
    pub best_arm_idx: Option<usize>,
}

/// Computes reward from training outcome.
///
/// # Arguments
///
/// * `speedup` - Achieved speedup factor (>1 is good)
/// * `loss_gap` - Relative loss gap vs baseline (lower is better)
/// * `stability` - Stability score 0-1 (higher is better)
///
/// # Returns
///
/// Combined reward signal.
#[must_use]
pub fn compute_reward(speedup: f64, loss_gap: f64, stability: f64) -> f64 {
    // Reward components with weights
    let speedup_reward = (speedup - 1.0).max(0.0).min(10.0); // Cap at 10x
    let quality_reward = (1.0 - loss_gap).max(0.0); // 0 if gap >= 100%
    let stability_reward = stability;

    // Weighted combination
    0.4 * speedup_reward + 0.4 * quality_reward + 0.2 * stability_reward
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arm_update() {
        let mut arm = Arm::new(0, "test".to_string(), 10.0);

        arm.update(1.0);
        arm.update(2.0);
        arm.update(3.0);

        assert_eq!(arm.selection_count, 3);
        assert!((arm.mean_reward - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_ucb_score_exploration() {
        let arm = Arm::new(0, "test".to_string(), 10.0);

        // Unselected arm should have infinite UCB score
        let score = arm.ucb_score(10, 1.0);
        assert!(score.is_infinite());
    }

    #[test]
    fn test_bandit_selection() {
        let mut bandit = BanditSelector::for_predict_length();
        let state = TrainingState::new();

        // First selection should work
        let (idx, value) = bandit.select(&state);
        assert!(idx < 5);
        assert!(value > 0.0);
    }

    #[test]
    fn test_bandit_update() {
        let mut bandit = BanditSelector::for_predict_length();
        let state = TrainingState::new();

        let (idx, _) = bandit.select(&state);
        bandit.update(idx, 1.0, &state);

        assert_eq!(bandit.arms[idx].selection_count, 1);
    }

    #[test]
    fn test_reward_computation() {
        let reward = compute_reward(2.0, 0.05, 0.9);
        assert!(reward > 0.0);

        // Better speedup should give higher reward
        let reward_fast = compute_reward(5.0, 0.05, 0.9);
        assert!(reward_fast > reward);
    }
}
