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

    #[test]
    fn test_linucb_score_computation() {
        // Create a bandit with two arms and update them with different context/reward patterns
        let arms = vec![("arm_a".to_string(), 10.0), ("arm_b".to_string(), 20.0)];
        let config = BanditConfig {
            use_context: true,
            exploration_factor: 1.0,
            ..BanditConfig::default()
        };
        let mut bandit = BanditSelector::new(&arms, config);

        // Create a state with some recorded steps so features are non-trivial
        let mut state = TrainingState::new();
        state.record_step(2.5, 1.0);
        state.record_step(2.3, 0.9);
        state.record_step(2.1, 0.8);

        // Update arm 0 with a high reward, arm 1 with a low reward
        bandit.update(0, 5.0, &state);
        bandit.update(1, 0.1, &state);

        // Compute LinUCB scores -- they should differ between arms
        let features = state.compute_features();
        let scores = bandit.compute_linucb_scores(&features);

        assert_eq!(scores.len(), 2);
        // Arm 0 got higher reward, so its exploitation term should be higher
        // (both arms have same context, so exploration terms are similar)
        assert!(
            (scores[0] - scores[1]).abs() > 1e-6,
            "Scores should differ: arm0={}, arm1={}",
            scores[0],
            scores[1]
        );
    }

    #[test]
    fn test_ab_matrix_update() {
        let arms = vec![("only_arm".to_string(), 10.0)];
        let config = BanditConfig {
            use_context: true,
            l2_regularization: 1.0,
            ..BanditConfig::default()
        };
        let mut bandit = BanditSelector::new(&arms, config);

        // Record initial A and b values (A starts at l2_reg = 1.0, b starts at 0.0)
        let feature_dim = bandit.a_matrices[0].len();
        assert!(feature_dim > 0);
        for i in 0..feature_dim {
            assert!(
                (bandit.a_matrices[0][i] - 1.0).abs() < 1e-9,
                "A should start at l2_reg"
            );
            assert!((bandit.b_vectors[0][i]).abs() < 1e-9, "b should start at 0");
        }

        // Create a state with non-zero features
        let mut state = TrainingState::new();
        state.record_step(3.0, 2.0);

        let features = state.compute_features();
        let reward = 2.5;

        // First update
        bandit.update(0, reward, &state);

        // Verify A_a += x^2 for each feature dimension
        for i in 0..feature_dim {
            let f = features[i];
            let expected_a = 1.0 + f64::from(f * f);
            assert!(
                (bandit.a_matrices[0][i] - expected_a).abs() < 1e-6,
                "A[{}] should be {} but got {}",
                i,
                expected_a,
                bandit.a_matrices[0][i]
            );
        }

        // Verify b_a += r * x for each feature dimension
        for i in 0..feature_dim {
            let f = features[i];
            let expected_b = reward * f64::from(f);
            assert!(
                (bandit.b_vectors[0][i] - expected_b).abs() < 1e-6,
                "b[{}] should be {} but got {}",
                i,
                expected_b,
                bandit.b_vectors[0][i]
            );
        }

        // Second update with same state and reward -- values should accumulate
        bandit.update(0, reward, &state);
        for i in 0..feature_dim {
            let f = features[i];
            let expected_a = 1.0 + 2.0 * f64::from(f * f);
            assert!(
                (bandit.a_matrices[0][i] - expected_a).abs() < 1e-6,
                "After 2 updates, A[{}] should be {} but got {}",
                i,
                expected_a,
                bandit.a_matrices[0][i]
            );
            let expected_b = 2.0 * reward * f64::from(f);
            assert!(
                (bandit.b_vectors[0][i] - expected_b).abs() < 1e-6,
                "After 2 updates, b[{}] should be {} but got {}",
                i,
                expected_b,
                bandit.b_vectors[0][i]
            );
        }
    }

    #[test]
    fn test_best_arm_identification() {
        let arms = vec![
            ("low".to_string(), 10.0),
            ("high".to_string(), 20.0),
            ("medium".to_string(), 15.0),
        ];
        let mut bandit = BanditSelector::new(&arms, BanditConfig::default());
        let state = TrainingState::new();

        // Update arm 0 with low rewards
        for _ in 0..10 {
            bandit.update(0, 0.2, &state);
        }

        // Update arm 1 with high rewards
        for _ in 0..10 {
            bandit.update(1, 5.0, &state);
        }

        // Update arm 2 with medium rewards
        for _ in 0..10 {
            bandit.update(2, 1.5, &state);
        }

        let best = bandit.best_arm().expect("Should have a best arm");
        assert_eq!(best.id, 1, "Arm 1 (high rewards) should be the best");
        assert!((best.mean_reward - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_reset_clears_state() {
        let arms = vec![("a".to_string(), 10.0), ("b".to_string(), 20.0)];
        let config = BanditConfig {
            l2_regularization: 2.0,
            ..BanditConfig::default()
        };
        let mut bandit = BanditSelector::new(&arms, config);
        let state = TrainingState::new();

        // Perform some updates
        bandit.update(0, 3.0, &state);
        bandit.update(1, 1.0, &state);
        bandit.total_selections = 5;
        bandit.last_selection = Some(1);

        // Verify state is non-zero before reset
        assert!(bandit.arms[0].selection_count > 0);
        assert!(bandit.total_selections > 0);

        // Reset
        bandit.reset();

        // Verify all arm statistics are zeroed
        for arm in &bandit.arms {
            assert_eq!(arm.selection_count, 0);
            assert!((arm.total_reward).abs() < 1e-9);
            assert!((arm.mean_reward).abs() < 1e-9);
            assert!((arm.reward_variance).abs() < 1e-9);
        }

        // Verify A matrices reset to l2_regularization (2.0)
        for a_matrix in &bandit.a_matrices {
            for &v in a_matrix {
                assert!(
                    (v - 2.0).abs() < 1e-9,
                    "A matrix should reset to l2_reg=2.0"
                );
            }
        }

        // Verify b vectors reset to 0
        for b_vec in &bandit.b_vectors {
            for &v in b_vec {
                assert!(v.abs() < 1e-9, "b vector should reset to 0");
            }
        }

        // Verify selector state reset
        assert_eq!(bandit.total_selections, 0);
        assert!(bandit.last_selection.is_none());
    }

    #[test]
    fn test_reward_speedup_component() {
        // High speedup with neutral quality and stability
        let reward_high_speedup = compute_reward(8.0, 0.0, 0.5);
        let reward_low_speedup = compute_reward(1.5, 0.0, 0.5);

        // Higher speedup should yield higher reward
        assert!(
            reward_high_speedup > reward_low_speedup,
            "High speedup ({}) should beat low speedup ({})",
            reward_high_speedup,
            reward_low_speedup
        );

        // Verify speedup component: (speedup - 1.0).max(0).min(10) * 0.4
        // For speedup=8.0: speedup_reward = 7.0 * 0.4 = 2.8
        // quality_reward = (1.0 - 0.0) * 0.4 = 0.4
        // stability_reward = 0.5 * 0.2 = 0.1
        // total = 3.3
        assert!((reward_high_speedup - 3.3).abs() < 1e-6);

        // Speedup <= 1.0 should contribute 0 speedup reward
        let reward_no_speedup = compute_reward(0.5, 0.0, 0.5);
        // speedup_reward = (0.5 - 1.0).max(0) = 0
        // quality_reward = 0.4, stability_reward = 0.1
        // total = 0.5
        assert!((reward_no_speedup - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_reward_quality_component() {
        // Negative loss_gap means improvement (loss_gap < 0)
        let reward_improved = compute_reward(1.0, -0.5, 0.5);
        // speedup_reward = (1.0-1.0).max(0) = 0
        // quality_reward = (1.0 - (-0.5)).max(0) = 1.5, but capped? No, formula is just (1.0 - loss_gap).max(0)
        // quality_reward = 1.5 * 0.4 = 0.6
        // stability_reward = 0.5 * 0.2 = 0.1
        // total = 0.7
        assert!((reward_improved - 0.7).abs() < 1e-6);

        // Positive loss_gap means degradation
        let reward_degraded = compute_reward(1.0, 0.8, 0.5);
        // quality_reward = (1.0 - 0.8).max(0) = 0.2 * 0.4 = 0.08
        // total = 0.0 + 0.08 + 0.1 = 0.18
        assert!((reward_degraded - 0.18).abs() < 1e-6);

        // Quality improvement should give higher total reward
        assert!(reward_improved > reward_degraded);

        // Loss gap >= 1.0 means quality component is 0
        let reward_total_loss = compute_reward(1.0, 1.5, 0.5);
        // quality_reward = (1.0 - 1.5).max(0) = 0
        // total = 0.0 + 0.0 + 0.1 = 0.1
        assert!((reward_total_loss - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_reward_stability_component() {
        // High stability
        let reward_stable = compute_reward(1.0, 0.0, 1.0);
        // speedup = 0, quality = 0.4, stability = 1.0 * 0.2 = 0.2
        // total = 0.6
        assert!((reward_stable - 0.6).abs() < 1e-6);

        // Low stability
        let reward_unstable = compute_reward(1.0, 0.0, 0.0);
        // speedup = 0, quality = 0.4, stability = 0.0
        // total = 0.4
        assert!((reward_unstable - 0.4).abs() < 1e-6);

        // Stability contributes positively
        assert!(reward_stable > reward_unstable);

        // Low prediction_error implies high stability (passed as stability param)
        // Test with stability = 0.95 (low error => high stability)
        let reward_low_error = compute_reward(1.0, 0.0, 0.95);
        // stability = 0.95 * 0.2 = 0.19
        // total = 0.0 + 0.4 + 0.19 = 0.59
        assert!((reward_low_error - 0.59).abs() < 1e-6);
    }

    #[test]
    fn test_epsilon_greedy_explores() {
        // Create a bandit where arm 0 has by far the best known reward
        let arms = vec![
            ("best".to_string(), 10.0),
            ("other1".to_string(), 20.0),
            ("other2".to_string(), 30.0),
        ];
        let config = BanditConfig {
            min_exploration_prob: 0.3, // 30% exploration to make test reliable
            use_context: false,        // Simpler, non-contextual selection
            ..BanditConfig::default()
        };
        let mut bandit = BanditSelector::new(&arms, config);
        let state = TrainingState::new();

        // Give arm 0 a very high reward so it's clearly the best
        for _ in 0..20 {
            bandit.update(0, 10.0, &state);
        }
        // Give other arms low rewards
        for _ in 0..20 {
            bandit.update(1, 0.1, &state);
            bandit.update(2, 0.1, &state);
        }

        // Run many selections and count how often non-best arms are chosen
        let mut non_best_count = 0;
        let num_trials = 500;
        for _ in 0..num_trials {
            let (idx, _) = bandit.select(&state);
            if idx != 0 {
                non_best_count += 1;
            }
        }

        // With 30% exploration and 3 arms, we expect ~20% non-greedy selections (30% * 2/3)
        // Allow wide margin for randomness but ensure some exploration happens
        assert!(
            non_best_count > 10,
            "Expected some exploration selections, got only {} out of {}",
            non_best_count,
            num_trials
        );
    }

    #[test]
    fn test_arm_statistics() {
        let arms = vec![
            ("a".to_string(), 10.0),
            ("b".to_string(), 20.0),
            ("c".to_string(), 30.0),
        ];
        let mut bandit = BanditSelector::new(&arms, BanditConfig::default());
        let state = TrainingState::new();

        // No updates yet
        let stats = bandit.statistics();
        assert_eq!(stats.total_selections, 0);
        assert_eq!(stats.selection_counts, vec![0, 0, 0]);
        assert_eq!(stats.mean_rewards, vec![0.0, 0.0, 0.0]);
        assert!(stats.best_arm_idx.is_none());

        // Update arms with known rewards
        bandit.update(0, 2.0, &state);
        bandit.update(0, 4.0, &state);
        // Arm 0: 2 selections, mean = 3.0

        bandit.update(1, 10.0, &state);
        // Arm 1: 1 selection, mean = 10.0

        // Arm 2: no selections

        let stats = bandit.statistics();
        assert_eq!(stats.selection_counts, vec![2, 1, 0]);
        assert!((stats.mean_rewards[0] - 3.0).abs() < 1e-6);
        assert!((stats.mean_rewards[1] - 10.0).abs() < 1e-6);
        assert!((stats.mean_rewards[2]).abs() < 1e-6);

        // Best arm should be arm 1 (highest mean reward among selected arms)
        assert_eq!(stats.best_arm_idx, Some(1));
    }
}
