//! Learning Rate Advisor
//!
//! Analyzes training dynamics to recommend learning rate adjustments in real-time.
//!
//! # Features
//!
//! - Loss curve analysis to detect oscillation or plateau
//! - Gradient magnitude monitoring for explosion/vanishing detection
//! - Phase-aware recommendations (warmup has higher tolerance)
//! - Urgency levels to prioritize critical adjustments
//!
//! # Example
//!
//! ```rust
//! use training_tools::lr_advisor::{analyze_lr, TrainingPhase};
//!
//! let recent_losses = vec![2.5, 2.3, 2.1, 2.0, 1.95, 1.92];
//! let recent_gradients = vec![0.5, 0.48, 0.47, 0.46, 0.45, 0.44];
//! let current_lr = 1e-4;
//! let current_step = 1000;
//! let phase = TrainingPhase::Stable;
//!
//! if let Some(advice) = analyze_lr(
//!     &recent_losses,
//!     &recent_gradients,
//!     current_lr,
//!     current_step,
//!     phase,
//! ) {
//!     println!("{}", advice.reason);
//!     println!("Suggested LR: {:.2e} → {:.2e}", advice.current_lr, advice.suggested_lr);
//! }
//! ```

use serde::{Deserialize, Serialize};

/// Urgency level for learning rate adjustments.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Urgency {
    /// Low priority - training is progressing well, but could be optimized
    Low,
    /// Medium priority - training could benefit from adjustment
    Medium,
    /// High priority - training is suboptimal or unstable
    High,
    /// Critical - training is likely to fail or diverge without immediate action
    Critical,
}

impl std::fmt::Display for Urgency {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Low => write!(f, "LOW"),
            Self::Medium => write!(f, "MEDIUM"),
            Self::High => write!(f, "HIGH"),
            Self::Critical => write!(f, "CRITICAL"),
        }
    }
}

/// Training phase for context-aware advice.
///
/// Re-exported from training_state to avoid circular dependencies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TrainingPhase {
    /// Warmup phase - higher tolerance for oscillation
    Warmup,
    /// Stable phase - main training period
    Stable,
    /// Predict phase - predictive gradient mode
    Predict,
    /// Correct phase - correction after prediction
    Correct,
}

impl From<crate::training_state::TrainingPhase> for TrainingPhase {
    fn from(phase: crate::training_state::TrainingPhase) -> Self {
        match phase {
            crate::training_state::TrainingPhase::Warmup => Self::Warmup,
            crate::training_state::TrainingPhase::Full => Self::Stable,
            crate::training_state::TrainingPhase::Predict => Self::Predict,
            crate::training_state::TrainingPhase::Correct => Self::Correct,
        }
    }
}

/// Learning rate advice with recommendation and reasoning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LRAdvice {
    /// Current learning rate
    pub current_lr: f32,
    /// Suggested learning rate
    pub suggested_lr: f32,
    /// Human-readable explanation of why this adjustment is recommended
    pub reason: String,
    /// Urgency level
    pub urgency: Urgency,
    /// Specific issue detected (for logging/debugging)
    pub issue: Issue,
}

impl LRAdvice {
    /// Get the suggested multiplier (suggested_lr / current_lr).
    pub fn multiplier(&self) -> f32 {
        if self.current_lr == 0.0 {
            1.0
        } else {
            self.suggested_lr / self.current_lr
        }
    }

    /// Get percentage change as a signed value (-50.0 = 50% reduction, 20.0 = 20% increase).
    pub fn percentage_change(&self) -> f32 {
        (self.multiplier() - 1.0) * 100.0
    }

    /// Format the advice as a concise string.
    pub fn format(&self) -> String {
        let change = self.percentage_change();
        let direction = if change > 0.0 { "increase" } else { "decrease" };
        format!(
            "[{}] {} LR by {:.1}%: {}",
            self.urgency,
            direction,
            change.abs(),
            self.reason
        )
    }
}

/// Specific issue detected during analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Issue {
    /// Loss is oscillating with high amplitude
    LossOscillation,
    /// Loss has plateaued (not decreasing)
    LossPlateau,
    /// Gradients are exploding
    GradientExplosion,
    /// Gradients are vanishing
    GradientVanishing,
    /// Loss is increasing significantly
    LossIncrease,
    /// Gradient norm is unstable
    GradientInstability,
}

impl std::fmt::Display for Issue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LossOscillation => write!(f, "Loss Oscillation"),
            Self::LossPlateau => write!(f, "Loss Plateau"),
            Self::GradientExplosion => write!(f, "Gradient Explosion"),
            Self::GradientVanishing => write!(f, "Gradient Vanishing"),
            Self::LossIncrease => write!(f, "Loss Increase"),
            Self::GradientInstability => write!(f, "Gradient Instability"),
        }
    }
}

/// Minimum number of samples required for reliable analysis.
const MIN_SAMPLES: usize = 10;

/// Analyze learning rate appropriateness and suggest adjustments.
///
/// Returns `Some(LRAdvice)` if an adjustment is recommended, or `None` if training
/// dynamics appear healthy.
///
/// # Arguments
///
/// * `recent_losses` - Loss values from recent steps (at least 10 recommended)
/// * `recent_gradients` - Gradient norms from recent steps
/// * `current_lr` - Current learning rate
/// * `current_step` - Current training step
/// * `phase` - Current training phase (affects tolerance)
///
/// # Analysis Criteria
///
/// - **Oscillation**: Loss amplitude > 10% of mean → reduce LR by 50%
/// - **Plateau**: Loss slope < 0.001 for 50+ steps → increase LR by 20-50%
/// - **Gradient Explosion**: Gradient > 10x baseline → emergency 75% reduction
/// - **Gradient Vanishing**: Gradient < 0.01x baseline → increase LR or flag architectural issue
/// - **Loss Increase**: Significant upward trend → reduce LR by 30-60%
///
/// # Phase-Specific Behavior
///
/// - **Warmup**: Higher tolerance for oscillation and instability
/// - **Stable/Predict/Correct**: Standard analysis
pub fn analyze_lr(
    recent_losses: &[f32],
    recent_gradients: &[f32],
    current_lr: f32,
    current_step: u64,
    phase: TrainingPhase,
) -> Option<LRAdvice> {
    // Need sufficient samples for reliable analysis
    if recent_losses.len() < MIN_SAMPLES || recent_gradients.len() < MIN_SAMPLES {
        return None;
    }

    // Phase-specific tolerance adjustments
    let (oscillation_tolerance, plateau_tolerance) = match phase {
        TrainingPhase::Warmup => (0.20, 0.0001), // 20% oscillation OK in warmup, less plateau detection
        _ => (0.10, 0.001),                      // 10% oscillation, standard plateau detection
    };

    // 1. Check for gradient explosion (critical priority)
    if let Some(advice) = check_gradient_explosion(recent_gradients, current_lr) {
        return Some(advice);
    }

    // 2. Check for loss oscillation (high priority)
    if let Some(advice) =
        check_loss_oscillation(recent_losses, current_lr, oscillation_tolerance, phase)
    {
        return Some(advice);
    }

    // 3. Check for significant loss increase (high priority)
    if let Some(advice) = check_loss_increase(recent_losses, current_lr, phase) {
        return Some(advice);
    }

    // 4. Check for gradient instability (medium priority)
    if let Some(advice) = check_gradient_instability(recent_gradients, current_lr) {
        return Some(advice);
    }

    // 5. Check for gradient vanishing (medium priority)
    if let Some(advice) = check_gradient_vanishing(recent_gradients, current_lr) {
        return Some(advice);
    }

    // 6. Check for loss plateau (low-medium priority, only after warmup)
    if phase != TrainingPhase::Warmup && recent_losses.len() >= 50 {
        if let Some(advice) =
            check_loss_plateau(recent_losses, current_lr, current_step, plateau_tolerance)
        {
            return Some(advice);
        }
    }

    // No issues detected
    None
}

/// Check for gradient explosion (>10x baseline).
fn check_gradient_explosion(recent_gradients: &[f32], current_lr: f32) -> Option<LRAdvice> {
    let baseline = compute_baseline_gradient(recent_gradients)?;
    let recent = &recent_gradients[recent_gradients.len().saturating_sub(5)..];
    let recent_mean = mean(recent);

    if recent_mean > baseline * 10.0 {
        let suggested_lr = current_lr * 0.25; // Reduce by 75%
        Some(LRAdvice {
            current_lr,
            suggested_lr,
            reason: format!(
                "Gradient explosion detected: {:.2e} → {:.2e} ({}x increase). Emergency LR reduction required.",
                baseline, recent_mean, (recent_mean / baseline) as u32
            ),
            urgency: Urgency::Critical,
            issue: Issue::GradientExplosion,
        })
    } else if recent_mean > baseline * 5.0 {
        let suggested_lr = current_lr * 0.5; // Reduce by 50%
        Some(LRAdvice {
            current_lr,
            suggested_lr,
            reason: format!(
                "Gradients growing rapidly: {:.2e} → {:.2e} ({}x increase). Preemptive LR reduction recommended.",
                baseline, recent_mean, (recent_mean / baseline) as u32
            ),
            urgency: Urgency::High,
            issue: Issue::GradientExplosion,
        })
    } else {
        None
    }
}

/// Check for loss oscillation (high variance).
fn check_loss_oscillation(
    recent_losses: &[f32],
    current_lr: f32,
    tolerance: f32,
    phase: TrainingPhase,
) -> Option<LRAdvice> {
    let loss_mean = mean(recent_losses);
    let loss_std = std_dev(recent_losses, loss_mean);

    // Amplitude as percentage of mean
    let amplitude_ratio = loss_std / loss_mean;

    if amplitude_ratio > tolerance {
        // More aggressive reduction if oscillation is severe
        let reduction_factor = if amplitude_ratio > tolerance * 2.0 {
            0.4 // 60% reduction
        } else {
            0.5 // 50% reduction
        };

        let suggested_lr = current_lr * reduction_factor;
        let urgency = match phase {
            TrainingPhase::Warmup => Urgency::Medium, // Less urgent in warmup
            _ => Urgency::High,
        };

        Some(LRAdvice {
            current_lr,
            suggested_lr,
            reason: format!(
                "Loss oscillating with {:.1}% amplitude (mean={:.3}, std={:.3}). LR too high.",
                amplitude_ratio * 100.0,
                loss_mean,
                loss_std
            ),
            urgency,
            issue: Issue::LossOscillation,
        })
    } else {
        None
    }
}

/// Check for loss plateau (slope near zero).
fn check_loss_plateau(
    recent_losses: &[f32],
    current_lr: f32,
    current_step: u64,
    _threshold: f32,
) -> Option<LRAdvice> {
    // Use at least last 50 steps for plateau detection
    let window = recent_losses.len().min(50);
    let losses = &recent_losses[recent_losses.len().saturating_sub(window)..];

    if losses.len() < 50 {
        return None;
    }

    let slope = compute_slope(losses);
    let mean_loss = mean(losses);

    // Use normalized slope relative to mean loss
    // This makes plateau detection scale-invariant
    let normalized_slope = slope.abs() / mean_loss.max(0.001);

    // Plateau: normalized slope < 0.001 (0.1% improvement per step = stalled)
    // Also check that slope isn't strongly positive (would indicate divergence, not plateau)
    if normalized_slope < 0.001 && slope <= 0.0 {
        // Increase LR to escape plateau - more aggressive (2x instead of 1.2-1.5x)
        let increase_factor = if normalized_slope < 0.0001 {
            2.0 // Very flat - aggressive increase
        } else {
            1.5 // Mild plateau - moderate increase
        };

        let suggested_lr = current_lr * increase_factor;

        // Lower urgency early in training
        let urgency = if current_step < 1000 {
            Urgency::Low
        } else {
            Urgency::Medium
        };

        Some(LRAdvice {
            current_lr,
            suggested_lr,
            reason: format!(
                "Loss plateau detected (slope={:.2e}, normalized={:.4} over {} steps). LR may be too conservative.",
                slope, normalized_slope, window
            ),
            urgency,
            issue: Issue::LossPlateau,
        })
    } else {
        None
    }
}

/// Check for significant loss increase.
fn check_loss_increase(
    recent_losses: &[f32],
    current_lr: f32,
    phase: TrainingPhase,
) -> Option<LRAdvice> {
    if recent_losses.len() < 20 {
        return None;
    }

    // Compare recent 10 steps to previous 10 steps
    let mid = recent_losses.len() - 10;
    let prev_mean = mean(&recent_losses[mid - 10..mid]);
    let recent_mean = mean(&recent_losses[mid..]);

    let increase_ratio = (recent_mean - prev_mean) / prev_mean;

    // Significant increase (>5% for non-warmup phases)
    let threshold = match phase {
        TrainingPhase::Warmup => 0.15, // 15% tolerance in warmup
        _ => 0.05,                     // 5% tolerance otherwise
    };

    if increase_ratio > threshold {
        let reduction_factor = if increase_ratio > 0.2 {
            0.4 // 60% reduction for severe increase
        } else {
            0.6 // 40% reduction for moderate increase
        };

        let suggested_lr = current_lr * reduction_factor;

        Some(LRAdvice {
            current_lr,
            suggested_lr,
            reason: format!(
                "Loss increasing: {:.3} → {:.3} ({:+.1}%). LR likely too high.",
                prev_mean,
                recent_mean,
                increase_ratio * 100.0
            ),
            urgency: Urgency::High,
            issue: Issue::LossIncrease,
        })
    } else {
        None
    }
}

/// Check for gradient vanishing (<0.01x baseline).
fn check_gradient_vanishing(recent_gradients: &[f32], current_lr: f32) -> Option<LRAdvice> {
    let baseline = compute_baseline_gradient(recent_gradients)?;
    let recent = &recent_gradients[recent_gradients.len().saturating_sub(5)..];
    let recent_mean = mean(recent);

    if recent_mean < baseline * 0.01 && baseline > 1e-8 {
        // Vanishing gradients - could be LR too low OR architectural issue
        let suggested_lr = current_lr * 2.0; // Try doubling first

        Some(LRAdvice {
            current_lr,
            suggested_lr,
            reason: format!(
                "Gradient vanishing: {:.2e} → {:.2e} ({}x decrease). Try increasing LR, or investigate architecture.",
                baseline, recent_mean, (baseline / recent_mean.max(1e-10)) as u32
            ),
            urgency: Urgency::Medium,
            issue: Issue::GradientVanishing,
        })
    } else {
        None
    }
}

/// Check for gradient instability (high variance in gradient norms).
fn check_gradient_instability(recent_gradients: &[f32], current_lr: f32) -> Option<LRAdvice> {
    if recent_gradients.len() < 20 {
        return None;
    }

    let grad_mean = mean(recent_gradients);
    let grad_std = std_dev(recent_gradients, grad_mean);

    // Coefficient of variation (CV) = std / mean
    // High CV indicates instability
    let cv = grad_std / grad_mean;

    if cv > 0.5 && grad_mean > 1e-8 {
        let suggested_lr = current_lr * 0.7; // 30% reduction

        Some(LRAdvice {
            current_lr,
            suggested_lr,
            reason: format!(
                "Gradient instability detected: CV={:.2} (mean={:.2e}, std={:.2e}). LR may be too high.",
                cv, grad_mean, grad_std
            ),
            urgency: Urgency::Medium,
            issue: Issue::GradientInstability,
        })
    } else {
        None
    }
}

/// Compute baseline gradient (median of first 50%, excluding outliers).
fn compute_baseline_gradient(gradients: &[f32]) -> Option<f32> {
    if gradients.is_empty() {
        return None;
    }

    let baseline_end = gradients.len() / 2;
    let baseline_grads = &gradients[..baseline_end.max(5)];

    // Use median to be robust to outliers
    Some(median(baseline_grads))
}

/// Compute mean of values.
fn mean(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f32>() / values.len() as f32
}

/// Compute standard deviation.
fn std_dev(values: &[f32], mean: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / values.len() as f32;
    variance.sqrt()
}

/// Compute median of values.
fn median(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }

    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[mid]
    }
}

/// Compute linear regression slope using least squares.
fn compute_slope(values: &[f32]) -> f32 {
    if values.len() < 2 {
        return 0.0;
    }

    let n = values.len() as f32;
    let x_mean = (n - 1.0) / 2.0; // Mean of 0, 1, 2, ..., n-1
    let y_mean = mean(values);

    let mut numerator = 0.0;
    let mut denominator = 0.0;

    for (i, &y) in values.iter().enumerate() {
        let x = i as f32;
        numerator += (x - x_mean) * (y - y_mean);
        denominator += (x - x_mean).powi(2);
    }

    if denominator == 0.0 {
        0.0
    } else {
        numerator / denominator
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mean() {
        assert_eq!(mean(&[1.0, 2.0, 3.0]), 2.0);
        assert_eq!(mean(&[]), 0.0);
        assert_eq!(mean(&[5.0]), 5.0);
    }

    #[test]
    fn test_std_dev() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let m = mean(&values);
        let std = std_dev(&values, m);
        assert!((std - 1.414).abs() < 0.01); // std ≈ sqrt(2)
    }

    #[test]
    fn test_median() {
        assert_eq!(median(&[1.0, 2.0, 3.0]), 2.0);
        assert_eq!(median(&[1.0, 2.0, 3.0, 4.0]), 2.5);
        assert_eq!(median(&[5.0]), 5.0);
    }

    #[test]
    fn test_compute_slope() {
        // Perfectly linear: y = 2x
        let values = vec![0.0, 2.0, 4.0, 6.0, 8.0];
        let slope = compute_slope(&values);
        assert!((slope - 2.0).abs() < 0.01);

        // Flat line
        let flat = vec![5.0, 5.0, 5.0, 5.0];
        assert_eq!(compute_slope(&flat), 0.0);

        // Decreasing: y = -x + 10
        let decreasing = vec![10.0, 9.0, 8.0, 7.0, 6.0];
        let slope_down = compute_slope(&decreasing);
        assert!((slope_down + 1.0).abs() < 0.01);
    }

    #[test]
    fn test_gradient_explosion() {
        // Test critical explosion (>10x baseline)
        let gradients = vec![0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.5, 1.5, 1.5, 1.5];
        let advice = check_gradient_explosion(&gradients, 1e-4);
        assert!(advice.is_some());
        let advice = advice.unwrap();
        assert_eq!(advice.urgency, Urgency::Critical);
        assert!(advice.suggested_lr < 1e-4);

        // Test high priority explosion (>5x but <10x baseline)
        let gradients2 = vec![0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.7, 0.7, 0.7, 0.7];
        let advice2 = check_gradient_explosion(&gradients2, 1e-4);
        assert!(advice2.is_some());
        let advice2 = advice2.unwrap();
        assert_eq!(advice2.urgency, Urgency::High);
        assert!(advice2.suggested_lr < 1e-4);
    }

    #[test]
    fn test_loss_oscillation() {
        let losses = vec![2.0, 1.5, 2.2, 1.4, 2.3, 1.3, 2.4, 1.2, 2.5, 1.1, 2.6, 1.0];
        let advice = check_loss_oscillation(&losses, 1e-4, 0.10, TrainingPhase::Stable);
        assert!(advice.is_some());
        let advice = advice.unwrap();
        assert_eq!(advice.issue, Issue::LossOscillation);
        assert!(advice.suggested_lr < 1e-4);
    }

    #[test]
    fn test_loss_plateau() {
        // Create a flat loss curve with tiny decreasing trend (to satisfy slope <= 0 condition)
        let mut losses = vec![2.0; 60];
        for (i, loss) in losses.iter_mut().enumerate() {
            // Very slight downward trend: 2.0 -> 1.9994 over 60 steps
            // This gives slope = -0.0001 which is well below the 0.001 threshold
            *loss -= i as f32 * 0.00001;
        }

        let advice = check_loss_plateau(&losses, 1e-4, 1000, 0.001);
        assert!(advice.is_some(), "Should detect plateau with normalized_slope < 0.001");
        let advice = advice.unwrap();
        assert_eq!(advice.issue, Issue::LossPlateau);
        assert!(advice.suggested_lr > 1e-4);
    }

    #[test]
    fn test_loss_increase() {
        let mut losses = vec![2.0; 20];
        // Simulate loss increase in second half
        for i in 10..20 {
            losses[i] = 2.3;
        }

        let advice = check_loss_increase(&losses, 1e-4, TrainingPhase::Stable);
        assert!(advice.is_some());
        let advice = advice.unwrap();
        assert_eq!(advice.issue, Issue::LossIncrease);
        assert!(advice.suggested_lr < 1e-4);
    }

    #[test]
    fn test_gradient_vanishing() {
        // Need more gradients in baseline and more extreme vanishing
        let mut gradients = vec![0.1; 10]; // Baseline
        gradients.extend(vec![0.0005, 0.0005, 0.0005, 0.0005, 0.0005]); // Recent vanishing
        let advice = check_gradient_vanishing(&gradients, 1e-4);
        assert!(advice.is_some());
        let advice = advice.unwrap();
        assert_eq!(advice.issue, Issue::GradientVanishing);
        assert!(advice.suggested_lr > 1e-4);
    }

    #[test]
    fn test_gradient_instability() {
        let gradients = vec![
            0.1, 0.5, 0.08, 0.6, 0.09, 0.55, 0.07, 0.65, 0.11, 0.5, 0.1, 0.6, 0.09, 0.55, 0.08,
            0.6, 0.1, 0.5, 0.09, 0.6,
        ];
        let advice = check_gradient_instability(&gradients, 1e-4);
        assert!(advice.is_some());
        let advice = advice.unwrap();
        assert_eq!(advice.issue, Issue::GradientInstability);
        assert!(advice.suggested_lr < 1e-4);
    }

    #[test]
    fn test_healthy_training() {
        // Realistic training: exponential-ish decay that stays in a reasonable range
        // Loss goes from 2.5 to ~2.0, smooth decrease
        let losses: Vec<f32> = (0..50)
            .map(|i| {
                let progress = i as f32 / 50.0;
                2.5 * (1.0 - 0.2 * progress) + (i as f32 * 0.001).sin() * 0.01 // Small noise
            })
            .collect();
        // Stable gradients
        let gradients = vec![0.3; 50];

        let advice = analyze_lr(&losses, &gradients, 1e-4, 500, TrainingPhase::Stable);
        if let Some(ref a) = advice {
            eprintln!("Unexpected advice: {:?}", a);
        }
        assert!(advice.is_none()); // No issues detected
    }

    #[test]
    fn test_warmup_tolerance() {
        // Same oscillation, but in warmup phase
        let losses = vec![2.0, 1.5, 2.2, 1.4, 2.3, 1.3, 2.4, 1.2, 2.5, 1.1, 2.6, 1.0];
        let gradients = vec![0.3; 12];

        // Should trigger in stable phase
        let advice_stable = analyze_lr(&losses, &gradients, 1e-4, 500, TrainingPhase::Stable);
        assert!(advice_stable.is_some());

        // Should be lower urgency in warmup
        let advice_warmup = analyze_lr(&losses, &gradients, 1e-4, 50, TrainingPhase::Warmup);
        if let Some(advice) = advice_warmup {
            assert_eq!(advice.urgency, Urgency::Medium); // Not High
        }
    }

    #[test]
    fn test_insufficient_samples() {
        let losses = vec![2.0, 1.9, 1.8];
        let gradients = vec![0.3, 0.3, 0.3];

        let advice = analyze_lr(&losses, &gradients, 1e-4, 100, TrainingPhase::Stable);
        assert!(advice.is_none()); // Not enough data
    }

    #[test]
    fn test_advice_formatting() {
        let advice = LRAdvice {
            current_lr: 1e-4,
            suggested_lr: 5e-5,
            reason: "Test reason".to_string(),
            urgency: Urgency::High,
            issue: Issue::LossOscillation,
        };

        assert_eq!(advice.multiplier(), 0.5);
        assert_eq!(advice.percentage_change(), -50.0);

        let formatted = advice.format();
        assert!(formatted.contains("HIGH"));
        assert!(formatted.contains("decrease"));
        assert!(formatted.contains("50.0%"));
    }
}
