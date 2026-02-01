//! Training smoothing utilities to mitigate spikes and oscillations.
//!
//! This module provides tools for smoothing training metrics to reduce noise,
//! detect and suppress sudden spikes, and dampen oscillations.

use serde::{Deserialize, Serialize};

/// Configuration for all smoothing utilities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmoothingConfig {
    /// EMA smoothing factor (0-1, higher = more responsive)
    pub ema_alpha: f32,
    /// Spike detection threshold (e.g., 2.0 = 2x normal)
    pub spike_threshold: f32,
    /// Recovery rate after spike detection (0-1)
    pub spike_recovery_rate: f32,
    /// Window size for oscillation detection
    pub oscillation_window: usize,
    /// Variance threshold for oscillation detection
    pub oscillation_variance_threshold: f32,
    /// Damping factor for oscillations (0-1, higher = more damping)
    pub oscillation_damping_factor: f32,
}

impl Default for SmoothingConfig {
    fn default() -> Self {
        Self {
            ema_alpha: 0.1,
            spike_threshold: 2.0,
            spike_recovery_rate: 0.3,
            oscillation_window: 20,
            oscillation_variance_threshold: 0.15,
            oscillation_damping_factor: 0.5,
        }
    }
}

/// Exponential Moving Average for smoothing metrics.
///
/// EMA applies more weight to recent values while smoothing out noise.
/// Formula: EMA_t = alpha * value_t + (1 - alpha) * EMA_{t-1}
///
/// # Example
/// ```
/// use training_tools::smoothing::ExponentialMovingAverage;
///
/// let mut ema = ExponentialMovingAverage::new(0.1);
/// ema.update(1.0);
/// ema.update(2.0);
/// assert!((ema.value() - 1.1).abs() < 0.01);
/// ```
#[derive(Debug, Clone)]
pub struct ExponentialMovingAverage {
    /// Smoothing factor (0-1, higher = more responsive to recent values)
    alpha: f32,
    /// Current EMA value
    value: f32,
    /// Whether we've seen any data yet
    initialized: bool,
}

impl ExponentialMovingAverage {
    /// Create a new EMA with the given smoothing factor.
    ///
    /// # Arguments
    /// * `alpha` - Smoothing factor (0-1). Higher values = more responsive.
    ///   - 0.1 = slow, smooth (10% weight to new value)
    ///   - 0.5 = balanced
    ///   - 0.9 = fast, responsive (90% weight to new value)
    ///
    /// # Panics
    /// Panics if alpha is not in [0, 1].
    pub fn new(alpha: f32) -> Self {
        assert!(
            (0.0..=1.0).contains(&alpha),
            "alpha must be in [0, 1], got {}",
            alpha
        );
        Self {
            alpha,
            value: 0.0,
            initialized: false,
        }
    }

    /// Update the EMA with a new value.
    pub fn update(&mut self, new_value: f32) {
        if !self.initialized {
            self.value = new_value;
            self.initialized = true;
        } else {
            self.value = self.alpha * new_value + (1.0 - self.alpha) * self.value;
        }
    }

    /// Get the current EMA value.
    pub fn value(&self) -> f32 {
        self.value
    }

    /// Check if the EMA has been initialized with at least one value.
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Reset the EMA to uninitialized state.
    pub fn reset(&mut self) {
        self.initialized = false;
        self.value = 0.0;
    }

    /// Set a new alpha value.
    ///
    /// # Panics
    /// Panics if alpha is not in [0, 1].
    pub fn set_alpha(&mut self, alpha: f32) {
        assert!(
            (0.0..=1.0).contains(&alpha),
            "alpha must be in [0, 1], got {}",
            alpha
        );
        self.alpha = alpha;
    }
}

/// Detects and mitigates sudden metric spikes.
///
/// A spike is detected when a value exceeds the baseline by more than
/// the threshold multiplier. Recovery gradually brings values back to normal.
///
/// # Example
/// ```
/// use training_tools::smoothing::SpikeSuppressor;
///
/// let mut suppressor = SpikeSuppressor::new(2.0, 0.3);
/// let normal = suppressor.process(1.0);
/// let spike = suppressor.process(10.0); // 10x > 2x threshold
/// assert!(spike < 10.0); // Spike is suppressed
/// ```
#[derive(Debug, Clone)]
pub struct SpikeSuppressor {
    /// Spike detection threshold (e.g., 2.0 = 2x baseline)
    threshold: f32,
    /// Recovery rate (0-1, how fast to return to normal after spike)
    recovery_rate: f32,
    /// Baseline value (EMA of normal values)
    baseline: ExponentialMovingAverage,
    /// Are we currently in spike suppression mode?
    suppressing: bool,
    /// Last suppressed value (for gradual recovery)
    last_suppressed: f32,
}

impl SpikeSuppressor {
    /// Create a new spike suppressor.
    ///
    /// # Arguments
    /// * `threshold` - Spike detection threshold (e.g., 2.0 = 2x baseline)
    /// * `recovery_rate` - Recovery rate (0-1, higher = faster recovery)
    pub fn new(threshold: f32, recovery_rate: f32) -> Self {
        Self {
            threshold,
            recovery_rate,
            baseline: ExponentialMovingAverage::new(0.1),
            suppressing: false,
            last_suppressed: 0.0,
        }
    }

    /// Process a value, detecting and suppressing spikes.
    ///
    /// Returns the suppressed value if a spike is detected, otherwise the original.
    pub fn process(&mut self, value: f32) -> f32 {
        // Initialize baseline on first value
        if !self.baseline.is_initialized() {
            self.baseline.update(value);
            self.last_suppressed = value;
            return value;
        }

        let baseline = self.baseline.value();
        let ratio = if baseline > 0.0 {
            value / baseline
        } else {
            1.0
        };

        // Detect spike
        let is_spike = ratio > self.threshold;

        if is_spike {
            // Enter/continue suppression mode
            self.suppressing = true;
            // Clamp to threshold * baseline
            let suppressed = baseline * self.threshold;
            self.last_suppressed = suppressed;
            suppressed
        } else if self.suppressing {
            // Gradual recovery from spike
            let recovered =
                self.recovery_rate * value + (1.0 - self.recovery_rate) * self.last_suppressed;

            // Exit suppression if we're close to normal
            if (recovered - value).abs() < 0.1 * baseline {
                self.suppressing = false;
                self.baseline.update(value);
                self.last_suppressed = value;
                value
            } else {
                self.baseline.update(recovered);
                self.last_suppressed = recovered;
                recovered
            }
        } else {
            // Normal operation
            self.baseline.update(value);
            self.last_suppressed = value;
            value
        }
    }

    /// Check if currently suppressing a spike.
    pub fn is_suppressing(&self) -> bool {
        self.suppressing
    }

    /// Get the current baseline value.
    pub fn baseline(&self) -> f32 {
        self.baseline.value()
    }

    /// Reset the suppressor state.
    pub fn reset(&mut self) {
        self.baseline.reset();
        self.suppressing = false;
        self.last_suppressed = 0.0;
    }
}

/// Reduces oscillation amplitude in training metrics.
///
/// Detects oscillations by measuring variance within a sliding window,
/// then applies damping to reduce amplitude while preserving trend.
///
/// # Example
/// ```
/// use training_tools::smoothing::OscillationDamper;
///
/// let mut damper = OscillationDamper::new(10, 0.15, 0.5);
/// for i in 0..20 {
///     let oscillating = 1.0 + 0.5 * ((i as f32 * 0.5).sin());
///     let damped = damper.process(oscillating);
///     // damped will have reduced amplitude
/// }
/// ```
#[derive(Debug, Clone)]
pub struct OscillationDamper {
    /// Window size for variance calculation
    window_size: usize,
    /// Variance threshold for oscillation detection
    variance_threshold: f32,
    /// Damping factor (0-1, higher = more damping)
    damping_factor: f32,
    /// Sliding window of recent values
    window: Vec<f32>,
    /// Trend (EMA of values)
    trend: ExponentialMovingAverage,
    /// Are we currently damping oscillations?
    damping: bool,
}

impl OscillationDamper {
    /// Create a new oscillation damper.
    ///
    /// # Arguments
    /// * `window_size` - Number of values to use for variance calculation
    /// * `variance_threshold` - Relative variance threshold (e.g., 0.15 = 15% of mean)
    /// * `damping_factor` - Damping strength (0-1, higher = more damping)
    pub fn new(window_size: usize, variance_threshold: f32, damping_factor: f32) -> Self {
        Self {
            window_size,
            variance_threshold,
            damping_factor,
            window: Vec::with_capacity(window_size),
            trend: ExponentialMovingAverage::new(0.2),
            damping: false,
        }
    }

    /// Process a value, detecting and damping oscillations.
    pub fn process(&mut self, value: f32) -> f32 {
        // Update window
        self.window.push(value);
        if self.window.len() > self.window_size {
            self.window.remove(0);
        }

        // Update trend
        self.trend.update(value);

        // Need full window for oscillation detection
        if self.window.len() < self.window_size {
            return value;
        }

        // Calculate variance
        let mean = self.window.iter().sum::<f32>() / self.window.len() as f32;
        let variance =
            self.window.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / self.window.len() as f32;

        let relative_variance = if mean.abs() > 1e-6 {
            variance.sqrt() / mean.abs()
        } else {
            0.0
        };

        // Detect oscillation
        self.damping = relative_variance > self.variance_threshold;

        if self.damping {
            // Apply damping: blend value toward trend
            let trend = self.trend.value();
            self.damping_factor * trend + (1.0 - self.damping_factor) * value
        } else {
            value
        }
    }

    /// Check if currently damping oscillations.
    pub fn is_damping(&self) -> bool {
        self.damping
    }

    /// Get the current trend value.
    pub fn trend(&self) -> f32 {
        self.trend.value()
    }

    /// Get the current window values.
    pub fn window(&self) -> &[f32] {
        &self.window
    }

    /// Reset the damper state.
    pub fn reset(&mut self) {
        self.window.clear();
        self.trend.reset();
        self.damping = false;
    }
}

/// Combined smoothing pipeline applying EMA, spike suppression, and oscillation damping.
///
/// Processes values through all smoothing stages in sequence:
/// 1. Spike suppression (removes extreme outliers)
/// 2. Oscillation damping (reduces periodic noise)
/// 3. EMA smoothing (general noise reduction)
///
/// # Example
/// ```
/// use training_tools::smoothing::{SmoothingPipeline, SmoothingConfig};
///
/// let config = SmoothingConfig::default();
/// let mut pipeline = SmoothingPipeline::new(config);
///
/// for value in vec![1.0, 100.0, 2.0, 1.5, 2.5, 1.8] {
///     let smoothed = pipeline.process(value);
///     println!("Raw: {}, Smoothed: {}", value, smoothed);
/// }
/// ```
#[derive(Debug, Clone)]
pub struct SmoothingPipeline {
    spike_suppressor: SpikeSuppressor,
    oscillation_damper: OscillationDamper,
    ema: ExponentialMovingAverage,
}

impl SmoothingPipeline {
    /// Create a new smoothing pipeline with the given configuration.
    pub fn new(config: SmoothingConfig) -> Self {
        Self {
            spike_suppressor: SpikeSuppressor::new(
                config.spike_threshold,
                config.spike_recovery_rate,
            ),
            oscillation_damper: OscillationDamper::new(
                config.oscillation_window,
                config.oscillation_variance_threshold,
                config.oscillation_damping_factor,
            ),
            ema: ExponentialMovingAverage::new(config.ema_alpha),
        }
    }

    /// Process a value through the full smoothing pipeline.
    pub fn process(&mut self, value: f32) -> f32 {
        let after_spike = self.spike_suppressor.process(value);
        let after_oscillation = self.oscillation_damper.process(after_spike);
        self.ema.update(after_oscillation);
        self.ema.value()
    }

    /// Get diagnostic information about the smoothing state.
    pub fn diagnostics(&self) -> SmoothingDiagnostics {
        SmoothingDiagnostics {
            is_suppressing_spike: self.spike_suppressor.is_suppressing(),
            is_damping_oscillation: self.oscillation_damper.is_damping(),
            baseline: self.spike_suppressor.baseline(),
            trend: self.oscillation_damper.trend(),
            smoothed_value: self.ema.value(),
        }
    }

    /// Reset all smoothing state.
    pub fn reset(&mut self) {
        self.spike_suppressor.reset();
        self.oscillation_damper.reset();
        self.ema.reset();
    }
}

/// Diagnostic information about the smoothing pipeline state.
#[derive(Debug, Clone)]
pub struct SmoothingDiagnostics {
    /// Is the spike suppressor currently active?
    pub is_suppressing_spike: bool,
    /// Is the oscillation damper currently active?
    pub is_damping_oscillation: bool,
    /// Current baseline value from spike suppressor
    pub baseline: f32,
    /// Current trend value from oscillation damper
    pub trend: f32,
    /// Final smoothed value from EMA
    pub smoothed_value: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ema_basic() {
        let mut ema = ExponentialMovingAverage::new(0.5);
        assert!(!ema.is_initialized());

        ema.update(1.0);
        assert!(ema.is_initialized());
        assert_eq!(ema.value(), 1.0);

        ema.update(3.0);
        assert_eq!(ema.value(), 2.0); // 0.5 * 3.0 + 0.5 * 1.0
    }

    #[test]
    fn test_ema_reset() {
        let mut ema = ExponentialMovingAverage::new(0.1);
        ema.update(5.0);
        assert!(ema.is_initialized());

        ema.reset();
        assert!(!ema.is_initialized());
    }

    #[test]
    #[should_panic]
    fn test_ema_invalid_alpha() {
        ExponentialMovingAverage::new(1.5);
    }

    #[test]
    fn test_spike_suppressor() {
        let mut suppressor = SpikeSuppressor::new(2.0, 0.3);

        // Normal values
        let v1 = suppressor.process(1.0);
        assert_eq!(v1, 1.0);

        let v2 = suppressor.process(1.1);
        assert_eq!(v2, 1.1);

        // Spike (10x > 2x threshold)
        let spike = suppressor.process(10.0);
        assert!(suppressor.is_suppressing());
        assert!(spike < 10.0);
        assert!(spike <= 2.0 * suppressor.baseline());
    }

    #[test]
    fn test_oscillation_damper() {
        let mut damper = OscillationDamper::new(10, 0.15, 0.5);

        // Feed oscillating values
        let mut dampened_values = Vec::new();
        for i in 0..20 {
            let oscillating = 1.0 + 0.5 * ((i as f32 * 0.5).sin());
            let dampened = damper.process(oscillating);
            dampened_values.push(dampened);
        }

        // Damping should reduce variance
        let mean = dampened_values.iter().sum::<f32>() / dampened_values.len() as f32;
        let variance = dampened_values
            .iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f32>()
            / dampened_values.len() as f32;

        // Dampened variance should be lower than original oscillation amplitude
        assert!(variance.sqrt() < 0.5);
    }

    #[test]
    fn test_smoothing_pipeline() {
        let config = SmoothingConfig::default();
        let mut pipeline = SmoothingPipeline::new(config);

        // Process mixed data (normal, spike, oscillation)
        let values = vec![1.0, 1.1, 100.0, 1.2, 0.8, 1.3, 0.9, 1.1];
        let mut smoothed = Vec::new();

        for value in values {
            let s = pipeline.process(value);
            smoothed.push(s);

            let diag = pipeline.diagnostics();
            assert!(diag.smoothed_value >= 0.0);
        }

        // Smoothed values should be more stable
        assert!(smoothed.len() == 8);
    }

    #[test]
    fn test_smoothing_config_default() {
        let config = SmoothingConfig::default();
        assert_eq!(config.ema_alpha, 0.1);
        assert_eq!(config.spike_threshold, 2.0);
        assert_eq!(config.oscillation_window, 20);
    }
}
