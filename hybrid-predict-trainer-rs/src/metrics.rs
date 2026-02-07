//! Training metrics collection and reporting.
//!
//! Provides comprehensive metrics collection for monitoring training
//! progress, debugging issues, and evaluating the effectiveness of
//! predictive training.
//!
//! # Why High-Precision Timing?
//!
//! Training operations span multiple orders of magnitude in duration:
//! - Full training steps: milliseconds (1-100ms typical)
//! - Prediction operations: microseconds (10-500μs)
//! - Individual kernels: nanoseconds (100ns-10μs)
//!
//! By capturing nanosecond-precision timing and providing picosecond
//! accessors, we enable:
//! - Accurate overhead analysis for fast operations
//! - Meaningful comparisons between CPU and GPU execution
//! - Identification of micro-bottlenecks in hot paths
//!
//! # Why Separate GPU Timing?
//!
//! Wall-clock time conflates multiple factors:
//! - Actual GPU computation
//! - Kernel launch overhead
//! - Memory transfer time
//! - CPU-GPU synchronization
//!
//! GPU compute time (via CUDA events) isolates the actual work,
//! enabling accurate efficiency analysis and overlap detection.
//!
//! # Collected Metrics
//!
//! - **Step-level**: Loss, gradient norm, prediction error, phase, timing
//! - **Phase-level**: Duration, steps, average metrics
//! - **Aggregate**: Total speedup, backward reduction, loss quality
//!
//! # Output Formats
//!
//! Metrics can be exported as:
//! - JSON for programmatic analysis
//! - Console summary for monitoring
//! - Parquet for efficient storage (future)

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::timing::{Duration, TimingMetrics, TimingStats};
use crate::Phase;

/// Metrics for a single training step.
///
/// Captures comprehensive timing information at multiple granularities
/// for both wall-clock and GPU compute time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepMetrics {
    /// Training step number.
    pub step: u64,

    /// Loss value.
    pub loss: f32,

    /// Gradient norm.
    pub gradient_norm: f32,

    /// Current phase.
    pub phase: Phase,

    /// Whether this step used predictions.
    pub was_predicted: bool,

    /// Prediction error (if applicable).
    pub prediction_error: Option<f32>,

    /// Predictor confidence.
    pub confidence: f32,

    /// High-precision timing metrics.
    ///
    /// Includes wall-clock time (always) and GPU compute time (when available).
    /// Access via:
    /// - `timing.wall_clock.as_millis()` - milliseconds
    /// - `timing.wall_clock.as_nanos()` - nanoseconds
    /// - `timing.wall_clock.as_picos()` - picoseconds (interpolated)
    /// - `timing.gpu_compute_nanos()` - GPU time in nanoseconds
    pub timing: TimingMetrics,

    /// Wall-clock time in milliseconds (convenience, derived from timing).
    ///
    /// This field is kept for backward compatibility. Prefer using
    /// `timing.wall_clock.as_millis_f64()` for new code.
    pub time_ms: f64,

    /// Learning rate (if available).
    pub learning_rate: Option<f32>,
}

impl StepMetrics {
    /// Returns wall-clock time in nanoseconds.
    #[inline]
    #[must_use]
    pub fn time_nanos(&self) -> u64 {
        self.timing.wall_clock.as_nanos()
    }

    /// Returns wall-clock time in picoseconds (interpolated from nanoseconds).
    #[inline]
    #[must_use]
    pub fn time_picos(&self) -> u128 {
        self.timing.wall_clock.as_picos()
    }

    /// Returns GPU compute time in milliseconds, if available.
    #[inline]
    #[must_use]
    pub fn gpu_time_ms(&self) -> Option<f64> {
        self.timing.gpu_compute_ms()
    }

    /// Returns GPU compute time in nanoseconds, if available.
    #[inline]
    #[must_use]
    pub fn gpu_time_nanos(&self) -> Option<u64> {
        self.timing.gpu_compute_nanos()
    }

    /// Returns GPU compute time in picoseconds, if available.
    #[inline]
    #[must_use]
    pub fn gpu_time_picos(&self) -> Option<u128> {
        self.timing.gpu_compute_picos()
    }
}

/// Metrics for a completed training phase.
///
/// Includes aggregate timing at all granularities for the entire phase.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseMetrics {
    /// Phase type.
    pub phase: Phase,

    /// Starting step.
    pub start_step: u64,

    /// Ending step.
    pub end_step: u64,

    /// Number of steps executed.
    pub steps_executed: usize,

    /// Average loss during phase.
    pub average_loss: f32,

    /// Final loss at phase end.
    pub final_loss: f32,

    /// Average gradient norm.
    pub average_gradient_norm: f32,

    /// High-precision timing for the entire phase.
    pub timing: TimingMetrics,

    /// Total phase duration in milliseconds (convenience, derived from timing).
    ///
    /// Kept for backward compatibility. Prefer `timing.wall_clock.as_millis_f64()`.
    pub duration_ms: f64,

    /// Whether phase completed normally.
    pub completed_normally: bool,

    /// Prediction error (for predict phase).
    pub prediction_error: Option<f32>,
}

impl PhaseMetrics {
    /// Returns phase duration in nanoseconds.
    #[inline]
    #[must_use]
    pub fn duration_nanos(&self) -> u64 {
        self.timing.wall_clock.as_nanos()
    }

    /// Returns phase duration in picoseconds.
    #[inline]
    #[must_use]
    pub fn duration_picos(&self) -> u128 {
        self.timing.wall_clock.as_picos()
    }

    /// Returns GPU compute time for the phase, if available.
    #[inline]
    #[must_use]
    pub fn gpu_duration_nanos(&self) -> Option<u64> {
        self.timing.gpu_compute_nanos()
    }
}

/// Aggregate training statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TrainingStatistics {
    /// Total training steps.
    pub total_steps: u64,

    /// Steps spent in warmup.
    pub warmup_steps: usize,

    /// Steps spent in full training.
    pub full_steps: usize,

    /// Steps spent in prediction.
    pub predict_steps: usize,

    /// Steps spent in correction.
    pub correct_steps: usize,

    /// Percentage of backward passes avoided.
    pub backward_reduction_pct: f32,

    /// Wall-clock speedup factor vs traditional training.
    pub wall_clock_speedup: f32,

    /// Final training loss.
    pub final_loss: f32,

    /// Estimated baseline loss (traditional training).
    pub baseline_loss_estimate: f32,

    /// Loss gap percentage.
    pub loss_gap_pct: f32,

    /// Average prediction length.
    pub avg_predict_length: f32,

    /// Maximum prediction length achieved.
    pub max_predict_length: usize,

    /// Average predictor confidence.
    pub avg_confidence: f32,

    /// Prediction accuracy statistics.
    pub prediction_accuracy: PredictionAccuracy,

    /// Number of divergence events.
    pub divergence_events: usize,

    /// Number of intra-horizon micro-corrections applied.
    pub micro_corrections_applied: usize,

    /// Predictor overhead statistics.
    pub predictor_overhead: PredictorOverhead,
}

/// Prediction accuracy statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PredictionAccuracy {
    /// Mean absolute error of loss predictions.
    pub loss_mae: f32,

    /// Correlation between predicted and actual loss.
    pub loss_correlation: f32,

    /// Cosine similarity of predicted vs actual weight updates.
    pub weight_cosine_similarity: f32,
}

/// Predictor overhead statistics with multi-granularity timing.
///
/// Tracks encoding, prediction, and update times at nanosecond precision
/// for both wall-clock and GPU compute time.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PredictorOverhead {
    /// Encoding time statistics (state → features).
    pub encode_time: TimingStats,

    /// Prediction time statistics (dynamics model forward pass).
    pub predict_time: TimingStats,

    /// Update time statistics (online learning update).
    pub update_time: TimingStats,

    /// GPU-specific encoding time statistics.
    pub gpu_encode_time: TimingStats,

    /// GPU-specific prediction time statistics.
    pub gpu_predict_time: TimingStats,

    /// Average encoding time in milliseconds (convenience accessor).
    pub encode_time_ms_avg: f64,

    /// Average prediction time in milliseconds (convenience accessor).
    pub predict_time_ms_avg: f64,

    /// Average update time in milliseconds (convenience accessor).
    pub update_time_ms_avg: f64,

    /// Memory used by predictor in MB.
    pub memory_used_mb: f32,

    /// Percentage of step time spent on prediction.
    pub pct_of_step_time: f32,
}

impl PredictorOverhead {
    /// Records an encoding operation timing.
    pub fn record_encode(&mut self, wall_clock: Duration, gpu_time: Option<Duration>) {
        self.encode_time.record(wall_clock);
        if let Some(gpu) = gpu_time {
            self.gpu_encode_time.record(gpu);
        }
        self.encode_time_ms_avg = self.encode_time.average_ms();
    }

    /// Records a prediction operation timing.
    pub fn record_predict(&mut self, wall_clock: Duration, gpu_time: Option<Duration>) {
        self.predict_time.record(wall_clock);
        if let Some(gpu) = gpu_time {
            self.gpu_predict_time.record(gpu);
        }
        self.predict_time_ms_avg = self.predict_time.average_ms();
    }

    /// Records an update operation timing.
    pub fn record_update(&mut self, wall_clock: Duration) {
        self.update_time.record(wall_clock);
        self.update_time_ms_avg = self.update_time.average_ms();
    }

    /// Returns average encoding time in nanoseconds.
    #[inline]
    #[must_use]
    pub fn encode_time_nanos_avg(&self) -> u64 {
        self.encode_time.average_nanos()
    }

    /// Returns average encoding time in picoseconds.
    #[inline]
    #[must_use]
    pub fn encode_time_picos_avg(&self) -> u128 {
        self.encode_time.average_picos()
    }

    /// Returns average prediction time in nanoseconds.
    #[inline]
    #[must_use]
    pub fn predict_time_nanos_avg(&self) -> u64 {
        self.predict_time.average_nanos()
    }

    /// Returns average prediction time in picoseconds.
    #[inline]
    #[must_use]
    pub fn predict_time_picos_avg(&self) -> u128 {
        self.predict_time.average_picos()
    }

    /// Returns average GPU encoding time in nanoseconds.
    #[inline]
    #[must_use]
    pub fn gpu_encode_time_nanos_avg(&self) -> u64 {
        self.gpu_encode_time.average_nanos()
    }

    /// Returns average GPU prediction time in nanoseconds.
    #[inline]
    #[must_use]
    pub fn gpu_predict_time_nanos_avg(&self) -> u64 {
        self.gpu_predict_time.average_nanos()
    }
}

/// Divergence event record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DivergenceEvent {
    /// Step where divergence was detected.
    pub step: u64,

    /// Severity level.
    pub severity: String,

    /// Action taken.
    pub action: String,

    /// Whether recovery was successful.
    pub recovery_successful: bool,
}

/// Collector for training metrics with high-precision timing.
///
/// Aggregates metrics at step, phase, and training-run levels with
/// nanosecond precision timing for both wall-clock and GPU compute time.
pub struct MetricsCollector {
    /// Whether collection is enabled.
    enabled: bool,

    /// Step-level metrics (limited buffer).
    step_metrics: Vec<StepMetrics>,

    /// Phase-level metrics.
    phase_metrics: Vec<PhaseMetrics>,

    /// Divergence events.
    divergence_events: Vec<DivergenceEvent>,

    /// Running statistics.
    statistics: TrainingStatistics,

    /// Maximum step metrics to keep in memory.
    max_step_metrics: usize,

    /// Per-phase step counters.
    phase_step_counts: HashMap<Phase, usize>,

    /// Total wall-clock time in each phase (nanoseconds for precision).
    phase_times: HashMap<Phase, Duration>,

    /// Total GPU time in each phase (nanoseconds).
    phase_gpu_times: HashMap<Phase, Duration>,

    /// Per-phase timing statistics.
    phase_timing_stats: HashMap<Phase, TimingStats>,

    /// Prediction errors for accuracy tracking.
    prediction_errors: Vec<f32>,
}

impl MetricsCollector {
    /// Creates a new metrics collector.
    #[must_use]
    pub fn new(enabled: bool) -> Self {
        Self {
            enabled,
            step_metrics: Vec::with_capacity(10000),
            phase_metrics: Vec::new(),
            divergence_events: Vec::new(),
            statistics: TrainingStatistics::default(),
            max_step_metrics: 10000,
            phase_step_counts: HashMap::new(),
            phase_times: HashMap::new(),
            phase_gpu_times: HashMap::new(),
            phase_timing_stats: HashMap::new(),
            prediction_errors: Vec::new(),
        }
    }

    /// Records metrics for a training step.
    pub fn record_step(&mut self, metrics: StepMetrics) {
        if !self.enabled {
            return;
        }

        // Update per-phase counters
        *self.phase_step_counts.entry(metrics.phase).or_insert(0) += 1;

        // Update per-phase timing (high precision)
        *self
            .phase_times
            .entry(metrics.phase)
            .or_insert(Duration::ZERO) += metrics.timing.wall_clock;

        // Update GPU timing if available
        if let Some(gpu_time) = metrics.timing.gpu_compute {
            *self
                .phase_gpu_times
                .entry(metrics.phase)
                .or_insert(Duration::ZERO) += gpu_time;
        }

        // Update timing statistics
        self.phase_timing_stats
            .entry(metrics.phase)
            .or_default()
            .record(metrics.timing.wall_clock);

        // Track prediction errors
        if let Some(error) = metrics.prediction_error {
            self.prediction_errors.push(error);
        }

        // Update statistics
        self.statistics.total_steps = metrics.step;
        self.statistics.final_loss = metrics.loss;

        // Store step metrics (with eviction if needed)
        if self.step_metrics.len() >= self.max_step_metrics {
            self.step_metrics.remove(0);
        }
        self.step_metrics.push(metrics);
    }

    /// Records metrics for a training step from individual values.
    ///
    /// Convenience method that creates a `StepMetrics` and records it.
    ///
    /// # Arguments
    ///
    /// * `step` - Current training step
    /// * `loss` - Loss value
    /// * `phase` - Current phase
    /// * `was_predicted` - Whether predictions were used
    /// * `prediction_error` - Error between predicted and actual (if applicable)
    /// * `confidence` - Predictor confidence level
    ///
    /// # Returns
    ///
    /// The created `StepMetrics` struct.
    pub fn record_step_data(
        &mut self,
        step: u64,
        loss: f32,
        phase: Phase,
        was_predicted: bool,
        prediction_error: Option<f32>,
        confidence: f32,
    ) -> StepMetrics {
        let metrics = StepMetrics {
            step,
            loss,
            gradient_norm: 0.0, // Updated separately if available
            phase,
            was_predicted,
            prediction_error,
            confidence,
            timing: TimingMetrics::default(), // Will be updated by caller
            time_ms: 0.0,                     // Will be updated by caller
            learning_rate: None,
        };

        self.record_step(metrics.clone());
        metrics
    }

    /// Records metrics for a training step with timing information.
    ///
    /// # Arguments
    ///
    /// * `step` - Current training step
    /// * `loss` - Loss value
    /// * `phase` - Current phase
    /// * `was_predicted` - Whether predictions were used
    /// * `prediction_error` - Error between predicted and actual (if applicable)
    /// * `confidence` - Predictor confidence level
    /// * `timing` - High-precision timing metrics
    ///
    /// # Returns
    ///
    /// The created `StepMetrics` struct.
    #[allow(clippy::too_many_arguments)] // Convenience method mirrors StepMetrics fields
    pub fn record_step_with_timing(
        &mut self,
        step: u64,
        loss: f32,
        phase: Phase,
        was_predicted: bool,
        prediction_error: Option<f32>,
        confidence: f32,
        timing: TimingMetrics,
    ) -> StepMetrics {
        let metrics = StepMetrics {
            step,
            loss,
            gradient_norm: 0.0,
            phase,
            was_predicted,
            prediction_error,
            confidence,
            time_ms: timing.wall_clock.as_millis_f64(),
            timing,
            learning_rate: None,
        };

        self.record_step(metrics.clone());
        metrics
    }

    /// Records metrics for a completed phase.
    pub fn record_phase(&mut self, metrics: PhaseMetrics) {
        if !self.enabled {
            return;
        }

        self.phase_metrics.push(metrics);
    }

    /// Records a divergence event.
    pub fn record_divergence(&mut self, event: DivergenceEvent) {
        if !self.enabled {
            return;
        }

        self.divergence_events.push(event);
        self.statistics.divergence_events += 1;
    }

    /// Finalizes statistics computation.
    pub fn finalize(&mut self) {
        // Update phase step counts
        self.statistics.warmup_steps = *self.phase_step_counts.get(&Phase::Warmup).unwrap_or(&0);
        self.statistics.full_steps = *self.phase_step_counts.get(&Phase::Full).unwrap_or(&0);
        self.statistics.predict_steps = *self.phase_step_counts.get(&Phase::Predict).unwrap_or(&0);
        self.statistics.correct_steps = *self.phase_step_counts.get(&Phase::Correct).unwrap_or(&0);

        // Compute backward reduction
        // Backward passes occur in: Warmup, Full, and Correct phases
        // Predict phase skips backward passes (that's the whole point!)
        let total = self.statistics.total_steps as f32;
        let backward_steps = (self.statistics.warmup_steps
            + self.statistics.full_steps
            + self.statistics.correct_steps) as f32;
        if total > 0.0 {
            self.statistics.backward_reduction_pct = 100.0 * (1.0 - backward_steps / total);
        }

        // Compute prediction accuracy
        if !self.prediction_errors.is_empty() {
            let sum: f32 = self.prediction_errors.iter().sum();
            self.statistics.prediction_accuracy.loss_mae =
                sum / self.prediction_errors.len() as f32;
        }

        // Compute average prediction length
        let predict_phases: Vec<_> = self
            .phase_metrics
            .iter()
            .filter(|p| p.phase == Phase::Predict)
            .collect();

        if !predict_phases.is_empty() {
            let total_predict_steps: usize = predict_phases.iter().map(|p| p.steps_executed).sum();
            self.statistics.avg_predict_length =
                total_predict_steps as f32 / predict_phases.len() as f32;
            self.statistics.max_predict_length = predict_phases
                .iter()
                .map(|p| p.steps_executed)
                .max()
                .unwrap_or(0);
        }

        // Compute wall-clock speedup using high-precision timing
        let full_time = self
            .phase_times
            .get(&Phase::Full)
            .copied()
            .unwrap_or(Duration::ZERO);
        let predict_time = self
            .phase_times
            .get(&Phase::Predict)
            .copied()
            .unwrap_or(Duration::ZERO);

        if !predict_time.is_zero() && self.statistics.predict_steps > 0 {
            let full_time_per_step = if self.statistics.full_steps > 0 {
                full_time.as_nanos_f64() / self.statistics.full_steps as f64
            } else {
                1.0
            };
            let predict_time_per_step =
                predict_time.as_nanos_f64() / self.statistics.predict_steps as f64;

            if predict_time_per_step > 0.0 {
                let speedup_factor = full_time_per_step / predict_time_per_step;
                self.statistics.wall_clock_speedup = speedup_factor as f32;
            }
        }
    }

    /// Returns timing statistics for a specific phase.
    #[must_use]
    pub fn phase_timing_stats(&self, phase: Phase) -> Option<&TimingStats> {
        self.phase_timing_stats.get(&phase)
    }

    /// Returns total wall-clock time for a phase in nanoseconds.
    #[must_use]
    pub fn phase_total_time_nanos(&self, phase: Phase) -> u64 {
        self.phase_times.get(&phase).map_or(0, Duration::as_nanos)
    }

    /// Returns total GPU time for a phase in nanoseconds.
    #[must_use]
    pub fn phase_total_gpu_time_nanos(&self, phase: Phase) -> u64 {
        self.phase_gpu_times
            .get(&phase)
            .map_or(0, Duration::as_nanos)
    }

    /// Returns the current statistics.
    ///
    /// This method automatically finalizes the statistics before returning them,
    /// ensuring all derived metrics (backward_reduction_pct, avg_confidence, etc.)
    /// are up-to-date.
    #[must_use]
    pub fn statistics(&mut self) -> TrainingStatistics {
        self.finalize();
        self.statistics.clone()
    }

    /// Exports metrics to JSON.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        let export = MetricsExport {
            training_summary: self.statistics.clone(),
            phase_history: self.phase_metrics.clone(),
            divergence_events: self.divergence_events.clone(),
        };
        serde_json::to_string_pretty(&export)
    }

    /// Returns a console-friendly summary.
    #[must_use]
    pub fn summary(&self) -> String {
        let stats = &self.statistics;
        format!(
            "Training Summary:\n\
             ├─ Total Steps: {}\n\
             ├─ Phases: W={}, F={}, P={}, C={}\n\
             ├─ Backward Reduction: {:.1}%\n\
             ├─ Wall-Clock Speedup: {:.1}x\n\
             ├─ Final Loss: {:.4}\n\
             ├─ Avg Predict Length: {:.1}\n\
             ├─ Prediction MAE: {:.4}\n\
             └─ Divergence Events: {}",
            stats.total_steps,
            stats.warmup_steps,
            stats.full_steps,
            stats.predict_steps,
            stats.correct_steps,
            stats.backward_reduction_pct,
            stats.wall_clock_speedup,
            stats.final_loss,
            stats.avg_predict_length,
            stats.prediction_accuracy.loss_mae,
            stats.divergence_events
        )
    }

    /// Records a micro-correction event.
    ///
    /// Increments the count of intra-horizon micro-corrections applied
    /// during prediction phases.
    pub fn record_micro_correction(&mut self) {
        if self.enabled {
            self.statistics.micro_corrections_applied += 1;
        }
    }

    /// Resets all collected metrics.
    pub fn reset(&mut self) {
        self.step_metrics.clear();
        self.phase_metrics.clear();
        self.divergence_events.clear();
        self.statistics = TrainingStatistics::default();
        self.phase_step_counts.clear();
        self.phase_times.clear();
        self.phase_gpu_times.clear();
        self.phase_timing_stats.clear();
        self.prediction_errors.clear();
    }
}

/// Export structure for metrics serialization.
#[derive(Debug, Serialize, Deserialize)]
pub struct MetricsExport {
    /// Training summary statistics.
    pub training_summary: TrainingStatistics,

    /// Phase-level metrics history.
    pub phase_history: Vec<PhaseMetrics>,

    /// Divergence events.
    pub divergence_events: Vec<DivergenceEvent>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_step_metrics(step: u64, phase: Phase, time_nanos: u64) -> StepMetrics {
        StepMetrics {
            step,
            loss: 2.5,
            gradient_norm: 1.0,
            phase,
            was_predicted: phase == Phase::Predict,
            prediction_error: if phase == Phase::Predict {
                Some(0.05)
            } else {
                None
            },
            confidence: 0.9,
            timing: TimingMetrics::wall_clock_only(Duration::from_nanos(time_nanos)),
            time_ms: Duration::from_nanos(time_nanos).as_millis_f64(),
            learning_rate: Some(0.001),
        }
    }

    #[test]
    fn test_collector_disabled() {
        let mut collector = MetricsCollector::new(false);

        collector.record_step(make_step_metrics(1, Phase::Warmup, 10_000_000));

        // Should not record when disabled
        assert!(collector.step_metrics.is_empty());
    }

    #[test]
    fn test_collector_enabled() {
        let mut collector = MetricsCollector::new(true);

        collector.record_step(make_step_metrics(1, Phase::Warmup, 10_000_000));

        assert_eq!(collector.step_metrics.len(), 1);
    }

    #[test]
    fn test_finalize_statistics() {
        let mut collector = MetricsCollector::new(true);

        // Record some warmup steps (10ms each)
        for i in 0..10 {
            collector.record_step(make_step_metrics(i, Phase::Warmup, 10_000_000));
        }

        // Record some predict steps (5ms each)
        for i in 10..30 {
            collector.record_step(make_step_metrics(i, Phase::Predict, 5_000_000));
        }

        collector.finalize();

        let stats = collector.statistics();
        assert_eq!(stats.warmup_steps, 10);
        assert_eq!(stats.predict_steps, 20);
        assert!(stats.backward_reduction_pct > 0.0);
    }

    #[test]
    fn test_json_export() {
        let collector = MetricsCollector::new(true);
        let json = collector.to_json().unwrap();

        assert!(json.contains("training_summary"));
        assert!(json.contains("phase_history"));
    }

    #[test]
    fn test_timing_granularity() {
        let timing = TimingMetrics::wall_clock_only(Duration::from_nanos(1_500_000));

        // Test multiple granularities
        assert_eq!(timing.wall_clock.as_millis(), 1);
        assert_eq!(timing.wall_clock.as_micros(), 1500);
        assert_eq!(timing.wall_clock.as_nanos(), 1_500_000);
        assert_eq!(timing.wall_clock.as_picos(), 1_500_000_000);

        // Test f64 precision
        assert!((timing.wall_clock.as_millis_f64() - 1.5).abs() < 0.0001);
    }

    #[test]
    fn test_gpu_timing() {
        let timing = TimingMetrics::with_gpu(Duration::from_millis(10), Duration::from_millis(8));

        assert_eq!(timing.wall_clock_ms(), 10.0);
        assert_eq!(timing.gpu_compute_ms(), Some(8.0));
        assert!(timing.has_gpu_timing());

        // CPU overhead = wall_clock - gpu_compute
        let overhead = timing.cpu_overhead().unwrap();
        assert_eq!(overhead.as_millis(), 2);
    }

    #[test]
    fn test_phase_timing_stats() {
        let mut collector = MetricsCollector::new(true);

        // Record steps with varying times
        for i in 0..100 {
            let time_nanos = 1_000_000 + (i * 10_000); // 1ms + variance
            collector.record_step(make_step_metrics(i, Phase::Full, time_nanos));
        }

        let stats = collector.phase_timing_stats(Phase::Full).unwrap();
        assert_eq!(stats.count, 100);
        assert!(stats.min.as_nanos() >= 1_000_000);
        assert!(stats.max.as_nanos() <= 2_000_000);
    }
}
