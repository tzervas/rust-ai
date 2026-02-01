//! High-precision timing utilities for training metrics.
//!
//! This module provides low-overhead timing with multiple granularity levels
//! for both wall-clock and GPU compute time measurement.
//!
//! # Why Multiple Granularities?
//!
//! Training operations span a wide range of durations:
//! - Full training steps: milliseconds
//! - Individual kernel launches: microseconds
//! - Memory transfers: nanoseconds
//! - Fine-grained profiling: sub-nanosecond (interpolated)
//!
//! By storing time in nanoseconds and providing accessors for all granularities,
//! we enable precise analysis without repeated unit conversions during collection.
//!
//! # Why Separate GPU Timing?
//!
//! Wall-clock time includes CPU overhead, synchronization, and scheduling delays.
//! GPU compute time measures only the actual kernel execution, providing:
//! - Accurate kernel performance analysis
//! - Overlap detection (CPU work during GPU execution)
//! - True computational cost without system noise
//!
//! # Overhead Considerations
//!
//! - `Instant::now()` is ~20-25ns on modern systems
//! - We store raw `u64` nanoseconds to avoid float conversion overhead
//! - GPU timing uses CUDA events which are essentially free (async recording)
//! - Conversions to other units are computed on-demand, not during collection

use serde::{Deserialize, Serialize};
use std::time::Instant;

// ============================================================================
// Core Duration Type
// ============================================================================

/// High-precision duration stored as nanoseconds.
///
/// This type provides zero-cost storage with on-demand conversion to any
/// time granularity. The internal representation is nanoseconds (u64),
/// which provides ~584 years of range - sufficient for any training run.
///
/// # Granularity Notes
///
/// - **Nanoseconds**: Native precision of `std::time::Instant`
/// - **Picoseconds**: Interpolated (multiply ns by 1000) - useful for API
///   consistency but not actual hardware precision
/// - **Microseconds/Milliseconds**: Derived from nanoseconds
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Duration {
    /// Duration in nanoseconds.
    nanos: u64,
}

impl Duration {
    /// Zero duration constant.
    pub const ZERO: Self = Self { nanos: 0 };

    /// Creates a duration from nanoseconds.
    #[inline]
    #[must_use]
    pub const fn from_nanos(nanos: u64) -> Self {
        Self { nanos }
    }

    /// Creates a duration from microseconds.
    #[inline]
    #[must_use]
    pub const fn from_micros(micros: u64) -> Self {
        Self {
            nanos: micros * 1_000,
        }
    }

    /// Creates a duration from milliseconds.
    #[inline]
    #[must_use]
    pub const fn from_millis(millis: u64) -> Self {
        Self {
            nanos: millis * 1_000_000,
        }
    }

    /// Creates a duration from seconds.
    #[inline]
    #[must_use]
    pub const fn from_secs(secs: u64) -> Self {
        Self {
            nanos: secs * 1_000_000_000,
        }
    }

    /// Returns the duration in picoseconds (interpolated).
    ///
    /// Note: Actual hardware precision is nanoseconds. Picosecond values
    /// are derived by multiplying nanoseconds by 1000 for API consistency.
    #[inline]
    #[must_use]
    pub const fn as_picos(&self) -> u128 {
        self.nanos as u128 * 1_000
    }

    /// Returns the duration in nanoseconds.
    #[inline]
    #[must_use]
    pub const fn as_nanos(&self) -> u64 {
        self.nanos
    }

    /// Returns the duration in microseconds.
    #[inline]
    #[must_use]
    pub const fn as_micros(&self) -> u64 {
        self.nanos / 1_000
    }

    /// Returns the duration in milliseconds.
    #[inline]
    #[must_use]
    pub const fn as_millis(&self) -> u64 {
        self.nanos / 1_000_000
    }

    /// Returns the duration in seconds.
    #[inline]
    #[must_use]
    pub const fn as_secs(&self) -> u64 {
        self.nanos / 1_000_000_000
    }

    /// Returns the duration in picoseconds as f64.
    #[inline]
    #[must_use]
    pub fn as_picos_f64(&self) -> f64 {
        self.nanos as f64 * 1_000.0
    }

    /// Returns the duration in nanoseconds as f64.
    #[inline]
    #[must_use]
    pub fn as_nanos_f64(&self) -> f64 {
        self.nanos as f64
    }

    /// Returns the duration in microseconds as f64.
    #[inline]
    #[must_use]
    pub fn as_micros_f64(&self) -> f64 {
        self.nanos as f64 / 1_000.0
    }

    /// Returns the duration in milliseconds as f64.
    #[inline]
    #[must_use]
    pub fn as_millis_f64(&self) -> f64 {
        self.nanos as f64 / 1_000_000.0
    }

    /// Returns the duration in seconds as f64.
    #[inline]
    #[must_use]
    pub fn as_secs_f64(&self) -> f64 {
        self.nanos as f64 / 1_000_000_000.0
    }

    /// Adds two durations.
    #[inline]
    #[must_use]
    pub const fn saturating_add(self, other: Self) -> Self {
        Self {
            nanos: self.nanos.saturating_add(other.nanos),
        }
    }

    /// Subtracts two durations.
    #[inline]
    #[must_use]
    pub const fn saturating_sub(self, other: Self) -> Self {
        Self {
            nanos: self.nanos.saturating_sub(other.nanos),
        }
    }

    /// Returns whether this duration is zero.
    #[inline]
    #[must_use]
    pub const fn is_zero(&self) -> bool {
        self.nanos == 0
    }
}

impl std::ops::Add for Duration {
    type Output = Self;

    #[inline]
    fn add(self, other: Self) -> Self {
        Self {
            nanos: self.nanos + other.nanos,
        }
    }
}

impl std::ops::AddAssign for Duration {
    #[inline]
    fn add_assign(&mut self, other: Self) {
        self.nanos += other.nanos;
    }
}

impl std::ops::Sub for Duration {
    type Output = Self;

    #[inline]
    fn sub(self, other: Self) -> Self {
        Self {
            nanos: self.nanos.saturating_sub(other.nanos),
        }
    }
}

impl From<std::time::Duration> for Duration {
    #[inline]
    fn from(d: std::time::Duration) -> Self {
        Self {
            nanos: d.as_nanos() as u64,
        }
    }
}

impl From<Duration> for std::time::Duration {
    #[inline]
    fn from(d: Duration) -> Self {
        std::time::Duration::from_nanos(d.nanos)
    }
}

// ============================================================================
// Timer for Wall-Clock Measurement
// ============================================================================

/// Low-overhead wall-clock timer.
///
/// Uses `std::time::Instant` internally, which provides nanosecond precision
/// on most platforms with ~20-25ns overhead per measurement.
///
/// # Example
///
/// ```
/// use hybrid_predict_trainer_rs::timing::Timer;
///
/// let timer = Timer::start();
/// // ... do work ...
/// let elapsed = timer.elapsed();
/// println!("Took {} ns", elapsed.as_nanos());
/// ```
#[derive(Debug, Clone)]
pub struct Timer {
    start: Instant,
}

impl Timer {
    /// Starts a new timer.
    #[inline]
    #[must_use]
    pub fn start() -> Self {
        Self {
            start: Instant::now(),
        }
    }

    /// Returns the elapsed duration since the timer was started.
    #[inline]
    #[must_use]
    pub fn elapsed(&self) -> Duration {
        Duration::from(self.start.elapsed())
    }

    /// Resets the timer and returns the elapsed duration.
    #[inline]
    pub fn reset(&mut self) -> Duration {
        let elapsed = self.elapsed();
        self.start = Instant::now();
        elapsed
    }

    /// Returns the elapsed time in nanoseconds (convenience method).
    #[inline]
    #[must_use]
    pub fn elapsed_nanos(&self) -> u64 {
        self.elapsed().as_nanos()
    }

    /// Returns the elapsed time in milliseconds (convenience method).
    #[inline]
    #[must_use]
    pub fn elapsed_millis(&self) -> u64 {
        self.elapsed().as_millis()
    }
}

impl Default for Timer {
    fn default() -> Self {
        Self::start()
    }
}

// ============================================================================
// Timing Metrics Container
// ============================================================================

/// Complete timing information for an operation.
///
/// Captures both wall-clock time and GPU compute time (when available),
/// enabling accurate performance analysis that separates CPU overhead
/// from actual GPU computation.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct TimingMetrics {
    /// Wall-clock duration (includes CPU overhead, sync, scheduling).
    pub wall_clock: Duration,

    /// GPU compute duration (actual kernel execution time).
    /// None if GPU timing is not available or operation was CPU-only.
    pub gpu_compute: Option<Duration>,
}

impl TimingMetrics {
    /// Creates timing metrics with only wall-clock time.
    #[inline]
    #[must_use]
    pub const fn wall_clock_only(duration: Duration) -> Self {
        Self {
            wall_clock: duration,
            gpu_compute: None,
        }
    }

    /// Creates timing metrics with both wall-clock and GPU time.
    #[inline]
    #[must_use]
    pub const fn with_gpu(wall_clock: Duration, gpu_compute: Duration) -> Self {
        Self {
            wall_clock,
            gpu_compute: Some(gpu_compute),
        }
    }

    /// Returns the wall-clock time in milliseconds.
    #[inline]
    #[must_use]
    pub fn wall_clock_ms(&self) -> f64 {
        self.wall_clock.as_millis_f64()
    }

    /// Returns the wall-clock time in nanoseconds.
    #[inline]
    #[must_use]
    pub fn wall_clock_nanos(&self) -> u64 {
        self.wall_clock.as_nanos()
    }

    /// Returns the wall-clock time in picoseconds.
    #[inline]
    #[must_use]
    pub fn wall_clock_picos(&self) -> u128 {
        self.wall_clock.as_picos()
    }

    /// Returns the GPU compute time in milliseconds, if available.
    #[inline]
    #[must_use]
    pub fn gpu_compute_ms(&self) -> Option<f64> {
        self.gpu_compute.map(|d| d.as_millis_f64())
    }

    /// Returns the GPU compute time in nanoseconds, if available.
    #[inline]
    #[must_use]
    pub fn gpu_compute_nanos(&self) -> Option<u64> {
        self.gpu_compute.map(|d| d.as_nanos())
    }

    /// Returns the GPU compute time in picoseconds, if available.
    #[inline]
    #[must_use]
    pub fn gpu_compute_picos(&self) -> Option<u128> {
        self.gpu_compute.map(|d| d.as_picos())
    }

    /// Returns the CPU overhead (wall-clock minus GPU compute).
    ///
    /// This represents time spent on CPU operations, synchronization,
    /// and kernel launch overhead.
    #[inline]
    #[must_use]
    pub fn cpu_overhead(&self) -> Option<Duration> {
        self.gpu_compute
            .map(|gpu| self.wall_clock.saturating_sub(gpu))
    }

    /// Returns whether GPU timing is available.
    #[inline]
    #[must_use]
    pub const fn has_gpu_timing(&self) -> bool {
        self.gpu_compute.is_some()
    }
}

// ============================================================================
// GPU Timer (Feature-Gated)
// ============================================================================

/// GPU timer using CUDA events for accurate kernel timing.
///
/// CUDA events are recorded asynchronously on the GPU command stream,
/// providing accurate measurement of actual kernel execution time without
/// including CPU-side overhead.
///
/// # Overhead
///
/// CUDA event recording is essentially free - it simply inserts a timestamp
/// marker into the command stream. The actual timing query happens later
/// and can be batched for efficiency.
#[cfg(feature = "cuda")]
pub struct GpuTimer {
    // Placeholder for CUDA event handles
    // In a real implementation, these would be cubecl::cuda::Event or similar
    _start_event: (),
    _end_event: (),
    started: bool,
}

#[cfg(feature = "cuda")]
impl GpuTimer {
    /// Creates a new GPU timer.
    #[must_use]
    pub fn new() -> Self {
        Self {
            _start_event: (),
            _end_event: (),
            started: false,
        }
    }

    /// Records the start event on the GPU stream.
    pub fn start(&mut self) {
        // In real implementation: cudaEventRecord(start_event, stream)
        self.started = true;
    }

    /// Records the end event on the GPU stream.
    pub fn stop(&mut self) {
        // In real implementation: cudaEventRecord(end_event, stream)
    }

    /// Synchronizes and returns the elapsed GPU time.
    ///
    /// This blocks until the GPU has completed all work up to the end event.
    #[must_use]
    pub fn elapsed(&self) -> Option<Duration> {
        if !self.started {
            return None;
        }

        // In real implementation:
        // cudaEventSynchronize(end_event)
        // cudaEventElapsedTime(&elapsed_ms, start_event, end_event)
        // Duration::from_nanos((elapsed_ms * 1_000_000.0) as u64)

        // Placeholder - returns None until CUDA integration is complete
        None
    }

    /// Resets the timer for reuse.
    pub fn reset(&mut self) {
        self.started = false;
    }
}

#[cfg(feature = "cuda")]
impl Default for GpuTimer {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Scoped Timer for RAII-style Timing
// ============================================================================

/// RAII-style timer that records duration when dropped.
///
/// Useful for timing scopes without explicit start/stop calls.
///
/// # Example
///
/// ```
/// use hybrid_predict_trainer_rs::timing::{ScopedTimer, Duration};
/// use std::cell::Cell;
///
/// let result: Cell<Duration> = Cell::new(Duration::ZERO);
/// {
///     let _timer = ScopedTimer::new(|d| result.set(d));
///     // ... do work ...
/// } // Duration recorded here
/// println!("Scope took {} ns", result.get().as_nanos());
/// ```
pub struct ScopedTimer<F: FnOnce(Duration)> {
    timer: Timer,
    callback: Option<F>,
}

impl<F: FnOnce(Duration)> ScopedTimer<F> {
    /// Creates a new scoped timer with a callback.
    #[inline]
    #[must_use]
    pub fn new(callback: F) -> Self {
        Self {
            timer: Timer::start(),
            callback: Some(callback),
        }
    }

    /// Manually stops the timer and invokes the callback.
    ///
    /// This consumes the timer, preventing the Drop impl from running.
    #[inline]
    pub fn stop(mut self) -> Duration {
        let elapsed = self.timer.elapsed();
        if let Some(cb) = self.callback.take() {
            cb(elapsed);
        }
        elapsed
    }
}

impl<F: FnOnce(Duration)> Drop for ScopedTimer<F> {
    fn drop(&mut self) {
        if let Some(cb) = self.callback.take() {
            cb(self.timer.elapsed());
        }
    }
}

// ============================================================================
// Statistics Accumulator
// ============================================================================

/// Accumulator for timing statistics with minimal overhead.
///
/// Tracks min, max, sum, and count for computing averages without
/// storing individual measurements.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingStats {
    /// Minimum observed duration.
    pub min: Duration,

    /// Maximum observed duration.
    pub max: Duration,

    /// Sum of all durations (for computing average).
    pub sum: Duration,

    /// Number of measurements.
    pub count: u64,
}

impl TimingStats {
    /// Creates a new empty stats accumulator.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            min: Duration { nanos: u64::MAX },
            max: Duration::ZERO,
            sum: Duration::ZERO,
            count: 0,
        }
    }

    /// Records a new duration measurement.
    #[inline]
    pub fn record(&mut self, duration: Duration) {
        if duration.nanos < self.min.nanos {
            self.min = duration;
        }
        if duration.nanos > self.max.nanos {
            self.max = duration;
        }
        self.sum = self.sum.saturating_add(duration);
        self.count += 1;
    }

    /// Returns the average duration.
    #[inline]
    #[must_use]
    pub fn average(&self) -> Duration {
        if self.count == 0 {
            Duration::ZERO
        } else {
            Duration::from_nanos(self.sum.nanos / self.count)
        }
    }

    /// Returns the average in milliseconds.
    #[inline]
    #[must_use]
    pub fn average_ms(&self) -> f64 {
        self.average().as_millis_f64()
    }

    /// Returns the average in nanoseconds.
    #[inline]
    #[must_use]
    pub fn average_nanos(&self) -> u64 {
        self.average().as_nanos()
    }

    /// Returns the average in picoseconds.
    #[inline]
    #[must_use]
    pub fn average_picos(&self) -> u128 {
        self.average().as_picos()
    }

    /// Resets all statistics.
    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// Merges another stats accumulator into this one.
    pub fn merge(&mut self, other: &Self) {
        if other.count == 0 {
            return;
        }
        if other.min.nanos < self.min.nanos {
            self.min = other.min;
        }
        if other.max.nanos > self.max.nanos {
            self.max = other.max;
        }
        self.sum = self.sum.saturating_add(other.sum);
        self.count += other.count;
    }
}

impl Default for TimingStats {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_duration_conversions() {
        let d = Duration::from_millis(1);
        assert_eq!(d.as_millis(), 1);
        assert_eq!(d.as_micros(), 1_000);
        assert_eq!(d.as_nanos(), 1_000_000);
        assert_eq!(d.as_picos(), 1_000_000_000);
    }

    #[test]
    fn test_duration_arithmetic() {
        let a = Duration::from_nanos(100);
        let b = Duration::from_nanos(50);

        assert_eq!((a + b).as_nanos(), 150);
        assert_eq!((a - b).as_nanos(), 50);
        assert_eq!(a.saturating_sub(Duration::from_nanos(200)).as_nanos(), 0);
    }

    #[test]
    fn test_timer_basic() {
        let timer = Timer::start();
        std::thread::sleep(std::time::Duration::from_millis(1));
        let elapsed = timer.elapsed();

        // Should be at least 1ms but allow for timing variance
        assert!(elapsed.as_micros() >= 900);
    }

    #[test]
    fn test_timing_stats() {
        let mut stats = TimingStats::new();

        stats.record(Duration::from_nanos(100));
        stats.record(Duration::from_nanos(200));
        stats.record(Duration::from_nanos(300));

        assert_eq!(stats.count, 3);
        assert_eq!(stats.min.as_nanos(), 100);
        assert_eq!(stats.max.as_nanos(), 300);
        assert_eq!(stats.average().as_nanos(), 200);
    }

    #[test]
    fn test_timing_metrics() {
        let metrics = TimingMetrics::with_gpu(Duration::from_millis(10), Duration::from_millis(8));

        assert_eq!(metrics.wall_clock_ms(), 10.0);
        assert_eq!(metrics.gpu_compute_ms(), Some(8.0));
        assert_eq!(metrics.cpu_overhead().unwrap().as_millis(), 2);
    }

    #[test]
    fn test_duration_f64_precision() {
        let d = Duration::from_nanos(1_500_000); // 1.5ms

        assert!((d.as_millis_f64() - 1.5).abs() < 0.0001);
        assert!((d.as_micros_f64() - 1500.0).abs() < 0.0001);
    }

    #[test]
    fn test_std_duration_conversion() {
        let std_dur = std::time::Duration::from_millis(42);
        let our_dur: Duration = std_dur.into();
        let back: std::time::Duration = our_dur.into();

        assert_eq!(std_dur, back);
    }
}
