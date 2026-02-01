//! Comprehensive GPU statistics monitoring.
//!
//! Queries nvidia-smi for detailed GPU metrics including:
//! - Memory usage and bandwidth
//! - Temperature and power
//! - Clock speeds and utilization
//! - Performance state

use std::process::Command;
use std::time::{Duration, Instant};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Comprehensive GPU statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuStats {
    /// GPU index
    pub device_idx: usize,
    /// GPU name (e.g., "NVIDIA GeForce RTX 5080")
    pub name: String,
    /// Driver version
    pub driver_version: String,
    /// CUDA version
    pub cuda_version: String,

    // Memory
    /// Total VRAM in bytes
    pub memory_total: u64,
    /// Used VRAM in bytes
    pub memory_used: u64,
    /// Free VRAM in bytes
    pub memory_free: u64,
    /// Memory utilization percentage
    pub memory_util: u32,

    // Utilization
    /// GPU compute utilization percentage
    pub gpu_util: u32,
    /// Encoder utilization percentage
    pub encoder_util: u32,
    /// Decoder utilization percentage
    pub decoder_util: u32,

    // Thermals
    /// GPU temperature in Celsius
    pub temperature: u32,
    /// Temperature throttle threshold
    pub temp_throttle: u32,
    /// Temperature shutdown threshold
    pub temp_shutdown: u32,
    /// Fan speed percentage
    pub fan_speed: u32,

    // Power
    /// Current power draw in watts
    pub power_draw: f32,
    /// Power limit in watts
    pub power_limit: f32,
    /// Default power limit in watts
    pub power_default: f32,
    /// Maximum power limit in watts
    pub power_max: f32,

    // Clocks
    /// Current graphics clock in MHz
    pub clock_graphics: u32,
    /// Maximum graphics clock in MHz
    pub clock_graphics_max: u32,
    /// Current memory clock in MHz
    pub clock_memory: u32,
    /// Maximum memory clock in MHz
    pub clock_memory_max: u32,
    /// Current SM clock in MHz
    pub clock_sm: u32,

    // Performance
    /// Performance state (P0 = max, P12 = min)
    pub pstate: String,
    /// Compute mode
    pub compute_mode: String,
    /// PCIe link generation
    pub pcie_gen: u32,
    /// PCIe link width
    pub pcie_width: u32,

    /// Timestamp when stats were collected
    pub timestamp: DateTime<Utc>,

    /// Internal instant for timing (not serialized)
    #[serde(skip)]
    pub instant: Option<Instant>,
}

impl GpuStats {
    /// Memory usage as percentage.
    pub fn memory_percent(&self) -> f64 {
        if self.memory_total == 0 {
            0.0
        } else {
            100.0 * self.memory_used as f64 / self.memory_total as f64
        }
    }

    /// Power usage as percentage of limit.
    pub fn power_percent(&self) -> f64 {
        if self.power_limit == 0.0 {
            0.0
        } else {
            100.0 * self.power_draw as f64 / self.power_limit as f64
        }
    }

    /// Check if GPU is thermal throttling.
    pub fn is_thermal_throttling(&self) -> bool {
        self.temperature >= self.temp_throttle.saturating_sub(5)
    }

    /// Check if GPU memory is under pressure (>90% used).
    pub fn is_memory_pressure(&self) -> bool {
        self.memory_percent() > 90.0
    }

    /// Check if power is near limit (>95%).
    pub fn is_power_limited(&self) -> bool {
        self.power_percent() > 95.0
    }

    /// Format memory as human-readable string.
    pub fn memory_string(&self) -> String {
        format!(
            "{:.1}/{:.1} GB ({:.0}%)",
            self.memory_used as f64 / 1e9,
            self.memory_total as f64 / 1e9,
            self.memory_percent()
        )
    }

    /// Format power as human-readable string.
    pub fn power_string(&self) -> String {
        format!(
            "{:.0}/{:.0}W ({:.0}%)",
            self.power_draw,
            self.power_limit,
            self.power_percent()
        )
    }

    /// Format temperature as human-readable string.
    pub fn temp_string(&self) -> String {
        format!(
            "{}°C (throttle: {}°C)",
            self.temperature, self.temp_throttle
        )
    }

    /// Format clocks as human-readable string.
    pub fn clocks_string(&self) -> String {
        format!(
            "GPU: {}/{}MHz, Mem: {}/{}MHz",
            self.clock_graphics, self.clock_graphics_max, self.clock_memory, self.clock_memory_max
        )
    }
}

/// Query comprehensive GPU stats from nvidia-smi.
pub fn query_gpu_stats(device_idx: usize) -> Option<GpuStats> {
    // Use a minimal, reliable set of fields that work across GPU generations
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,driver_version,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu,power.draw,power.limit,clocks.current.graphics,clocks.current.memory,pstate,pcie.link.gen.current,pcie.link.width.current",
            "--format=csv,noheader,nounits",
            &format!("--id={}", device_idx),
        ])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let parts: Vec<&str> = stdout.trim().split(',').map(|s| s.trim()).collect();

    if parts.len() < 10 {
        return None;
    }

    // Parse values, using defaults for unavailable fields
    let parse_u64 = |s: &str| s.replace("[N/A]", "0").parse::<u64>().unwrap_or(0);
    let parse_u32 = |s: &str| s.replace("[N/A]", "0").parse::<u32>().unwrap_or(0);
    let parse_f32 = |s: &str| s.replace("[N/A]", "0").parse::<f32>().unwrap_or(0.0);

    Some(GpuStats {
        device_idx,
        name: parts[0].to_string(),
        driver_version: parts[1].to_string(),
        cuda_version: String::new(), // Not queried for reliability

        memory_total: parse_u64(parts[2]) * 1024 * 1024,
        memory_used: parse_u64(parts[3]) * 1024 * 1024,
        memory_free: parse_u64(parts[4]) * 1024 * 1024,
        memory_util: parse_u32(parts.get(6).unwrap_or(&"0")),

        gpu_util: parse_u32(parts[5]),
        encoder_util: 0,
        decoder_util: 0,

        temperature: parse_u32(parts.get(7).unwrap_or(&"0")),
        temp_throttle: 83, // Default for RTX 50 series
        temp_shutdown: 93,
        fan_speed: 0,

        power_draw: parse_f32(parts.get(8).unwrap_or(&"0")),
        power_limit: parse_f32(parts.get(9).unwrap_or(&"0")),
        power_default: 0.0,
        power_max: 0.0,

        clock_graphics: parse_u32(parts.get(10).unwrap_or(&"0")),
        clock_graphics_max: 0,
        clock_memory: parse_u32(parts.get(11).unwrap_or(&"0")),
        clock_memory_max: 0,
        clock_sm: 0,

        pstate: parts.get(12).map(|s| s.to_string()).unwrap_or_default(),
        compute_mode: String::new(),
        pcie_gen: parse_u32(parts.get(13).unwrap_or(&"0")),
        pcie_width: parse_u32(parts.get(14).unwrap_or(&"0")),

        timestamp: Utc::now(),
        instant: Some(Instant::now()),
    })
}

/// GPU statistics monitor that tracks history.
#[derive(Debug)]
pub struct GpuStatsMonitor {
    device_idx: usize,
    current: Option<GpuStats>,
    peak_memory: u64,
    peak_temperature: u32,
    peak_power: f32,
    sample_count: u64,
    gpu_time_estimate_ms: u64,
}

impl GpuStatsMonitor {
    /// Create a new GPU stats monitor.
    pub fn new(device_idx: usize) -> Self {
        Self {
            device_idx,
            current: None,
            peak_memory: 0,
            peak_temperature: 0,
            peak_power: 0.0,
            sample_count: 0,
            gpu_time_estimate_ms: 0,
        }
    }

    /// Sample current GPU stats.
    pub fn sample(&mut self) -> Option<&GpuStats> {
        if let Some(stats) = query_gpu_stats(self.device_idx) {
            // Update peaks
            if stats.memory_used > self.peak_memory {
                self.peak_memory = stats.memory_used;
            }
            if stats.temperature > self.peak_temperature {
                self.peak_temperature = stats.temperature;
            }
            if stats.power_draw > self.peak_power {
                self.peak_power = stats.power_draw;
            }

            // Estimate GPU time based on utilization
            // This is a rough estimate: (sample_interval * gpu_util / 100)
            self.gpu_time_estimate_ms += (500 * stats.gpu_util as u64) / 100;

            self.sample_count += 1;
            self.current = Some(stats);
        }

        self.current.as_ref()
    }

    /// Get current stats (without sampling).
    pub fn current(&self) -> Option<&GpuStats> {
        self.current.as_ref()
    }

    /// Get peak memory usage.
    pub fn peak_memory(&self) -> u64 {
        self.peak_memory
    }

    /// Get peak temperature.
    pub fn peak_temperature(&self) -> u32 {
        self.peak_temperature
    }

    /// Get peak power draw.
    pub fn peak_power(&self) -> f32 {
        self.peak_power
    }

    /// Get estimated GPU time in milliseconds.
    pub fn gpu_time_ms(&self) -> u64 {
        self.gpu_time_estimate_ms
    }

    /// Get sample count.
    pub fn sample_count(&self) -> u64 {
        self.sample_count
    }

    /// Check if GPU is healthy (no thermal throttling, memory OK).
    pub fn is_healthy(&self) -> bool {
        self.current
            .as_ref()
            .map(|s| !s.is_thermal_throttling() && !s.is_memory_pressure())
            .unwrap_or(true)
    }

    /// Get health status string.
    pub fn health_status(&self) -> &'static str {
        match &self.current {
            Some(stats) => {
                if stats.is_thermal_throttling() {
                    "THERMAL THROTTLING"
                } else if stats.is_memory_pressure() {
                    "MEMORY PRESSURE"
                } else if stats.is_power_limited() {
                    "POWER LIMITED"
                } else {
                    "HEALTHY"
                }
            }
            None => "UNKNOWN",
        }
    }
}

/// Checkpoint timing information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointTiming {
    /// Checkpoint step number
    pub step: u64,
    /// When checkpoint started
    pub started_at: chrono::DateTime<chrono::Utc>,
    /// When checkpoint completed (None if in progress)
    pub completed_at: Option<chrono::DateTime<chrono::Utc>>,
    /// Checkpoint size in bytes
    pub size_bytes: Option<u64>,
    /// Whether uploaded to HuggingFace
    pub uploaded: bool,
    /// HuggingFace URL if uploaded
    pub hf_url: Option<String>,
}

impl CheckpointTiming {
    /// Create a new checkpoint timing record.
    pub fn new(step: u64) -> Self {
        Self {
            step,
            started_at: chrono::Utc::now(),
            completed_at: None,
            size_bytes: None,
            uploaded: false,
            hf_url: None,
        }
    }

    /// Mark checkpoint as complete.
    pub fn complete(&mut self, size_bytes: u64) {
        self.completed_at = Some(chrono::Utc::now());
        self.size_bytes = Some(size_bytes);
    }

    /// Duration to save checkpoint.
    pub fn duration(&self) -> Option<chrono::Duration> {
        self.completed_at.map(|end| end - self.started_at)
    }

    /// Mark as uploaded.
    pub fn mark_uploaded(&mut self, url: &str) {
        self.uploaded = true;
        self.hf_url = Some(url.to_string());
    }
}

/// ETA calculator for training runs.
#[derive(Debug)]
pub struct EtaCalculator {
    /// Start time
    start_time: Instant,
    /// Total steps
    total_steps: u64,
    /// Recent step times for smoothing
    recent_step_times: Vec<Duration>,
    /// Window size for averaging
    window_size: usize,
}

impl EtaCalculator {
    /// Create a new ETA calculator.
    pub fn new(total_steps: u64) -> Self {
        Self {
            start_time: Instant::now(),
            total_steps,
            recent_step_times: Vec::new(),
            window_size: 50,
        }
    }

    /// Record a step completion.
    pub fn record_step(&mut self, step_time: Duration) {
        self.recent_step_times.push(step_time);
        if self.recent_step_times.len() > self.window_size {
            self.recent_step_times.remove(0);
        }
    }

    /// Get elapsed wall time.
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Get average step time.
    pub fn avg_step_time(&self) -> Option<Duration> {
        if self.recent_step_times.is_empty() {
            return None;
        }
        let total: Duration = self.recent_step_times.iter().sum();
        Some(total / self.recent_step_times.len() as u32)
    }

    /// Calculate ETA for completion.
    pub fn eta(&self, current_step: u64) -> Option<Duration> {
        let remaining_steps = self.total_steps.saturating_sub(current_step);
        self.avg_step_time().map(|avg| avg * remaining_steps as u32)
    }

    /// Format ETA as human-readable string.
    pub fn eta_string(&self, current_step: u64) -> String {
        match self.eta(current_step) {
            Some(eta) => {
                let secs = eta.as_secs();
                if secs < 60 {
                    format!("{}s", secs)
                } else if secs < 3600 {
                    format!("{}m {}s", secs / 60, secs % 60)
                } else {
                    format!("{}h {}m", secs / 3600, (secs % 3600) / 60)
                }
            }
            None => "calculating...".to_string(),
        }
    }

    /// Format elapsed as human-readable string.
    pub fn elapsed_string(&self) -> String {
        let secs = self.elapsed().as_secs();
        if secs < 60 {
            format!("{}s", secs)
        } else if secs < 3600 {
            format!("{}m {}s", secs / 60, secs % 60)
        } else {
            format!("{}h {}m {}s", secs / 3600, (secs % 3600) / 60, secs % 60)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eta_calculator() {
        let mut calc = EtaCalculator::new(100);

        // Record some step times
        for _ in 0..10 {
            calc.record_step(Duration::from_millis(500));
        }

        let eta = calc.eta(50);
        assert!(eta.is_some());

        // 50 steps remaining * 500ms = 25s
        let eta_secs = eta.unwrap().as_secs();
        assert!(eta_secs >= 20 && eta_secs <= 30);
    }

    #[test]
    fn test_checkpoint_timing() {
        let mut timing = CheckpointTiming::new(100);
        assert!(timing.completed_at.is_none());

        timing.complete(1024 * 1024);
        assert!(timing.completed_at.is_some());
        assert_eq!(timing.size_bytes, Some(1024 * 1024));

        timing.mark_uploaded("https://huggingface.co/test");
        assert!(timing.uploaded);
    }
}
