//! GPU memory management and monitoring.
//!
//! Provides:
//! - Real-time VRAM usage monitoring
//! - Intelligent batch sizing based on available memory
//! - RAM as hot VRAM cache for large activations
//! - Memory-aware training configuration

use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

/// GPU memory information.
#[derive(Debug, Clone, Copy)]
pub struct GpuMemoryInfo {
    /// Total VRAM in bytes
    pub total: u64,
    /// Used VRAM in bytes
    pub used: u64,
    /// Free VRAM in bytes
    pub free: u64,
    /// GPU utilization percentage
    pub gpu_util: u32,
    /// Memory utilization percentage
    pub mem_util: u32,
}

impl GpuMemoryInfo {
    /// Get usage as percentage.
    pub fn usage_percent(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            100.0 * self.used as f64 / self.total as f64
        }
    }

    /// Check if VRAM is under pressure (>90% used).
    pub fn is_under_pressure(&self) -> bool {
        self.usage_percent() > 90.0
    }

    /// Check if OOM is likely (>95% used).
    pub fn oom_likely(&self) -> bool {
        self.usage_percent() > 95.0
    }
}

/// Query GPU memory from nvidia-smi.
pub fn query_gpu_memory(device_idx: usize) -> Option<GpuMemoryInfo> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=memory.total,memory.used,memory.free,utilization.gpu,utilization.memory",
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

    if parts.len() < 5 {
        return None;
    }

    Some(GpuMemoryInfo {
        total: parts[0].parse::<u64>().ok()? * 1024 * 1024, // MB to bytes
        used: parts[1].parse::<u64>().ok()? * 1024 * 1024,
        free: parts[2].parse::<u64>().ok()? * 1024 * 1024,
        gpu_util: parts[3].parse().unwrap_or(0),
        mem_util: parts[4].parse().unwrap_or(0),
    })
}

/// Memory budget for training.
#[derive(Debug, Clone)]
pub struct MemoryBudget {
    /// Target maximum VRAM usage (leave some headroom)
    pub max_vram: u64,
    /// Reserved VRAM for system/other apps
    pub reserved_vram: u64,
    /// Available VRAM for training
    pub training_vram: u64,
    /// RAM available for VRAM cache
    pub cache_ram: u64,
}

impl MemoryBudget {
    /// Create a memory budget from GPU info.
    pub fn from_gpu(info: &GpuMemoryInfo, reserve_percent: f64) -> Self {
        let reserved = (info.total as f64 * reserve_percent / 100.0) as u64;
        let training = info.total.saturating_sub(reserved);

        // Use up to 10GB RAM for VRAM cache (rolling limit)
        // Leverages resizable BAR for fast PCIe transfers
        let cache_ram = 10 * 1024 * 1024 * 1024;

        Self {
            max_vram: info.total,
            reserved_vram: reserved,
            training_vram: training,
            cache_ram,
        }
    }

    /// Default budget for 16GB VRAM (RTX 5080).
    pub fn default_16gb() -> Self {
        let gb = 1024 * 1024 * 1024;
        Self {
            max_vram: 16 * gb,
            reserved_vram: 2 * gb,  // 2GB reserved
            training_vram: 14 * gb, // 14GB for training
            cache_ram: 16 * gb,     // 16GB RAM cache
        }
    }
}

/// Optimal training parameters for memory budget.
#[derive(Debug, Clone)]
pub struct OptimalTrainingParams {
    /// Batch size
    pub batch_size: usize,
    /// Sequence length
    pub seq_length: usize,
    /// Gradient checkpoint interval
    pub checkpoint_interval: usize,
    /// Use gradient accumulation
    pub gradient_accumulation: usize,
    /// Effective batch size (batch * accumulation)
    pub effective_batch: usize,
}

/// Estimate memory requirements for a training configuration.
pub fn estimate_training_memory(
    num_params: usize,
    batch_size: usize,
    seq_length: usize,
    hidden_size: usize,
    num_layers: usize,
    num_heads: usize,
    checkpoint_interval: usize,
) -> u64 {
    let bytes_per_param = 4; // FP32

    // Model parameters
    let param_mem = num_params * bytes_per_param;

    // Optimizer states (Adam: m + v = 2x params)
    let optimizer_mem = 2 * param_mem;

    // Gradients
    let gradient_mem = param_mem;

    // Activation memory per layer (approximate)
    // Hidden states: batch * seq * hidden
    let hidden_mem = batch_size * seq_length * hidden_size * bytes_per_param;
    // Attention: batch * heads * seq * seq
    let attention_mem = batch_size * num_heads * seq_length * seq_length * bytes_per_param;
    // MLP intermediates: batch * seq * (4 * hidden) for typical FFN
    let mlp_mem = batch_size * seq_length * 4 * hidden_size * bytes_per_param;

    let per_layer_activation = hidden_mem + attention_mem + mlp_mem;

    // With checkpointing, we only store every N layers
    let stored_layers = (num_layers + checkpoint_interval - 1) / checkpoint_interval;
    let activation_mem = stored_layers * per_layer_activation;

    // Total with some overhead
    let total = param_mem + optimizer_mem + gradient_mem + activation_mem;
    let with_overhead = (total as f64 * 1.2) as u64; // 20% overhead

    with_overhead as u64
}

/// Find optimal training parameters for memory budget.
pub fn find_optimal_params(
    budget: &MemoryBudget,
    num_params: usize,
    hidden_size: usize,
    num_layers: usize,
    num_heads: usize,
    target_batch: usize,
    target_seq: usize,
) -> OptimalTrainingParams {
    // Try different configurations
    let checkpoint_intervals = [2, 4, 8];
    let batch_sizes = [1, 2, 4, 8, 16];
    let seq_lengths = [128, 256, 512, 1024, 2048];

    let mut best: Option<OptimalTrainingParams> = None;
    let mut best_effective_batch = 0;

    for &checkpoint_interval in &checkpoint_intervals {
        for &batch_size in &batch_sizes {
            for &seq_length in &seq_lengths {
                let mem = estimate_training_memory(
                    num_params,
                    batch_size,
                    seq_length,
                    hidden_size,
                    num_layers,
                    num_heads,
                    checkpoint_interval,
                );

                if mem <= budget.training_vram {
                    // Calculate gradient accumulation to reach target batch
                    let grad_accum = (target_batch + batch_size - 1) / batch_size;
                    let effective_batch = batch_size * grad_accum;

                    // Prefer configs closer to target while maximizing effective batch
                    let score = effective_batch * seq_length;

                    if best.is_none()
                        || score > best_effective_batch * (best.as_ref().unwrap().seq_length)
                    {
                        best = Some(OptimalTrainingParams {
                            batch_size,
                            seq_length: seq_length.min(target_seq),
                            checkpoint_interval,
                            gradient_accumulation: grad_accum,
                            effective_batch,
                        });
                        best_effective_batch = effective_batch;
                    }
                }
            }
        }
    }

    // Fallback to minimal config if nothing fits
    best.unwrap_or(OptimalTrainingParams {
        batch_size: 1,
        seq_length: 128,
        checkpoint_interval: 2,
        gradient_accumulation: target_batch,
        effective_batch: target_batch,
    })
}

/// Memory monitor that tracks VRAM usage during training.
#[derive(Debug)]
pub struct MemoryMonitor {
    device_idx: usize,
    peak_used: AtomicU64,
    samples: AtomicU64,
}

impl MemoryMonitor {
    /// Create a new memory monitor.
    pub fn new(device_idx: usize) -> Self {
        Self {
            device_idx,
            peak_used: AtomicU64::new(0),
            samples: AtomicU64::new(0),
        }
    }

    /// Sample current memory usage.
    pub fn sample(&self) -> Option<GpuMemoryInfo> {
        let info = query_gpu_memory(self.device_idx)?;

        // Update peak
        let current_peak = self.peak_used.load(Ordering::Relaxed);
        if info.used > current_peak {
            self.peak_used.store(info.used, Ordering::Relaxed);
        }

        self.samples.fetch_add(1, Ordering::Relaxed);

        Some(info)
    }

    /// Get peak memory usage.
    pub fn peak_used(&self) -> u64 {
        self.peak_used.load(Ordering::Relaxed)
    }

    /// Get sample count.
    pub fn sample_count(&self) -> u64 {
        self.samples.load(Ordering::Relaxed)
    }

    /// Check if memory is healthy.
    pub fn is_healthy(&self) -> bool {
        self.sample()
            .map(|info| !info.is_under_pressure())
            .unwrap_or(true)
    }

    /// Format memory status string.
    pub fn status_string(&self) -> String {
        match self.sample() {
            Some(info) => {
                let used_gb = info.used as f64 / 1e9;
                let total_gb = info.total as f64 / 1e9;
                let peak_gb = self.peak_used() as f64 / 1e9;
                format!(
                    "{:.1}/{:.1}GB ({:.0}%) [peak: {:.1}GB]",
                    used_gb,
                    total_gb,
                    info.usage_percent(),
                    peak_gb
                )
            }
            None => "GPU memory unavailable".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_estimation() {
        // 100M model parameters
        let mem = estimate_training_memory(
            185_000_000, // 185M params
            4,           // batch
            256,         // seq
            768,         // hidden
            12,          // layers
            12,          // heads
            4,           // checkpoint interval
        );

        // Should be a few GB
        assert!(mem > 1_000_000_000); // > 1GB
        assert!(mem < 16_000_000_000); // < 16GB
    }

    #[test]
    fn test_find_optimal_params() {
        let budget = MemoryBudget::default_16gb();

        let params = find_optimal_params(
            &budget,
            185_000_000,
            768,
            12,
            12,
            32,  // target batch
            512, // target seq
        );

        assert!(params.batch_size > 0);
        assert!(params.seq_length > 0);
        assert!(params.effective_batch > 0);
    }
}
