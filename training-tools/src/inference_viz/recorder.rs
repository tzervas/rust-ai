//! Activation Recording Module
//!
//! Provides utilities for recording and managing layer activations
//! during the forward pass of a transformer model.

use super::{ActivationStats, InferenceVizError, LayerActivation, LayerType, Result};
use candle_core::Tensor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Record of a single layer's activation during forward pass.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerRecord {
    /// Layer index
    pub layer_idx: usize,
    /// Layer type
    pub layer_type: LayerType,
    /// Shape of the activation tensor
    pub shape: Vec<usize>,
    /// Activation statistics
    pub stats: ActivationStats,
    /// Sample of activation values (downsampled for efficiency)
    pub sample_values: Vec<f32>,
    /// Timestamp when recorded (relative to start)
    pub timestamp_ms: u64,
}

impl LayerRecord {
    /// Create a new layer record from a tensor.
    pub fn from_tensor(
        layer_idx: usize,
        layer_type: LayerType,
        tensor: &Tensor,
        sample_size: usize,
        timestamp_ms: u64,
    ) -> Result<Self> {
        let shape = tensor.dims().to_vec();
        let flat: Vec<f32> = tensor.flatten_all()?.to_vec1()?;

        let stats = ActivationStats::from_values(&flat, 0.01);

        // Downsample for storage efficiency
        let sample_values = if flat.len() <= sample_size {
            flat
        } else {
            let step = flat.len() / sample_size;
            flat.iter()
                .step_by(step)
                .take(sample_size)
                .copied()
                .collect()
        };

        Ok(Self {
            layer_idx,
            layer_type,
            shape,
            stats,
            sample_values,
            timestamp_ms,
        })
    }
}

/// Recorder for capturing activations during forward pass.
///
/// Maintains a buffer of recent activations and computes running statistics.
#[derive(Debug)]
pub struct ActivationRecorder {
    /// Configuration
    config: RecorderConfig,
    /// Recorded layer activations by layer index
    records: HashMap<usize, Vec<LayerRecord>>,
    /// Start time for relative timestamps
    start_time: std::time::Instant,
    /// Running statistics per layer
    running_stats: HashMap<usize, RunningStats>,
}

/// Configuration for the activation recorder.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecorderConfig {
    /// Maximum number of records to keep per layer
    pub max_records_per_layer: usize,
    /// Sample size for activation values
    pub sample_size: usize,
    /// Whether to compute running statistics
    pub compute_running_stats: bool,
    /// Minimum interval between recordings (ms)
    pub min_interval_ms: u64,
}

impl Default for RecorderConfig {
    fn default() -> Self {
        Self {
            max_records_per_layer: 100,
            sample_size: 1000,
            compute_running_stats: true,
            min_interval_ms: 0,
        }
    }
}

/// Running statistics for a layer.
#[derive(Debug, Clone, Default)]
pub struct RunningStats {
    /// Number of observations
    pub count: usize,
    /// Running mean of mean activations
    pub mean_of_means: f32,
    /// Running mean of std activations
    pub mean_of_stds: f32,
    /// Maximum observed mean
    pub max_mean: f32,
    /// Minimum observed mean
    pub min_mean: f32,
    /// Running mean of sparsity
    pub mean_sparsity: f32,
}

impl RunningStats {
    /// Update running statistics with a new observation.
    pub fn update(&mut self, stats: &ActivationStats) {
        self.count += 1;
        let n = self.count as f32;

        // Welford's online algorithm for mean
        let delta_mean = stats.mean - self.mean_of_means;
        self.mean_of_means += delta_mean / n;

        let delta_std = stats.std - self.mean_of_stds;
        self.mean_of_stds += delta_std / n;

        let delta_sparsity = stats.sparsity - self.mean_sparsity;
        self.mean_sparsity += delta_sparsity / n;

        // Track extremes
        if self.count == 1 || stats.mean > self.max_mean {
            self.max_mean = stats.mean;
        }
        if self.count == 1 || stats.mean < self.min_mean {
            self.min_mean = stats.mean;
        }
    }
}

impl ActivationRecorder {
    /// Create a new activation recorder with default configuration.
    pub fn new() -> Self {
        Self::with_config(RecorderConfig::default())
    }

    /// Create a new activation recorder with custom configuration.
    pub fn with_config(config: RecorderConfig) -> Self {
        Self {
            config,
            records: HashMap::new(),
            start_time: std::time::Instant::now(),
            running_stats: HashMap::new(),
        }
    }

    /// Record activations for a layer.
    pub fn record(
        &mut self,
        layer_idx: usize,
        layer_type: LayerType,
        tensor: &Tensor,
    ) -> Result<()> {
        let elapsed = self.start_time.elapsed().as_millis() as u64;

        // Check minimum interval
        if let Some(records) = self.records.get(&layer_idx) {
            if let Some(last) = records.last() {
                if elapsed - last.timestamp_ms < self.config.min_interval_ms {
                    return Ok(());
                }
            }
        }

        let record = LayerRecord::from_tensor(
            layer_idx,
            layer_type,
            tensor,
            self.config.sample_size,
            elapsed,
        )?;

        // Update running stats
        if self.config.compute_running_stats {
            self.running_stats
                .entry(layer_idx)
                .or_default()
                .update(&record.stats);
        }

        // Store record
        let records = self.records.entry(layer_idx).or_default();
        records.push(record);

        // Trim if exceeding max
        if records.len() > self.config.max_records_per_layer {
            records.remove(0);
        }

        Ok(())
    }

    /// Get all records for a specific layer.
    pub fn get_layer_records(&self, layer_idx: usize) -> Option<&Vec<LayerRecord>> {
        self.records.get(&layer_idx)
    }

    /// Get the most recent record for a layer.
    pub fn get_latest(&self, layer_idx: usize) -> Option<&LayerRecord> {
        self.records.get(&layer_idx).and_then(|r| r.last())
    }

    /// Get running statistics for a layer.
    pub fn get_running_stats(&self, layer_idx: usize) -> Option<&RunningStats> {
        self.running_stats.get(&layer_idx)
    }

    /// Get all layer indices that have been recorded.
    pub fn recorded_layers(&self) -> Vec<usize> {
        let mut layers: Vec<_> = self.records.keys().copied().collect();
        layers.sort();
        layers
    }

    /// Clear all records.
    pub fn clear(&mut self) {
        self.records.clear();
        self.running_stats.clear();
        self.start_time = std::time::Instant::now();
    }

    /// Reset running statistics without clearing records.
    pub fn reset_running_stats(&mut self) {
        self.running_stats.clear();
    }

    /// Get total number of records across all layers.
    pub fn total_records(&self) -> usize {
        self.records.values().map(|v| v.len()).sum()
    }

    /// Export all records as a serializable structure.
    pub fn export(&self) -> RecorderExport {
        RecorderExport {
            config: self.config.clone(),
            records: self.records.clone(),
            running_stats: self
                .running_stats
                .iter()
                .map(|(&k, v)| (k, v.clone()))
                .collect(),
        }
    }

    /// Compute layer-wise activation trends.
    pub fn compute_trends(&self) -> Vec<LayerTrend> {
        self.records
            .iter()
            .map(|(&layer_idx, records)| {
                let means: Vec<f32> = records.iter().map(|r| r.stats.mean).collect();
                let stds: Vec<f32> = records.iter().map(|r| r.stats.std).collect();
                let sparsities: Vec<f32> = records.iter().map(|r| r.stats.sparsity).collect();

                LayerTrend {
                    layer_idx,
                    mean_trend: compute_trend(&means),
                    std_trend: compute_trend(&stds),
                    sparsity_trend: compute_trend(&sparsities),
                    num_samples: records.len(),
                }
            })
            .collect()
    }
}

impl Default for ActivationRecorder {
    fn default() -> Self {
        Self::new()
    }
}

/// Exported recorder state for serialization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecorderExport {
    /// Configuration
    pub config: RecorderConfig,
    /// All records by layer
    pub records: HashMap<usize, Vec<LayerRecord>>,
    /// Running statistics by layer
    pub running_stats: HashMap<usize, RunningStats>,
}

impl Serialize for RunningStats {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut s = serializer.serialize_struct("RunningStats", 6)?;
        s.serialize_field("count", &self.count)?;
        s.serialize_field("mean_of_means", &self.mean_of_means)?;
        s.serialize_field("mean_of_stds", &self.mean_of_stds)?;
        s.serialize_field("max_mean", &self.max_mean)?;
        s.serialize_field("min_mean", &self.min_mean)?;
        s.serialize_field("mean_sparsity", &self.mean_sparsity)?;
        s.end()
    }
}

impl<'de> Deserialize<'de> for RunningStats {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Helper {
            count: usize,
            mean_of_means: f32,
            mean_of_stds: f32,
            max_mean: f32,
            min_mean: f32,
            mean_sparsity: f32,
        }

        let helper = Helper::deserialize(deserializer)?;
        Ok(RunningStats {
            count: helper.count,
            mean_of_means: helper.mean_of_means,
            mean_of_stds: helper.mean_of_stds,
            max_mean: helper.max_mean,
            min_mean: helper.min_mean,
            mean_sparsity: helper.mean_sparsity,
        })
    }
}

/// Trend information for a layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerTrend {
    /// Layer index
    pub layer_idx: usize,
    /// Trend in mean activation (positive = increasing)
    pub mean_trend: f32,
    /// Trend in activation std
    pub std_trend: f32,
    /// Trend in sparsity
    pub sparsity_trend: f32,
    /// Number of samples used
    pub num_samples: usize,
}

/// Compute linear trend (slope) of a series.
fn compute_trend(values: &[f32]) -> f32 {
    if values.len() < 2 {
        return 0.0;
    }

    let n = values.len() as f32;
    let x_mean = (n - 1.0) / 2.0;
    let y_mean: f32 = values.iter().sum::<f32>() / n;

    let mut numerator = 0.0f32;
    let mut denominator = 0.0f32;

    for (i, &y) in values.iter().enumerate() {
        let x = i as f32;
        numerator += (x - x_mean) * (y - y_mean);
        denominator += (x - x_mean).powi(2);
    }

    if denominator.abs() < 1e-10 {
        0.0
    } else {
        numerator / denominator
    }
}

/// Recorder for attention patterns.
#[derive(Debug)]
pub struct AttentionRecorder {
    /// Recorded patterns by (layer, head) tuple
    patterns: HashMap<(usize, usize), Vec<AttentionRecord>>,
    /// Maximum patterns to keep per head
    max_patterns: usize,
    /// Start time
    start_time: std::time::Instant,
}

/// Record of attention weights.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionRecord {
    /// Layer index
    pub layer_idx: usize,
    /// Head index
    pub head_idx: usize,
    /// Attention weights (seq x seq)
    pub weights: Vec<Vec<f32>>,
    /// Entropy of attention distribution per query position
    pub entropy: Vec<f32>,
    /// Timestamp
    pub timestamp_ms: u64,
}

impl AttentionRecorder {
    /// Create a new attention recorder.
    pub fn new(max_patterns: usize) -> Self {
        Self {
            patterns: HashMap::new(),
            max_patterns,
            start_time: std::time::Instant::now(),
        }
    }

    /// Record attention weights.
    pub fn record(&mut self, layer_idx: usize, head_idx: usize, weights: &Tensor) -> Result<()> {
        let elapsed = self.start_time.elapsed().as_millis() as u64;
        let shape = weights.dims();

        if shape.len() < 2 {
            return Err(InferenceVizError::ShapeMismatch {
                expected: "2D attention weights".to_string(),
                got: format!("{:?}", shape),
            });
        }

        let seq_len = shape[shape.len() - 2];
        let weights_2d: Vec<Vec<f32>> = (0..seq_len)
            .map(|i| {
                weights
                    .narrow(shape.len() - 2, i, 1)
                    .and_then(|t| t.flatten_all())
                    .and_then(|t| t.to_vec1())
                    .unwrap_or_default()
            })
            .collect();

        // Compute entropy for each query position
        let entropy: Vec<f32> = weights_2d
            .iter()
            .map(|row| {
                row.iter()
                    .filter(|&&p| p > 1e-10)
                    .map(|&p| -p * p.ln())
                    .sum()
            })
            .collect();

        let record = AttentionRecord {
            layer_idx,
            head_idx,
            weights: weights_2d,
            entropy,
            timestamp_ms: elapsed,
        };

        let patterns = self.patterns.entry((layer_idx, head_idx)).or_default();
        patterns.push(record);

        if patterns.len() > self.max_patterns {
            patterns.remove(0);
        }

        Ok(())
    }

    /// Get patterns for a specific layer and head.
    pub fn get_patterns(&self, layer_idx: usize, head_idx: usize) -> Option<&Vec<AttentionRecord>> {
        self.patterns.get(&(layer_idx, head_idx))
    }

    /// Get the latest pattern for a layer and head.
    pub fn get_latest(&self, layer_idx: usize, head_idx: usize) -> Option<&AttentionRecord> {
        self.patterns
            .get(&(layer_idx, head_idx))
            .and_then(|p| p.last())
    }

    /// Get all recorded (layer, head) pairs.
    pub fn recorded_heads(&self) -> Vec<(usize, usize)> {
        let mut heads: Vec<_> = self.patterns.keys().copied().collect();
        heads.sort();
        heads
    }

    /// Clear all records.
    pub fn clear(&mut self) {
        self.patterns.clear();
        self.start_time = std::time::Instant::now();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_running_stats() {
        let mut stats = RunningStats::default();

        let s1 = ActivationStats {
            mean: 1.0,
            std: 0.5,
            max: 2.0,
            min: 0.0,
            sparsity: 0.1,
        };
        stats.update(&s1);

        assert_eq!(stats.count, 1);
        assert!((stats.mean_of_means - 1.0).abs() < 1e-6);

        let s2 = ActivationStats {
            mean: 3.0,
            std: 0.5,
            max: 4.0,
            min: 2.0,
            sparsity: 0.3,
        };
        stats.update(&s2);

        assert_eq!(stats.count, 2);
        assert!((stats.mean_of_means - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_compute_trend() {
        // Increasing trend
        let increasing = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let trend = compute_trend(&increasing);
        assert!(trend > 0.9, "Expected positive trend, got {}", trend);

        // Decreasing trend
        let decreasing = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let trend = compute_trend(&decreasing);
        assert!(trend < -0.9, "Expected negative trend, got {}", trend);

        // Flat
        let flat = vec![3.0, 3.0, 3.0, 3.0, 3.0];
        let trend = compute_trend(&flat);
        assert!(trend.abs() < 1e-6, "Expected zero trend, got {}", trend);
    }

    #[test]
    fn test_layer_record_from_tensor() {
        let device = Device::Cpu;
        let tensor = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0, 5.0], &device).unwrap();

        let record = LayerRecord::from_tensor(0, LayerType::FeedForward, &tensor, 10, 0).unwrap();

        assert_eq!(record.layer_idx, 0);
        assert_eq!(record.shape, vec![5]);
        assert!((record.stats.mean - 3.0).abs() < 1e-6);
    }
}
