//! Training Visualization Module
//!
//! Unified system for efficient activation capture during training with
//! real-time 3D animated visualization in a tiled dashboard.
//!
//! # Architecture
//!
//! ```text
//! Training Loop
//!      │
//!      ▼
//! ┌─────────────────────────────────────────────────────────────┐
//! │  TrainingVizCapture (efficient capture)                     │
//! │  ├── ActivationSampler (every N steps)                      │
//! │  ├── LayerStatsCollector (streaming stats, <100 bytes/layer)│
//! │  └── EventTrigger (capture on anomalies)                    │
//! └─────────────────────────────────────────────────────────────┘
//!      │
//!      ▼
//! ┌─────────────────────────────────────────────────────────────┐
//! │  TrainingVizStream (data pipeline)                          │
//! │  ├── Ring buffer (last 1000 snapshots)                      │
//! │  ├── Aggregation (per-layer summaries)                      │
//! │  └── Broadcast to visualizers                               │
//! └─────────────────────────────────────────────────────────────┘
//!      │
//!      ▼
//! ┌─────────────────────────────────────────────────────────────┐
//! │  TrainingDashboard (unified tiled view)                     │
//! │  ┌─────────┬─────────┬─────────┐                           │
//! │  │ Network │ Activ.  │ Grad    │                           │
//! │  │   3D    │ Heatmap │ Flow    │                           │
//! │  ├─────────┼─────────┼─────────┤                           │
//! │  │ Loss    │ Attn    │ Embed   │                           │
//! │  │ Chart   │ Pattern │ Cloud   │                           │
//! │  └─────────┴─────────┴─────────┘                           │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Efficient Capture Strategy
//!
//! Full activation capture is prohibitively expensive (~100GB for 7B model).
//! We use a multi-tier capture strategy:
//!
//! | Tier | What | When | Memory |
//! |------|------|------|--------|
//! | Always | Layer stats (mean, std, min, max, sparsity) | Every step | ~100 bytes/layer |
//! | Sampled | Downsampled activations (1K values/layer) | Every N steps | ~4KB/layer |
//! | Triggered | Full layer snapshot | On anomaly | One-shot |
//!
//! # Example
//!
//! ```rust,ignore
//! use training_tools::training_viz::{TrainingVizCapture, TrainingDashboard, CaptureConfig};
//!
//! // Configure capture
//! let config = CaptureConfig::builder()
//!     .sample_every_n_steps(100)
//!     .max_history(1000)
//!     .trigger_on_loss_spike(2.0)  // 2 std devs
//!     .trigger_on_gradient_explosion(10.0)
//!     .build();
//!
//! let mut capture = TrainingVizCapture::new(config);
//!
//! // In training loop
//! for step in 0..total_steps {
//!     let output = model.forward(&batch);
//!
//!     // Efficient: only stats (always)
//!     capture.record_step_stats(step, &model);
//!
//!     // Sampled: full capture every N steps
//!     if capture.should_sample(step) {
//!         capture.record_full_snapshot(step, &model);
//!     }
//!
//!     // Triggered: on anomaly
//!     if capture.detect_anomaly(&metrics) {
//!         capture.record_anomaly_snapshot(step, &model, &metrics);
//!     }
//!
//!     let loss = loss_fn(&output, &targets);
//!     loss.backward();
//!     optimizer.step();
//! }
//!
//! // Launch dashboard
//! let dashboard = TrainingDashboard::new(capture.stream());
//! dashboard.run();  // Opens interactive 3D tiled view
//! ```

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

// Re-export key types from other modules
pub use crate::inference_viz::{ActivationStats, LayerType};

// ============================================================================
// Capture Configuration
// ============================================================================

/// Configuration for activation capture.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaptureConfig {
    /// Sample full activations every N steps (0 = disabled).
    pub sample_every_n_steps: usize,
    /// Maximum history to retain (ring buffer size).
    pub max_history: usize,
    /// Number of activation values to downsample per layer.
    pub sample_size_per_layer: usize,
    /// Trigger full capture when loss exceeds baseline by N std devs.
    pub trigger_loss_spike_std: f32,
    /// Trigger full capture when gradient norm exceeds baseline by factor.
    pub trigger_gradient_explosion_factor: f32,
    /// Trigger on gradient vanishing (below factor of baseline).
    pub trigger_gradient_vanishing_factor: f32,
    /// Minimum interval between triggered captures (ms).
    pub trigger_cooldown_ms: u64,
    /// Whether to track attention patterns.
    pub track_attention: bool,
    /// Maximum attention heads to track (0 = all).
    pub max_attention_heads: usize,
}

impl Default for CaptureConfig {
    fn default() -> Self {
        Self {
            sample_every_n_steps: 100,
            max_history: 1000,
            sample_size_per_layer: 1000,
            trigger_loss_spike_std: 3.0,
            trigger_gradient_explosion_factor: 10.0,
            trigger_gradient_vanishing_factor: 0.01,
            trigger_cooldown_ms: 5000,
            track_attention: true,
            max_attention_heads: 12, // First 12 heads
        }
    }
}

impl CaptureConfig {
    /// Create a builder for capture configuration.
    pub fn builder() -> CaptureConfigBuilder {
        CaptureConfigBuilder::default()
    }

    /// Lightweight config: only stats, no sampling.
    pub fn lightweight() -> Self {
        Self {
            sample_every_n_steps: 0,
            max_history: 100,
            sample_size_per_layer: 100,
            track_attention: false,
            ..Default::default()
        }
    }

    /// Full config: frequent sampling for detailed analysis.
    pub fn detailed() -> Self {
        Self {
            sample_every_n_steps: 10,
            max_history: 5000,
            sample_size_per_layer: 5000,
            track_attention: true,
            max_attention_heads: 0, // All heads
            ..Default::default()
        }
    }
}

/// Builder for capture configuration.
#[derive(Debug, Default)]
pub struct CaptureConfigBuilder {
    config: CaptureConfig,
}

impl CaptureConfigBuilder {
    pub fn sample_every_n_steps(mut self, n: usize) -> Self {
        self.config.sample_every_n_steps = n;
        self
    }

    pub fn max_history(mut self, n: usize) -> Self {
        self.config.max_history = n;
        self
    }

    pub fn sample_size_per_layer(mut self, n: usize) -> Self {
        self.config.sample_size_per_layer = n;
        self
    }

    pub fn trigger_on_loss_spike(mut self, std_devs: f32) -> Self {
        self.config.trigger_loss_spike_std = std_devs;
        self
    }

    pub fn trigger_on_gradient_explosion(mut self, factor: f32) -> Self {
        self.config.trigger_gradient_explosion_factor = factor;
        self
    }

    pub fn trigger_on_gradient_vanishing(mut self, factor: f32) -> Self {
        self.config.trigger_gradient_vanishing_factor = factor;
        self
    }

    pub fn track_attention(mut self, enabled: bool) -> Self {
        self.config.track_attention = enabled;
        self
    }

    pub fn max_attention_heads(mut self, n: usize) -> Self {
        self.config.max_attention_heads = n;
        self
    }

    pub fn build(self) -> CaptureConfig {
        self.config
    }
}

// ============================================================================
// Activation Snapshot Types
// ============================================================================

/// Lightweight per-step statistics (always captured).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepStats {
    /// Training step number.
    pub step: u64,
    /// Timestamp (ms since training start).
    pub timestamp_ms: u64,
    /// Loss value.
    pub loss: f32,
    /// Overall gradient norm.
    pub gradient_norm: f32,
    /// Per-layer activation statistics.
    pub layer_stats: Vec<LayerStats>,
    /// Training phase (warmup, full, predict, correct).
    pub phase: String,
    /// Learning rate.
    pub learning_rate: f32,
}

/// Per-layer statistics (lightweight, ~80 bytes per layer).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct LayerStats {
    /// Layer index.
    pub layer_idx: usize,
    /// Activation mean.
    pub activation_mean: f32,
    /// Activation standard deviation.
    pub activation_std: f32,
    /// Activation min.
    pub activation_min: f32,
    /// Activation max.
    pub activation_max: f32,
    /// Sparsity (fraction near zero).
    pub sparsity: f32,
    /// Gradient norm for this layer.
    pub gradient_norm: f32,
    /// Dead neuron fraction (always zero activation).
    pub dead_neuron_fraction: f32,
}

impl From<ActivationStats> for LayerStats {
    fn from(stats: ActivationStats) -> Self {
        Self {
            layer_idx: 0,
            activation_mean: stats.mean,
            activation_std: stats.std,
            activation_min: stats.min,
            activation_max: stats.max,
            sparsity: stats.sparsity,
            gradient_norm: 0.0,
            dead_neuron_fraction: 0.0,
        }
    }
}

/// Full activation snapshot (sampled, ~4KB per layer).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivationSnapshot {
    /// Training step number.
    pub step: u64,
    /// Timestamp (ms since training start).
    pub timestamp_ms: u64,
    /// Per-layer activation samples.
    pub layer_activations: Vec<LayerActivationSample>,
    /// Trigger reason (if triggered capture).
    pub trigger_reason: Option<TriggerReason>,
}

/// Sampled activation values for a single layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerActivationSample {
    /// Layer index.
    pub layer_idx: usize,
    /// Layer name/type.
    pub layer_name: String,
    /// Downsampled activation values.
    pub values: Vec<f32>,
    /// Full statistics.
    pub stats: ActivationStats,
    /// 10-bin histogram for distribution shape.
    pub histogram: [u32; 10],
    /// Histogram bin edges (min to max).
    pub histogram_edges: (f32, f32),
}

impl LayerActivationSample {
    /// Create a new sample with histogram.
    pub fn new(layer_idx: usize, layer_name: String, values: Vec<f32>) -> Self {
        let stats = ActivationStats::from_values(&values, 0.01);
        let histogram = Self::compute_histogram(&values, stats.min, stats.max);

        Self {
            layer_idx,
            layer_name,
            values,
            stats,
            histogram,
            histogram_edges: (stats.min, stats.max),
        }
    }

    fn compute_histogram(values: &[f32], min: f32, max: f32) -> [u32; 10] {
        let mut histogram = [0u32; 10];
        if (max - min).abs() < 1e-10 || values.is_empty() {
            return histogram;
        }

        let range = max - min;
        for &v in values {
            let bin = ((v - min) / range * 10.0).min(9.0).max(0.0) as usize;
            histogram[bin] += 1;
        }
        histogram
    }
}

/// Reason for triggered capture.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TriggerReason {
    /// Loss spiked above threshold.
    LossSpike { loss: f32, threshold: f32 },
    /// Gradient exploded.
    GradientExplosion { norm: f32, baseline: f32 },
    /// Gradient vanished.
    GradientVanishing { norm: f32, baseline: f32 },
    /// Manual trigger.
    Manual,
    /// Phase transition.
    PhaseTransition { from: String, to: String },
}

// ============================================================================
// Attention Snapshot
// ============================================================================

/// Attention pattern snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionSnapshot {
    /// Training step.
    pub step: u64,
    /// Layer index.
    pub layer_idx: usize,
    /// Head index.
    pub head_idx: usize,
    /// Attention weights (query_len x key_len), downsampled if needed.
    pub weights: Vec<Vec<f32>>,
    /// Per-position entropy.
    pub entropy: Vec<f32>,
    /// Average attention span (how far attention reaches).
    pub avg_span: f32,
}

// ============================================================================
// Anomaly Detection
// ============================================================================

/// Anomaly detection state.
#[derive(Debug, Clone)]
pub struct AnomalyDetector {
    /// Running mean of loss.
    loss_mean: f32,
    /// Running std of loss.
    loss_std: f32,
    /// Running mean of gradient norm.
    grad_mean: f32,
    /// Number of observations.
    count: usize,
    /// Last trigger time.
    last_trigger: Option<Instant>,
    /// Configuration.
    config: CaptureConfig,
}

impl AnomalyDetector {
    pub fn new(config: CaptureConfig) -> Self {
        Self {
            loss_mean: 0.0,
            loss_std: 1.0,
            grad_mean: 1.0,
            count: 0,
            last_trigger: None,
            config,
        }
    }

    /// Update running statistics with new observation.
    pub fn update(&mut self, loss: f32, gradient_norm: f32) {
        self.count += 1;
        let n = self.count as f32;

        // Welford's online algorithm for mean/variance
        let delta_loss = loss - self.loss_mean;
        self.loss_mean += delta_loss / n;

        if self.count > 1 {
            // Update running variance
            let new_delta = loss - self.loss_mean;
            let m2 = (self.loss_std.powi(2) * (n - 1.0)) + delta_loss * new_delta;
            self.loss_std = (m2 / n).sqrt();
        }

        // EMA for gradient norm baseline
        let alpha = 0.01;
        self.grad_mean = alpha * gradient_norm + (1.0 - alpha) * self.grad_mean;
    }

    /// Check if current metrics indicate an anomaly.
    pub fn detect(&mut self, loss: f32, gradient_norm: f32) -> Option<TriggerReason> {
        // Check cooldown
        if let Some(last) = self.last_trigger {
            if last.elapsed() < Duration::from_millis(self.config.trigger_cooldown_ms) {
                return None;
            }
        }

        // Check loss spike
        if self.count > 10 && self.loss_std > 1e-6 {
            let z_score = (loss - self.loss_mean) / self.loss_std;
            if z_score > self.config.trigger_loss_spike_std {
                self.last_trigger = Some(Instant::now());
                return Some(TriggerReason::LossSpike {
                    loss,
                    threshold: self.loss_mean + self.config.trigger_loss_spike_std * self.loss_std,
                });
            }
        }

        // Check gradient explosion
        if self.grad_mean > 1e-6 {
            let ratio = gradient_norm / self.grad_mean;
            if ratio > self.config.trigger_gradient_explosion_factor {
                self.last_trigger = Some(Instant::now());
                return Some(TriggerReason::GradientExplosion {
                    norm: gradient_norm,
                    baseline: self.grad_mean,
                });
            }

            // Check gradient vanishing
            if ratio < self.config.trigger_gradient_vanishing_factor {
                self.last_trigger = Some(Instant::now());
                return Some(TriggerReason::GradientVanishing {
                    norm: gradient_norm,
                    baseline: self.grad_mean,
                });
            }
        }

        None
    }
}

// ============================================================================
// Training Visualization Capture
// ============================================================================

/// Main capture system for training visualization.
pub struct TrainingVizCapture {
    /// Configuration.
    config: CaptureConfig,
    /// Per-step statistics history (ring buffer).
    step_stats: VecDeque<StepStats>,
    /// Full activation snapshots (ring buffer).
    snapshots: VecDeque<ActivationSnapshot>,
    /// Attention snapshots.
    attention_snapshots: VecDeque<AttentionSnapshot>,
    /// Anomaly detector.
    anomaly_detector: AnomalyDetector,
    /// Training start time.
    start_time: Instant,
    /// Current step.
    current_step: u64,
    /// Shared stream for real-time visualization.
    stream: Arc<RwLock<VizStream>>,
}

impl TrainingVizCapture {
    /// Create a new capture system.
    pub fn new(config: CaptureConfig) -> Self {
        let stream = Arc::new(RwLock::new(VizStream::new(config.max_history)));
        Self {
            anomaly_detector: AnomalyDetector::new(config.clone()),
            config,
            step_stats: VecDeque::with_capacity(1000),
            snapshots: VecDeque::with_capacity(100),
            attention_snapshots: VecDeque::with_capacity(100),
            start_time: Instant::now(),
            current_step: 0,
            stream,
        }
    }

    /// Get a handle to the visualization stream.
    pub fn stream(&self) -> Arc<RwLock<VizStream>> {
        Arc::clone(&self.stream)
    }

    /// Check if we should sample full activations at this step.
    pub fn should_sample(&self, step: u64) -> bool {
        self.config.sample_every_n_steps > 0 && step % self.config.sample_every_n_steps as u64 == 0
    }

    /// Record lightweight per-step statistics.
    pub fn record_step_stats(&mut self, stats: StepStats) {
        self.current_step = stats.step;

        // Update anomaly detector
        self.anomaly_detector
            .update(stats.loss, stats.gradient_norm);

        // Store in local buffer
        self.step_stats.push_back(stats.clone());
        if self.step_stats.len() > self.config.max_history {
            self.step_stats.pop_front();
        }

        // Broadcast to stream
        if let Ok(mut stream) = self.stream.write() {
            stream.push_stats(stats);
        }
    }

    /// Record full activation snapshot.
    pub fn record_snapshot(&mut self, snapshot: ActivationSnapshot) {
        self.snapshots.push_back(snapshot.clone());
        if self.snapshots.len() > 100 {
            self.snapshots.pop_front();
        }

        if let Ok(mut stream) = self.stream.write() {
            stream.push_snapshot(snapshot);
        }
    }

    /// Detect anomaly and return trigger reason if found.
    pub fn detect_anomaly(&mut self, loss: f32, gradient_norm: f32) -> Option<TriggerReason> {
        self.anomaly_detector.detect(loss, gradient_norm)
    }

    /// Record attention snapshot.
    pub fn record_attention(&mut self, snapshot: AttentionSnapshot) {
        if !self.config.track_attention {
            return;
        }

        // Check head limit
        if self.config.max_attention_heads > 0
            && snapshot.head_idx >= self.config.max_attention_heads
        {
            return;
        }

        self.attention_snapshots.push_back(snapshot.clone());
        if self.attention_snapshots.len() > 100 {
            self.attention_snapshots.pop_front();
        }

        if let Ok(mut stream) = self.stream.write() {
            stream.push_attention(snapshot);
        }
    }

    /// Get elapsed time since training start.
    pub fn elapsed_ms(&self) -> u64 {
        self.start_time.elapsed().as_millis() as u64
    }

    /// Get current step stats.
    pub fn latest_stats(&self) -> Option<&StepStats> {
        self.step_stats.back()
    }

    /// Get latest snapshot.
    pub fn latest_snapshot(&self) -> Option<&ActivationSnapshot> {
        self.snapshots.back()
    }

    /// Export all captured data.
    pub fn export(&self) -> CaptureExport {
        CaptureExport {
            config: self.config.clone(),
            step_stats: self.step_stats.iter().cloned().collect(),
            snapshots: self.snapshots.iter().cloned().collect(),
            attention_snapshots: self.attention_snapshots.iter().cloned().collect(),
        }
    }
}

/// Exported capture data for serialization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaptureExport {
    pub config: CaptureConfig,
    pub step_stats: Vec<StepStats>,
    pub snapshots: Vec<ActivationSnapshot>,
    pub attention_snapshots: Vec<AttentionSnapshot>,
}

// ============================================================================
// Visualization Stream
// ============================================================================

/// Real-time stream of visualization data.
#[derive(Debug)]
pub struct VizStream {
    /// Recent step stats.
    pub stats: VecDeque<StepStats>,
    /// Recent snapshots.
    pub snapshots: VecDeque<ActivationSnapshot>,
    /// Recent attention.
    pub attention: VecDeque<AttentionSnapshot>,
    /// Maximum history size.
    max_history: usize,
    /// Subscribers to notify on new data.
    version: u64,
}

impl VizStream {
    pub fn new(max_history: usize) -> Self {
        Self {
            stats: VecDeque::with_capacity(max_history),
            snapshots: VecDeque::with_capacity(100),
            attention: VecDeque::with_capacity(100),
            max_history,
            version: 0,
        }
    }

    pub fn push_stats(&mut self, stats: StepStats) {
        self.stats.push_back(stats);
        if self.stats.len() > self.max_history {
            self.stats.pop_front();
        }
        self.version += 1;
    }

    pub fn push_snapshot(&mut self, snapshot: ActivationSnapshot) {
        self.snapshots.push_back(snapshot);
        if self.snapshots.len() > 100 {
            self.snapshots.pop_front();
        }
        self.version += 1;
    }

    pub fn push_attention(&mut self, attention: AttentionSnapshot) {
        self.attention.push_back(attention);
        if self.attention.len() > 100 {
            self.attention.pop_front();
        }
        self.version += 1;
    }

    /// Get the current version (incremented on each update).
    pub fn version(&self) -> u64 {
        self.version
    }

    /// Get latest N step stats.
    pub fn recent_stats(&self, n: usize) -> Vec<&StepStats> {
        self.stats.iter().rev().take(n).collect()
    }

    /// Get per-layer statistics aggregated over recent steps.
    pub fn layer_aggregates(&self, steps: usize) -> HashMap<usize, LayerAggregate> {
        let mut aggregates: HashMap<usize, LayerAggregate> = HashMap::new();

        for stats in self.stats.iter().rev().take(steps) {
            for layer in &stats.layer_stats {
                let agg = aggregates
                    .entry(layer.layer_idx)
                    .or_insert_with(|| LayerAggregate {
                        layer_idx: layer.layer_idx,
                        mean_activation: 0.0,
                        mean_gradient: 0.0,
                        mean_sparsity: 0.0,
                        activation_trend: 0.0,
                        gradient_trend: 0.0,
                        count: 0,
                    });
                agg.mean_activation += layer.activation_mean;
                agg.mean_gradient += layer.gradient_norm;
                agg.mean_sparsity += layer.sparsity;
                agg.count += 1;
            }
        }

        // Normalize
        for agg in aggregates.values_mut() {
            if agg.count > 0 {
                let n = agg.count as f32;
                agg.mean_activation /= n;
                agg.mean_gradient /= n;
                agg.mean_sparsity /= n;
            }
        }

        aggregates
    }
}

/// Aggregated statistics for a layer.
#[derive(Debug, Clone, Default)]
pub struct LayerAggregate {
    pub layer_idx: usize,
    pub mean_activation: f32,
    pub mean_gradient: f32,
    pub mean_sparsity: f32,
    pub activation_trend: f32,
    pub gradient_trend: f32,
    pub count: usize,
}

// ============================================================================
// Dashboard Layout Presets
// ============================================================================

/// Predefined dashboard layouts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DashboardPreset {
    /// Training overview: loss chart + gradient flow + activations.
    TrainingOverview,
    /// Attention analysis: attention heatmap + chord diagram + token flow.
    AttentionAnalysis,
    /// Layer deep-dive: activation histograms + gradient flow + 3D network.
    LayerAnalysis,
    /// Full dashboard: all visualizations.
    Full,
    /// Minimal: loss + gradient only.
    Minimal,
}

impl DashboardPreset {
    /// Get panel configuration for this preset.
    pub fn panels(&self) -> Vec<PanelConfig> {
        match self {
            DashboardPreset::TrainingOverview => vec![
                PanelConfig::new("loss_chart", PanelKind::LossChart, 0, 0, 2, 1),
                PanelConfig::new("gradient_flow", PanelKind::GradientFlow, 2, 0, 1, 1),
                PanelConfig::new("activations", PanelKind::ActivationHeatmap, 0, 1, 2, 1),
                PanelConfig::new("layer_stats", PanelKind::LayerStats, 2, 1, 1, 1),
            ],
            DashboardPreset::AttentionAnalysis => vec![
                PanelConfig::new("attention_heatmap", PanelKind::AttentionHeatmap, 0, 0, 2, 2),
                PanelConfig::new("chord_diagram", PanelKind::ChordDiagram, 2, 0, 1, 1),
                PanelConfig::new("token_flow", PanelKind::TokenFlow, 2, 1, 1, 1),
            ],
            DashboardPreset::LayerAnalysis => vec![
                PanelConfig::new("network_3d", PanelKind::Network3D, 0, 0, 2, 2),
                PanelConfig::new(
                    "activation_hist",
                    PanelKind::ActivationHistogram,
                    2,
                    0,
                    1,
                    1,
                ),
                PanelConfig::new("gradient_flow", PanelKind::GradientFlow, 2, 1, 1, 1),
            ],
            DashboardPreset::Full => vec![
                PanelConfig::new("network_3d", PanelKind::Network3D, 0, 0, 1, 1),
                PanelConfig::new("loss_chart", PanelKind::LossChart, 1, 0, 1, 1),
                PanelConfig::new("gradient_flow", PanelKind::GradientFlow, 2, 0, 1, 1),
                PanelConfig::new("activations", PanelKind::ActivationHeatmap, 0, 1, 1, 1),
                PanelConfig::new("attention", PanelKind::AttentionHeatmap, 1, 1, 1, 1),
                PanelConfig::new("embeddings", PanelKind::EmbeddingCloud, 2, 1, 1, 1),
            ],
            DashboardPreset::Minimal => vec![
                PanelConfig::new("loss_chart", PanelKind::LossChart, 0, 0, 2, 1),
                PanelConfig::new("gradient_norm", PanelKind::GradientChart, 2, 0, 1, 1),
            ],
        }
    }
}

/// Panel configuration.
#[derive(Debug, Clone)]
pub struct PanelConfig {
    /// Panel identifier.
    pub id: String,
    /// Panel type.
    pub kind: PanelKind,
    /// Column position.
    pub col: usize,
    /// Row position.
    pub row: usize,
    /// Width in grid cells.
    pub width: usize,
    /// Height in grid cells.
    pub height: usize,
}

impl PanelConfig {
    pub fn new(
        id: &str,
        kind: PanelKind,
        col: usize,
        row: usize,
        width: usize,
        height: usize,
    ) -> Self {
        Self {
            id: id.to_string(),
            kind,
            col,
            row,
            width,
            height,
        }
    }
}

/// Types of visualization panels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PanelKind {
    /// 3D network visualization.
    Network3D,
    /// Loss over time chart.
    LossChart,
    /// Gradient norm chart.
    GradientChart,
    /// Gradient flow through layers.
    GradientFlow,
    /// Per-layer activation heatmap.
    ActivationHeatmap,
    /// Activation value histograms.
    ActivationHistogram,
    /// Layer statistics table.
    LayerStats,
    /// Attention pattern heatmap.
    AttentionHeatmap,
    /// Chord diagram for attention.
    ChordDiagram,
    /// Token flow visualization.
    TokenFlow,
    /// 3D embedding point cloud.
    EmbeddingCloud,
    /// Loss landscape surface.
    LossLandscape,
}

// ============================================================================
// 3D Animation Data Structures
// ============================================================================

/// Animated training visualization frame.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnimationFrame {
    /// Frame timestamp (ms).
    pub timestamp_ms: u64,
    /// Training step.
    pub step: u64,
    /// Phase of animation (forward/backward/idle).
    pub phase: AnimPhase,
    /// Layer glow intensities (for highlighting active layers).
    pub layer_glow: Vec<f32>,
    /// Connection flow intensities (for data flow visualization).
    pub connection_flow: Vec<f32>,
    /// Particle positions for gradient flow.
    pub gradient_particles: Vec<(f32, f32, f32)>,
    /// Color mapping for current loss/gradient.
    pub heat_color: [f32; 4],
}

/// Animation phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnimPhase {
    /// No animation.
    Idle,
    /// Forward pass (input → output).
    Forward,
    /// Backward pass (output → input).
    Backward,
    /// Weight update.
    Update,
}

impl Default for AnimPhase {
    fn default() -> Self {
        Self::Idle
    }
}

/// Animation controller for 3D training visualization.
pub struct AnimationController {
    /// Animation speed multiplier.
    pub speed: f32,
    /// Current phase.
    pub phase: AnimPhase,
    /// Phase progress (0.0 to 1.0).
    pub progress: f32,
    /// Number of layers.
    num_layers: usize,
    /// Forward pass duration (frames).
    forward_frames: usize,
    /// Backward pass duration (frames).
    backward_frames: usize,
    /// Current frame.
    current_frame: usize,
}

impl AnimationController {
    pub fn new(num_layers: usize) -> Self {
        Self {
            speed: 1.0,
            phase: AnimPhase::Idle,
            progress: 0.0,
            num_layers,
            forward_frames: num_layers * 10,
            backward_frames: num_layers * 10,
            current_frame: 0,
        }
    }

    /// Start forward pass animation.
    pub fn start_forward(&mut self) {
        self.phase = AnimPhase::Forward;
        self.progress = 0.0;
        self.current_frame = 0;
    }

    /// Start backward pass animation.
    pub fn start_backward(&mut self) {
        self.phase = AnimPhase::Backward;
        self.progress = 0.0;
        self.current_frame = 0;
    }

    /// Update animation state.
    pub fn update(&mut self, delta_frames: usize) -> AnimationFrame {
        self.current_frame += delta_frames;

        let total_frames = match self.phase {
            AnimPhase::Forward => self.forward_frames,
            AnimPhase::Backward => self.backward_frames,
            AnimPhase::Update => 30,
            AnimPhase::Idle => 0,
        };

        if total_frames > 0 {
            self.progress = (self.current_frame as f32 / total_frames as f32).min(1.0);
        }

        // Compute layer glow based on animation progress
        let layer_glow = self.compute_layer_glow();
        let connection_flow = self.compute_connection_flow();
        let gradient_particles = self.compute_particles();

        // Transition phases
        if self.progress >= 1.0 {
            match self.phase {
                AnimPhase::Forward => self.start_backward(),
                AnimPhase::Backward => {
                    self.phase = AnimPhase::Update;
                    self.current_frame = 0;
                }
                AnimPhase::Update => {
                    self.phase = AnimPhase::Idle;
                }
                AnimPhase::Idle => {}
            }
        }

        AnimationFrame {
            timestamp_ms: 0,
            step: 0,
            phase: self.phase,
            layer_glow,
            connection_flow,
            gradient_particles,
            heat_color: [1.0, 0.5, 0.0, 1.0],
        }
    }

    fn compute_layer_glow(&self) -> Vec<f32> {
        let mut glow = vec![0.0; self.num_layers];
        let active_layer = match self.phase {
            AnimPhase::Forward => (self.progress * self.num_layers as f32) as usize,
            AnimPhase::Backward => self
                .num_layers
                .saturating_sub((self.progress * self.num_layers as f32) as usize + 1),
            AnimPhase::Update => {
                // All layers glow during update
                for g in &mut glow {
                    *g = 0.8;
                }
                return glow;
            }
            AnimPhase::Idle => return glow,
        };

        // Gaussian glow around active layer
        for (i, g) in glow.iter_mut().enumerate() {
            let dist = (i as f32 - active_layer as f32).abs();
            *g = (-dist.powi(2) / 2.0).exp();
        }
        glow
    }

    fn compute_connection_flow(&self) -> Vec<f32> {
        let mut flow = vec![0.0; self.num_layers.saturating_sub(1)];
        let active_conn = match self.phase {
            AnimPhase::Forward => (self.progress * flow.len() as f32) as usize,
            AnimPhase::Backward => flow
                .len()
                .saturating_sub((self.progress * flow.len() as f32) as usize + 1),
            _ => return flow,
        };

        if active_conn < flow.len() {
            flow[active_conn] = 1.0;
        }
        flow
    }

    fn compute_particles(&self) -> Vec<(f32, f32, f32)> {
        // Generate particle positions along the active flow
        let mut particles = Vec::new();
        if self.phase == AnimPhase::Idle {
            return particles;
        }

        let num_particles = 50;
        for i in 0..num_particles {
            let t = self.progress + (i as f32 * 0.02);
            let x = t * 10.0;
            let y = (t * 3.14159 * 2.0).sin() * 0.5;
            let z = (t * 3.14159 * 2.0).cos() * 0.5;
            particles.push((x, y, z));
        }
        particles
    }
}

// ============================================================================
// Training Analysis & Assessment
// ============================================================================

/// Comprehensive training analysis summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingSummary {
    /// Current training step.
    pub step: u64,
    /// Overall health score (0-100).
    pub health_score: u8,
    /// Health status.
    pub health_status: HealthStatus,
    /// Loss analysis.
    pub loss_analysis: LossAnalysis,
    /// Gradient analysis.
    pub gradient_analysis: GradientAnalysis,
    /// Layer-wise analysis.
    pub layer_analysis: Vec<LayerAnalysis>,
    /// Attention analysis (if available).
    pub attention_analysis: Option<AttentionAnalysis>,
    /// Recommendations.
    pub recommendations: Vec<Recommendation>,
    /// Detected anomalies.
    pub anomalies: Vec<DetectedAnomaly>,
    /// Performance metrics.
    pub performance: PerformanceMetrics,
}

/// Overall health status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealthStatus {
    /// Training is progressing well.
    Healthy,
    /// Minor issues detected.
    Warning,
    /// Significant problems requiring attention.
    Critical,
    /// Training has stalled or diverged.
    Failed,
}

impl HealthStatus {
    pub fn emoji(&self) -> &'static str {
        match self {
            HealthStatus::Healthy => "✅",
            HealthStatus::Warning => "⚠️",
            HealthStatus::Critical => "🔴",
            HealthStatus::Failed => "💀",
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            HealthStatus::Healthy => "Training progressing normally",
            HealthStatus::Warning => "Minor issues detected - monitor closely",
            HealthStatus::Critical => "Significant problems - intervention recommended",
            HealthStatus::Failed => "Training failed or diverged",
        }
    }
}

/// Loss trajectory analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LossAnalysis {
    /// Current loss value.
    pub current: f32,
    /// Best loss achieved.
    pub best: f32,
    /// Step at which best loss was achieved.
    pub best_step: u64,
    /// Loss trend (positive = increasing, negative = decreasing).
    pub trend: f32,
    /// Loss volatility (std dev of recent losses).
    pub volatility: f32,
    /// Whether loss is plateauing.
    pub is_plateau: bool,
    /// Plateau duration (steps).
    pub plateau_duration: Option<u64>,
    /// Whether loss is diverging.
    pub is_diverging: bool,
    /// Expected final loss (extrapolated).
    pub expected_final: Option<f32>,
    /// Estimated steps to target loss.
    pub steps_to_target: Option<u64>,
}

/// Gradient flow analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientAnalysis {
    /// Current gradient norm.
    pub current_norm: f32,
    /// Average gradient norm.
    pub mean_norm: f32,
    /// Gradient norm trend.
    pub trend: f32,
    /// Number of layers with vanishing gradients.
    pub vanishing_layers: usize,
    /// Number of layers with exploding gradients.
    pub exploding_layers: usize,
    /// Gradient entropy (diversity of gradient directions).
    pub entropy: Option<f32>,
    /// Whether gradients are healthy.
    pub is_healthy: bool,
    /// Problematic layer names.
    pub problematic_layers: Vec<String>,
}

/// Per-layer analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerAnalysis {
    /// Layer index.
    pub layer_idx: usize,
    /// Layer name.
    pub layer_name: String,
    /// Layer health score (0-100).
    pub health_score: u8,
    /// Activation statistics.
    pub activation_stats: ActivationStats,
    /// Gradient statistics.
    pub gradient_stats: GradientStats,
    /// Issues detected.
    pub issues: Vec<LayerIssue>,
    /// Activation trend.
    pub activation_trend: f32,
    /// Dead neuron percentage.
    pub dead_neurons_pct: f32,
    /// Saturation percentage.
    pub saturation_pct: f32,
}

/// Gradient statistics for a layer.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct GradientStats {
    /// Gradient norm.
    pub norm: f32,
    /// Gradient mean.
    pub mean: f32,
    /// Gradient std.
    pub std: f32,
    /// Max absolute gradient.
    pub max_abs: f32,
}

/// Issues detected in a layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LayerIssue {
    /// Gradients vanishing.
    VanishingGradient { norm: f32 },
    /// Gradients exploding.
    ExplodingGradient { norm: f32 },
    /// High percentage of dead neurons.
    DeadNeurons { percentage: f32 },
    /// Activation saturation.
    Saturation { percentage: f32 },
    /// Unstable activations.
    UnstableActivations { volatility: f32 },
    /// Unusual sparsity.
    AbnormalSparsity { sparsity: f32, expected: f32 },
}

impl LayerIssue {
    pub fn severity(&self) -> IssueSeverity {
        match self {
            LayerIssue::VanishingGradient { norm } if *norm < 1e-8 => IssueSeverity::Critical,
            LayerIssue::VanishingGradient { .. } => IssueSeverity::Warning,
            LayerIssue::ExplodingGradient { norm } if *norm > 1000.0 => IssueSeverity::Critical,
            LayerIssue::ExplodingGradient { .. } => IssueSeverity::Warning,
            LayerIssue::DeadNeurons { percentage } if *percentage > 50.0 => IssueSeverity::Critical,
            LayerIssue::DeadNeurons { .. } => IssueSeverity::Warning,
            LayerIssue::Saturation { percentage } if *percentage > 80.0 => IssueSeverity::Critical,
            LayerIssue::Saturation { .. } => IssueSeverity::Warning,
            LayerIssue::UnstableActivations { .. } => IssueSeverity::Warning,
            LayerIssue::AbnormalSparsity { .. } => IssueSeverity::Info,
        }
    }

    pub fn description(&self) -> String {
        match self {
            LayerIssue::VanishingGradient { norm } => {
                format!("Vanishing gradient (norm: {:.2e})", norm)
            }
            LayerIssue::ExplodingGradient { norm } => {
                format!("Exploding gradient (norm: {:.2e})", norm)
            }
            LayerIssue::DeadNeurons { percentage } => {
                format!("{:.1}% dead neurons", percentage)
            }
            LayerIssue::Saturation { percentage } => {
                format!("{:.1}% activation saturation", percentage)
            }
            LayerIssue::UnstableActivations { volatility } => {
                format!("Unstable activations (volatility: {:.2})", volatility)
            }
            LayerIssue::AbnormalSparsity { sparsity, expected } => {
                format!(
                    "Unusual sparsity: {:.1}% (expected ~{:.1}%)",
                    sparsity * 100.0,
                    expected * 100.0
                )
            }
        }
    }
}

/// Issue severity level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum IssueSeverity {
    Info,
    Warning,
    Critical,
}

/// Attention pattern analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionAnalysis {
    /// Average attention entropy.
    pub avg_entropy: f32,
    /// Attention concentration (how focused).
    pub concentration: f32,
    /// Average attention span.
    pub avg_span: f32,
    /// Heads with degenerate patterns.
    pub degenerate_heads: Vec<(usize, usize)>,
    /// Heads with interesting patterns.
    pub notable_heads: Vec<(usize, usize, String)>,
}

/// Actionable recommendation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recommendation {
    /// Recommendation priority.
    pub priority: RecommendationPriority,
    /// Category of recommendation.
    pub category: RecommendationCategory,
    /// Short title.
    pub title: String,
    /// Detailed description.
    pub description: String,
    /// Suggested action.
    pub action: String,
    /// Expected impact.
    pub expected_impact: String,
}

/// Recommendation priority.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RecommendationPriority {
    Low,
    Medium,
    High,
    Urgent,
}

/// Recommendation category.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecommendationCategory {
    LearningRate,
    BatchSize,
    Architecture,
    Regularization,
    DataQuality,
    Checkpointing,
    EarlyStopping,
    Optimization,
}

/// Detected anomaly.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedAnomaly {
    /// Step when detected.
    pub step: u64,
    /// Anomaly type.
    pub anomaly_type: AnomalyType,
    /// Severity.
    pub severity: IssueSeverity,
    /// Description.
    pub description: String,
    /// Associated metrics.
    pub metrics: HashMap<String, f32>,
}

/// Types of anomalies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnomalyType {
    LossSpike,
    GradientExplosion,
    GradientVanishing,
    ActivationCollapse,
    AttentionDegeneration,
    NaNDetected,
    InfDetected,
    MemorySpike,
}

/// Performance metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Steps per second.
    pub steps_per_second: f32,
    /// Tokens per second.
    pub tokens_per_second: f32,
    /// GPU utilization percentage.
    pub gpu_utilization: Option<f32>,
    /// GPU memory used (MB).
    pub gpu_memory_mb: Option<f32>,
    /// Time per forward pass (ms).
    pub forward_time_ms: f32,
    /// Time per backward pass (ms).
    pub backward_time_ms: f32,
    /// Estimated time to completion.
    pub eta_seconds: Option<u64>,
    /// Prediction phase ratio (for hybrid training).
    pub prediction_ratio: Option<f32>,
}

/// Training analyzer that produces summaries.
pub struct TrainingAnalyzer {
    /// History of step stats for analysis.
    stats_history: VecDeque<StepStats>,
    /// History of anomalies.
    anomaly_history: Vec<DetectedAnomaly>,
    /// Best loss seen.
    best_loss: f32,
    /// Step of best loss.
    best_step: u64,
    /// Plateau detection window.
    plateau_window: usize,
    /// Plateau threshold (relative change).
    plateau_threshold: f32,
}

impl TrainingAnalyzer {
    pub fn new() -> Self {
        Self {
            stats_history: VecDeque::with_capacity(1000),
            anomaly_history: Vec::new(),
            best_loss: f32::MAX,
            best_step: 0,
            plateau_window: 100,
            plateau_threshold: 0.001,
        }
    }

    /// Add a step to the analyzer.
    pub fn add_step(&mut self, stats: StepStats) {
        if stats.loss < self.best_loss {
            self.best_loss = stats.loss;
            self.best_step = stats.step;
        }

        self.stats_history.push_back(stats);
        if self.stats_history.len() > 1000 {
            self.stats_history.pop_front();
        }
    }

    /// Generate comprehensive training summary.
    pub fn analyze(&self) -> TrainingSummary {
        let current_stats = self.stats_history.back();

        let loss_analysis = self.analyze_loss();
        let gradient_analysis = self.analyze_gradients();
        let layer_analysis = self.analyze_layers();
        let recommendations =
            self.generate_recommendations(&loss_analysis, &gradient_analysis, &layer_analysis);
        let health_score =
            self.compute_health_score(&loss_analysis, &gradient_analysis, &layer_analysis);
        let health_status = self.determine_health_status(health_score);
        let performance = self.compute_performance();

        TrainingSummary {
            step: current_stats.map(|s| s.step).unwrap_or(0),
            health_score,
            health_status,
            loss_analysis,
            gradient_analysis,
            layer_analysis,
            attention_analysis: None, // TODO: implement
            recommendations,
            anomalies: self.anomaly_history.clone(),
            performance,
        }
    }

    fn analyze_loss(&self) -> LossAnalysis {
        let losses: Vec<f32> = self.stats_history.iter().map(|s| s.loss).collect();
        let current = losses.last().copied().unwrap_or(0.0);

        let trend = if losses.len() >= 10 {
            compute_linear_trend(&losses[losses.len() - 10..])
        } else {
            0.0
        };

        let volatility = if losses.len() >= 10 {
            let recent = &losses[losses.len() - 10..];
            let mean = recent.iter().sum::<f32>() / recent.len() as f32;
            (recent.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / recent.len() as f32).sqrt()
        } else {
            0.0
        };

        let is_plateau = losses.len() >= self.plateau_window && {
            let window = &losses[losses.len() - self.plateau_window..];
            let first_half = &window[..self.plateau_window / 2];
            let second_half = &window[self.plateau_window / 2..];
            let first_mean = first_half.iter().sum::<f32>() / first_half.len() as f32;
            let second_mean = second_half.iter().sum::<f32>() / second_half.len() as f32;
            (first_mean - second_mean).abs() / first_mean.max(0.001) < self.plateau_threshold
        };

        let is_diverging = trend > 0.01 && losses.len() > 50;

        LossAnalysis {
            current,
            best: self.best_loss,
            best_step: self.best_step,
            trend,
            volatility,
            is_plateau,
            plateau_duration: if is_plateau {
                Some(self.plateau_window as u64)
            } else {
                None
            },
            is_diverging,
            expected_final: None,
            steps_to_target: None,
        }
    }

    fn analyze_gradients(&self) -> GradientAnalysis {
        let grads: Vec<f32> = self.stats_history.iter().map(|s| s.gradient_norm).collect();
        let current_norm = grads.last().copied().unwrap_or(0.0);
        let mean_norm = if grads.is_empty() {
            0.0
        } else {
            grads.iter().sum::<f32>() / grads.len() as f32
        };

        let trend = if grads.len() >= 10 {
            compute_linear_trend(&grads[grads.len() - 10..])
        } else {
            0.0
        };

        let mut vanishing_layers = 0;
        let mut exploding_layers = 0;
        let mut problematic_layers = Vec::new();

        if let Some(stats) = self.stats_history.back() {
            for layer in &stats.layer_stats {
                if layer.gradient_norm < 1e-6 {
                    vanishing_layers += 1;
                    problematic_layers.push(format!("Layer {} (vanishing)", layer.layer_idx));
                } else if layer.gradient_norm > 100.0 {
                    exploding_layers += 1;
                    problematic_layers.push(format!("Layer {} (exploding)", layer.layer_idx));
                }
            }
        }

        let is_healthy = vanishing_layers == 0
            && exploding_layers == 0
            && current_norm > 1e-6
            && current_norm < 100.0;

        GradientAnalysis {
            current_norm,
            mean_norm,
            trend,
            vanishing_layers,
            exploding_layers,
            entropy: None,
            is_healthy,
            problematic_layers,
        }
    }

    fn analyze_layers(&self) -> Vec<LayerAnalysis> {
        let Some(stats) = self.stats_history.back() else {
            return Vec::new();
        };

        stats
            .layer_stats
            .iter()
            .map(|layer| {
                let mut issues = Vec::new();

                // Check for vanishing/exploding gradients
                if layer.gradient_norm < 1e-6 {
                    issues.push(LayerIssue::VanishingGradient {
                        norm: layer.gradient_norm,
                    });
                } else if layer.gradient_norm > 100.0 {
                    issues.push(LayerIssue::ExplodingGradient {
                        norm: layer.gradient_norm,
                    });
                }

                // Check for dead neurons
                if layer.dead_neuron_fraction > 0.1 {
                    issues.push(LayerIssue::DeadNeurons {
                        percentage: layer.dead_neuron_fraction * 100.0,
                    });
                }

                // Check for saturation
                let saturation_pct = if layer.activation_max - layer.activation_min < 0.1 {
                    ((layer.activation_max - layer.activation_mean).abs()
                        / (layer.activation_max - layer.activation_min + 0.001))
                        .min(1.0)
                } else {
                    0.0
                };
                if saturation_pct > 0.5 {
                    issues.push(LayerIssue::Saturation {
                        percentage: saturation_pct * 100.0,
                    });
                }

                let health_score = if issues.is_empty() {
                    100
                } else {
                    100 - (issues
                        .iter()
                        .map(|i| match i.severity() {
                            IssueSeverity::Critical => 40,
                            IssueSeverity::Warning => 20,
                            IssueSeverity::Info => 5,
                        })
                        .sum::<u8>())
                    .min(100)
                };

                LayerAnalysis {
                    layer_idx: layer.layer_idx,
                    layer_name: format!("layer_{}", layer.layer_idx),
                    health_score,
                    activation_stats: ActivationStats {
                        mean: layer.activation_mean,
                        std: layer.activation_std,
                        min: layer.activation_min,
                        max: layer.activation_max,
                        sparsity: layer.sparsity,
                    },
                    gradient_stats: GradientStats {
                        norm: layer.gradient_norm,
                        mean: 0.0,
                        std: 0.0,
                        max_abs: 0.0,
                    },
                    issues,
                    activation_trend: 0.0,
                    dead_neurons_pct: layer.dead_neuron_fraction * 100.0,
                    saturation_pct: saturation_pct * 100.0,
                }
            })
            .collect()
    }

    fn generate_recommendations(
        &self,
        loss: &LossAnalysis,
        grads: &GradientAnalysis,
        _layers: &[LayerAnalysis],
    ) -> Vec<Recommendation> {
        let mut recs = Vec::new();

        // Learning rate recommendations
        if loss.is_plateau {
            recs.push(Recommendation {
                priority: RecommendationPriority::High,
                category: RecommendationCategory::LearningRate,
                title: "Plateau detected".to_string(),
                description: "Loss has stopped decreasing significantly".to_string(),
                action: "Increase learning rate by 20-50% or apply learning rate warmup restart"
                    .to_string(),
                expected_impact: "May help escape local minimum".to_string(),
            });
        }

        if loss.volatility > 0.5 {
            recs.push(Recommendation {
                priority: RecommendationPriority::Medium,
                category: RecommendationCategory::LearningRate,
                title: "High loss volatility".to_string(),
                description: format!("Loss standard deviation is {:.3}", loss.volatility),
                action: "Reduce learning rate by 30-50%".to_string(),
                expected_impact: "Smoother convergence".to_string(),
            });
        }

        if loss.is_diverging {
            recs.push(Recommendation {
                priority: RecommendationPriority::Urgent,
                category: RecommendationCategory::LearningRate,
                title: "Loss diverging".to_string(),
                description: "Loss is consistently increasing".to_string(),
                action: "Immediately reduce learning rate by 50-75% or restore from checkpoint"
                    .to_string(),
                expected_impact: "Prevent training failure".to_string(),
            });
        }

        // Gradient recommendations
        if grads.vanishing_layers > 0 {
            recs.push(Recommendation {
                priority: RecommendationPriority::High,
                category: RecommendationCategory::Architecture,
                title: "Vanishing gradients".to_string(),
                description: format!("{} layers have vanishing gradients", grads.vanishing_layers),
                action:
                    "Consider using skip connections, better initialization, or gradient clipping"
                        .to_string(),
                expected_impact: "Improved gradient flow".to_string(),
            });
        }

        if grads.exploding_layers > 0 {
            recs.push(Recommendation {
                priority: RecommendationPriority::Urgent,
                category: RecommendationCategory::Optimization,
                title: "Exploding gradients".to_string(),
                description: format!("{} layers have exploding gradients", grads.exploding_layers),
                action: "Apply gradient clipping (max_norm=1.0) immediately".to_string(),
                expected_impact: "Prevent NaN and training collapse".to_string(),
            });
        }

        recs
    }

    fn compute_health_score(
        &self,
        loss: &LossAnalysis,
        grads: &GradientAnalysis,
        layers: &[LayerAnalysis],
    ) -> u8 {
        let mut score = 100i32;

        // Loss penalties
        if loss.is_diverging {
            score -= 50;
        }
        if loss.is_plateau {
            score -= 15;
        }
        if loss.volatility > 0.5 {
            score -= 10;
        }

        // Gradient penalties
        score -= (grads.vanishing_layers * 10) as i32;
        score -= (grads.exploding_layers * 20) as i32;

        // Layer penalties
        let avg_layer_health = if layers.is_empty() {
            100
        } else {
            layers.iter().map(|l| l.health_score as u32).sum::<u32>() / layers.len() as u32
        };
        score -= (100 - avg_layer_health as i32) / 4;

        score.max(0).min(100) as u8
    }

    fn determine_health_status(&self, score: u8) -> HealthStatus {
        match score {
            80..=100 => HealthStatus::Healthy,
            50..=79 => HealthStatus::Warning,
            20..=49 => HealthStatus::Critical,
            _ => HealthStatus::Failed,
        }
    }

    fn compute_performance(&self) -> PerformanceMetrics {
        let steps_per_second = if self.stats_history.len() >= 2 {
            let first = self.stats_history.front().unwrap();
            let last = self.stats_history.back().unwrap();
            let time_diff = (last.timestamp_ms - first.timestamp_ms) as f32 / 1000.0;
            let step_diff = (last.step - first.step) as f32;
            if time_diff > 0.0 {
                step_diff / time_diff
            } else {
                0.0
            }
        } else {
            0.0
        };

        PerformanceMetrics {
            steps_per_second,
            tokens_per_second: 0.0, // TODO: compute from batch size
            gpu_utilization: None,
            gpu_memory_mb: None,
            forward_time_ms: 0.0,
            backward_time_ms: 0.0,
            eta_seconds: None,
            prediction_ratio: None,
        }
    }
}

impl Default for TrainingAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute linear trend (slope) of values.
fn compute_linear_trend(values: &[f32]) -> f32 {
    if values.len() < 2 {
        return 0.0;
    }

    let n = values.len() as f32;
    let x_mean = (n - 1.0) / 2.0;
    let y_mean = values.iter().sum::<f32>() / n;

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

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_capture_config_builder() {
        let config = CaptureConfig::builder()
            .sample_every_n_steps(50)
            .max_history(500)
            .trigger_on_loss_spike(2.5)
            .build();

        assert_eq!(config.sample_every_n_steps, 50);
        assert_eq!(config.max_history, 500);
        assert!((config.trigger_loss_spike_std - 2.5).abs() < 1e-6);
    }

    #[test]
    fn test_anomaly_detector() {
        let config = CaptureConfig::default();
        let mut detector = AnomalyDetector::new(config);

        // Feed normal data
        for i in 0..100 {
            detector.update(2.0 + (i as f32 * 0.01), 1.0);
        }

        // No anomaly for normal loss
        assert!(detector.detect(2.5, 1.0).is_none());

        // Loss spike should trigger
        // Need to wait for cooldown in real usage
    }

    #[test]
    fn test_layer_activation_sample() {
        let values = vec![0.0, 0.5, 1.0, 1.5, 2.0, 0.1, 0.9, 1.1];
        let sample = LayerActivationSample::new(0, "layer_0".to_string(), values);

        assert_eq!(sample.layer_idx, 0);
        assert!(sample.stats.mean > 0.0);
        assert!(sample.histogram.iter().sum::<u32>() == 8);
    }

    #[test]
    fn test_dashboard_presets() {
        let panels = DashboardPreset::Full.panels();
        assert_eq!(panels.len(), 6);

        let minimal = DashboardPreset::Minimal.panels();
        assert_eq!(minimal.len(), 2);
    }

    #[test]
    fn test_animation_controller() {
        let mut controller = AnimationController::new(12);
        controller.start_forward();

        assert_eq!(controller.phase, AnimPhase::Forward);

        // Update a few frames
        for _ in 0..50 {
            controller.update(1);
        }

        assert!(controller.progress > 0.0);
    }
}
