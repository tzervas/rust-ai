//! Live training monitor with real-time streaming and multi-pane views.
//!
//! Features:
//! - Multi-pane views with Tab navigation (Overview, Charts, GPU, History)
//! - Multiple chart types: line, scatter with trend, flame graphs
//! - Live mode vs History mode toggle
//! - Comprehensive GPU monitoring
//! - High-contrast visualization

use std::collections::{HashMap, VecDeque};
use std::fs::File;
use std::io::{self, BufRead, BufReader, Seek, SeekFrom, Stdout};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use chrono::Utc;
use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    symbols,
    text::{Line, Span},
    widgets::{
        Axis, BarChart, Block, Borders, Chart, Dataset, Gauge, GraphType, List, ListItem,
        Paragraph, Tabs, Wrap,
    },
    Frame, Terminal,
};

use crate::gpu_stats::GpuStatsMonitor;
use crate::training_state::{RunManager, StepMetrics, TrainingPhase, TrainingRun, TrainingStatus};

/// RAII guard to ensure terminal state is restored even on panic.
struct TerminalCleanup;

impl Drop for TerminalCleanup {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let _ = execute!(std::io::stdout(), LeaveAlternateScreen);
    }
}

/// High contrast color palette for better visibility.
pub mod colors {
    use ratatui::style::Color;

    // Chart colors - optimized for dark backgrounds
    pub const LOSS_LINE: Color = Color::Rgb(0, 255, 255);      // Bright cyan - primary data
    pub const LOSS_SCATTER: Color = Color::Rgb(255, 255, 0);   // Bright yellow dots
    pub const TREND_LINE: Color = Color::Rgb(255, 100, 255);   // Magenta trend
    pub const GRAD_NORM: Color = Color::Rgb(50, 255, 50);      // Brighter green for visibility
    pub const STEP_TIME: Color = Color::Rgb(255, 180, 50);     // Orange
    pub const PREDICTION: Color = Color::Rgb(120, 220, 255);   // Brighter light blue

    // Phase colors
    pub const WARMUP: Color = Color::Rgb(255, 200, 0);         // Orange
    pub const FULL: Color = Color::Rgb(0, 180, 255);           // Brighter blue
    pub const PREDICT: Color = Color::Rgb(0, 255, 100);        // Green
    pub const CORRECT: Color = Color::Rgb(255, 100, 255);      // Magenta

    // Status colors
    pub const ACTIVE_RUN: Color = Color::Rgb(0, 255, 0);       // Bright green
    pub const COMPLETED_RUN: Color = Color::Rgb(100, 200, 255);// Light blue
    pub const FAILED_RUN: Color = Color::Rgb(255, 80, 80);     // Red
    pub const PAUSED_RUN: Color = Color::Rgb(255, 255, 0);     // Yellow

    // GPU colors
    pub const MEMORY_OK: Color = Color::Rgb(0, 220, 100);
    pub const MEMORY_WARN: Color = Color::Rgb(255, 200, 0);
    pub const MEMORY_CRIT: Color = Color::Rgb(255, 50, 50);
    pub const TEMP_OK: Color = Color::Rgb(100, 220, 100);
    pub const TEMP_WARN: Color = Color::Rgb(255, 200, 0);
    pub const TEMP_CRIT: Color = Color::Rgb(255, 50, 50);

    // UI colors - improved contrast
    pub const HEADER_BG: Color = Color::Rgb(30, 30, 50);
    pub const SELECTED_BG: Color = Color::Rgb(60, 60, 100);    // Brighter selection
    pub const TAB_ACTIVE: Color = Color::Rgb(0, 220, 255);     // Brighter active tab
    pub const TAB_INACTIVE: Color = Color::Rgb(120, 120, 160); // More readable inactive
    pub const TAB_ACTIVE_BG: Color = Color::Rgb(40, 60, 90);   // Background for active tab
    pub const BORDER: Color = Color::Rgb(100, 100, 150);       // Brighter border
    pub const BORDER_ACTIVE: Color = Color::Rgb(0, 180, 255);  // Active panel border
    pub const HELP_KEY: Color = Color::Rgb(255, 220, 80);      // Brighter help keys
    pub const LIVE_INDICATOR: Color = Color::Rgb(255, 50, 50);
    pub const LABEL_DIM: Color = Color::Rgb(150, 150, 170);    // Dimmed labels
    pub const LABEL_BRIGHT: Color = Color::Rgb(220, 220, 240); // Bright labels

    // Flame graph colors
    pub const FLAME_HOT: Color = Color::Rgb(255, 80, 0);
    pub const FLAME_WARM: Color = Color::Rgb(255, 180, 0);
    pub const FLAME_COOL: Color = Color::Rgb(100, 200, 100);

    // Advanced chart colors - distinct and readable
    pub const EMA_FAST: Color = Color::Rgb(0, 255, 200);        // Cyan-green for EMA-10
    pub const EMA_SLOW: Color = Color::Rgb(255, 100, 255);      // Pink-magenta for EMA-50
    pub const LINEAR_TREND: Color = Color::Rgb(255, 255, 100);  // Bright yellow for regression
    pub const BOLLINGER_BAND: Color = Color::Rgb(140, 140, 220);// Brighter blue for bands
    pub const BOLLINGER_FILL: Color = Color::Rgb(50, 50, 100);  // Darker for fill area
    pub const VELOCITY_POS: Color = Color::Rgb(0, 255, 100);    // Green for improving
    pub const VELOCITY_NEG: Color = Color::Rgb(255, 80, 80);    // Red for worsening

    // Signal colors - VERY PROMINENT for warnings
    pub const SIGNAL_SPIKE: Color = Color::Rgb(255, 0, 0);      // Pure red for spike
    pub const SIGNAL_PLATEAU: Color = Color::Rgb(255, 200, 0);  // Orange-yellow - ATTENTION
    pub const SIGNAL_PLATEAU_BG: Color = Color::Rgb(80, 50, 0); // Dark orange background
    pub const SIGNAL_INFLECTION: Color = Color::Rgb(100, 180, 255);// Brighter blue
    pub const SIGNAL_RECOVERY: Color = Color::Rgb(0, 255, 0);   // Pure green for recovery

    // Learning rate display
    pub const LR_NORMAL: Color = Color::Rgb(255, 200, 0);       // Standard LR color
    pub const LR_CHANGED: Color = Color::Rgb(255, 100, 255);    // LR recently changed
    pub const LR_CRITICAL: Color = Color::Rgb(255, 50, 50);     // LR needs attention

    // Chart grid and axis
    pub const AXIS_LABEL: Color = Color::Rgb(180, 180, 200);    // Readable axis labels
    pub const GRID_LINE: Color = Color::Rgb(60, 60, 80);        // Subtle grid
}

/// View mode for the monitor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViewMode {
    Live,
    History,
}

/// Main tab selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MainTab {
    Overview,
    Charts,
    Dimensions,
    GPU,
    History,
    Help,
}

impl MainTab {
    fn all() -> &'static [MainTab] {
        &[
            MainTab::Overview,
            MainTab::Charts,
            MainTab::Dimensions,
            MainTab::GPU,
            MainTab::History,
            MainTab::Help,
        ]
    }

    fn title(&self) -> &'static str {
        match self {
            MainTab::Overview => "Overview",
            MainTab::Charts => "Charts",
            MainTab::Dimensions => "Dims",
            MainTab::GPU => "GPU",
            MainTab::History => "History",
            MainTab::Help => "Help",
        }
    }

    fn index(&self) -> usize {
        match self {
            MainTab::Overview => 0,
            MainTab::Charts => 1,
            MainTab::Dimensions => 2,
            MainTab::GPU => 3,
            MainTab::History => 4,
            MainTab::Help => 5,
        }
    }
}

/// Chart type selection (within Charts tab).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChartType {
    LossLine,
    LossScatter,
    GradientNorm,
    StepTime,
    PhaseBreakdown,
    PredictionAccuracy,
}

impl ChartType {
    fn all() -> &'static [ChartType] {
        &[
            ChartType::LossLine,
            ChartType::LossScatter,
            ChartType::GradientNorm,
            ChartType::StepTime,
            ChartType::PhaseBreakdown,
            ChartType::PredictionAccuracy,
        ]
    }

    fn title(&self) -> &'static str {
        match self {
            ChartType::LossLine => "Loss (Line)",
            ChartType::LossScatter => "Loss (Scatter+Trend)",
            ChartType::GradientNorm => "Gradient Norm",
            ChartType::StepTime => "Step Time",
            ChartType::PhaseBreakdown => "Phase Breakdown",
            ChartType::PredictionAccuracy => "Prediction Accuracy",
        }
    }
}

/// Smoothing method for chart data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SmoothingMethod {
    #[default]
    Raw,
    Ema,
    Sma,
    Bilateral,
}

impl SmoothingMethod {
    fn all() -> &'static [SmoothingMethod] {
        &[
            SmoothingMethod::Raw,
            SmoothingMethod::Ema,
            SmoothingMethod::Sma,
            SmoothingMethod::Bilateral,
        ]
    }

    fn title(&self) -> &'static str {
        match self {
            SmoothingMethod::Raw => "Raw",
            SmoothingMethod::Ema => "EMA",
            SmoothingMethod::Sma => "SMA",
            SmoothingMethod::Bilateral => "Bilateral",
        }
    }

    fn next(&self) -> Self {
        let all = Self::all();
        let idx = all.iter().position(|m| m == self).unwrap_or(0);
        all[(idx + 1) % all.len()]
    }
}

/// Signal types for training dynamics detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignalType {
    /// Loss spike detected (sudden increase)
    Spike,
    /// Plateau warning (loss not improving)
    Plateau,
    /// Inflection point (change in curvature)
    Inflection,
    /// Recovery confirmed (sustained improvement after issue)
    Recovery,
}

impl SignalType {
    fn emoji(&self) -> &'static str {
        match self {
            SignalType::Spike => "!!",
            SignalType::Plateau => "~~",
            SignalType::Inflection => "<>",
            SignalType::Recovery => "OK",
        }
    }

    fn color(&self) -> Color {
        match self {
            SignalType::Spike => colors::SIGNAL_SPIKE,
            SignalType::Plateau => colors::SIGNAL_PLATEAU,
            SignalType::Inflection => colors::SIGNAL_INFLECTION,
            SignalType::Recovery => colors::SIGNAL_RECOVERY,
        }
    }

    fn description(&self) -> &'static str {
        match self {
            SignalType::Spike => "Loss spike detected",
            SignalType::Plateau => "Plateau warning",
            SignalType::Inflection => "Inflection point",
            SignalType::Recovery => "Recovery confirmed",
        }
    }
}

/// GPU metrics sub-view.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuView {
    Summary,
    Memory,
    Thermal,
    Utilization,
    FlameGraph,
}

impl GpuView {
    fn all() -> &'static [GpuView] {
        &[GpuView::Summary, GpuView::Memory, GpuView::Thermal, GpuView::Utilization, GpuView::FlameGraph]
    }

    fn title(&self) -> &'static str {
        match self {
            GpuView::Summary => "Summary",
            GpuView::Memory => "Memory",
            GpuView::Thermal => "Thermal",
            GpuView::Utilization => "Util",
            GpuView::FlameGraph => "Flame",
        }
    }
}

/// 5D projection modes for dimensional visualization.
///
/// The 5 dimensions are: layer, time, gradient, loss, and attention.
/// Each projection mode selects 2 dimensions for the X and Y axes,
/// with remaining dimensions encoded as color/character intensity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DimensionProjection {
    /// Layer (Y) x Time (X), gradient as intensity, loss as character
    LayerTime,
    /// Layer (Y) x Attention head (X), activity as intensity
    LayerAttention,
    /// Time (X) x Gradient (Y), layer as color gradient
    TimeGradient,
    /// Layer (Y) x Loss contribution (X), time as animation
    LayerLoss,
}

impl DimensionProjection {
    fn all() -> &'static [DimensionProjection] {
        &[
            DimensionProjection::LayerTime,
            DimensionProjection::LayerAttention,
            DimensionProjection::TimeGradient,
            DimensionProjection::LayerLoss,
        ]
    }

    fn title(&self) -> &'static str {
        match self {
            DimensionProjection::LayerTime => "Layer x Time",
            DimensionProjection::LayerAttention => "Layer x Attn",
            DimensionProjection::TimeGradient => "Time x Grad",
            DimensionProjection::LayerLoss => "Layer x Loss",
        }
    }

    fn description(&self) -> &'static str {
        match self {
            DimensionProjection::LayerTime => "Gradient heatmap over time (Y: layers, X: steps)",
            DimensionProjection::LayerAttention => "Attention head activity per layer",
            DimensionProjection::TimeGradient => "Temporal gradient flow patterns",
            DimensionProjection::LayerLoss => "Layer-wise loss contribution",
        }
    }
}

/// Sub-view modes within the Dimensions tab.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DimensionView {
    /// 4D heatmap: layer x time x gradient x loss
    Heatmap4D,
    /// 5D slice view with projection selector
    Slice5D,
    /// Animated temporal evolution
    Animation,
    /// Summary statistics
    Summary,
}

impl DimensionView {
    fn all() -> &'static [DimensionView] {
        &[
            DimensionView::Heatmap4D,
            DimensionView::Slice5D,
            DimensionView::Animation,
            DimensionView::Summary,
        ]
    }

    fn title(&self) -> &'static str {
        match self {
            DimensionView::Heatmap4D => "4D Heatmap",
            DimensionView::Slice5D => "5D Slice",
            DimensionView::Animation => "Animation",
            DimensionView::Summary => "Summary",
        }
    }
}

/// Historical gradient data per layer for temporal visualization.
#[derive(Clone, Default)]
pub struct LayerGradientHistory {
    /// Map of layer name -> vec of (step, gradient_norm)
    pub layer_data: HashMap<String, VecDeque<(u64, f32)>>,
    /// Loss values per step
    pub loss_history: VecDeque<(u64, f32)>,
    /// Maximum number of steps to retain
    pub max_steps: usize,
}

impl LayerGradientHistory {
    pub fn new(max_steps: usize) -> Self {
        Self {
            layer_data: HashMap::new(),
            loss_history: VecDeque::with_capacity(max_steps),
            max_steps,
        }
    }

    /// Add gradient data for a step from a StepMetrics.
    pub fn add_from_metrics(&mut self, metrics: &StepMetrics) {
        // Store loss
        self.loss_history.push_back((metrics.step, metrics.loss));
        while self.loss_history.len() > self.max_steps {
            self.loss_history.pop_front();
        }

        // Store layer gradients
        if let Some(ref layer_grads) = metrics.layer_gradients {
            for (layer_name, &grad_norm) in layer_grads {
                let history = self
                    .layer_data
                    .entry(layer_name.clone())
                    .or_insert_with(|| VecDeque::with_capacity(self.max_steps));
                history.push_back((metrics.step, grad_norm));
                while history.len() > self.max_steps {
                    history.pop_front();
                }
            }
        }
    }

    /// Get gradient at a specific step for a layer.
    pub fn get_gradient(&self, layer: &str, step: u64) -> Option<f32> {
        self.layer_data.get(layer).and_then(|history| {
            for (s, g) in history {
                if *s == step {
                    return Some(*g);
                }
            }
            None
        })
    }

    /// Get loss at a specific step.
    pub fn get_loss(&self, step: u64) -> Option<f32> {
        for (s, l) in &self.loss_history {
            if *s == step {
                return Some(*l);
            }
        }
        None
    }

    /// Get all layers sorted by name.
    pub fn sorted_layers(&self) -> Vec<String> {
        let mut layers: Vec<_> = self.layer_data.keys().cloned().collect();
        layers.sort();
        layers
    }

    /// Get step range covered by the data.
    pub fn step_range(&self) -> Option<(u64, u64)> {
        let mut min_step = u64::MAX;
        let mut max_step = 0;

        for history in self.layer_data.values() {
            if let Some((first, _)) = history.front() {
                min_step = min_step.min(*first);
            }
            if let Some((last, _)) = history.back() {
                max_step = max_step.max(*last);
            }
        }

        if min_step <= max_step {
            Some((min_step, max_step))
        } else {
            None
        }
    }

    /// Get recent steps (last N).
    pub fn recent_steps(&self, n: usize) -> Vec<u64> {
        self.loss_history
            .iter()
            .rev()
            .take(n)
            .map(|(s, _)| *s)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect()
    }
}

/// Live metrics reader that tails the metrics file.
pub struct LiveMetricsReader {
    file_path: PathBuf,
    file: Option<BufReader<File>>,
    last_position: u64,
    metrics: VecDeque<StepMetrics>,
    max_history: usize,
}

impl LiveMetricsReader {
    pub fn new(file_path: PathBuf, max_history: usize) -> Self {
        Self {
            file_path,
            file: None,
            last_position: 0,
            metrics: VecDeque::with_capacity(max_history),
            max_history,
        }
    }

    /// Load entire file history (for history mode).
    pub fn load_full_history(&mut self) -> io::Result<()> {
        self.metrics.clear();
        self.last_position = 0;
        self.file = None;

        if !self.file_path.exists() {
            return Ok(());
        }

        let file = File::open(&self.file_path)?;
        let reader = BufReader::new(file);

        for line in reader.lines() {
            if let Ok(line) = line {
                if let Ok(metric) = serde_json::from_str::<StepMetrics>(&line) {
                    self.metrics.push_back(metric);
                }
            }
        }

        Ok(())
    }

    /// Open or reopen the file.
    fn ensure_file(&mut self) -> io::Result<()> {
        if self.file.is_none() && self.file_path.exists() {
            let file = File::open(&self.file_path)?;
            let mut reader = BufReader::new(file);
            reader.seek(SeekFrom::Start(self.last_position))?;
            self.file = Some(reader);
        }
        Ok(())
    }

    /// Read new metrics from the file (non-blocking).
    pub fn poll(&mut self) -> Vec<StepMetrics> {
        let mut new_metrics = Vec::new();

        if self.ensure_file().is_err() {
            return new_metrics;
        }

        if let Some(ref mut reader) = self.file {
            let mut line = String::new();
            loop {
                line.clear();
                match reader.read_line(&mut line) {
                    Ok(0) => break,
                    Ok(_) => {
                        if let Ok(metric) = serde_json::from_str::<StepMetrics>(&line) {
                            new_metrics.push(metric.clone());
                            self.metrics.push_back(metric);
                            if self.metrics.len() > self.max_history {
                                self.metrics.pop_front();
                            }
                        }
                    }
                    Err(_) => break,
                }
            }
            if let Ok(pos) = reader.stream_position() {
                self.last_position = pos;
            }
        }

        new_metrics
    }

    pub fn all_metrics(&self) -> &VecDeque<StepMetrics> {
        &self.metrics
    }

    pub fn loss_data(&self) -> Vec<(f64, f64)> {
        self.metrics.iter().map(|m| (m.step as f64, m.loss as f64)).collect()
    }

    pub fn gradient_data(&self) -> Vec<(f64, f64)> {
        self.metrics
            .iter()
            .map(|m| (m.step as f64, m.gradient_norm as f64))
            .collect()
    }

    pub fn step_time_data(&self) -> Vec<(f64, f64)> {
        self.metrics.iter().map(|m| (m.step as f64, m.step_time_ms as f64)).collect()
    }

    pub fn prediction_data(&self) -> Vec<(f64, f64)> {
        self.metrics
            .iter()
            .filter(|m| m.prediction_error.is_some())
            .map(|m| (m.step as f64, m.prediction_error.unwrap() as f64))
            .collect()
    }

    pub fn confidence_data(&self) -> Vec<(f64, f64)> {
        self.metrics
            .iter()
            .map(|m| (m.step as f64, m.confidence as f64 * 100.0))
            .collect()
    }

    pub fn token_throughput_data(&self) -> Vec<(f64, f64)> {
        if self.metrics.len() < 2 {
            return Vec::new();
        }

        // Compute tokens per second for each step
        let mut result = Vec::new();
        let tokens_per_step = self.metrics.front().map(|m| m.tokens_this_step).unwrap_or(1024);

        for (i, m) in self.metrics.iter().enumerate().skip(1) {
            if let Some(prev) = self.metrics.get(i - 1) {
                let time_delta = (m.timestamp - prev.timestamp).num_milliseconds() as f64 / 1000.0;
                if time_delta > 0.0 {
                    let tokens_sec = tokens_per_step as f64 / time_delta;
                    result.push((m.step as f64, tokens_sec));
                }
            }
        }
        result
    }

    /// Compute smoothed/EMA trend line.
    pub fn smoothed_loss(&self, alpha: f64) -> Vec<(f64, f64)> {
        let mut result = Vec::new();
        let mut ema: Option<f64> = None;

        for m in &self.metrics {
            let loss = m.loss as f64;
            ema = Some(match ema {
                Some(prev) => alpha * loss + (1.0 - alpha) * prev,
                None => loss,
            });
            result.push((m.step as f64, ema.unwrap()));
        }

        result
    }

    pub fn latest(&self) -> Option<&StepMetrics> {
        self.metrics.back()
    }

    pub fn clear(&mut self) {
        self.metrics.clear();
        self.last_position = 0;
        self.file = None;
    }

    /// Phase statistics.
    pub fn phase_stats(&self) -> HashMap<TrainingPhase, (usize, f64, f64)> {
        let mut stats: HashMap<TrainingPhase, (usize, f64, f64)> = HashMap::new();

        for m in &self.metrics {
            let entry = stats.entry(m.phase).or_insert((0, 0.0, 0.0));
            entry.0 += 1;
            entry.1 += m.loss as f64;
            entry.2 += m.step_time_ms as f64;
        }

        for (_, (count, loss_sum, time_sum)) in stats.iter_mut() {
            if *count > 0 {
                *loss_sum /= *count as f64;
                *time_sum /= *count as f64;
            }
        }

        stats
    }
}

// =============================================================================
// Chart Analysis Helper Functions
// =============================================================================

/// Compute exponential moving average (EMA) with given period.
/// The alpha is calculated as 2 / (period + 1).
pub fn compute_ema(data: &[f64], period: usize) -> Vec<f64> {
    if data.is_empty() || period == 0 {
        return Vec::new();
    }
    let alpha = 2.0 / (period as f64 + 1.0);
    let mut result = Vec::with_capacity(data.len());
    let mut ema = data[0];

    for &value in data {
        ema = alpha * value + (1.0 - alpha) * ema;
        result.push(ema);
    }

    result
}

/// Compute simple moving average (SMA) with given period.
pub fn compute_sma(data: &[f64], period: usize) -> Vec<f64> {
    if data.is_empty() || period == 0 {
        return Vec::new();
    }

    let mut result = Vec::with_capacity(data.len());
    let mut window_sum: f64 = 0.0;

    for (i, &value) in data.iter().enumerate() {
        window_sum += value;
        if i >= period {
            window_sum -= data[i - period];
            result.push(window_sum / period as f64);
        } else {
            // For initial values, use what we have
            result.push(window_sum / (i + 1) as f64);
        }
    }

    result
}

/// Apply bilateral filter for spike removal.
/// This preserves edges while smoothing noise.
pub fn compute_bilateral_filter(data: &[f64], window: usize, sigma_space: f64, sigma_range: f64) -> Vec<f64> {
    if data.is_empty() {
        return Vec::new();
    }

    let half_window = window / 2;
    let mut result = Vec::with_capacity(data.len());

    for i in 0..data.len() {
        let mut weighted_sum = 0.0;
        let mut weight_sum = 0.0;

        let start = i.saturating_sub(half_window);
        let end = (i + half_window + 1).min(data.len());

        for j in start..end {
            // Spatial weight (distance in index)
            let spatial_dist = (i as f64 - j as f64).abs();
            let spatial_weight = (-spatial_dist.powi(2) / (2.0 * sigma_space.powi(2))).exp();

            // Range weight (difference in value)
            let range_dist = (data[i] - data[j]).abs();
            let range_weight = (-range_dist.powi(2) / (2.0 * sigma_range.powi(2))).exp();

            let weight = spatial_weight * range_weight;
            weighted_sum += data[j] * weight;
            weight_sum += weight;
        }

        if weight_sum > 0.0 {
            result.push(weighted_sum / weight_sum);
        } else {
            result.push(data[i]);
        }
    }

    result
}

/// Compute velocity (first derivative) of the data.
/// Uses central difference where possible, forward/backward at edges.
pub fn compute_velocity(data: &[f64]) -> Vec<f64> {
    if data.len() < 2 {
        return vec![0.0; data.len()];
    }

    let mut result = Vec::with_capacity(data.len());

    // Forward difference for first point
    result.push(data[1] - data[0]);

    // Central difference for middle points
    for i in 1..data.len() - 1 {
        result.push((data[i + 1] - data[i - 1]) / 2.0);
    }

    // Backward difference for last point
    result.push(data[data.len() - 1] - data[data.len() - 2]);

    result
}

/// Compute acceleration (second derivative) of the data.
pub fn compute_acceleration(data: &[f64]) -> Vec<f64> {
    let velocity = compute_velocity(data);
    compute_velocity(&velocity)
}

/// Compute Bollinger Bands: returns (upper_band, lower_band).
/// Uses SMA as the middle band with std_mult standard deviations.
pub fn compute_bollinger_bands(data: &[f64], period: usize, std_mult: f64) -> (Vec<f64>, Vec<f64>) {
    if data.is_empty() || period == 0 {
        return (Vec::new(), Vec::new());
    }

    let sma = compute_sma(data, period);
    let mut upper = Vec::with_capacity(data.len());
    let mut lower = Vec::with_capacity(data.len());

    for i in 0..data.len() {
        // Compute standard deviation for this window
        let start = i.saturating_sub(period.saturating_sub(1));
        let window: Vec<f64> = data[start..=i].to_vec();
        let mean = sma[i];

        let variance: f64 = if window.len() > 1 {
            window.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (window.len() - 1) as f64
        } else {
            0.0
        };
        let std_dev = variance.sqrt();

        upper.push(mean + std_mult * std_dev);
        lower.push(mean - std_mult * std_dev);
    }

    (upper, lower)
}

/// Compute linear regression and return the trend line values.
pub fn compute_linear_regression(data: &[f64]) -> Vec<f64> {
    if data.len() < 2 {
        return data.to_vec();
    }

    let n = data.len() as f64;

    // x values are just indices
    let sum_x: f64 = (0..data.len()).map(|i| i as f64).sum();
    let sum_y: f64 = data.iter().sum();
    let sum_xy: f64 = data.iter().enumerate().map(|(i, &y)| i as f64 * y).sum();
    let sum_x2: f64 = (0..data.len()).map(|i| (i as f64).powi(2)).sum();

    let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x.powi(2));
    let intercept = (sum_y - slope * sum_x) / n;

    (0..data.len())
        .map(|i| slope * i as f64 + intercept)
        .collect()
}

/// Detect training signals from loss data and velocity.
/// Returns a list of (index, SignalType) pairs.
pub fn detect_signals(data: &[f64], velocity: &[f64]) -> Vec<(usize, SignalType)> {
    if data.len() < 10 {
        return Vec::new();
    }

    let mut signals = Vec::new();

    // Compute some statistics
    let mean: f64 = data.iter().sum::<f64>() / data.len() as f64;
    let std_dev: f64 = (data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64).sqrt();

    // Recent mean for plateau detection
    let recent_window = data.len().min(50);
    let recent_data = &data[data.len() - recent_window..];
    let recent_mean: f64 = recent_data.iter().sum::<f64>() / recent_window as f64;

    // Detect spikes (loss > 2 std devs above recent mean)
    for i in data.len().saturating_sub(20)..data.len() {
        if data[i] > recent_mean + 2.0 * std_dev {
            // Check it's actually a spike (preceded by lower values)
            if i > 0 && data[i] > data[i - 1] * 1.5 {
                signals.push((i, SignalType::Spike));
            }
        }
    }

    // Detect plateau (very low slope in recent data)
    if recent_window >= 20 {
        let recent_slope = (recent_data[recent_window - 1] - recent_data[0]) / recent_window as f64;
        if recent_slope.abs() < 0.001 * mean.abs().max(0.01) {
            signals.push((data.len() - 1, SignalType::Plateau));
        }
    }

    // Detect inflection points (sign change in acceleration)
    let acceleration = compute_acceleration(data);
    for i in 1..acceleration.len().saturating_sub(1) {
        if acceleration[i - 1] * acceleration[i + 1] < 0.0 && acceleration[i].abs() < 0.0001 {
            // Only report significant inflections
            if velocity[i].abs() > 0.001 {
                signals.push((i, SignalType::Inflection));
            }
        }
    }

    // Detect recovery (sustained improvement after a spike)
    if velocity.len() >= 10 {
        let recent_velocity = &velocity[velocity.len() - 10..];
        let improving_count = recent_velocity.iter().filter(|&&v| v < 0.0).count();

        // Check if we had a spike recently and are now recovering
        let had_recent_spike = signals.iter().any(|(i, s)| {
            *s == SignalType::Spike && *i > data.len().saturating_sub(30)
        });

        if improving_count >= 7 && had_recent_spike {
            signals.push((data.len() - 1, SignalType::Recovery));
        }
    }

    signals
}

/// GPU timing sample for flame graph.
#[derive(Clone)]
pub struct GpuTimingSample {
    pub timestamp: Instant,
    pub memory_mb: f64,
    pub util_percent: f64,
    pub temp_c: u32,
    pub power_w: f64,
}

/// Live training monitor with streaming updates.
pub struct LiveMonitor {
    _runs_dir: PathBuf,
    run_manager: RunManager,
    selected_run_id: Option<String>,
    metrics_readers: HashMap<String, LiveMetricsReader>,
    gpu_monitor: GpuStatsMonitor,
    gpu_history: VecDeque<GpuTimingSample>,

    // UI state
    view_mode: ViewMode,
    main_tab: MainTab,
    chart_type: ChartType,
    gpu_view: GpuView,
    show_help_overlay: bool,
    should_quit: bool,

    // Chart enhancement state
    smoothing_method: SmoothingMethod,
    show_velocity_overlay: bool,
    show_bollinger_bands: bool,
    show_trend_lines: bool,

    // Dimensions tab state
    dimension_view: DimensionView,
    dimension_projection: DimensionProjection,
    layer_gradient_history: HashMap<String, LayerGradientHistory>,
    animation_frame: usize,
    animation_playing: bool,

    // Timing
    refresh_ms: u64,
    last_gpu_sample: Instant,
    start_time: Instant,
}

impl LiveMonitor {
    pub fn new(runs_dir: PathBuf) -> Self {
        Self {
            _runs_dir: runs_dir.clone(),
            run_manager: RunManager::new(runs_dir),
            selected_run_id: None,
            metrics_readers: HashMap::new(),
            gpu_monitor: GpuStatsMonitor::new(0),
            gpu_history: VecDeque::with_capacity(300),
            view_mode: ViewMode::Live,
            main_tab: MainTab::Overview,
            chart_type: ChartType::LossLine,
            gpu_view: GpuView::Summary,
            show_help_overlay: false,
            should_quit: false,
            smoothing_method: SmoothingMethod::default(),
            show_velocity_overlay: false,
            show_bollinger_bands: true,
            show_trend_lines: true,
            dimension_view: DimensionView::Heatmap4D,
            dimension_projection: DimensionProjection::LayerTime,
            layer_gradient_history: HashMap::new(),
            animation_frame: 0,
            animation_playing: false,
            refresh_ms: 100,
            last_gpu_sample: Instant::now(),
            start_time: Instant::now(),
        }
    }

    pub fn run(&mut self) -> anyhow::Result<()> {
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen)?;

        // RAII guard ensures terminal cleanup even on panic
        let _cleanup = TerminalCleanup;

        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend)?;

        self.run_manager.discover_runs()?;
        self.setup_metrics_readers();

        // Auto-select first run
        if let Some(run) = self.run_manager.active_runs().next() {
            self.selected_run_id = Some(run.run_id.clone());
        } else if let Some(run) = self.run_manager.runs().next() {
            self.selected_run_id = Some(run.run_id.clone());
        }

        let result = self.main_loop(&mut terminal);

        // Explicit cleanup (guard will also clean up on drop, but explicit is clearer)
        disable_raw_mode()?;
        execute!(terminal.backend_mut(), LeaveAlternateScreen)?;

        // Forget the guard to avoid double cleanup
        std::mem::forget(_cleanup);

        result
    }

    fn setup_metrics_readers(&mut self) {
        for run in self.run_manager.runs() {
            if !self.metrics_readers.contains_key(&run.run_id) {
                let reader = LiveMetricsReader::new(run.metrics_file.clone(), 2000);
                self.metrics_readers.insert(run.run_id.clone(), reader);
            }
        }
    }

    fn main_loop(&mut self, terminal: &mut Terminal<CrosstermBackend<Stdout>>) -> anyhow::Result<()> {
        while !self.should_quit {
            // Discover new runs
            self.run_manager.discover_runs()?;
            self.setup_metrics_readers();

            // Poll metrics in live mode
            if self.view_mode == ViewMode::Live {
                // Find the currently active (running) run
                let active_run_id = self.run_manager.active_runs().next().map(|r| r.run_id.clone());

                // Auto-select the active run in live mode
                if let Some(ref active_id) = active_run_id {
                    if self.selected_run_id.as_ref() != Some(active_id) {
                        self.selected_run_id = Some(active_id.clone());
                    }
                }

                // Poll metrics for all runs
                for (_run_id, reader) in &mut self.metrics_readers {
                    reader.poll();
                }
            }

            // Sample GPU stats
            if self.last_gpu_sample.elapsed() > Duration::from_millis(500) {
                self.gpu_monitor.sample();
                if let Some(stats) = self.gpu_monitor.current() {
                    self.gpu_history.push_back(GpuTimingSample {
                        timestamp: Instant::now(),
                        memory_mb: stats.memory_used as f64 / 1e6,
                        util_percent: stats.gpu_util as f64,
                        temp_c: stats.temperature,
                        power_w: stats.power_draw as f64,
                    });
                    if self.gpu_history.len() > 300 {
                        self.gpu_history.pop_front();
                    }
                }
                self.last_gpu_sample = Instant::now();
            }

            terminal.draw(|f| self.draw(f))?;

            if event::poll(Duration::from_millis(self.refresh_ms))? {
                if let Event::Key(key) = event::read()? {
                    if key.kind == KeyEventKind::Press {
                        self.handle_key(key.code, key.modifiers);
                    }
                }
            }
        }

        Ok(())
    }

    fn handle_key(&mut self, key: KeyCode, _modifiers: KeyModifiers) {
        // Help overlay takes priority
        if self.show_help_overlay {
            match key {
                KeyCode::Esc | KeyCode::Char('?') | KeyCode::Char('h') => {
                    self.show_help_overlay = false;
                }
                _ => {}
            }
            return;
        }

        match key {
            // Quit
            KeyCode::Char('q') | KeyCode::Esc => self.should_quit = true,

            // Help
            KeyCode::Char('?') | KeyCode::F(1) => {
                self.show_help_overlay = true;
            }

            // Tab navigation
            KeyCode::Tab => {
                let tabs = MainTab::all();
                let idx = self.main_tab.index();
                self.main_tab = tabs[(idx + 1) % tabs.len()];
            }
            KeyCode::BackTab => {
                let tabs = MainTab::all();
                let idx = self.main_tab.index();
                self.main_tab = tabs[(idx + tabs.len() - 1) % tabs.len()];
            }

            // Number keys for tabs
            KeyCode::Char('1') => self.main_tab = MainTab::Overview,
            KeyCode::Char('2') => self.main_tab = MainTab::Charts,
            KeyCode::Char('3') => self.main_tab = MainTab::Dimensions,
            KeyCode::Char('4') => self.main_tab = MainTab::GPU,
            KeyCode::Char('5') => self.main_tab = MainTab::History,
            KeyCode::Char('6') => self.main_tab = MainTab::Help,

            // Live/History mode toggle
            KeyCode::Char('l') => {
                self.view_mode = match self.view_mode {
                    ViewMode::Live => ViewMode::History,
                    ViewMode::History => ViewMode::Live,
                };
                // Load history if switching to history mode
                if self.view_mode == ViewMode::History {
                    if let Some(ref id) = self.selected_run_id {
                        if let Some(reader) = self.metrics_readers.get_mut(id) {
                            let _ = reader.load_full_history();
                        }
                    }
                }
            }

            // Run navigation
            KeyCode::Up | KeyCode::Char('k') => self.select_previous_run(),
            KeyCode::Down | KeyCode::Char('j') => self.select_next_run(),

            // Sub-view cycling (Charts, Dimensions, GPU tabs)
            KeyCode::Left | KeyCode::Char('[') => {
                if self.main_tab == MainTab::Charts {
                    let types = ChartType::all();
                    let idx = types.iter().position(|t| *t == self.chart_type).unwrap_or(0);
                    self.chart_type = types[(idx + types.len() - 1) % types.len()];
                } else if self.main_tab == MainTab::GPU {
                    let views = GpuView::all();
                    let idx = views.iter().position(|v| *v == self.gpu_view).unwrap_or(0);
                    self.gpu_view = views[(idx + views.len() - 1) % views.len()];
                } else if self.main_tab == MainTab::Dimensions {
                    let views = DimensionView::all();
                    let idx = views.iter().position(|v| *v == self.dimension_view).unwrap_or(0);
                    self.dimension_view = views[(idx + views.len() - 1) % views.len()];
                }
            }
            KeyCode::Right | KeyCode::Char(']') => {
                if self.main_tab == MainTab::Charts {
                    let types = ChartType::all();
                    let idx = types.iter().position(|t| *t == self.chart_type).unwrap_or(0);
                    self.chart_type = types[(idx + 1) % types.len()];
                } else if self.main_tab == MainTab::GPU {
                    let views = GpuView::all();
                    let idx = views.iter().position(|v| *v == self.gpu_view).unwrap_or(0);
                    self.gpu_view = views[(idx + 1) % views.len()];
                } else if self.main_tab == MainTab::Dimensions {
                    let views = DimensionView::all();
                    let idx = views.iter().position(|v| *v == self.dimension_view).unwrap_or(0);
                    self.dimension_view = views[(idx + 1) % views.len()];
                }
            }

            // Projection mode cycling (Dimensions tab only, using up/down when not on runs)
            KeyCode::Char('p') => {
                if self.main_tab == MainTab::Dimensions {
                    let projs = DimensionProjection::all();
                    let idx = projs.iter().position(|p| *p == self.dimension_projection).unwrap_or(0);
                    self.dimension_projection = projs[(idx + 1) % projs.len()];
                }
            }

            // Animation control (Dimensions tab)
            KeyCode::Char(' ') => {
                if self.main_tab == MainTab::Dimensions && self.dimension_view == DimensionView::Animation {
                    self.animation_playing = !self.animation_playing;
                }
            }

            // Refresh
            KeyCode::Char('r') => {
                let _ = self.run_manager.discover_runs();
                self.setup_metrics_readers();
            }

            // Clear history
            KeyCode::Char('c') => {
                if let Some(ref id) = self.selected_run_id {
                    if let Some(reader) = self.metrics_readers.get_mut(id) {
                        reader.clear();
                    }
                }
            }

            // Chart enhancement toggles (only in Charts tab)
            KeyCode::Char('s') => {
                if self.main_tab == MainTab::Charts {
                    self.smoothing_method = self.smoothing_method.next();
                }
            }
            KeyCode::Char('v') => {
                if self.main_tab == MainTab::Charts {
                    self.show_velocity_overlay = !self.show_velocity_overlay;
                }
            }
            KeyCode::Char('b') => {
                if self.main_tab == MainTab::Charts {
                    self.show_bollinger_bands = !self.show_bollinger_bands;
                }
            }
            KeyCode::Char('t') => {
                if self.main_tab == MainTab::Charts {
                    self.show_trend_lines = !self.show_trend_lines;
                }
            }

            _ => {}
        }
    }

    fn select_previous_run(&mut self) {
        let runs: Vec<_> = self.run_manager.runs().collect();
        if runs.is_empty() {
            return;
        }
        let idx = self
            .selected_run_id
            .as_ref()
            .and_then(|id| runs.iter().position(|r| &r.run_id == id))
            .unwrap_or(0);
        let new_idx = if idx == 0 { runs.len() - 1 } else { idx - 1 };
        self.selected_run_id = Some(runs[new_idx].run_id.clone());
    }

    fn select_next_run(&mut self) {
        let runs: Vec<_> = self.run_manager.runs().collect();
        if runs.is_empty() {
            return;
        }
        let idx = self
            .selected_run_id
            .as_ref()
            .and_then(|id| runs.iter().position(|r| &r.run_id == id))
            .unwrap_or(0);
        let new_idx = (idx + 1) % runs.len();
        self.selected_run_id = Some(runs[new_idx].run_id.clone());
    }

    fn draw(&self, f: &mut Frame) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3), // Header with tabs
                Constraint::Min(10),   // Main content
                Constraint::Length(2), // Footer/status bar
            ])
            .split(f.area());

        self.draw_header(f, chunks[0]);
        self.draw_main_content(f, chunks[1]);
        self.draw_footer(f, chunks[2]);

        // Help overlay
        if self.show_help_overlay {
            self.draw_help_overlay(f);
        }
    }

    fn draw_header(&self, f: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Min(40), Constraint::Length(25)])
            .split(area);

        // Tabs - Enhanced visibility with clear active indication
        let titles: Vec<Line> = MainTab::all()
            .iter()
            .enumerate()
            .map(|(i, t)| {
                let is_active = *t == self.main_tab;
                let style = if is_active {
                    // Active tab: bright color, bold, with visual brackets
                    Style::default()
                        .fg(colors::TAB_ACTIVE)
                        .bg(colors::TAB_ACTIVE_BG)
                        .add_modifier(Modifier::BOLD)
                } else {
                    Style::default().fg(colors::TAB_INACTIVE)
                };
                // Add number prefix for quick navigation hint
                let prefix = if is_active {
                    format!("[{}:", i + 1)
                } else {
                    format!(" {}:", i + 1)
                };
                let suffix = if is_active { "]" } else { " " };
                Line::from(vec![
                    Span::styled(prefix, style),
                    Span::styled(t.title(), style),
                    Span::styled(suffix, style),
                ])
            })
            .collect();

        let tabs = Tabs::new(titles)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(colors::BORDER_ACTIVE))
                    .title(Span::styled(
                        " RUST-AI MONITOR ",
                        Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
                    )),
            )
            .select(self.main_tab.index())
            .style(Style::default())
            .highlight_style(Style::default().fg(colors::TAB_ACTIVE).add_modifier(Modifier::BOLD | Modifier::UNDERLINED));

        f.render_widget(tabs, chunks[0]);

        // Live/History mode indicator
        let mode_str = match self.view_mode {
            ViewMode::Live => "● LIVE",
            ViewMode::History => "◆ HISTORY",
        };
        let mode_color = match self.view_mode {
            ViewMode::Live => colors::LIVE_INDICATOR,
            ViewMode::History => colors::COMPLETED_RUN,
        };

        let active = self.run_manager.active_runs().count();

        let status = Paragraph::new(Line::from(vec![
            Span::styled(mode_str, Style::default().fg(mode_color).add_modifier(Modifier::BOLD)),
            Span::raw("  "),
            Span::styled(format!("{} active", active), Style::default().fg(colors::ACTIVE_RUN)),
        ]))
        .alignment(Alignment::Right)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(colors::BORDER)),
        );

        f.render_widget(status, chunks[1]);
    }

    fn draw_main_content(&self, f: &mut Frame, area: Rect) {
        match self.main_tab {
            MainTab::Overview => self.draw_overview(f, area),
            MainTab::Charts => self.draw_charts_tab(f, area),
            MainTab::Dimensions => self.draw_dimensions_tab(f, area),
            MainTab::GPU => self.draw_gpu_tab(f, area),
            MainTab::History => self.draw_history_tab(f, area),
            MainTab::Help => self.draw_help_tab(f, area),
        }
    }

    fn draw_overview(&self, f: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Length(35), Constraint::Min(50)])
            .split(area);

        self.draw_run_list(f, chunks[0]);

        // Right side: details
        let right_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),  // Progress
                Constraint::Length(10), // Stats (expanded for tokens)
                Constraint::Min(8),     // Chart
                Constraint::Length(5),  // GPU summary
            ])
            .split(chunks[1]);

        let selected_run = self.selected_run_id.as_ref().and_then(|id| self.run_manager.get_run(id));
        let reader = self.selected_run_id.as_ref().and_then(|id| self.metrics_readers.get(id));

        match (selected_run, reader) {
            (Some(run), Some(reader)) => {
                self.draw_progress(f, right_chunks[0], run, reader);
                self.draw_stats(f, right_chunks[1], run, reader);
                self.draw_loss_line_chart(f, right_chunks[2], reader, "Loss");
                self.draw_gpu_summary(f, right_chunks[3]);
            }
            _ => {
                let msg = Paragraph::new("No run selected. Use j/k to navigate runs.")
                    .style(Style::default().fg(Color::Gray))
                    .block(Block::default().borders(Borders::ALL).title(" Details "));
                f.render_widget(msg, chunks[1]);
            }
        }
    }

    fn draw_run_list(&self, f: &mut Frame, area: Rect) {
        let runs: Vec<_> = self.run_manager.runs().collect();

        let items: Vec<ListItem> = runs
            .iter()
            .map(|run| {
                let is_selected = self.selected_run_id.as_ref().map(|id| id == &run.run_id).unwrap_or(false);

                let (status_icon, status_color) = match run.status {
                    TrainingStatus::Running => ("▶", colors::ACTIVE_RUN),
                    TrainingStatus::Completed => ("✓", colors::COMPLETED_RUN),
                    TrainingStatus::Failed => ("✗", colors::FAILED_RUN),
                    TrainingStatus::Paused => ("⏸", colors::PAUSED_RUN),
                    TrainingStatus::Cancelled => ("⏹", Color::Gray),
                    TrainingStatus::Initializing => ("..", Color::Cyan),
                };

                let reader = self.metrics_readers.get(&run.run_id);
                let step_info = reader
                    .and_then(|r| r.latest())
                    .map(|m| format!(" #{}", m.step))
                    .unwrap_or_default();

                let loss_info = reader
                    .and_then(|r| r.latest())
                    .map(|m| format!(" L:{:.3}", m.loss))
                    .unwrap_or_default();

                let style = if is_selected {
                    Style::default().fg(Color::White).bg(colors::SELECTED_BG).add_modifier(Modifier::BOLD)
                } else {
                    Style::default()
                };

                ListItem::new(Line::from(vec![
                    Span::styled(format!("{} ", status_icon), Style::default().fg(status_color)),
                    Span::styled(run.run_name.clone(), style.fg(if is_selected { Color::White } else { status_color })),
                    Span::styled(step_info, Style::default().fg(Color::Gray)),
                    Span::styled(loss_info, Style::default().fg(colors::LOSS_LINE)),
                ]))
            })
            .collect();

        let list = List::new(items).block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(colors::BORDER))
                .title(Span::styled(" Runs ", Style::default().fg(Color::White).add_modifier(Modifier::BOLD))),
        );

        f.render_widget(list, area);
    }

    fn draw_progress(&self, f: &mut Frame, area: Rect, run: &TrainingRun, reader: &LiveMetricsReader) {
        let latest = reader.latest();
        let step = latest.map(|m| m.step).unwrap_or(run.current_step);
        let progress = step as f64 / run.config.max_steps.max(1) as f64;

        let phase = latest.map(|m| m.phase).unwrap_or(run.current_phase);
        let phase_color = match phase {
            TrainingPhase::Warmup => colors::WARMUP,
            TrainingPhase::Full => colors::FULL,
            TrainingPhase::Predict => colors::PREDICT,
            TrainingPhase::Correct => colors::CORRECT,
        };

        let title = format!(" {} │ Step {}/{} │ {:?} ", run.run_name, step, run.config.max_steps, phase);

        let gauge = Gauge::default()
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(phase_color))
                    .title(Span::styled(title, Style::default().fg(phase_color))),
            )
            .gauge_style(Style::default().fg(phase_color))
            .ratio(progress.clamp(0.0, 1.0))
            .label(format!("{:.1}%", progress * 100.0));

        f.render_widget(gauge, area);
    }

    fn draw_stats(&self, f: &mut Frame, area: Rect, run: &TrainingRun, reader: &LiveMetricsReader) {
        let latest = reader.latest();
        let loss = latest.map(|m| m.loss).unwrap_or(run.current_loss);

        let metrics = reader.all_metrics();
        let steps_per_sec = if metrics.len() >= 2 {
            let first = metrics.front().unwrap();
            let last = metrics.back().unwrap();
            let time_diff = (last.timestamp - first.timestamp).num_milliseconds() as f64 / 1000.0;
            if time_diff > 0.0 {
                (last.step - first.step) as f64 / time_diff
            } else {
                0.0
            }
        } else {
            0.0
        };

        let remaining_steps = run.config.max_steps.saturating_sub(latest.map(|m| m.step).unwrap_or(0));
        let eta_secs = if steps_per_sec > 0.0 { remaining_steps as f64 / steps_per_sec } else { 0.0 };

        let eta_str = format_duration(eta_secs);
        let elapsed = self.start_time.elapsed().as_secs_f64();
        let elapsed_str = format_duration(elapsed);

        let loss_trend = if metrics.len() >= 20 {
            let recent: f64 = metrics.iter().rev().take(10).map(|m| m.loss as f64).sum::<f64>() / 10.0;
            let older: f64 = metrics.iter().rev().skip(10).take(10).map(|m| m.loss as f64).sum::<f64>() / 10.0;
            if recent < older * 0.99 { "↓ improving" } else if recent > older * 1.01 { "↑ rising" } else { "─ stable" }
        } else {
            "─ gathering data"
        };

        let trend_color = if loss_trend.starts_with('↓') {
            colors::PREDICT
        } else if loss_trend.starts_with('↑') {
            colors::FAILED_RUN
        } else {
            Color::Gray
        };

        // Token metrics
        let tokens_trained = latest.map(|m| m.total_tokens_trained).unwrap_or(run.total_tokens_trained);
        let tokens_remaining = latest.map(|m| m.tokens_remaining).unwrap_or(0);
        let tokens_per_step = latest.map(|m| m.tokens_this_step).unwrap_or(run.tokens_per_step);
        let tokens_per_sec = run.tokens_per_second.unwrap_or_else(|| {
            if steps_per_sec > 0.0 { steps_per_sec * tokens_per_step as f64 } else { 0.0 }
        });

        // Gradient norm and learning rate
        let grad_norm = latest.map(|m| m.gradient_norm).unwrap_or(0.0);
        let confidence = latest.map(|m| m.confidence).unwrap_or(0.0);
        let learning_rate = latest.map(|m| m.learning_rate).unwrap_or(run.config.learning_rate);

        // Plateau detection - if loss hasn't improved by >1% in last 500 steps, show warning
        let plateau_warning = if metrics.len() >= 500 {
            let recent_avg: f64 = metrics.iter().rev().take(100).map(|m| m.loss as f64).sum::<f64>() / 100.0;
            let older_avg: f64 = metrics.iter().rev().skip(400).take(100).map(|m| m.loss as f64).sum::<f64>() / 100.0;
            recent_avg >= older_avg * 0.99  // Less than 1% improvement
        } else {
            false
        };

        // Build stats with clear visual hierarchy
        let mut stats = vec![
            // Primary: Current loss with trend - most important
            Line::from(vec![
                Span::styled(" Loss: ", Style::default().fg(colors::LABEL_DIM)),
                Span::styled(format!("{:.4}", loss), Style::default().fg(colors::LOSS_LINE).add_modifier(Modifier::BOLD)),
                Span::styled(format!("  {}", loss_trend), Style::default().fg(trend_color).add_modifier(Modifier::BOLD)),
            ]),
            // Best loss reference
            Line::from(vec![
                Span::styled(" Best: ", Style::default().fg(colors::LABEL_DIM)),
                Span::styled(format!("{:.4}", run.best_loss), Style::default().fg(colors::PREDICT)),
                Span::styled(format!(" @ step {}", run.best_step), Style::default().fg(colors::LABEL_DIM)),
            ]),
            // Separator for visual grouping
            Line::from(Span::styled(" ────────────────────────────────", Style::default().fg(colors::BORDER))),
            // Speed and ETA - important for monitoring
            Line::from(vec![
                Span::styled(" Speed: ", Style::default().fg(colors::LABEL_DIM)),
                Span::styled(format!("{:.2} steps/s", steps_per_sec), Style::default().fg(colors::LABEL_BRIGHT)),
                Span::styled("  │  ETA: ", Style::default().fg(colors::LABEL_DIM)),
                Span::styled(eta_str, Style::default().fg(colors::WARMUP).add_modifier(Modifier::BOLD)),
            ]),
            Line::from(vec![
                Span::styled(" Elapsed: ", Style::default().fg(colors::LABEL_DIM)),
                Span::styled(elapsed_str, Style::default().fg(colors::LABEL_BRIGHT)),
                Span::styled("  │  Backward: ", Style::default().fg(colors::LABEL_DIM)),
                Span::styled(
                    format!("{:.1}% reduction", run.backward_reduction()),
                    Style::default().fg(colors::PREDICT),
                ),
            ]),
            // Token metrics
            Line::from(vec![
                Span::styled(" Tokens: ", Style::default().fg(colors::LABEL_DIM)),
                Span::styled(format_tokens(tokens_trained), Style::default().fg(colors::WARMUP)),
                Span::styled(" trained  │  ", Style::default().fg(colors::LABEL_DIM)),
                Span::styled(format_tokens(tokens_remaining), Style::default().fg(colors::LABEL_BRIGHT)),
                Span::styled(" remaining", Style::default().fg(colors::LABEL_DIM)),
            ]),
            Line::from(vec![
                Span::styled(" Throughput: ", Style::default().fg(colors::LABEL_DIM)),
                Span::styled(format!("{}/s", format_tokens(tokens_per_sec as u64)), Style::default().fg(colors::STEP_TIME)),
                Span::styled(format!("  │  {} tok/step", format_tokens(tokens_per_step)), Style::default().fg(colors::LABEL_DIM)),
            ]),
            // Gradient and confidence
            Line::from(vec![
                Span::styled(" Grad Norm: ", Style::default().fg(colors::LABEL_DIM)),
                Span::styled(format!("{:.4}", grad_norm), Style::default().fg(colors::GRAD_NORM)),
                Span::styled("  │  Confidence: ", Style::default().fg(colors::LABEL_DIM)),
                Span::styled(format!("{:.1}%", confidence * 100.0), Style::default().fg(colors::PREDICTION)),
            ]),
        ];

        // Learning Rate - PROMINENT display
        stats.push(Line::from(Span::styled(" ────────────────────────────────", Style::default().fg(colors::BORDER))));
        stats.push(Line::from(vec![
            Span::styled(" Learning Rate: ", Style::default().fg(colors::LABEL_BRIGHT)),
            Span::styled(
                format!("{:.2e}", learning_rate),
                Style::default().fg(colors::LR_NORMAL).add_modifier(Modifier::BOLD)
            ),
        ]));

        // PLATEAU WARNING - MAXIMUM VISIBILITY with blinking effect via REVERSED + BOLD
        if plateau_warning {
            stats.push(Line::from(Span::raw("")));
            stats.push(Line::from(vec![
                Span::styled(
                    " !! PLATEAU DETECTED !! ",
                    Style::default()
                        .fg(Color::Black)
                        .bg(colors::SIGNAL_PLATEAU)
                        .add_modifier(Modifier::BOLD | Modifier::SLOW_BLINK)
                ),
            ]));
            stats.push(Line::from(vec![
                Span::styled(
                    " Loss not improving - Consider: ",
                    Style::default().fg(colors::SIGNAL_PLATEAU).add_modifier(Modifier::BOLD)
                ),
            ]));
            stats.push(Line::from(vec![
                Span::styled("   - Increase LR by 20-50%", Style::default().fg(colors::LABEL_BRIGHT)),
            ]));
            stats.push(Line::from(vec![
                Span::styled("   - Check for gradient issues", Style::default().fg(colors::LABEL_BRIGHT)),
            ]));
            stats.push(Line::from(vec![
                Span::styled("   - Review data quality", Style::default().fg(colors::LABEL_BRIGHT)),
            ]));
        }

        let para = Paragraph::new(stats).block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(colors::BORDER))
                .title(" Statistics "),
        );

        f.render_widget(para, area);
    }

    /// Enhanced loss chart with multiple trend lines, Bollinger bands, velocity overlay, and signals.
    fn draw_enhanced_loss_chart(&self, f: &mut Frame, area: Rect, reader: &LiveMetricsReader) {
        let raw_data = reader.loss_data();

        if raw_data.is_empty() {
            let msg = Paragraph::new("Waiting for training data...")
                .style(Style::default().fg(Color::Gray))
                .block(Block::default().borders(Borders::ALL).title(" Enhanced Loss Chart "));
            f.render_widget(msg, area);
            return;
        }

        // Split area for main chart and signal indicators
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(8), Constraint::Length(3)])
            .split(area);

        // Extract loss values for analysis
        let loss_values: Vec<f64> = raw_data.iter().map(|(_, l)| *l).collect();
        let steps: Vec<f64> = raw_data.iter().map(|(s, _)| *s).collect();

        // Apply smoothing based on selected method
        let smoothed_values = match self.smoothing_method {
            SmoothingMethod::Raw => loss_values.clone(),
            SmoothingMethod::Ema => compute_ema(&loss_values, 10),
            SmoothingMethod::Sma => compute_sma(&loss_values, 10),
            SmoothingMethod::Bilateral => compute_bilateral_filter(&loss_values, 7, 3.0, 0.1),
        };

        // Combine steps with smoothed values for the main data
        let main_data: Vec<(f64, f64)> = steps.iter()
            .zip(smoothed_values.iter())
            .map(|(&s, &l)| (s, l))
            .collect();

        // Compute bounds
        let min_step = steps.first().copied().unwrap_or(0.0);
        let max_step = steps.last().copied().unwrap_or(1.0);

        // Find min/max across all data we'll display
        let mut all_values = smoothed_values.clone();

        // Add Bollinger bands data if enabled
        let (upper_band, lower_band) = if self.show_bollinger_bands && loss_values.len() >= 20 {
            let (u, l) = compute_bollinger_bands(&loss_values, 20, 2.0);
            all_values.extend(u.iter().copied());
            all_values.extend(l.iter().copied());
            (u, l)
        } else {
            (Vec::new(), Vec::new())
        };

        // Compute trend lines if enabled
        let ema_fast = if self.show_trend_lines { compute_ema(&loss_values, 10) } else { Vec::new() };
        let ema_slow = if self.show_trend_lines { compute_ema(&loss_values, 50) } else { Vec::new() };
        let linear_trend = if self.show_trend_lines { compute_linear_regression(&loss_values) } else { Vec::new() };

        if self.show_trend_lines {
            all_values.extend(ema_fast.iter().copied());
            all_values.extend(ema_slow.iter().copied());
            all_values.extend(linear_trend.iter().copied());
        }

        let min_loss = all_values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_loss = all_values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        let y_margin = (max_loss - min_loss).max(0.1) * 0.15;
        let y_min = (min_loss - y_margin).max(0.0);
        let y_max = max_loss + y_margin;

        // Build datasets
        let mut datasets: Vec<Dataset> = Vec::new();

        // Main loss line (smoothed or raw)
        datasets.push(
            Dataset::default()
                .name(format!("Loss ({})", self.smoothing_method.title()))
                .marker(symbols::Marker::Braille)
                .graph_type(GraphType::Line)
                .style(Style::default().fg(colors::LOSS_LINE))
                .data(&main_data),
        );

        // Bollinger bands
        let upper_band_data: Vec<(f64, f64)>;
        let lower_band_data: Vec<(f64, f64)>;
        if self.show_bollinger_bands && !upper_band.is_empty() {
            upper_band_data = steps.iter().zip(upper_band.iter()).map(|(&s, &l)| (s, l)).collect();
            lower_band_data = steps.iter().zip(lower_band.iter()).map(|(&s, &l)| (s, l)).collect();

            datasets.push(
                Dataset::default()
                    .name("BB Upper")
                    .marker(symbols::Marker::Braille)
                    .graph_type(GraphType::Line)
                    .style(Style::default().fg(colors::BOLLINGER_BAND))
                    .data(&upper_band_data),
            );
            datasets.push(
                Dataset::default()
                    .name("BB Lower")
                    .marker(symbols::Marker::Braille)
                    .graph_type(GraphType::Line)
                    .style(Style::default().fg(colors::BOLLINGER_BAND))
                    .data(&lower_band_data),
            );
        }

        // Trend lines
        let ema_fast_data: Vec<(f64, f64)>;
        let ema_slow_data: Vec<(f64, f64)>;
        let linear_data: Vec<(f64, f64)>;
        if self.show_trend_lines && !ema_fast.is_empty() {
            ema_fast_data = steps.iter().zip(ema_fast.iter()).map(|(&s, &l)| (s, l)).collect();
            ema_slow_data = steps.iter().zip(ema_slow.iter()).map(|(&s, &l)| (s, l)).collect();
            linear_data = steps.iter().zip(linear_trend.iter()).map(|(&s, &l)| (s, l)).collect();

            datasets.push(
                Dataset::default()
                    .name("EMA-10")
                    .marker(symbols::Marker::Braille)
                    .graph_type(GraphType::Line)
                    .style(Style::default().fg(colors::EMA_FAST))
                    .data(&ema_fast_data),
            );
            datasets.push(
                Dataset::default()
                    .name("EMA-50")
                    .marker(symbols::Marker::Braille)
                    .graph_type(GraphType::Line)
                    .style(Style::default().fg(colors::EMA_SLOW))
                    .data(&ema_slow_data),
            );
            datasets.push(
                Dataset::default()
                    .name("Linear")
                    .marker(symbols::Marker::Braille)
                    .graph_type(GraphType::Line)
                    .style(Style::default().fg(colors::LINEAR_TREND))
                    .data(&linear_data),
            );
        }

        // Velocity overlay (as additional markers on the chart)
        let velocity_data: Vec<(f64, f64)>;
        if self.show_velocity_overlay && loss_values.len() >= 3 {
            let velocity = compute_velocity(&loss_values);
            // Normalize velocity to fit on the chart
            let vel_max = velocity.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
            let scale = if vel_max > 0.0 { (y_max - y_min) * 0.3 / vel_max } else { 1.0 };
            let mid_y = (y_max + y_min) / 2.0;

            velocity_data = steps.iter()
                .zip(velocity.iter())
                .map(|(&s, &v)| (s, mid_y + v * scale))
                .collect();

            datasets.push(
                Dataset::default()
                    .name("Velocity")
                    .marker(symbols::Marker::Dot)
                    .graph_type(GraphType::Scatter)
                    .style(Style::default().fg(if velocity.last().copied().unwrap_or(0.0) < 0.0 {
                        colors::VELOCITY_POS  // Negative velocity = loss decreasing = good
                    } else {
                        colors::VELOCITY_NEG  // Positive velocity = loss increasing = bad
                    }))
                    .data(&velocity_data),
            );
        }

        // Build title with color-coded legend for clarity
        let mut title_spans: Vec<Span> = vec![
            Span::styled(" Loss ", Style::default().fg(colors::LOSS_LINE).add_modifier(Modifier::BOLD)),
            Span::styled(format!("({})", self.smoothing_method.title()), Style::default().fg(colors::LABEL_DIM)),
        ];
        if self.show_trend_lines {
            title_spans.push(Span::styled(" | ", Style::default().fg(colors::BORDER)));
            title_spans.push(Span::styled("EMA-10", Style::default().fg(colors::EMA_FAST)));
            title_spans.push(Span::styled(" ", Style::default()));
            title_spans.push(Span::styled("EMA-50", Style::default().fg(colors::EMA_SLOW)));
            title_spans.push(Span::styled(" ", Style::default()));
            title_spans.push(Span::styled("Linear", Style::default().fg(colors::LINEAR_TREND)));
        }
        if self.show_bollinger_bands {
            title_spans.push(Span::styled(" | ", Style::default().fg(colors::BORDER)));
            title_spans.push(Span::styled("BB", Style::default().fg(colors::BOLLINGER_BAND)));
        }
        if self.show_velocity_overlay {
            title_spans.push(Span::styled(" | ", Style::default().fg(colors::BORDER)));
            title_spans.push(Span::styled("Vel", Style::default().fg(colors::VELOCITY_POS)));
        }

        // Calculate step interval for better axis labels
        let step_range = max_step - min_step;
        let mid_step = min_step + step_range / 2.0;
        let y_range = y_max - y_min;
        let mid_y = y_min + y_range / 2.0;

        let chart = Chart::new(datasets)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(colors::BORDER_ACTIVE))
                    .title(Line::from(title_spans)),
            )
            .x_axis(
                Axis::default()
                    .title(Span::styled("Step", Style::default().fg(colors::AXIS_LABEL).add_modifier(Modifier::BOLD)))
                    .style(Style::default().fg(colors::AXIS_LABEL))
                    .bounds([min_step, max_step])
                    .labels(vec![
                        Span::styled(format!("{:.0}", min_step), Style::default().fg(colors::AXIS_LABEL)),
                        Span::styled(format!("{:.0}", mid_step), Style::default().fg(colors::LABEL_DIM)),
                        Span::styled(format!("{:.0}", max_step), Style::default().fg(colors::AXIS_LABEL)),
                    ]),
            )
            .y_axis(
                Axis::default()
                    .title(Span::styled("Loss", Style::default().fg(colors::AXIS_LABEL).add_modifier(Modifier::BOLD)))
                    .style(Style::default().fg(colors::AXIS_LABEL))
                    .bounds([y_min, y_max])
                    .labels(vec![
                        Span::styled(format!("{:.4}", y_min), Style::default().fg(colors::AXIS_LABEL)),
                        Span::styled(format!("{:.4}", mid_y), Style::default().fg(colors::LABEL_DIM)),
                        Span::styled(format!("{:.4}", y_max), Style::default().fg(colors::AXIS_LABEL)),
                    ]),
            );

        f.render_widget(chart, chunks[0]);

        // Signal indicators panel - PROMINENT warnings
        let velocity = compute_velocity(&loss_values);
        let signals = detect_signals(&loss_values, &velocity);

        // Check for critical signals (plateau, spike)
        let has_plateau = signals.iter().any(|(_, s)| matches!(s, SignalType::Plateau));
        let has_spike = signals.iter().any(|(_, s)| matches!(s, SignalType::Spike));

        let mut signal_spans: Vec<Span> = vec![
            Span::styled(" Signals: ", Style::default().fg(colors::LABEL_BRIGHT).add_modifier(Modifier::BOLD))
        ];

        if signals.is_empty() {
            signal_spans.push(Span::styled(
                "OK - No issues detected",
                Style::default().fg(colors::SIGNAL_RECOVERY)
            ));
        } else {
            // Show most recent signals (up to 4) with enhanced styling
            for (i, (idx, signal)) in signals.iter().rev().take(4).enumerate() {
                if i > 0 {
                    signal_spans.push(Span::styled(" | ", Style::default().fg(colors::BORDER)));
                }

                // Make plateau and spike warnings VERY prominent
                let style = match signal {
                    SignalType::Plateau => Style::default()
                        .fg(Color::Black)
                        .bg(colors::SIGNAL_PLATEAU)
                        .add_modifier(Modifier::BOLD),
                    SignalType::Spike => Style::default()
                        .fg(Color::White)
                        .bg(colors::SIGNAL_SPIKE)
                        .add_modifier(Modifier::BOLD),
                    _ => Style::default().fg(signal.color()),
                };

                signal_spans.push(Span::styled(
                    format!(" {} {} @{} ", signal.emoji(), signal.description(), idx),
                    style,
                ));
            }
        }

        // Use attention-grabbing border for warnings
        let border_color = if has_spike {
            colors::SIGNAL_SPIKE
        } else if has_plateau {
            colors::SIGNAL_PLATEAU
        } else {
            colors::BORDER
        };

        let title_style = if has_spike || has_plateau {
            Style::default().fg(border_color).add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(colors::LABEL_BRIGHT)
        };

        let signals_para = Paragraph::new(Line::from(signal_spans))
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(border_color))
                    .title(Span::styled(" Training Signals ", title_style)),
            );
        f.render_widget(signals_para, chunks[1]);
    }

    #[allow(dead_code)]
    fn draw_loss_line_chart(&self, f: &mut Frame, area: Rect, reader: &LiveMetricsReader, title: &str) {
        let data = reader.loss_data();

        if data.is_empty() {
            let msg = Paragraph::new("Waiting for training data...")
                .style(Style::default().fg(Color::Gray))
                .block(Block::default().borders(Borders::ALL).title(format!(" {} ", title)));
            f.render_widget(msg, area);
            return;
        }

        let min_loss = data.iter().map(|(_, l)| *l).fold(f64::INFINITY, f64::min);
        let max_loss = data.iter().map(|(_, l)| *l).fold(f64::NEG_INFINITY, f64::max);
        let min_step = data.first().map(|(s, _)| *s).unwrap_or(0.0);
        let max_step = data.last().map(|(s, _)| *s).unwrap_or(1.0);

        let y_margin = (max_loss - min_loss).max(0.1) * 0.15;
        let y_min = (min_loss - y_margin).max(0.0);
        let y_max = max_loss + y_margin;

        let datasets = vec![Dataset::default()
            .name("Loss")
            .marker(symbols::Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::default().fg(colors::LOSS_LINE))
            .data(&data)];

        let chart = Chart::new(datasets)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(colors::BORDER))
                    .title(Span::styled(format!(" {} ", title), Style::default().fg(Color::White))),
            )
            .x_axis(
                Axis::default()
                    .style(Style::default().fg(Color::Gray))
                    .bounds([min_step, max_step])
                    .labels(vec![
                        Span::raw(format!("{:.0}", min_step)),
                        Span::raw(format!("{:.0}", max_step)),
                    ]),
            )
            .y_axis(
                Axis::default()
                    .style(Style::default().fg(Color::Gray))
                    .bounds([y_min, y_max])
                    .labels(vec![
                        Span::raw(format!("{:.2}", y_min)),
                        Span::raw(format!("{:.2}", y_max)),
                    ]),
            );

        f.render_widget(chart, area);
    }

    fn draw_scatter_with_trend(&self, f: &mut Frame, area: Rect, reader: &LiveMetricsReader) {
        let scatter_data = reader.loss_data();
        let trend_data = reader.smoothed_loss(0.1);

        if scatter_data.is_empty() {
            let msg = Paragraph::new("Waiting for training data...")
                .style(Style::default().fg(Color::Gray))
                .block(Block::default().borders(Borders::ALL).title(" Loss (Scatter + Trend) "));
            f.render_widget(msg, area);
            return;
        }

        let min_loss = scatter_data.iter().map(|(_, l)| *l).fold(f64::INFINITY, f64::min);
        let max_loss = scatter_data.iter().map(|(_, l)| *l).fold(f64::NEG_INFINITY, f64::max);
        let min_step = scatter_data.first().map(|(s, _)| *s).unwrap_or(0.0);
        let max_step = scatter_data.last().map(|(s, _)| *s).unwrap_or(1.0);

        let y_margin = (max_loss - min_loss).max(0.1) * 0.15;
        let y_min = (min_loss - y_margin).max(0.0);
        let y_max = max_loss + y_margin;

        let datasets = vec![
            Dataset::default()
                .name("Scatter")
                .marker(symbols::Marker::Dot)
                .graph_type(GraphType::Scatter)
                .style(Style::default().fg(colors::LOSS_SCATTER))
                .data(&scatter_data),
            Dataset::default()
                .name("Trend (EMA)")
                .marker(symbols::Marker::Braille)
                .graph_type(GraphType::Line)
                .style(Style::default().fg(colors::TREND_LINE))
                .data(&trend_data),
        ];

        let chart = Chart::new(datasets)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(colors::BORDER))
                    .title(" Loss (Scatter + EMA Trend) "),
            )
            .x_axis(
                Axis::default()
                    .title("Step")
                    .style(Style::default().fg(Color::Gray))
                    .bounds([min_step, max_step])
                    .labels(vec![
                        Span::raw(format!("{:.0}", min_step)),
                        Span::raw(format!("{:.0}", max_step)),
                    ]),
            )
            .y_axis(
                Axis::default()
                    .title("Loss")
                    .style(Style::default().fg(Color::Gray))
                    .bounds([y_min, y_max])
                    .labels(vec![
                        Span::raw(format!("{:.2}", y_min)),
                        Span::raw(format!("{:.2}", y_max)),
                    ]),
            );

        f.render_widget(chart, area);
    }

    fn draw_charts_tab(&self, f: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),  // Chart type selector
                Constraint::Length(2),  // Enhancement controls status bar
                Constraint::Min(10),    // Main chart area
            ])
            .split(area);

        // Chart type selector
        let chart_titles: Vec<Span> = ChartType::all()
            .iter()
            .map(|t| {
                let style = if *t == self.chart_type {
                    Style::default().fg(colors::TAB_ACTIVE).add_modifier(Modifier::BOLD)
                } else {
                    Style::default().fg(colors::TAB_INACTIVE)
                };
                Span::styled(format!(" {} ", t.title()), style)
            })
            .collect();

        let selector = Paragraph::new(Line::from(chart_titles))
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(colors::BORDER))
                    .title(" Chart Type [</>] "),
            );

        f.render_widget(selector, chunks[0]);

        // Enhancement controls status bar
        let smoothing_color = if self.smoothing_method != SmoothingMethod::Raw {
            colors::TAB_ACTIVE
        } else {
            colors::TAB_INACTIVE
        };
        let trend_color = if self.show_trend_lines { colors::TAB_ACTIVE } else { colors::TAB_INACTIVE };
        let bollinger_color = if self.show_bollinger_bands { colors::TAB_ACTIVE } else { colors::TAB_INACTIVE };
        let velocity_color = if self.show_velocity_overlay { colors::TAB_ACTIVE } else { colors::TAB_INACTIVE };

        let controls = Line::from(vec![
            Span::styled(" [s]mooth: ", Style::default().fg(colors::HELP_KEY)),
            Span::styled(self.smoothing_method.title(), Style::default().fg(smoothing_color)),
            Span::raw("  "),
            Span::styled("[t]rends: ", Style::default().fg(colors::HELP_KEY)),
            Span::styled(if self.show_trend_lines { "ON" } else { "OFF" }, Style::default().fg(trend_color)),
            Span::raw("  "),
            Span::styled("[b]ollinger: ", Style::default().fg(colors::HELP_KEY)),
            Span::styled(if self.show_bollinger_bands { "ON" } else { "OFF" }, Style::default().fg(bollinger_color)),
            Span::raw("  "),
            Span::styled("[v]elocity: ", Style::default().fg(colors::HELP_KEY)),
            Span::styled(if self.show_velocity_overlay { "ON" } else { "OFF" }, Style::default().fg(velocity_color)),
        ]);

        let controls_para = Paragraph::new(controls)
            .style(Style::default().fg(Color::White));
        f.render_widget(controls_para, chunks[1]);

        let reader = self.selected_run_id.as_ref().and_then(|id| self.metrics_readers.get(id));

        if let Some(reader) = reader {
            match self.chart_type {
                ChartType::LossLine => self.draw_enhanced_loss_chart(f, chunks[2], reader),
                ChartType::LossScatter => self.draw_scatter_with_trend(f, chunks[2], reader),
                ChartType::GradientNorm => self.draw_gradient_chart(f, chunks[2], reader),
                ChartType::StepTime => self.draw_step_time_chart(f, chunks[2], reader),
                ChartType::PhaseBreakdown => self.draw_phase_breakdown(f, chunks[2], reader),
                ChartType::PredictionAccuracy => self.draw_prediction_chart(f, chunks[2], reader),
            }
        } else {
            let msg = Paragraph::new("Select a run to view charts")
                .style(Style::default().fg(Color::Gray))
                .block(Block::default().borders(Borders::ALL));
            f.render_widget(msg, chunks[2]);
        }
    }

    fn draw_gradient_chart(&self, f: &mut Frame, area: Rect, reader: &LiveMetricsReader) {
        let data = reader.gradient_data();

        if data.is_empty() {
            let msg = Paragraph::new("No gradient norm data available")
                .style(Style::default().fg(Color::Gray))
                .block(Block::default().borders(Borders::ALL).title(" Gradient Norm "));
            f.render_widget(msg, area);
            return;
        }

        let min_val = data.iter().map(|(_, v)| *v).fold(f64::INFINITY, f64::min);
        let max_val = data.iter().map(|(_, v)| *v).fold(f64::NEG_INFINITY, f64::max);
        let min_step = data.first().map(|(s, _)| *s).unwrap_or(0.0);
        let max_step = data.last().map(|(s, _)| *s).unwrap_or(1.0);

        let y_margin = (max_val - min_val).max(0.1) * 0.15;
        let y_min = (min_val - y_margin).max(0.0);
        let y_max = max_val + y_margin;

        let datasets = vec![Dataset::default()
            .name("Gradient Norm")
            .marker(symbols::Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::default().fg(colors::GRAD_NORM))
            .data(&data)];

        let chart = Chart::new(datasets)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(colors::BORDER))
                    .title(" Gradient Norm "),
            )
            .x_axis(
                Axis::default()
                    .style(Style::default().fg(Color::Gray))
                    .bounds([min_step, max_step]),
            )
            .y_axis(
                Axis::default()
                    .style(Style::default().fg(Color::Gray))
                    .bounds([y_min, y_max]),
            );

        f.render_widget(chart, area);
    }

    fn draw_step_time_chart(&self, f: &mut Frame, area: Rect, reader: &LiveMetricsReader) {
        let data = reader.step_time_data();

        if data.is_empty() {
            let msg = Paragraph::new("No step time data available")
                .style(Style::default().fg(Color::Gray))
                .block(Block::default().borders(Borders::ALL).title(" Step Time "));
            f.render_widget(msg, area);
            return;
        }

        let min_val = data.iter().map(|(_, v)| *v).fold(f64::INFINITY, f64::min);
        let max_val = data.iter().map(|(_, v)| *v).fold(f64::NEG_INFINITY, f64::max);
        let min_step = data.first().map(|(s, _)| *s).unwrap_or(0.0);
        let max_step = data.last().map(|(s, _)| *s).unwrap_or(1.0);

        let y_margin = (max_val - min_val).max(10.0) * 0.15;
        let y_max = max_val + y_margin;

        let datasets = vec![Dataset::default()
            .name("Step Time (ms)")
            .marker(symbols::Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::default().fg(colors::STEP_TIME))
            .data(&data)];

        let chart = Chart::new(datasets)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(colors::BORDER))
                    .title(" Step Time (ms) "),
            )
            .x_axis(
                Axis::default()
                    .style(Style::default().fg(Color::Gray))
                    .bounds([min_step, max_step]),
            )
            .y_axis(
                Axis::default()
                    .style(Style::default().fg(Color::Gray))
                    .bounds([0.0, y_max]),
            );

        f.render_widget(chart, area);
    }

    fn draw_phase_breakdown(&self, f: &mut Frame, area: Rect, reader: &LiveMetricsReader) {
        let stats = reader.phase_stats();

        if stats.is_empty() {
            let msg = Paragraph::new("No phase data available")
                .style(Style::default().fg(Color::Gray))
                .block(Block::default().borders(Borders::ALL).title(" Phase Breakdown "));
            f.render_widget(msg, area);
            return;
        }

        let mut lines = vec![
            Line::from(Span::styled(
                " Phase         │ Steps │ Avg Loss │ Avg Time",
                Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
            )),
            Line::from(Span::raw(" ──────────────┼───────┼──────────┼──────────")),
        ];

        for phase in [TrainingPhase::Warmup, TrainingPhase::Full, TrainingPhase::Predict, TrainingPhase::Correct] {
            if let Some((count, avg_loss, avg_time)) = stats.get(&phase) {
                let color = match phase {
                    TrainingPhase::Warmup => colors::WARMUP,
                    TrainingPhase::Full => colors::FULL,
                    TrainingPhase::Predict => colors::PREDICT,
                    TrainingPhase::Correct => colors::CORRECT,
                };

                lines.push(Line::from(vec![
                    Span::styled(format!(" {:12?}", phase), Style::default().fg(color)),
                    Span::raw(" │ "),
                    Span::styled(format!("{:5}", count), Style::default().fg(Color::White)),
                    Span::raw(" │ "),
                    Span::styled(format!("{:8.4}", avg_loss), Style::default().fg(colors::LOSS_LINE)),
                    Span::raw(" │ "),
                    Span::styled(format!("{:6.0}ms", avg_time), Style::default().fg(colors::STEP_TIME)),
                ]));
            }
        }

        let para = Paragraph::new(lines).block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(colors::BORDER))
                .title(" Phase Breakdown "),
        );

        f.render_widget(para, area);
    }

    fn draw_prediction_chart(&self, f: &mut Frame, area: Rect, reader: &LiveMetricsReader) {
        let data = reader.prediction_data();

        if data.is_empty() {
            let msg = Paragraph::new("No prediction data available (shown during PREDICT/CORRECT phases)")
                .style(Style::default().fg(Color::Gray))
                .block(Block::default().borders(Borders::ALL).title(" Prediction Accuracy "));
            f.render_widget(msg, area);
            return;
        }

        let min_step = data.first().map(|(s, _)| *s).unwrap_or(0.0);
        let max_step = data.last().map(|(s, _)| *s).unwrap_or(1.0);
        let max_err = data.iter().map(|(_, v)| *v).fold(0.0f64, f64::max);

        let datasets = vec![Dataset::default()
            .name("Prediction Error")
            .marker(symbols::Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::default().fg(colors::PREDICTION))
            .data(&data)];

        let chart = Chart::new(datasets)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(colors::BORDER))
                    .title(" Prediction Error "),
            )
            .x_axis(
                Axis::default()
                    .style(Style::default().fg(Color::Gray))
                    .bounds([min_step, max_step]),
            )
            .y_axis(
                Axis::default()
                    .style(Style::default().fg(Color::Gray))
                    .bounds([0.0, max_err * 1.1]),
            );

        f.render_widget(chart, area);
    }

    fn draw_gpu_tab(&self, f: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(3), Constraint::Min(10)])
            .split(area);

        // GPU view selector
        let view_titles: Vec<Span> = GpuView::all()
            .iter()
            .map(|v| {
                let style = if *v == self.gpu_view {
                    Style::default().fg(colors::TAB_ACTIVE).add_modifier(Modifier::BOLD)
                } else {
                    Style::default().fg(colors::TAB_INACTIVE)
                };
                Span::styled(format!(" {} ", v.title()), style)
            })
            .collect();

        let selector = Paragraph::new(Line::from(view_titles))
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(colors::BORDER))
                    .title(" GPU View [←/→] "),
            );

        f.render_widget(selector, chunks[0]);

        match self.gpu_view {
            GpuView::Summary => self.draw_gpu_summary_full(f, chunks[1]),
            GpuView::Memory => self.draw_gpu_memory_chart(f, chunks[1]),
            GpuView::Thermal => self.draw_gpu_thermal_chart(f, chunks[1]),
            GpuView::Utilization => self.draw_gpu_util_chart(f, chunks[1]),
            GpuView::FlameGraph => self.draw_gpu_flame_graph(f, chunks[1]),
        }
    }

    fn draw_gpu_summary(&self, f: &mut Frame, area: Rect) {
        let content = if let Some(stats) = self.gpu_monitor.current() {
            let mem_color = if stats.memory_percent() > 90.0 {
                colors::MEMORY_CRIT
            } else if stats.memory_percent() > 70.0 {
                colors::MEMORY_WARN
            } else {
                colors::MEMORY_OK
            };

            let temp_color = if stats.temperature > 80 {
                colors::TEMP_CRIT
            } else if stats.temperature > 70 {
                colors::TEMP_WARN
            } else {
                colors::TEMP_OK
            };

            vec![
                Line::from(vec![
                    Span::styled(" GPU: ", Style::default().fg(Color::Gray)),
                    Span::styled(
                        format!("{:.1}/{:.1}GB", stats.memory_used as f64 / 1e9, stats.memory_total as f64 / 1e9),
                        Style::default().fg(mem_color).add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(format!(" ({:.0}%)", stats.memory_percent()), Style::default().fg(mem_color)),
                    Span::styled("  │  ", Style::default().fg(colors::BORDER)),
                    Span::styled(format!("{}°C", stats.temperature), Style::default().fg(temp_color)),
                    Span::styled("  │  ", Style::default().fg(colors::BORDER)),
                    Span::styled(format!("{:.0}W", stats.power_draw), Style::default().fg(Color::White)),
                    Span::styled("  │  ", Style::default().fg(colors::BORDER)),
                    Span::styled(format!("{}%", stats.gpu_util), Style::default().fg(Color::White)),
                ]),
                Line::from(vec![
                    Span::styled(" Peak: ", Style::default().fg(Color::Gray)),
                    Span::styled(
                        format!("{:.1}GB", self.gpu_monitor.peak_memory() as f64 / 1e9),
                        Style::default().fg(mem_color),
                    ),
                    Span::styled("  │  ", Style::default().fg(colors::BORDER)),
                    Span::styled(format!("{}°C", self.gpu_monitor.peak_temperature()), Style::default().fg(temp_color)),
                    Span::styled("  │  ", Style::default().fg(colors::BORDER)),
                    Span::styled(format!("{:.0}W", self.gpu_monitor.peak_power()), Style::default().fg(Color::White)),
                    Span::styled("  │  ", Style::default().fg(colors::BORDER)),
                    Span::styled(stats.pstate.clone(), Style::default().fg(Color::Yellow)),
                ]),
            ]
        } else {
            vec![Line::from(Span::styled(" GPU stats unavailable", Style::default().fg(Color::Gray)))]
        };

        let para = Paragraph::new(content).block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(colors::BORDER))
                .title(" GPU "),
        );

        f.render_widget(para, area);
    }

    fn draw_gpu_summary_full(&self, f: &mut Frame, area: Rect) {
        let stats = self.gpu_monitor.current();

        let content = if let Some(stats) = stats {
            let mem_pct = stats.memory_percent();
            let mem_color = if mem_pct > 90.0 {
                colors::MEMORY_CRIT
            } else if mem_pct > 70.0 {
                colors::MEMORY_WARN
            } else {
                colors::MEMORY_OK
            };

            let temp_color = if stats.temperature > 80 {
                colors::TEMP_CRIT
            } else if stats.temperature > 70 {
                colors::TEMP_WARN
            } else {
                colors::TEMP_OK
            };

            vec![
                Line::from(vec![
                    Span::styled(" Device:    ", Style::default().fg(Color::Gray)),
                    Span::styled(&stats.name, Style::default().fg(Color::White)),
                ]),
                Line::from(vec![
                    Span::styled(" Driver:    ", Style::default().fg(Color::Gray)),
                    Span::styled(&stats.driver_version, Style::default().fg(Color::White)),
                ]),
                Line::from(Span::raw("")),
                Line::from(vec![
                    Span::styled(" Memory:    ", Style::default().fg(Color::Gray)),
                    Span::styled(
                        format!("{:.2} / {:.2} GB", stats.memory_used as f64 / 1e9, stats.memory_total as f64 / 1e9),
                        Style::default().fg(mem_color).add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(format!("  ({:.1}%)", mem_pct), Style::default().fg(mem_color)),
                ]),
                Line::from(vec![
                    Span::styled(" Peak Mem:  ", Style::default().fg(Color::Gray)),
                    Span::styled(
                        format!("{:.2} GB", self.gpu_monitor.peak_memory() as f64 / 1e9),
                        Style::default().fg(mem_color),
                    ),
                ]),
                Line::from(Span::raw("")),
                Line::from(vec![
                    Span::styled(" Temp:      ", Style::default().fg(Color::Gray)),
                    Span::styled(format!("{}°C", stats.temperature), Style::default().fg(temp_color).add_modifier(Modifier::BOLD)),
                    Span::styled(format!("  (peak {}°C)", self.gpu_monitor.peak_temperature()), Style::default().fg(Color::Gray)),
                ]),
                Line::from(vec![
                    Span::styled(" Power:     ", Style::default().fg(Color::Gray)),
                    Span::styled(format!("{:.0} / {:.0} W", stats.power_draw, stats.power_limit), Style::default().fg(Color::White)),
                    Span::styled(format!("  (peak {:.0}W)", self.gpu_monitor.peak_power()), Style::default().fg(Color::Gray)),
                ]),
                Line::from(Span::raw("")),
                Line::from(vec![
                    Span::styled(" GPU Util:  ", Style::default().fg(Color::Gray)),
                    Span::styled(format!("{}%", stats.gpu_util), Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
                ]),
                Line::from(vec![
                    Span::styled(" Mem Util:  ", Style::default().fg(Color::Gray)),
                    Span::styled(format!("{}%", stats.memory_util), Style::default().fg(Color::White)),
                ]),
                Line::from(vec![
                    Span::styled(" P-State:   ", Style::default().fg(Color::Gray)),
                    Span::styled(&stats.pstate, Style::default().fg(Color::Yellow)),
                ]),
                Line::from(vec![
                    Span::styled(" Clocks:    ", Style::default().fg(Color::Gray)),
                    Span::styled(format!("{} MHz (GPU)", stats.clock_graphics), Style::default().fg(Color::White)),
                    Span::styled(format!("  {} MHz (Mem)", stats.clock_memory), Style::default().fg(Color::Gray)),
                ]),
            ]
        } else {
            vec![Line::from(Span::styled(" GPU stats unavailable - nvidia-smi not found?", Style::default().fg(Color::Gray)))]
        };

        let para = Paragraph::new(content).block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(colors::BORDER))
                .title(" GPU Summary "),
        );

        f.render_widget(para, area);
    }

    fn draw_gpu_memory_chart(&self, f: &mut Frame, area: Rect) {
        if self.gpu_history.is_empty() {
            let msg = Paragraph::new("Collecting GPU memory data...")
                .style(Style::default().fg(Color::Gray))
                .block(Block::default().borders(Borders::ALL).title(" GPU Memory "));
            f.render_widget(msg, area);
            return;
        }

        let data: Vec<(f64, f64)> = self
            .gpu_history
            .iter()
            .enumerate()
            .map(|(i, s)| (i as f64, s.memory_mb / 1024.0))
            .collect();

        let _max_mem = data.iter().map(|(_, v)| *v).fold(0.0f64, f64::max);
        let total_mem = self.gpu_monitor.current().map(|s| s.memory_total as f64 / 1e9).unwrap_or(16.0);

        let datasets = vec![Dataset::default()
            .name("Memory (GB)")
            .marker(symbols::Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::default().fg(colors::MEMORY_OK))
            .data(&data)];

        let chart = Chart::new(datasets)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(colors::BORDER))
                    .title(" GPU Memory Usage "),
            )
            .x_axis(Axis::default().style(Style::default().fg(Color::Gray)).bounds([0.0, data.len() as f64]))
            .y_axis(
                Axis::default()
                    .style(Style::default().fg(Color::Gray))
                    .bounds([0.0, total_mem])
                    .labels(vec![
                        Span::raw("0"),
                        Span::raw(format!("{:.0}GB", total_mem / 2.0)),
                        Span::raw(format!("{:.0}GB", total_mem)),
                    ]),
            );

        f.render_widget(chart, area);
    }

    fn draw_gpu_thermal_chart(&self, f: &mut Frame, area: Rect) {
        if self.gpu_history.is_empty() {
            let msg = Paragraph::new("Collecting GPU thermal data...")
                .style(Style::default().fg(Color::Gray))
                .block(Block::default().borders(Borders::ALL).title(" GPU Temperature "));
            f.render_widget(msg, area);
            return;
        }

        let data: Vec<(f64, f64)> = self.gpu_history.iter().enumerate().map(|(i, s)| (i as f64, s.temp_c as f64)).collect();

        let _max_temp = data.iter().map(|(_, v)| *v).fold(0.0f64, f64::max);

        let datasets = vec![Dataset::default()
            .name("Temp (°C)")
            .marker(symbols::Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::default().fg(colors::TEMP_WARN))
            .data(&data)];

        let chart = Chart::new(datasets)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(colors::BORDER))
                    .title(" GPU Temperature "),
            )
            .x_axis(Axis::default().style(Style::default().fg(Color::Gray)).bounds([0.0, data.len() as f64]))
            .y_axis(
                Axis::default()
                    .style(Style::default().fg(Color::Gray))
                    .bounds([0.0, 100.0])
                    .labels(vec![Span::raw("0°C"), Span::raw("50°C"), Span::raw("100°C")]),
            );

        f.render_widget(chart, area);
    }

    fn draw_gpu_util_chart(&self, f: &mut Frame, area: Rect) {
        if self.gpu_history.is_empty() {
            let msg = Paragraph::new("Collecting GPU utilization data...")
                .style(Style::default().fg(Color::Gray))
                .block(Block::default().borders(Borders::ALL).title(" GPU Utilization "));
            f.render_widget(msg, area);
            return;
        }

        let data: Vec<(f64, f64)> = self.gpu_history.iter().enumerate().map(|(i, s)| (i as f64, s.util_percent)).collect();

        let datasets = vec![Dataset::default()
            .name("Util (%)")
            .marker(symbols::Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::default().fg(colors::PREDICT))
            .data(&data)];

        let chart = Chart::new(datasets)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(colors::BORDER))
                    .title(" GPU Utilization "),
            )
            .x_axis(Axis::default().style(Style::default().fg(Color::Gray)).bounds([0.0, data.len() as f64]))
            .y_axis(
                Axis::default()
                    .style(Style::default().fg(Color::Gray))
                    .bounds([0.0, 100.0])
                    .labels(vec![Span::raw("0%"), Span::raw("50%"), Span::raw("100%")]),
            );

        f.render_widget(chart, area);
    }

    fn draw_gpu_flame_graph(&self, f: &mut Frame, area: Rect) {
        // Flame graph style visualization showing GPU activity over time
        if self.gpu_history.len() < 10 {
            let msg = Paragraph::new("Collecting data for flame graph (need 10+ samples)...")
                .style(Style::default().fg(Color::Gray))
                .block(Block::default().borders(Borders::ALL).title(" GPU Flame Graph "));
            f.render_widget(msg, area);
            return;
        }

        // Create bar chart data from GPU history
        let bar_width = 3;
        let max_bars = (area.width as usize - 4) / (bar_width + 1);
        let step = (self.gpu_history.len() / max_bars).max(1);

        let data: Vec<(&str, u64)> = self
            .gpu_history
            .iter()
            .step_by(step)
            .take(max_bars)
            .enumerate()
            .map(|(_, s)| {
                // Empty label for flame bars
                ("", s.util_percent as u64)
            })
            .collect();

        let bar_chart = BarChart::default()
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(colors::BORDER))
                    .title(" GPU Activity Flame "),
            )
            .data(&data)
            .bar_width(bar_width as u16)
            .bar_gap(1)
            .bar_style(Style::default().fg(colors::FLAME_WARM))
            .value_style(Style::default().fg(Color::Black).bg(colors::FLAME_WARM))
            .max(100);

        f.render_widget(bar_chart, area);
    }

    fn draw_history_tab(&self, f: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(40), Constraint::Percentage(60)])
            .split(area);

        // Run list
        self.draw_run_list(f, chunks[0]);

        // Run details/history
        let selected_run = self.selected_run_id.as_ref().and_then(|id| self.run_manager.get_run(id));
        let reader = self.selected_run_id.as_ref().and_then(|id| self.metrics_readers.get(id));

        match (selected_run, reader) {
            (Some(run), Some(reader)) => {
                let detail_chunks = Layout::default()
                    .direction(Direction::Vertical)
                    .constraints([Constraint::Length(10), Constraint::Min(5)])
                    .split(chunks[1]);

                self.draw_run_history_details(f, detail_chunks[0], run, reader);
                self.draw_loss_line_chart(f, detail_chunks[1], reader, "Loss History");
            }
            _ => {
                let msg = Paragraph::new("Select a run to view history")
                    .style(Style::default().fg(Color::Gray))
                    .block(Block::default().borders(Borders::ALL).title(" Run History "));
                f.render_widget(msg, chunks[1]);
            }
        }
    }

    fn draw_run_history_details(&self, f: &mut Frame, area: Rect, run: &TrainingRun, reader: &LiveMetricsReader) {
        let metrics = reader.all_metrics();
        let _first = metrics.front();
        let _last = metrics.back();

        let start_time = run.started_at.format("%Y-%m-%d %H:%M:%S").to_string();
        let end_time = run.ended_at.map(|t| t.format("%H:%M:%S").to_string()).unwrap_or_else(|| "running...".to_string());

        let duration = if let Some(end) = run.ended_at {
            let dur = (end - run.started_at).num_seconds();
            format_duration(dur as f64)
        } else {
            let dur = (Utc::now() - run.started_at).num_seconds();
            format_duration(dur as f64)
        };

        let content = vec![
            Line::from(vec![
                Span::styled(" Run ID:     ", Style::default().fg(Color::Gray)),
                Span::styled(&run.run_id, Style::default().fg(Color::White)),
            ]),
            Line::from(vec![
                Span::styled(" Status:     ", Style::default().fg(Color::Gray)),
                Span::styled(
                    format!("{:?}", run.status),
                    Style::default().fg(match run.status {
                        TrainingStatus::Running => colors::ACTIVE_RUN,
                        TrainingStatus::Completed => colors::COMPLETED_RUN,
                        TrainingStatus::Failed => colors::FAILED_RUN,
                        _ => Color::Gray,
                    }),
                ),
            ]),
            Line::from(vec![
                Span::styled(" Start:      ", Style::default().fg(Color::Gray)),
                Span::styled(start_time, Style::default().fg(Color::White)),
            ]),
            Line::from(vec![
                Span::styled(" End:        ", Style::default().fg(Color::Gray)),
                Span::styled(end_time, Style::default().fg(Color::White)),
            ]),
            Line::from(vec![
                Span::styled(" Duration:   ", Style::default().fg(Color::Gray)),
                Span::styled(duration, Style::default().fg(Color::White)),
            ]),
            Line::from(vec![
                Span::styled(" Data pts:   ", Style::default().fg(Color::Gray)),
                Span::styled(format!("{}", metrics.len()), Style::default().fg(Color::White)),
            ]),
            Line::from(vec![
                Span::styled(" Best Loss:  ", Style::default().fg(Color::Gray)),
                Span::styled(format!("{:.4} @ step {}", run.best_loss, run.best_step), Style::default().fg(colors::PREDICT)),
            ]),
        ];

        let para = Paragraph::new(content).block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(colors::BORDER))
                .title(" Run Details "),
        );

        f.render_widget(para, area);
    }

    fn draw_help_tab(&self, f: &mut Frame, area: Rect) {
        let help_text = vec![
            Line::from(Span::styled(
                " KEYBOARD SHORTCUTS",
                Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
            )),
            Line::from(Span::raw("")),
            Line::from(vec![
                Span::styled(" Navigation", Style::default().fg(colors::WARMUP).add_modifier(Modifier::BOLD)),
            ]),
            Line::from(vec![
                Span::styled("   Tab / Shift+Tab  ", Style::default().fg(colors::HELP_KEY)),
                Span::raw("Cycle through main tabs"),
            ]),
            Line::from(vec![
                Span::styled("   1-5              ", Style::default().fg(colors::HELP_KEY)),
                Span::raw("Jump to specific tab (Overview/Charts/GPU/History/Help)"),
            ]),
            Line::from(vec![
                Span::styled("   j/k or ↑/↓       ", Style::default().fg(colors::HELP_KEY)),
                Span::raw("Navigate between runs"),
            ]),
            Line::from(vec![
                Span::styled("   [/] or ←/→       ", Style::default().fg(colors::HELP_KEY)),
                Span::raw("Cycle chart types / GPU views within tabs"),
            ]),
            Line::from(Span::raw("")),
            Line::from(vec![
                Span::styled(" Mode Toggle", Style::default().fg(colors::WARMUP).add_modifier(Modifier::BOLD)),
            ]),
            Line::from(vec![
                Span::styled("   l                ", Style::default().fg(colors::HELP_KEY)),
                Span::raw("Toggle LIVE ↔ HISTORY mode"),
            ]),
            Line::from(Span::raw("")),
            Line::from(vec![
                Span::styled(" Actions", Style::default().fg(colors::WARMUP).add_modifier(Modifier::BOLD)),
            ]),
            Line::from(vec![
                Span::styled("   r                ", Style::default().fg(colors::HELP_KEY)),
                Span::raw("Refresh / rediscover runs"),
            ]),
            Line::from(vec![
                Span::styled("   c                ", Style::default().fg(colors::HELP_KEY)),
                Span::raw("Clear cached metrics for selected run"),
            ]),
            Line::from(vec![
                Span::styled("   ? or F1          ", Style::default().fg(colors::HELP_KEY)),
                Span::raw("Show help overlay"),
            ]),
            Line::from(vec![
                Span::styled("   q or Esc         ", Style::default().fg(colors::HELP_KEY)),
                Span::raw("Quit"),
            ]),
            Line::from(Span::raw("")),
            Line::from(vec![
                Span::styled(" Chart Types (Charts Tab)", Style::default().fg(colors::WARMUP).add_modifier(Modifier::BOLD)),
            ]),
            Line::from(vec![
                Span::styled("   Loss (Line)       ", Style::default().fg(colors::LOSS_LINE)),
                Span::raw("Standard line chart of loss over steps"),
            ]),
            Line::from(vec![
                Span::styled("   Loss (Scatter)    ", Style::default().fg(colors::LOSS_SCATTER)),
                Span::raw("Scatter plot with EMA trend overlay"),
            ]),
            Line::from(vec![
                Span::styled("   Gradient Norm     ", Style::default().fg(colors::GRAD_NORM)),
                Span::raw("Gradient magnitude over training"),
            ]),
            Line::from(vec![
                Span::styled("   Step Time         ", Style::default().fg(colors::STEP_TIME)),
                Span::raw("Time per training step (ms)"),
            ]),
            Line::from(vec![
                Span::styled("   Phase Breakdown   ", Style::default().fg(Color::White)),
                Span::raw("Statistics by training phase"),
            ]),
            Line::from(vec![
                Span::styled("   Prediction Acc    ", Style::default().fg(colors::PREDICTION)),
                Span::raw("Prediction error in hybrid training"),
            ]),
            Line::from(Span::raw("")),
            Line::from(vec![
                Span::styled(" Chart Enhancements (Charts Tab)", Style::default().fg(colors::WARMUP).add_modifier(Modifier::BOLD)),
            ]),
            Line::from(vec![
                Span::styled("   s                 ", Style::default().fg(colors::HELP_KEY)),
                Span::raw("Cycle smoothing: Raw -> EMA -> SMA -> Bilateral"),
            ]),
            Line::from(vec![
                Span::styled("   t                 ", Style::default().fg(colors::HELP_KEY)),
                Span::raw("Toggle trend lines (EMA-10/50, linear regression)"),
            ]),
            Line::from(vec![
                Span::styled("   b                 ", Style::default().fg(colors::HELP_KEY)),
                Span::raw("Toggle Bollinger Bands (2 std dev, 20-step)"),
            ]),
            Line::from(vec![
                Span::styled("   v                 ", Style::default().fg(colors::HELP_KEY)),
                Span::raw("Toggle velocity overlay (loss derivative)"),
            ]),
            Line::from(Span::raw("")),
            Line::from(vec![
                Span::styled(" Signal Indicators", Style::default().fg(colors::WARMUP).add_modifier(Modifier::BOLD)),
            ]),
            Line::from(vec![
                Span::styled("   !! (Red)          ", Style::default().fg(colors::SIGNAL_SPIKE)),
                Span::raw("Loss spike detected"),
            ]),
            Line::from(vec![
                Span::styled("   ~~ (Yellow)       ", Style::default().fg(colors::SIGNAL_PLATEAU)),
                Span::raw("Plateau warning (loss not improving)"),
            ]),
            Line::from(vec![
                Span::styled("   <> (Blue)         ", Style::default().fg(colors::SIGNAL_INFLECTION)),
                Span::raw("Inflection point (curvature change)"),
            ]),
            Line::from(vec![
                Span::styled("   OK (Green)        ", Style::default().fg(colors::SIGNAL_RECOVERY)),
                Span::raw("Recovery confirmed after issue"),
            ]),
            Line::from(Span::raw("")),
            Line::from(vec![
                Span::styled(" GPU Views (GPU Tab)", Style::default().fg(colors::WARMUP).add_modifier(Modifier::BOLD)),
            ]),
            Line::from(vec![
                Span::styled("   Summary           ", Style::default().fg(Color::White)),
                Span::raw("Full GPU info: device, clocks, power, etc."),
            ]),
            Line::from(vec![
                Span::styled("   Memory            ", Style::default().fg(colors::MEMORY_OK)),
                Span::raw("VRAM usage over time"),
            ]),
            Line::from(vec![
                Span::styled("   Thermal           ", Style::default().fg(colors::TEMP_WARN)),
                Span::raw("GPU temperature history"),
            ]),
            Line::from(vec![
                Span::styled("   Utilization       ", Style::default().fg(colors::PREDICT)),
                Span::raw("GPU compute utilization %"),
            ]),
            Line::from(vec![
                Span::styled("   Flame             ", Style::default().fg(colors::FLAME_WARM)),
                Span::raw("Activity flame graph visualization"),
            ]),
        ];

        let para = Paragraph::new(help_text)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(colors::BORDER))
                    .title(" Help & Keybindings "),
            )
            .wrap(Wrap { trim: false });

        f.render_widget(para, area);
    }

    fn draw_help_overlay(&self, f: &mut Frame) {
        let area = f.area();
        let popup_width = 65.min(area.width - 4);
        let popup_height = 24.min(area.height - 4);

        let popup_area = Rect {
            x: (area.width - popup_width) / 2,
            y: (area.height - popup_height) / 2,
            width: popup_width,
            height: popup_height,
        };

        // Clear background with semi-transparent effect
        let clear = Paragraph::new("").style(Style::default().bg(Color::Rgb(15, 15, 25)));
        f.render_widget(clear, popup_area);

        let help_text = vec![
            Line::from(Span::styled(
                " QUICK REFERENCE ",
                Style::default().fg(colors::TAB_ACTIVE).add_modifier(Modifier::BOLD)
            )),
            Line::from(Span::styled("─────────────────────────────────────────", Style::default().fg(colors::BORDER))),
            Line::from(Span::raw("")),
            // Navigation section
            Line::from(Span::styled(" Navigation", Style::default().fg(colors::WARMUP).add_modifier(Modifier::BOLD))),
            Line::from(vec![
                Span::styled("   Tab      ", Style::default().fg(colors::HELP_KEY).add_modifier(Modifier::BOLD)),
                Span::styled("Cycle through main tabs", Style::default().fg(colors::LABEL_BRIGHT)),
            ]),
            Line::from(vec![
                Span::styled("   1-6      ", Style::default().fg(colors::HELP_KEY).add_modifier(Modifier::BOLD)),
                Span::styled("Jump to specific tab", Style::default().fg(colors::LABEL_BRIGHT)),
            ]),
            Line::from(vec![
                Span::styled("   j/k ↑↓   ", Style::default().fg(colors::HELP_KEY).add_modifier(Modifier::BOLD)),
                Span::styled("Navigate between runs", Style::default().fg(colors::LABEL_BRIGHT)),
            ]),
            Line::from(vec![
                Span::styled("   [/] ←→   ", Style::default().fg(colors::HELP_KEY).add_modifier(Modifier::BOLD)),
                Span::styled("Cycle chart types / sub-views", Style::default().fg(colors::LABEL_BRIGHT)),
            ]),
            Line::from(Span::raw("")),
            // Actions section
            Line::from(Span::styled(" Actions", Style::default().fg(colors::WARMUP).add_modifier(Modifier::BOLD))),
            Line::from(vec![
                Span::styled("   l        ", Style::default().fg(colors::HELP_KEY).add_modifier(Modifier::BOLD)),
                Span::styled("Toggle ", Style::default().fg(colors::LABEL_BRIGHT)),
                Span::styled("LIVE", Style::default().fg(colors::LIVE_INDICATOR)),
                Span::styled(" / ", Style::default().fg(colors::LABEL_BRIGHT)),
                Span::styled("HISTORY", Style::default().fg(colors::COMPLETED_RUN)),
                Span::styled(" mode", Style::default().fg(colors::LABEL_BRIGHT)),
            ]),
            Line::from(vec![
                Span::styled("   r        ", Style::default().fg(colors::HELP_KEY).add_modifier(Modifier::BOLD)),
                Span::styled("Refresh / rediscover runs", Style::default().fg(colors::LABEL_BRIGHT)),
            ]),
            Line::from(vec![
                Span::styled("   c        ", Style::default().fg(colors::HELP_KEY).add_modifier(Modifier::BOLD)),
                Span::styled("Clear cached metrics", Style::default().fg(colors::LABEL_BRIGHT)),
            ]),
            Line::from(Span::raw("")),
            // Chart enhancements
            Line::from(Span::styled(" Chart Enhancements (Charts tab)", Style::default().fg(colors::WARMUP).add_modifier(Modifier::BOLD))),
            Line::from(vec![
                Span::styled("   s        ", Style::default().fg(colors::HELP_KEY).add_modifier(Modifier::BOLD)),
                Span::styled("Cycle smoothing methods", Style::default().fg(colors::LABEL_BRIGHT)),
            ]),
            Line::from(vec![
                Span::styled("   t/b/v    ", Style::default().fg(colors::HELP_KEY).add_modifier(Modifier::BOLD)),
                Span::styled("Toggle trends/Bollinger/velocity", Style::default().fg(colors::LABEL_BRIGHT)),
            ]),
            Line::from(Span::raw("")),
            Line::from(Span::styled("─────────────────────────────────────────", Style::default().fg(colors::BORDER))),
            Line::from(vec![
                Span::styled("   q/Esc    ", Style::default().fg(colors::HELP_KEY).add_modifier(Modifier::BOLD)),
                Span::styled("Quit  ", Style::default().fg(colors::LABEL_BRIGHT)),
                Span::styled("│  ", Style::default().fg(colors::BORDER)),
                Span::styled("?       ", Style::default().fg(colors::HELP_KEY).add_modifier(Modifier::BOLD)),
                Span::styled("Close this overlay", Style::default().fg(colors::LABEL_DIM)),
            ]),
        ];

        let popup = Paragraph::new(help_text).block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(colors::TAB_ACTIVE))
                .style(Style::default().bg(Color::Rgb(25, 25, 40)))
                .title(Span::styled(" ? Help ", Style::default().fg(colors::TAB_ACTIVE).add_modifier(Modifier::BOLD))),
        );

        f.render_widget(popup, popup_area);
    }

    fn draw_footer(&self, f: &mut Frame, area: Rect) {
        let mode_str = match self.view_mode {
            ViewMode::Live => "LIVE",
            ViewMode::History => "HIST",
        };
        let mode_color = match self.view_mode {
            ViewMode::Live => colors::LIVE_INDICATOR,
            ViewMode::History => colors::COMPLETED_RUN,
        };

        // Mode indicator with clear visual distinction
        let mode_indicator = match self.view_mode {
            ViewMode::Live => Span::styled(
                format!(" {} {} ", '\u{25CF}', mode_str), // Bullet point
                Style::default().fg(Color::Black).bg(mode_color).add_modifier(Modifier::BOLD)
            ),
            ViewMode::History => Span::styled(
                format!(" {} {} ", '\u{25C6}', mode_str), // Diamond
                Style::default().fg(Color::Black).bg(mode_color).add_modifier(Modifier::BOLD)
            ),
        };

        let footer = Paragraph::new(Line::from(vec![
            mode_indicator,
            Span::styled(" ", Style::default()),
            Span::styled("Tab", Style::default().fg(colors::HELP_KEY).add_modifier(Modifier::BOLD)),
            Span::styled(":tabs ", Style::default().fg(colors::LABEL_DIM)),
            Span::styled("│ ", Style::default().fg(colors::BORDER)),
            Span::styled("j/k", Style::default().fg(colors::HELP_KEY).add_modifier(Modifier::BOLD)),
            Span::styled(":runs ", Style::default().fg(colors::LABEL_DIM)),
            Span::styled("│ ", Style::default().fg(colors::BORDER)),
            Span::styled("[/]", Style::default().fg(colors::HELP_KEY).add_modifier(Modifier::BOLD)),
            Span::styled(":views ", Style::default().fg(colors::LABEL_DIM)),
            Span::styled("│ ", Style::default().fg(colors::BORDER)),
            Span::styled("l", Style::default().fg(colors::HELP_KEY).add_modifier(Modifier::BOLD)),
            Span::styled(":mode ", Style::default().fg(colors::LABEL_DIM)),
            Span::styled("│ ", Style::default().fg(colors::BORDER)),
            Span::styled("?", Style::default().fg(colors::HELP_KEY).add_modifier(Modifier::BOLD)),
            Span::styled(":help ", Style::default().fg(colors::LABEL_DIM)),
            Span::styled("│ ", Style::default().fg(colors::BORDER)),
            Span::styled("q", Style::default().fg(colors::HELP_KEY).add_modifier(Modifier::BOLD)),
            Span::styled(":quit", Style::default().fg(colors::LABEL_DIM)),
        ]))
        .style(Style::default().bg(colors::HEADER_BG));

        f.render_widget(footer, area);
    }

    // =========================================================================
    // 4D/5D DIMENSIONAL VISUALIZATION
    // =========================================================================

    /// Draw the Dimensions tab with 4D/5D layer visualization.
    fn draw_dimensions_tab(&self, f: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(3), Constraint::Min(10)])
            .split(area);

        // View selector bar
        self.draw_dimension_view_selector(f, chunks[0]);

        // Main visualization content
        let selected_run = self.selected_run_id.as_ref().and_then(|id| self.run_manager.get_run(id));
        let reader = self.selected_run_id.as_ref().and_then(|id| self.metrics_readers.get(id));

        match (selected_run, reader) {
            (Some(run), Some(reader)) => {
                match self.dimension_view {
                    DimensionView::Heatmap4D => self.draw_4d_heatmap(f, chunks[1], run, reader),
                    DimensionView::Slice5D => self.draw_5d_slice(f, chunks[1], run, reader),
                    DimensionView::Animation => self.draw_animation_view(f, chunks[1], run, reader),
                    DimensionView::Summary => self.draw_dimension_summary(f, chunks[1], run, reader),
                }
            }
            _ => {
                let msg = Paragraph::new(vec![
                    Line::from(""),
                    Line::from(Span::styled(
                        "  No run selected",
                        Style::default().fg(Color::Gray),
                    )),
                    Line::from(""),
                    Line::from(Span::styled(
                        "  Use j/k to navigate runs, or start a training run.",
                        Style::default().fg(Color::DarkGray),
                    )),
                ])
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .border_style(Style::default().fg(colors::BORDER))
                        .title(" 4D/5D Visualization "),
                );
                f.render_widget(msg, chunks[1]);
            }
        }
    }

    fn draw_dimension_view_selector(&self, f: &mut Frame, area: Rect) {
        let view_titles: Vec<Span> = DimensionView::all()
            .iter()
            .map(|v| {
                let style = if *v == self.dimension_view {
                    Style::default().fg(colors::TAB_ACTIVE).add_modifier(Modifier::BOLD)
                } else {
                    Style::default().fg(colors::TAB_INACTIVE)
                };
                Span::styled(format!(" {} ", v.title()), style)
            })
            .collect();

        let proj_display = Span::styled(
            format!("  |  Projection: {} ", self.dimension_projection.title()),
            Style::default().fg(colors::PREDICTION),
        );

        let mut all_spans = view_titles;
        all_spans.push(proj_display);

        let selector = Paragraph::new(Line::from(all_spans)).block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(colors::BORDER))
                .title(" View [</>/] | Projection [p] "),
        );

        f.render_widget(selector, area);
    }

    /// Draw 4D Layer x Time heatmap with gradient intensity and loss state.
    fn draw_4d_heatmap(&self, f: &mut Frame, area: Rect, run: &TrainingRun, reader: &LiveMetricsReader) {
        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Min(50), Constraint::Length(35)])
            .split(area);

        self.draw_4d_heatmap_grid(f, chunks[0], run, reader);
        self.draw_4d_heatmap_legend(f, chunks[1], reader);
    }

    fn draw_4d_heatmap_grid(&self, f: &mut Frame, area: Rect, run: &TrainingRun, reader: &LiveMetricsReader) {
        let metrics = reader.all_metrics();
        let num_layers = run.config.num_layers;

        let inner_width = (area.width as usize).saturating_sub(12);
        let inner_height = (area.height as usize).saturating_sub(4);

        let steps_to_show = inner_width.min(60);
        let layers_to_show = inner_height.min(num_layers).max(1);

        let recent_metrics: Vec<_> = metrics.iter().rev().take(steps_to_show).collect();
        let recent_metrics: Vec<_> = recent_metrics.into_iter().rev().collect();

        let mut lines: Vec<Line> = Vec::new();

        lines.push(Line::from(Span::styled(
            format!("4D Layer x Time Heatmap (gradient intensity, last {} steps)", recent_metrics.len()),
            Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
        )));

        lines.push(Line::from(Span::styled(
            "-".repeat(inner_width + 8),
            Style::default().fg(colors::BORDER),
        )));

        let step_labels: String = if recent_metrics.len() > 10 {
            let first_step = recent_metrics.first().map(|m| m.step).unwrap_or(0);
            let last_step = recent_metrics.last().map(|m| m.step).unwrap_or(0);
            format!("Layer   {:>6} {:>width$} {:>6}", first_step, "Time (steps)", last_step, width = steps_to_show.saturating_sub(14))
        } else {
            "Layer   <-- Time (steps) -->".to_string()
        };
        lines.push(Line::from(Span::styled(step_labels, Style::default().fg(Color::Gray))));

        let layer_step = if num_layers > layers_to_show { num_layers / layers_to_show } else { 1 };

        for display_idx in 0..layers_to_show {
            let layer_idx = if num_layers > layers_to_show {
                num_layers - 1 - (display_idx * layer_step)
            } else {
                num_layers - 1 - display_idx
            };

            let layer_name = format!("layer_{}", layer_idx);
            let mut row_spans: Vec<Span> = Vec::new();

            row_spans.push(Span::styled(format!("{:>5}  ", layer_idx), Style::default().fg(Color::Gray)));

            for metric in &recent_metrics {
                let grad_norm = metric
                    .layer_gradients
                    .as_ref()
                    .and_then(|g| g.get(&layer_name).or_else(|| g.get(&format!("transformer.layer.{}", layer_idx))))
                    .copied()
                    .unwrap_or(metric.gradient_norm / num_layers as f32);

                let cell_char = self.loss_to_char(metric.loss);
                let cell_color = self.gradient_to_color(grad_norm);
                row_spans.push(Span::styled(cell_char.to_string(), Style::default().fg(cell_color)));
            }

            let remaining = steps_to_show.saturating_sub(recent_metrics.len());
            if remaining > 0 {
                row_spans.push(Span::styled(" ".repeat(remaining), Style::default().fg(Color::DarkGray)));
            }

            lines.push(Line::from(row_spans));
        }

        if num_layers > layers_to_show {
            lines.push(Line::from(Span::styled(
                format!("  ...  ({} more layers)", num_layers - layers_to_show),
                Style::default().fg(Color::DarkGray),
            )));
        }

        lines.push(Line::from(Span::styled(
            "       ^ warmup        ^ spike        ^ stable",
            Style::default().fg(Color::Gray),
        )));

        let para = Paragraph::new(lines).block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(colors::BORDER))
                .title(" Layer x Time Heatmap "),
        );

        f.render_widget(para, area);
    }

    fn draw_4d_heatmap_legend(&self, f: &mut Frame, area: Rect, reader: &LiveMetricsReader) {
        let latest = reader.latest();
        let mut lines: Vec<Line> = Vec::new();

        lines.push(Line::from(Span::styled(" Legend", Style::default().fg(Color::White).add_modifier(Modifier::BOLD))));
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(" Character = Loss State:", Style::default().fg(Color::Gray))));
        lines.push(Line::from(vec![Span::raw("   "), Span::styled("X", Style::default().fg(colors::PREDICT)), Span::raw(" healthy  (loss < 2.0)")]));
        lines.push(Line::from(vec![Span::raw("   "), Span::styled("#", Style::default().fg(colors::PAUSED_RUN)), Span::raw(" elevated (2.0 - 5.0)")]));
        lines.push(Line::from(vec![Span::raw("   "), Span::styled("%", Style::default().fg(colors::FLAME_WARM)), Span::raw(" warning  (5.0 - 10.0)")]));
        lines.push(Line::from(vec![Span::raw("   "), Span::styled("@", Style::default().fg(colors::FAILED_RUN)), Span::raw(" critical (> 10.0)")]));
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(" Color = Gradient Norm:", Style::default().fg(Color::Gray))));
        lines.push(Line::from(vec![Span::raw("   "), Span::styled("*", Style::default().fg(Color::Rgb(50, 50, 200))), Span::raw(" vanishing (< 0.001)")]));
        lines.push(Line::from(vec![Span::raw("   "), Span::styled("*", Style::default().fg(Color::Rgb(100, 255, 100))), Span::raw(" healthy   (0.01-1.0)")]));
        lines.push(Line::from(vec![Span::raw("   "), Span::styled("*", Style::default().fg(Color::Rgb(255, 200, 50))), Span::raw(" elevated  (1.0-10.0)")]));
        lines.push(Line::from(vec![Span::raw("   "), Span::styled("*", Style::default().fg(Color::Rgb(255, 50, 50))), Span::raw(" exploding (> 10.0)")]));
        lines.push(Line::from(""));

        if let Some(m) = latest {
            lines.push(Line::from(Span::styled(" Current Step:", Style::default().fg(Color::White).add_modifier(Modifier::BOLD))));
            lines.push(Line::from(vec![Span::raw("   Step: "), Span::styled(format!("{}", m.step), Style::default().fg(colors::PREDICTION))]));
            lines.push(Line::from(vec![Span::raw("   Loss: "), Span::styled(format!("{:.4}", m.loss), Style::default().fg(colors::LOSS_LINE))]));
            lines.push(Line::from(vec![Span::raw("   Grad: "), Span::styled(format!("{:.4}", m.gradient_norm), Style::default().fg(self.gradient_to_color(m.gradient_norm)))]));
        }

        let para = Paragraph::new(lines).block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(colors::BORDER))
                .title(" Legend "),
        );

        f.render_widget(para, area);
    }

    fn draw_5d_slice(&self, f: &mut Frame, area: Rect, run: &TrainingRun, reader: &LiveMetricsReader) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(3), Constraint::Min(10)])
            .split(area);

        let proj_info = Paragraph::new(Line::from(vec![
            Span::styled(" Active Projection: ", Style::default().fg(Color::Gray)),
            Span::styled(self.dimension_projection.title(), Style::default().fg(colors::TAB_ACTIVE).add_modifier(Modifier::BOLD)),
            Span::raw(" | "),
            Span::styled(self.dimension_projection.description(), Style::default().fg(Color::White)),
            Span::raw(" | Press 'p' to cycle"),
        ]))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(colors::BORDER))
                .title(" 5D Projection Mode "),
        );
        f.render_widget(proj_info, chunks[0]);

        match self.dimension_projection {
            DimensionProjection::LayerTime => self.draw_4d_heatmap_grid(f, chunks[1], run, reader),
            DimensionProjection::LayerAttention => self.draw_layer_attention_projection(f, chunks[1], run, reader),
            DimensionProjection::TimeGradient => self.draw_time_gradient_projection(f, chunks[1], reader),
            DimensionProjection::LayerLoss => self.draw_layer_loss_projection(f, chunks[1], run, reader),
        }
    }

    fn draw_layer_attention_projection(&self, f: &mut Frame, area: Rect, run: &TrainingRun, reader: &LiveMetricsReader) {
        let metrics = reader.all_metrics();
        let num_layers = run.config.num_layers;
        let num_heads = run.config.num_heads.max(8);

        let inner_width = (area.width as usize).saturating_sub(12);
        let inner_height = (area.height as usize).saturating_sub(4);
        let heads_to_show = inner_width.min(num_heads).min(16);
        let layers_to_show = inner_height.min(num_layers).max(1);

        let mut lines: Vec<Line> = Vec::new();

        lines.push(Line::from(Span::styled(
            format!("Layer x Attention Head Activity ({} heads x {} layers)", heads_to_show, layers_to_show),
            Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
        )));
        lines.push(Line::from(Span::styled("-".repeat(inner_width + 8), Style::default().fg(colors::BORDER))));

        let header: String = format!("Layer   {}", (0..heads_to_show).map(|h| format!("{:>2}", h % 100)).collect::<Vec<_>>().join(""));
        lines.push(Line::from(Span::styled(header, Style::default().fg(Color::Gray))));

        let latest = metrics.back();

        for display_idx in 0..layers_to_show {
            let layer_idx = num_layers - 1 - (display_idx * (num_layers / layers_to_show).max(1));
            if layer_idx >= num_layers { continue; }

            let layer_name = format!("layer_{}", layer_idx);
            let base_grad = latest
                .and_then(|m| m.layer_gradients.as_ref())
                .and_then(|g| g.get(&layer_name))
                .copied()
                .unwrap_or(0.1);

            let mut row_spans: Vec<Span> = Vec::new();
            row_spans.push(Span::styled(format!("{:>5}  ", layer_idx), Style::default().fg(Color::Gray)));

            for head_idx in 0..heads_to_show {
                let seed = (layer_idx * 17 + head_idx * 31) as f32;
                let activity = ((seed.sin() + 1.0) / 2.0) * base_grad;
                let cell_char = if activity < 0.01 { '.' } else if activity < 0.1 { 'o' } else if activity < 0.5 { 'O' } else { '@' };
                let color = self.gradient_to_color(activity);
                row_spans.push(Span::styled(format!("{:>2}", cell_char), Style::default().fg(color)));
            }

            lines.push(Line::from(row_spans));
        }

        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "Note: Patterns simulated from gradient data",
            Style::default().fg(Color::DarkGray),
        )));

        let para = Paragraph::new(lines).block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(colors::BORDER))
                .title(" Layer x Attention "),
        );

        f.render_widget(para, area);
    }

    fn draw_time_gradient_projection(&self, f: &mut Frame, area: Rect, reader: &LiveMetricsReader) {
        let metrics = reader.all_metrics();
        let inner_width = (area.width as usize).saturating_sub(15);
        let inner_height = (area.height as usize).saturating_sub(5);

        let buckets = ["< 0.001", "0.001-0.01", "0.01-0.1", "0.1-1.0", "1.0-10", "> 10"];
        let num_buckets = buckets.len().min(inner_height);
        let steps_to_show = inner_width.min(80);

        let recent_metrics: Vec<_> = metrics.iter().rev().take(steps_to_show).collect();
        let recent_metrics: Vec<_> = recent_metrics.into_iter().rev().collect();

        let mut lines: Vec<Line> = Vec::new();

        lines.push(Line::from(Span::styled(
            format!("Time x Gradient Distribution (last {} steps)", recent_metrics.len()),
            Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
        )));
        lines.push(Line::from(Span::styled("-".repeat(inner_width + 12), Style::default().fg(colors::BORDER))));
        lines.push(Line::from(Span::styled("Grad Range  <-- Time (steps) -->", Style::default().fg(Color::Gray))));

        for (bucket_idx, bucket_label) in buckets.iter().enumerate().take(num_buckets) {
            let mut row_spans: Vec<Span> = Vec::new();
            row_spans.push(Span::styled(format!("{:>10} ", bucket_label), Style::default().fg(Color::Gray)));

            for metric in &recent_metrics {
                let grad = metric.gradient_norm;
                let in_bucket = match bucket_idx {
                    0 => grad < 0.001,
                    1 => (0.001..0.01).contains(&grad),
                    2 => (0.01..0.1).contains(&grad),
                    3 => (0.1..1.0).contains(&grad),
                    4 => (1.0..10.0).contains(&grad),
                    _ => grad >= 10.0,
                };
                let cell_char = if in_bucket { '#' } else { '.' };
                let color = if in_bucket { self.gradient_to_color(grad) } else { Color::Rgb(40, 40, 40) };
                row_spans.push(Span::styled(cell_char.to_string(), Style::default().fg(color)));
            }

            lines.push(Line::from(row_spans));
        }

        lines.push(Line::from(""));

        let grads: Vec<f32> = metrics.iter().map(|m| m.gradient_norm).collect();
        if !grads.is_empty() {
            let mean = grads.iter().sum::<f32>() / grads.len() as f32;
            let max = grads.iter().cloned().fold(0.0f32, f32::max);
            let min = grads.iter().cloned().fold(f32::MAX, f32::min);
            lines.push(Line::from(vec![
                Span::styled("  Stats: ", Style::default().fg(Color::White)),
                Span::raw("Min="),
                Span::styled(format!("{:.4}", min), Style::default().fg(self.gradient_to_color(min))),
                Span::raw("  Mean="),
                Span::styled(format!("{:.4}", mean), Style::default().fg(self.gradient_to_color(mean))),
                Span::raw("  Max="),
                Span::styled(format!("{:.4}", max), Style::default().fg(self.gradient_to_color(max))),
            ]));
        }

        let para = Paragraph::new(lines).block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(colors::BORDER))
                .title(" Time x Gradient Flow "),
        );

        f.render_widget(para, area);
    }

    fn draw_layer_loss_projection(&self, f: &mut Frame, area: Rect, run: &TrainingRun, reader: &LiveMetricsReader) {
        let metrics = reader.all_metrics();
        let num_layers = run.config.num_layers;
        let inner_width = (area.width as usize).saturating_sub(15);
        let inner_height = (area.height as usize).saturating_sub(5);
        let layers_to_show = inner_height.min(num_layers).max(1);

        let mut lines: Vec<Line> = Vec::new();

        lines.push(Line::from(Span::styled(
            "Layer x Loss Contribution (estimated from gradient magnitude)",
            Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
        )));
        lines.push(Line::from(Span::styled("-".repeat(inner_width + 12), Style::default().fg(colors::BORDER))));

        let latest = metrics.back();
        let layer_gradients = latest.and_then(|m| m.layer_gradients.as_ref());

        let header = format!("Layer      {:>width$}", "Low <--- Loss Contribution ---> High", width = inner_width.min(40));
        lines.push(Line::from(Span::styled(header, Style::default().fg(Color::Gray))));

        let mut layer_contributions: Vec<(usize, f32)> = Vec::new();
        let total_grad: f32 = if let Some(grads) = layer_gradients {
            for i in 0..num_layers {
                let layer_name = format!("layer_{}", i);
                let grad = grads.get(&layer_name)
                    .or_else(|| grads.get(&format!("transformer.layer.{}", i)))
                    .copied()
                    .unwrap_or(0.0);
                layer_contributions.push((i, grad));
            }
            layer_contributions.iter().map(|(_, g)| g).sum()
        } else { 1.0 };

        layer_contributions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        for (layer_idx, contribution) in layer_contributions.iter().take(layers_to_show) {
            let normalized = if total_grad > 0.0 { contribution / total_grad } else { 0.0 };
            let bar_width = inner_width.min(50);
            let filled = (normalized * bar_width as f32).round() as usize;
            let empty = bar_width.saturating_sub(filled);

            let bar_color = if normalized > 0.2 { colors::FAILED_RUN }
                else if normalized > 0.1 { colors::FLAME_WARM }
                else if normalized > 0.05 { colors::PAUSED_RUN }
                else { colors::PREDICT };

            lines.push(Line::from(vec![
                Span::styled(format!("{:>5}     ", layer_idx), Style::default().fg(Color::Gray)),
                Span::styled("#".repeat(filled), Style::default().fg(bar_color)),
                Span::styled(".".repeat(empty), Style::default().fg(Color::Rgb(40, 40, 40))),
                Span::styled(format!(" {:.1}%", normalized * 100.0), Style::default().fg(Color::White)),
            ]));
        }

        if num_layers > layers_to_show {
            lines.push(Line::from(Span::styled(
                format!("  ... {} more layers", num_layers - layers_to_show),
                Style::default().fg(Color::DarkGray),
            )));
        }

        let para = Paragraph::new(lines).block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(colors::BORDER))
                .title(" Layer x Loss Contribution "),
        );

        f.render_widget(para, area);
    }

    fn draw_animation_view(&self, f: &mut Frame, area: Rect, run: &TrainingRun, reader: &LiveMetricsReader) {
        let metrics = reader.all_metrics();
        let num_layers = run.config.num_layers;
        let inner_height = (area.height as usize).saturating_sub(6);
        let layers_to_show = inner_height.min(num_layers).max(1);

        let frame_idx = if self.animation_playing {
            (self.start_time.elapsed().as_millis() / 200) as usize % metrics.len().max(1)
        } else {
            self.animation_frame.min(metrics.len().saturating_sub(1))
        };

        let current_metric = metrics.iter().nth(frame_idx);
        let mut lines: Vec<Line> = Vec::new();

        let play_indicator = if self.animation_playing { "> Playing" } else { "|| Paused" };
        lines.push(Line::from(vec![
            Span::styled(format!(" Animation Frame: {}/{} ", frame_idx + 1, metrics.len()), Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
            Span::styled(format!(" [{}] ", play_indicator), Style::default().fg(if self.animation_playing { colors::ACTIVE_RUN } else { colors::PAUSED_RUN })),
            Span::styled(" Press SPACE to toggle", Style::default().fg(Color::Gray)),
        ]));

        lines.push(Line::from(Span::styled("-".repeat((area.width as usize).saturating_sub(4)), Style::default().fg(colors::BORDER))));

        if let Some(metric) = current_metric {
            lines.push(Line::from(vec![
                Span::styled("  Step: ", Style::default().fg(Color::Gray)),
                Span::styled(format!("{}", metric.step), Style::default().fg(colors::PREDICTION)),
                Span::styled("  Loss: ", Style::default().fg(Color::Gray)),
                Span::styled(format!("{:.4}", metric.loss), Style::default().fg(colors::LOSS_LINE)),
                Span::styled("  Grad: ", Style::default().fg(Color::Gray)),
                Span::styled(format!("{:.4}", metric.gradient_norm), Style::default().fg(self.gradient_to_color(metric.gradient_norm))),
            ]));

            lines.push(Line::from(""));

            if let Some(ref layer_grads) = metric.layer_gradients {
                lines.push(Line::from(Span::styled("  Layer Gradients at this step:", Style::default().fg(Color::White))));

                let mut sorted_layers: Vec<_> = layer_grads.iter().collect();
                sorted_layers.sort_by(|a, b| a.0.cmp(b.0));

                for (layer_name, grad) in sorted_layers.iter().take(layers_to_show) {
                    let bar_width: usize = 30;
                    let grad_val = **grad;
                    let intensity = ((grad_val as f64).log10() + 3.0).clamp(0.0, 5.0) / 5.0;
                    let filled = (intensity * bar_width as f64) as usize;
                    let empty = bar_width.saturating_sub(filled);
                    let color = self.gradient_to_color(grad_val);

                    lines.push(Line::from(vec![
                        Span::styled(format!("    {:>12} ", layer_name), Style::default().fg(Color::Gray)),
                        Span::styled("#".repeat(filled), Style::default().fg(color)),
                        Span::styled(".".repeat(empty), Style::default().fg(Color::Rgb(40, 40, 40))),
                        Span::styled(format!(" {:.4}", grad), Style::default().fg(color)),
                    ]));
                }

                if layer_grads.len() > layers_to_show {
                    lines.push(Line::from(Span::styled(format!("    ... {} more layers", layer_grads.len() - layers_to_show), Style::default().fg(Color::DarkGray))));
                }
            } else {
                lines.push(Line::from(Span::styled("  Layer gradient data not available for this step.", Style::default().fg(Color::DarkGray))));
            }
        } else {
            lines.push(Line::from(Span::styled("  No data available for animation.", Style::default().fg(Color::DarkGray))));
        }

        lines.push(Line::from(""));
        let timeline_width = (area.width as usize).saturating_sub(10);
        let progress = if metrics.is_empty() { 0.0 } else { frame_idx as f64 / metrics.len() as f64 };
        let progress_pos = (progress * timeline_width as f64) as usize;

        let mut timeline_spans = vec![Span::styled("  [", Style::default().fg(colors::BORDER))];
        for i in 0..timeline_width {
            let char_style = if i == progress_pos { Style::default().fg(colors::TAB_ACTIVE).add_modifier(Modifier::BOLD) }
                else if i < progress_pos { Style::default().fg(colors::PREDICTION) }
                else { Style::default().fg(Color::Rgb(60, 60, 60)) };
            let ch = if i == progress_pos { "V" } else { "-" };
            timeline_spans.push(Span::styled(ch, char_style));
        }
        timeline_spans.push(Span::styled("]", Style::default().fg(colors::BORDER)));
        lines.push(Line::from(timeline_spans));

        let para = Paragraph::new(lines).block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(colors::BORDER))
                .title(" Temporal Animation "),
        );

        f.render_widget(para, area);
    }

    fn draw_dimension_summary(&self, f: &mut Frame, area: Rect, run: &TrainingRun, reader: &LiveMetricsReader) {
        let metrics = reader.all_metrics();
        let num_layers = run.config.num_layers;
        let mut lines: Vec<Line> = Vec::new();

        lines.push(Line::from(Span::styled(" Dimensional Analysis Summary", Style::default().fg(Color::White).add_modifier(Modifier::BOLD))));
        lines.push(Line::from(""));

        lines.push(Line::from(Span::styled(" Model Configuration:", Style::default().fg(colors::TAB_ACTIVE))));
        lines.push(Line::from(vec![
            Span::raw("   Layers: "),
            Span::styled(format!("{}", num_layers), Style::default().fg(colors::PREDICTION)),
            Span::raw("  |  Hidden: "),
            Span::styled(format!("{}", run.config.hidden_size), Style::default().fg(colors::PREDICTION)),
            Span::raw("  |  Heads: "),
            Span::styled(format!("{}", run.config.num_heads), Style::default().fg(colors::PREDICTION)),
        ]));
        lines.push(Line::from(""));

        lines.push(Line::from(Span::styled(" Gradient Statistics:", Style::default().fg(colors::TAB_ACTIVE))));

        let grads: Vec<f32> = metrics.iter().map(|m| m.gradient_norm).collect();
        if !grads.is_empty() {
            let mean = grads.iter().sum::<f32>() / grads.len() as f32;
            let max = grads.iter().cloned().fold(0.0f32, f32::max);
            let min = grads.iter().cloned().fold(f32::MAX, f32::min);
            let variance = grads.iter().map(|g| (g - mean).powi(2)).sum::<f32>() / grads.len() as f32;
            let std_dev = variance.sqrt();

            lines.push(Line::from(vec![
                Span::raw("   Mean: "),
                Span::styled(format!("{:.6}", mean), Style::default().fg(self.gradient_to_color(mean))),
                Span::raw("  |  Std: "),
                Span::styled(format!("{:.6}", std_dev), Style::default().fg(Color::White)),
            ]));
            lines.push(Line::from(vec![
                Span::raw("   Min:  "),
                Span::styled(format!("{:.6}", min), Style::default().fg(self.gradient_to_color(min))),
                Span::raw("  |  Max: "),
                Span::styled(format!("{:.6}", max), Style::default().fg(self.gradient_to_color(max))),
            ]));

            let health = if max > 10.0 || min < 0.0001 { ("Critical - Gradient problems detected", colors::FAILED_RUN) }
                else if max > 1.0 || min < 0.001 { ("Warning - Gradients outside optimal range", colors::PAUSED_RUN) }
                else { ("Healthy - Gradients in good range", colors::PREDICT) };
            lines.push(Line::from(vec![Span::raw("   Health: "), Span::styled(health.0, Style::default().fg(health.1))]));
        }

        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(" Loss Statistics:", Style::default().fg(colors::TAB_ACTIVE))));

        let losses: Vec<f32> = metrics.iter().map(|m| m.loss).collect();
        if losses.len() >= 2 {
            let first_loss = losses.first().copied().unwrap_or(0.0);
            let last_loss = losses.last().copied().unwrap_or(0.0);
            let reduction = if first_loss > 0.0 { (1.0 - last_loss / first_loss) * 100.0 } else { 0.0 };

            lines.push(Line::from(vec![
                Span::raw("   Initial: "),
                Span::styled(format!("{:.4}", first_loss), Style::default().fg(colors::LOSS_LINE)),
                Span::raw("  ->  Current: "),
                Span::styled(format!("{:.4}", last_loss), Style::default().fg(colors::LOSS_LINE)),
            ]));
            lines.push(Line::from(vec![
                Span::raw("   Reduction: "),
                Span::styled(format!("{:.1}%", reduction), Style::default().fg(if reduction > 0.0 { colors::PREDICT } else { colors::FAILED_RUN })),
            ]));
        }

        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(" Tip: Use [/] to switch views, 'p' to change projection mode", Style::default().fg(Color::DarkGray))));

        let para = Paragraph::new(lines).block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(colors::BORDER))
                .title(" Dimensional Summary "),
        );

        f.render_widget(para, area);
    }

    fn loss_to_char(&self, loss: f32) -> char {
        if loss < 2.0 { 'X' }
        else if loss < 5.0 { '#' }
        else if loss < 10.0 { '%' }
        else { '@' }
    }

    fn gradient_to_color(&self, grad_norm: f32) -> Color {
        if grad_norm < 0.001 { Color::Rgb(50, 50, 200) }
        else if grad_norm < 0.01 { Color::Rgb(50, 150, 200) }
        else if grad_norm < 0.1 { Color::Rgb(50, 200, 100) }
        else if grad_norm < 1.0 { Color::Rgb(100, 255, 100) }
        else if grad_norm < 10.0 { Color::Rgb(255, 200, 50) }
        else { Color::Rgb(255, 50, 50) }
    }
}

fn format_duration(secs: f64) -> String {
    if secs < 60.0 {
        format!("{:.0}s", secs)
    } else if secs < 3600.0 {
        let mins = secs / 60.0;
        let rem_secs = secs % 60.0;
        format!("{:.0}m {:.0}s", mins, rem_secs)
    } else {
        let hours = secs / 3600.0;
        let rem_mins = (secs % 3600.0) / 60.0;
        format!("{:.0}h {:.0}m", hours, rem_mins)
    }
}

/// Format token count with K/M/B suffixes.
fn format_tokens(tokens: u64) -> String {
    if tokens >= 1_000_000_000 {
        format!("{:.2}B", tokens as f64 / 1_000_000_000.0)
    } else if tokens >= 1_000_000 {
        format!("{:.2}M", tokens as f64 / 1_000_000.0)
    } else if tokens >= 1_000 {
        format!("{:.1}K", tokens as f64 / 1_000.0)
    } else {
        format!("{}", tokens)
    }
}
