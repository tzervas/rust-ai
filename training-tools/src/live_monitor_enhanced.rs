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
        Axis, BarChart, Block, Borders, Chart, Clear, Dataset, Gauge, GraphType, List, ListItem,
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

    // Chart colors
    pub const LOSS_LINE: Color = Color::Rgb(0, 255, 255); // Bright cyan
    pub const LOSS_SCATTER: Color = Color::Rgb(255, 255, 0); // Bright yellow dots
    pub const TREND_LINE: Color = Color::Rgb(255, 100, 255); // Magenta trend
    pub const GRAD_NORM: Color = Color::Rgb(100, 255, 100); // Green
    pub const STEP_TIME: Color = Color::Rgb(255, 180, 50); // Orange
    pub const PREDICTION: Color = Color::Rgb(100, 200, 255); // Light blue
    pub const LEARNING_RATE: Color = Color::Rgb(255, 150, 255); // Pink
    pub const THROUGHPUT: Color = Color::Rgb(150, 255, 150); // Light green
    pub const MEMORY: Color = Color::Rgb(200, 150, 255); // Purple
    pub const MOVING_AVG: Color = Color::Rgb(0, 200, 200); // Darker cyan

    // Phase colors
    pub const WARMUP: Color = Color::Rgb(255, 200, 0); // Orange
    pub const FULL: Color = Color::Rgb(0, 150, 255); // Blue
    pub const PREDICT: Color = Color::Rgb(0, 255, 100); // Green
    pub const CORRECT: Color = Color::Rgb(255, 100, 255); // Magenta

    // Status colors
    pub const ACTIVE_RUN: Color = Color::Rgb(0, 255, 0); // Bright green
    pub const COMPLETED_RUN: Color = Color::Rgb(100, 200, 255); // Light blue
    pub const FAILED_RUN: Color = Color::Rgb(255, 80, 80); // Red
    pub const PAUSED_RUN: Color = Color::Rgb(255, 255, 0); // Yellow

    // GPU colors
    pub const MEMORY_OK: Color = Color::Rgb(0, 200, 100);
    pub const MEMORY_WARN: Color = Color::Rgb(255, 200, 0);
    pub const MEMORY_CRIT: Color = Color::Rgb(255, 50, 50);
    pub const TEMP_OK: Color = Color::Rgb(100, 200, 100);
    pub const TEMP_WARN: Color = Color::Rgb(255, 200, 0);
    pub const TEMP_CRIT: Color = Color::Rgb(255, 50, 50);

    // UI colors
    pub const HEADER_BG: Color = Color::Rgb(30, 30, 50);
    pub const SELECTED_BG: Color = Color::Rgb(50, 50, 80);
    pub const TAB_ACTIVE: Color = Color::Rgb(0, 200, 255);
    pub const TAB_INACTIVE: Color = Color::Rgb(100, 100, 140);
    pub const BORDER: Color = Color::Rgb(80, 80, 120);
    pub const HELP_KEY: Color = Color::Rgb(255, 200, 0);
    pub const LIVE_INDICATOR: Color = Color::Rgb(255, 50, 50);

    // Flame graph colors
    pub const FLAME_HOT: Color = Color::Rgb(255, 80, 0);
    pub const FLAME_WARM: Color = Color::Rgb(255, 180, 0);
    pub const FLAME_COOL: Color = Color::Rgb(100, 200, 100);
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
    Dashboard, // NEW: Unified dashboard with all key metrics
    Overview,
    Charts,
    Analysis,
    Network,
    Concepts,
    GPU,
    History,
    Help,
}

impl MainTab {
    fn all() -> &'static [MainTab] {
        &[
            MainTab::Dashboard,
            MainTab::Overview,
            MainTab::Charts,
            MainTab::Analysis,
            MainTab::Network,
            MainTab::Concepts,
            MainTab::GPU,
            MainTab::History,
            MainTab::Help,
        ]
    }

    fn title(&self) -> &'static str {
        match self {
            MainTab::Dashboard => "Dashboard",
            MainTab::Overview => "Overview",
            MainTab::Charts => "Charts",
            MainTab::Analysis => "Analysis",
            MainTab::Network => "Network",
            MainTab::Concepts => "Concepts",
            MainTab::GPU => "GPU",
            MainTab::History => "History",
            MainTab::Help => "Help",
        }
    }

    fn index(&self) -> usize {
        match self {
            MainTab::Dashboard => 0,
            MainTab::Overview => 1,
            MainTab::Charts => 2,
            MainTab::Analysis => 3,
            MainTab::Network => 4,
            MainTab::Concepts => 5,
            MainTab::GPU => 6,
            MainTab::History => 7,
            MainTab::Help => 8,
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
    LearningRate,
    Throughput,
    MemoryUsage,
    LossVsTokens,
    PhaseDistribution,
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
            ChartType::LearningRate,
            ChartType::Throughput,
            ChartType::MemoryUsage,
            ChartType::LossVsTokens,
            ChartType::PhaseDistribution,
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
            ChartType::LearningRate => "Learning Rate",
            ChartType::Throughput => "Throughput (tok/s)",
            ChartType::MemoryUsage => "Memory Usage",
            ChartType::LossVsTokens => "Loss vs Tokens",
            ChartType::PhaseDistribution => "Phase Distribution",
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
        &[
            GpuView::Summary,
            GpuView::Memory,
            GpuView::Thermal,
            GpuView::Utilization,
            GpuView::FlameGraph,
        ]
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

/// Training curve quality assessment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CurveQuality {
    Healthy,     // Normal descent
    TooSlow,     // Underfitting - loss not decreasing fast enough
    TooFast,     // Overfitting risk - loss dropping too quickly
    Oscillating, // Learning rate too high
    Plateau,     // Stuck, needs intervention
    Diverging,   // Training failing
}

impl CurveQuality {
    fn icon(&self) -> &'static str {
        match self {
            CurveQuality::Healthy => "✓",
            CurveQuality::TooSlow => "⚠",
            CurveQuality::TooFast => "⚡",
            CurveQuality::Oscillating => "~",
            CurveQuality::Plateau => "─",
            CurveQuality::Diverging => "✗",
        }
    }

    fn description(&self) -> &'static str {
        match self {
            CurveQuality::Healthy => "Healthy descent",
            CurveQuality::TooSlow => "Too slow - underfitting",
            CurveQuality::TooFast => "Too fast - overfitting risk",
            CurveQuality::Oscillating => "Oscillating - LR too high",
            CurveQuality::Plateau => "Plateau - needs intervention",
            CurveQuality::Diverging => "Diverging - training failing",
        }
    }

    fn color(&self) -> Color {
        match self {
            CurveQuality::Healthy => colors::PREDICT,
            CurveQuality::TooSlow => colors::MEMORY_WARN,
            CurveQuality::TooFast => colors::WARMUP,
            CurveQuality::Oscillating => colors::MEMORY_WARN,
            CurveQuality::Plateau => colors::MEMORY_WARN, // Yellow warning, not gray
            CurveQuality::Diverging => colors::FAILED_RUN,
        }
    }
}

/// Cached computed data to avoid recalculation every frame.
#[derive(Default)]
struct MetricsCache {
    // Data vectors (reused buffers)
    loss_data: Vec<(f64, f64)>,
    gradient_data: Vec<(f64, f64)>,
    step_time_data: Vec<(f64, f64)>,
    smoothed_loss: Vec<(f64, f64)>,

    // Statistics
    phase_stats: HashMap<TrainingPhase, (usize, f64, f64)>,

    // Cache validity
    last_metrics_count: usize,
}

impl MetricsCache {
    fn is_valid(&self, current_count: usize) -> bool {
        // Cache is valid if metrics count hasn't changed
        self.last_metrics_count == current_count
    }

    fn invalidate(&mut self) {
        self.last_metrics_count = 0;
    }
}

/// Live metrics reader that tails the metrics file.
pub struct LiveMetricsReader {
    file_path: PathBuf,
    file: Option<BufReader<File>>,
    last_position: u64,
    metrics: VecDeque<StepMetrics>,
    max_history: usize,

    // Cache for expensive computations
    cache: MetricsCache,
}

impl LiveMetricsReader {
    pub fn new(file_path: PathBuf, max_history: usize) -> Self {
        Self {
            file_path,
            file: None,
            last_position: 0,
            metrics: VecDeque::with_capacity(max_history),
            max_history,
            cache: MetricsCache::default(),
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

        // Invalidate cache if new metrics arrived
        if !new_metrics.is_empty() {
            self.cache.invalidate();
        }

        new_metrics
    }

    pub fn all_metrics(&self) -> &VecDeque<StepMetrics> {
        &self.metrics
    }

    pub fn loss_data(&mut self) -> &[(f64, f64)] {
        if !self.cache.is_valid(self.metrics.len()) {
            self.cache.loss_data.clear();
            self.cache
                .loss_data
                .extend(self.metrics.iter().map(|m| (m.step as f64, m.loss as f64)));
            self.cache.last_metrics_count = self.metrics.len();
        }
        &self.cache.loss_data
    }

    /// Get loss data with range limit.
    pub fn loss_data_ranged(&mut self, range: ChartRange) -> Vec<(f64, f64)> {
        let data = self.loss_data();
        self.apply_range_slice(data, range)
    }

    /// Apply a range slice to data
    fn apply_range_slice(&self, data: &[(f64, f64)], range: ChartRange) -> Vec<(f64, f64)> {
        match range {
            ChartRange::Full => data.to_vec(),
            ChartRange::Trailing(n) => {
                let start = data.len().saturating_sub(n);
                data[start..].to_vec()
            }
        }
    }

    pub fn gradient_data(&mut self) -> &[(f64, f64)] {
        if !self.cache.is_valid(self.metrics.len()) {
            self.cache.gradient_data.clear();
            self.cache.gradient_data.extend(
                self.metrics
                    .iter()
                    .map(|m| (m.step as f64, m.gradient_norm as f64)),
            );
            self.cache.last_metrics_count = self.metrics.len();
        }
        &self.cache.gradient_data
    }

    /// Get gradient data with range limit.
    pub fn gradient_data_ranged(&mut self, range: ChartRange) -> Vec<(f64, f64)> {
        let data = self.gradient_data();
        self.apply_range_slice(data, range)
    }

    pub fn step_time_data(&mut self) -> &[(f64, f64)] {
        if !self.cache.is_valid(self.metrics.len()) {
            self.cache.step_time_data.clear();
            self.cache.step_time_data.extend(
                self.metrics
                    .iter()
                    .map(|m| (m.step as f64, m.step_time_ms as f64)),
            );
            self.cache.last_metrics_count = self.metrics.len();
        }
        &self.cache.step_time_data
    }

    /// Get step time data with range limit.
    pub fn step_time_data_ranged(&mut self, range: ChartRange) -> Vec<(f64, f64)> {
        let data = self.step_time_data();
        self.apply_range_slice(data, range)
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

    pub fn learning_rate_data(&self) -> Vec<(f64, f64)> {
        self.metrics
            .iter()
            .filter(|m| m.learning_rate > 0.0)
            .map(|m| (m.step as f64, m.learning_rate as f64))
            .collect()
    }

    pub fn perplexity_data(&self) -> Vec<(f64, f64)> {
        self.metrics
            .iter()
            .map(|m| {
                let ppl = if m.perplexity > 0.0 {
                    m.perplexity as f64
                } else {
                    (m.loss as f64).exp() // Compute from loss if not stored
                };
                (m.step as f64, ppl.min(10000.0)) // Cap for display
            })
            .collect()
    }

    /// Simple Moving Average of loss.
    pub fn loss_sma(&self, window: usize) -> Vec<(f64, f64)> {
        if self.metrics.len() < window {
            return Vec::new();
        }
        let losses: Vec<f64> = self.metrics.iter().map(|m| m.loss as f64).collect();
        let mut result = Vec::new();
        for i in window..=losses.len() {
            let sum: f64 = losses[i - window..i].iter().sum();
            let avg = sum / window as f64;
            let step = self.metrics.get(i - 1).map(|m| m.step).unwrap_or(0);
            result.push((step as f64, avg));
        }
        result
    }

    /// Velocity (rate of change) of loss.
    pub fn loss_velocity(&self, window: usize) -> Vec<(f64, f64)> {
        if self.metrics.len() < window + 1 {
            return Vec::new();
        }
        let losses: Vec<f64> = self.metrics.iter().map(|m| m.loss as f64).collect();
        let mut result = Vec::new();
        for i in window..losses.len() {
            let prev_avg: f64 = losses[i - window..i].iter().sum::<f64>() / window as f64;
            let curr = losses[i];
            let velocity = curr - prev_avg; // Negative = improving
            let step = self.metrics.get(i).map(|m| m.step).unwrap_or(0);
            result.push((step as f64, velocity));
        }
        result
    }

    pub fn token_throughput_data(&self) -> Vec<(f64, f64)> {
        if self.metrics.len() < 2 {
            return Vec::new();
        }

        // Compute tokens per second for each step
        let mut result = Vec::new();
        let tokens_per_step = self
            .metrics
            .front()
            .map(|m| m.tokens_this_step)
            .unwrap_or(1024);

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
    pub fn smoothed_loss(&mut self, alpha: f64) -> &[(f64, f64)] {
        if !self.cache.is_valid(self.metrics.len()) {
            self.cache.smoothed_loss.clear();
            let mut ema: Option<f64> = None;

            for m in &self.metrics {
                let loss = m.loss as f64;
                ema = Some(match ema {
                    Some(prev) => alpha * loss + (1.0 - alpha) * prev,
                    None => loss,
                });
                self.cache.smoothed_loss.push((m.step as f64, ema.unwrap()));
            }
            self.cache.last_metrics_count = self.metrics.len();
        }
        &self.cache.smoothed_loss
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

    /// Get loss vs total tokens trained (for LossVsTokens chart).
    pub fn loss_vs_tokens_data(&self) -> Vec<(f64, f64)> {
        self.metrics
            .iter()
            .filter(|m| m.total_tokens_trained > 0)
            .map(|m| (m.total_tokens_trained as f64, m.loss as f64))
            .collect()
    }

    /// Calculate min/max/mean statistics for a dataset.
    pub fn stats(data: &[(f64, f64)]) -> (f64, f64, f64) {
        if data.is_empty() {
            return (0.0, 0.0, 0.0);
        }
        let values: Vec<f64> = data.iter().map(|(_, v)| *v).collect();
        let min = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        (min, max, mean)
    }
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

/// Concept node for the concept cloud visualization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConceptNode {
    Loss,
    Gradient,
    LearningRate,
    Phase,
    Perplexity,
    Tokens,
    Confidence,
    StepTime,
    Memory,
    Efficiency,
}

impl ConceptNode {
    fn all() -> &'static [ConceptNode] {
        &[
            ConceptNode::Loss,
            ConceptNode::Gradient,
            ConceptNode::LearningRate,
            ConceptNode::Phase,
            ConceptNode::Perplexity,
            ConceptNode::Tokens,
            ConceptNode::Confidence,
            ConceptNode::StepTime,
            ConceptNode::Memory,
            ConceptNode::Efficiency,
        ]
    }

    fn title(&self) -> &'static str {
        match self {
            ConceptNode::Loss => "Loss",
            ConceptNode::Gradient => "Gradient",
            ConceptNode::LearningRate => "LR",
            ConceptNode::Phase => "Phase",
            ConceptNode::Perplexity => "PPL",
            ConceptNode::Tokens => "Tokens",
            ConceptNode::Confidence => "Conf",
            ConceptNode::StepTime => "Time",
            ConceptNode::Memory => "Mem",
            ConceptNode::Efficiency => "Eff",
        }
    }

    fn description(&self) -> &'static str {
        match self {
            ConceptNode::Loss => "Cross-entropy loss between predicted and target tokens. Measures model's prediction errors - lower means better token prediction",
            ConceptNode::Gradient => "Gradient norm ||∇|| - magnitude of the learning signal. Too low (<0.01) = vanishing gradients, too high (>10) = exploding. Ideal: 0.1-5.0",
            ConceptNode::LearningRate => "Step size for weight updates. Higher = faster but unstable learning. WSD schedule: warmup → stable → decay",
            ConceptNode::Phase => "Hybrid training cycle: WARMUP (calibrate predictor) → FULL (compute all gradients) → PREDICT (skip computation) → CORRECT (verify predictions)",
            ConceptNode::Perplexity => "exp(loss) - interpretable as 'how many equally likely choices the model sees'. PPL 10 = model is ~choosing between 10 equally likely tokens. Good LLMs achieve PPL < 20",
            ConceptNode::Tokens => "Count of subword units processed. Training throughput measured in tokens/second. More tokens = more learning",
            ConceptNode::Confidence => "Predictor's certainty that gradient prediction is accurate. Above 85% → skip computation (PREDICT phase)",
            ConceptNode::StepTime => "Step duration - time per training iteration",
            ConceptNode::Memory => "GPU memory - VRAM utilization",
            ConceptNode::Efficiency => "Training efficiency - hardware utilization score",
        }
    }

    fn color(&self) -> Color {
        match self {
            ConceptNode::Loss => colors::LOSS_LINE,
            ConceptNode::Gradient => colors::GRAD_NORM,
            ConceptNode::LearningRate => colors::WARMUP,
            ConceptNode::Phase => colors::FULL,
            ConceptNode::Perplexity => colors::LOSS_SCATTER,
            ConceptNode::Tokens => colors::STEP_TIME,
            ConceptNode::Confidence => colors::PREDICTION,
            ConceptNode::StepTime => colors::STEP_TIME,
            ConceptNode::Memory => colors::MEMORY_OK,
            ConceptNode::Efficiency => colors::PREDICT,
        }
    }
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

    // Chart zoom/range
    chart_range: ChartRange,

    // Concept cloud state
    selected_concept: ConceptNode,

    // Timing
    refresh_ms: u64,
    last_gpu_sample: Instant,
    start_time: Instant,
}

/// Chart view range for zoom control.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChartRange {
    /// Show all data
    Full,
    /// Show last N steps
    Trailing(usize),
}

impl ChartRange {
    fn all() -> &'static [ChartRange] {
        &[
            ChartRange::Full,
            ChartRange::Trailing(100),
            ChartRange::Trailing(500),
            ChartRange::Trailing(1000),
            ChartRange::Trailing(2000),
        ]
    }

    fn label(&self) -> String {
        match self {
            ChartRange::Full => "Full".to_string(),
            ChartRange::Trailing(n) => format!("Last {}", n),
        }
    }

    fn next(&self) -> ChartRange {
        let all = Self::all();
        let idx = all.iter().position(|r| r == self).unwrap_or(0);
        all[(idx + 1) % all.len()]
    }

    fn prev(&self) -> ChartRange {
        let all = Self::all();
        let idx = all.iter().position(|r| r == self).unwrap_or(0);
        all[(idx + all.len() - 1) % all.len()]
    }
}

/// Assess training curve quality from recent loss history.
///
/// Analyzes:
/// - Loss slope (derivative) - too steep = overfitting, too flat = underfitting
/// - Loss variance - high variance = oscillating
/// - Gradient norm trend - vanishing = plateau, exploding = diverging
/// - Phase-appropriate thresholds (early training should have steeper loss drop)
fn assess_curve_quality(losses: &[f32], gradients: &[f32], step: u64) -> CurveQuality {
    if losses.len() < 10 || gradients.len() < 10 {
        return CurveQuality::Healthy; // Not enough data
    }

    // Calculate loss statistics over recent window
    let recent_losses = &losses[losses.len().saturating_sub(20)..];
    let loss_mean: f32 = recent_losses.iter().sum::<f32>() / recent_losses.len() as f32;
    let loss_variance: f32 = recent_losses
        .iter()
        .map(|&x| (x - loss_mean).powi(2))
        .sum::<f32>()
        / recent_losses.len() as f32;
    let loss_std = loss_variance.sqrt();

    // Calculate loss slope (linear regression over last 20 points)
    let recent_n = recent_losses.len() as f32;
    let x_mean = (recent_n - 1.0) / 2.0;
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for (i, &loss) in recent_losses.iter().enumerate() {
        let x_diff = i as f32 - x_mean;
        numerator += x_diff * (loss - loss_mean);
        denominator += x_diff * x_diff;
    }
    let loss_slope = if denominator > 0.0 {
        numerator / denominator
    } else {
        0.0
    };

    // Calculate gradient statistics
    let recent_grads = &gradients[gradients.len().saturating_sub(20)..];
    let grad_mean: f32 = recent_grads.iter().sum::<f32>() / recent_grads.len() as f32;

    // Check for divergence (loss increasing or gradients exploding)
    if loss_slope > 0.001 || grad_mean > 100.0 {
        return CurveQuality::Diverging;
    }

    // Check for plateau (loss not decreasing, gradients too small)
    if loss_slope.abs() < 0.0001 && grad_mean < 0.01 {
        return CurveQuality::Plateau;
    }

    // Check for oscillation (high variance relative to mean)
    let loss_cv = loss_std / loss_mean.max(1e-6); // Coefficient of variation
    if loss_cv > 0.3 {
        return CurveQuality::Oscillating;
    }

    // Phase-appropriate slope thresholds
    let early_phase = step < 1000;
    let expected_slope = if early_phase { -0.01 } else { -0.001 };

    // Check for too slow (slope not steep enough for phase)
    if loss_slope > expected_slope * 0.2 && loss_slope < 0.0 {
        return CurveQuality::TooSlow;
    }

    // Check for too fast (overfitting risk - very steep drop)
    if loss_slope < expected_slope * 5.0 {
        return CurveQuality::TooFast;
    }

    CurveQuality::Healthy
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
            main_tab: MainTab::Dashboard,
            chart_type: ChartType::LossLine,
            gpu_view: GpuView::Summary,
            show_help_overlay: false,
            should_quit: false,
            chart_range: ChartRange::Full,
            selected_concept: ConceptNode::Loss,
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

    fn main_loop(
        &mut self,
        terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    ) -> anyhow::Result<()> {
        while !self.should_quit {
            // Discover new runs
            self.run_manager.discover_runs()?;
            self.setup_metrics_readers();

            // Mark stale runs (no updates in 5 minutes) as cancelled
            let stale_threshold = chrono::Duration::minutes(5);
            self.run_manager.mark_stale_runs(stale_threshold);

            // Poll metrics in live mode
            if self.view_mode == ViewMode::Live {
                // Find the currently active (running) run
                let active_run_id = self
                    .run_manager
                    .active_runs()
                    .next()
                    .map(|r| r.run_id.clone());

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

            // Number keys for tabs (0 = Dashboard)
            KeyCode::Char('0') => self.main_tab = MainTab::Dashboard,
            KeyCode::Char('1') => self.main_tab = MainTab::Overview,
            KeyCode::Char('2') => self.main_tab = MainTab::Charts,
            KeyCode::Char('3') => self.main_tab = MainTab::Analysis,
            KeyCode::Char('4') => self.main_tab = MainTab::Network,
            KeyCode::Char('5') => self.main_tab = MainTab::Concepts,
            KeyCode::Char('6') => self.main_tab = MainTab::GPU,
            KeyCode::Char('7') => self.main_tab = MainTab::History,
            KeyCode::Char('8') => self.main_tab = MainTab::Help,

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

            // Chart type cycling (within Charts tab) and concept navigation
            KeyCode::Left | KeyCode::Char('[') => {
                if self.main_tab == MainTab::Charts {
                    let types = ChartType::all();
                    let idx = types
                        .iter()
                        .position(|t| *t == self.chart_type)
                        .unwrap_or(0);
                    self.chart_type = types[(idx + types.len() - 1) % types.len()];
                } else if self.main_tab == MainTab::GPU {
                    let views = GpuView::all();
                    let idx = views.iter().position(|v| *v == self.gpu_view).unwrap_or(0);
                    self.gpu_view = views[(idx + views.len() - 1) % views.len()];
                } else if self.main_tab == MainTab::Concepts {
                    let concepts = ConceptNode::all();
                    let idx = concepts
                        .iter()
                        .position(|c| *c == self.selected_concept)
                        .unwrap_or(0);
                    self.selected_concept = concepts[(idx + concepts.len() - 1) % concepts.len()];
                }
            }
            KeyCode::Right | KeyCode::Char(']') => {
                if self.main_tab == MainTab::Charts {
                    let types = ChartType::all();
                    let idx = types
                        .iter()
                        .position(|t| *t == self.chart_type)
                        .unwrap_or(0);
                    self.chart_type = types[(idx + 1) % types.len()];
                } else if self.main_tab == MainTab::GPU {
                    let views = GpuView::all();
                    let idx = views.iter().position(|v| *v == self.gpu_view).unwrap_or(0);
                    self.gpu_view = views[(idx + 1) % views.len()];
                } else if self.main_tab == MainTab::Concepts {
                    let concepts = ConceptNode::all();
                    let idx = concepts
                        .iter()
                        .position(|c| *c == self.selected_concept)
                        .unwrap_or(0);
                    self.selected_concept = concepts[(idx + 1) % concepts.len()];
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

            // Zoom controls for charts (+/- or =/-)
            KeyCode::Char('+') | KeyCode::Char('=') => {
                self.chart_range = self.chart_range.next();
            }
            KeyCode::Char('-') | KeyCode::Char('_') => {
                self.chart_range = self.chart_range.prev();
            }
            // Full view shortcut
            KeyCode::Char('f') => {
                self.chart_range = ChartRange::Full;
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

        // Tabs
        let titles: Vec<Line> = MainTab::all()
            .iter()
            .map(|t| {
                let style = if *t == self.main_tab {
                    Style::default()
                        .fg(colors::TAB_ACTIVE)
                        .add_modifier(Modifier::BOLD)
                } else {
                    Style::default().fg(colors::TAB_INACTIVE)
                };
                Line::from(Span::styled(t.title(), style))
            })
            .collect();

        let tabs = Tabs::new(titles)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(colors::BORDER))
                    .title(Span::styled(
                        " RUST-AI MONITOR ",
                        Style::default()
                            .fg(Color::White)
                            .add_modifier(Modifier::BOLD),
                    )),
            )
            .select(self.main_tab.index())
            .style(Style::default())
            .highlight_style(Style::default().fg(colors::TAB_ACTIVE));

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
            Span::styled(
                mode_str,
                Style::default().fg(mode_color).add_modifier(Modifier::BOLD),
            ),
            Span::raw("  "),
            Span::styled(
                format!("{} active", active),
                Style::default().fg(colors::ACTIVE_RUN),
            ),
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
            MainTab::Dashboard => self.draw_dashboard(f, area),
            MainTab::Overview => self.draw_overview(f, area),
            MainTab::Charts => self.draw_charts_tab(f, area),
            MainTab::Analysis => self.draw_analysis_tab(f, area),
            MainTab::Network => self.draw_network_tab(f, area),
            MainTab::Concepts => self.draw_concepts_tab(f, area),
            MainTab::GPU => self.draw_gpu_tab(f, area),
            MainTab::History => self.draw_history_tab(f, area),
            MainTab::Help => self.draw_help_tab(f, area),
        }
    }

    /// Unified dashboard showing all key metrics in one view.
    fn draw_dashboard(&self, f: &mut Frame, area: Rect) {
        // Main layout: 3 columns
        let cols = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(25), // Left: Run list + Training Health
                Constraint::Percentage(50), // Center: Charts
                Constraint::Percentage(25), // Right: GPU + Metrics
            ])
            .split(area);

        // Left column: Run list (top) + Training Health (bottom)
        let left_rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Percentage(40), Constraint::Percentage(60)])
            .split(cols[0]);

        self.draw_run_list(f, left_rows[0]);
        self.draw_dashboard_health(f, left_rows[1]);

        // Center column: Charts (loss, gradient, LR)
        let center_rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Percentage(40), // Loss chart
                Constraint::Percentage(30), // Gradient chart
                Constraint::Percentage(30), // Key metrics bar
            ])
            .split(cols[1]);

        let run_id = self.selected_run_id.clone();
        if let (Some(ref run_id_ref), Some(reader)) = (
            &run_id,
            run_id.as_ref().and_then(|id| self.metrics_readers.get_mut(id)),
        ) {
            let run = self.run_manager.get_run(run_id_ref);
            self.draw_loss_line_chart(f, center_rows[0], reader, "📈 Loss Curve");
            self.draw_dashboard_gradient_chart(f, center_rows[1], reader);
            if let Some(run) = run {
                self.draw_dashboard_metrics_bar(f, center_rows[2], run, reader);
            } else {
                self.draw_empty_panel(f, center_rows[2], "Metrics");
            }
        } else {
            self.draw_empty_panel(f, center_rows[0], "Loss");
            self.draw_empty_panel(f, center_rows[1], "Gradient");
            self.draw_empty_panel(f, center_rows[2], "Metrics");
        }

        // Right column: GPU (top) + Phase/Progress (bottom)
        let right_rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Percentage(35), // GPU stats
                Constraint::Percentage(35), // Progress & Phase
                Constraint::Percentage(30), // Recommendations
            ])
            .split(cols[2]);

        self.draw_dashboard_gpu(f, right_rows[0]);

        if let (Some(run_id), Some(reader)) = (
            &self.selected_run_id,
            self.selected_run_id
                .as_ref()
                .and_then(|id| self.metrics_readers.get(id)),
        ) {
            if let Some(run) = self.run_manager.get_run(run_id) {
                self.draw_dashboard_phase(f, right_rows[1], run, reader);
                self.draw_dashboard_recommendations(f, right_rows[2], reader);
            } else {
                self.draw_empty_panel(f, right_rows[1], "Phase");
                self.draw_empty_panel(f, right_rows[2], "Recommendations");
            }
        } else {
            self.draw_empty_panel(f, right_rows[1], "Phase");
            self.draw_empty_panel(f, right_rows[2], "Recommendations");
        }
    }

    fn draw_empty_panel(&self, f: &mut Frame, area: Rect, title: &str) {
        let msg = Paragraph::new("No data")
            .style(Style::default().fg(Color::DarkGray))
            .alignment(Alignment::Center)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(colors::BORDER))
                    .title(format!(" {} ", title)),
            );
        f.render_widget(msg, area);
    }

    fn draw_dashboard_health(&self, f: &mut Frame, area: Rect) {
        let reader = self
            .selected_run_id
            .as_ref()
            .and_then(|id| self.metrics_readers.get(id));
        let metrics: Vec<_> = reader
            .map(|r| r.all_metrics().iter().cloned().collect())
            .unwrap_or_default();

        let mut lines = vec![
            Line::from(Span::styled(
                "TRAINING HEALTH",
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD),
            )),
            Line::from(""),
        ];

        if metrics.len() >= 10 {
            let losses: Vec<f32> = metrics.iter().map(|m| m.loss).collect();
            let gradients: Vec<f32> = metrics.iter().map(|m| m.gradient_norm).collect();
            let latest = metrics.last();
            let step = latest.map(|m| m.step).unwrap_or(0);

            let curve_quality = assess_curve_quality(&losses, &gradients, step);

            // Curve quality indicator with better colors
            let (status_icon, status_color) = match curve_quality {
                CurveQuality::Healthy => ("✓", Color::Green),
                CurveQuality::TooSlow | CurveQuality::Plateau => ("⚠", Color::Yellow),
                CurveQuality::TooFast => ("⚡", Color::Yellow),
                CurveQuality::Oscillating | CurveQuality::Diverging => ("✗", Color::Red),
            };

            lines.push(Line::from(vec![
                Span::styled(
                    format!("{} ", status_icon),
                    Style::default()
                        .fg(status_color)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    curve_quality.description(),
                    Style::default().fg(status_color),
                ),
            ]));

            // Loss sparkline (last 20 points)
            let sparkline_data: Vec<f64> = losses[losses.len().saturating_sub(20)..]
                .iter()
                .map(|&x| x as f64)
                .collect();
            let sparkline = create_sparkline(&sparkline_data);
            lines.push(Line::from(""));
            lines.push(Line::from(vec![
                Span::styled("Trend:     ", Style::default().fg(Color::Gray)),
                Span::styled(sparkline, Style::default().fg(colors::LOSS_LINE)),
            ]));

            // Loss statistics
            let recent = &losses[losses.len().saturating_sub(100)..];
            let avg_loss = recent.iter().sum::<f32>() / recent.len() as f32;
            let min_loss = recent.iter().cloned().fold(f32::INFINITY, f32::min);
            let max_loss = recent.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let loss_std = (recent.iter().map(|x| (x - avg_loss).powi(2)).sum::<f32>()
                / recent.len() as f32)
                .sqrt();
            let volatility = (loss_std / avg_loss.max(0.001) * 100.0).min(100.0);

            lines.push(Line::from(""));
            lines.push(Line::from(vec![
                Span::styled("Avg Loss:  ", Style::default().fg(Color::Gray)),
                Span::styled(
                    format!("{:.4}", avg_loss),
                    Style::default().fg(colors::LOSS_LINE),
                ),
            ]));
            lines.push(Line::from(vec![
                Span::styled("Min/Max:   ", Style::default().fg(Color::Gray)),
                Span::styled(
                    format!("{:.4} / {:.4}", min_loss, max_loss),
                    Style::default().fg(Color::White),
                ),
            ]));

            // Volatility with clear color coding: green=good, yellow=moderate, red=bad
            let vol_color = if volatility < 10.0 {
                Color::Green
            } else if volatility < 30.0 {
                Color::Yellow
            } else {
                Color::Red
            };
            lines.push(Line::from(vec![
                Span::styled("Volatility:", Style::default().fg(Color::Gray)),
                Span::styled(
                    format!(" {:.1}%", volatility),
                    Style::default().fg(vol_color).add_modifier(Modifier::BOLD),
                ),
            ]));

            // Gradient health with better color coding
            let recent_grads = &gradients[gradients.len().saturating_sub(20)..];
            let avg_grad = recent_grads.iter().sum::<f32>() / recent_grads.len() as f32;
            let grad_color = if avg_grad < 0.01 {
                Color::Yellow // Vanishing - warning
            } else if avg_grad > 10.0 {
                Color::Red // Exploding - bad
            } else {
                Color::Green // Healthy - good
            };
            lines.push(Line::from(vec![
                Span::styled("Grad Norm: ", Style::default().fg(Color::Gray)),
                Span::styled(
                    format!("{:.4}", avg_grad),
                    Style::default().fg(grad_color).add_modifier(Modifier::BOLD),
                ),
            ]));

            // Perplexity
            if let Some(latest) = latest {
                let ppl = latest.loss.exp();
                lines.push(Line::from(vec![
                    Span::styled("PPL:       ", Style::default().fg(Color::Gray)),
                    Span::styled(
                        format!("{:.1}", ppl),
                        Style::default().fg(colors::LOSS_SCATTER),
                    ),
                ]));
            }
        } else {
            lines.push(Line::from(Span::styled(
                "Collecting data...",
                Style::default().fg(Color::DarkGray),
            )));
        }

        let para = Paragraph::new(lines).block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(colors::BORDER))
                .title(" 🏥 Health "),
        );
        f.render_widget(para, area);
    }

    fn draw_dashboard_gradient_chart(&self, f: &mut Frame, area: Rect, reader: &LiveMetricsReader) {
        let metrics: Vec<_> = reader.all_metrics().iter().cloned().collect();
        if metrics.is_empty() {
            self.draw_empty_panel(f, area, "Gradient");
            return;
        }

        let data: Vec<(f64, f64)> = metrics
            .iter()
            .enumerate()
            .map(|(i, m)| (i as f64, m.gradient_norm as f64))
            .collect();

        let max_grad = data
            .iter()
            .map(|(_, g)| *g)
            .fold(0.0_f64, f64::max)
            .max(0.1);

        let dataset = Dataset::default()
            .marker(symbols::Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::default().fg(colors::GRAD_NORM))
            .data(&data);

        let chart = Chart::new(vec![dataset])
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(colors::BORDER))
                    .title(" 📊 Gradient Norm "),
            )
            .x_axis(
                Axis::default()
                    .bounds([0.0, data.len() as f64])
                    .style(Style::default().fg(Color::Gray)),
            )
            .y_axis(
                Axis::default()
                    .bounds([0.0, max_grad * 1.1])
                    .labels(vec![
                        Span::raw("0"),
                        Span::raw(format!("{:.2}", max_grad / 2.0)),
                        Span::raw(format!("{:.2}", max_grad)),
                    ])
                    .style(Style::default().fg(Color::Gray)),
            );

        f.render_widget(chart, area);
    }

    fn draw_dashboard_metrics_bar(
        &self,
        f: &mut Frame,
        area: Rect,
        run: &TrainingRun,
        reader: &LiveMetricsReader,
    ) {
        let latest = reader.latest();
        let metrics = reader.all_metrics();

        let step = latest.map(|m| m.step).unwrap_or(0);
        let loss = latest.map(|m| m.loss).unwrap_or(0.0);
        let lr = latest
            .map(|m| m.learning_rate)
            .unwrap_or(run.config.learning_rate);
        let tokens_trained = latest.map(|m| m.total_tokens_trained).unwrap_or(0);
        let phase = latest.map(|m| m.phase).unwrap_or(TrainingPhase::Warmup);

        // Calculate throughput
        let recent: Vec<_> = metrics.iter().rev().take(10).collect();
        let tokens_per_sec = if recent.len() >= 2 {
            let time_span_ms: f64 = recent.iter().map(|m| m.step_time_ms).sum();
            let tokens: u64 = recent.iter().map(|m| m.tokens_this_step).sum();
            if time_span_ms > 0.0 {
                tokens as f64 / (time_span_ms / 1000.0)
            } else {
                0.0
            }
        } else {
            0.0
        };

        let phase_color = match phase {
            TrainingPhase::Warmup => colors::WARMUP,
            TrainingPhase::Full => colors::FULL,
            TrainingPhase::Predict => colors::PREDICT,
            TrainingPhase::Correct => colors::CORRECT,
        };

        let lines = vec![
            Line::from(vec![
                Span::styled("Step: ", Style::default().fg(Color::Gray)),
                Span::styled(
                    format!("{:>7}", step),
                    Style::default()
                        .fg(Color::White)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled("  │  Loss: ", Style::default().fg(Color::Gray)),
                Span::styled(
                    format!("{:.4}", loss),
                    Style::default()
                        .fg(colors::LOSS_LINE)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled("  │  LR: ", Style::default().fg(Color::Gray)),
                Span::styled(format!("{:.2e}", lr), Style::default().fg(colors::WARMUP)),
            ]),
            Line::from(vec![
                Span::styled("Tokens: ", Style::default().fg(Color::Gray)),
                Span::styled(
                    format_tokens(tokens_trained),
                    Style::default().fg(colors::PREDICTION),
                ),
                Span::styled("  │  Throughput: ", Style::default().fg(Color::Gray)),
                Span::styled(
                    format!("{}/s", format_tokens(tokens_per_sec as u64)),
                    Style::default().fg(colors::STEP_TIME),
                ),
                Span::styled("  │  Phase: ", Style::default().fg(Color::Gray)),
                Span::styled(
                    format!("{:?}", phase),
                    Style::default()
                        .fg(phase_color)
                        .add_modifier(Modifier::BOLD),
                ),
            ]),
            Line::from(vec![
                Span::styled("Best Loss: ", Style::default().fg(Color::Gray)),
                Span::styled(
                    format!("{:.4}", run.best_loss),
                    Style::default().fg(colors::PREDICT),
                ),
                Span::styled(
                    format!(" @ step {}", run.best_step),
                    Style::default().fg(Color::DarkGray),
                ),
                Span::styled("  │  Progress: ", Style::default().fg(Color::Gray)),
                Span::styled(
                    format!(
                        "{:.1}%",
                        step as f64 / run.config.max_steps.max(1) as f64 * 100.0
                    ),
                    Style::default().fg(Color::White),
                ),
            ]),
        ];

        let para = Paragraph::new(lines).block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(colors::BORDER))
                .title(" 📋 Key Metrics "),
        );
        f.render_widget(para, area);
    }

    fn draw_dashboard_gpu(&self, f: &mut Frame, area: Rect) {
        let stats = self.gpu_monitor.current();

        let lines = if let Some(s) = stats {
            let mem_used_gb = s.memory_used as f32 / (1024.0 * 1024.0 * 1024.0);
            let mem_total_gb = s.memory_total as f32 / (1024.0 * 1024.0 * 1024.0);
            let mem_pct = mem_used_gb / mem_total_gb.max(0.001) * 100.0;
            let mem_color = if mem_pct < 70.0 {
                colors::MEMORY_OK
            } else if mem_pct < 90.0 {
                colors::MEMORY_WARN
            } else {
                colors::MEMORY_CRIT
            };
            let temp_color = if s.temperature < 70 {
                colors::TEMP_OK
            } else if s.temperature < 85 {
                colors::TEMP_WARN
            } else {
                colors::TEMP_CRIT
            };

            vec![
                Line::from(vec![
                    Span::styled("Memory:  ", Style::default().fg(Color::Gray)),
                    Span::styled(
                        format!("{:.1}%", mem_pct),
                        Style::default().fg(mem_color).add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(
                        format!(" ({:.1}/{:.1}GB)", mem_used_gb, mem_total_gb),
                        Style::default().fg(Color::DarkGray),
                    ),
                ]),
                Line::from(vec![
                    Span::styled("GPU Util:", Style::default().fg(Color::Gray)),
                    Span::styled(
                        format!(" {}%", s.gpu_util),
                        Style::default().fg(colors::PREDICTION),
                    ),
                ]),
                Line::from(vec![
                    Span::styled("Temp:    ", Style::default().fg(Color::Gray)),
                    Span::styled(
                        format!("{}°C", s.temperature),
                        Style::default().fg(temp_color),
                    ),
                ]),
                Line::from(vec![
                    Span::styled("Power:   ", Style::default().fg(Color::Gray)),
                    Span::styled(
                        format!("{:.0}W / {:.0}W", s.power_draw, s.power_limit),
                        Style::default().fg(Color::White),
                    ),
                ]),
            ]
        } else {
            vec![Line::from(Span::styled(
                "No GPU data",
                Style::default().fg(Color::DarkGray),
            ))]
        };

        let para = Paragraph::new(lines).block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(colors::BORDER))
                .title(" 🖥️ GPU "),
        );
        f.render_widget(para, area);
    }

    fn draw_dashboard_phase(
        &self,
        f: &mut Frame,
        area: Rect,
        run: &TrainingRun,
        reader: &LiveMetricsReader,
    ) {
        let latest = reader.latest();
        let metrics: Vec<_> = reader.all_metrics().iter().cloned().collect();
        let phase = latest.map(|m| m.phase).unwrap_or(TrainingPhase::Warmup);
        let confidence = latest.map(|m| m.confidence).unwrap_or(0.0);
        let was_predicted = latest.map(|m| m.was_predicted).unwrap_or(false);
        let step = latest.map(|m| m.step).unwrap_or(0);

        let phase_str = match phase {
            TrainingPhase::Warmup => "WARMUP",
            TrainingPhase::Full => "FULL",
            TrainingPhase::Predict => "PREDICT",
            TrainingPhase::Correct => "CORRECT",
        };
        let phase_color = match phase {
            TrainingPhase::Warmup => colors::WARMUP,
            TrainingPhase::Full => colors::FULL,
            TrainingPhase::Predict => colors::PREDICT,
            TrainingPhase::Correct => colors::CORRECT,
        };

        // Progress bar
        let max_steps = run.config.max_steps.max(1);
        let progress_pct = (step as f64 / max_steps as f64 * 100.0).min(100.0);
        let progress_bar = generate_progress_bar(progress_pct, 20);

        // Estimate time remaining
        let time_remaining = if metrics.len() >= 5 {
            let recent: Vec<_> = metrics.iter().rev().take(50).collect();
            let avg_step_time: f64 =
                recent.iter().map(|m| m.step_time_ms).sum::<f64>() / recent.len() as f64;
            let remaining_steps = max_steps.saturating_sub(step);
            let remaining_secs = (remaining_steps as f64 * avg_step_time) / 1000.0;
            format_duration(remaining_secs)
        } else {
            "calculating...".to_string()
        };

        let mut lines = vec![
            Line::from(vec![
                Span::styled("Progress:  ", Style::default().fg(Color::Gray)),
                Span::styled(
                    format!("{:.1}%", progress_pct),
                    Style::default()
                        .fg(Color::White)
                        .add_modifier(Modifier::BOLD),
                ),
            ]),
            Line::from(vec![Span::styled(
                progress_bar,
                Style::default().fg(Color::Cyan),
            )]),
            Line::from(vec![Span::styled(
                format!("{} / {} steps", step, max_steps),
                Style::default().fg(Color::DarkGray),
            )]),
            Line::from(""),
            Line::from(vec![
                Span::styled("Remaining: ", Style::default().fg(Color::Gray)),
                Span::styled(time_remaining, Style::default().fg(Color::White)),
            ]),
            Line::from(""),
        ];

        // Phase status
        lines.push(Line::from(vec![
            Span::styled("Phase:     ", Style::default().fg(Color::Gray)),
            Span::styled(
                phase_str,
                Style::default()
                    .fg(phase_color)
                    .add_modifier(Modifier::BOLD),
            ),
        ]));

        // Confidence with color coding: green=high, yellow=moderate, red=low
        let conf_pct = confidence * 100.0;
        let conf_color = if conf_pct >= 80.0 {
            Color::Green
        } else if conf_pct >= 50.0 {
            Color::Yellow
        } else {
            Color::Red
        };
        lines.push(Line::from(vec![
            Span::styled("Confidence:", Style::default().fg(Color::Gray)),
            Span::styled(
                format!(" {:.1}%", conf_pct),
                Style::default().fg(conf_color),
            ),
        ]));

        lines.push(Line::from(vec![
            Span::styled("Predicted: ", Style::default().fg(Color::Gray)),
            Span::styled(
                if was_predicted { "Yes ✓" } else { "No" },
                Style::default().fg(if was_predicted {
                    Color::Green
                } else {
                    Color::Gray
                }),
            ),
        ]));

        let backward_savings = 100.0 - run.backward_reduction();
        lines.push(Line::from(vec![
            Span::styled("Savings:   ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{:.1}%", backward_savings),
                Style::default().fg(Color::Green),
            ),
            Span::styled(" backward", Style::default().fg(Color::DarkGray)),
        ]));

        let para = Paragraph::new(lines).block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(colors::BORDER))
                .title(" 🔄 Progress & Phase "),
        );
        f.render_widget(para, area);
    }

    fn draw_dashboard_recommendations(
        &self,
        f: &mut Frame,
        area: Rect,
        reader: &LiveMetricsReader,
    ) {
        let metrics: Vec<_> = reader.all_metrics().iter().cloned().collect();
        let mut recommendations = Vec::new();

        if metrics.len() >= 10 {
            let losses: Vec<f32> = metrics.iter().map(|m| m.loss).collect();
            let gradients: Vec<f32> = metrics.iter().map(|m| m.gradient_norm).collect();
            let step = metrics.last().map(|m| m.step).unwrap_or(0);

            let curve_quality = assess_curve_quality(&losses, &gradients, step);

            match curve_quality {
                CurveQuality::Healthy => {
                    recommendations.push(Line::from(vec![
                        Span::styled("✓ ", Style::default().fg(colors::PREDICT)),
                        Span::styled("Training healthy", Style::default().fg(colors::PREDICT)),
                    ]));
                }
                CurveQuality::TooSlow => {
                    recommendations.push(Line::from(vec![
                        Span::styled("⚠ ", Style::default().fg(colors::WARMUP)),
                        Span::styled(
                            "Increase LR or batch size",
                            Style::default().fg(colors::WARMUP),
                        ),
                    ]));
                }
                CurveQuality::TooFast => {
                    recommendations.push(Line::from(vec![
                        Span::styled("⚡ ", Style::default().fg(colors::WARMUP)),
                        Span::styled(
                            "Decrease LR, add regularization",
                            Style::default().fg(colors::WARMUP),
                        ),
                    ]));
                }
                CurveQuality::Oscillating => {
                    recommendations.push(Line::from(vec![
                        Span::styled("~ ", Style::default().fg(colors::FAILED_RUN)),
                        Span::styled(
                            "Reduce LR significantly",
                            Style::default().fg(colors::FAILED_RUN),
                        ),
                    ]));
                }
                CurveQuality::Plateau => {
                    recommendations.push(Line::from(vec![
                        Span::styled("─ ", Style::default().fg(colors::WARMUP)),
                        Span::styled(
                            "Try LR warmup or reset",
                            Style::default().fg(colors::WARMUP),
                        ),
                    ]));
                }
                CurveQuality::Diverging => {
                    recommendations.push(Line::from(vec![
                        Span::styled("✗ ", Style::default().fg(colors::FAILED_RUN)),
                        Span::styled(
                            "STOP - reduce LR 10x",
                            Style::default().fg(colors::FAILED_RUN),
                        ),
                    ]));
                }
            }

            // Check gradient health
            let recent_grads = &gradients[gradients.len().saturating_sub(20)..];
            let avg_grad = recent_grads.iter().sum::<f32>() / recent_grads.len() as f32;
            if avg_grad < 0.001 {
                recommendations.push(Line::from(vec![
                    Span::styled("⚠ ", Style::default().fg(colors::WARMUP)),
                    Span::styled("Vanishing gradients", Style::default().fg(colors::WARMUP)),
                ]));
            } else if avg_grad > 50.0 {
                recommendations.push(Line::from(vec![
                    Span::styled("⚠ ", Style::default().fg(colors::FAILED_RUN)),
                    Span::styled(
                        "Exploding gradients - clip",
                        Style::default().fg(colors::FAILED_RUN),
                    ),
                ]));
            }
        } else {
            recommendations.push(Line::from(Span::styled(
                "Collecting data...",
                Style::default().fg(Color::DarkGray),
            )));
        }

        let para = Paragraph::new(recommendations).block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(colors::BORDER))
                .title(" 💡 Recommendations "),
        );
        f.render_widget(para, area);
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

        let selected_run = self
            .selected_run_id
            .as_ref()
            .and_then(|id| self.run_manager.get_run(id));
        let run_id = self.selected_run_id.clone();
        let reader = run_id.as_ref().and_then(|id| self.metrics_readers.get_mut(id));

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
                let is_selected = self
                    .selected_run_id
                    .as_ref()
                    .map(|id| id == &run.run_id)
                    .unwrap_or(false);

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
                    Style::default()
                        .fg(Color::White)
                        .bg(colors::SELECTED_BG)
                        .add_modifier(Modifier::BOLD)
                } else {
                    Style::default()
                };

                ListItem::new(Line::from(vec![
                    Span::styled(
                        format!("{} ", status_icon),
                        Style::default().fg(status_color),
                    ),
                    Span::styled(
                        run.run_name.clone(),
                        style.fg(if is_selected {
                            Color::White
                        } else {
                            status_color
                        }),
                    ),
                    Span::styled(step_info, Style::default().fg(Color::Gray)),
                    Span::styled(loss_info, Style::default().fg(colors::LOSS_LINE)),
                ]))
            })
            .collect();

        let list = List::new(items).block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(colors::BORDER))
                .title(Span::styled(
                    " Runs ",
                    Style::default()
                        .fg(Color::White)
                        .add_modifier(Modifier::BOLD),
                )),
        );

        f.render_widget(list, area);
    }

    fn draw_progress(
        &self,
        f: &mut Frame,
        area: Rect,
        run: &TrainingRun,
        reader: &LiveMetricsReader,
    ) {
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

        let title = format!(
            " {} │ Step {}/{} │ {:?} ",
            run.run_name, step, run.config.max_steps, phase
        );

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

        let remaining_steps = run
            .config
            .max_steps
            .saturating_sub(latest.map(|m| m.step).unwrap_or(0));
        let eta_secs = if steps_per_sec > 0.0 {
            remaining_steps as f64 / steps_per_sec
        } else {
            0.0
        };

        let eta_str = format_duration(eta_secs);
        let elapsed = self.start_time.elapsed().as_secs_f64();
        let elapsed_str = format_duration(elapsed);

        let loss_trend = if metrics.len() >= 20 {
            let recent: f64 = metrics
                .iter()
                .rev()
                .take(10)
                .map(|m| m.loss as f64)
                .sum::<f64>()
                / 10.0;
            let older: f64 = metrics
                .iter()
                .rev()
                .skip(10)
                .take(10)
                .map(|m| m.loss as f64)
                .sum::<f64>()
                / 10.0;
            if recent < older * 0.99 {
                "↓ improving"
            } else if recent > older * 1.01 {
                "↑ rising"
            } else {
                "─ stable"
            }
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
        let tokens_trained = latest
            .map(|m| m.total_tokens_trained)
            .unwrap_or(run.total_tokens_trained);
        let tokens_remaining = latest.map(|m| m.tokens_remaining).unwrap_or(0);
        let tokens_per_step = latest
            .map(|m| m.tokens_this_step)
            .unwrap_or(run.tokens_per_step);
        let tokens_per_sec = run.tokens_per_second.unwrap_or_else(|| {
            if steps_per_sec > 0.0 {
                steps_per_sec * tokens_per_step as f64
            } else {
                0.0
            }
        });

        // Gradient norm and learning rate
        let grad_norm = latest.map(|m| m.gradient_norm).unwrap_or(0.0);
        let confidence = latest.map(|m| m.confidence).unwrap_or(0.0);
        let learning_rate = latest
            .map(|m| m.learning_rate)
            .unwrap_or(run.config.learning_rate);
        let perplexity = (loss as f64).exp();

        // Assess training curve quality
        let curve_quality = if metrics.len() >= 10 {
            let losses: Vec<f32> = metrics.iter().map(|m| m.loss).collect();
            let gradients: Vec<f32> = metrics.iter().map(|m| m.gradient_norm).collect();
            let current_step = latest.map(|m| m.step).unwrap_or(0);
            assess_curve_quality(&losses, &gradients, current_step)
        } else {
            CurveQuality::Healthy // Not enough data yet
        };

        let stats = vec![
            Line::from(vec![
                Span::styled(" Loss: ", Style::default().fg(Color::Gray)),
                Span::styled(
                    format!("{:.4}", loss),
                    Style::default()
                        .fg(colors::LOSS_LINE)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    format!("  {}", loss_trend),
                    Style::default().fg(trend_color),
                ),
                Span::styled("  │  PPL: ", Style::default().fg(Color::Gray)),
                Span::styled(
                    format!("{:.1}", perplexity),
                    Style::default().fg(colors::LOSS_SCATTER),
                ),
            ]),
            Line::from(vec![
                Span::styled(" Best: ", Style::default().fg(Color::Gray)),
                Span::styled(
                    format!("{:.4}", run.best_loss),
                    Style::default().fg(colors::PREDICT),
                ),
                Span::styled(
                    format!(" @ step {}", run.best_step),
                    Style::default().fg(Color::Gray),
                ),
                Span::styled("  │  LR: ", Style::default().fg(Color::Gray)),
                Span::styled(
                    format!("{:.2e}", learning_rate),
                    Style::default().fg(colors::WARMUP),
                ),
            ]),
            Line::from(vec![
                Span::styled(" Speed: ", Style::default().fg(Color::Gray)),
                Span::styled(
                    format!("{:.2} steps/s", steps_per_sec),
                    Style::default().fg(Color::White),
                ),
                Span::styled("  │  ETA: ", Style::default().fg(Color::Gray)),
                Span::styled(eta_str, Style::default().fg(colors::WARMUP)),
            ]),
            Line::from(vec![
                Span::styled(" Elapsed: ", Style::default().fg(Color::Gray)),
                Span::styled(elapsed_str, Style::default().fg(Color::White)),
                Span::styled("  │  Backward: ", Style::default().fg(Color::Gray)),
                Span::styled(
                    format!("{:.1}% saved", 100.0 - run.backward_reduction()),
                    Style::default().fg(colors::PREDICT),
                ),
            ]),
            Line::from(vec![
                Span::styled(" Tokens: ", Style::default().fg(Color::Gray)),
                Span::styled(
                    format_tokens(tokens_trained),
                    Style::default().fg(colors::WARMUP),
                ),
                Span::styled(" trained  │  ", Style::default().fg(Color::Gray)),
                Span::styled(
                    format_tokens(tokens_remaining),
                    Style::default().fg(Color::White),
                ),
                Span::styled(" remaining", Style::default().fg(Color::Gray)),
            ]),
            Line::from(vec![
                Span::styled(" Throughput: ", Style::default().fg(Color::Gray)),
                Span::styled(
                    format!("{}/s", format_tokens(tokens_per_sec as u64)),
                    Style::default().fg(colors::STEP_TIME),
                ),
                Span::styled(
                    format!("  │  {} tok/step", format_tokens(tokens_per_step)),
                    Style::default().fg(Color::Gray),
                ),
            ]),
            Line::from(vec![
                Span::styled(" Grad ‖∇‖: ", Style::default().fg(Color::Gray)),
                Span::styled(
                    format!("{:.3}", grad_norm),
                    Style::default().fg(colors::GRAD_NORM),
                ),
                Span::styled("  │  Confidence: ", Style::default().fg(Color::Gray)),
                Span::styled(
                    format!("{:.1}%", confidence * 100.0),
                    Style::default().fg(colors::PREDICTION),
                ),
            ]),
            Line::from(vec![
                Span::styled(" Curve: ", Style::default().fg(Color::Gray)),
                Span::styled(
                    format!("{} ", curve_quality.icon()),
                    Style::default().fg(curve_quality.color()),
                ),
                Span::styled(
                    curve_quality.description(),
                    Style::default().fg(curve_quality.color()),
                ),
            ]),
        ];

        let para = Paragraph::new(stats).block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(colors::BORDER))
                .title(" Statistics "),
        );

        f.render_widget(para, area);
    }

    fn draw_loss_line_chart(
        &self,
        f: &mut Frame,
        area: Rect,
        reader: &mut LiveMetricsReader,
        title: &str,
    ) {
        let data = reader.loss_data();

        if data.is_empty() {
            let msg = Paragraph::new("Waiting for training data...")
                .style(Style::default().fg(Color::Gray))
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .title(format!(" {} ", title)),
                );
            f.render_widget(msg, area);
            return;
        }

        let min_loss = data.iter().map(|(_, l)| *l).fold(f64::INFINITY, f64::min);
        let max_loss = data
            .iter()
            .map(|(_, l)| *l)
            .fold(f64::NEG_INFINITY, f64::max);
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
                    .title(Span::styled(
                        format!(" {} ", title),
                        Style::default().fg(Color::White),
                    )),
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

    fn draw_scatter_with_trend(&self, f: &mut Frame, area: Rect, reader: &mut LiveMetricsReader) {
        let scatter_data = reader.loss_data();
        let trend_data = reader.smoothed_loss(0.1);

        if scatter_data.is_empty() {
            let msg = Paragraph::new("Waiting for training data...")
                .style(Style::default().fg(Color::Gray))
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .title(" Loss (Scatter + Trend) "),
                );
            f.render_widget(msg, area);
            return;
        }

        let min_loss = scatter_data
            .iter()
            .map(|(_, l)| *l)
            .fold(f64::INFINITY, f64::min);
        let max_loss = scatter_data
            .iter()
            .map(|(_, l)| *l)
            .fold(f64::NEG_INFINITY, f64::max);
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
            .constraints([Constraint::Length(3), Constraint::Min(10)])
            .split(area);

        // Chart type selector
        let chart_titles: Vec<Span> = ChartType::all()
            .iter()
            .map(|t| {
                let style = if *t == self.chart_type {
                    Style::default()
                        .fg(colors::TAB_ACTIVE)
                        .add_modifier(Modifier::BOLD)
                } else {
                    Style::default().fg(colors::TAB_INACTIVE)
                };
                Span::styled(format!(" {} ", t.title()), style)
            })
            .collect();

        let selector = Paragraph::new(Line::from(chart_titles)).block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(colors::BORDER))
                .title(" Chart Type [←/→] "),
        );

        f.render_widget(selector, chunks[0]);

        let run_id = self.selected_run_id.clone();
        let reader = run_id
            .as_ref()
            .and_then(|id| self.metrics_readers.get_mut(id));

        if let Some(reader) = reader {
            match self.chart_type {
                ChartType::LossLine => {
                    self.draw_loss_line_chart(f, chunks[1], reader, "Loss (Line)")
                }
                ChartType::LossScatter => self.draw_scatter_with_trend(f, chunks[1], reader),
                ChartType::GradientNorm => self.draw_gradient_chart(f, chunks[1], reader),
                ChartType::StepTime => self.draw_step_time_chart(f, chunks[1], reader),
                ChartType::PhaseBreakdown => self.draw_phase_breakdown(f, chunks[1], reader),
                ChartType::PredictionAccuracy => self.draw_prediction_chart(f, chunks[1], reader),
                ChartType::LearningRate => self.draw_learning_rate_chart(f, chunks[1], reader),
                ChartType::Throughput => self.draw_throughput_chart(f, chunks[1], reader),
                ChartType::MemoryUsage => self.draw_memory_chart(f, chunks[1], reader),
                ChartType::LossVsTokens => self.draw_loss_vs_tokens_chart(f, chunks[1], reader),
                ChartType::PhaseDistribution => {
                    self.draw_phase_distribution_chart(f, chunks[1], reader)
                }
            }
        } else {
            let msg = Paragraph::new("Select a run to view charts")
                .style(Style::default().fg(Color::Gray))
                .block(Block::default().borders(Borders::ALL));
            f.render_widget(msg, chunks[1]);
        }
    }

    fn draw_gradient_chart(&self, f: &mut Frame, area: Rect, reader: &mut LiveMetricsReader) {
        let data = reader.gradient_data();

        if data.is_empty() {
            let msg = Paragraph::new("No gradient norm data available")
                .style(Style::default().fg(Color::Gray))
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .title(" Gradient Norm "),
                );
            f.render_widget(msg, area);
            return;
        }

        let min_val = data.iter().map(|(_, v)| *v).fold(f64::INFINITY, f64::min);
        let max_val = data
            .iter()
            .map(|(_, v)| *v)
            .fold(f64::NEG_INFINITY, f64::max);
        let min_step = data.first().map(|(s, _)| *s).unwrap_or(0.0);
        let max_step = data.last().map(|(s, _)| *s).unwrap_or(1.0);

        let y_margin = (max_val - min_val).max(0.1) * 0.15;
        let y_min = (min_val - y_margin).max(0.0);
        let y_max = max_val + y_margin;
        let y_mid = (y_min + y_max) / 2.0;

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
                    .title(Span::styled("Step", Style::default().fg(Color::Gray)))
                    .style(Style::default().fg(Color::Gray))
                    .bounds([min_step, max_step])
                    .labels(vec![
                        Span::raw(format!("{:.0}", min_step)),
                        Span::raw(format!("{:.0}", (min_step + max_step) / 2.0)),
                        Span::raw(format!("{:.0}", max_step)),
                    ]),
            )
            .y_axis(
                Axis::default()
                    .title(Span::styled("‖∇‖", Style::default().fg(colors::GRAD_NORM)))
                    .style(Style::default().fg(Color::Gray))
                    .bounds([y_min, y_max])
                    .labels(vec![
                        Span::raw(format_axis_value(y_min)),
                        Span::raw(format_axis_value(y_mid)),
                        Span::raw(format_axis_value(y_max)),
                    ]),
            );

        f.render_widget(chart, area);
    }

    fn draw_step_time_chart(&self, f: &mut Frame, area: Rect, reader: &mut LiveMetricsReader) {
        let data = reader.step_time_data();

        if data.is_empty() {
            let msg = Paragraph::new("No step time data available")
                .style(Style::default().fg(Color::Gray))
                .block(Block::default().borders(Borders::ALL).title(" Step Time "));
            f.render_widget(msg, area);
            return;
        }

        let min_val = data.iter().map(|(_, v)| *v).fold(f64::INFINITY, f64::min);
        let max_val = data
            .iter()
            .map(|(_, v)| *v)
            .fold(f64::NEG_INFINITY, f64::max);
        let min_step = data.first().map(|(s, _)| *s).unwrap_or(0.0);
        let max_step = data.last().map(|(s, _)| *s).unwrap_or(1.0);

        let y_margin = (max_val - min_val).max(10.0) * 0.15;
        let y_max = max_val + y_margin;
        let y_mid = y_max / 2.0;

        // Calculate average step time
        let avg_time: f64 = data.iter().map(|(_, v)| *v).sum::<f64>() / data.len() as f64;

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
                    .title(format!(" Step Time (avg: {:.0}ms) ", avg_time)),
            )
            .x_axis(
                Axis::default()
                    .title(Span::styled("Step", Style::default().fg(Color::Gray)))
                    .style(Style::default().fg(Color::Gray))
                    .bounds([min_step, max_step])
                    .labels(vec![
                        Span::raw(format!("{:.0}", min_step)),
                        Span::raw(format!("{:.0}", (min_step + max_step) / 2.0)),
                        Span::raw(format!("{:.0}", max_step)),
                    ]),
            )
            .y_axis(
                Axis::default()
                    .title(Span::styled("ms", Style::default().fg(colors::STEP_TIME)))
                    .style(Style::default().fg(Color::Gray))
                    .bounds([0.0, y_max])
                    .labels(vec![
                        Span::raw("0"),
                        Span::raw(format!("{:.0}", y_mid)),
                        Span::raw(format!("{:.0}", y_max)),
                    ]),
            );

        f.render_widget(chart, area);
    }

    fn draw_phase_breakdown(&self, f: &mut Frame, area: Rect, reader: &LiveMetricsReader) {
        let stats = reader.phase_stats();

        if stats.is_empty() {
            let msg = Paragraph::new("No phase data available")
                .style(Style::default().fg(Color::Gray))
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .title(" Phase Breakdown "),
                );
            f.render_widget(msg, area);
            return;
        }

        let mut lines = vec![
            Line::from(Span::styled(
                " Phase         │ Steps │ Avg Loss │ Avg Time",
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD),
            )),
            Line::from(Span::raw(" ──────────────┼───────┼──────────┼──────────")),
        ];

        for phase in [
            TrainingPhase::Warmup,
            TrainingPhase::Full,
            TrainingPhase::Predict,
            TrainingPhase::Correct,
        ] {
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
                    Span::styled(
                        format!("{:8.4}", avg_loss),
                        Style::default().fg(colors::LOSS_LINE),
                    ),
                    Span::raw(" │ "),
                    Span::styled(
                        format!("{:6.0}ms", avg_time),
                        Style::default().fg(colors::STEP_TIME),
                    ),
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
            let msg = Paragraph::new(
                "No prediction data available (shown during PREDICT/CORRECT phases)",
            )
            .style(Style::default().fg(Color::Gray))
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(" Prediction Accuracy "),
            );
            f.render_widget(msg, area);
            return;
        }

        let min_step = data.first().map(|(s, _)| *s).unwrap_or(0.0);
        let max_step = data.last().map(|(s, _)| *s).unwrap_or(1.0);
        let min_err = data.iter().map(|(_, v)| *v).fold(f64::INFINITY, f64::min);
        let max_err = data.iter().map(|(_, v)| *v).fold(0.0f64, f64::max);
        let latest_err = data.last().map(|(_, v)| *v).unwrap_or(0.0);
        let y_max = max_err * 1.1;
        let y_mid = y_max / 2.0;

        let datasets = vec![Dataset::default()
            .name("Prediction Error")
            .marker(symbols::Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::default().fg(colors::PREDICTION))
            .data(&data)];

        // Show trend in title
        let trend = if data.len() > 10 {
            let recent: f64 = data.iter().rev().take(5).map(|(_, v)| *v).sum::<f64>() / 5.0;
            let older: f64 = data
                .iter()
                .rev()
                .skip(5)
                .take(5)
                .map(|(_, v)| *v)
                .sum::<f64>()
                / 5.0;
            if recent < older * 0.95 {
                " ↓"
            } else if recent > older * 1.05 {
                " ↑"
            } else {
                " →"
            }
        } else {
            ""
        };

        let chart = Chart::new(datasets)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(colors::BORDER))
                    .title(format!(
                        " Prediction Error (latest: {:.3}){} ",
                        latest_err, trend
                    )),
            )
            .x_axis(
                Axis::default()
                    .title(Span::styled("Step", Style::default().fg(Color::Gray)))
                    .style(Style::default().fg(Color::Gray))
                    .bounds([min_step, max_step])
                    .labels(vec![
                        Span::raw(format!("{:.0}", min_step)),
                        Span::raw(format!("{:.0}", (min_step + max_step) / 2.0)),
                        Span::raw(format!("{:.0}", max_step)),
                    ]),
            )
            .y_axis(
                Axis::default()
                    .title(Span::styled(
                        "Error",
                        Style::default().fg(colors::PREDICTION),
                    ))
                    .style(Style::default().fg(Color::Gray))
                    .bounds([0.0, y_max])
                    .labels(vec![
                        Span::raw("0"),
                        Span::raw(format_axis_value(y_mid)),
                        Span::raw(format_axis_value(y_max)),
                    ]),
            );

        f.render_widget(chart, area);
    }


    fn draw_learning_rate_chart(&self, f: &mut Frame, area: Rect, reader: &LiveMetricsReader) {
        let data = reader.learning_rate_data();

        if data.is_empty() {
            let msg = Paragraph::new("No learning rate data available")
                .style(Style::default().fg(Color::Gray))
                .block(Block::default().borders(Borders::ALL).title(" Learning Rate "));
            f.render_widget(msg, area);
            return;
        }

        let min_step = data.first().map(|(s, _)| *s).unwrap_or(0.0);
        let max_step = data.last().map(|(s, _)| *s).unwrap_or(1.0);
        let min_lr = data.iter().map(|(_, v)| *v).fold(f64::INFINITY, f64::min);
        let max_lr = data.iter().map(|(_, v)| *v).fold(f64::NEG_INFINITY, f64::max);
        let current_lr = data.last().map(|(_, v)| *v).unwrap_or(0.0);

        let y_margin = (max_lr - min_lr).max(1e-6) * 0.15;
        let y_min = (min_lr - y_margin).max(0.0);
        let y_max = max_lr + y_margin;
        let y_mid = (y_min + y_max) / 2.0;

        let datasets = vec![Dataset::default()
            .name("Learning Rate")
            .marker(symbols::Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::default().fg(colors::LEARNING_RATE))
            .data(&data)];

        let chart = Chart::new(datasets)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(colors::BORDER))
                    .title(format!(" Learning Rate (current: {:.2e}) ", current_lr)),
            )
            .x_axis(
                Axis::default()
                    .title(Span::styled("Step", Style::default().fg(Color::Gray)))
                    .style(Style::default().fg(Color::Gray))
                    .bounds([min_step, max_step])
                    .labels(vec![
                        Span::raw(format!("{:.0}", min_step)),
                        Span::raw(format!("{:.0}", (min_step + max_step) / 2.0)),
                        Span::raw(format!("{:.0}", max_step)),
                    ]),
            )
            .y_axis(
                Axis::default()
                    .title(Span::styled("LR", Style::default().fg(colors::LEARNING_RATE)))
                    .style(Style::default().fg(Color::Gray))
                    .bounds([y_min, y_max])
                    .labels(vec![
                        Span::raw(format!("{:.2e}", y_min)),
                        Span::raw(format!("{:.2e}", y_mid)),
                        Span::raw(format!("{:.2e}", y_max)),
                    ]),
            );

        f.render_widget(chart, area);
    }

    fn draw_throughput_chart(&self, f: &mut Frame, area: Rect, reader: &LiveMetricsReader) {
        let data = reader.token_throughput_data();

        if data.is_empty() {
            let msg = Paragraph::new("No throughput data available")
                .style(Style::default().fg(Color::Gray))
                .block(Block::default().borders(Borders::ALL).title(" Throughput "));
            f.render_widget(msg, area);
            return;
        }

        let min_step = data.first().map(|(s, _)| *s).unwrap_or(0.0);
        let max_step = data.last().map(|(s, _)| *s).unwrap_or(1.0);
        let (min_val, max_val, mean_val) = LiveMetricsReader::stats(&data);

        let y_margin = (max_val - min_val).max(10.0) * 0.15;
        let y_max = max_val + y_margin;
        let y_mid = y_max / 2.0;

        let datasets = vec![Dataset::default()
            .name("Tokens/sec")
            .marker(symbols::Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::default().fg(colors::THROUGHPUT))
            .data(&data)];

        let chart = Chart::new(datasets)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(colors::BORDER))
                    .title(format!(" Throughput (avg: {:.0} tok/s, range: {:.0}-{:.0}) ", mean_val, min_val, max_val)),
            )
            .x_axis(
                Axis::default()
                    .title(Span::styled("Step", Style::default().fg(Color::Gray)))
                    .style(Style::default().fg(Color::Gray))
                    .bounds([min_step, max_step])
                    .labels(vec![
                        Span::raw(format!("{:.0}", min_step)),
                        Span::raw(format!("{:.0}", (min_step + max_step) / 2.0)),
                        Span::raw(format!("{:.0}", max_step)),
                    ]),
            )
            .y_axis(
                Axis::default()
                    .title(Span::styled("tok/s", Style::default().fg(colors::THROUGHPUT)))
                    .style(Style::default().fg(Color::Gray))
                    .bounds([0.0, y_max])
                    .labels(vec![
                        Span::raw("0"),
                        Span::raw(format!("{:.0}", y_mid)),
                        Span::raw(format!("{:.0}", y_max)),
                    ]),
            );

        f.render_widget(chart, area);
    }

    fn draw_memory_chart(&self, f: &mut Frame, area: Rect, reader: &LiveMetricsReader) {
        let msg = Paragraph::new("Memory tracking not yet implemented in StepMetrics\n\nGPU memory is tracked in TrainingRun but not per-step.\nSee GPU tab for current memory usage.")
            .style(Style::default().fg(Color::Gray))
            .block(Block::default().borders(Borders::ALL).title(" Memory Usage "));
        f.render_widget(msg, area);
    }

    fn draw_loss_vs_tokens_chart(&self, f: &mut Frame, area: Rect, reader: &LiveMetricsReader) {
        let data = reader.loss_vs_tokens_data();

        if data.is_empty() {
            let msg = Paragraph::new("No token tracking data available")
                .style(Style::default().fg(Color::Gray))
                .block(Block::default().borders(Borders::ALL).title(" Loss vs Tokens "));
            f.render_widget(msg, area);
            return;
        }

        let min_tokens = data.first().map(|(t, _)| *t).unwrap_or(0.0);
        let max_tokens = data.last().map(|(t, _)| *t).unwrap_or(1.0);
        let min_loss = data.iter().map(|(_, l)| *l).fold(f64::INFINITY, f64::min);
        let max_loss = data.iter().map(|(_, l)| *l).fold(f64::NEG_INFINITY, f64::max);

        let y_margin = (max_loss - min_loss).max(0.1) * 0.15;
        let y_min = (min_loss - y_margin).max(0.0);
        let y_max = max_loss + y_margin;

        // Add EMA trend line
        let mut ema_data = Vec::new();
        let mut ema: Option<f64> = None;
        let alpha = 0.1;
        for (tokens, loss) in &data {
            ema = Some(match ema {
                Some(prev) => alpha * loss + (1.0 - alpha) * prev,
                None => *loss,
            });
            ema_data.push((*tokens, ema.unwrap()));
        }

        let datasets = vec![
            Dataset::default()
                .name("Loss")
                .marker(symbols::Marker::Dot)
                .graph_type(GraphType::Scatter)
                .style(Style::default().fg(colors::LOSS_SCATTER))
                .data(&data),
            Dataset::default()
                .name("Trend")
                .marker(symbols::Marker::Braille)
                .graph_type(GraphType::Line)
                .style(Style::default().fg(colors::TREND_LINE))
                .data(&ema_data),
        ];

        let chart = Chart::new(datasets)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(colors::BORDER))
                    .title(" Loss vs Total Tokens Trained "),
            )
            .x_axis(
                Axis::default()
                    .title(Span::styled("Tokens", Style::default().fg(Color::Gray)))
                    .style(Style::default().fg(Color::Gray))
                    .bounds([min_tokens, max_tokens])
                    .labels(vec![
                        Span::raw(format!("{:.1e}", min_tokens)),
                        Span::raw(format!("{:.1e}", (min_tokens + max_tokens) / 2.0)),
                        Span::raw(format!("{:.1e}", max_tokens)),
                    ]),
            )
            .y_axis(
                Axis::default()
                    .title(Span::styled("Loss", Style::default().fg(colors::LOSS_LINE)))
                    .style(Style::default().fg(Color::Gray))
                    .bounds([y_min, y_max])
                    .labels(vec![
                        Span::raw(format_axis_value(y_min)),
                        Span::raw(format_axis_value((y_min + y_max) / 2.0)),
                        Span::raw(format_axis_value(y_max)),
                    ]),
            );

        f.render_widget(chart, area);
    }

    fn draw_phase_distribution_chart(&self, f: &mut Frame, area: Rect, reader: &LiveMetricsReader) {
        let stats = reader.phase_stats();

        if stats.is_empty() {
            let msg = Paragraph::new("No phase data available")
                .style(Style::default().fg(Color::Gray))
                .block(Block::default().borders(Borders::ALL).title(" Phase Distribution "));
            f.render_widget(msg, area);
            return;
        }

        let total_steps: usize = stats.values().map(|(count, _, _)| *count).sum();

        let mut lines = vec![
            Line::from(Span::styled(
                format!(" Phase Distribution (Total: {} steps) ", total_steps),
                Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
            )),
            Line::from(Span::raw("")),
        ];

        for phase in [TrainingPhase::Warmup, TrainingPhase::Full, TrainingPhase::Predict, TrainingPhase::Correct] {
            if let Some((count, avg_loss, avg_time)) = stats.get(&phase) {
                let color = match phase {
                    TrainingPhase::Warmup => colors::WARMUP,
                    TrainingPhase::Full => colors::FULL,
                    TrainingPhase::Predict => colors::PREDICT,
                    TrainingPhase::Correct => colors::CORRECT,
                };

                let percentage = (*count as f64 / total_steps as f64) * 100.0;
                let bar_width = (percentage / 2.0) as usize;
                let bar = "█".repeat(bar_width.max(1));

                lines.push(Line::from(vec![
                    Span::styled(format!(" {:>8} │ ", format!("{:?}", phase)), Style::default().fg(Color::Gray)),
                    Span::styled(format!("{:>5} steps ", count), Style::default().fg(color)),
                    Span::styled(format!("({:>5.1}%) ", percentage), Style::default().fg(Color::Gray)),
                    Span::styled(bar, Style::default().fg(color)),
                ]));

                lines.push(Line::from(vec![
                    Span::raw("            │ "),
                    Span::styled(
                        format!("Avg Loss: {:.4}  Avg Time: {:.0}ms", avg_loss, avg_time),
                        Style::default().fg(Color::DarkGray),
                    ),
                ]));
                lines.push(Line::from(Span::raw("")));
            }
        }

        let paragraph = Paragraph::new(lines)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(colors::BORDER))
                    .title(" Phase Distribution "),
            );

        f.render_widget(paragraph, area);
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
                    Style::default()
                        .fg(colors::TAB_ACTIVE)
                        .add_modifier(Modifier::BOLD)
                } else {
                    Style::default().fg(colors::TAB_INACTIVE)
                };
                Span::styled(format!(" {} ", v.title()), style)
            })
            .collect();

        let selector = Paragraph::new(Line::from(view_titles)).block(
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
                        format!(
                            "{:.1}/{:.1}GB",
                            stats.memory_used as f64 / 1e9,
                            stats.memory_total as f64 / 1e9
                        ),
                        Style::default().fg(mem_color).add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(
                        format!(" ({:.0}%)", stats.memory_percent()),
                        Style::default().fg(mem_color),
                    ),
                    Span::styled("  │  ", Style::default().fg(colors::BORDER)),
                    Span::styled(
                        format!("{}°C", stats.temperature),
                        Style::default().fg(temp_color),
                    ),
                    Span::styled("  │  ", Style::default().fg(colors::BORDER)),
                    Span::styled(
                        format!("{:.0}W", stats.power_draw),
                        Style::default().fg(Color::White),
                    ),
                    Span::styled("  │  ", Style::default().fg(colors::BORDER)),
                    Span::styled(
                        format!("{}%", stats.gpu_util),
                        Style::default().fg(Color::White),
                    ),
                ]),
                Line::from(vec![
                    Span::styled(" Peak: ", Style::default().fg(Color::Gray)),
                    Span::styled(
                        format!("{:.1}GB", self.gpu_monitor.peak_memory() as f64 / 1e9),
                        Style::default().fg(mem_color),
                    ),
                    Span::styled("  │  ", Style::default().fg(colors::BORDER)),
                    Span::styled(
                        format!("{}°C", self.gpu_monitor.peak_temperature()),
                        Style::default().fg(temp_color),
                    ),
                    Span::styled("  │  ", Style::default().fg(colors::BORDER)),
                    Span::styled(
                        format!("{:.0}W", self.gpu_monitor.peak_power()),
                        Style::default().fg(Color::White),
                    ),
                    Span::styled("  │  ", Style::default().fg(colors::BORDER)),
                    Span::styled(stats.pstate.clone(), Style::default().fg(Color::Yellow)),
                ]),
            ]
        } else {
            vec![Line::from(Span::styled(
                " GPU stats unavailable",
                Style::default().fg(Color::Gray),
            ))]
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
                        format!(
                            "{:.2} / {:.2} GB",
                            stats.memory_used as f64 / 1e9,
                            stats.memory_total as f64 / 1e9
                        ),
                        Style::default().fg(mem_color).add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(
                        format!("  ({:.1}%)", mem_pct),
                        Style::default().fg(mem_color),
                    ),
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
                    Span::styled(
                        format!("{}°C", stats.temperature),
                        Style::default().fg(temp_color).add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(
                        format!("  (peak {}°C)", self.gpu_monitor.peak_temperature()),
                        Style::default().fg(Color::Gray),
                    ),
                ]),
                Line::from(vec![
                    Span::styled(" Power:     ", Style::default().fg(Color::Gray)),
                    Span::styled(
                        format!("{:.0} / {:.0} W", stats.power_draw, stats.power_limit),
                        Style::default().fg(Color::White),
                    ),
                    Span::styled(
                        format!("  (peak {:.0}W)", self.gpu_monitor.peak_power()),
                        Style::default().fg(Color::Gray),
                    ),
                ]),
                Line::from(Span::raw("")),
                Line::from(vec![
                    Span::styled(" GPU Util:  ", Style::default().fg(Color::Gray)),
                    Span::styled(
                        format!("{}%", stats.gpu_util),
                        Style::default()
                            .fg(Color::White)
                            .add_modifier(Modifier::BOLD),
                    ),
                ]),
                Line::from(vec![
                    Span::styled(" Mem Util:  ", Style::default().fg(Color::Gray)),
                    Span::styled(
                        format!("{}%", stats.memory_util),
                        Style::default().fg(Color::White),
                    ),
                ]),
                Line::from(vec![
                    Span::styled(" P-State:   ", Style::default().fg(Color::Gray)),
                    Span::styled(&stats.pstate, Style::default().fg(Color::Yellow)),
                ]),
                Line::from(vec![
                    Span::styled(" Clocks:    ", Style::default().fg(Color::Gray)),
                    Span::styled(
                        format!("{} MHz (GPU)", stats.clock_graphics),
                        Style::default().fg(Color::White),
                    ),
                    Span::styled(
                        format!("  {} MHz (Mem)", stats.clock_memory),
                        Style::default().fg(Color::Gray),
                    ),
                ]),
            ]
        } else {
            vec![Line::from(Span::styled(
                " GPU stats unavailable - nvidia-smi not found?",
                Style::default().fg(Color::Gray),
            ))]
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
        let total_mem = self
            .gpu_monitor
            .current()
            .map(|s| s.memory_total as f64 / 1e9)
            .unwrap_or(16.0);

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
            .x_axis(
                Axis::default()
                    .style(Style::default().fg(Color::Gray))
                    .bounds([0.0, data.len() as f64]),
            )
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
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .title(" GPU Temperature "),
                );
            f.render_widget(msg, area);
            return;
        }

        let data: Vec<(f64, f64)> = self
            .gpu_history
            .iter()
            .enumerate()
            .map(|(i, s)| (i as f64, s.temp_c as f64))
            .collect();

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
            .x_axis(
                Axis::default()
                    .style(Style::default().fg(Color::Gray))
                    .bounds([0.0, data.len() as f64]),
            )
            .y_axis(
                Axis::default()
                    .style(Style::default().fg(Color::Gray))
                    .bounds([0.0, 100.0])
                    .labels(vec![
                        Span::raw("0°C"),
                        Span::raw("50°C"),
                        Span::raw("100°C"),
                    ]),
            );

        f.render_widget(chart, area);
    }

    fn draw_gpu_util_chart(&self, f: &mut Frame, area: Rect) {
        if self.gpu_history.is_empty() {
            let msg = Paragraph::new("Collecting GPU utilization data...")
                .style(Style::default().fg(Color::Gray))
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .title(" GPU Utilization "),
                );
            f.render_widget(msg, area);
            return;
        }

        let data: Vec<(f64, f64)> = self
            .gpu_history
            .iter()
            .enumerate()
            .map(|(i, s)| (i as f64, s.util_percent))
            .collect();

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
            .x_axis(
                Axis::default()
                    .style(Style::default().fg(Color::Gray))
                    .bounds([0.0, data.len() as f64]),
            )
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
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .title(" GPU Flame Graph "),
                );
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
        let selected_run = self
            .selected_run_id
            .as_ref()
            .and_then(|id| self.run_manager.get_run(id));
        let reader = self
            .selected_run_id
            .as_ref()
            .and_then(|id| self.metrics_readers.get(id));

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
                    .block(
                        Block::default()
                            .borders(Borders::ALL)
                            .title(" Run History "),
                    );
                f.render_widget(msg, chunks[1]);
            }
        }
    }

    fn draw_analysis_tab(&self, f: &mut Frame, area: Rect) {
        let selected_run = self
            .selected_run_id
            .as_ref()
            .and_then(|id| self.run_manager.get_run(id));
        let reader = self
            .selected_run_id
            .as_ref()
            .and_then(|id| self.metrics_readers.get(id));

        match (selected_run, reader) {
            (Some(run), Some(reader)) => {
                let chunks = Layout::default()
                    .direction(Direction::Horizontal)
                    .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
                    .split(area);

                // Left: Training Health & Recommendations
                self.draw_training_health(f, chunks[0], run, reader);
                // Right: Hardware Optimization
                self.draw_optimization_insights(f, chunks[1], run, reader);
            }
            _ => {
                let msg = Paragraph::new("Select a run to view analysis")
                    .style(Style::default().fg(Color::Gray))
                    .block(Block::default().borders(Borders::ALL).title(" Analysis "));
                f.render_widget(msg, area);
            }
        }
    }

    fn draw_training_health(
        &self,
        f: &mut Frame,
        area: Rect,
        run: &TrainingRun,
        reader: &LiveMetricsReader,
    ) {
        let metrics = reader.all_metrics();
        if metrics.len() < 10 {
            let msg = Paragraph::new("Collecting data for analysis (need 10+ steps)...")
                .style(Style::default().fg(Color::Gray))
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .title(" Training Health "),
                );
            f.render_widget(msg, area);
            return;
        }

        let losses: Vec<f64> = metrics.iter().map(|m| m.loss as f64).collect();
        let grads: Vec<f64> = metrics.iter().map(|m| m.gradient_norm as f64).collect();

        // Calculate metrics
        let recent_100 = losses.len().saturating_sub(100);
        let recent_losses = &losses[recent_100..];
        let avg_loss = recent_losses.iter().sum::<f64>() / recent_losses.len() as f64;
        let loss_variance = recent_losses
            .iter()
            .map(|l| (l - avg_loss).powi(2))
            .sum::<f64>()
            / recent_losses.len() as f64;
        let loss_std = loss_variance.sqrt();

        let avg_grad = grads.iter().sum::<f64>() / grads.len() as f64;
        let recent_grads = &grads[recent_100..];
        let grad_trend = if recent_grads.len() > 20 {
            let first_half: f64 = recent_grads[..recent_grads.len() / 2].iter().sum::<f64>()
                / (recent_grads.len() / 2) as f64;
            let second_half: f64 = recent_grads[recent_grads.len() / 2..].iter().sum::<f64>()
                / (recent_grads.len() / 2) as f64;
            second_half - first_half
        } else {
            0.0
        };

        // Loss trend
        let loss_trend = if losses.len() > 100 {
            let first_100: f64 = losses[..100].iter().sum::<f64>() / 100.0;
            let last_100: f64 = losses[losses.len() - 100..].iter().sum::<f64>() / 100.0;
            (last_100 - first_100) / first_100 * 100.0
        } else {
            0.0
        };

        // Diagnose issues
        let mut issues: Vec<(Color, String)> = Vec::new();
        let mut recommendations: Vec<String> = Vec::new();

        // High volatility check
        let volatility_ratio = loss_std / avg_loss;
        if volatility_ratio > 0.5 {
            issues.push((
                colors::FAILED_RUN,
                format!(
                    "⚠ High volatility: {:.1}% std/mean",
                    volatility_ratio * 100.0
                ),
            ));
            recommendations.push("→ Reduce learning rate by 2-3x".to_string());
            recommendations.push("→ Increase batch size (gradient accumulation)".to_string());
        } else if volatility_ratio > 0.3 {
            issues.push((
                colors::WARMUP,
                format!("△ Moderate volatility: {:.1}%", volatility_ratio * 100.0),
            ));
            recommendations.push("→ Consider lowering learning rate".to_string());
        } else {
            issues.push((
                colors::PREDICT,
                format!(
                    "✓ Stable training: {:.1}% volatility",
                    volatility_ratio * 100.0
                ),
            ));
        }

        // Gradient health
        if avg_grad < 0.01 {
            issues.push((
                colors::FAILED_RUN,
                "⚠ Very low gradients - potential vanishing".to_string(),
            ));
            recommendations.push("→ Check for dead ReLUs, increase LR".to_string());
        } else if avg_grad > 10.0 {
            issues.push((
                colors::FAILED_RUN,
                format!("⚠ High gradients: {:.2} avg", avg_grad),
            ));
            recommendations.push("→ Add/reduce gradient clipping".to_string());
        } else {
            issues.push((
                colors::PREDICT,
                format!("✓ Healthy gradients: {:.2} avg", avg_grad),
            ));
        }

        // Loss progress
        if loss_trend > 10.0 {
            issues.push((
                colors::FAILED_RUN,
                format!("⚠ Loss increasing: +{:.1}%", loss_trend),
            ));
            recommendations.push("→ Learning rate too high or data issue".to_string());
        } else if loss_trend > 0.0 {
            issues.push((
                colors::WARMUP,
                format!("△ Loss stagnant: {:.1}% change", loss_trend),
            ));
        } else {
            issues.push((
                colors::PREDICT,
                format!("✓ Loss decreasing: {:.1}%", loss_trend),
            ));
        }

        // Backward ratio
        let backward_pct = 100.0 * run.total_backward as f64 / run.total_forward.max(1) as f64;
        if backward_pct > 99.0 {
            issues.push((
                colors::WARMUP,
                "△ Predictive training not engaging".to_string(),
            ));
            recommendations.push("→ Lower confidence threshold".to_string());
        } else if backward_pct < 80.0 {
            issues.push((
                colors::PREDICT,
                format!("✓ {:.1}% backward savings", 100.0 - backward_pct),
            ));
        }

        // Build display
        let mut lines = vec![
            Line::from(Span::styled(
                " TRAINING HEALTH",
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD),
            )),
            Line::from(Span::raw("")),
        ];

        for (color, msg) in &issues {
            lines.push(Line::from(Span::styled(
                format!(" {}", msg),
                Style::default().fg(*color),
            )));
        }

        lines.push(Line::from(Span::raw("")));
        lines.push(Line::from(Span::styled(
            " RECOMMENDATIONS",
            Style::default()
                .fg(colors::WARMUP)
                .add_modifier(Modifier::BOLD),
        )));

        if recommendations.is_empty() {
            lines.push(Line::from(Span::styled(
                " Training looks healthy!",
                Style::default().fg(colors::PREDICT),
            )));
        } else {
            for rec in &recommendations {
                lines.push(Line::from(Span::styled(
                    format!(" {}", rec),
                    Style::default().fg(Color::White),
                )));
            }
        }

        // Add quick stats
        lines.push(Line::from(Span::raw("")));
        lines.push(Line::from(Span::styled(
            " QUICK STATS",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )));
        lines.push(Line::from(vec![
            Span::styled(" Avg Loss (100): ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{:.4}", avg_loss),
                Style::default().fg(colors::LOSS_LINE),
            ),
        ]));
        lines.push(Line::from(vec![
            Span::styled(" Loss Std Dev:   ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{:.4}", loss_std),
                Style::default().fg(Color::White),
            ),
        ]));
        lines.push(Line::from(vec![
            Span::styled(" Perplexity:     ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{:.1}", avg_loss.exp()),
                Style::default().fg(colors::LOSS_SCATTER),
            ),
        ]));

        let para = Paragraph::new(lines)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(colors::BORDER))
                    .title(" Training Health & Recommendations "),
            )
            .wrap(Wrap { trim: false });

        f.render_widget(para, area);
    }

    fn draw_optimization_insights(
        &self,
        f: &mut Frame,
        area: Rect,
        run: &TrainingRun,
        reader: &LiveMetricsReader,
    ) {
        let metrics = reader.all_metrics();
        let gpu_stats = self.gpu_monitor.current();

        let mut lines = vec![
            Line::from(Span::styled(
                " HARDWARE UTILIZATION",
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD),
            )),
            Line::from(Span::raw("")),
        ];

        // GPU stats
        if let Some(stats) = gpu_stats {
            let mem_pct = stats.memory_percent();
            let mem_color = if mem_pct > 90.0 {
                colors::FAILED_RUN
            } else if mem_pct > 70.0 {
                colors::WARMUP
            } else {
                colors::PREDICT
            };

            lines.push(Line::from(vec![
                Span::styled(" GPU Memory:  ", Style::default().fg(Color::Gray)),
                Span::styled(
                    format!(
                        "{:.1}% ({:.1}/{:.1}GB)",
                        mem_pct,
                        stats.memory_used as f64 / 1e9,
                        stats.memory_total as f64 / 1e9
                    ),
                    Style::default().fg(mem_color),
                ),
            ]));
            lines.push(Line::from(vec![
                Span::styled(" GPU Util:    ", Style::default().fg(Color::Gray)),
                Span::styled(
                    format!("{}%", stats.gpu_util),
                    Style::default().fg(if stats.gpu_util > 80 {
                        colors::PREDICT
                    } else {
                        colors::WARMUP
                    }),
                ),
            ]));
            lines.push(Line::from(vec![
                Span::styled(" Temperature: ", Style::default().fg(Color::Gray)),
                Span::styled(
                    format!("{}°C", stats.temperature),
                    Style::default().fg(if stats.temperature > 80 {
                        colors::FAILED_RUN
                    } else {
                        colors::PREDICT
                    }),
                ),
            ]));
            lines.push(Line::from(vec![
                Span::styled(" Power:       ", Style::default().fg(Color::Gray)),
                Span::styled(
                    format!("{:.0}W / {:.0}W", stats.power_draw, stats.power_limit),
                    Style::default().fg(Color::White),
                ),
            ]));

            // Memory optimization suggestions
            lines.push(Line::from(Span::raw("")));
            lines.push(Line::from(Span::styled(
                " MEMORY OPTIMIZATION",
                Style::default()
                    .fg(colors::WARMUP)
                    .add_modifier(Modifier::BOLD),
            )));

            if mem_pct < 60.0 {
                lines.push(Line::from(Span::styled(
                    " → Can increase batch size",
                    Style::default().fg(colors::PREDICT),
                )));
                let suggested_batch =
                    (run.config.batch_size as f64 * (90.0 / mem_pct)).min(32.0) as usize;
                lines.push(Line::from(Span::styled(
                    format!("   Suggested: batch_size={}", suggested_batch),
                    Style::default().fg(Color::White),
                )));
            } else if mem_pct > 90.0 {
                lines.push(Line::from(Span::styled(
                    " → Memory pressure high",
                    Style::default().fg(colors::FAILED_RUN),
                )));
                lines.push(Line::from(Span::styled(
                    "   Enable gradient checkpointing",
                    Style::default().fg(Color::White),
                )));
            } else {
                lines.push(Line::from(Span::styled(
                    " ✓ Memory usage optimal",
                    Style::default().fg(colors::PREDICT),
                )));
            }
        } else {
            lines.push(Line::from(Span::styled(
                " GPU stats unavailable",
                Style::default().fg(Color::Gray),
            )));
        }

        // Throughput analysis
        lines.push(Line::from(Span::raw("")));
        lines.push(Line::from(Span::styled(
            " THROUGHPUT ANALYSIS",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )));

        if !metrics.is_empty() {
            let step_times: Vec<f64> = metrics.iter().map(|m| m.step_time_ms).collect();
            let avg_time = step_times.iter().sum::<f64>() / step_times.len() as f64;
            let steps_per_sec = 1000.0 / avg_time;
            let tokens_per_step = run.config.batch_size * run.config.max_seq_length;
            let tokens_per_sec = steps_per_sec * tokens_per_step as f64;

            lines.push(Line::from(vec![
                Span::styled(" Avg Step:    ", Style::default().fg(Color::Gray)),
                Span::styled(
                    format!("{:.0}ms", avg_time),
                    Style::default().fg(colors::STEP_TIME),
                ),
            ]));
            lines.push(Line::from(vec![
                Span::styled(" Throughput:  ", Style::default().fg(Color::Gray)),
                Span::styled(
                    format!("{:.1} steps/s", steps_per_sec),
                    Style::default().fg(Color::White),
                ),
            ]));
            lines.push(Line::from(vec![
                Span::styled(" Tokens/sec:  ", Style::default().fg(Color::Gray)),
                Span::styled(
                    format!("{:.0}", tokens_per_sec),
                    Style::default().fg(colors::WARMUP),
                ),
            ]));

            // Efficiency score
            let efficiency = if let Some(stats) = gpu_stats {
                (stats.gpu_util as f64 / 100.0) * (tokens_per_sec / 10000.0).min(1.0)
            } else {
                0.5
            };

            lines.push(Line::from(Span::raw("")));
            lines.push(Line::from(vec![
                Span::styled(" Efficiency:  ", Style::default().fg(Color::Gray)),
                Span::styled(
                    format!("{:.0}%", efficiency * 100.0),
                    Style::default().fg(if efficiency > 0.7 {
                        colors::PREDICT
                    } else if efficiency > 0.4 {
                        colors::WARMUP
                    } else {
                        colors::FAILED_RUN
                    }),
                ),
            ]));
        }

        let para = Paragraph::new(lines)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(colors::BORDER))
                    .title(" Hardware & Optimization "),
            )
            .wrap(Wrap { trim: false });

        f.render_widget(para, area);
    }


    /// Enhanced Analysis Tab - Comprehensive diagnostic view with 3 columns
    fn draw_analysis_tab(&self, f: &mut Frame, area: Rect) {
        let selected_run = self.selected_run_id.as_ref().and_then(|id| self.run_manager.get_run(id));
        let reader = self.selected_run_id.as_ref().and_then(|id| self.metrics_readers.get(id));

        match (selected_run, reader) {
            (Some(run), Some(reader)) => {
                let columns = Layout::default()
                    .direction(Direction::Horizontal)
                    .constraints([
                        Constraint::Percentage(33),
                        Constraint::Percentage(34),
                        Constraint::Percentage(33),
                    ])
                    .split(area);

                // Column 1: Loss Trend & Gradient Health Analysis
                self.draw_loss_gradient_analysis(f, columns[0], run, reader);
                // Column 2: Prediction Accuracy & Phase Efficiency
                self.draw_prediction_phase_analysis(f, columns[1], run, reader);
                // Column 3: Memory, Throughput & Performance
                self.draw_performance_analysis(f, columns[2], run, reader);
            }
            _ => {
                let msg = Paragraph::new("Select a run to view comprehensive training analysis")
                    .style(Style::default().fg(Color::Gray))
                    .block(Block::default().borders(Borders::ALL).title(" Comprehensive Analysis "));
                f.render_widget(msg, area);
            }
        }
    }


    // Enhanced Analysis Tab - Column 1: Loss Trend & Gradient Health Analysis
    fn draw_loss_gradient_analysis(&self, f: &mut Frame, area: Rect, run: &TrainingRun, reader: &LiveMetricsReader) {
        let metrics = reader.all_metrics();
        if metrics.len() < 10 {
            let msg = Paragraph::new("Collecting data...\n(need 10+ steps)")
                .style(Style::default().fg(Color::Gray))
                .block(Block::default().borders(Borders::ALL).title(" Loss & Gradient Analysis "));
            f.render_widget(msg, area);
            return;
        }

        let losses: Vec<f64> = metrics.iter().map(|m| m.loss as f64).collect();
        let grads: Vec<f64> = metrics.iter().map(|m| m.gradient_norm as f64).collect();

        // Calculate loss trend with linear regression slope
        let n = losses.len() as f64;
        let x_mean = (n - 1.0) / 2.0;
        let y_mean = losses.iter().sum::<f64>() / n;

        let mut numerator = 0.0;
        let mut denominator = 0.0;
        for (i, &loss) in losses.iter().enumerate() {
            let x_diff = i as f64 - x_mean;
            numerator += x_diff * (loss - y_mean);
            denominator += x_diff * x_diff;
        }
        let slope = if denominator != 0.0 { numerator / denominator } else { 0.0 };
        let slope_indicator = if slope < -0.001 { "↓↓" } else if slope < -0.0001 { "↓" } else if slope > 0.001 { "↑↑" } else if slope > 0.0001 { "↑" } else { "→" };
        let slope_color = if slope < -0.001 { colors::PREDICT } else if slope > 0.001 { colors::FAILED_RUN } else { colors::WARMUP };

        // Gradient health score (0-100)
        let avg_grad = grads.iter().sum::<f64>() / grads.len() as f64;
        let grad_variance = grads.iter().map(|g| (g - avg_grad).powi(2)).sum::<f64>() / grads.len() as f64;
        let grad_std = grad_variance.sqrt();
        let grad_cv = if avg_grad > 0.0 { grad_std / avg_grad } else { 999.0 };

        // Score based on: healthy magnitude (0.1-5.0), low variance, stable trend
        let magnitude_score = if avg_grad >= 0.1 && avg_grad <= 5.0 { 40.0 } else if avg_grad >= 0.01 && avg_grad <= 10.0 { 25.0 } else { 10.0 };
        let stability_score = if grad_cv < 0.3 { 40.0 } else if grad_cv < 0.6 { 25.0 } else { 10.0 };
        let trend_score = if grad_std < avg_grad * 0.5 { 20.0 } else { 10.0 };
        let gradient_health_score = (magnitude_score + stability_score + trend_score) as u8;
        let health_color = if gradient_health_score >= 75 { colors::PREDICT } else if gradient_health_score >= 50 { colors::WARMUP } else { colors::FAILED_RUN };

        // Loss statistics
        let recent_100 = losses.len().saturating_sub(100);
        let recent_losses = &losses[recent_100..];
        let avg_loss = recent_losses.iter().sum::<f64>() / recent_losses.len() as f64;
        let loss_variance = recent_losses.iter().map(|l| (l - avg_loss).powi(2)).sum::<f64>() / recent_losses.len() as f64;
        let loss_std = loss_variance.sqrt();
        let volatility_ratio = loss_std / avg_loss;

        let mut lines = vec![
            Line::from(Span::styled(
                " LOSS TREND ANALYSIS",
                Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
            )),
            Line::from(Span::raw("")),
        ];

        // Loss trend with slope indicator
        lines.push(Line::from(vec![
            Span::styled(" Trend:         ", Style::default().fg(Color::Gray)),
            Span::styled(format!("{} {:.2e}/step", slope_indicator, slope), Style::default().fg(slope_color)),
        ]));
        lines.push(Line::from(vec![
            Span::styled(" Avg Loss (100):", Style::default().fg(Color::Gray)),
            Span::styled(format!(" {:.4}", avg_loss), Style::default().fg(colors::LOSS_LINE)),
        ]));
        lines.push(Line::from(vec![
            Span::styled(" Std Dev:       ", Style::default().fg(Color::Gray)),
            Span::styled(format!(" {:.4}", loss_std), Style::default().fg(Color::White)),
        ]));
        lines.push(Line::from(vec![
            Span::styled(" Volatility:    ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!(" {:.1}%", volatility_ratio * 100.0),
                Style::default().fg(if volatility_ratio > 0.5 { colors::FAILED_RUN } else if volatility_ratio > 0.3 { colors::WARMUP } else { colors::PREDICT })
            ),
        ]));
        lines.push(Line::from(vec![
            Span::styled(" Perplexity:    ", Style::default().fg(Color::Gray)),
            Span::styled(format!(" {:.1}", avg_loss.exp()), Style::default().fg(colors::LOSS_SCATTER)),
        ]));

        // Gradient health
        lines.push(Line::from(Span::raw("")));
        lines.push(Line::from(Span::styled(
            " GRADIENT HEALTH",
            Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
        )));
        lines.push(Line::from(vec![
            Span::styled(" Health Score:  ", Style::default().fg(Color::Gray)),
            Span::styled(format!(" {}/100", gradient_health_score), Style::default().fg(health_color).add_modifier(Modifier::BOLD)),
        ]));
        lines.push(Line::from(vec![
            Span::styled(" Avg Magnitude: ", Style::default().fg(Color::Gray)),
            Span::styled(format!(" {:.4}", avg_grad), Style::default().fg(colors::GRAD_NORM)),
        ]));
        lines.push(Line::from(vec![
            Span::styled(" Std Dev:       ", Style::default().fg(Color::Gray)),
            Span::styled(format!(" {:.4}", grad_std), Style::default().fg(Color::White)),
        ]));
        lines.push(Line::from(vec![
            Span::styled(" Coeff. Var:    ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!(" {:.2}", grad_cv),
                Style::default().fg(if grad_cv < 0.3 { colors::PREDICT } else if grad_cv < 0.6 { colors::WARMUP } else { colors::FAILED_RUN })
            ),
        ]));

        // Diagnostic recommendations
        lines.push(Line::from(Span::raw("")));
        lines.push(Line::from(Span::styled(
            " RECOMMENDATIONS",
            Style::default().fg(colors::WARMUP).add_modifier(Modifier::BOLD),
        )));

        if gradient_health_score < 50 {
            if avg_grad < 0.01 {
                lines.push(Line::from(Span::styled(" → Vanishing gradients!", Style::default().fg(colors::FAILED_RUN))));
                lines.push(Line::from(Span::styled("   Increase learning rate", Style::default().fg(Color::Gray))));
            } else if avg_grad > 10.0 {
                lines.push(Line::from(Span::styled(" → Exploding gradients!", Style::default().fg(colors::FAILED_RUN))));
                lines.push(Line::from(Span::styled("   Reduce LR or clip grads", Style::default().fg(Color::Gray))));
            }
            if grad_cv > 0.6 {
                lines.push(Line::from(Span::styled(" → High instability", Style::default().fg(colors::FAILED_RUN))));
                lines.push(Line::from(Span::styled("   Reduce learning rate", Style::default().fg(Color::Gray))));
            }
        } else if volatility_ratio > 0.5 {
            lines.push(Line::from(Span::styled(" → Loss too volatile", Style::default().fg(colors::WARMUP))));
            lines.push(Line::from(Span::styled("   Increase batch size", Style::default().fg(Color::Gray))));
        } else {
            lines.push(Line::from(Span::styled(" ✓ Training healthy!", Style::default().fg(colors::PREDICT))));
        }

        let para = Paragraph::new(lines)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(colors::BORDER))
                    .title(" Loss & Gradient Analysis "),
            )
            .wrap(Wrap { trim: false });

        f.render_widget(para, area);
    }

    // Enhanced Analysis Tab - Column 2: Prediction Accuracy & Phase Efficiency
    fn draw_prediction_phase_analysis(&self, f: &mut Frame, area: Rect, run: &TrainingRun, reader: &LiveMetricsReader) {
        let metrics = reader.all_metrics();
        if metrics.is_empty() {
            let msg = Paragraph::new("No metrics available")
                .style(Style::default().fg(Color::Gray))
                .block(Block::default().borders(Borders::ALL).title(" Prediction & Phase Analysis "));
            f.render_widget(msg, area);
            return;
        }

        // Prediction accuracy statistics
        let total_predictions = metrics.iter().filter(|m| m.was_predicted).count();
        let predictions_with_error: Vec<f32> = metrics.iter()
            .filter_map(|m| if m.was_predicted { m.prediction_error } else { None })
            .collect();

        let avg_prediction_error = if !predictions_with_error.is_empty() {
            predictions_with_error.iter().sum::<f32>() / predictions_with_error.len() as f32
        } else {
            0.0
        };

        let prediction_accuracy_pct = if !predictions_with_error.is_empty() {
            let good_predictions = predictions_with_error.iter().filter(|&&e| e < 0.1).count();
            (good_predictions as f32 / predictions_with_error.len() as f32) * 100.0
        } else {
            0.0
        };

        // Phase efficiency metrics - time spent in each phase
        let mut phase_durations: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
        let mut phase_counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();

        for m in metrics.iter() {
            let phase_name = format!("{}", m.phase);
            *phase_durations.entry(phase_name.clone()).or_insert(0.0) += m.step_time_ms;
            *phase_counts.entry(phase_name).or_insert(0) += 1;
        }

        let total_time: f64 = phase_durations.values().sum();

        let mut lines = vec![
            Line::from(Span::styled(
                " PREDICTION ACCURACY",
                Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
            )),
            Line::from(Span::raw("")),
        ];

        // Prediction stats
        let prediction_rate = (total_predictions as f64 / metrics.len() as f64) * 100.0;
        lines.push(Line::from(vec![
            Span::styled(" Predictions:   ", Style::default().fg(Color::Gray)),
            Span::styled(format!(" {}/{} ({:.1}%)", total_predictions, metrics.len(), prediction_rate), Style::default().fg(colors::PREDICTION)),
        ]));

        if total_predictions > 0 {
            lines.push(Line::from(vec![
                Span::styled(" Avg Error:     ", Style::default().fg(Color::Gray)),
                Span::styled(
                    format!(" {:.4}", avg_prediction_error),
                    Style::default().fg(if avg_prediction_error < 0.05 { colors::PREDICT } else if avg_prediction_error < 0.15 { colors::WARMUP } else { colors::FAILED_RUN })
                ),
            ]));
            lines.push(Line::from(vec![
                Span::styled(" Accuracy:      ", Style::default().fg(Color::Gray)),
                Span::styled(
                    format!(" {:.1}% good", prediction_accuracy_pct),
                    Style::default().fg(if prediction_accuracy_pct > 75.0 { colors::PREDICT } else if prediction_accuracy_pct > 50.0 { colors::WARMUP } else { colors::FAILED_RUN })
                ),
            ]));
        } else {
            lines.push(Line::from(Span::styled(" No predictions yet", Style::default().fg(Color::Gray))));
        }

        // Compute savings from predictions
        let backward_saved = run.total_forward.saturating_sub(run.total_backward);
        let savings_pct = if run.total_forward > 0 {
            (backward_saved as f64 / run.total_forward as f64) * 100.0
        } else {
            0.0
        };
        lines.push(Line::from(vec![
            Span::styled(" Compute Saved: ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!(" {:.1}%", savings_pct),
                Style::default().fg(if savings_pct > 15.0 { colors::PREDICT } else if savings_pct > 5.0 { colors::WARMUP } else { Color::Gray })
            ),
        ]));

        // Phase efficiency
        lines.push(Line::from(Span::raw("")));
        lines.push(Line::from(Span::styled(
            " PHASE EFFICIENCY",
            Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
        )));

        let phases = ["WARMUP", "FULL", "PREDICT", "CORRECT"];
        let phase_colors = [colors::WARMUP, colors::FULL, colors::PREDICT, colors::CORRECT];

        for (phase, color) in phases.iter().zip(phase_colors.iter()) {
            if let Some(&duration) = phase_durations.get(*phase) {
                let pct = (duration / total_time) * 100.0;
                let count = phase_counts.get(*phase).unwrap_or(&0);
                let avg_time = if *count > 0 { duration / *count as f64 } else { 0.0 };

                lines.push(Line::from(vec![
                    Span::styled(format!(" {:<8}", phase), Style::default().fg(*color)),
                    Span::styled(format!(" {:.1}%  ", pct), Style::default().fg(Color::White)),
                    Span::styled(format!("({:.0}ms avg)", avg_time), Style::default().fg(Color::Gray)),
                ]));
            }
        }

        // Phase transition insights
        lines.push(Line::from(Span::raw("")));
        lines.push(Line::from(Span::styled(
            " INSIGHTS",
            Style::default().fg(colors::WARMUP).add_modifier(Modifier::BOLD),
        )));

        if prediction_accuracy_pct > 80.0 {
            lines.push(Line::from(Span::styled(" ✓ Excellent predictions!", Style::default().fg(colors::PREDICT))));
        } else if prediction_accuracy_pct > 60.0 {
            lines.push(Line::from(Span::styled(" △ Predictions improving", Style::default().fg(colors::WARMUP))));
        } else if total_predictions > 10 {
            lines.push(Line::from(Span::styled(" ⚠ Low prediction quality", Style::default().fg(colors::FAILED_RUN))));
            lines.push(Line::from(Span::styled("   Tune confidence thresh", Style::default().fg(Color::Gray))));
        }

        if savings_pct > 20.0 {
            lines.push(Line::from(Span::styled(" ✓ Great compute savings!", Style::default().fg(colors::PREDICT))));
        }

        let para = Paragraph::new(lines)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(colors::BORDER))
                    .title(" Prediction & Phase Analysis "),
            )
            .wrap(Wrap { trim: false });

        f.render_widget(para, area);
    }

    // Enhanced Analysis Tab - Column 3: Memory, Throughput & Performance
    fn draw_performance_analysis(&self, f: &mut Frame, area: Rect, run: &TrainingRun, reader: &LiveMetricsReader) {
        let metrics = reader.all_metrics();
        let gpu_stats = self.gpu_monitor.current();

        let mut lines = vec![
            Line::from(Span::styled(
                " MEMORY EFFICIENCY",
                Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
            )),
            Line::from(Span::raw("")),
        ];

        // Memory statistics
        if let Some(stats) = gpu_stats {
            let mem_pct = stats.memory_percent();
            let mem_used_gb = stats.memory_used as f64 / 1e9;
            let mem_total_gb = stats.memory_total as f64 / 1e9;
            let mem_color = if mem_pct > 90.0 { colors::MEMORY_CRIT } else if mem_pct > 70.0 { colors::MEMORY_WARN } else { colors::MEMORY_OK };

            // Memory efficiency indicator (0-100)
            let memory_efficiency = if mem_pct > 85.0 && mem_pct < 95.0 {
                95
            } else if mem_pct > 70.0 && mem_pct < 85.0 {
                85
            } else if mem_pct > 95.0 {
                60
            } else {
                ((mem_pct / 70.0) * 70.0) as u8
            };

            lines.push(Line::from(vec![
                Span::styled(" GPU Memory:    ", Style::default().fg(Color::Gray)),
                Span::styled(format!(" {:.1}%", mem_pct), Style::default().fg(mem_color)),
            ]));
            lines.push(Line::from(vec![
                Span::styled(" Used/Total:    ", Style::default().fg(Color::Gray)),
                Span::styled(format!(" {:.1}/{:.1} GB", mem_used_gb, mem_total_gb), Style::default().fg(Color::White)),
            ]));
            lines.push(Line::from(vec![
                Span::styled(" Efficiency:    ", Style::default().fg(Color::Gray)),
                Span::styled(
                    format!(" {}/100", memory_efficiency),
                    Style::default().fg(if memory_efficiency > 80 { colors::PREDICT } else if memory_efficiency > 60 { colors::WARMUP } else { colors::FAILED_RUN }).add_modifier(Modifier::BOLD)
                ),
            ]));
            lines.push(Line::from(vec![
                Span::styled(" GPU Util:      ", Style::default().fg(Color::Gray)),
                Span::styled(format!(" {}%", stats.gpu_util), Style::default().fg(if stats.gpu_util > 80 { colors::PREDICT } else { colors::WARMUP })),
            ]));
            lines.push(Line::from(vec![
                Span::styled(" Temperature:   ", Style::default().fg(Color::Gray)),
                Span::styled(
                    format!(" {}°C", stats.temperature),
                    Style::default().fg(if stats.temperature > 80 { colors::TEMP_WARN } else { colors::TEMP_OK })
                ),
            ]));
        } else {
            lines.push(Line::from(Span::styled(" GPU stats unavailable", Style::default().fg(Color::Gray))));
        }

        // Throughput comparison
        lines.push(Line::from(Span::raw("")));
        lines.push(Line::from(Span::styled(
            " THROUGHPUT ANALYSIS",
            Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
        )));

        if !metrics.is_empty() {
            let step_times: Vec<f64> = metrics.iter().map(|m| m.step_time_ms).collect();
            let avg_time = step_times.iter().sum::<f64>() / step_times.len() as f64;
            let steps_per_sec = 1000.0 / avg_time;
            let tokens_per_step = run.config.batch_size * run.config.max_seq_length;
            let tokens_per_sec = steps_per_sec * tokens_per_step as f64;

            // Expected throughput (baseline: ~200 tokens/sec on CPU, ~2000 on GPU)
            let expected_throughput = if gpu_stats.is_some() { 2000.0 } else { 200.0 };
            let throughput_ratio = tokens_per_sec / expected_throughput;
            let throughput_pct = (throughput_ratio * 100.0).min(999.9);

            lines.push(Line::from(vec![
                Span::styled(" Avg Step Time: ", Style::default().fg(Color::Gray)),
                Span::styled(format!(" {:.0}ms", avg_time), Style::default().fg(colors::STEP_TIME)),
            ]));
            lines.push(Line::from(vec![
                Span::styled(" Steps/sec:     ", Style::default().fg(Color::Gray)),
                Span::styled(format!(" {:.2}", steps_per_sec), Style::default().fg(Color::White)),
            ]));
            lines.push(Line::from(vec![
                Span::styled(" Tokens/sec:    ", Style::default().fg(Color::Gray)),
                Span::styled(format!(" {:.0}", tokens_per_sec), Style::default().fg(colors::WARMUP)),
            ]));
            lines.push(Line::from(vec![
                Span::styled(" vs Expected:   ", Style::default().fg(Color::Gray)),
                Span::styled(
                    format!(" {:.0}%", throughput_pct),
                    Style::default().fg(if throughput_ratio > 0.8 { colors::PREDICT } else if throughput_ratio > 0.5 { colors::WARMUP } else { colors::FAILED_RUN })
                ),
            ]));

            // Recent trend
            if step_times.len() > 20 {
                let recent_20: f64 = step_times.iter().rev().take(20).sum::<f64>() / 20.0;
                let trend = ((recent_20 - avg_time) / avg_time) * 100.0;
                let trend_indicator = if trend < -5.0 { "↓ Improving" } else if trend > 5.0 { "↑ Slowing" } else { "→ Stable" };
                let trend_color = if trend < -5.0 { colors::PREDICT } else if trend > 5.0 { colors::FAILED_RUN } else { colors::WARMUP };

                lines.push(Line::from(vec![
                    Span::styled(" Recent Trend:  ", Style::default().fg(Color::Gray)),
                    Span::styled(format!(" {}", trend_indicator), Style::default().fg(trend_color)),
                ]));
            }
        }

        // Overall performance score
        lines.push(Line::from(Span::raw("")));
        lines.push(Line::from(Span::styled(
            " RECOMMENDATIONS",
            Style::default().fg(colors::WARMUP).add_modifier(Modifier::BOLD),
        )));

        if let Some(stats) = gpu_stats {
            let mem_pct = stats.memory_percent();
            if mem_pct < 60.0 {
                lines.push(Line::from(Span::styled(" → Increase batch size", Style::default().fg(colors::PREDICT))));
                let suggested_batch = ((run.config.batch_size as f64 * (85.0 / mem_pct)).min(64.0)) as usize;
                lines.push(Line::from(Span::styled(format!("   Try batch_size={}", suggested_batch), Style::default().fg(Color::Gray))));
            } else if mem_pct > 95.0 {
                lines.push(Line::from(Span::styled(" → Memory critical!", Style::default().fg(colors::FAILED_RUN))));
                lines.push(Line::from(Span::styled("   Enable grad checkpoint", Style::default().fg(Color::Gray))));
            } else if mem_pct > 85.0 && mem_pct < 95.0 {
                lines.push(Line::from(Span::styled(" ✓ Optimal memory usage", Style::default().fg(colors::PREDICT))));
            }

            if stats.gpu_util < 70 {
                lines.push(Line::from(Span::styled(" △ GPU underutilized", Style::default().fg(colors::WARMUP))));
                lines.push(Line::from(Span::styled("   Check data loading", Style::default().fg(Color::Gray))));
            }
        }

        let para = Paragraph::new(lines)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(colors::BORDER))
                    .title(" Performance & Efficiency "),
            )
            .wrap(Wrap { trim: false });

        f.render_widget(para, area);
    }


    /// Draw the Network Flow tab - visualizes transformer architecture and data flow.
    fn draw_network_tab(&self, f: &mut Frame, area: Rect) {
        let selected_run = self
            .selected_run_id
            .as_ref()
            .and_then(|id| self.run_manager.get_run(id));
        let reader = self
            .selected_run_id
            .as_ref()
            .and_then(|id| self.metrics_readers.get(id));

        match (selected_run, reader) {
            (Some(run), Some(reader)) => {
                let chunks = Layout::default()
                    .direction(Direction::Horizontal)
                    .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
                    .split(area);

                // Left: Network architecture visualization
                self.draw_network_architecture(f, chunks[0], run, reader);
                // Right: Layer-wise statistics
                self.draw_layer_stats(f, chunks[1], run, reader);
            }
            _ => {
                let msg = Paragraph::new("Select a run to view network visualization")
                    .style(Style::default().fg(Color::Gray))
                    .block(
                        Block::default()
                            .borders(Borders::ALL)
                            .title(" Network Flow "),
                    );
                f.render_widget(msg, area);
            }
        }
    }

    /// Draw ASCII representation of transformer architecture with data flow.
    fn draw_network_architecture(
        &self,
        f: &mut Frame,
        area: Rect,
        run: &TrainingRun,
        reader: &LiveMetricsReader,
    ) {
        let metrics = reader.all_metrics();
        let num_layers = run.config.num_layers;

        // Calculate gradient intensity for coloring (using recent gradient norm)
        let recent_grad = metrics
            .iter()
            .rev()
            .take(10)
            .map(|m| m.gradient_norm as f64)
            .sum::<f64>()
            / 10.0_f64.max(metrics.len() as f64);

        // Normalize gradient to 0-1 for color intensity
        let grad_intensity = (recent_grad / 5.0).min(1.0);

        let mut lines: Vec<Line> = Vec::new();

        // Title with data flow direction indicator
        lines.push(Line::from(Span::styled(
            " ╔═══════════════════════════════════════════════════╗",
            Style::default().fg(colors::BORDER),
        )));
        lines.push(Line::from(vec![
            Span::styled(" ║", Style::default().fg(colors::BORDER)),
            Span::styled(
                "  TRANSFORMER ARCHITECTURE  ",
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                "     DATA FLOW →          ",
                Style::default().fg(colors::PREDICTION),
            ),
            Span::styled("║", Style::default().fg(colors::BORDER)),
        ]));
        lines.push(Line::from(Span::styled(
            " ╠═══════════════════════════════════════════════════╣",
            Style::default().fg(colors::BORDER),
        )));

        // Input embedding layer
        lines.push(Line::from(vec![
            Span::styled(" ║ ", Style::default().fg(colors::BORDER)),
            Span::styled(
                "┌─────────────────────────────────────────────┐",
                Style::default().fg(colors::WARMUP),
            ),
            Span::styled("   ║", Style::default().fg(colors::BORDER)),
        ]));
        lines.push(Line::from(vec![
            Span::styled(" ║ ", Style::default().fg(colors::BORDER)),
            Span::styled("│  ", Style::default().fg(colors::WARMUP)),
            Span::styled(
                "INPUT EMBEDDING",
                Style::default()
                    .fg(colors::WARMUP)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!(
                    " ({}→{})",
                    run.config.max_seq_length, run.config.hidden_size
                ),
                Style::default().fg(Color::Gray),
            ),
            Span::styled("              │", Style::default().fg(colors::WARMUP)),
            Span::styled("   ║", Style::default().fg(colors::BORDER)),
        ]));
        lines.push(Line::from(vec![
            Span::styled(" ║ ", Style::default().fg(colors::BORDER)),
            Span::styled(
                "└──────────────────────┬──────────────────────┘",
                Style::default().fg(colors::WARMUP),
            ),
            Span::styled("   ║", Style::default().fg(colors::BORDER)),
        ]));

        // Data flow arrow
        lines.push(Line::from(vec![
            Span::styled(
                " ║                        ",
                Style::default().fg(colors::BORDER),
            ),
            Span::styled("│", Style::default().fg(colors::LOSS_LINE)),
            Span::styled(
                "                              ║",
                Style::default().fg(colors::BORDER),
            ),
        ]));
        lines.push(Line::from(vec![
            Span::styled(
                " ║                        ",
                Style::default().fg(colors::BORDER),
            ),
            Span::styled("▼", Style::default().fg(colors::LOSS_LINE)),
            Span::styled(
                "                              ║",
                Style::default().fg(colors::BORDER),
            ),
        ]));

        // Show first few layers and last layer with gradient intensity
        let layers_to_show = 3.min(num_layers);
        for i in 0..layers_to_show {
            let layer_color =
                intensity_to_color(grad_intensity * (1.0 - i as f64 / num_layers as f64));
            self.draw_transformer_layer(&mut lines, i, run.config.num_heads, layer_color);
        }

        // Ellipsis for middle layers
        if num_layers > 4 {
            lines.push(Line::from(vec![
                Span::styled(
                    " ║                        ",
                    Style::default().fg(colors::BORDER),
                ),
                Span::styled("⋮", Style::default().fg(Color::Gray)),
                Span::styled(
                    format!("  ({} more layers)", num_layers - 4),
                    Style::default().fg(Color::Gray),
                ),
                Span::styled("               ║", Style::default().fg(colors::BORDER)),
            ]));
            lines.push(Line::from(vec![
                Span::styled(
                    " ║                        ",
                    Style::default().fg(colors::BORDER),
                ),
                Span::styled("│", Style::default().fg(colors::LOSS_LINE)),
                Span::styled(
                    "                              ║",
                    Style::default().fg(colors::BORDER),
                ),
            ]));

            // Last layer
            let layer_color = intensity_to_color(grad_intensity * 0.3);
            self.draw_transformer_layer(
                &mut lines,
                num_layers - 1,
                run.config.num_heads,
                layer_color,
            );
        }

        // Output layer
        lines.push(Line::from(vec![
            Span::styled(" ║ ", Style::default().fg(colors::BORDER)),
            Span::styled(
                "┌─────────────────────────────────────────────┐",
                Style::default().fg(colors::PREDICT),
            ),
            Span::styled("   ║", Style::default().fg(colors::BORDER)),
        ]));
        lines.push(Line::from(vec![
            Span::styled(" ║ ", Style::default().fg(colors::BORDER)),
            Span::styled("│  ", Style::default().fg(colors::PREDICT)),
            Span::styled(
                "OUTPUT HEAD",
                Style::default()
                    .fg(colors::PREDICT)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                " (logits → softmax → loss)          │",
                Style::default().fg(Color::Gray),
            ),
            Span::styled("   ║", Style::default().fg(colors::BORDER)),
        ]));
        lines.push(Line::from(vec![
            Span::styled(" ║ ", Style::default().fg(colors::BORDER)),
            Span::styled(
                "└─────────────────────────────────────────────┘",
                Style::default().fg(colors::PREDICT),
            ),
            Span::styled("   ║", Style::default().fg(colors::BORDER)),
        ]));

        // Gradient backprop indicator
        lines.push(Line::from(Span::styled(
            " ╠═══════════════════════════════════════════════════╣",
            Style::default().fg(colors::BORDER),
        )));
        lines.push(Line::from(vec![
            Span::styled(" ║ ", Style::default().fg(colors::BORDER)),
            Span::styled(
                " ← GRADIENT BACKPROP ",
                Style::default()
                    .fg(colors::GRAD_NORM)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!(" ‖∇‖ = {:.3}", recent_grad),
                Style::default().fg(colors::GRAD_NORM),
            ),
            Span::styled("               ║", Style::default().fg(colors::BORDER)),
        ]));
        lines.push(Line::from(Span::styled(
            " ╚═══════════════════════════════════════════════════╝",
            Style::default().fg(colors::BORDER),
        )));

        let para = Paragraph::new(lines).block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(colors::BORDER))
                .title(Span::styled(
                    " Network Architecture ",
                    Style::default()
                        .fg(Color::White)
                        .add_modifier(Modifier::BOLD),
                )),
        );

        f.render_widget(para, area);
    }

    /// Helper to draw a single transformer layer.
    fn draw_transformer_layer(
        &self,
        lines: &mut Vec<Line>,
        layer_idx: usize,
        num_heads: usize,
        color: Color,
    ) {
        // Layer box
        lines.push(Line::from(vec![
            Span::styled(" ║ ", Style::default().fg(colors::BORDER)),
            Span::styled(
                "┌───────────────────────────────────────────┐  ",
                Style::default().fg(color),
            ),
            Span::styled("║", Style::default().fg(colors::BORDER)),
        ]));
        lines.push(Line::from(vec![
            Span::styled(" ║ ", Style::default().fg(colors::BORDER)),
            Span::styled(
                format!("│ Layer {} ", layer_idx),
                Style::default().fg(color).add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!("[Attn:{} heads] [FFN] [Norm]", num_heads),
                Style::default().fg(Color::Gray),
            ),
            Span::styled("    │  ", Style::default().fg(color)),
            Span::styled("║", Style::default().fg(colors::BORDER)),
        ]));
        lines.push(Line::from(vec![
            Span::styled(" ║ ", Style::default().fg(colors::BORDER)),
            Span::styled("│ ", Style::default().fg(color)),
            Span::styled("█", Style::default().fg(color)),
            Span::styled(
                "░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░",
                Style::default().fg(Color::Rgb(60, 60, 80)),
            ),
            Span::styled(" │  ", Style::default().fg(color)),
            Span::styled("║", Style::default().fg(colors::BORDER)),
        ]));
        lines.push(Line::from(vec![
            Span::styled(" ║ ", Style::default().fg(colors::BORDER)),
            Span::styled(
                "└──────────────────────┬────────────────────┘  ",
                Style::default().fg(color),
            ),
            Span::styled("║", Style::default().fg(colors::BORDER)),
        ]));

        // Connection to next layer
        lines.push(Line::from(vec![
            Span::styled(
                " ║                        ",
                Style::default().fg(colors::BORDER),
            ),
            Span::styled("│", Style::default().fg(colors::LOSS_LINE)),
            Span::styled(
                "                            ║",
                Style::default().fg(colors::BORDER),
            ),
        ]));
    }

    /// Draw layer-wise statistics panel.
    fn draw_layer_stats(
        &self,
        f: &mut Frame,
        area: Rect,
        run: &TrainingRun,
        reader: &LiveMetricsReader,
    ) {
        let metrics = reader.all_metrics();

        let mut lines = vec![
            Line::from(Span::styled(
                " LAYER STATISTICS",
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD),
            )),
            Line::from(Span::raw("")),
        ];

        // Current phase indicator with visual
        let phase = metrics
            .back()
            .map(|m| m.phase)
            .unwrap_or(TrainingPhase::Warmup);
        let phase_color = match phase {
            TrainingPhase::Warmup => colors::WARMUP,
            TrainingPhase::Full => colors::FULL,
            TrainingPhase::Predict => colors::PREDICT,
            TrainingPhase::Correct => colors::CORRECT,
        };

        lines.push(Line::from(vec![
            Span::styled(" Phase: ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{:?}", phase),
                Style::default()
                    .fg(phase_color)
                    .add_modifier(Modifier::BOLD),
            ),
        ]));

        // Phase flow visualization
        lines.push(Line::from(Span::raw("")));
        lines.push(Line::from(Span::styled(
            " Training Phase Flow:",
            Style::default().fg(Color::Gray),
        )));

        let warmup_style = if phase == TrainingPhase::Warmup {
            Style::default()
                .fg(colors::WARMUP)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::Rgb(80, 60, 0))
        };
        let full_style = if phase == TrainingPhase::Full {
            Style::default()
                .fg(colors::FULL)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::Rgb(0, 50, 80))
        };
        let predict_style = if phase == TrainingPhase::Predict {
            Style::default()
                .fg(colors::PREDICT)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::Rgb(0, 60, 30))
        };
        let correct_style = if phase == TrainingPhase::Correct {
            Style::default()
                .fg(colors::CORRECT)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::Rgb(60, 30, 60))
        };

        lines.push(Line::from(vec![
            Span::styled(" [WARM]", warmup_style),
            Span::styled("→", Style::default().fg(Color::Gray)),
            Span::styled("[FULL]", full_style),
            Span::styled("→", Style::default().fg(Color::Gray)),
            Span::styled("[PRED]", predict_style),
            Span::styled("→", Style::default().fg(Color::Gray)),
            Span::styled("[CORR]", correct_style),
            Span::styled("↺", Style::default().fg(Color::Gray)),
        ]));

        // Model configuration
        lines.push(Line::from(Span::raw("")));
        lines.push(Line::from(Span::styled(
            " Model Config:",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )));
        lines.push(Line::from(vec![
            Span::styled(" Layers:     ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{}", run.config.num_layers),
                Style::default().fg(Color::White),
            ),
        ]));
        lines.push(Line::from(vec![
            Span::styled(" Hidden:     ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{}", run.config.hidden_size),
                Style::default().fg(Color::White),
            ),
        ]));
        lines.push(Line::from(vec![
            Span::styled(" Heads:      ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{}", run.config.num_heads),
                Style::default().fg(Color::White),
            ),
        ]));
        lines.push(Line::from(vec![
            Span::styled(" Params:     ", Style::default().fg(Color::Gray)),
            Span::styled(
                format_tokens(run.config.num_parameters as u64),
                Style::default().fg(colors::WARMUP),
            ),
        ]));

        // Gradient checkpointing status
        lines.push(Line::from(Span::raw("")));
        lines.push(Line::from(Span::styled(
            " Memory Optimization:",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )));
        if run.config.gradient_checkpointing {
            lines.push(Line::from(vec![
                Span::styled(" ✓ ", Style::default().fg(colors::PREDICT)),
                Span::styled(
                    "Gradient checkpointing enabled",
                    Style::default().fg(Color::White),
                ),
            ]));
            lines.push(Line::from(vec![
                Span::styled("   Interval: ", Style::default().fg(Color::Gray)),
                Span::styled(
                    format!("every {} layers", run.config.checkpoint_interval),
                    Style::default().fg(Color::White),
                ),
            ]));
        } else {
            lines.push(Line::from(vec![
                Span::styled(" ○ ", Style::default().fg(Color::Gray)),
                Span::styled(
                    "No gradient checkpointing",
                    Style::default().fg(Color::Gray),
                ),
            ]));
        }

        // Recent gradient/loss trends
        if metrics.len() >= 10 {
            lines.push(Line::from(Span::raw("")));
            lines.push(Line::from(Span::styled(
                " Recent Trends:",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            )));

            let recent_losses: Vec<f64> = metrics
                .iter()
                .rev()
                .take(10)
                .map(|m| m.loss as f64)
                .collect();
            let recent_grads: Vec<f64> = metrics
                .iter()
                .rev()
                .take(10)
                .map(|m| m.gradient_norm as f64)
                .collect();

            // Loss mini-sparkline
            let loss_spark = create_sparkline(&recent_losses);
            lines.push(Line::from(vec![
                Span::styled(" Loss:  ", Style::default().fg(Color::Gray)),
                Span::styled(loss_spark, Style::default().fg(colors::LOSS_LINE)),
            ]));

            // Gradient mini-sparkline
            let grad_spark = create_sparkline(&recent_grads);
            lines.push(Line::from(vec![
                Span::styled(" Grad:  ", Style::default().fg(Color::Gray)),
                Span::styled(grad_spark, Style::default().fg(colors::GRAD_NORM)),
            ]));
        }

        let para = Paragraph::new(lines)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(colors::BORDER))
                    .title(" Layer & Flow Stats "),
            )
            .wrap(Wrap { trim: false });

        f.render_widget(para, area);
    }

    /// Draw the Concepts tab - interactive concept cloud for exploring metrics.
    fn draw_concepts_tab(&self, f: &mut Frame, area: Rect) {
        let selected_run = self
            .selected_run_id
            .as_ref()
            .and_then(|id| self.run_manager.get_run(id));
        let reader = self
            .selected_run_id
            .as_ref()
            .and_then(|id| self.metrics_readers.get(id));

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(12), Constraint::Min(8)])
            .split(area);

        // Top: Concept cloud with selection
        self.draw_concept_cloud(f, chunks[0]);

        // Bottom: Details for selected concept
        if let (Some(run), Some(reader)) = (selected_run, reader) {
            self.draw_concept_details(f, chunks[1], run, reader);
        } else {
            let msg = Paragraph::new("Select a run to view concept details")
                .style(Style::default().fg(Color::Gray))
                .block(Block::default().borders(Borders::ALL).title(" Details "));
            f.render_widget(msg, chunks[1]);
        }
    }

    /// Draw the concept cloud visualization.
    fn draw_concept_cloud(&self, f: &mut Frame, area: Rect) {
        let concepts = ConceptNode::all();
        let selected = self.selected_concept;

        // Create a visual cloud layout
        let mut lines: Vec<Line> = Vec::new();

        lines.push(Line::from(Span::styled(
            " CONCEPT CLOUD  [←/→ to navigate, see details below]",
            Style::default()
                .fg(Color::White)
                .add_modifier(Modifier::BOLD),
        )));
        lines.push(Line::from(Span::raw("")));

        // Row 1: Loss, Gradient, LR, Phase
        let mut row1: Vec<Span> = vec![Span::styled("     ", Style::default())];
        for concept in &[
            ConceptNode::Loss,
            ConceptNode::Gradient,
            ConceptNode::LearningRate,
            ConceptNode::Phase,
        ] {
            let is_selected = *concept == selected;
            let style = if is_selected {
                Style::default()
                    .fg(Color::Black)
                    .bg(concept.color())
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(concept.color())
            };
            row1.push(Span::styled(format!(" {} ", concept.title()), style));
            row1.push(Span::styled("   ", Style::default()));
        }
        lines.push(Line::from(row1));

        // Connections
        lines.push(Line::from(vec![
            Span::styled("       ╲", Style::default().fg(Color::Gray)),
            Span::styled("    │    ", Style::default().fg(Color::Gray)),
            Span::styled("╱", Style::default().fg(Color::Gray)),
            Span::styled("           ", Style::default()),
        ]));
        lines.push(Line::from(vec![
            Span::styled("        ╲", Style::default().fg(Color::Gray)),
            Span::styled("   │   ", Style::default().fg(Color::Gray)),
            Span::styled("╱", Style::default().fg(Color::Gray)),
            Span::styled("            ", Style::default()),
        ]));

        // Row 2: Perplexity, Tokens, Confidence
        let mut row2: Vec<Span> = vec![Span::styled("        ", Style::default())];
        for concept in &[
            ConceptNode::Perplexity,
            ConceptNode::Tokens,
            ConceptNode::Confidence,
        ] {
            let is_selected = *concept == selected;
            let style = if is_selected {
                Style::default()
                    .fg(Color::Black)
                    .bg(concept.color())
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(concept.color())
            };
            row2.push(Span::styled(format!(" {} ", concept.title()), style));
            row2.push(Span::styled("     ", Style::default()));
        }
        lines.push(Line::from(row2));

        // More connections
        lines.push(Line::from(vec![
            Span::styled("          │", Style::default().fg(Color::Gray)),
            Span::styled("      │      ", Style::default().fg(Color::Gray)),
            Span::styled("│", Style::default().fg(Color::Gray)),
        ]));

        // Row 3: StepTime, Memory, Efficiency
        let mut row3: Vec<Span> = vec![Span::styled("       ", Style::default())];
        for concept in &[
            ConceptNode::StepTime,
            ConceptNode::Memory,
            ConceptNode::Efficiency,
        ] {
            let is_selected = *concept == selected;
            let style = if is_selected {
                Style::default()
                    .fg(Color::Black)
                    .bg(concept.color())
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(concept.color())
            };
            row3.push(Span::styled(format!(" {} ", concept.title()), style));
            row3.push(Span::styled("    ", Style::default()));
        }
        lines.push(Line::from(row3));

        let para = Paragraph::new(lines).block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(colors::BORDER))
                .title(Span::styled(
                    " Training Concepts ",
                    Style::default().fg(Color::White),
                )),
        );

        f.render_widget(para, area);
    }

    /// Draw details for the selected concept.
    fn draw_concept_details(
        &self,
        f: &mut Frame,
        area: Rect,
        run: &TrainingRun,
        reader: &LiveMetricsReader,
    ) {
        let metrics = reader.all_metrics();
        let selected = self.selected_concept;
        let gpu_stats = self.gpu_monitor.current();

        let mut lines: Vec<Line> = Vec::new();

        // Title with selected concept
        lines.push(Line::from(vec![
            Span::styled(" Selected: ", Style::default().fg(Color::Gray)),
            Span::styled(
                selected.title(),
                Style::default()
                    .fg(selected.color())
                    .add_modifier(Modifier::BOLD),
            ),
        ]));
        lines.push(Line::from(Span::raw("")));

        // Description
        lines.push(Line::from(Span::styled(
            format!(" {}", selected.description()),
            Style::default().fg(Color::White),
        )));
        lines.push(Line::from(Span::raw("")));

        // Current value and statistics
        lines.push(Line::from(Span::styled(
            " Current Value:",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )));

        match selected {
            ConceptNode::Loss => {
                let loss = metrics.back().map(|m| m.loss).unwrap_or(0.0);
                let best = run.best_loss;
                lines.push(Line::from(vec![
                    Span::styled("   Current: ", Style::default().fg(Color::Gray)),
                    Span::styled(
                        format!("{:.4}", loss),
                        Style::default().fg(colors::LOSS_LINE),
                    ),
                ]));
                lines.push(Line::from(vec![
                    Span::styled("   Best:    ", Style::default().fg(Color::Gray)),
                    Span::styled(
                        format!("{:.4} @ step {}", best, run.best_step),
                        Style::default().fg(colors::PREDICT),
                    ),
                ]));
                if metrics.len() >= 100 {
                    let recent: f64 = metrics
                        .iter()
                        .rev()
                        .take(100)
                        .map(|m| m.loss as f64)
                        .sum::<f64>()
                        / 100.0;
                    lines.push(Line::from(vec![
                        Span::styled("   Avg(100):", Style::default().fg(Color::Gray)),
                        Span::styled(format!("{:.4}", recent), Style::default().fg(Color::White)),
                    ]));
                }
            }
            ConceptNode::Gradient => {
                let grad = metrics.back().map(|m| m.gradient_norm).unwrap_or(0.0);
                lines.push(Line::from(vec![
                    Span::styled("   ‖∇‖: ", Style::default().fg(Color::Gray)),
                    Span::styled(
                        format!("{:.4}", grad),
                        Style::default().fg(colors::GRAD_NORM),
                    ),
                ]));
                if metrics.len() >= 10 {
                    let avg: f64 = metrics
                        .iter()
                        .rev()
                        .take(10)
                        .map(|m| m.gradient_norm as f64)
                        .sum::<f64>()
                        / 10.0;
                    let health = if avg < 0.01 {
                        "VANISHING"
                    } else if avg > 10.0 {
                        "EXPLODING"
                    } else {
                        "HEALTHY"
                    };
                    let health_color = if avg < 0.01 || avg > 10.0 {
                        colors::FAILED_RUN
                    } else {
                        colors::PREDICT
                    };
                    lines.push(Line::from(vec![
                        Span::styled("   Status: ", Style::default().fg(Color::Gray)),
                        Span::styled(health, Style::default().fg(health_color)),
                    ]));
                }
            }
            ConceptNode::LearningRate => {
                let lr = metrics
                    .back()
                    .map(|m| m.learning_rate)
                    .unwrap_or(run.config.learning_rate);
                lines.push(Line::from(vec![
                    Span::styled("   Current: ", Style::default().fg(Color::Gray)),
                    Span::styled(format!("{:.2e}", lr), Style::default().fg(colors::WARMUP)),
                ]));
                lines.push(Line::from(vec![
                    Span::styled("   Config:  ", Style::default().fg(Color::Gray)),
                    Span::styled(
                        format!("{:.2e}", run.config.learning_rate),
                        Style::default().fg(Color::White),
                    ),
                ]));
            }
            ConceptNode::Phase => {
                let phase = metrics
                    .back()
                    .map(|m| m.phase)
                    .unwrap_or(TrainingPhase::Warmup);
                let backward_pct =
                    100.0 * run.total_backward as f64 / run.total_forward.max(1) as f64;
                lines.push(Line::from(vec![
                    Span::styled("   Current: ", Style::default().fg(Color::Gray)),
                    Span::styled(
                        format!("{:?}", phase),
                        Style::default().fg(match phase {
                            TrainingPhase::Warmup => colors::WARMUP,
                            TrainingPhase::Full => colors::FULL,
                            TrainingPhase::Predict => colors::PREDICT,
                            TrainingPhase::Correct => colors::CORRECT,
                        }),
                    ),
                ]));
                lines.push(Line::from(vec![
                    Span::styled("   Backward: ", Style::default().fg(Color::Gray)),
                    Span::styled(
                        format!("{:.1}% of steps", backward_pct),
                        Style::default().fg(Color::White),
                    ),
                ]));
            }
            ConceptNode::Perplexity => {
                let ppl = metrics.back().map(|m| (m.loss as f64).exp()).unwrap_or(0.0);
                lines.push(Line::from(vec![
                    Span::styled("   PPL: ", Style::default().fg(Color::Gray)),
                    Span::styled(
                        format!("{:.2}", ppl.min(99999.0)),
                        Style::default().fg(colors::LOSS_SCATTER),
                    ),
                ]));
                lines.push(Line::from(vec![
                    Span::styled("   Note: ", Style::default().fg(Color::Gray)),
                    Span::styled(
                        "Lower = better prediction quality",
                        Style::default().fg(Color::White),
                    ),
                ]));
            }
            ConceptNode::Tokens => {
                let total = run.total_tokens_trained;
                let per_step = run.tokens_per_step;
                let per_sec = run.tokens_per_second.unwrap_or(0.0);
                lines.push(Line::from(vec![
                    Span::styled("   Total:   ", Style::default().fg(Color::Gray)),
                    Span::styled(format_tokens(total), Style::default().fg(colors::WARMUP)),
                ]));
                lines.push(Line::from(vec![
                    Span::styled("   Per step:", Style::default().fg(Color::Gray)),
                    Span::styled(format_tokens(per_step), Style::default().fg(Color::White)),
                ]));
                lines.push(Line::from(vec![
                    Span::styled("   Per sec: ", Style::default().fg(Color::Gray)),
                    Span::styled(
                        format_tokens(per_sec as u64),
                        Style::default().fg(colors::STEP_TIME),
                    ),
                ]));
            }
            ConceptNode::Confidence => {
                let conf = metrics.back().map(|m| m.confidence).unwrap_or(0.0);
                lines.push(Line::from(vec![
                    Span::styled("   Predictor: ", Style::default().fg(Color::Gray)),
                    Span::styled(
                        format!("{:.1}%", conf * 100.0),
                        Style::default().fg(colors::PREDICTION),
                    ),
                ]));
                lines.push(Line::from(vec![
                    Span::styled("   Threshold: ", Style::default().fg(Color::Gray)),
                    Span::styled("85% (default)", Style::default().fg(Color::Gray)),
                ]));
            }
            ConceptNode::StepTime => {
                let time = metrics.back().map(|m| m.step_time_ms).unwrap_or(0.0);
                let avg: f64 = if !metrics.is_empty() {
                    metrics.iter().map(|m| m.step_time_ms).sum::<f64>() / metrics.len() as f64
                } else {
                    0.0
                };
                lines.push(Line::from(vec![
                    Span::styled("   Current: ", Style::default().fg(Color::Gray)),
                    Span::styled(
                        format!("{:.0} ms", time),
                        Style::default().fg(colors::STEP_TIME),
                    ),
                ]));
                lines.push(Line::from(vec![
                    Span::styled("   Average: ", Style::default().fg(Color::Gray)),
                    Span::styled(format!("{:.0} ms", avg), Style::default().fg(Color::White)),
                ]));
            }
            ConceptNode::Memory => {
                if let Some(stats) = gpu_stats {
                    let pct = stats.memory_percent();
                    lines.push(Line::from(vec![
                        Span::styled("   Used:  ", Style::default().fg(Color::Gray)),
                        Span::styled(
                            format!("{:.1} GB ({:.0}%)", stats.memory_used as f64 / 1e9, pct),
                            Style::default().fg(if pct > 90.0 {
                                colors::FAILED_RUN
                            } else {
                                colors::MEMORY_OK
                            }),
                        ),
                    ]));
                    lines.push(Line::from(vec![
                        Span::styled("   Total: ", Style::default().fg(Color::Gray)),
                        Span::styled(
                            format!("{:.1} GB", stats.memory_total as f64 / 1e9),
                            Style::default().fg(Color::White),
                        ),
                    ]));
                } else {
                    lines.push(Line::from(Span::styled(
                        "   GPU stats unavailable",
                        Style::default().fg(Color::Gray),
                    )));
                }
            }
            ConceptNode::Efficiency => {
                if let (Some(stats), false) = (gpu_stats, metrics.is_empty()) {
                    let util = stats.gpu_util as f64;
                    let time_avg: f64 =
                        metrics.iter().map(|m| m.step_time_ms).sum::<f64>() / metrics.len() as f64;
                    let throughput = 1000.0 / time_avg;
                    let efficiency = (util / 100.0) * (throughput / 10.0).min(1.0);
                    lines.push(Line::from(vec![
                        Span::styled("   GPU Util: ", Style::default().fg(Color::Gray)),
                        Span::styled(
                            format!("{}%", stats.gpu_util),
                            Style::default().fg(if util > 80.0 {
                                colors::PREDICT
                            } else {
                                colors::WARMUP
                            }),
                        ),
                    ]));
                    lines.push(Line::from(vec![
                        Span::styled("   Score:    ", Style::default().fg(Color::Gray)),
                        Span::styled(
                            format!("{:.0}%", efficiency * 100.0),
                            Style::default().fg(if efficiency > 0.7 {
                                colors::PREDICT
                            } else if efficiency > 0.4 {
                                colors::WARMUP
                            } else {
                                colors::FAILED_RUN
                            }),
                        ),
                    ]));
                } else {
                    lines.push(Line::from(Span::styled(
                        "   Calculating...",
                        Style::default().fg(Color::Gray),
                    )));
                }
            }
        }

        let para = Paragraph::new(lines)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(selected.color()))
                    .title(Span::styled(
                        format!(" {} Details ", selected.title()),
                        Style::default().fg(selected.color()),
                    )),
            )
            .wrap(Wrap { trim: false });

        f.render_widget(para, area);
    }

    fn draw_run_history_details(
        &self,
        f: &mut Frame,
        area: Rect,
        run: &TrainingRun,
        reader: &LiveMetricsReader,
    ) {
        let metrics = reader.all_metrics();
        let _first = metrics.front();
        let _last = metrics.back();

        let start_time = run.started_at.format("%Y-%m-%d %H:%M:%S").to_string();
        let end_time = run
            .ended_at
            .map(|t| t.format("%H:%M:%S").to_string())
            .unwrap_or_else(|| "running...".to_string());

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
                Span::styled(
                    format!("{}", metrics.len()),
                    Style::default().fg(Color::White),
                ),
            ]),
            Line::from(vec![
                Span::styled(" Best Loss:  ", Style::default().fg(Color::Gray)),
                Span::styled(
                    format!("{:.4} @ step {}", run.best_loss, run.best_step),
                    Style::default().fg(colors::PREDICT),
                ),
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
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD),
            )),
            Line::from(Span::raw("")),
            Line::from(vec![Span::styled(
                " Navigation",
                Style::default()
                    .fg(colors::WARMUP)
                    .add_modifier(Modifier::BOLD),
            )]),
            Line::from(vec![
                Span::styled(
                    "   Tab / Shift+Tab  ",
                    Style::default().fg(colors::HELP_KEY),
                ),
                Span::raw("Cycle through main tabs"),
            ]),
            Line::from(vec![
                Span::styled(
                    "   1-5              ",
                    Style::default().fg(colors::HELP_KEY),
                ),
                Span::raw("Jump to specific tab (Overview/Charts/GPU/History/Help)"),
            ]),
            Line::from(vec![
                Span::styled(
                    "   j/k or ↑/↓       ",
                    Style::default().fg(colors::HELP_KEY),
                ),
                Span::raw("Navigate between runs"),
            ]),
            Line::from(vec![
                Span::styled(
                    "   [/] or ←/→       ",
                    Style::default().fg(colors::HELP_KEY),
                ),
                Span::raw("Cycle chart types / GPU views within tabs"),
            ]),
            Line::from(Span::raw("")),
            Line::from(vec![Span::styled(
                " Mode Toggle",
                Style::default()
                    .fg(colors::WARMUP)
                    .add_modifier(Modifier::BOLD),
            )]),
            Line::from(vec![
                Span::styled(
                    "   l                ",
                    Style::default().fg(colors::HELP_KEY),
                ),
                Span::raw("Toggle LIVE ↔ HISTORY mode"),
            ]),
            Line::from(Span::raw("")),
            Line::from(vec![Span::styled(
                " Actions",
                Style::default()
                    .fg(colors::WARMUP)
                    .add_modifier(Modifier::BOLD),
            )]),
            Line::from(vec![
                Span::styled(
                    "   r                ",
                    Style::default().fg(colors::HELP_KEY),
                ),
                Span::raw("Refresh / rediscover runs"),
            ]),
            Line::from(vec![
                Span::styled(
                    "   c                ",
                    Style::default().fg(colors::HELP_KEY),
                ),
                Span::raw("Clear cached metrics (reloads from disk)"),
            ]),
            Line::from(vec![
                Span::styled(
                    "   ? or F1          ",
                    Style::default().fg(colors::HELP_KEY),
                ),
                Span::raw("Show help overlay"),
            ]),
            Line::from(vec![
                Span::styled(
                    "   q or Esc         ",
                    Style::default().fg(colors::HELP_KEY),
                ),
                Span::raw("Quit"),
            ]),
            Line::from(Span::raw("")),
            Line::from(vec![Span::styled(
                " Chart Types (Charts Tab)",
                Style::default()
                    .fg(colors::WARMUP)
                    .add_modifier(Modifier::BOLD),
            )]),
            Line::from(vec![
                Span::styled(
                    "   Loss (Line)       ",
                    Style::default().fg(colors::LOSS_LINE),
                ),
                Span::raw("Standard line chart of loss over steps"),
            ]),
            Line::from(vec![
                Span::styled(
                    "   Loss (Scatter)    ",
                    Style::default().fg(colors::LOSS_SCATTER),
                ),
                Span::raw("Scatter plot with EMA trend overlay"),
            ]),
            Line::from(vec![
                Span::styled(
                    "   Gradient Norm     ",
                    Style::default().fg(colors::GRAD_NORM),
                ),
                Span::raw("Gradient magnitude over training"),
            ]),
            Line::from(vec![
                Span::styled(
                    "   Step Time         ",
                    Style::default().fg(colors::STEP_TIME),
                ),
                Span::raw("Time per training step (ms)"),
            ]),
            Line::from(vec![
                Span::styled("   Phase Breakdown   ", Style::default().fg(Color::White)),
                Span::raw("Statistics by training phase"),
            ]),
            Line::from(vec![
                Span::styled(
                    "   Prediction Acc    ",
                    Style::default().fg(colors::PREDICTION),
                ),
                Span::raw("Prediction error in hybrid training"),
            ]),
            Line::from(Span::raw("")),
            Line::from(vec![Span::styled(
                " GPU Views (GPU Tab)",
                Style::default()
                    .fg(colors::WARMUP)
                    .add_modifier(Modifier::BOLD),
            )]),
            Line::from(vec![
                Span::styled("   Summary           ", Style::default().fg(Color::White)),
                Span::raw("Full GPU info: device, clocks, power, etc."),
            ]),
            Line::from(vec![
                Span::styled(
                    "   Memory            ",
                    Style::default().fg(colors::MEMORY_OK),
                ),
                Span::raw("VRAM usage over time"),
            ]),
            Line::from(vec![
                Span::styled(
                    "   Thermal           ",
                    Style::default().fg(colors::TEMP_WARN),
                ),
                Span::raw("GPU temperature history"),
            ]),
            Line::from(vec![
                Span::styled(
                    "   Utilization       ",
                    Style::default().fg(colors::PREDICT),
                ),
                Span::raw("GPU compute utilization %"),
            ]),
            Line::from(vec![
                Span::styled(
                    "   Flame             ",
                    Style::default().fg(colors::FLAME_WARM),
                ),
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
        let popup_width = 72.min(area.width - 4);
        let popup_height = 42.min(area.height - 4);

        let popup_area = Rect {
            x: (area.width - popup_width) / 2,
            y: (area.height - popup_height) / 2,
            width: popup_width,
            height: popup_height,
        };

        // Clear the area completely first
        f.render_widget(Clear, popup_area);

        // Render solid opaque background
        let bg = Block::default().style(Style::default().bg(Color::Rgb(25, 25, 40)));
        let help_text = vec![
            Line::from(Span::styled(
                " Training Monitor - Quick Reference",
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD),
            )),
            Line::from(Span::raw("")),
            Line::from(Span::styled(
                " NAVIGATION",
                Style::default()
                    .fg(colors::WARMUP)
                    .add_modifier(Modifier::BOLD),
            )),
            Line::from(vec![
                Span::styled(" Tab/Shift+Tab ", Style::default().fg(colors::HELP_KEY)),
                Span::raw("Cycle through tabs"),
            ]),
            Line::from(vec![
                Span::styled(" 0-8           ", Style::default().fg(colors::HELP_KEY)),
                Span::raw("Jump to tab (0=Dashboard, 1=Overview... 8=Help)"),
            ]),
            Line::from(vec![
                Span::styled(" j/k or ↑↓    ", Style::default().fg(colors::HELP_KEY)),
                Span::raw("Navigate between runs"),
            ]),
            Line::from(vec![
                Span::styled(" [/] or ←→    ", Style::default().fg(colors::HELP_KEY)),
                Span::raw("Cycle charts/views/concepts within tabs"),
            ]),
            Line::from(Span::raw("")),
            Line::from(Span::styled(
                " TABS",
                Style::default()
                    .fg(colors::WARMUP)
                    .add_modifier(Modifier::BOLD),
            )),
            Line::from(vec![
                Span::styled(" 0 Dashboard  ", Style::default().fg(colors::HELP_KEY)),
                Span::raw("Unified view with key metrics & curve quality"),
            ]),
            Line::from(vec![
                Span::styled(" 1 Overview   ", Style::default().fg(colors::HELP_KEY)),
                Span::raw("Run status, metrics, phase breakdown"),
            ]),
            Line::from(vec![
                Span::styled(" 2 Charts     ", Style::default().fg(colors::HELP_KEY)),
                Span::raw("Loss, gradients, step time, predictions"),
            ]),
            Line::from(vec![
                Span::styled(" 3 Analysis   ", Style::default().fg(colors::HELP_KEY)),
                Span::raw("LR recommendations & training diagnostics"),
            ]),
            Line::from(vec![
                Span::styled(" 4 Network    ", Style::default().fg(colors::HELP_KEY)),
                Span::raw("Layer flow visualization"),
            ]),
            Line::from(vec![
                Span::styled(" 5 Concepts   ", Style::default().fg(colors::HELP_KEY)),
                Span::raw("Interactive concept cloud"),
            ]),
            Line::from(vec![
                Span::styled(" 6 GPU        ", Style::default().fg(colors::HELP_KEY)),
                Span::raw("Memory, thermal, utilization, flame graph"),
            ]),
            Line::from(vec![
                Span::styled(" 7 History    ", Style::default().fg(colors::HELP_KEY)),
                Span::raw("Past runs and checkpoints"),
            ]),
            Line::from(Span::raw("")),
            Line::from(Span::styled(
                " CURVE QUALITY INDICATORS",
                Style::default()
                    .fg(colors::WARMUP)
                    .add_modifier(Modifier::BOLD),
            )),
            Line::from(vec![
                Span::styled(" ✓ ", Style::default().fg(colors::PREDICT)),
                Span::raw("Healthy - Normal descent, training on track"),
            ]),
            Line::from(vec![
                Span::styled(" ⚠ ", Style::default().fg(colors::MEMORY_WARN)),
                Span::raw("Too Slow - Underfitting, increase LR/batch size"),
            ]),
            Line::from(vec![
                Span::styled(" ⚡ ", Style::default().fg(colors::WARMUP)),
                Span::raw("Too Fast - Overfitting risk, reduce LR"),
            ]),
            Line::from(vec![
                Span::styled(" ~ ", Style::default().fg(colors::MEMORY_WARN)),
                Span::raw("Oscillating - LR too high, reduce significantly"),
            ]),
            Line::from(vec![
                Span::styled(" ─ ", Style::default().fg(Color::Gray)),
                Span::raw("Plateau - Loss stuck, needs intervention"),
            ]),
            Line::from(vec![
                Span::styled(" ✗ ", Style::default().fg(colors::FAILED_RUN)),
                Span::raw("Diverging - Training failing, stop & reduce LR 10x"),
            ]),
            Line::from(Span::raw("")),
            Line::from(Span::styled(
                " ACTIONS",
                Style::default()
                    .fg(colors::WARMUP)
                    .add_modifier(Modifier::BOLD),
            )),
            Line::from(vec![
                Span::styled(" +/-          ", Style::default().fg(colors::HELP_KEY)),
                Span::raw("Zoom in/out (trailing steps)"),
            ]),
            Line::from(vec![
                Span::styled(" f            ", Style::default().fg(colors::HELP_KEY)),
                Span::raw("Full view (reset zoom)"),
            ]),
            Line::from(vec![
                Span::styled(" l            ", Style::default().fg(colors::HELP_KEY)),
                Span::raw("Toggle LIVE ↔ HISTORY mode"),
            ]),
            Line::from(vec![
                Span::styled(" r            ", Style::default().fg(colors::HELP_KEY)),
                Span::raw("Refresh / rediscover runs"),
            ]),
            Line::from(vec![
                Span::styled(" c            ", Style::default().fg(colors::HELP_KEY)),
                Span::raw("Clear cached metrics for selected run"),
            ]),
            Line::from(vec![
                Span::styled(" ? or F1      ", Style::default().fg(colors::HELP_KEY)),
                Span::raw("Show/hide this help overlay"),
            ]),
            Line::from(vec![
                Span::styled(" q or Esc     ", Style::default().fg(colors::HELP_KEY)),
                Span::raw("Quit monitor"),
            ]),
            Line::from(Span::raw("")),
            Line::from(Span::styled(
                " Press ? or Esc to close",
                Style::default().fg(Color::Gray),
            )),
        ];

        let popup = Paragraph::new(help_text).block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(colors::TAB_ACTIVE))
                .style(Style::default().bg(Color::Rgb(30, 30, 50)))
                .title(Span::styled(
                    " Help ",
                    Style::default().fg(colors::TAB_ACTIVE),
                )),
        );

        f.render_widget(popup, popup_area);
    }

    fn draw_footer(&self, f: &mut Frame, area: Rect) {
        let mode_str = match self.view_mode {
            ViewMode::Live => "●LIVE",
            ViewMode::History => "◆HIST",
        };
        let mode_color = match self.view_mode {
            ViewMode::Live => colors::LIVE_INDICATOR,
            ViewMode::History => colors::COMPLETED_RUN,
        };

        let footer = Paragraph::new(Line::from(vec![
            Span::styled(format!(" [{}] ", mode_str), Style::default().fg(mode_color)),
            Span::styled("Tab", Style::default().fg(colors::HELP_KEY)),
            Span::raw(":tabs  "),
            Span::styled("j/k", Style::default().fg(colors::HELP_KEY)),
            Span::raw(":runs  "),
            Span::styled("[/]", Style::default().fg(colors::HELP_KEY)),
            Span::raw(":charts  "),
            Span::styled("l", Style::default().fg(colors::HELP_KEY)),
            Span::raw(":mode  "),
            Span::styled("?", Style::default().fg(colors::HELP_KEY)),
            Span::raw(":help  "),
            Span::styled("q", Style::default().fg(colors::HELP_KEY)),
            Span::raw(":quit"),
        ]))
        .style(Style::default().bg(colors::HEADER_BG));

        f.render_widget(footer, area);
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

/// Format axis values smartly (handles both large and small numbers).
fn format_axis_value(val: f64) -> String {
    if val.abs() >= 1000.0 {
        format!("{:.1}K", val / 1000.0)
    } else if val.abs() >= 100.0 {
        format!("{:.0}", val)
    } else if val.abs() >= 10.0 {
        format!("{:.1}", val)
    } else if val.abs() >= 1.0 {
        format!("{:.2}", val)
    } else if val.abs() >= 0.01 {
        format!("{:.3}", val)
    } else if val.abs() == 0.0 {
        "0".to_string()
    } else {
        format!("{:.2e}", val)
    }
}

/// Generate a sparkline from data using unicode block characters.
fn generate_sparkline(data: &[f32]) -> String {
    if data.is_empty() {
        return String::new();
    }

    let chars = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let range = (max - min).max(0.0001);

    data.iter()
        .map(|&v| {
            let normalized = ((v - min) / range).clamp(0.0, 1.0);
            let idx = (normalized * (chars.len() - 1) as f32) as usize;
            chars[idx]
        })
        .collect()
}

/// Generate a progress bar with filled/unfilled blocks.
fn generate_progress_bar(percent: f64, width: usize) -> String {
    let filled = ((percent / 100.0) * width as f64) as usize;
    let filled = filled.min(width);
    let empty = width.saturating_sub(filled);

    format!("{}{}", "█".repeat(filled), "░".repeat(empty))
}

/// Apply chart range to data vector.
fn apply_range(data: Vec<(f64, f64)>, range: ChartRange) -> Vec<(f64, f64)> {
    match range {
        ChartRange::Full => data,
        ChartRange::Trailing(n) => {
            if data.len() <= n {
                data
            } else {
                data[data.len() - n..].to_vec()
            }
        }
    }
}

/// Convert intensity (0.0-1.0) to a color gradient (cool to hot).
fn intensity_to_color(intensity: f64) -> Color {
    let i = intensity.clamp(0.0, 1.0);
    if i < 0.33 {
        // Cool (blue-ish)
        let t = i / 0.33;
        Color::Rgb(
            (50.0 + t * 50.0) as u8,
            (100.0 + t * 50.0) as u8,
            (200.0 - t * 50.0) as u8,
        )
    } else if i < 0.66 {
        // Warm (yellow-ish)
        let t = (i - 0.33) / 0.33;
        Color::Rgb(
            (100.0 + t * 155.0) as u8,
            (150.0 + t * 105.0) as u8,
            (150.0 - t * 100.0) as u8,
        )
    } else {
        // Hot (red-ish)
        let t = (i - 0.66) / 0.34;
        Color::Rgb(255, (255.0 - t * 155.0) as u8, (50.0 - t * 50.0) as u8)
    }
}

/// Create a mini sparkline from values using Unicode block characters.
fn create_sparkline(values: &[f64]) -> String {
    if values.is_empty() {
        return String::new();
    }

    let min = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let range = (max - min).max(0.001);

    // Unicode block characters for sparkline (from lowest to highest)
    const BLOCKS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

    values
        .iter()
        .map(|&v| {
            let normalized = ((v - min) / range).clamp(0.0, 0.999);
            let idx = (normalized * 8.0) as usize;
            BLOCKS[idx]
        })
        .collect()
}
