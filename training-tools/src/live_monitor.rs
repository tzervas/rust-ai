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

    // Chart colors
    pub const LOSS_LINE: Color = Color::Rgb(0, 255, 255);      // Bright cyan
    pub const LOSS_SCATTER: Color = Color::Rgb(255, 255, 0);   // Bright yellow dots
    pub const TREND_LINE: Color = Color::Rgb(255, 100, 255);   // Magenta trend
    pub const GRAD_NORM: Color = Color::Rgb(100, 255, 100);    // Green
    pub const STEP_TIME: Color = Color::Rgb(255, 180, 50);     // Orange
    pub const PREDICTION: Color = Color::Rgb(100, 200, 255);   // Light blue

    // Phase colors
    pub const WARMUP: Color = Color::Rgb(255, 200, 0);         // Orange
    pub const FULL: Color = Color::Rgb(0, 150, 255);           // Blue
    pub const PREDICT: Color = Color::Rgb(0, 255, 100);        // Green
    pub const CORRECT: Color = Color::Rgb(255, 100, 255);      // Magenta

    // Status colors
    pub const ACTIVE_RUN: Color = Color::Rgb(0, 255, 0);       // Bright green
    pub const COMPLETED_RUN: Color = Color::Rgb(100, 200, 255);// Light blue
    pub const FAILED_RUN: Color = Color::Rgb(255, 80, 80);     // Red
    pub const PAUSED_RUN: Color = Color::Rgb(255, 255, 0);     // Yellow

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
    Overview,
    Charts,
    GPU,
    History,
    Help,
}

impl MainTab {
    fn all() -> &'static [MainTab] {
        &[MainTab::Overview, MainTab::Charts, MainTab::GPU, MainTab::History, MainTab::Help]
    }

    fn title(&self) -> &'static str {
        match self {
            MainTab::Overview => "Overview",
            MainTab::Charts => "Charts",
            MainTab::GPU => "GPU",
            MainTab::History => "History",
            MainTab::Help => "Help",
        }
    }

    fn index(&self) -> usize {
        match self {
            MainTab::Overview => 0,
            MainTab::Charts => 1,
            MainTab::GPU => 2,
            MainTab::History => 3,
            MainTab::Help => 4,
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
            KeyCode::Char('3') => self.main_tab = MainTab::GPU,
            KeyCode::Char('4') => self.main_tab = MainTab::History,
            KeyCode::Char('5') => self.main_tab = MainTab::Help,

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

            // Chart type cycling (within Charts tab)
            KeyCode::Left | KeyCode::Char('[') => {
                if self.main_tab == MainTab::Charts {
                    let types = ChartType::all();
                    let idx = types.iter().position(|t| *t == self.chart_type).unwrap_or(0);
                    self.chart_type = types[(idx + types.len() - 1) % types.len()];
                } else if self.main_tab == MainTab::GPU {
                    let views = GpuView::all();
                    let idx = views.iter().position(|v| *v == self.gpu_view).unwrap_or(0);
                    self.gpu_view = views[(idx + views.len() - 1) % views.len()];
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
                    Style::default().fg(colors::TAB_ACTIVE).add_modifier(Modifier::BOLD)
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
                        Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
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

        // Gradient norm
        let grad_norm = latest.map(|m| m.gradient_norm).unwrap_or(0.0);
        let confidence = latest.map(|m| m.confidence).unwrap_or(0.0);

        let stats = vec![
            Line::from(vec![
                Span::styled(" Loss: ", Style::default().fg(Color::Gray)),
                Span::styled(format!("{:.4}", loss), Style::default().fg(colors::LOSS_LINE).add_modifier(Modifier::BOLD)),
                Span::styled(format!("  {}", loss_trend), Style::default().fg(trend_color)),
            ]),
            Line::from(vec![
                Span::styled(" Best: ", Style::default().fg(Color::Gray)),
                Span::styled(format!("{:.4}", run.best_loss), Style::default().fg(colors::PREDICT)),
                Span::styled(format!(" @ step {}", run.best_step), Style::default().fg(Color::Gray)),
            ]),
            Line::from(vec![
                Span::styled(" Speed: ", Style::default().fg(Color::Gray)),
                Span::styled(format!("{:.2} steps/s", steps_per_sec), Style::default().fg(Color::White)),
                Span::styled("  │  ETA: ", Style::default().fg(Color::Gray)),
                Span::styled(eta_str, Style::default().fg(colors::WARMUP)),
            ]),
            Line::from(vec![
                Span::styled(" Elapsed: ", Style::default().fg(Color::Gray)),
                Span::styled(elapsed_str, Style::default().fg(Color::White)),
                Span::styled("  │  Backward: ", Style::default().fg(Color::Gray)),
                Span::styled(
                    format!("{:.1}% reduction", run.backward_reduction()),
                    Style::default().fg(colors::PREDICT),
                ),
            ]),
            Line::from(vec![
                Span::styled(" Tokens: ", Style::default().fg(Color::Gray)),
                Span::styled(format_tokens(tokens_trained), Style::default().fg(colors::WARMUP)),
                Span::styled(" trained  │  ", Style::default().fg(Color::Gray)),
                Span::styled(format_tokens(tokens_remaining), Style::default().fg(Color::White)),
                Span::styled(" remaining", Style::default().fg(Color::Gray)),
            ]),
            Line::from(vec![
                Span::styled(" Throughput: ", Style::default().fg(Color::Gray)),
                Span::styled(format!("{}/s", format_tokens(tokens_per_sec as u64)), Style::default().fg(colors::STEP_TIME)),
                Span::styled(format!("  │  {} tok/step", format_tokens(tokens_per_step)), Style::default().fg(Color::Gray)),
            ]),
            Line::from(vec![
                Span::styled(" Grad Norm: ", Style::default().fg(Color::Gray)),
                Span::styled(format!("{:.4}", grad_norm), Style::default().fg(colors::GRAD_NORM)),
                Span::styled("  │  Confidence: ", Style::default().fg(Color::Gray)),
                Span::styled(format!("{:.1}%", confidence * 100.0), Style::default().fg(colors::PREDICTION)),
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
            .constraints([Constraint::Length(3), Constraint::Min(10)])
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
                    .title(" Chart Type [←/→] "),
            );

        f.render_widget(selector, chunks[0]);

        let reader = self.selected_run_id.as_ref().and_then(|id| self.metrics_readers.get(id));

        if let Some(reader) = reader {
            match self.chart_type {
                ChartType::LossLine => self.draw_loss_line_chart(f, chunks[1], reader, "Loss (Line)"),
                ChartType::LossScatter => self.draw_scatter_with_trend(f, chunks[1], reader),
                ChartType::GradientNorm => self.draw_gradient_chart(f, chunks[1], reader),
                ChartType::StepTime => self.draw_step_time_chart(f, chunks[1], reader),
                ChartType::PhaseBreakdown => self.draw_phase_breakdown(f, chunks[1], reader),
                ChartType::PredictionAccuracy => self.draw_prediction_chart(f, chunks[1], reader),
            }
        } else {
            let msg = Paragraph::new("Select a run to view charts")
                .style(Style::default().fg(Color::Gray))
                .block(Block::default().borders(Borders::ALL));
            f.render_widget(msg, chunks[1]);
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
        let popup_width = 60.min(area.width - 4);
        let popup_height = 20.min(area.height - 4);

        let popup_area = Rect {
            x: (area.width - popup_width) / 2,
            y: (area.height - popup_height) / 2,
            width: popup_width,
            height: popup_height,
        };

        // Clear background
        let clear = Paragraph::new("").style(Style::default().bg(Color::Rgb(20, 20, 30)));
        f.render_widget(clear, popup_area);

        let help_text = vec![
            Line::from(Span::styled(" Quick Reference", Style::default().fg(Color::White).add_modifier(Modifier::BOLD))),
            Line::from(Span::raw("")),
            Line::from(vec![
                Span::styled(" Tab      ", Style::default().fg(colors::HELP_KEY)),
                Span::raw("Cycle tabs"),
            ]),
            Line::from(vec![
                Span::styled(" 1-5      ", Style::default().fg(colors::HELP_KEY)),
                Span::raw("Jump to tab"),
            ]),
            Line::from(vec![
                Span::styled(" j/k ↑↓   ", Style::default().fg(colors::HELP_KEY)),
                Span::raw("Navigate runs"),
            ]),
            Line::from(vec![
                Span::styled(" [/] ←→   ", Style::default().fg(colors::HELP_KEY)),
                Span::raw("Cycle sub-views"),
            ]),
            Line::from(vec![
                Span::styled(" l        ", Style::default().fg(colors::HELP_KEY)),
                Span::raw("Toggle LIVE/HISTORY"),
            ]),
            Line::from(vec![
                Span::styled(" r        ", Style::default().fg(colors::HELP_KEY)),
                Span::raw("Refresh"),
            ]),
            Line::from(vec![
                Span::styled(" c        ", Style::default().fg(colors::HELP_KEY)),
                Span::raw("Clear metrics"),
            ]),
            Line::from(vec![
                Span::styled(" q/Esc    ", Style::default().fg(colors::HELP_KEY)),
                Span::raw("Quit"),
            ]),
            Line::from(Span::raw("")),
            Line::from(Span::styled(" Press ? or Esc to close", Style::default().fg(Color::Gray))),
        ];

        let popup = Paragraph::new(help_text).block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(colors::TAB_ACTIVE))
                .style(Style::default().bg(Color::Rgb(30, 30, 50)))
                .title(Span::styled(" Help ", Style::default().fg(colors::TAB_ACTIVE))),
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
