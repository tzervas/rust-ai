//! Real-time training monitor with TUI.
//!
//! Provides a terminal UI for monitoring active training runs,
//! displaying loss curves, phase information, GPU stats, memory usage, and more.
//!
//! # Views
//!
//! - **Unified**: All metrics on one screen
//! - **Loss**: Detailed loss metrics and chart
//! - **GPU**: GPU utilization, memory, thermals, power
//! - **Timing**: Elapsed time, ETA, checkpoints
//! - **Phase**: Training phase details

use std::io::{self, Stdout};
use std::path::Path;
use std::time::Duration;

use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    symbols,
    text::{Line, Span},
    widgets::{
        Axis, Block, Borders, Chart, Dataset, Gauge, List, ListItem, Paragraph, Row, Sparkline,
        Table, Tabs,
    },
    Frame, Terminal,
};

use crate::gpu_stats::{GpuStats, GpuStatsMonitor};
use crate::training_state::{RunManager, StepMetrics, TrainingPhase, TrainingRun, TrainingStatus};

/// View mode for the monitor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViewMode {
    /// All metrics unified
    Unified,
    /// Detailed loss view
    Loss,
    /// GPU stats view
    Gpu,
    /// Timing and ETA view
    Timing,
    /// Phase details view
    Phase,
}

impl ViewMode {
    fn title(&self) -> &'static str {
        match self {
            Self::Unified => "Unified",
            Self::Loss => "Loss",
            Self::Gpu => "GPU",
            Self::Timing => "Timing",
            Self::Phase => "Phase",
        }
    }

    fn all() -> &'static [ViewMode] {
        &[
            ViewMode::Unified,
            ViewMode::Loss,
            ViewMode::Gpu,
            ViewMode::Timing,
            ViewMode::Phase,
        ]
    }

    fn index(&self) -> usize {
        match self {
            Self::Unified => 0,
            Self::Loss => 1,
            Self::Gpu => 2,
            Self::Timing => 3,
            Self::Phase => 4,
        }
    }

    fn from_index(idx: usize) -> Self {
        match idx {
            0 => Self::Unified,
            1 => Self::Loss,
            2 => Self::Gpu,
            3 => Self::Timing,
            4 => Self::Phase,
            _ => Self::Unified,
        }
    }
}

/// Training monitor configuration.
#[derive(Debug, Clone)]
pub struct MonitorConfig {
    /// Refresh interval in milliseconds
    pub refresh_ms: u64,
    /// Number of loss history points to display
    pub loss_history_size: usize,
    /// Show detailed metrics
    pub show_details: bool,
}

impl Default for MonitorConfig {
    fn default() -> Self {
        Self {
            refresh_ms: 500,
            loss_history_size: 100,
            show_details: true,
        }
    }
}

/// Real-time training monitor.
pub struct TrainingMonitor {
    config: MonitorConfig,
    run_manager: RunManager,
    selected_run: Option<String>,
    loss_history: Vec<(f64, f64)>,
    should_quit: bool,
    view_mode: ViewMode,
    gpu_monitor: GpuStatsMonitor,
    all_metrics: Vec<StepMetrics>,
}

impl TrainingMonitor {
    /// Create a new training monitor.
    pub fn new(runs_dir: &Path, config: MonitorConfig) -> Self {
        Self {
            config,
            run_manager: RunManager::new(runs_dir.to_path_buf()),
            selected_run: None,
            loss_history: Vec::new(),
            should_quit: false,
            view_mode: ViewMode::Unified,
            gpu_monitor: GpuStatsMonitor::new(0),
            all_metrics: Vec::new(),
        }
    }

    /// Run the monitor TUI.
    pub fn run(&mut self) -> anyhow::Result<()> {
        // Setup terminal
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen)?;
        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend)?;

        // Initial discovery
        self.run_manager.discover_runs()?;

        // Select first active run if any
        if let Some(run) = self.run_manager.active_runs().next() {
            self.selected_run = Some(run.run_id.clone());
        }

        // Main loop
        let result = self.main_loop(&mut terminal);

        // Restore terminal
        disable_raw_mode()?;
        execute!(terminal.backend_mut(), LeaveAlternateScreen)?;

        result
    }

    fn main_loop(
        &mut self,
        terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    ) -> anyhow::Result<()> {
        while !self.should_quit {
            // Refresh runs
            self.run_manager.discover_runs()?;

            // Sample GPU stats
            self.gpu_monitor.sample();

            // Update loss history for selected run
            if let Some(ref run_id) = self.selected_run {
                if let Some(run) = self.run_manager.get_run(run_id) {
                    if let Ok(metrics) = run.read_recent_metrics(self.config.loss_history_size) {
                        self.all_metrics = metrics.clone();
                        self.loss_history = metrics
                            .iter()
                            .map(|m| (m.step as f64, m.loss as f64))
                            .collect();
                    }
                }
            }

            // Draw UI
            terminal.draw(|f| self.draw(f))?;

            // Handle input
            if event::poll(Duration::from_millis(self.config.refresh_ms))? {
                if let Event::Key(key) = event::read()? {
                    if key.kind == KeyEventKind::Press {
                        self.handle_key(key.code);
                    }
                }
            }
        }

        Ok(())
    }

    fn handle_key(&mut self, key: KeyCode) {
        match key {
            KeyCode::Char('q') | KeyCode::Esc => self.should_quit = true,
            KeyCode::Up | KeyCode::Char('k') => self.select_previous_run(),
            KeyCode::Down | KeyCode::Char('j') => self.select_next_run(),
            KeyCode::Char('r') => {
                let _ = self.run_manager.discover_runs();
            }
            KeyCode::Char('d') => self.config.show_details = !self.config.show_details,
            // View mode switching
            KeyCode::Char('1') => self.view_mode = ViewMode::Unified,
            KeyCode::Char('2') => self.view_mode = ViewMode::Loss,
            KeyCode::Char('3') => self.view_mode = ViewMode::Gpu,
            KeyCode::Char('4') => self.view_mode = ViewMode::Timing,
            KeyCode::Char('5') => self.view_mode = ViewMode::Phase,
            KeyCode::Tab => {
                let next = (self.view_mode.index() + 1) % ViewMode::all().len();
                self.view_mode = ViewMode::from_index(next);
            }
            KeyCode::BackTab => {
                let prev = if self.view_mode.index() == 0 {
                    ViewMode::all().len() - 1
                } else {
                    self.view_mode.index() - 1
                };
                self.view_mode = ViewMode::from_index(prev);
            }
            _ => {}
        }
    }

    fn select_previous_run(&mut self) {
        let runs: Vec<_> = self.run_manager.runs().collect();
        if runs.is_empty() {
            return;
        }

        let current_idx = self
            .selected_run
            .as_ref()
            .and_then(|id| runs.iter().position(|r| &r.run_id == id))
            .unwrap_or(0);

        let new_idx = if current_idx == 0 {
            runs.len() - 1
        } else {
            current_idx - 1
        };

        self.selected_run = Some(runs[new_idx].run_id.clone());
    }

    fn select_next_run(&mut self) {
        let runs: Vec<_> = self.run_manager.runs().collect();
        if runs.is_empty() {
            return;
        }

        let current_idx = self
            .selected_run
            .as_ref()
            .and_then(|id| runs.iter().position(|r| &r.run_id == id))
            .unwrap_or(0);

        let new_idx = (current_idx + 1) % runs.len();
        self.selected_run = Some(runs[new_idx].run_id.clone());
    }

    fn draw(&self, f: &mut Frame) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3), // Header + tabs
                Constraint::Min(10),   // Main content
                Constraint::Length(3), // Footer
            ])
            .split(f.area());

        self.draw_header(f, chunks[0]);
        self.draw_main(f, chunks[1]);
        self.draw_footer(f, chunks[2]);
    }

    fn draw_header(&self, f: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(area);

        // Title and status
        let active_count = self.run_manager.active_runs().count();
        let total_count = self.run_manager.runs().count();

        let header = Paragraph::new(vec![Line::from(vec![
            Span::styled(
                " RUST-AI TRAINING MONITOR ",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw(" | "),
            Span::styled(
                format!("{} active", active_count),
                Style::default().fg(if active_count > 0 {
                    Color::Green
                } else {
                    Color::Gray
                }),
            ),
            Span::raw(" / "),
            Span::raw(format!("{} total", total_count)),
        ])])
        .block(Block::default().borders(Borders::ALL));

        f.render_widget(header, chunks[0]);

        // View tabs
        let titles: Vec<Line> = ViewMode::all()
            .iter()
            .enumerate()
            .map(|(i, mode)| {
                Line::from(Span::styled(
                    format!(" {} {} ", i + 1, mode.title()),
                    if *mode == self.view_mode {
                        Style::default()
                            .fg(Color::Cyan)
                            .add_modifier(Modifier::BOLD)
                    } else {
                        Style::default().fg(Color::Gray)
                    },
                ))
            })
            .collect();

        let tabs = Tabs::new(titles)
            .block(Block::default().borders(Borders::ALL).title(" Views "))
            .select(self.view_mode.index())
            .highlight_style(Style::default().fg(Color::Cyan));

        f.render_widget(tabs, chunks[1]);
    }

    fn draw_main(&self, f: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Length(30), Constraint::Min(50)])
            .split(area);

        self.draw_run_list(f, chunks[0]);
        self.draw_run_details(f, chunks[1]);
    }

    fn draw_run_list(&self, f: &mut Frame, area: Rect) {
        let runs: Vec<_> = self.run_manager.runs().collect();

        let items: Vec<ListItem> = runs
            .iter()
            .map(|run| {
                let status_color = match run.status {
                    TrainingStatus::Running => Color::Green,
                    TrainingStatus::Paused => Color::Yellow,
                    TrainingStatus::Completed => Color::Cyan,
                    TrainingStatus::Failed => Color::Red,
                    TrainingStatus::Cancelled => Color::Gray,
                    TrainingStatus::Initializing => Color::Blue,
                };

                let selected = self
                    .selected_run
                    .as_ref()
                    .map(|id| id == &run.run_id)
                    .unwrap_or(false);

                let style = if selected {
                    Style::default()
                        .fg(Color::White)
                        .bg(Color::DarkGray)
                        .add_modifier(Modifier::BOLD)
                } else {
                    Style::default()
                };

                let status_icon = match run.status {
                    TrainingStatus::Running => ">>",
                    TrainingStatus::Paused => "||",
                    TrainingStatus::Completed => "OK",
                    TrainingStatus::Failed => "XX",
                    TrainingStatus::Cancelled => "--",
                    TrainingStatus::Initializing => "..",
                };

                let content = format!("{} {} ({:.1}%)", status_icon, run.run_name, run.progress());

                ListItem::new(Line::from(vec![Span::styled(
                    content,
                    style.fg(status_color),
                )]))
            })
            .collect();

        let list = List::new(items).block(Block::default().borders(Borders::ALL).title(" Runs "));

        f.render_widget(list, area);
    }

    fn draw_run_details(&self, f: &mut Frame, area: Rect) {
        let selected_run = self
            .selected_run
            .as_ref()
            .and_then(|id| self.run_manager.get_run(id));

        match selected_run {
            Some(run) => match self.view_mode {
                ViewMode::Unified => self.draw_unified_view(f, area, run),
                ViewMode::Loss => self.draw_loss_view(f, area, run),
                ViewMode::Gpu => self.draw_gpu_view(f, area, run),
                ViewMode::Timing => self.draw_timing_view(f, area, run),
                ViewMode::Phase => self.draw_phase_view(f, area, run),
            },
            None => {
                let msg = Paragraph::new("No run selected. Use j/k to navigate.")
                    .block(Block::default().borders(Borders::ALL).title(" Details "));
                f.render_widget(msg, area);
            }
        }
    }

    fn draw_unified_view(&self, f: &mut Frame, area: Rect, run: &TrainingRun) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3), // Progress bar
                Constraint::Length(6), // Stats
                Constraint::Min(8),    // Loss chart
                Constraint::Length(4), // GPU quick stats
                Constraint::Length(3), // Timing
            ])
            .split(area);

        self.draw_progress_bar(f, chunks[0], run);
        self.draw_quick_stats(f, chunks[1], run);
        self.draw_loss_chart(f, chunks[2]);
        self.draw_gpu_quick(f, chunks[3]);
        self.draw_timing_quick(f, chunks[4], run);
    }

    fn draw_loss_view(&self, f: &mut Frame, area: Rect, run: &TrainingRun) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),  // Progress
                Constraint::Length(10), // Loss stats
                Constraint::Min(10),    // Large chart
                Constraint::Length(4),  // Sparkline
            ])
            .split(area);

        self.draw_progress_bar(f, chunks[0], run);

        // Detailed loss stats
        let loss_stats = self.format_loss_stats(run);
        let stats_widget = Paragraph::new(loss_stats).block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Loss Statistics "),
        );
        f.render_widget(stats_widget, chunks[1]);

        self.draw_loss_chart(f, chunks[2]);

        // Loss sparkline
        if !self.loss_history.is_empty() {
            let spark_data: Vec<u64> = self
                .loss_history
                .iter()
                .map(|(_, l)| (l * 100.0) as u64)
                .collect();
            let sparkline = Sparkline::default()
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .title(" Recent Loss Trend "),
                )
                .data(&spark_data)
                .style(Style::default().fg(Color::Cyan));
            f.render_widget(sparkline, chunks[3]);
        }
    }

    fn draw_gpu_view(&self, f: &mut Frame, area: Rect, run: &TrainingRun) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),  // Progress
                Constraint::Length(12), // GPU info
                Constraint::Min(8),     // Memory gauge
                Constraint::Length(8),  // Thermals and power
            ])
            .split(area);

        self.draw_progress_bar(f, chunks[0], run);

        // GPU info table
        if let Some(stats) = self.gpu_monitor.current() {
            let gpu_info = self.format_gpu_info(stats);
            let info_widget = Paragraph::new(gpu_info).block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(format!(" {} ", stats.name)),
            );
            f.render_widget(info_widget, chunks[1]);

            // Memory usage gauge
            let mem_percent = stats.memory_percent() / 100.0;
            let mem_color = if stats.is_memory_pressure() {
                Color::Red
            } else if mem_percent > 0.7 {
                Color::Yellow
            } else {
                Color::Green
            };

            let mem_gauge = Gauge::default()
                .block(Block::default().borders(Borders::ALL).title(" VRAM Usage "))
                .gauge_style(Style::default().fg(mem_color))
                .ratio(mem_percent.clamp(0.0, 1.0))
                .label(format!(
                    "{:.1} / {:.1} GB ({:.0}%) [Peak: {:.1} GB]",
                    stats.memory_used as f64 / 1e9,
                    stats.memory_total as f64 / 1e9,
                    stats.memory_percent(),
                    self.gpu_monitor.peak_memory() as f64 / 1e9
                ));
            f.render_widget(mem_gauge, chunks[2]);

            // Thermals and power
            let thermal_info = self.format_thermal_power(stats);
            let thermal_widget = Paragraph::new(thermal_info).block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(" Thermals & Power "),
            );
            f.render_widget(thermal_widget, chunks[3]);
        } else {
            let no_gpu = Paragraph::new("GPU stats unavailable - nvidia-smi not responding")
                .block(Block::default().borders(Borders::ALL).title(" GPU Info "));
            f.render_widget(no_gpu, chunks[1]);
        }
    }

    fn draw_timing_view(&self, f: &mut Frame, area: Rect, run: &TrainingRun) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),  // Progress
                Constraint::Length(10), // Timing stats
                Constraint::Min(6),     // Checkpoints table
                Constraint::Length(5),  // Run timestamps
            ])
            .split(area);

        self.draw_progress_bar(f, chunks[0], run);

        // Timing stats
        let timing_info = self.format_timing_stats(run);
        let timing_widget = Paragraph::new(timing_info)
            .block(Block::default().borders(Borders::ALL).title(" Timing "));
        f.render_widget(timing_widget, chunks[1]);

        // Checkpoints table
        self.draw_checkpoint_table(f, chunks[2], run);

        // Run timestamps
        let timestamps = self.format_run_timestamps(run);
        let ts_widget = Paragraph::new(timestamps).block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Run Timestamps "),
        );
        f.render_widget(ts_widget, chunks[3]);
    }

    fn draw_phase_view(&self, f: &mut Frame, area: Rect, run: &TrainingRun) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3), // Progress
                Constraint::Length(8), // Current phase
                Constraint::Min(8),    // Phase history
                Constraint::Length(6), // Phase stats
            ])
            .split(area);

        self.draw_progress_bar(f, chunks[0], run);

        // Current phase detail
        let phase_detail = self.format_phase_detail(run);
        let phase_widget = Paragraph::new(phase_detail).block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Current Phase "),
        );
        f.render_widget(phase_widget, chunks[1]);

        // Phase transitions table
        self.draw_phase_transitions(f, chunks[2], run);

        // Phase statistics
        let phase_stats = self.format_phase_stats(run);
        let stats_widget = Paragraph::new(phase_stats).block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Phase Statistics "),
        );
        f.render_widget(stats_widget, chunks[3]);
    }

    fn draw_progress_bar(&self, f: &mut Frame, area: Rect, run: &TrainingRun) {
        let progress = run.progress() / 100.0;
        let gauge = Gauge::default()
            .block(Block::default().borders(Borders::ALL).title(format!(
                " {} - Step {}/{} ",
                run.run_name, run.current_step, run.config.max_steps
            )))
            .gauge_style(Style::default().fg(Color::Cyan))
            .ratio(progress.clamp(0.0, 1.0))
            .label(format!("{:.1}%", run.progress()));

        f.render_widget(gauge, area);
    }

    fn draw_quick_stats(&self, f: &mut Frame, area: Rect, run: &TrainingRun) {
        let elapsed = run.elapsed();
        let elapsed_str = format!(
            "{}:{:02}:{:02}",
            elapsed.num_hours(),
            elapsed.num_minutes() % 60,
            elapsed.num_seconds() % 60
        );

        let phase_color = match run.current_phase {
            TrainingPhase::Warmup => Color::Yellow,
            TrainingPhase::Full => Color::Blue,
            TrainingPhase::Predict => Color::Green,
            TrainingPhase::Correct => Color::Magenta,
        };

        let stats_text = vec![
            Line::from(vec![
                Span::raw(" Model: "),
                Span::styled(
                    format!(
                        "{} ({:.1}M params)",
                        run.config.model_size,
                        run.config.num_parameters as f64 / 1e6
                    ),
                    Style::default().add_modifier(Modifier::BOLD),
                ),
            ]),
            Line::from(vec![
                Span::raw(" Loss: "),
                Span::styled(
                    format!("{:.4}", run.current_loss),
                    Style::default().fg(Color::Cyan),
                ),
                Span::raw(" (best: "),
                Span::styled(
                    format!("{:.4}", run.best_loss),
                    Style::default().fg(Color::Green),
                ),
                Span::raw(format!(" @ step {})", run.best_step)),
            ]),
            Line::from(vec![
                Span::raw(" Phase: "),
                Span::styled(
                    run.current_phase.to_string(),
                    Style::default()
                        .fg(phase_color)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::raw(" | Bwd reduction: "),
                Span::styled(
                    format!("{:.1}%", run.backward_reduction()),
                    Style::default().fg(Color::Green),
                ),
            ]),
            Line::from(vec![
                Span::raw(" Elapsed: "),
                Span::raw(elapsed_str),
                Span::raw(" | ETA: "),
                Span::styled(run.eta_string(), Style::default().fg(Color::Yellow)),
            ]),
        ];

        let stats = Paragraph::new(stats_text)
            .block(Block::default().borders(Borders::ALL).title(" Statistics "));

        f.render_widget(stats, area);
    }

    fn draw_loss_chart(&self, f: &mut Frame, area: Rect) {
        if self.loss_history.is_empty() {
            let msg = Paragraph::new("Waiting for data...")
                .block(Block::default().borders(Borders::ALL).title(" Loss "));
            f.render_widget(msg, area);
            return;
        }

        let min_loss = self
            .loss_history
            .iter()
            .map(|(_, l)| *l)
            .fold(f64::INFINITY, f64::min);
        let max_loss = self
            .loss_history
            .iter()
            .map(|(_, l)| *l)
            .fold(f64::NEG_INFINITY, f64::max);
        let min_step = self.loss_history.first().map(|(s, _)| *s).unwrap_or(0.0);
        let max_step = self.loss_history.last().map(|(s, _)| *s).unwrap_or(1.0);

        let y_margin = (max_loss - min_loss) * 0.1;
        let y_min = (min_loss - y_margin).max(0.0);
        let y_max = max_loss + y_margin;

        let datasets = vec![Dataset::default()
            .name("Loss")
            .marker(symbols::Marker::Braille)
            .style(Style::default().fg(Color::Cyan))
            .data(&self.loss_history)];

        let chart = Chart::new(datasets)
            .block(Block::default().borders(Borders::ALL).title(" Loss Curve "))
            .x_axis(
                Axis::default()
                    .title("Step")
                    .style(Style::default().fg(Color::Gray))
                    .bounds([min_step, max_step]),
            )
            .y_axis(
                Axis::default()
                    .title("Loss")
                    .style(Style::default().fg(Color::Gray))
                    .bounds([y_min, y_max])
                    .labels::<Vec<Line>>(vec![
                        format!("{:.2}", y_min).into(),
                        format!("{:.2}", (y_min + y_max) / 2.0).into(),
                        format!("{:.2}", y_max).into(),
                    ]),
            );

        f.render_widget(chart, area);
    }

    fn draw_gpu_quick(&self, f: &mut Frame, area: Rect) {
        let content = if let Some(stats) = self.gpu_monitor.current() {
            let health_color = if stats.is_thermal_throttling() {
                Color::Red
            } else if stats.is_memory_pressure() {
                Color::Yellow
            } else {
                Color::Green
            };

            vec![
                Line::from(vec![
                    Span::raw(" GPU: "),
                    Span::styled(
                        format!("{}%", stats.gpu_util),
                        Style::default().fg(Color::Cyan),
                    ),
                    Span::raw(" | Mem: "),
                    Span::styled(stats.memory_string(), Style::default().fg(Color::Cyan)),
                    Span::raw(" | Temp: "),
                    Span::styled(
                        format!("{}°C", stats.temperature),
                        Style::default().fg(if stats.temperature > 80 {
                            Color::Red
                        } else if stats.temperature > 70 {
                            Color::Yellow
                        } else {
                            Color::Green
                        }),
                    ),
                    Span::raw(" | Power: "),
                    Span::styled(stats.power_string(), Style::default().fg(Color::Cyan)),
                ]),
                Line::from(vec![
                    Span::raw(" Status: "),
                    Span::styled(
                        self.gpu_monitor.health_status(),
                        Style::default()
                            .fg(health_color)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::raw(format!(
                        " | Peak Mem: {:.1} GB | PState: {}",
                        self.gpu_monitor.peak_memory() as f64 / 1e9,
                        stats.pstate
                    )),
                ]),
            ]
        } else {
            vec![Line::from(Span::raw(" GPU stats unavailable"))]
        };

        let widget = Paragraph::new(content).block(
            Block::default()
                .borders(Borders::ALL)
                .title(" GPU Quick Stats "),
        );
        f.render_widget(widget, area);
    }

    fn draw_timing_quick(&self, f: &mut Frame, area: Rect, run: &TrainingRun) {
        let elapsed = run.elapsed();
        let gpu_time_ms = self.gpu_monitor.gpu_time_ms();

        let content = Line::from(vec![
            Span::raw(" Wall: "),
            Span::styled(
                format!(
                    "{}:{:02}:{:02}",
                    elapsed.num_hours(),
                    elapsed.num_minutes() % 60,
                    elapsed.num_seconds() % 60
                ),
                Style::default().fg(Color::Cyan),
            ),
            Span::raw(" | GPU: "),
            Span::styled(
                format!(
                    "{}:{:02}:{:02}",
                    gpu_time_ms / 3600000,
                    (gpu_time_ms % 3600000) / 60000,
                    (gpu_time_ms % 60000) / 1000
                ),
                Style::default().fg(Color::Yellow),
            ),
            Span::raw(" | ETA: "),
            Span::styled(run.eta_string(), Style::default().fg(Color::Green)),
            Span::raw(format!(
                " | {:.1} steps/s",
                run.steps_per_second.unwrap_or(0.0)
            )),
        ]);

        let widget =
            Paragraph::new(content).block(Block::default().borders(Borders::ALL).title(" Timing "));
        f.render_widget(widget, area);
    }

    fn draw_checkpoint_table(&self, f: &mut Frame, area: Rect, run: &TrainingRun) {
        let rows: Vec<Row> = run
            .checkpoints
            .iter()
            .rev()
            .take(5)
            .map(|cp| {
                let save_dur = cp
                    .save_duration()
                    .map(|d| format!("{:.1}s", d.num_milliseconds() as f64 / 1000.0))
                    .unwrap_or_else(|| "...".to_string());

                let upload_status = if cp.upload_completed.is_some() {
                    "Uploaded"
                } else if cp.upload_started.is_some() {
                    "Uploading..."
                } else {
                    "Local"
                };

                let size_str = cp
                    .size_bytes
                    .map(|s| format!("{:.1}MB", s as f64 / 1e6))
                    .unwrap_or_else(|| "...".to_string());

                Row::new(vec![
                    format!("Step {}", cp.step),
                    cp.save_started.format("%H:%M:%S").to_string(),
                    save_dur,
                    size_str,
                    upload_status.to_string(),
                ])
            })
            .collect();

        let table = Table::new(
            rows,
            [
                Constraint::Length(12),
                Constraint::Length(10),
                Constraint::Length(8),
                Constraint::Length(10),
                Constraint::Length(12),
            ],
        )
        .header(
            Row::new(vec!["Step", "Started", "Duration", "Size", "Status"])
                .style(Style::default().add_modifier(Modifier::BOLD)),
        )
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Recent Checkpoints "),
        );

        f.render_widget(table, area);
    }

    fn draw_phase_transitions(&self, f: &mut Frame, area: Rect, run: &TrainingRun) {
        let rows: Vec<Row> = run
            .phase_transitions
            .iter()
            .rev()
            .take(8)
            .map(|pt| {
                let _from_color = match pt.from_phase {
                    TrainingPhase::Warmup => Color::Yellow,
                    TrainingPhase::Full => Color::Blue,
                    TrainingPhase::Predict => Color::Green,
                    TrainingPhase::Correct => Color::Magenta,
                };
                let _to_color = match pt.to_phase {
                    TrainingPhase::Warmup => Color::Yellow,
                    TrainingPhase::Full => Color::Blue,
                    TrainingPhase::Predict => Color::Green,
                    TrainingPhase::Correct => Color::Magenta,
                };

                Row::new(vec![
                    format!("Step {}", pt.step),
                    pt.timestamp.format("%H:%M:%S").to_string(),
                    pt.from_phase.to_string(),
                    "->".to_string(),
                    pt.to_phase.to_string(),
                ])
                .style(Style::default())
            })
            .collect();

        let table = Table::new(
            rows,
            [
                Constraint::Length(12),
                Constraint::Length(10),
                Constraint::Length(10),
                Constraint::Length(4),
                Constraint::Length(10),
            ],
        )
        .header(
            Row::new(vec!["Step", "Time", "From", "", "To"])
                .style(Style::default().add_modifier(Modifier::BOLD)),
        )
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Phase Transitions "),
        );

        f.render_widget(table, area);
    }

    fn format_loss_stats(&self, run: &TrainingRun) -> Vec<Line<'static>> {
        let loss_trend = if self.loss_history.len() >= 10 {
            let recent: Vec<f64> = self
                .loss_history
                .iter()
                .rev()
                .take(10)
                .map(|(_, l)| *l)
                .collect();
            let older: Vec<f64> = self
                .loss_history
                .iter()
                .rev()
                .skip(10)
                .take(10)
                .map(|(_, l)| *l)
                .collect();
            if !older.is_empty() {
                let recent_avg: f64 = recent.iter().sum::<f64>() / recent.len() as f64;
                let older_avg: f64 = older.iter().sum::<f64>() / older.len() as f64;
                if recent_avg < older_avg {
                    ("Decreasing", Color::Green)
                } else {
                    ("Increasing", Color::Red)
                }
            } else {
                ("Collecting...", Color::Gray)
            }
        } else {
            ("Collecting...", Color::Gray)
        };

        vec![
            Line::from(vec![
                Span::raw(" Current Loss: "),
                Span::styled(
                    format!("{:.6}", run.current_loss),
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD),
                ),
            ]),
            Line::from(vec![
                Span::raw(" Best Loss: "),
                Span::styled(
                    format!("{:.6}", run.best_loss),
                    Style::default().fg(Color::Green),
                ),
                Span::raw(format!(" (step {})", run.best_step)),
            ]),
            Line::from(vec![
                Span::raw(" Loss Trend: "),
                Span::styled(loss_trend.0.to_string(), Style::default().fg(loss_trend.1)),
            ]),
            Line::from(vec![
                Span::raw(" Improvement: "),
                Span::raw(format!("{:.4}", run.current_loss - run.best_loss)),
                Span::raw(" from best"),
            ]),
            Line::from(vec![
                Span::raw(" Samples: "),
                Span::raw(format!("{}", self.loss_history.len())),
            ]),
        ]
    }

    fn format_gpu_info(&self, stats: &GpuStats) -> Vec<Line<'static>> {
        vec![
            Line::from(vec![
                Span::raw(" Driver: "),
                Span::raw(stats.driver_version.clone()),
                Span::raw(" | CUDA: "),
                Span::raw(stats.cuda_version.clone()),
            ]),
            Line::from(vec![
                Span::raw(" GPU Util: "),
                Span::styled(
                    format!("{}%", stats.gpu_util),
                    Style::default().fg(Color::Cyan),
                ),
                Span::raw(" | Memory Util: "),
                Span::styled(
                    format!("{}%", stats.memory_util),
                    Style::default().fg(Color::Cyan),
                ),
            ]),
            Line::from(vec![
                Span::raw(" Clocks - Graphics: "),
                Span::raw(format!(
                    "{}/{} MHz",
                    stats.clock_graphics, stats.clock_graphics_max
                )),
            ]),
            Line::from(vec![
                Span::raw("        - Memory: "),
                Span::raw(format!(
                    "{}/{} MHz",
                    stats.clock_memory, stats.clock_memory_max
                )),
            ]),
            Line::from(vec![
                Span::raw(" PCIe: "),
                Span::raw(format!("Gen{} x{}", stats.pcie_gen, stats.pcie_width)),
                Span::raw(" | PState: "),
                Span::styled(stats.pstate.clone(), Style::default().fg(Color::Yellow)),
            ]),
            Line::from(vec![
                Span::raw(" Compute Mode: "),
                Span::raw(stats.compute_mode.clone()),
            ]),
        ]
    }

    fn format_thermal_power(&self, stats: &GpuStats) -> Vec<Line<'static>> {
        let temp_color = if stats.temperature > 85 {
            Color::Red
        } else if stats.temperature > 75 {
            Color::Yellow
        } else {
            Color::Green
        };

        let power_color = if stats.is_power_limited() {
            Color::Yellow
        } else {
            Color::Green
        };

        vec![
            Line::from(vec![
                Span::raw(" Temperature: "),
                Span::styled(
                    format!("{}°C", stats.temperature),
                    Style::default().fg(temp_color),
                ),
                Span::raw(format!(
                    " (throttle: {}°C, shutdown: {}°C)",
                    stats.temp_throttle, stats.temp_shutdown
                )),
            ]),
            Line::from(vec![
                Span::raw(" Fan Speed: "),
                Span::raw(format!("{}%", stats.fan_speed)),
            ]),
            Line::from(vec![
                Span::raw(" Power: "),
                Span::styled(
                    format!("{:.0}W", stats.power_draw),
                    Style::default().fg(power_color),
                ),
                Span::raw(format!(
                    " / {:.0}W limit ({:.0}W max)",
                    stats.power_limit, stats.power_max
                )),
            ]),
            Line::from(vec![
                Span::raw(" Peak Power: "),
                Span::raw(format!("{:.0}W", self.gpu_monitor.peak_power())),
                Span::raw(" | Peak Temp: "),
                Span::raw(format!("{}°C", self.gpu_monitor.peak_temperature())),
            ]),
        ]
    }

    fn format_timing_stats(&self, run: &TrainingRun) -> Vec<Line<'static>> {
        let elapsed = run.elapsed();
        let gpu_time_ms = self.gpu_monitor.gpu_time_ms();

        let efficiency = if elapsed.num_milliseconds() > 0 {
            100.0 * gpu_time_ms as f64 / elapsed.num_milliseconds() as f64
        } else {
            0.0
        };

        vec![
            Line::from(vec![
                Span::raw(" Wall Time: "),
                Span::styled(
                    format!(
                        "{}h {}m {}s",
                        elapsed.num_hours(),
                        elapsed.num_minutes() % 60,
                        elapsed.num_seconds() % 60
                    ),
                    Style::default().fg(Color::Cyan),
                ),
            ]),
            Line::from(vec![
                Span::raw(" GPU Time (est): "),
                Span::styled(
                    format!(
                        "{}h {}m {}s",
                        gpu_time_ms / 3600000,
                        (gpu_time_ms % 3600000) / 60000,
                        (gpu_time_ms % 60000) / 1000
                    ),
                    Style::default().fg(Color::Yellow),
                ),
            ]),
            Line::from(vec![
                Span::raw(" GPU Efficiency: "),
                Span::styled(
                    format!("{:.1}%", efficiency),
                    Style::default().fg(Color::Green),
                ),
            ]),
            Line::from(vec![
                Span::raw(" ETA: "),
                Span::styled(
                    run.eta_string(),
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD),
                ),
            ]),
            Line::from(vec![
                Span::raw(" Steps/sec: "),
                Span::raw(format!("{:.2}", run.steps_per_second.unwrap_or(0.0))),
                Span::raw(" | Tokens/sec: "),
                Span::raw(format!("{:.0}", run.tokens_per_second())),
            ]),
        ]
    }

    fn format_run_timestamps(&self, run: &TrainingRun) -> Vec<Line<'static>> {
        vec![
            Line::from(vec![
                Span::raw(" Started: "),
                Span::styled(
                    run.started_at.format("%Y-%m-%d %H:%M:%S UTC").to_string(),
                    Style::default().fg(Color::Cyan),
                ),
            ]),
            Line::from(vec![
                Span::raw(" Updated: "),
                Span::raw(run.updated_at.format("%Y-%m-%d %H:%M:%S UTC").to_string()),
            ]),
            Line::from(vec![
                Span::raw(" Ended: "),
                Span::raw(
                    run.ended_at
                        .map(|t| t.format("%Y-%m-%d %H:%M:%S UTC").to_string())
                        .unwrap_or_else(|| "In progress...".to_string()),
                ),
            ]),
        ]
    }

    fn format_phase_detail(&self, run: &TrainingRun) -> Vec<Line<'static>> {
        let (phase_color, phase_desc) = match run.current_phase {
            TrainingPhase::Warmup => (
                Color::Yellow,
                "Collecting baseline statistics for gradient prediction",
            ),
            TrainingPhase::Full => (Color::Blue, "Computing full forward and backward passes"),
            TrainingPhase::Predict => (
                Color::Green,
                "Using VSA to predict gradients, skipping backward pass",
            ),
            TrainingPhase::Correct => (
                Color::Magenta,
                "Correcting prediction errors with full backward",
            ),
        };

        vec![
            Line::from(vec![
                Span::raw(" Current: "),
                Span::styled(
                    run.current_phase.to_string(),
                    Style::default()
                        .fg(phase_color)
                        .add_modifier(Modifier::BOLD),
                ),
            ]),
            Line::from(Span::raw("")),
            Line::from(vec![Span::styled(
                format!(" {}", phase_desc),
                Style::default().fg(Color::Gray),
            )]),
            Line::from(Span::raw("")),
            Line::from(vec![
                Span::raw(" Forward passes: "),
                Span::raw(format!("{}", run.total_forward)),
            ]),
            Line::from(vec![
                Span::raw(" Backward passes: "),
                Span::raw(format!("{}", run.total_backward)),
                Span::raw(" ("),
                Span::styled(
                    format!("{:.1}% reduction", run.backward_reduction()),
                    Style::default().fg(Color::Green),
                ),
                Span::raw(")"),
            ]),
        ]
    }

    fn format_phase_stats(&self, _run: &TrainingRun) -> Vec<Line<'static>> {
        // Count phases from metrics
        let mut warmup = 0;
        let mut full = 0;
        let mut predict = 0;
        let mut correct = 0;

        for m in &self.all_metrics {
            match m.phase {
                TrainingPhase::Warmup => warmup += 1,
                TrainingPhase::Full => full += 1,
                TrainingPhase::Predict => predict += 1,
                TrainingPhase::Correct => correct += 1,
            }
        }

        let total = warmup + full + predict + correct;
        let pct = |n: usize| {
            if total > 0 {
                100.0 * n as f64 / total as f64
            } else {
                0.0
            }
        };

        vec![
            Line::from(vec![
                Span::styled(" WARMUP: ", Style::default().fg(Color::Yellow)),
                Span::raw(format!("{} ({:.1}%)", warmup, pct(warmup))),
            ]),
            Line::from(vec![
                Span::styled(" FULL: ", Style::default().fg(Color::Blue)),
                Span::raw(format!("{} ({:.1}%)", full, pct(full))),
            ]),
            Line::from(vec![
                Span::styled(" PREDICT: ", Style::default().fg(Color::Green)),
                Span::raw(format!("{} ({:.1}%)", predict, pct(predict))),
            ]),
            Line::from(vec![
                Span::styled(" CORRECT: ", Style::default().fg(Color::Magenta)),
                Span::raw(format!("{} ({:.1}%)", correct, pct(correct))),
            ]),
        ]
    }

    fn draw_footer(&self, f: &mut Frame, area: Rect) {
        let footer = Paragraph::new(Line::from(vec![
            Span::styled(" q", Style::default().fg(Color::Yellow)),
            Span::raw(" Quit | "),
            Span::styled("j/k", Style::default().fg(Color::Yellow)),
            Span::raw(" Navigate | "),
            Span::styled("Tab", Style::default().fg(Color::Yellow)),
            Span::raw(" Switch view | "),
            Span::styled("1-5", Style::default().fg(Color::Yellow)),
            Span::raw(" Direct view | "),
            Span::styled("r", Style::default().fg(Color::Yellow)),
            Span::raw(" Refresh"),
        ]))
        .block(Block::default().borders(Borders::ALL));

        f.render_widget(footer, area);
    }
}
