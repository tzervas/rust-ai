//! Visual verification tests for the Training Monitor TUI.
//!
//! These tests capture screenshots of each TUI component to:
//! 1. Verify visual appearance during development
//! 2. Detect unintended visual regressions
//! 3. Generate documentation screenshots
//!
//! Run with: `cargo test -p training-tools visual_tests -- --nocapture`

use std::collections::{HashMap, VecDeque};
use std::fs;
use std::path::PathBuf;

use chrono::Utc;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::symbols;
use ratatui::text::{Line, Span};
use ratatui::widgets::{
    Axis, Block, Borders, Chart, Dataset, Gauge, List, ListItem, Paragraph, Tabs,
};

use training_tools::{
    compare_buffers, ComparisonResult, ScreenshotCapture, ScreenshotFormat, StepMetrics,
    TrainingPhase,
};

// Color palette matching live_monitor.rs
mod colors {
    use ratatui::style::Color;

    pub const LOSS_LINE: Color = Color::Rgb(0, 255, 255);
    pub const GRAD_NORM: Color = Color::Rgb(100, 255, 100);
    pub const STEP_TIME: Color = Color::Rgb(255, 180, 50);
    pub const PREDICTION: Color = Color::Rgb(100, 200, 255);

    pub const WARMUP: Color = Color::Rgb(255, 200, 0);
    pub const FULL: Color = Color::Rgb(0, 150, 255);
    pub const PREDICT: Color = Color::Rgb(0, 255, 100);
    pub const CORRECT: Color = Color::Rgb(255, 100, 255);

    pub const ACTIVE_RUN: Color = Color::Rgb(0, 255, 0);
    pub const COMPLETED_RUN: Color = Color::Rgb(100, 200, 255);
    pub const FAILED_RUN: Color = Color::Rgb(255, 80, 80);

    pub const TAB_ACTIVE: Color = Color::Rgb(0, 200, 255);
    pub const TAB_INACTIVE: Color = Color::Rgb(100, 100, 140);
    pub const BORDER: Color = Color::Rgb(80, 80, 120);
}

/// Create sample step metrics for testing.
fn create_sample_metrics(num_steps: usize) -> VecDeque<StepMetrics> {
    let mut metrics = VecDeque::with_capacity(num_steps);

    for step in 0..num_steps {
        let loss = 3.5 - (step as f32 * 0.02).min(2.0) + (step as f32 * 0.1).sin() * 0.1;
        let phase = if step < 50 {
            TrainingPhase::Warmup
        } else if step < 100 {
            TrainingPhase::Full
        } else if step % 3 == 0 {
            TrainingPhase::Predict
        } else {
            TrainingPhase::Correct
        };

        metrics.push_back(StepMetrics {
            step: step as u64,
            loss,
            gradient_norm: 0.5 + (step as f32 * 0.05).cos() * 0.2,
            phase,
            was_predicted: phase == TrainingPhase::Predict,
            prediction_error: if phase == TrainingPhase::Predict {
                Some(0.05)
            } else {
                None
            },
            step_time_ms: 100.0 + (step as f64 * 0.1).sin() * 20.0,
            timestamp: Utc::now(),
            tokens_this_step: 4096,
            total_tokens_trained: step as u64 * 4096,
            tokens_remaining: (1000 - step as u64) * 4096,
            confidence: 0.7 + (step as f32 * 0.002).min(0.25),
            learning_rate: 1e-4,
            perplexity: loss.exp(),
            train_val_gap: None,
            loss_velocity: -0.01,
            loss_acceleration: 0.001,
            gradient_entropy: Some(2.5),
            layer_gradients: None,
            layer_gradient_stats: None,
        });
    }

    metrics
}

/// Get screenshot output directory.
fn screenshot_dir() -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("screenshots");
    fs::create_dir_all(&dir).expect("Failed to create screenshots directory");
    dir
}

// ============================================================================
// Tab Rendering Tests
// ============================================================================

#[test]
fn test_dashboard_tab_renders() {
    let mut capture = ScreenshotCapture::new(120, 40, &screenshot_dir());

    // Empty dashboard
    capture
        .terminal_mut()
        .draw(|f| {
            draw_dashboard_tab(f, f.area(), None);
        })
        .unwrap();

    let path = capture.capture("dashboard_empty").unwrap();
    assert!(path.exists());
    println!("Saved: {}", path.display());

    // Dashboard with data
    let metrics = create_sample_metrics(200);
    capture
        .terminal_mut()
        .draw(|f| {
            draw_dashboard_tab(f, f.area(), Some(&metrics));
        })
        .unwrap();

    let path = capture.capture("dashboard_with_data").unwrap();
    assert!(path.exists());
    println!("Saved: {}", path.display());
}

#[test]
fn test_overview_tab_renders() {
    let mut capture = ScreenshotCapture::new(120, 40, &screenshot_dir());
    let metrics = create_sample_metrics(150);

    capture
        .terminal_mut()
        .draw(|f| {
            draw_overview_tab(f, f.area(), &metrics);
        })
        .unwrap();

    let path = capture.capture("overview_tab").unwrap();
    assert!(path.exists());
    println!("Saved: {}", path.display());
}

#[test]
fn test_charts_tab_renders() {
    let mut capture = ScreenshotCapture::new(120, 40, &screenshot_dir());
    let metrics = create_sample_metrics(200);

    // Loss line chart
    capture
        .terminal_mut()
        .draw(|f| {
            draw_charts_tab(f, f.area(), &metrics, "loss_line");
        })
        .unwrap();

    let path = capture.capture("charts_loss_line").unwrap();
    assert!(path.exists());

    // Gradient norm chart
    capture
        .terminal_mut()
        .draw(|f| {
            draw_charts_tab(f, f.area(), &metrics, "gradient_norm");
        })
        .unwrap();

    let path = capture.capture("charts_gradient_norm").unwrap();
    assert!(path.exists());
}

#[test]
fn test_network_tab_renders() {
    let mut capture = ScreenshotCapture::new(120, 40, &screenshot_dir());

    capture
        .terminal_mut()
        .draw(|f| {
            draw_network_tab(f, f.area());
        })
        .unwrap();

    let path = capture.capture("network_tab").unwrap();
    assert!(path.exists());
    println!("Saved: {}", path.display());
}

#[test]
fn test_all_tabs_render() {
    let mut capture = ScreenshotCapture::new(120, 40, &screenshot_dir());
    let metrics = create_sample_metrics(200);

    let tabs = [
        ("Dashboard", "dashboard"),
        ("Overview", "overview"),
        ("Charts", "charts"),
        ("Network", "network"),
        ("Analysis", "analysis"),
        ("GPU", "gpu"),
        ("History", "history"),
        ("Help", "help"),
    ];

    for (tab_name, file_prefix) in tabs {
        capture
            .terminal_mut()
            .draw(|f| {
                let chunks = Layout::default()
                    .direction(Direction::Vertical)
                    .constraints([Constraint::Length(3), Constraint::Min(10)])
                    .split(f.area());

                draw_tabs_header(f, chunks[0], tab_name);

                match tab_name {
                    "Dashboard" => draw_dashboard_tab(f, chunks[1], Some(&metrics)),
                    "Overview" => draw_overview_tab(f, chunks[1], &metrics),
                    "Charts" => draw_charts_tab(f, chunks[1], &metrics, "loss_line"),
                    "Network" => draw_network_tab(f, chunks[1]),
                    "Analysis" => draw_analysis_tab(f, chunks[1], &metrics),
                    "GPU" => draw_gpu_tab(f, chunks[1]),
                    "History" => draw_history_tab(f, chunks[1]),
                    "Help" => draw_help_tab(f, chunks[1]),
                    _ => {}
                }
            })
            .unwrap();

        let path = capture.capture(&format!("tab_{}", file_prefix)).unwrap();
        assert!(path.exists());
        println!("Saved tab {}: {}", tab_name, path.display());
    }
}

// ============================================================================
// Color Coding Tests
// ============================================================================

#[test]
fn test_color_coding_correct() {
    let mut capture = ScreenshotCapture::new(80, 30, &screenshot_dir());

    capture
        .terminal_mut()
        .draw(|f| {
            draw_color_legend(f, f.area());
        })
        .unwrap();

    let path = capture.capture("color_legend").unwrap();
    assert!(path.exists());
    println!("Saved color legend: {}", path.display());
}

#[test]
fn test_phase_colors() {
    let mut capture = ScreenshotCapture::new(60, 20, &screenshot_dir());

    capture
        .terminal_mut()
        .draw(|f| {
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(3),
                    Constraint::Length(3),
                    Constraint::Length(3),
                    Constraint::Length(3),
                    Constraint::Min(0),
                ])
                .split(f.area());

            let phases = [
                ("WARMUP", colors::WARMUP),
                ("FULL", colors::FULL),
                ("PREDICT", colors::PREDICT),
                ("CORRECT", colors::CORRECT),
            ];

            for (i, (name, color)) in phases.iter().enumerate() {
                let gauge = Gauge::default()
                    .block(
                        Block::default()
                            .borders(Borders::ALL)
                            .title(format!(" {} ", name)),
                    )
                    .gauge_style(Style::default().fg(*color))
                    .ratio(0.7)
                    .label(format!("Phase: {}", name));
                f.render_widget(gauge, chunks[i]);
            }
        })
        .unwrap();

    let path = capture.capture("phase_colors").unwrap();
    assert!(path.exists());
}

// ============================================================================
// Animation Sequence Tests
// ============================================================================

#[test]
fn test_tab_navigation_sequence() {
    let mut capture = ScreenshotCapture::new(120, 40, &screenshot_dir());
    let metrics = create_sample_metrics(200);

    let tabs = ["Dashboard", "Overview", "Charts", "Network"];

    let paths = capture
        .capture_sequence("navigation", tabs.len(), |terminal, frame| {
            terminal
                .draw(|f| {
                    let chunks = Layout::default()
                        .direction(Direction::Vertical)
                        .constraints([Constraint::Length(3), Constraint::Min(10)])
                        .split(f.area());

                    draw_tabs_header(f, chunks[0], tabs[frame]);

                    match tabs[frame] {
                        "Dashboard" => draw_dashboard_tab(f, chunks[1], Some(&metrics)),
                        "Overview" => draw_overview_tab(f, chunks[1], &metrics),
                        "Charts" => draw_charts_tab(f, chunks[1], &metrics, "loss_line"),
                        "Network" => draw_network_tab(f, chunks[1]),
                        _ => {}
                    }
                })
                .unwrap();
        })
        .unwrap();

    assert_eq!(paths.len(), 4);
    println!("Saved {} navigation frames", paths.len());
}

#[test]
fn test_training_progress_sequence() {
    let mut capture = ScreenshotCapture::new(100, 30, &screenshot_dir());

    let paths = capture
        .capture_sequence("training_progress", 10, |terminal, frame| {
            let progress = (frame as f64 + 1.0) / 10.0;
            let step = (frame + 1) * 100;
            let loss = 3.5 - (frame as f32 * 0.2);

            terminal
                .draw(|f| {
                    let chunks = Layout::default()
                        .direction(Direction::Vertical)
                        .constraints([
                            Constraint::Length(3),
                            Constraint::Length(6),
                            Constraint::Min(10),
                        ])
                        .split(f.area());

                    // Progress bar
                    let gauge = Gauge::default()
                        .block(
                            Block::default()
                                .borders(Borders::ALL)
                                .title(format!(" Training - Step {}/1000 ", step)),
                        )
                        .gauge_style(Style::default().fg(colors::PREDICT))
                        .ratio(progress)
                        .label(format!("{:.0}%", progress * 100.0));
                    f.render_widget(gauge, chunks[0]);

                    // Stats
                    let stats = Paragraph::new(vec![
                        Line::from(vec![
                            Span::raw(" Loss: "),
                            Span::styled(
                                format!("{:.4}", loss),
                                Style::default().fg(colors::LOSS_LINE),
                            ),
                        ]),
                        Line::from(vec![Span::raw(" Step: "), Span::raw(format!("{}", step))]),
                        Line::from(vec![
                            Span::raw(" Phase: "),
                            Span::styled(
                                if frame < 3 { "WARMUP" } else { "FULL" },
                                Style::default().fg(if frame < 3 {
                                    colors::WARMUP
                                } else {
                                    colors::FULL
                                }),
                            ),
                        ]),
                    ])
                    .block(Block::default().borders(Borders::ALL).title(" Statistics "));
                    f.render_widget(stats, chunks[1]);

                    // Mini chart
                    let data: Vec<(f64, f64)> = (0..=frame)
                        .map(|i| (i as f64 * 100.0, 3.5 - i as f64 * 0.2))
                        .collect();

                    let dataset = Dataset::default()
                        .name("Loss")
                        .marker(symbols::Marker::Braille)
                        .style(Style::default().fg(colors::LOSS_LINE))
                        .data(&data);

                    let chart = Chart::new(vec![dataset])
                        .block(Block::default().borders(Borders::ALL).title(" Loss Curve "))
                        .x_axis(
                            Axis::default()
                                .title("Step")
                                .bounds([0.0, 1000.0])
                                .style(Style::default().fg(Color::Gray)),
                        )
                        .y_axis(
                            Axis::default()
                                .title("Loss")
                                .bounds([1.0, 4.0])
                                .style(Style::default().fg(Color::Gray)),
                        );
                    f.render_widget(chart, chunks[2]);
                })
                .unwrap();
        })
        .unwrap();

    assert_eq!(paths.len(), 10);
    println!("Saved {} training progress frames", paths.len());
}

#[test]
fn test_phase_transition_sequence() {
    let mut capture = ScreenshotCapture::new(80, 20, &screenshot_dir());

    let phases = [
        (TrainingPhase::Warmup, "WARMUP", colors::WARMUP),
        (TrainingPhase::Full, "FULL", colors::FULL),
        (TrainingPhase::Predict, "PREDICT", colors::PREDICT),
        (TrainingPhase::Correct, "CORRECT", colors::CORRECT),
    ];

    let paths = capture
        .capture_sequence("phase_transition", 4, |terminal, frame| {
            let (phase, name, color) = phases[frame];

            terminal
                .draw(|f| {
                    let chunks = Layout::default()
                        .direction(Direction::Vertical)
                        .constraints([
                            Constraint::Length(5),
                            Constraint::Length(3),
                            Constraint::Min(5),
                        ])
                        .split(f.area());

                    // Phase indicator
                    let phase_block = Paragraph::new(vec![
                        Line::from(Span::styled(
                            format!("   {}   ", name),
                            Style::default().fg(color).add_modifier(Modifier::BOLD),
                        )),
                        Line::from(Span::styled(
                            match phase {
                                TrainingPhase::Warmup => "Collecting baseline statistics",
                                TrainingPhase::Full => "Full forward/backward passes",
                                TrainingPhase::Predict => "Gradient prediction active",
                                TrainingPhase::Correct => "Correcting prediction errors",
                            },
                            Style::default().fg(Color::Gray),
                        )),
                    ])
                    .alignment(ratatui::layout::Alignment::Center)
                    .block(
                        Block::default()
                            .borders(Borders::ALL)
                            .border_style(Style::default().fg(color))
                            .title(" Current Phase "),
                    );
                    f.render_widget(phase_block, chunks[0]);

                    // Progress
                    let progress = match phase {
                        TrainingPhase::Warmup => 0.15,
                        TrainingPhase::Full => 0.35,
                        TrainingPhase::Predict => 0.75,
                        TrainingPhase::Correct => 0.90,
                    };

                    let gauge = Gauge::default()
                        .block(Block::default().borders(Borders::ALL))
                        .gauge_style(Style::default().fg(color))
                        .ratio(progress)
                        .label(format!("{:.0}%", progress * 100.0));
                    f.render_widget(gauge, chunks[1]);

                    // Description
                    let desc = Paragraph::new(vec![
                        Line::from(vec![
                            Span::raw(" Forward passes: "),
                            Span::styled("1000", Style::default().fg(Color::White)),
                        ]),
                        Line::from(vec![
                            Span::raw(" Backward passes: "),
                            Span::styled(
                                match phase {
                                    TrainingPhase::Predict => "250",
                                    _ => "1000",
                                },
                                Style::default().fg(Color::White),
                            ),
                            if phase == TrainingPhase::Predict {
                                Span::styled(
                                    " (75% reduction)",
                                    Style::default().fg(colors::PREDICT),
                                )
                            } else {
                                Span::raw("")
                            },
                        ]),
                    ])
                    .block(Block::default().borders(Borders::ALL).title(" Statistics "));
                    f.render_widget(desc, chunks[2]);
                })
                .unwrap();
        })
        .unwrap();

    assert_eq!(paths.len(), 4);
    println!("Saved {} phase transition frames", paths.len());
}

// ============================================================================
// HTML Documentation Generation
// ============================================================================

#[test]
fn test_generate_html_screenshots() {
    let mut capture = ScreenshotCapture::new(120, 40, &screenshot_dir());
    let metrics = create_sample_metrics(200);

    // Generate HTML version of dashboard
    capture
        .terminal_mut()
        .draw(|f| {
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Length(3), Constraint::Min(10)])
                .split(f.area());

            draw_tabs_header(f, chunks[0], "Dashboard");
            draw_dashboard_tab(f, chunks[1], Some(&metrics));
        })
        .unwrap();

    let path = capture
        .capture_with_format("dashboard_full", ScreenshotFormat::Html)
        .unwrap();

    assert!(path.exists());
    let content = fs::read_to_string(&path).unwrap();
    assert!(content.contains("<!DOCTYPE html>"));
    println!("Saved HTML screenshot: {}", path.display());
}

// ============================================================================
// Visual Regression Tests
// ============================================================================

#[test]
fn test_visual_consistency() {
    let mut capture1 = ScreenshotCapture::new(80, 24, &screenshot_dir());
    let mut capture2 = ScreenshotCapture::new(80, 24, &screenshot_dir());

    // Render same content twice
    let render = |terminal: &mut ratatui::Terminal<ratatui::backend::TestBackend>| {
        terminal
            .draw(|f| {
                let para = Paragraph::new("Test content")
                    .style(Style::default().fg(Color::Cyan))
                    .block(Block::default().borders(Borders::ALL).title(" Test "));
                f.render_widget(para, f.area());
            })
            .unwrap();
    };

    render(capture1.terminal_mut());
    render(capture2.terminal_mut());

    let buf1 = capture1.terminal_mut().backend().buffer().clone();
    let buf2 = capture2.terminal_mut().backend().buffer().clone();

    let result = compare_buffers(&buf1, &buf2);
    assert!(result.matches, "Identical renders should match");
    assert_eq!(result.match_percentage, 100.0);
}

// ============================================================================
// Helper Drawing Functions
// ============================================================================

fn draw_tabs_header(f: &mut ratatui::Frame, area: Rect, active_tab: &str) {
    let tabs = [
        "Dashboard",
        "Overview",
        "Charts",
        "Network",
        "Analysis",
        "GPU",
        "History",
        "Help",
    ];
    let active_idx = tabs.iter().position(|t| *t == active_tab).unwrap_or(0);

    let titles: Vec<Line> = tabs
        .iter()
        .enumerate()
        .map(|(i, t)| {
            let style = if i == active_idx {
                Style::default()
                    .fg(colors::TAB_ACTIVE)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(colors::TAB_INACTIVE)
            };
            Line::from(Span::styled(*t, style))
        })
        .collect();

    let tabs_widget = Tabs::new(titles)
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
        .select(active_idx);

    f.render_widget(tabs_widget, area);
}

fn draw_dashboard_tab(f: &mut ratatui::Frame, area: Rect, metrics: Option<&VecDeque<StepMetrics>>) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(6), Constraint::Min(10)])
        .split(area);

    // Health indicators
    let health_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(25),
            Constraint::Percentage(25),
            Constraint::Percentage(25),
            Constraint::Percentage(25),
        ])
        .split(chunks[0]);

    let (status, status_color) = if metrics.is_some() {
        ("RUNNING", colors::ACTIVE_RUN)
    } else {
        ("NO DATA", Color::Gray)
    };

    let status_widget = Paragraph::new(vec![
        Line::from(Span::styled(
            if metrics.is_some() { ">>>" } else { "---" },
            Style::default()
                .fg(status_color)
                .add_modifier(Modifier::BOLD),
        )),
        Line::from(Span::styled(status, Style::default().fg(status_color))),
    ])
    .alignment(ratatui::layout::Alignment::Center)
    .block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(status_color))
            .title(" Status "),
    );
    f.render_widget(status_widget, health_chunks[0]);

    let (loss_val, loss_trend) = if let Some(m) = metrics.and_then(|m| m.back()) {
        (format!("{:.4}", m.loss), "Improving")
    } else {
        ("--".to_string(), "No data")
    };

    let loss_widget = Paragraph::new(vec![
        Line::from(Span::styled(
            loss_val,
            Style::default()
                .fg(colors::LOSS_LINE)
                .add_modifier(Modifier::BOLD),
        )),
        Line::from(Span::styled(
            loss_trend,
            Style::default().fg(colors::PREDICT),
        )),
    ])
    .alignment(ratatui::layout::Alignment::Center)
    .block(Block::default().borders(Borders::ALL).title(" Loss "));
    f.render_widget(loss_widget, health_chunks[1]);

    let speed_widget = Paragraph::new(vec![
        Line::from(Span::styled(
            if metrics.is_some() { "2.5/s" } else { "--" },
            Style::default()
                .fg(colors::STEP_TIME)
                .add_modifier(Modifier::BOLD),
        )),
        Line::from(Span::styled(
            if metrics.is_some() { "10K tok/s" } else { "--" },
            Style::default().fg(Color::Gray),
        )),
    ])
    .alignment(ratatui::layout::Alignment::Center)
    .block(Block::default().borders(Borders::ALL).title(" Speed "));
    f.render_widget(speed_widget, health_chunks[2]);

    let gpu_widget = Paragraph::new(vec![
        Line::from(Span::styled(
            "85%",
            Style::default()
                .fg(Color::Green)
                .add_modifier(Modifier::BOLD),
        )),
        Line::from(Span::styled("72C", Style::default().fg(Color::Gray))),
    ])
    .alignment(ratatui::layout::Alignment::Center)
    .block(Block::default().borders(Borders::ALL).title(" GPU "));
    f.render_widget(gpu_widget, health_chunks[3]);

    // Phase breakdown
    let phase_content = if let Some(metrics) = metrics {
        let mut phase_counts: HashMap<TrainingPhase, usize> = HashMap::new();
        for m in metrics {
            *phase_counts.entry(m.phase).or_insert(0) += 1;
        }
        let total = metrics.len();

        let mut lines = vec![Line::from(Span::styled(
            " Phase Distribution",
            Style::default()
                .fg(Color::White)
                .add_modifier(Modifier::BOLD),
        ))];

        for phase in [
            TrainingPhase::Warmup,
            TrainingPhase::Full,
            TrainingPhase::Predict,
            TrainingPhase::Correct,
        ] {
            let count = phase_counts.get(&phase).copied().unwrap_or(0);
            let pct = if total > 0 {
                count as f64 / total as f64 * 100.0
            } else {
                0.0
            };
            let color = match phase {
                TrainingPhase::Warmup => colors::WARMUP,
                TrainingPhase::Full => colors::FULL,
                TrainingPhase::Predict => colors::PREDICT,
                TrainingPhase::Correct => colors::CORRECT,
            };
            let bar_width = 20;
            let filled = ((pct / 100.0) * bar_width as f64) as usize;
            let bar: String = "|".repeat(filled) + &" ".repeat(bar_width - filled);

            lines.push(Line::from(vec![
                Span::styled(format!(" {:8?} ", phase), Style::default().fg(color)),
                Span::styled(bar, Style::default().fg(color)),
                Span::styled(format!(" {:5.1}%", pct), Style::default().fg(Color::White)),
            ]));
        }
        lines
    } else {
        vec![Line::from(Span::styled(
            "No data available",
            Style::default().fg(Color::Gray),
        ))]
    };

    let phase_widget = Paragraph::new(phase_content).block(
        Block::default()
            .borders(Borders::ALL)
            .title(" Training Phases "),
    );
    f.render_widget(phase_widget, chunks[1]);
}

fn draw_overview_tab(f: &mut ratatui::Frame, area: Rect, metrics: &VecDeque<StepMetrics>) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Length(30), Constraint::Min(50)])
        .split(area);

    // Run list
    let runs = vec![
        (">>> tritter_100m", colors::ACTIVE_RUN, "#150 L:2.34"),
        ("    tritter_500m", colors::COMPLETED_RUN, "#1000 L:1.89"),
        ("    tritter_1b", colors::FAILED_RUN, "#500 L:3.21"),
    ];

    let items: Vec<ListItem> = runs
        .iter()
        .map(|(name, color, info)| {
            ListItem::new(Line::from(vec![
                Span::styled(*name, Style::default().fg(*color)),
                Span::styled(format!(" {}", info), Style::default().fg(Color::Gray)),
            ]))
        })
        .collect();

    let list = List::new(items).block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(colors::BORDER))
            .title(" Runs "),
    );
    f.render_widget(list, chunks[0]);

    // Right side with chart
    let right_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(3), Constraint::Min(10)])
        .split(chunks[1]);

    // Progress
    let gauge = Gauge::default()
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(" tritter_100m - Step 150/1000 "),
        )
        .gauge_style(Style::default().fg(colors::FULL))
        .ratio(0.15)
        .label("15.0%");
    f.render_widget(gauge, right_chunks[0]);

    // Chart
    let data: Vec<(f64, f64)> = metrics
        .iter()
        .map(|m| (m.step as f64, m.loss as f64))
        .collect();

    let dataset = Dataset::default()
        .name("Loss")
        .marker(symbols::Marker::Braille)
        .style(Style::default().fg(colors::LOSS_LINE))
        .data(&data);

    let y_min = data.iter().map(|(_, y)| *y).fold(f64::INFINITY, f64::min);
    let y_max = data
        .iter()
        .map(|(_, y)| *y)
        .fold(f64::NEG_INFINITY, f64::max);
    let x_max = data.last().map(|(x, _)| *x).unwrap_or(100.0);

    let chart = Chart::new(vec![dataset])
        .block(Block::default().borders(Borders::ALL).title(" Loss Curve "))
        .x_axis(
            Axis::default()
                .title("Step")
                .bounds([0.0, x_max])
                .style(Style::default().fg(Color::Gray)),
        )
        .y_axis(
            Axis::default()
                .title("Loss")
                .bounds([y_min * 0.9, y_max * 1.1])
                .style(Style::default().fg(Color::Gray)),
        );
    f.render_widget(chart, right_chunks[1]);
}

fn draw_charts_tab(
    f: &mut ratatui::Frame,
    area: Rect,
    metrics: &VecDeque<StepMetrics>,
    chart_type: &str,
) {
    let (data, title, color): (Vec<(f64, f64)>, &str, Color) = match chart_type {
        "gradient_norm" => (
            metrics
                .iter()
                .map(|m| (m.step as f64, m.gradient_norm as f64))
                .collect(),
            "Gradient Norm",
            colors::GRAD_NORM,
        ),
        "step_time" => (
            metrics
                .iter()
                .map(|m| (m.step as f64, m.step_time_ms))
                .collect(),
            "Step Time (ms)",
            colors::STEP_TIME,
        ),
        _ => (
            metrics
                .iter()
                .map(|m| (m.step as f64, m.loss as f64))
                .collect(),
            "Loss",
            colors::LOSS_LINE,
        ),
    };

    let dataset = Dataset::default()
        .name(title)
        .marker(symbols::Marker::Braille)
        .style(Style::default().fg(color))
        .data(&data);

    let y_min = data.iter().map(|(_, y)| *y).fold(f64::INFINITY, f64::min);
    let y_max = data
        .iter()
        .map(|(_, y)| *y)
        .fold(f64::NEG_INFINITY, f64::max);
    let x_max = data.last().map(|(x, _)| *x).unwrap_or(100.0);

    let chart = Chart::new(vec![dataset])
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(format!(" {} Chart ", title)),
        )
        .x_axis(
            Axis::default()
                .title("Step")
                .bounds([0.0, x_max])
                .style(Style::default().fg(Color::Gray))
                .labels::<Vec<Line>>(vec![
                    "0".into(),
                    format!("{:.0}", x_max / 2.0).into(),
                    format!("{:.0}", x_max).into(),
                ]),
        )
        .y_axis(
            Axis::default()
                .title(title)
                .bounds([y_min * 0.9, y_max * 1.1])
                .style(Style::default().fg(Color::Gray))
                .labels::<Vec<Line>>(vec![
                    format!("{:.2}", y_min).into(),
                    format!("{:.2}", (y_min + y_max) / 2.0).into(),
                    format!("{:.2}", y_max).into(),
                ]),
        );

    f.render_widget(chart, area);
}

fn draw_network_tab(f: &mut ratatui::Frame, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(3), Constraint::Min(10)])
        .split(area);

    // Title
    let title = Paragraph::new(Line::from(vec![
        Span::raw(" Network Architecture: "),
        Span::styled(
            "Transformer (12 layers, 768 hidden)",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
    ]))
    .block(Block::default().borders(Borders::ALL));
    f.render_widget(title, chunks[0]);

    // 3D layer visualization (ASCII art representation)
    let layer_vis = vec![
        Line::from(""),
        Line::from(vec![
            Span::raw("     "),
            Span::styled("Input Embedding", Style::default().fg(Color::White)),
        ]),
        Line::from(vec![
            Span::raw("         "),
            Span::styled("|", Style::default().fg(Color::Gray)),
        ]),
        Line::from(vec![
            Span::raw("         "),
            Span::styled("v", Style::default().fg(Color::Gray)),
        ]),
        Line::from(vec![
            Span::raw("   "),
            Span::styled("+-------------------+", Style::default().fg(colors::FULL)),
        ]),
        Line::from(vec![
            Span::raw("   "),
            Span::styled("|", Style::default().fg(colors::FULL)),
            Span::raw(" Attention (L0)   "),
            Span::styled("|", Style::default().fg(colors::FULL)),
            Span::raw(" grad: "),
            Span::styled("0.52", Style::default().fg(colors::GRAD_NORM)),
        ]),
        Line::from(vec![
            Span::raw("   "),
            Span::styled("+-------------------+", Style::default().fg(colors::FULL)),
        ]),
        Line::from(vec![
            Span::raw("         "),
            Span::styled("|", Style::default().fg(Color::Gray)),
        ]),
        Line::from(vec![
            Span::raw("   "),
            Span::styled(
                "+-------------------+",
                Style::default().fg(colors::PREDICT),
            ),
        ]),
        Line::from(vec![
            Span::raw("   "),
            Span::styled("|", Style::default().fg(colors::PREDICT)),
            Span::raw(" FFN (L0)         "),
            Span::styled("|", Style::default().fg(colors::PREDICT)),
            Span::raw(" grad: "),
            Span::styled("0.48", Style::default().fg(colors::GRAD_NORM)),
        ]),
        Line::from(vec![
            Span::raw("   "),
            Span::styled(
                "+-------------------+",
                Style::default().fg(colors::PREDICT),
            ),
        ]),
        Line::from(vec![
            Span::raw("         "),
            Span::styled("|", Style::default().fg(Color::Gray)),
        ]),
        Line::from(vec![
            Span::raw("         "),
            Span::styled(
                "... (10 more layers) ...",
                Style::default().fg(Color::DarkGray),
            ),
        ]),
        Line::from(vec![
            Span::raw("         "),
            Span::styled("|", Style::default().fg(Color::Gray)),
        ]),
        Line::from(vec![
            Span::raw("         "),
            Span::styled("v", Style::default().fg(Color::Gray)),
        ]),
        Line::from(vec![
            Span::raw("     "),
            Span::styled("Output Projection", Style::default().fg(Color::White)),
        ]),
    ];

    let vis_widget =
        Paragraph::new(layer_vis).block(Block::default().borders(Borders::ALL).title(" Layers "));
    f.render_widget(vis_widget, chunks[1]);
}

fn draw_analysis_tab(f: &mut ratatui::Frame, area: Rect, metrics: &VecDeque<StepMetrics>) {
    let content = if let Some(latest) = metrics.back() {
        vec![
            Line::from(Span::styled(
                " Training Analysis",
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD),
            )),
            Line::from(""),
            Line::from(vec![
                Span::raw(" Loss Velocity:     "),
                Span::styled(
                    format!("{:.4}", latest.loss_velocity),
                    Style::default().fg(if latest.loss_velocity < 0.0 {
                        colors::PREDICT
                    } else {
                        colors::FAILED_RUN
                    }),
                ),
                Span::raw(if latest.loss_velocity < 0.0 {
                    " (improving)"
                } else {
                    " (worsening)"
                }),
            ]),
            Line::from(vec![
                Span::raw(" Loss Acceleration: "),
                Span::styled(
                    format!("{:.4}", latest.loss_acceleration),
                    Style::default().fg(Color::White),
                ),
            ]),
            Line::from(vec![
                Span::raw(" Confidence:        "),
                Span::styled(
                    format!("{:.1}%", latest.confidence * 100.0),
                    Style::default().fg(colors::PREDICTION),
                ),
            ]),
            Line::from(""),
            Line::from(Span::styled(
                " Recommendations:",
                Style::default().fg(Color::Yellow),
            )),
            Line::from(Span::raw("   - Training is progressing normally")),
            Line::from(Span::raw(
                "   - Consider increasing batch size for stability",
            )),
        ]
    } else {
        vec![Line::from(Span::styled(
            "No data for analysis",
            Style::default().fg(Color::Gray),
        ))]
    };

    let widget =
        Paragraph::new(content).block(Block::default().borders(Borders::ALL).title(" Analysis "));
    f.render_widget(widget, area);
}

fn draw_gpu_tab(f: &mut ratatui::Frame, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(5),
            Constraint::Length(3),
            Constraint::Min(5),
        ])
        .split(area);

    // GPU info
    let gpu_info = vec![
        Line::from(vec![
            Span::raw(" GPU: "),
            Span::styled(
                "NVIDIA RTX 4090",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::from(vec![Span::raw(" Driver: 545.23 | CUDA: 12.3")]),
        Line::from(vec![
            Span::raw(" Utilization: "),
            Span::styled("85%", Style::default().fg(Color::Green)),
            Span::raw(" | Memory: "),
            Span::styled("18.5/24 GB", Style::default().fg(Color::Yellow)),
        ]),
    ];

    let info_widget =
        Paragraph::new(gpu_info).block(Block::default().borders(Borders::ALL).title(" GPU Info "));
    f.render_widget(info_widget, chunks[0]);

    // Memory gauge
    let mem_gauge = Gauge::default()
        .block(Block::default().borders(Borders::ALL).title(" VRAM Usage "))
        .gauge_style(Style::default().fg(Color::Yellow))
        .ratio(0.77)
        .label("18.5 / 24 GB (77%)");
    f.render_widget(mem_gauge, chunks[1]);

    // Thermals
    let thermal_info = vec![
        Line::from(vec![
            Span::raw(" Temperature: "),
            Span::styled("72C", Style::default().fg(Color::Green)),
            Span::raw(" (throttle: 83C)"),
        ]),
        Line::from(vec![Span::raw(" Fan Speed: "), Span::raw("65%")]),
        Line::from(vec![
            Span::raw(" Power: "),
            Span::styled("320W", Style::default().fg(Color::Yellow)),
            Span::raw(" / 450W limit"),
        ]),
    ];

    let thermal_widget = Paragraph::new(thermal_info).block(
        Block::default()
            .borders(Borders::ALL)
            .title(" Thermals & Power "),
    );
    f.render_widget(thermal_widget, chunks[2]);
}

fn draw_history_tab(f: &mut ratatui::Frame, area: Rect) {
    let history = vec![
        (
            "tritter_100m_001",
            "2024-01-15 10:30",
            "Completed",
            colors::COMPLETED_RUN,
        ),
        (
            "tritter_100m_002",
            "2024-01-16 14:20",
            "Failed",
            colors::FAILED_RUN,
        ),
        (
            "tritter_500m_001",
            "2024-01-17 09:00",
            "Completed",
            colors::COMPLETED_RUN,
        ),
        (
            "tritter_1b_001",
            "2024-01-18 16:45",
            "Running",
            colors::ACTIVE_RUN,
        ),
    ];

    let items: Vec<ListItem> = history
        .iter()
        .map(|(name, date, status, color)| {
            ListItem::new(Line::from(vec![
                Span::styled(format!(" {:20} ", name), Style::default().fg(Color::White)),
                Span::styled(format!("{:16} ", date), Style::default().fg(Color::Gray)),
                Span::styled(*status, Style::default().fg(*color)),
            ]))
        })
        .collect();

    let list = List::new(items).block(
        Block::default()
            .borders(Borders::ALL)
            .title(" Training History "),
    );
    f.render_widget(list, area);
}

fn draw_help_tab(f: &mut ratatui::Frame, area: Rect) {
    let help_text = vec![
        Line::from(Span::styled(
            " Keyboard Shortcuts",
            Style::default()
                .fg(Color::White)
                .add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        Line::from(vec![
            Span::styled(" Tab      ", Style::default().fg(Color::Yellow)),
            Span::raw("Next tab"),
        ]),
        Line::from(vec![
            Span::styled(" Shift+Tab", Style::default().fg(Color::Yellow)),
            Span::raw("Previous tab"),
        ]),
        Line::from(vec![
            Span::styled(" 0-7      ", Style::default().fg(Color::Yellow)),
            Span::raw("Jump to tab"),
        ]),
        Line::from(vec![
            Span::styled(" j/k      ", Style::default().fg(Color::Yellow)),
            Span::raw("Navigate runs"),
        ]),
        Line::from(vec![
            Span::styled(" l        ", Style::default().fg(Color::Yellow)),
            Span::raw("Toggle Live/History mode"),
        ]),
        Line::from(vec![
            Span::styled(" r        ", Style::default().fg(Color::Yellow)),
            Span::raw("Refresh"),
        ]),
        Line::from(vec![
            Span::styled(" ?/F1     ", Style::default().fg(Color::Yellow)),
            Span::raw("Show help"),
        ]),
        Line::from(vec![
            Span::styled(" q/Esc    ", Style::default().fg(Color::Yellow)),
            Span::raw("Quit"),
        ]),
    ];

    let help_widget =
        Paragraph::new(help_text).block(Block::default().borders(Borders::ALL).title(" Help "));
    f.render_widget(help_widget, area);
}

fn draw_color_legend(f: &mut ratatui::Frame, area: Rect) {
    let legend = vec![
        Line::from(Span::styled(
            " Color Legend",
            Style::default()
                .fg(Color::White)
                .add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        Line::from(Span::styled(
            " Training Phases:",
            Style::default().fg(Color::Gray),
        )),
        Line::from(vec![
            Span::raw("   "),
            Span::styled("|||", Style::default().fg(colors::WARMUP)),
            Span::raw(" WARMUP  - Collecting baseline statistics"),
        ]),
        Line::from(vec![
            Span::raw("   "),
            Span::styled("|||", Style::default().fg(colors::FULL)),
            Span::raw(" FULL    - Full forward/backward passes"),
        ]),
        Line::from(vec![
            Span::raw("   "),
            Span::styled("|||", Style::default().fg(colors::PREDICT)),
            Span::raw(" PREDICT - Gradient prediction active"),
        ]),
        Line::from(vec![
            Span::raw("   "),
            Span::styled("|||", Style::default().fg(colors::CORRECT)),
            Span::raw(" CORRECT - Correcting prediction errors"),
        ]),
        Line::from(""),
        Line::from(Span::styled(
            " Run Status:",
            Style::default().fg(Color::Gray),
        )),
        Line::from(vec![
            Span::raw("   "),
            Span::styled(">>>", Style::default().fg(colors::ACTIVE_RUN)),
            Span::raw(" Running"),
        ]),
        Line::from(vec![
            Span::raw("   "),
            Span::styled(" v ", Style::default().fg(colors::COMPLETED_RUN)),
            Span::raw(" Completed"),
        ]),
        Line::from(vec![
            Span::raw("   "),
            Span::styled(" X ", Style::default().fg(colors::FAILED_RUN)),
            Span::raw(" Failed"),
        ]),
        Line::from(""),
        Line::from(Span::styled(" Metrics:", Style::default().fg(Color::Gray))),
        Line::from(vec![
            Span::raw("   "),
            Span::styled("---", Style::default().fg(colors::LOSS_LINE)),
            Span::raw(" Loss curve"),
        ]),
        Line::from(vec![
            Span::raw("   "),
            Span::styled("---", Style::default().fg(colors::GRAD_NORM)),
            Span::raw(" Gradient norm"),
        ]),
        Line::from(vec![
            Span::raw("   "),
            Span::styled("---", Style::default().fg(colors::STEP_TIME)),
            Span::raw(" Step time"),
        ]),
        Line::from(vec![
            Span::raw("   "),
            Span::styled("---", Style::default().fg(colors::PREDICTION)),
            Span::raw(" Prediction confidence"),
        ]),
    ];

    let legend_widget = Paragraph::new(legend).block(
        Block::default()
            .borders(Borders::ALL)
            .title(" Color Legend "),
    );
    f.render_widget(legend_widget, area);
}
