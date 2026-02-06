// CHARTS TAB ENHANCEMENTS - IMPLEMENTATION CODE
// This file contains all the code additions needed for the Charts tab enhancements

// ============================================================================
// 1. ENUM ADDITIONS (line ~169)
// Add these variants to the ChartType enum after PredictionAccuracy
// ============================================================================

    LearningRate,
    Throughput,
    MemoryUsage,
    LossVsTokens,
    PhaseDistribution,

// ============================================================================
// 2. ENUM all() METHOD UPDATE (line ~185)
// Add these to the ChartType::all() array
// ============================================================================

            ChartType::LearningRate,
            ChartType::Throughput,
            ChartType::MemoryUsage,
            ChartType::LossVsTokens,
            ChartType::PhaseDistribution,

// ============================================================================
// 3. ENUM title() METHOD UPDATE (line ~201)
// Add these cases to the match statement
// ============================================================================

            ChartType::LearningRate => "Learning Rate",
            ChartType::Throughput => "Throughput (tok/s)",
            ChartType::MemoryUsage => "Memory Usage",
            ChartType::LossVsTokens => "Loss vs Tokens",
            ChartType::PhaseDistribution => "Phase Distribution",

// ============================================================================
// 4. COLOR CONSTANTS (add after line 58 in colors module)
// ============================================================================

    pub const LEARNING_RATE: Color = Color::Rgb(255, 150, 255); // Pink
    pub const THROUGHPUT: Color = Color::Rgb(150, 255, 150); // Light green
    pub const MEMORY: Color = Color::Rgb(200, 150, 255); // Purple
    pub const MOVING_AVG: Color = Color::Rgb(0, 200, 200); // Darker cyan

// ============================================================================
// 5. LiveMetricsReader DATA METHODS (add before closing brace at ~line 620)
// ============================================================================

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

// ============================================================================
// 6. CHART SWITCH CASES (add after ChartType::PredictionAccuracy case ~line 2557)
// ============================================================================

                ChartType::LearningRate => self.draw_learning_rate_chart(f, chunks[1], reader),
                ChartType::Throughput => self.draw_throughput_chart(f, chunks[1], reader),
                ChartType::MemoryUsage => self.draw_memory_chart(f, chunks[1], reader),
                ChartType::LossVsTokens => self.draw_loss_vs_tokens_chart(f, chunks[1], reader),
                ChartType::PhaseDistribution => self.draw_phase_distribution_chart(f, chunks[1], reader),

// ============================================================================
// 7. CHART DRAWING FUNCTIONS (insert before draw_gpu_tab at ~line 2866)
// ============================================================================

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

    fn draw_memory_chart(&self, f: &mut Frame, area: Rect, _reader: &LiveMetricsReader) {
        let msg = Paragraph::new(
            "Memory tracking not yet implemented in StepMetrics\n\n\
             GPU memory is tracked in TrainingRun but not per-step.\n\
             See GPU tab for current memory usage."
        )
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
