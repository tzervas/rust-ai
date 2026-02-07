//! Screenshot capture utility for TUI visual testing.
//!
//! Provides functionality to capture terminal UI state to text files for:
//! - Visual regression testing
//! - Documentation generation
//! - Animation sequence capture
//!
//! # Example
//!
//! ```no_run
//! use training_tools::screenshot::{ScreenshotCapture, ScreenshotFormat};
//! use std::path::Path;
//!
//! let mut capture = ScreenshotCapture::new(120, 40, Path::new("screenshots"));
//! capture.capture("dashboard_empty").unwrap();
//! ```

use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};

use ratatui::backend::TestBackend;
use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier};
use ratatui::Terminal;

/// Output format for screenshots.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScreenshotFormat {
    /// Plain text (no formatting)
    Plain,
    /// ANSI escape sequences for terminal colors
    Ansi,
    /// HTML with inline styles
    Html,
}

/// Screenshot capture utility for TUI visual testing.
///
/// Uses ratatui's TestBackend to capture terminal state without
/// requiring an actual terminal connection.
pub struct ScreenshotCapture {
    terminal: Terminal<TestBackend>,
    output_dir: PathBuf,
    frame_counter: usize,
}

impl ScreenshotCapture {
    /// Create a new screenshot capture utility.
    ///
    /// # Arguments
    /// - `width`: Terminal width in columns
    /// - `height`: Terminal height in rows
    /// - `output_dir`: Directory to save screenshots
    pub fn new(width: u16, height: u16, output_dir: &Path) -> Self {
        let backend = TestBackend::new(width, height);
        let terminal = Terminal::new(backend).expect("Failed to create test terminal");

        // Ensure output directory exists
        fs::create_dir_all(output_dir).expect("Failed to create screenshot directory");

        Self {
            terminal,
            output_dir: output_dir.to_path_buf(),
            frame_counter: 0,
        }
    }

    /// Get a mutable reference to the terminal for rendering.
    pub fn terminal_mut(&mut self) -> &mut Terminal<TestBackend> {
        &mut self.terminal
    }

    /// Get the terminal area.
    pub fn area(&self) -> Rect {
        match self.terminal.size() {
            Ok(size) => Rect::new(0, 0, size.width, size.height),
            Err(_) => Rect::new(0, 0, 80, 24),
        }
    }

    /// Capture current terminal state to file.
    ///
    /// # Arguments
    /// - `name`: Base name for the screenshot file
    ///
    /// # Returns
    /// Path to the saved screenshot file
    pub fn capture(&mut self, name: &str) -> std::io::Result<PathBuf> {
        self.capture_with_format(name, ScreenshotFormat::Ansi)
    }

    /// Capture current terminal state with specific format.
    pub fn capture_with_format(
        &mut self,
        name: &str,
        format: ScreenshotFormat,
    ) -> std::io::Result<PathBuf> {
        let buffer = self.terminal.backend().buffer().clone();
        let content = match format {
            ScreenshotFormat::Plain => buffer_to_plain(&buffer),
            ScreenshotFormat::Ansi => buffer_to_ansi(&buffer),
            ScreenshotFormat::Html => buffer_to_html(&buffer, name),
        };

        let extension = match format {
            ScreenshotFormat::Plain => "txt",
            ScreenshotFormat::Ansi => "ansi",
            ScreenshotFormat::Html => "html",
        };

        let filename = format!("{}.{}", name, extension);
        let path = self.output_dir.join(&filename);

        let mut file = File::create(&path)?;
        file.write_all(content.as_bytes())?;

        self.frame_counter += 1;
        Ok(path)
    }

    /// Capture a sequence of frames.
    ///
    /// # Arguments
    /// - `name`: Base name for the sequence
    /// - `frames`: Number of frames to capture
    /// - `render_fn`: Function to render each frame
    ///
    /// # Returns
    /// Vector of paths to saved files
    pub fn capture_sequence<F>(
        &mut self,
        name: &str,
        frames: usize,
        mut render_fn: F,
    ) -> std::io::Result<Vec<PathBuf>>
    where
        F: FnMut(&mut Terminal<TestBackend>, usize),
    {
        let mut paths = Vec::with_capacity(frames);

        for frame in 0..frames {
            render_fn(&mut self.terminal, frame);
            let frame_name = format!("{}_{:04}", name, frame);
            let path = self.capture(&frame_name)?;
            paths.push(path);
        }

        Ok(paths)
    }

    /// Generate markdown documentation with inline screenshots.
    ///
    /// # Arguments
    /// - `screenshots`: List of screenshot paths with descriptions
    ///
    /// # Returns
    /// Markdown content as a string
    pub fn generate_docs(screenshots: &[(PathBuf, &str)]) -> String {
        let mut markdown = String::new();
        markdown.push_str("# Training Monitor UI Guide\n\n");
        markdown.push_str("This document shows the visual appearance of each UI component.\n\n");

        for (path, description) in screenshots {
            let filename = path.file_name().unwrap_or_default().to_string_lossy();
            markdown.push_str(&format!("## {}\n\n", description));
            markdown.push_str(&format!("{}\n\n", description));
            markdown.push_str("```ansi\n");
            // Include file contents if it's an ANSI file
            if let Ok(content) = fs::read_to_string(path) {
                markdown.push_str(&content);
            } else {
                markdown.push_str(&format!("<!-- See {} -->\n", filename));
            }
            markdown.push_str("\n```\n\n");
        }

        markdown
    }

    /// Clear the terminal buffer.
    pub fn clear(&mut self) {
        if let Ok(size) = self.terminal.size() {
            let backend = self.terminal.backend_mut();
            backend.resize(size.width, size.height);
        }
    }

    /// Get current frame counter.
    pub fn frame_count(&self) -> usize {
        self.frame_counter
    }

    /// Reset frame counter.
    pub fn reset_counter(&mut self) {
        self.frame_counter = 0;
    }
}

/// Convert buffer to plain text (no colors).
fn buffer_to_plain(buffer: &Buffer) -> String {
    let area = buffer.area;
    let mut output = String::new();

    for y in area.y..area.y + area.height {
        for x in area.x..area.x + area.width {
            let cell = buffer.get(x, y);
            output.push_str(cell.symbol());
        }
        output.push('\n');
    }

    output
}

/// Convert buffer to ANSI escape sequences.
fn buffer_to_ansi(buffer: &Buffer) -> String {
    let area = buffer.area;
    let mut output = String::new();
    let mut last_fg = Color::Reset;
    let mut last_bg = Color::Reset;
    let mut last_modifiers = Modifier::empty();

    for y in area.y..area.y + area.height {
        for x in area.x..area.x + area.width {
            let cell = buffer.get(x, y);

            // Check if style changed
            if cell.fg != last_fg || cell.bg != last_bg || cell.modifier != last_modifiers {
                output.push_str("\x1b[0m"); // Reset

                // Apply new style
                if cell.modifier.contains(Modifier::BOLD) {
                    output.push_str("\x1b[1m");
                }
                if cell.modifier.contains(Modifier::DIM) {
                    output.push_str("\x1b[2m");
                }
                if cell.modifier.contains(Modifier::ITALIC) {
                    output.push_str("\x1b[3m");
                }
                if cell.modifier.contains(Modifier::UNDERLINED) {
                    output.push_str("\x1b[4m");
                }
                if cell.modifier.contains(Modifier::REVERSED) {
                    output.push_str("\x1b[7m");
                }

                // Foreground color
                output.push_str(&color_to_ansi_fg(cell.fg));

                // Background color
                output.push_str(&color_to_ansi_bg(cell.bg));

                last_fg = cell.fg;
                last_bg = cell.bg;
                last_modifiers = cell.modifier;
            }

            output.push_str(cell.symbol());
        }
        output.push_str("\x1b[0m\n"); // Reset at end of line
        last_fg = Color::Reset;
        last_bg = Color::Reset;
        last_modifiers = Modifier::empty();
    }

    output
}

/// Convert buffer to HTML with inline styles.
fn buffer_to_html(buffer: &Buffer, title: &str) -> String {
    let area = buffer.area;
    let mut output = String::new();

    output.push_str("<!DOCTYPE html>\n<html>\n<head>\n");
    output.push_str(&format!("  <title>{}</title>\n", title));
    output.push_str("  <style>\n");
    output.push_str("    body { background-color: #1e1e2e; margin: 0; padding: 20px; }\n");
    output.push_str("    pre { font-family: 'JetBrains Mono', 'Fira Code', monospace; ");
    output.push_str("font-size: 14px; line-height: 1.2; margin: 0; }\n");
    output.push_str("    .bold { font-weight: bold; }\n");
    output.push_str("    .dim { opacity: 0.6; }\n");
    output.push_str("    .italic { font-style: italic; }\n");
    output.push_str("    .underline { text-decoration: underline; }\n");
    output.push_str("  </style>\n");
    output.push_str("</head>\n<body>\n<pre>");

    for y in area.y..area.y + area.height {
        for x in area.x..area.x + area.width {
            let cell = buffer.get(x, y);

            let mut classes = Vec::new();
            if cell.modifier.contains(Modifier::BOLD) {
                classes.push("bold");
            }
            if cell.modifier.contains(Modifier::DIM) {
                classes.push("dim");
            }
            if cell.modifier.contains(Modifier::ITALIC) {
                classes.push("italic");
            }
            if cell.modifier.contains(Modifier::UNDERLINED) {
                classes.push("underline");
            }

            let fg_css = color_to_css(cell.fg, "#cdd6f4");
            let bg_css = color_to_css(cell.bg, "transparent");

            let symbol = html_escape(cell.symbol());

            if !classes.is_empty() || fg_css != "#cdd6f4" || bg_css != "transparent" {
                output.push_str(&format!(
                    "<span class=\"{}\" style=\"color: {}; background-color: {}\">{}</span>",
                    classes.join(" "),
                    fg_css,
                    bg_css,
                    symbol
                ));
            } else {
                output.push_str(&symbol);
            }
        }
        output.push('\n');
    }

    output.push_str("</pre>\n</body>\n</html>");
    output
}

/// Convert ratatui Color to ANSI foreground escape sequence.
fn color_to_ansi_fg(color: Color) -> String {
    match color {
        Color::Reset => String::new(),
        Color::Black => "\x1b[30m".to_string(),
        Color::Red => "\x1b[31m".to_string(),
        Color::Green => "\x1b[32m".to_string(),
        Color::Yellow => "\x1b[33m".to_string(),
        Color::Blue => "\x1b[34m".to_string(),
        Color::Magenta => "\x1b[35m".to_string(),
        Color::Cyan => "\x1b[36m".to_string(),
        Color::Gray => "\x1b[37m".to_string(),
        Color::DarkGray => "\x1b[90m".to_string(),
        Color::LightRed => "\x1b[91m".to_string(),
        Color::LightGreen => "\x1b[92m".to_string(),
        Color::LightYellow => "\x1b[93m".to_string(),
        Color::LightBlue => "\x1b[94m".to_string(),
        Color::LightMagenta => "\x1b[95m".to_string(),
        Color::LightCyan => "\x1b[96m".to_string(),
        Color::White => "\x1b[97m".to_string(),
        Color::Rgb(r, g, b) => format!("\x1b[38;2;{};{};{}m", r, g, b),
        Color::Indexed(i) => format!("\x1b[38;5;{}m", i),
    }
}

/// Convert ratatui Color to ANSI background escape sequence.
fn color_to_ansi_bg(color: Color) -> String {
    match color {
        Color::Reset => String::new(),
        Color::Black => "\x1b[40m".to_string(),
        Color::Red => "\x1b[41m".to_string(),
        Color::Green => "\x1b[42m".to_string(),
        Color::Yellow => "\x1b[43m".to_string(),
        Color::Blue => "\x1b[44m".to_string(),
        Color::Magenta => "\x1b[45m".to_string(),
        Color::Cyan => "\x1b[46m".to_string(),
        Color::Gray => "\x1b[47m".to_string(),
        Color::DarkGray => "\x1b[100m".to_string(),
        Color::LightRed => "\x1b[101m".to_string(),
        Color::LightGreen => "\x1b[102m".to_string(),
        Color::LightYellow => "\x1b[103m".to_string(),
        Color::LightBlue => "\x1b[104m".to_string(),
        Color::LightMagenta => "\x1b[105m".to_string(),
        Color::LightCyan => "\x1b[106m".to_string(),
        Color::White => "\x1b[107m".to_string(),
        Color::Rgb(r, g, b) => format!("\x1b[48;2;{};{};{}m", r, g, b),
        Color::Indexed(i) => format!("\x1b[48;5;{}m", i),
    }
}

/// Convert ratatui Color to CSS color string.
fn color_to_css(color: Color, default: &str) -> String {
    match color {
        Color::Reset => default.to_string(),
        Color::Black => "#000000".to_string(),
        Color::Red => "#f38ba8".to_string(),
        Color::Green => "#a6e3a1".to_string(),
        Color::Yellow => "#f9e2af".to_string(),
        Color::Blue => "#89b4fa".to_string(),
        Color::Magenta => "#cba6f7".to_string(),
        Color::Cyan => "#94e2d5".to_string(),
        Color::Gray => "#6c7086".to_string(),
        Color::DarkGray => "#585b70".to_string(),
        Color::LightRed => "#f38ba8".to_string(),
        Color::LightGreen => "#a6e3a1".to_string(),
        Color::LightYellow => "#f9e2af".to_string(),
        Color::LightBlue => "#89dceb".to_string(),
        Color::LightMagenta => "#f5c2e7".to_string(),
        Color::LightCyan => "#89dceb".to_string(),
        Color::White => "#cdd6f4".to_string(),
        Color::Rgb(r, g, b) => format!("#{:02x}{:02x}{:02x}", r, g, b),
        Color::Indexed(i) => format!("var(--color-{})", i),
    }
}

/// Escape HTML special characters.
fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

/// Test helper: Create a sample metrics file for testing.
pub fn create_sample_metrics_file(path: &Path, num_steps: usize) -> std::io::Result<()> {
    use chrono::Utc;
    use serde_json;

    let mut file = File::create(path)?;

    for step in 0..num_steps {
        let loss = 3.5 - (step as f32 * 0.02).min(2.0);
        let phase = if step < 50 {
            "warmup"
        } else if step < 100 {
            "full"
        } else if step % 3 == 0 {
            "predict"
        } else {
            "correct"
        };

        let metrics = serde_json::json!({
            "step": step,
            "loss": loss + (step as f32 * 0.01).sin() * 0.1,
            "gradient_norm": 0.5 + (step as f32 * 0.05).cos() * 0.2,
            "phase": phase,
            "was_predicted": phase == "predict",
            "prediction_error": if phase == "predict" { Some(0.05) } else { None::<f32> },
            "step_time_ms": 100.0 + (step as f64 * 0.1).sin() * 20.0,
            "timestamp": Utc::now(),
            "tokens_this_step": 4096,
            "total_tokens_trained": step * 4096,
            "tokens_remaining": (1000 - step) * 4096,
            "confidence": 0.7 + (step as f32 * 0.002).min(0.25),
            "learning_rate": 1e-4,
            "perplexity": loss.exp(),
        });

        writeln!(file, "{}", serde_json::to_string(&metrics).unwrap())?;
    }

    Ok(())
}

/// Comparison result for visual tests.
#[derive(Debug, Clone)]
pub struct ComparisonResult {
    /// Whether the screenshots match
    pub matches: bool,
    /// Number of different cells
    pub diff_count: usize,
    /// Total number of cells
    pub total_cells: usize,
    /// Percentage of cells that match
    pub match_percentage: f64,
}

impl ComparisonResult {
    /// Check if screenshots match within a tolerance.
    pub fn matches_within(&self, tolerance_percent: f64) -> bool {
        self.match_percentage >= (100.0 - tolerance_percent)
    }
}

/// Compare two buffers for visual regression testing.
pub fn compare_buffers(expected: &Buffer, actual: &Buffer) -> ComparisonResult {
    if expected.area != actual.area {
        return ComparisonResult {
            matches: false,
            diff_count: (expected.area.width * expected.area.height) as usize,
            total_cells: (expected.area.width * expected.area.height) as usize,
            match_percentage: 0.0,
        };
    }

    let area = expected.area;
    let total_cells = (area.width * area.height) as usize;
    let mut diff_count = 0;

    for y in area.y..area.y + area.height {
        for x in area.x..area.x + area.width {
            let expected_cell = expected.get(x, y);
            let actual_cell = actual.get(x, y);

            if expected_cell.symbol() != actual_cell.symbol()
                || expected_cell.fg != actual_cell.fg
                || expected_cell.bg != actual_cell.bg
            {
                diff_count += 1;
            }
        }
    }

    let match_percentage = if total_cells > 0 {
        ((total_cells - diff_count) as f64 / total_cells as f64) * 100.0
    } else {
        100.0
    };

    ComparisonResult {
        matches: diff_count == 0,
        diff_count,
        total_cells,
        match_percentage,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::style::Style;
    use ratatui::widgets::{Block, Borders, Paragraph};

    #[test]
    fn test_screenshot_capture_creation() {
        let temp_dir = tempfile::tempdir().unwrap();
        let capture = ScreenshotCapture::new(80, 24, temp_dir.path());

        assert_eq!(capture.area().width, 80);
        assert_eq!(capture.area().height, 24);
        assert_eq!(capture.frame_count(), 0);
    }

    #[test]
    fn test_capture_plain_text() {
        let temp_dir = tempfile::tempdir().unwrap();
        let mut capture = ScreenshotCapture::new(40, 10, temp_dir.path());

        capture
            .terminal_mut()
            .draw(|f| {
                let block = Block::default().borders(Borders::ALL).title("Test");
                f.render_widget(block, f.area());
            })
            .unwrap();

        let path = capture
            .capture_with_format("test_plain", ScreenshotFormat::Plain)
            .unwrap();

        assert!(path.exists());
        let content = fs::read_to_string(&path).unwrap();
        assert!(content.contains("Test"));
    }

    #[test]
    fn test_capture_ansi() {
        let temp_dir = tempfile::tempdir().unwrap();
        let mut capture = ScreenshotCapture::new(40, 10, temp_dir.path());

        capture
            .terminal_mut()
            .draw(|f| {
                let para = Paragraph::new("Hello")
                    .style(Style::default().fg(Color::Red))
                    .block(Block::default().borders(Borders::ALL));
                f.render_widget(para, f.area());
            })
            .unwrap();

        let path = capture.capture("test_ansi").unwrap();

        assert!(path.exists());
        let content = fs::read_to_string(&path).unwrap();
        // Should contain ANSI escape sequences
        assert!(content.contains("\x1b["));
    }

    #[test]
    fn test_capture_html() {
        let temp_dir = tempfile::tempdir().unwrap();
        let mut capture = ScreenshotCapture::new(40, 10, temp_dir.path());

        capture
            .terminal_mut()
            .draw(|f| {
                let para = Paragraph::new("Hello")
                    .style(Style::default().fg(Color::Cyan))
                    .block(Block::default().borders(Borders::ALL));
                f.render_widget(para, f.area());
            })
            .unwrap();

        let path = capture
            .capture_with_format("test_html", ScreenshotFormat::Html)
            .unwrap();

        assert!(path.exists());
        let content = fs::read_to_string(&path).unwrap();
        assert!(content.contains("<!DOCTYPE html>"));
        assert!(content.contains("<pre>"));
    }

    #[test]
    fn test_capture_sequence() {
        let temp_dir = tempfile::tempdir().unwrap();
        let mut capture = ScreenshotCapture::new(40, 10, temp_dir.path());

        let paths = capture
            .capture_sequence("seq", 3, |terminal, frame| {
                terminal
                    .draw(|f| {
                        let para = Paragraph::new(format!("Frame {}", frame))
                            .block(Block::default().borders(Borders::ALL));
                        f.render_widget(para, f.area());
                    })
                    .unwrap();
            })
            .unwrap();

        assert_eq!(paths.len(), 3);
        for path in &paths {
            assert!(path.exists());
        }
    }

    #[test]
    fn test_buffer_comparison() {
        let mut buf1 = Buffer::empty(Rect::new(0, 0, 10, 5));
        let mut buf2 = Buffer::empty(Rect::new(0, 0, 10, 5));

        // Identical buffers
        let result = compare_buffers(&buf1, &buf2);
        assert!(result.matches);
        assert_eq!(result.diff_count, 0);
        assert_eq!(result.match_percentage, 100.0);

        // Modify one cell
        buf2.get_mut(5, 2).set_char('X');
        let result = compare_buffers(&buf1, &buf2);
        assert!(!result.matches);
        assert_eq!(result.diff_count, 1);
        assert!(result.matches_within(5.0)); // Within 5% tolerance
    }

    #[test]
    fn test_color_to_ansi() {
        assert_eq!(color_to_ansi_fg(Color::Red), "\x1b[31m");
        assert_eq!(
            color_to_ansi_fg(Color::Rgb(255, 0, 0)),
            "\x1b[38;2;255;0;0m"
        );
        assert_eq!(color_to_ansi_bg(Color::Blue), "\x1b[44m");
    }

    #[test]
    fn test_html_escape() {
        assert_eq!(html_escape("<test>"), "&lt;test&gt;");
        assert_eq!(html_escape("a&b"), "a&amp;b");
    }
}
