#!/usr/bin/env python3
"""Insert enhanced analysis tab functions into live_monitor.rs"""

import sys

# The new code to insert
NEW_CODE = '''
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

'''

ANALYSIS_HELPERS = open('/home/kang/Documents/projects/rust-ai/training-tools/src/analysis_helpers.txt').read()

# Read the file
with open('/home/kang/Documents/projects/rust-ai/training-tools/src/live_monitor.rs', 'r') as f:
    lines = f.readlines()

# Find the insertion point (before "/// Draw the Network Flow tab")
insert_idx = None
for i, line in enumerate(lines):
    if '/// Draw the Network Flow tab' in line:
        insert_idx = i
        break

if insert_idx is None:
    print("ERROR: Could not find insertion point", file=sys.stderr)
    sys.exit(1)

# Insert the new code
new_lines = lines[:insert_idx] + [NEW_CODE, '\n', ANALYSIS_HELPERS, '\n\n'] + lines[insert_idx:]

# Write back
with open('/home/kang/Documents/projects/rust-ai/training-tools/src/live_monitor.rs', 'w') as f:
    f.writelines(new_lines)

print(f"Successfully inserted analysis tab code at line {insert_idx}")
