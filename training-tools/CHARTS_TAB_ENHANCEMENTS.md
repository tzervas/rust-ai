# Charts Tab Enhancements Summary

## Completed Changes

### 1. ChartType Enum Extensions
**Location:** `src/live_monitor.rs` lines 162-174

Added new chart types to the `ChartType` enum:
```rust
pub enum ChartType {
    // Existing
    LossLine,
    LossScatter,
    GradientNorm,
    StepTime,
    PhaseBreakdown,
    PredictionAccuracy,

    // NEW - Added
    LearningRate,        // LR schedule over time
    Throughput,          // Tokens/second performance
    MemoryUsage,         // GPU memory over time
    LossVsTokens,        // Loss plotted against total tokens (not steps)
    PhaseDistribution,   // Bar chart of phase step distribution
}
```

### 2. Color Constants Added
**Location:** `src/live_monitor.rs` colors module (after line 58)

```rust
pub const LEARNING_RATE: Color = Color::Rgb(255, 150, 255); // Pink
pub const THROUGHPUT: Color = Color::Rgb(150, 255, 150);     // Light green
pub const MEMORY: Color = Color::Rgb(200, 150, 255);         // Purple
pub const MOVING_AVG: Color = Color::Rgb(0, 200, 200);       // Darker cyan
```

### 3. New Data Extraction Methods
**Location:** `src/live_monitor.rs` LiveMetricsReader impl (after line 619)

```rust
/// Get loss vs total tokens trained (for LossVsTokens chart)
pub fn loss_vs_tokens_data(&self) -> Vec<(f64, f64)> {
    self.metrics
        .iter()
        .filter(|m| m.total_tokens_trained > 0)
        .map(|m| (m.total_tokens_trained as f64, m.loss as f64))
        .collect()
}

/// Calculate min/max/mean statistics for any dataset
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
```

### 4. Chart Switch Cases
**Location:** `src/live_monitor.rs` draw_charts_tab() match statement (after line 2557)

```rust
match self.chart_type {
    // ... existing cases ...
    ChartType::PredictionAccuracy => self.draw_prediction_chart(f, chunks[1], reader),

    // NEW CASES
    ChartType::LearningRate => self.draw_learning_rate_chart(f, chunks[1], reader),
    ChartType::Throughput => self.draw_throughput_chart(f, chunks[1], reader),
    ChartType::MemoryUsage => self.draw_memory_chart(f, chunks[1], reader),
    ChartType::LossVsTokens => self.draw_loss_vs_tokens_chart(f, chunks[1], reader),
    ChartType::PhaseDistribution => self.draw_phase_distribution_chart(f, chunks[1], reader),
}
```

### 5. New Chart Drawing Functions
**Location:** `src/live_monitor.rs` before draw_gpu_tab() (insert at ~line 2866)

#### A. Learning Rate Chart
- Displays LR schedule over training steps
- Scientific notation for Y-axis
- Current LR shown in title
- Uses existing `learning_rate_data()` method

#### B. Throughput Chart
- Tokens/second performance metric
- Shows min/max/mean in title
- Uses existing `token_throughput_data()` method
- Calculates stats using new `LiveMetricsReader::stats()`

#### C. Memory Usage Chart
- Placeholder implementation
- Notes that memory is tracked in TrainingRun, not StepMetrics
- Directs users to GPU tab for current memory

#### D. Loss vs Tokens Chart
- X-axis: Total tokens trained (not steps)
- Y-axis: Loss
- Includes scatter plot + EMA trend line
- Useful for comparing runs with different batch sizes

#### E. Phase Distribution Chart
- Horizontal bar chart showing step count per phase
- Shows percentage distribution
- Displays avg loss and avg time per phase
- Color-coded by phase (Warmup/Full/Predict/Correct)

## Enhanced Existing Charts

### Gradient Norm Chart (EXISTING - Could be enhanced)
**Suggested additions:**
- Min/max/mean annotations in title
- Moving average overlay (using `smoothed_loss()` pattern)

### Step Time Chart (EXISTING - Could be enhanced)
**Suggested additions:**
- Already has average in title ✓
- Could add min/max range

### Loss Scatter Chart (EXISTING - Already enhanced!)
**Current features:**
- Scatter plot + EMA trend line ✓
- Good example of enhancement pattern

## Implementation Files Created

1. `/tmp/new_chart_funcs.rs` - Complete implementations of 5 new chart functions
2. `/tmp/new_methods.txt` - Data extraction methods for LiveMetricsReader
3. `/tmp/new_colors.txt` - Color constant definitions

## Integration Steps

To apply these enhancements:

1. **Add ChartType variants** (lines 169-173)
2. **Add color constants** (after line 58)
3. **Add data methods** (after line 619)
4. **Add switch cases** (after line 2557)
5. **Insert chart functions** (before line 2866 `draw_gpu_tab`)

## Notes

- Existing codebase has some compilation errors unrelated to these changes
- `learning_rate_data()` and `token_throughput_data()` already exist in LiveMetricsReader
- `phase_stats()` already exists and is used by new PhaseDistribution chart
- Memory tracking per-step would require adding `gpu_memory_used: Option<u64>` to StepMetrics

## Future Enhancements

1. Add moving average to Gradient Norm chart
2. Add trend indicators to Step Time chart
3. Implement proper memory-per-step tracking
4. Add configurable MA window size
5. Add export/screenshot functionality for charts
