# Training Monitor UI Guide

This document provides a comprehensive visual guide to the Training Monitor TUI (Terminal User Interface).

## Overview

The Training Monitor provides real-time visualization of training runs with multiple tabs:

| Tab | Key | Description |
|-----|-----|-------------|
| Dashboard | `0` | High-level overview with health indicators |
| Overview | `1` | Run list with loss chart |
| Charts | `2` | Detailed metric charts with multiple views |
| Network | `3` | Neural network layer visualization |
| Analysis | `4` | Training dynamics analysis |
| GPU | `5` | GPU utilization, memory, and thermals |
| History | `6` | Historical training runs |
| Help | `7` | Keyboard shortcuts reference |

## Keyboard Shortcuts

### Navigation

| Key | Action |
|-----|--------|
| `Tab` | Next tab |
| `Shift+Tab` | Previous tab |
| `0-7` | Jump to specific tab |
| `j` or `Down` | Next run |
| `k` or `Up` | Previous run |

### Views

| Key | Action |
|-----|--------|
| `l` | Toggle Live/History mode |
| `[` or `Left` | Previous chart type (Charts/GPU tabs) |
| `]` or `Right` | Next chart type (Charts/GPU tabs) |

### Actions

| Key | Action |
|-----|--------|
| `r` | Refresh data |
| `c` | Clear history buffer |
| `?` or `F1` | Show help overlay |
| `q` or `Esc` | Quit |

## Color Legend

### Training Phases

The hybrid predictive training system operates in four phases, each with a distinct color:

| Color | Phase | Description |
|-------|-------|-------------|
| Orange (RGB 255,200,0) | **WARMUP** | Collecting baseline statistics for gradient prediction |
| Blue (RGB 0,150,255) | **FULL** | Computing full forward and backward passes |
| Green (RGB 0,255,100) | **PREDICT** | Using VSA to predict gradients, skipping backward pass |
| Magenta (RGB 255,100,255) | **CORRECT** | Correcting prediction errors with full backward |

### Run Status

| Color | Icon | Status |
|-------|------|--------|
| Bright Green | `>>>` | Running |
| Light Blue | `v` | Completed |
| Red | `X` | Failed |
| Yellow | `||` | Paused |
| Cyan | `..` | Initializing |
| Gray | `--` | Cancelled |

### Metrics Visualization

| Color | Metric |
|-------|--------|
| Cyan (RGB 0,255,255) | Loss curve |
| Green (RGB 100,255,100) | Gradient norm |
| Orange (RGB 255,180,50) | Step time |
| Light Blue (RGB 100,200,255) | Prediction confidence |
| Magenta (RGB 255,0,255) | Trend line (EMA) |
| Yellow (RGB 255,255,0) | Linear regression |

### GPU Health

| Color | Condition |
|-------|-----------|
| Green | Normal operation |
| Yellow | Warning (high memory usage or temperature) |
| Red | Critical (thermal throttling or OOM risk) |

## Tab Descriptions

### Dashboard Tab

The Dashboard provides a quick overview of training health:

```
+-------------+-------------+-------------+-------------+
|   Status    |    Loss     |    Speed    |     GPU     |
|    >>>      |   2.3456    |   2.5/s     |    85%      |
|  RUNNING    |  Improving  |  10K tok/s  |    72C      |
+-------------+-------------+-------------+-------------+
|              Training Phases                          |
|  WARMUP   ||||||||                         25.0%      |
|  FULL     ||||||||||||||||||               50.0%      |
|  PREDICT  ||||||||||||||||||||||||||       75.0%      |
|  CORRECT  ||||||                           15.0%      |
+-------------------------------------------------------+
```

### Overview Tab

The Overview shows the run list and current metrics:

```
+---------------------+----------------------------------------+
|       Runs          |  Progress: Step 150/1000 [15%]         |
|---------------------|----------------------------------------|
| >>> tritter_100m    |                                        |
|     #150 L:2.34     |  +----------------------------------+  |
|                     |  |           Loss Curve             |  |
|     tritter_500m    |  |  *                               |  |
|     #1000 L:1.89    |  |   *  *                           |  |
|                     |  |      *  *  *                     |  |
|     tritter_1b      |  |           *  *  *  *             |  |
|     #500 L:3.21     |  +----------------------------------+  |
+---------------------+----------------------------------------+
```

### Charts Tab

Multiple chart types available via `[`/`]` keys:

1. **Loss (Line)** - Loss over time with Braille markers
2. **Loss (Scatter+Trend)** - Scatter plot with EMA trend line
3. **Gradient Norm** - Gradient magnitude over time
4. **Step Time** - Training step duration
5. **Phase Breakdown** - Bar chart of phase distribution
6. **Prediction Accuracy** - Prediction error over time

### Network Tab

ASCII visualization of network architecture with per-layer gradient norms:

```
     Input Embedding
         |
         v
   +-------------------+
   | Attention (L0)    | grad: 0.52
   +-------------------+
         |
   +-------------------+
   | FFN (L0)          | grad: 0.48
   +-------------------+
         |
    ... (10 more layers) ...
         |
         v
     Output Projection
```

Layer colors indicate gradient health:
- Normal: Blue/Green
- Vanishing (<1e-7): Red warning
- Exploding (>100): Red warning

### Analysis Tab

Training dynamics analysis with recommendations:

- **Loss Velocity**: Rate of loss change (negative = improving)
- **Loss Acceleration**: Second derivative of loss
- **Confidence**: Predictor confidence level
- **Recommendations**: Automated suggestions based on metrics

### GPU Tab

Detailed GPU monitoring:

- GPU utilization percentage
- VRAM usage with gauge
- Temperature and thermal throttle threshold
- Power consumption vs limit
- PCIe bandwidth and P-state

### History Tab

Browse past training runs with:

- Run name and timestamp
- Final status (Completed/Failed)
- Best loss achieved
- Total training time

## Example Workflows

### Monitoring a Training Run

1. Start the monitor: `train-monitor --runs-dir ./runs`
2. Use `Tab` to navigate to **Dashboard** for overview
3. Use `j/k` to select the active run
4. Press `2` for **Charts** tab to see loss curve
5. Use `[/]` to cycle between different chart types
6. Press `5` for **GPU** tab to check thermals

### Investigating Training Issues

1. Check **Dashboard** for overall health indicators:
   - Red status = run failed
   - Rising loss trend = possible divergence

2. If loss is oscillating:
   - Check **Analysis** tab for recommendations
   - May suggest reducing learning rate

3. Check **Charts** tab gradient norm:
   - Spikes indicate gradient explosion
   - Near-zero indicates vanishing gradients

4. Check **GPU** tab:
   - High temperature may cause throttling
   - High memory may cause OOM errors

5. Review **Network** tab:
   - Identify layers with problematic gradients
   - Check gradient flow through network

### Comparing Training Runs

1. Press `6` for **History** tab
2. Navigate with `j/k` to select runs
3. Note best loss and final status
4. Press `l` to switch to History mode for full data
5. Compare loss curves in **Charts** tab

## Generating Screenshots

The visual tests capture screenshots for regression testing and documentation:

```bash
# Run visual tests only
cargo test -p training-tools visual_tests -- --nocapture

# Generate screenshots with script
./scripts/capture_screenshots.sh

# Also generate PNG images
./scripts/capture_screenshots.sh --png

# Regenerate this documentation
./scripts/capture_screenshots.sh --docs
```

Screenshots are saved to `training-tools/screenshots/` in multiple formats:
- `.ansi` - ANSI escape sequences (view in terminal)
- `.html` - HTML with inline styles (view in browser)
- `.txt` - Plain text (no colors)

## Troubleshooting

### Monitor not showing data

1. Check that training run is writing to metrics.jsonl
2. Verify runs directory path is correct
3. Press `r` to refresh
4. Check that run status is not "Initializing"

### Charts are empty

1. Wait for sufficient data points (minimum ~10)
2. Switch to History mode with `l` to load all data
3. Check that metrics file is being written

### GPU stats unavailable

1. Verify nvidia-smi is installed and working
2. Check GPU permissions
3. GPU monitoring requires NVIDIA drivers

### High CPU usage

1. Increase refresh interval in config
2. Close other terminal applications
3. Reduce terminal window size
