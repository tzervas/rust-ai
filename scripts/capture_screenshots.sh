#!/bin/bash
# Capture TUI screenshots for documentation
#
# This script runs visual tests to generate screenshots, then optionally
# converts them to PNG format for embedding in documentation.
#
# Prerequisites:
# - Rust toolchain
# - Optional: aha (ANSI to HTML) - for color conversion
# - Optional: wkhtmltoimage - for PNG generation
#
# Usage:
#   ./scripts/capture_screenshots.sh              # Run tests only
#   ./scripts/capture_screenshots.sh --png        # Also generate PNGs
#   ./scripts/capture_screenshots.sh --docs       # Generate documentation

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SCREENSHOTS_DIR="$PROJECT_ROOT/training-tools/screenshots"
DOCS_DIR="$PROJECT_ROOT/training-tools/docs"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Parse arguments
GENERATE_PNG=false
GENERATE_DOCS=false

for arg in "$@"; do
    case $arg in
        --png)
            GENERATE_PNG=true
            shift
            ;;
        --docs)
            GENERATE_DOCS=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --png     Convert ANSI screenshots to PNG images"
            echo "  --docs    Generate UI documentation with screenshots"
            echo "  --help    Show this help message"
            exit 0
            ;;
    esac
done

# Step 1: Run visual tests to capture screenshots
log_info "Running visual tests to capture screenshots..."

cd "$PROJECT_ROOT"

# Ensure screenshots directory exists
mkdir -p "$SCREENSHOTS_DIR"

# Run the visual tests (with --nocapture to see output)
if cargo test -p training-tools visual_tests -- --nocapture 2>&1 | tee /tmp/visual_tests_output.txt; then
    log_success "Visual tests completed successfully"
else
    log_error "Visual tests failed"
    exit 1
fi

# Count captured screenshots
SCREENSHOT_COUNT=$(find "$SCREENSHOTS_DIR" -name "*.ansi" -o -name "*.html" 2>/dev/null | wc -l)
log_info "Captured $SCREENSHOT_COUNT screenshots in $SCREENSHOTS_DIR"

# Step 2: Convert ANSI to PNG (optional)
if [ "$GENERATE_PNG" = true ]; then
    log_info "Converting ANSI files to PNG..."

    # Check for required tools
    if ! command -v aha &> /dev/null; then
        log_warning "aha not found. Install with: sudo apt install aha"
        log_warning "Skipping PNG conversion"
    elif ! command -v wkhtmltoimage &> /dev/null; then
        log_warning "wkhtmltoimage not found. Install with: sudo apt install wkhtmltopdf"
        log_warning "Skipping PNG conversion"
    else
        PNG_DIR="$SCREENSHOTS_DIR/png"
        mkdir -p "$PNG_DIR"

        for ansi_file in "$SCREENSHOTS_DIR"/*.ansi; do
            if [ -f "$ansi_file" ]; then
                basename=$(basename "$ansi_file" .ansi)
                html_temp="/tmp/${basename}_temp.html"
                png_file="$PNG_DIR/${basename}.png"

                log_info "Converting $basename.ansi to PNG..."

                # Convert ANSI to HTML
                cat "$ansi_file" | aha --black > "$html_temp"

                # Add styling for better rendering
                sed -i 's/<body>/<body style="background-color: #1e1e2e; padding: 10px; font-family: monospace; font-size: 14px;">/' "$html_temp"

                # Convert HTML to PNG
                wkhtmltoimage --quality 100 --width 1200 "$html_temp" "$png_file"

                rm -f "$html_temp"
                log_success "Created $png_file"
            fi
        done

        PNG_COUNT=$(find "$PNG_DIR" -name "*.png" 2>/dev/null | wc -l)
        log_success "Generated $PNG_COUNT PNG images in $PNG_DIR"
    fi
fi

# Step 3: Generate documentation (optional)
if [ "$GENERATE_DOCS" = true ]; then
    log_info "Generating UI documentation..."

    mkdir -p "$DOCS_DIR"
    DOC_FILE="$DOCS_DIR/UI_GUIDE.md"

    cat > "$DOC_FILE" << 'EOF'
# Training Monitor UI Guide

This document provides a visual guide to the Training Monitor TUI (Terminal User Interface).

## Overview

The Training Monitor provides real-time visualization of training runs with multiple tabs:

1. **Dashboard** - High-level overview with health indicators
2. **Overview** - Run list with loss chart
3. **Charts** - Detailed metric charts
4. **Network** - Neural network layer visualization
5. **Analysis** - Training dynamics analysis
6. **GPU** - GPU utilization and thermals
7. **History** - Historical training runs
8. **Help** - Keyboard shortcuts

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Tab` | Next tab |
| `Shift+Tab` | Previous tab |
| `0-7` | Jump to specific tab |
| `j/k` | Navigate runs up/down |
| `l` | Toggle Live/History mode |
| `[/]` | Cycle chart types |
| `r` | Refresh data |
| `?` or `F1` | Show help overlay |
| `q` or `Esc` | Quit |

## Color Legend

### Training Phases

| Color | Phase | Description |
|-------|-------|-------------|
| Orange | WARMUP | Collecting baseline statistics |
| Blue | FULL | Full forward/backward passes |
| Green | PREDICT | Gradient prediction active |
| Magenta | CORRECT | Correcting prediction errors |

### Run Status

| Color | Status |
|-------|--------|
| Bright Green | Running |
| Light Blue | Completed |
| Red | Failed |
| Yellow | Paused |
| Gray | Cancelled |

### Metrics

| Color | Metric |
|-------|--------|
| Cyan | Loss curve |
| Green | Gradient norm |
| Orange | Step time |
| Light Blue | Prediction confidence |

## Screenshots

EOF

    # Add screenshots to documentation
    for ansi_file in "$SCREENSHOTS_DIR"/*.ansi; do
        if [ -f "$ansi_file" ]; then
            basename=$(basename "$ansi_file" .ansi)
            # Convert underscore to space and capitalize for title
            title=$(echo "$basename" | sed 's/_/ /g' | sed 's/\b\(.\)/\u\1/g')

            echo "" >> "$DOC_FILE"
            echo "### $title" >> "$DOC_FILE"
            echo "" >> "$DOC_FILE"
            echo '```' >> "$DOC_FILE"
            # Strip ANSI codes for markdown display
            sed 's/\x1b\[[0-9;]*m//g' "$ansi_file" >> "$DOC_FILE"
            echo '```' >> "$DOC_FILE"
            echo "" >> "$DOC_FILE"
        fi
    done

    # Add example workflows
    cat >> "$DOC_FILE" << 'EOF'

## Example Workflows

### Monitoring a Training Run

1. Start the monitor: `train-monitor --runs-dir ./runs`
2. Use `Tab` to navigate to **Dashboard** for overview
3. Use `j/k` to select the active run
4. Press `2` for **Charts** tab to see loss curve
5. Use `[/]` to cycle between different chart types
6. Press `5` for **GPU** tab to check thermals

### Investigating Issues

1. Check **Dashboard** for overall health indicators
2. If loss is oscillating, check **Analysis** tab for recommendations
3. Use **Charts** tab to visualize gradient norm for explosions
4. Check **GPU** tab for thermal throttling
5. Review **Network** tab for per-layer gradient flow

### Reviewing History

1. Press `6` for **History** tab
2. Navigate with `j/k` to select past runs
3. Press `l` to switch to History mode for full data
4. Use **Charts** to compare metrics across runs

## Generating Screenshots

To regenerate screenshots for this documentation:

```bash
./scripts/capture_screenshots.sh --docs
```

To also generate PNG images:

```bash
./scripts/capture_screenshots.sh --png --docs
```
EOF

    log_success "Generated documentation at $DOC_FILE"
fi

# Summary
echo ""
log_info "=== Summary ==="
echo "Screenshots directory: $SCREENSHOTS_DIR"
echo "Screenshots captured: $SCREENSHOT_COUNT"
if [ "$GENERATE_DOCS" = true ]; then
    echo "Documentation: $DOC_FILE"
fi

log_success "Done!"
