#!/bin/bash
# Dataset Download and Preparation Script for Tritter Training
# Run on homelab: ssh homelab && bash /path/to/download_datasets.sh

set -e

# Configuration
DATA_DIR="/data/datasets/tritter/datasets"
HF_CLI="/home/kang/.local/bin/hf"
LOG_FILE="/data/datasets/tritter/download.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

warn() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1" | tee -a "$LOG_FILE"
}

# Create directories
mkdir -p "$DATA_DIR"
mkdir -p /data/datasets/tritter/processed
mkdir -p /data/datasets/tritter/tokenized

log "=== Tritter Dataset Download Script ==="
log "Data directory: $DATA_DIR"
log "Log file: $LOG_FILE"

# Check HF CLI
if ! command -v "$HF_CLI" &> /dev/null; then
    error "HuggingFace CLI not found at $HF_CLI"
    exit 1
fi

log "HuggingFace CLI found: $HF_CLI"
$HF_CLI auth whoami

# ============================================
# 1. FineWeb-Edu 10B Sample (Primary - 40%)
# ============================================
log ""
log "=== Downloading FineWeb-Edu 10B Sample (~28GB) ==="
FINEWEB_DIR="$DATA_DIR/fineweb-edu-10B"

if [ -d "$FINEWEB_DIR" ] && [ "$(find "$FINEWEB_DIR" -name '*.parquet' | wc -l)" -gt 10 ]; then
    log "FineWeb-Edu 10B already exists with $(find "$FINEWEB_DIR" -name '*.parquet' | wc -l) parquet files"
else
    log "Downloading FineWeb-Edu 10B sample..."
    $HF_CLI download HuggingFaceFW/fineweb-edu \
        --include "sample/10BT/*" \
        --local-dir "$FINEWEB_DIR" \
        --local-dir-use-symlinks False

    # Flatten directory structure if needed
    if [ -d "$FINEWEB_DIR/sample/10BT" ]; then
        mv "$FINEWEB_DIR/sample/10BT"/* "$FINEWEB_DIR/" 2>/dev/null || true
        rm -rf "$FINEWEB_DIR/sample" 2>/dev/null || true
    fi

    log "FineWeb-Edu 10B download complete"
fi

# Create metadata
cat > "$FINEWEB_DIR/metadata.json" << 'EOF'
{
  "name": "fineweb-edu-10B",
  "source": "HuggingFaceFW/fineweb-edu",
  "subset": "sample-10BT",
  "tokens": "10B",
  "license": "ODC-BY",
  "purpose": "base_knowledge",
  "mix_ratio": 0.40
}
EOF

# ============================================
# 2. Stack-Edu (Code - 25%)
# ============================================
log ""
log "=== Downloading Stack-Edu (Code datasets) ==="

# Python code
STACK_PYTHON_DIR="$DATA_DIR/stack-python"
if [ -d "$STACK_PYTHON_DIR" ] && [ "$(find "$STACK_PYTHON_DIR" -name '*.parquet' 2>/dev/null | wc -l)" -gt 0 ]; then
    log "Stack Python already exists"
else
    log "Downloading Stack Python subset..."
    $HF_CLI download bigcode/the-stack-v2-train-smol-ids \
        --include "data/python/*" \
        --local-dir "$STACK_PYTHON_DIR" \
        --local-dir-use-symlinks False 2>/dev/null || warn "Stack Python download failed - may need different source"
fi

# Rust code
STACK_RUST_DIR="$DATA_DIR/stack-rust"
if [ -d "$STACK_RUST_DIR" ] && [ "$(find "$STACK_RUST_DIR" -name '*.parquet' 2>/dev/null | wc -l)" -gt 0 ]; then
    log "Stack Rust already exists"
else
    log "Downloading Stack Rust subset..."
    $HF_CLI download bigcode/the-stack-v2-train-smol-ids \
        --include "data/rust/*" \
        --local-dir "$STACK_RUST_DIR" \
        --local-dir-use-symlinks False 2>/dev/null || warn "Stack Rust download failed - may need different source"
fi

# TypeScript code
STACK_TS_DIR="$DATA_DIR/stack-typescript"
if [ -d "$STACK_TS_DIR" ] && [ "$(find "$STACK_TS_DIR" -name '*.parquet' 2>/dev/null | wc -l)" -gt 0 ]; then
    log "Stack TypeScript already exists"
else
    log "Downloading Stack TypeScript subset..."
    $HF_CLI download bigcode/the-stack-v2-train-smol-ids \
        --include "data/typescript/*" \
        --local-dir "$STACK_TS_DIR" \
        --local-dir-use-symlinks False 2>/dev/null || warn "Stack TypeScript download failed"
fi

# SmolLM Stack-Edu (curated, better quality)
SMOL_STACK_DIR="$DATA_DIR/smollm-stack-edu"
if [ -d "$SMOL_STACK_DIR" ] && [ "$(find "$SMOL_STACK_DIR" -name '*.parquet' 2>/dev/null | wc -l)" -gt 0 ]; then
    log "SmolLM Stack-Edu already exists"
else
    log "Downloading SmolLM Stack-Edu (curated code)..."
    $HF_CLI download HuggingFaceTB/smollm-corpus \
        --include "stack-edu-python/*" \
        --local-dir "$SMOL_STACK_DIR" \
        --local-dir-use-symlinks False 2>/dev/null || warn "SmolLM Stack-Edu download failed"
fi

# ============================================
# 3. FineMath (Math & Reasoning - 15%)
# ============================================
log ""
log "=== Downloading FineMath ==="
FINEMATH_DIR="$DATA_DIR/finemath"

if [ -d "$FINEMATH_DIR" ] && [ "$(find "$FINEMATH_DIR" -name '*.parquet' 2>/dev/null | wc -l)" -gt 0 ]; then
    log "FineMath already exists"
else
    log "Downloading FineMath (math reasoning)..."
    # Try the 4+ score subset for highest quality
    $HF_CLI download HuggingFaceTB/finemath \
        --include "finemath-4plus/*" \
        --local-dir "$FINEMATH_DIR" \
        --local-dir-use-symlinks False 2>/dev/null || {
        warn "FineMath 4+ failed, trying full dataset..."
        $HF_CLI download HuggingFaceTB/finemath \
            --local-dir "$FINEMATH_DIR" \
            --local-dir-use-symlinks False 2>/dev/null || warn "FineMath download failed"
    }
fi

# ============================================
# 4. TinyStories (Testing/Validation)
# ============================================
log ""
log "=== Downloading TinyStories (for testing) ==="
TINYSTORIES_DIR="$DATA_DIR/tinystories"

if [ -d "$TINYSTORIES_DIR" ] && [ "$(find "$TINYSTORIES_DIR" -type f | wc -l)" -gt 0 ]; then
    log "TinyStories already exists"
else
    log "Downloading TinyStories..."
    $HF_CLI download roneneldan/TinyStories \
        --local-dir "$TINYSTORIES_DIR" \
        --local-dir-use-symlinks False
fi

# ============================================
# 5. Cosmopedia (Already have 2.6GB - keep it)
# ============================================
log ""
log "=== Checking Cosmopedia ==="
COSMOPEDIA_DIR="$DATA_DIR/cosmopedia"
if [ -d "$COSMOPEDIA_DIR" ]; then
    log "Cosmopedia exists: $(du -sh "$COSMOPEDIA_DIR" | cut -f1)"
else
    log "Downloading Cosmopedia sample..."
    $HF_CLI download HuggingFaceTB/cosmopedia \
        --include "data/auto_math_text/*" \
        --local-dir "$COSMOPEDIA_DIR" \
        --local-dir-use-symlinks False 2>/dev/null || warn "Cosmopedia download failed"
fi

# ============================================
# 6. ArXiv ML Papers (AI/ML domain - 10%)
# ============================================
log ""
log "=== Downloading ArXiv ML subset ==="
ARXIV_DIR="$DATA_DIR/arxiv-ml"

if [ -d "$ARXIV_DIR" ] && [ "$(find "$ARXIV_DIR" -name '*.parquet' 2>/dev/null | wc -l)" -gt 0 ]; then
    log "ArXiv ML already exists"
else
    log "Downloading ArXiv ML papers..."
    $HF_CLI download togethercomputer/RedPajama-Data-V2 \
        --include "arxiv/*" \
        --local-dir "$ARXIV_DIR" \
        --local-dir-use-symlinks False 2>/dev/null || warn "ArXiv download failed - large dataset"
fi

# ============================================
# Summary
# ============================================
log ""
log "=== Download Summary ==="
log ""

for dir in "$DATA_DIR"/*/; do
    if [ -d "$dir" ]; then
        name=$(basename "$dir")
        size=$(du -sh "$dir" 2>/dev/null | cut -f1)
        files=$(find "$dir" -type f \( -name "*.parquet" -o -name "*.json" -o -name "*.jsonl" \) 2>/dev/null | wc -l)
        log "$name: $size ($files files)"
    fi
done

log ""
log "=== Dataset Download Complete ==="
log "Total size: $(du -sh "$DATA_DIR" | cut -f1)"
