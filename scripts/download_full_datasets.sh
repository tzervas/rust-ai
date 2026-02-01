#!/bin/bash
# Full Dataset Download Script for Tritter Training
# Downloads complete parquet datasets optimized for the hybrid predictive BitNet model
#
# Run on homelab: ssh homelab && cd /data/datasets/tritter && bash download_full_datasets.sh
#
# Target mix for 100M model training:
# - 40% Natural language (FineWeb-Edu 10B, Wikipedia)
# - 25% Code (SmolLM Stack-Edu Python/Rust/TS)
# - 15% Math (FineMath, OpenWebMath)
# - 10% AI/ML (Cosmopedia)
# - 10% Reasoning (Proof-Pile-2, AutoMathText)

set -e

# Configuration
DATA_DIR="/data/datasets/tritter/pretrain"
HF="/home/kang/.local/bin/hf"
LOG="/data/datasets/tritter/logs/download_full.log"

log() { echo -e "\033[0;32m[$(date '+%H:%M:%S')]\033[0m $1" | tee -a "$LOG"; }
warn() { echo -e "\033[1;33m[$(date '+%H:%M:%S')] WARN:\033[0m $1" | tee -a "$LOG"; }
err() { echo -e "\033[0;31m[$(date '+%H:%M:%S')] ERROR:\033[0m $1" | tee -a "$LOG"; }

mkdir -p "$DATA_DIR"
mkdir -p "$(dirname "$LOG")"
cd "$DATA_DIR"

log "=== Full Dataset Download for Tritter ==="
log "Target: Dense parquet format, ~50GB total"
log "Data dir: $DATA_DIR"
log ""

# ============================================
# 1. FineWeb-Edu 10B (Full Sample - ~28GB)
#    Primary natural language, educational content
# ============================================
log "=== 1. FineWeb-Edu 10B Sample (28GB) ==="
if [ -d "fineweb-edu-10B" ] && [ "$(find fineweb-edu-10B -name '*.parquet' 2>/dev/null | wc -l)" -ge 90 ]; then
    log "FineWeb-Edu 10B complete ($(find fineweb-edu-10B -name '*.parquet' | wc -l) parquet files)"
else
    log "Downloading FineWeb-Edu 10B sample (full)..."
    mkdir -p fineweb-edu-10B

    $HF download HuggingFaceFW/fineweb-edu \
        --repo-type dataset \
        --include "sample/10BT/*.parquet" \
        --local-dir fineweb-edu-10B

    # Flatten structure
    if [ -d "fineweb-edu-10B/sample/10BT" ]; then
        mv fineweb-edu-10B/sample/10BT/*.parquet fineweb-edu-10B/ 2>/dev/null || true
        rm -rf fineweb-edu-10B/sample 2>/dev/null || true
    fi

    log "FineWeb-Edu 10B download complete"
fi

# ============================================
# 2. SmolLM Corpus - Curated Educational Code
#    High-quality filtered code datasets
# ============================================
log ""
log "=== 2. SmolLM Corpus (Curated Code + Math) ==="

# Python code (educational, filtered)
if [ -d "smollm-python" ] && [ "$(find smollm-python -name '*.parquet' 2>/dev/null | wc -l)" -gt 0 ]; then
    log "SmolLM Python exists"
else
    log "Downloading SmolLM Python (filtered educational code)..."
    mkdir -p smollm-python
    $HF download HuggingFaceTB/smollm-corpus \
        --repo-type dataset \
        --include "python-edu/*.parquet" \
        --local-dir smollm-python 2>&1 | tee -a "$LOG" || warn "SmolLM Python failed"
fi

# Cosmopedia v2 (synthetic educational content)
if [ -d "cosmopedia-v2" ] && [ "$(find cosmopedia-v2 -name '*.parquet' 2>/dev/null | wc -l)" -gt 5 ]; then
    log "Cosmopedia v2 exists"
else
    log "Downloading Cosmopedia v2 (synthetic educational)..."
    mkdir -p cosmopedia-v2
    $HF download HuggingFaceTB/cosmopedia-v2 \
        --repo-type dataset \
        --include "data/*.parquet" \
        --local-dir cosmopedia-v2 2>&1 | tee -a "$LOG" || warn "Cosmopedia v2 failed"
fi

# ============================================
# 3. FineMath (Math reasoning - 15%)
# ============================================
log ""
log "=== 3. FineMath (Math Reasoning) ==="

if [ -d "finemath-4plus" ] && [ "$(find finemath-4plus -name '*.parquet' 2>/dev/null | wc -l)" -gt 0 ]; then
    log "FineMath 4+ exists"
else
    log "Downloading FineMath 4+ (high-quality math)..."
    mkdir -p finemath-4plus
    $HF download HuggingFaceTB/finemath \
        --repo-type dataset \
        --include "finemath-4plus/*.parquet" \
        --local-dir finemath-4plus 2>&1 | tee -a "$LOG" || {
        warn "FineMath 4+ not available, trying finemath-3plus..."
        $HF download HuggingFaceTB/finemath \
            --repo-type dataset \
            --include "finemath-3plus/*.parquet" \
            --local-dir finemath-4plus 2>&1 | tee -a "$LOG" || warn "FineMath failed"
    }
fi

# ============================================
# 4. The Stack v2 - Code (Python, Rust, TS)
# ============================================
log ""
log "=== 4. The Stack v2 (Code) ==="

# Stack Python
if [ -d "stack-v2-python" ] && [ "$(find stack-v2-python -name '*.parquet' 2>/dev/null | wc -l)" -gt 0 ]; then
    log "Stack v2 Python exists"
else
    log "Downloading Stack v2 Python..."
    mkdir -p stack-v2-python
    # Use the dedup version for cleaner training
    $HF download bigcode/the-stack-v2-dedup \
        --repo-type dataset \
        --include "data/python/*.parquet" \
        --local-dir stack-v2-python 2>&1 | tee -a "$LOG" || warn "Stack Python failed"
fi

# Stack Rust
if [ -d "stack-v2-rust" ] && [ "$(find stack-v2-rust -name '*.parquet' 2>/dev/null | wc -l)" -gt 0 ]; then
    log "Stack v2 Rust exists"
else
    log "Downloading Stack v2 Rust..."
    mkdir -p stack-v2-rust
    $HF download bigcode/the-stack-v2-dedup \
        --repo-type dataset \
        --include "data/rust/*.parquet" \
        --local-dir stack-v2-rust 2>&1 | tee -a "$LOG" || warn "Stack Rust failed"
fi

# Stack TypeScript
if [ -d "stack-v2-typescript" ] && [ "$(find stack-v2-typescript -name '*.parquet' 2>/dev/null | wc -l)" -gt 0 ]; then
    log "Stack v2 TypeScript exists"
else
    log "Downloading Stack v2 TypeScript..."
    mkdir -p stack-v2-typescript
    $HF download bigcode/the-stack-v2-dedup \
        --repo-type dataset \
        --include "data/typescript/*.parquet" \
        --local-dir stack-v2-typescript 2>&1 | tee -a "$LOG" || warn "Stack TypeScript failed"
fi

# ============================================
# 5. OpenWebMath (Additional math)
# ============================================
log ""
log "=== 5. OpenWebMath ==="

if [ -d "openwebmath" ] && [ "$(find openwebmath -name '*.parquet' 2>/dev/null | wc -l)" -gt 0 ]; then
    log "OpenWebMath exists"
else
    log "Downloading OpenWebMath..."
    mkdir -p openwebmath
    $HF download open-web-math/open-web-math \
        --repo-type dataset \
        --include "*.parquet" \
        --local-dir openwebmath 2>&1 | tee -a "$LOG" || warn "OpenWebMath failed"
fi

# ============================================
# 6. TinyStories (Testing/Validation)
# ============================================
log ""
log "=== 6. TinyStories (Quick Testing) ==="

if [ -d "tinystories" ] && [ "$(find tinystories -type f | wc -l)" -gt 0 ]; then
    log "TinyStories exists"
else
    log "Downloading TinyStories..."
    mkdir -p tinystories
    $HF download roneneldan/TinyStories \
        --repo-type dataset \
        --local-dir tinystories
fi

# ============================================
# 7. Proof-Pile-2 (Math proofs, reasoning)
# ============================================
log ""
log "=== 7. Proof-Pile-2 (Math Proofs) ==="

if [ -d "proof-pile-2" ] && [ "$(find proof-pile-2 -name '*.parquet' 2>/dev/null | wc -l)" -gt 0 ]; then
    log "Proof-Pile-2 exists"
else
    log "Downloading Proof-Pile-2 algebraic-stack..."
    mkdir -p proof-pile-2
    $HF download EleutherAI/proof-pile-2 \
        --repo-type dataset \
        --include "algebraic-stack/*.parquet" \
        --local-dir proof-pile-2 2>&1 | tee -a "$LOG" || warn "Proof-Pile-2 failed"
fi

# ============================================
# 8. AutoMathText (Synthetic math problems)
# ============================================
log ""
log "=== 8. AutoMathText ==="

if [ -d "automathtext" ] && [ "$(find automathtext -name '*.parquet' 2>/dev/null | wc -l)" -gt 0 ]; then
    log "AutoMathText exists"
else
    log "Downloading AutoMathText..."
    mkdir -p automathtext
    $HF download math-ai/AutoMathText \
        --repo-type dataset \
        --include "*.parquet" \
        --local-dir automathtext 2>&1 | tee -a "$LOG" || warn "AutoMathText failed"
fi

# ============================================
# Summary
# ============================================
log ""
log "=== Download Summary ==="
log ""

total_size=0
for dir in */; do
    if [ -d "$dir" ]; then
        name="${dir%/}"
        size=$(du -sh "$dir" 2>/dev/null | cut -f1)
        files=$(find "$dir" -name "*.parquet" 2>/dev/null | wc -l)
        bytes=$(du -sb "$dir" 2>/dev/null | cut -f1)
        total_size=$((total_size + bytes))
        log "  $name: $size ($files parquet files)"
    fi
done

log ""
log "Total: $(numfmt --to=iec $total_size 2>/dev/null || echo "$((total_size / 1024 / 1024 / 1024))G")"
log ""
log "=== Full Dataset Download Complete ==="
log "Run prepare_datasets.py to create mixed training set"
