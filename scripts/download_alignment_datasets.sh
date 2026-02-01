#!/bin/bash
# Ethics & Alignment Dataset Download Script
# Teaches the model right/wrong, good/evil through curated data
#
# Datasets for value learning:
# 1. Helpful vs Harmful distinction
# 2. Truthful vs Deceptive content
# 3. Ethical reasoning and moral philosophy
# 4. Safe vs Unsafe code practices
# 5. Constructive vs Destructive communication

set -e

DATA_DIR="/data/datasets/tritter/alignment"
HF="/home/kang/.local/bin/hf"
LOG="/data/datasets/tritter/logs/alignment_download.log"

log() { echo -e "\033[0;32m[$(date '+%H:%M:%S')]\033[0m $1" | tee -a "$LOG"; }
warn() { echo -e "\033[1;33m[$(date '+%H:%M:%S')] WARN:\033[0m $1" | tee -a "$LOG"; }

mkdir -p "$DATA_DIR"
mkdir -p "$(dirname "$LOG")"

log "=== Alignment & Ethics Dataset Download ==="
log "Teaching right/wrong, good/evil through data curation"
log ""

# ============================================
# 1. Anthropic HH (Helpful and Harmless)
#    Core alignment: helpful vs harmful responses
# ============================================
log "=== 1. Anthropic Helpful-Harmless (already have) ==="
if [ -d "$DATA_DIR/anthropic-hh" ] && [ "$(find "$DATA_DIR/anthropic-hh" -type f | wc -l)" -gt 0 ]; then
    log "Anthropic HH exists ($(du -sh "$DATA_DIR/anthropic-hh" | cut -f1))"
else
    log "Downloading Anthropic HH..."
    mkdir -p "$DATA_DIR/anthropic-hh"
    $HF download Anthropic/hh-rlhf \
        --repo-type dataset \
        --local-dir "$DATA_DIR/anthropic-hh" || warn "Anthropic HH failed"
fi

# ============================================
# 2. TruthfulQA - Truthfulness benchmark
#    Distinguishes true from false claims
# ============================================
log ""
log "=== 2. TruthfulQA (Truthfulness) ==="
if [ -d "$DATA_DIR/truthfulqa" ] && [ "$(find "$DATA_DIR/truthfulqa" -type f | wc -l)" -gt 0 ]; then
    log "TruthfulQA exists"
else
    log "Downloading TruthfulQA..."
    mkdir -p "$DATA_DIR/truthfulqa"
    $HF download truthful_qa \
        --repo-type dataset \
        --local-dir "$DATA_DIR/truthfulqa" || warn "TruthfulQA failed"
fi

# ============================================
# 3. ETHICS Dataset (Hendrycks)
#    Moral reasoning across scenarios
# ============================================
log ""
log "=== 3. ETHICS Dataset (Moral Reasoning) ==="
if [ -d "$DATA_DIR/ethics" ] && [ "$(find "$DATA_DIR/ethics" -type f | wc -l)" -gt 0 ]; then
    log "ETHICS exists"
else
    log "Downloading ETHICS..."
    mkdir -p "$DATA_DIR/ethics"
    $HF download hendrycks/ethics \
        --repo-type dataset \
        --local-dir "$DATA_DIR/ethics" || warn "ETHICS failed"
fi

# ============================================
# 4. PKU SafeRLHF - Safety alignment
#    Safe vs unsafe response pairs
# ============================================
log ""
log "=== 4. PKU SafeRLHF (Safety) ==="
if [ -d "$DATA_DIR/saferlhf" ] && [ "$(find "$DATA_DIR/saferlhf" -type f | wc -l)" -gt 0 ]; then
    log "SafeRLHF exists"
else
    log "Downloading PKU SafeRLHF..."
    mkdir -p "$DATA_DIR/saferlhf"
    $HF download PKU-Alignment/PKU-SafeRLHF \
        --repo-type dataset \
        --local-dir "$DATA_DIR/saferlhf" || warn "SafeRLHF failed"
fi

# ============================================
# 5. Prosocial Dialog - Constructive communication
#    Teaches helpful, prosocial responses
# ============================================
log ""
log "=== 5. Prosocial Dialog ==="
if [ -d "$DATA_DIR/prosocial" ] && [ "$(find "$DATA_DIR/prosocial" -type f | wc -l)" -gt 0 ]; then
    log "Prosocial Dialog exists"
else
    log "Downloading Prosocial Dialog..."
    mkdir -p "$DATA_DIR/prosocial"
    $HF download allenai/prosocial-dialog \
        --repo-type dataset \
        --local-dir "$DATA_DIR/prosocial" || warn "Prosocial Dialog failed"
fi

# ============================================
# 6. Do-Not-Answer - Refusal patterns
#    When and how to refuse harmful requests
# ============================================
log ""
log "=== 6. Do-Not-Answer (Refusal Patterns) ==="
if [ -d "$DATA_DIR/do-not-answer" ] && [ "$(find "$DATA_DIR/do-not-answer" -type f | wc -l)" -gt 0 ]; then
    log "Do-Not-Answer exists"
else
    log "Downloading Do-Not-Answer..."
    mkdir -p "$DATA_DIR/do-not-answer"
    $HF download LibrAI/do-not-answer \
        --repo-type dataset \
        --local-dir "$DATA_DIR/do-not-answer" || warn "Do-Not-Answer failed"
fi

# ============================================
# 7. Moral Stories - Narrative ethics
#    Stories with moral lessons
# ============================================
log ""
log "=== 7. Moral Stories ==="
if [ -d "$DATA_DIR/moral-stories" ] && [ "$(find "$DATA_DIR/moral-stories" -type f | wc -l)" -gt 0 ]; then
    log "Moral Stories exists"
else
    log "Downloading Moral Stories..."
    mkdir -p "$DATA_DIR/moral-stories"
    $HF download demelin/moral_stories \
        --repo-type dataset \
        --local-dir "$DATA_DIR/moral-stories" || warn "Moral Stories failed"
fi

# ============================================
# 8. HarmBench - Harmful behavior detection
#    Understanding what constitutes harm
# ============================================
log ""
log "=== 8. HarmBench ==="
if [ -d "$DATA_DIR/harmbench" ] && [ "$(find "$DATA_DIR/harmbench" -type f | wc -l)" -gt 0 ]; then
    log "HarmBench exists"
else
    log "Downloading HarmBench..."
    mkdir -p "$DATA_DIR/harmbench"
    $HF download cais/HarmBench \
        --repo-type dataset \
        --local-dir "$DATA_DIR/harmbench" || warn "HarmBench failed"
fi

# ============================================
# Summary
# ============================================
log ""
log "=== Alignment Dataset Summary ==="
log ""

total_size=0
for dir in "$DATA_DIR"/*/; do
    if [ -d "$dir" ]; then
        name=$(basename "$dir")
        size=$(du -sh "$dir" 2>/dev/null | cut -f1)
        files=$(find "$dir" -type f 2>/dev/null | wc -l)
        bytes=$(du -sb "$dir" 2>/dev/null | cut -f1)
        total_size=$((total_size + bytes))
        log "  $name: $size ($files files)"
    fi
done

log ""
log "Total alignment data: $(numfmt --to=iec $total_size 2>/dev/null || echo "$((total_size / 1024 / 1024))M")"
log ""
log "=== Alignment Download Complete ==="
log ""
log "Value learning categories covered:"
log "  - Helpful vs Harmful (Anthropic HH)"
log "  - True vs False (TruthfulQA)"
log "  - Ethical vs Unethical (ETHICS)"
log "  - Safe vs Unsafe (SafeRLHF)"
log "  - Constructive vs Destructive (Prosocial)"
log "  - Accept vs Refuse (Do-Not-Answer)"
log "  - Right vs Wrong narratives (Moral Stories)"
log "  - Harm recognition (HarmBench)"
