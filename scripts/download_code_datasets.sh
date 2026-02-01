#!/bin/bash
# Code Dataset Download Script for Tritter Training
# Downloads Rust, TypeScript, and Go code from public HuggingFace sources
#
# Usage: bash /home/kang/Documents/projects/rust-ai/scripts/download_code_datasets.sh
#
# Target datasets:
# - codeparrot/github-code (115M files, 1TB, includes Rust/TS/Go)
# - bigcode/starcoderdata (783GB code across 86 languages + Rust/TS/Go)
# - ammarnasr/the-stack-rust-clean (993k Rust files, deduplicated)
# - bigcode/CodeSearchNet (Go, Python, Java, PHP, Ruby, JavaScript)
# - codeparrot/codeparrot-clean (5.17M diverse code files)

set -e

# ============================================
# Configuration
# ============================================
DATA_DIR="/data/datasets/tritter/pretrain"
CODE_DIR="$DATA_DIR/code"
HF_CLI="/home/kang/.local/bin/hf"
LOG_DIR="/data/datasets/tritter/logs"
LOG_FILE="$LOG_DIR/download_code_datasets_$(date +%Y%m%d_%H%M%S).log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================
# Utility Functions
# ============================================
log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

warn() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARN:${NC} $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1" | tee -a "$LOG_FILE"
}

success() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')] SUCCESS:${NC} $1" | tee -a "$LOG_FILE"
}

# Get directory size in human readable format
get_size() {
    du -sh "$1" 2>/dev/null | cut -f1 || echo "unknown"
}

# Count files by type
count_files() {
    find "$1" -type f \( -name "*.parquet" -o -name "*.json" -o -name "*.jsonl" -o -name "*.arrow" \) 2>/dev/null | wc -l || echo 0
}

# ============================================
# Initialization
# ============================================
mkdir -p "$CODE_DIR"
mkdir -p "$LOG_DIR"

log "=========================================="
log "Code Dataset Download Script for Tritter"
log "=========================================="
log ""
log "Configuration:"
log "  Data directory: $DATA_DIR"
log "  Code directory: $CODE_DIR"
log "  HF CLI: $HF_CLI"
log "  Log file: $LOG_FILE"
log ""

# Check HF CLI
if ! command -v "$HF_CLI" &> /dev/null; then
    error "HuggingFace CLI not found at $HF_CLI"
    exit 1
fi

log "HuggingFace CLI found: $HF_CLI"
log ""

# ============================================
# 1. CodeParrot GitHub Code (Universal - All Languages)
# ============================================
log "========== 1. CodeParrot GitHub Code (115M files, 1TB) =========="
log "Contains: Rust, TypeScript, Go, Python, Java, C/C++, etc."
log "Source: codeparrot/github-code (30 languages, MIT/Apache/GPL/Other licenses)"
log ""

CODEPARROT_DIR="$CODE_DIR/codeparrot-github-code"
if [ -d "$CODEPARROT_DIR" ] && [ "$(count_files "$CODEPARROT_DIR")" -gt 100 ]; then
    log "CodeParrot GitHub Code exists: $(get_size "$CODEPARROT_DIR") ($(count_files "$CODEPARROT_DIR") files)"
else
    log "Downloading CodeParrot GitHub Code..."
    mkdir -p "$CODEPARROT_DIR"

    if $HF_CLI download codeparrot/github-code \
        --repo-type dataset \
        --local-dir "$CODEPARROT_DIR" \
         \
        2>&1 | tee -a "$LOG_FILE"; then
        success "CodeParrot GitHub Code downloaded: $(get_size "$CODEPARROT_DIR")"
    else
        warn "CodeParrot GitHub Code download failed - may need manual intervention"
    fi
fi

# Create metadata
cat > "$CODEPARROT_DIR/metadata.json" << 'EOF' 2>/dev/null || true
{
  "name": "codeparrot-github-code",
  "source": "codeparrot/github-code",
  "description": "115M code files from GitHub",
  "size_gb": 1000,
  "languages": ["Rust", "TypeScript", "Go", "Python", "Java", "C", "C++", "JavaScript", "Shell", "Ruby", "PHP"],
  "license": "Mixed (MIT, Apache, GPL, etc)",
  "access": "Public - no authentication required",
  "format": "parquet/json"
}
EOF

log ""

# ============================================
# 2. BigCode StarCoder Data (786GB, 86 languages)
# ============================================
log "========== 2. BigCode StarCoder Data (786GB, 86 languages) =========="
log "Contains: Rust, TypeScript, Go + many others"
log "Source: bigcode/starcoderdata (requires Terms of Use agreement)"
log ""

STARCODE_DIR="$CODE_DIR/starcoderdata"
if [ -d "$STARCODE_DIR" ] && [ "$(count_files "$STARCODE_DIR")" -gt 100 ]; then
    log "StarCoder Data exists: $(get_size "$STARCODE_DIR") ($(count_files "$STARCODE_DIR") files)"
else
    log "Note: StarCoder data requires accepting Terms of Use"
    log "Attempting download of StarCoder Rust subset..."
    mkdir -p "$STARCODE_DIR"

    # Try downloading Rust subset
    if $HF_CLI download bigcode/starcoderdata \
        --repo-type dataset \
        --include "rust/*" \
        --local-dir "$STARCODE_DIR" \
         \
        2>&1 | tee -a "$LOG_FILE"; then
        log "StarCoder Rust subset downloaded"
    else
        warn "StarCoder Rust download may need Terms of Use acceptance"
    fi

    # Try downloading TypeScript subset
    if $HF_CLI download bigcode/starcoderdata \
        --repo-type dataset \
        --include "typescript/*" \
        --local-dir "$STARCODE_DIR" \
         \
        2>&1 | tee -a "$LOG_FILE"; then
        log "StarCoder TypeScript subset downloaded"
    else
        warn "StarCoder TypeScript download may need Terms of Use acceptance"
    fi

    # Try downloading Go subset
    if $HF_CLI download bigcode/starcoderdata \
        --repo-type dataset \
        --include "go/*" \
        --local-dir "$STARCODE_DIR" \
         \
        2>&1 | tee -a "$LOG_FILE"; then
        log "StarCoder Go subset downloaded"
    else
        warn "StarCoder Go download may need Terms of Use acceptance"
    fi
fi

# Create metadata
cat > "$STARCODE_DIR/metadata.json" << 'EOF' 2>/dev/null || true
{
  "name": "starcoderdata",
  "source": "bigcode/starcoderdata",
  "description": "StarCoder training data - 86 programming languages",
  "size_gb": 786,
  "languages": ["Rust", "TypeScript", "Go", "Python", "Java", "C", "C++", "JavaScript"],
  "includes": ["code", "github-issues", "jupyter-notebooks", "git-commits"],
  "license": "OpenRAIL",
  "access": "Public - requires Terms of Use agreement",
  "format": "parquet"
}
EOF

log ""

# ============================================
# 3. Stack Rust Clean (993k Rust files)
# ============================================
log "========== 3. The Stack - Rust Clean (993k Rust files) =========="
log "Source: ammarnasr/the-stack-rust-clean"
log "Deduplicated and cleaned Rust code from GitHub"
log ""

RUST_CLEAN_DIR="$CODE_DIR/the-stack-rust-clean"
if [ -d "$RUST_CLEAN_DIR" ] && [ "$(count_files "$RUST_CLEAN_DIR")" -gt 100 ]; then
    log "Rust Clean dataset exists: $(get_size "$RUST_CLEAN_DIR") ($(count_files "$RUST_CLEAN_DIR") files)"
else
    log "Downloading Rust Clean dataset..."
    mkdir -p "$RUST_CLEAN_DIR"

    if $HF_CLI download ammarnasr/the-stack-rust-clean \
        --repo-type dataset \
        --local-dir "$RUST_CLEAN_DIR" \
         \
        2>&1 | tee -a "$LOG_FILE"; then
        success "Rust Clean dataset downloaded: $(get_size "$RUST_CLEAN_DIR")"
    else
        warn "Rust Clean dataset download failed"
    fi
fi

# Create metadata
cat > "$RUST_CLEAN_DIR/metadata.json" << 'EOF' 2>/dev/null || true
{
  "name": "the-stack-rust-clean",
  "source": "ammarnasr/the-stack-rust-clean",
  "description": "Deduplicated and cleaned Rust code from GitHub",
  "files": 993000,
  "split": {
    "train": 900000,
    "validation": 50000,
    "test": 50000
  },
  "language": "Rust",
  "format": "parquet",
  "columns": ["hexsha", "size", "content", "avg_line_length", "max_line_length", "alphanum_fraction"],
  "license": "OpenRAIL",
  "access": "Public - no authentication required"
}
EOF

log ""

# ============================================
# 4. BigCode CodeSearchNet (Go + others)
# ============================================
log "========== 4. BigCode CodeSearchNet (Go focus) =========="
log "Source: bigcode/codesearchnet"
log "Contains: Go, Python, Java, PHP, Ruby, JavaScript"
log ""

CODESEARCHNET_DIR="$CODE_DIR/codesearchnet"
if [ -d "$CODESEARCHNET_DIR" ] && [ "$(count_files "$CODESEARCHNET_DIR")" -gt 50 ]; then
    log "CodeSearchNet exists: $(get_size "$CODESEARCHNET_DIR") ($(count_files "$CODESEARCHNET_DIR") files)"
else
    log "Downloading CodeSearchNet..."
    mkdir -p "$CODESEARCHNET_DIR"

    if $HF_CLI download bigcode/codesearchnet \
        --repo-type dataset \
        --local-dir "$CODESEARCHNET_DIR" \
         \
        2>&1 | tee -a "$LOG_FILE"; then
        success "CodeSearchNet downloaded: $(get_size "$CODESEARCHNET_DIR")"
    else
        warn "CodeSearchNet download failed - may not be available or requires authentication"
    fi
fi

# Create metadata
cat > "$CODESEARCHNET_DIR/metadata.json" << 'EOF' 2>/dev/null || true
{
  "name": "codesearchnet",
  "source": "bigcode/codesearchnet",
  "description": "Code search and clone detection benchmark",
  "languages": ["Go", "Python", "Java", "PHP", "Ruby", "JavaScript"],
  "format": "parquet",
  "license": "MIT",
  "access": "Public - no authentication required"
}
EOF

log ""

# ============================================
# 5. CodeParrot Clean (5.17M files, diverse)
# ============================================
log "========== 5. CodeParrot Clean (5.17M diverse code files) =========="
log "Source: codeparrot/codeparrot-clean"
log "General-purpose code dataset across multiple languages"
log ""

CODEPARROT_CLEAN_DIR="$CODE_DIR/codeparrot-clean"
if [ -d "$CODEPARROT_CLEAN_DIR" ] && [ "$(count_files "$CODEPARROT_CLEAN_DIR")" -gt 100 ]; then
    log "CodeParrot Clean exists: $(get_size "$CODEPARROT_CLEAN_DIR") ($(count_files "$CODEPARROT_CLEAN_DIR") files)"
else
    log "Downloading CodeParrot Clean..."
    mkdir -p "$CODEPARROT_CLEAN_DIR"

    if $HF_CLI download codeparrot/codeparrot-clean \
        --repo-type dataset \
        --local-dir "$CODEPARROT_CLEAN_DIR" \
         \
        2>&1 | tee -a "$LOG_FILE"; then
        success "CodeParrot Clean downloaded: $(get_size "$CODEPARROT_CLEAN_DIR")"
    else
        warn "CodeParrot Clean download failed"
    fi
fi

# Create metadata
cat > "$CODEPARROT_CLEAN_DIR/metadata.json" << 'EOF' 2>/dev/null || true
{
  "name": "codeparrot-clean",
  "source": "codeparrot/codeparrot-clean",
  "description": "Cleaned and deduplicated code dataset",
  "files": 5170000,
  "languages": ["Python", "JavaScript", "Go", "Rust", "TypeScript", "Java", "C++", "Shell"],
  "format": "parquet",
  "license": "OpenRAIL",
  "access": "Public - no authentication required"
}
EOF

log ""

# ============================================
# Alternative Fallback Sources (if above fail)
# ============================================
log "========== Alternative/Fallback Sources =========="
log ""
log "If primary sources fail, consider these alternatives:"
log ""
log "1. FALLBACK: tokyotech-llm/swallow-code-v2"
log "   - 147M items, large-scale multi-language code"
log "   - URL: https://huggingface.co/datasets/tokyotech-llm/swallow-code-v2"
log ""
log "2. FALLBACK: codeparrot-code-clones (clone detection)"
log "   - URL: https://huggingface.co/datasets/codeparrot/codeparrot-code-clones"
log ""
log "3. FALLBACK: NTU-NLP-sg/xCodeEval (cross-language evaluation)"
log "   - URL: https://huggingface.co/datasets/NTU-NLP-sg/xCodeEval"
log ""
log "4. LOCAL GITHUB: Clone repositories directly from GitHub"
log "   - Recommended for specific frameworks (Tokio, Bevy, Deno, etc.)"
log ""

# ============================================
# Script for downloading from Stack v2 if auth available
# ============================================
cat > "$CODE_DIR/STACK_V2_AUTH.md" << 'EOF'
# Stack v2 Dataset (Requires Authentication)

If you have HuggingFace Pro access or can accept the gated dataset terms:

## Dataset URL
https://huggingface.co/datasets/bigcode/the-stack-v2-dedup

## Access Steps
1. Visit the dataset page (requires HF account)
2. Accept the dataset terms of use
3. Generate or use your HF token from https://huggingface.co/settings/tokens
4. Authenticate: hf auth login
5. Download:

```bash
HF="/home/kang/.local/bin/hf"
STACK_V2_DIR="/data/datasets/tritter/pretrain/code/stack-v2"
mkdir -p "$STACK_V2_DIR"

# Download Rust subset
$HF download bigcode/the-stack-v2-dedup \
    --repo-type dataset \
    --include "data/rust/*" \
    --local-dir "$STACK_V2_DIR" \
    

# Download TypeScript subset
$HF download bigcode/the-stack-v2-dedup \
    --repo-type dataset \
    --include "data/typescript/*" \
    --local-dir "$STACK_V2_DIR" \
    

# Download Go subset
$HF download bigcode/the-stack-v2-dedup \
    --repo-type dataset \
    --include "data/go/*" \
    --local-dir "$STACK_V2_DIR" \
    
```

## Dataset Stats
- Rust: ~50GB
- TypeScript: ~40GB
- Go: ~35GB
- Total available: 3TB across all languages
EOF

log "Stack v2 authentication guide written to: $CODE_DIR/STACK_V2_AUTH.md"
log ""

# ============================================
# Summary
# ============================================
log "========== Download Summary =========="
log ""

total_size=0
total_files=0

for dir in "$CODE_DIR"/*/; do
    if [ -d "$dir" ]; then
        name=$(basename "$dir")
        size=$(get_size "$dir")
        files=$(count_files "$dir")

        if [ "$files" -gt 0 ]; then
            log "✓ $name: $size ($files files)"
            # Note: size command may not work for all filesystem types
            total_files=$((total_files + files))
        else
            log "○ $name: pending/empty"
        fi
    fi
done

log ""
log "Total files downloaded: $total_files"
log "Total size: $(get_size "$CODE_DIR")"
log ""

# ============================================
# Generate dataset index
# ============================================
log "Generating dataset index..."

cat > "$CODE_DIR/README.md" << 'EOF'
# Code Datasets for Tritter Training

This directory contains code datasets downloaded from public HuggingFace sources.

## Available Datasets

### 1. CodeParrot GitHub Code
- **Size**: ~1TB (estimated)
- **Files**: 115M
- **Languages**: 30 (including Rust, TypeScript, Go)
- **Format**: Parquet
- **License**: Mixed (MIT, Apache, GPL, etc.)
- **Source**: https://huggingface.co/datasets/codeparrot/github-code
- **Access**: Public - no authentication required

### 2. StarCoder Data
- **Size**: ~786GB
- **Languages**: 86 (including Rust, TypeScript, Go)
- **Format**: Parquet
- **Components**: Code, GitHub Issues, Jupyter, Git Commits
- **License**: OpenRAIL
- **Source**: https://huggingface.co/datasets/bigcode/starcoderdata
- **Access**: Public - requires Terms of Use agreement

### 3. The Stack - Rust Clean
- **Size**: ~5-10GB (estimated)
- **Files**: 993,000 (900k train, 50k val, 50k test)
- **Language**: Rust only
- **Format**: Parquet
- **License**: OpenRAIL
- **Source**: https://huggingface.co/datasets/ammarnasr/the-stack-rust-clean
- **Access**: Public - no authentication required
- **Columns**: hexsha, size, content, avg_line_length, max_line_length, alphanum_fraction

### 4. CodeSearchNet
- **Size**: Variable
- **Languages**: Go, Python, Java, PHP, Ruby, JavaScript
- **Format**: Parquet
- **License**: MIT
- **Source**: https://huggingface.co/datasets/bigcode/codesearchnet
- **Access**: Public - no authentication required

### 5. CodeParrot Clean
- **Size**: Variable
- **Files**: 5.17M
- **Languages**: Multiple (Rust, TypeScript, Go, etc.)
- **Format**: Parquet
- **License**: OpenRAIL
- **Source**: https://huggingface.co/datasets/codeparrot/codeparrot-clean
- **Access**: Public - no authentication required

## Dataset Statistics

Run the following to get dataset statistics:

```bash
CODE_DIR="/data/datasets/tritter/pretrain/code"

# Count files by dataset
for dir in "$CODE_DIR"/*/; do
    name=$(basename "$dir")
    count=$(find "$dir" -type f \( -name "*.parquet" -o -name "*.json" -o -name "*.jsonl" \) 2>/dev/null | wc -l)
    size=$(du -sh "$dir" 2>/dev/null | cut -f1)
    echo "$name: $size ($count files)"
done
```

## Processing

To process these datasets for training:

1. **Extract Rust, TypeScript, Go** from multi-language datasets
2. **Deduplicate** across datasets
3. **Clean** (remove invalid files, normalize formatting)
4. **Tokenize** for your specific tokenizer
5. **Convert to training format** (e.g., PyArrow, binary)

## Stack v2 Alternative

If you have HuggingFace authentication, the Stack v2 dataset (bigcode/the-stack-v2-dedup) offers:
- Higher quality deduplicated code
- Larger language-specific subsets
- See `STACK_V2_AUTH.md` for instructions

## Download Script

This directory was populated using:
```bash
/home/kang/Documents/projects/rust-ai/scripts/download_code_datasets.sh
```
EOF

success "Dataset index created: $CODE_DIR/README.md"
log ""

# ============================================
# Verification and Checksum
# ============================================
log "Verifying downloads..."

# List all parquet files
parquet_count=$(find "$CODE_DIR" -name "*.parquet" 2>/dev/null | wc -l)
json_count=$(find "$CODE_DIR" -name "*.json" -o -name "*.jsonl" 2>/dev/null | wc -l)
arrow_count=$(find "$CODE_DIR" -name "*.arrow" 2>/dev/null | wc -l)

log ""
log "========== Final Summary =========="
log "Parquet files: $parquet_count"
log "JSON/JSONL files: $json_count"
log "Arrow files: $arrow_count"
log "Total code directory size: $(get_size "$CODE_DIR")"
log ""
log "Data saved to: $CODE_DIR"
log "Log file: $LOG_FILE"
log ""
log "========== Download Complete =========="
log ""
log "Next steps:"
log "1. Verify dataset integrity: du -sh $CODE_DIR/*"
log "2. Process datasets: python /path/to/process_datasets.py"
log "3. Check README.md for dataset details"
log ""
