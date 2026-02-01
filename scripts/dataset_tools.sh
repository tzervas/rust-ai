#!/bin/bash
# Utility Tools for Code Datasets
# Helper functions for inspecting, processing, and managing downloaded datasets

set -e

# Configuration
CODE_DIR="/data/datasets/tritter/pretrain/code"
DATA_DIR="/data/datasets/tritter/pretrain"
LOG_DIR="/data/datasets/tritter/logs"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# ============================================
# Helper Functions
# ============================================

log() { echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"; }
warn() { echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARN:${NC} $1"; }
error() { echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1"; }
info() { echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')] INFO:${NC} $1"; }

# ============================================
# Main Functions
# ============================================

# Show dataset statistics
show_stats() {
    log "=== Code Dataset Statistics ==="
    echo ""

    if [ ! -d "$CODE_DIR" ]; then
        error "Code directory not found: $CODE_DIR"
        return 1
    fi

    local total_size=0
    local total_files=0

    echo -e "${BLUE}Dataset Summary:${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    printf "%-30s %12s %12s\n" "Dataset" "Size" "Files"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    for dir in "$CODE_DIR"/*/; do
        if [ -d "$dir" ]; then
            local name=$(basename "$dir")
            local size=$(du -sh "$dir" 2>/dev/null | cut -f1)
            local files=$(find "$dir" -type f \( -name "*.parquet" -o -name "*.json" -o -name "*.jsonl" -o -name "*.arrow" \) 2>/dev/null | wc -l)

            if [ "$files" -gt 0 ]; then
                printf "%-30s %12s %12d\n" "$name" "$size" "$files"
                total_files=$((total_files + files))
            fi
        fi
    done

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    total_size=$(du -sh "$CODE_DIR" 2>/dev/null | cut -f1)
    printf "%-30s %12s %12d\n" "TOTAL" "$total_size" "$total_files"
    echo ""
}

# List available datasets
list_datasets() {
    log "=== Available Datasets in Code Directory ==="
    echo ""

    if [ ! -d "$CODE_DIR" ]; then
        error "Code directory not found: $CODE_DIR"
        return 1
    fi

    for dir in "$CODE_DIR"/*/; do
        if [ -d "$dir" ]; then
            local name=$(basename "$dir")
            echo ""
            echo -e "${BLUE}$name:${NC}"

            if [ -f "$dir/metadata.json" ]; then
                echo "  Metadata: $(grep -o '"name"[^}]*' "$dir/metadata.json" | head -3)"
            fi

            local file_count=$(find "$dir" -type f \( -name "*.parquet" -o -name "*.json" -o -name "*.jsonl" \) 2>/dev/null | wc -l)
            if [ "$file_count" -gt 0 ]; then
                echo "  Files: $file_count"
                echo "  Size: $(du -sh "$dir" 2>/dev/null | cut -f1)"
                echo "  File types: $(find "$dir" -type f ! -name ".*" -exec basename {} \; | sed 's/.*\.//' | sort | uniq -c)"
            else
                echo "  Status: Empty or downloading..."
            fi
        fi
    done
    echo ""
}

# Show download status
download_status() {
    log "=== Download Status ==="
    echo ""

    if [ ! -d "$LOG_DIR" ]; then
        warn "No logs found yet"
        return 0
    fi

    for log_file in "$LOG_DIR"/download_code_datasets_*.log; do
        if [ -f "$log_file" ]; then
            echo -e "${BLUE}Latest Download Log:${NC}"
            echo "  File: $(basename "$log_file")"
            echo "  Last Updated: $(stat -c %y "$log_file" 2>/dev/null | cut -d' ' -f1-2)"
            echo ""
            echo "Recent Activity:"
            tail -20 "$log_file" | grep -E "(Downloading|downloaded|SUCCESS|ERROR|WARN)" || echo "  No recent activity"
            echo ""
        fi
    done
}

# Extract specific language
extract_language() {
    local language=$1
    local output_dir=${2:-.}

    if [ -z "$language" ]; then
        error "Usage: extract_language <rust|typescript|go> [output_dir]"
        return 1
    fi

    log "=== Extracting $language Code ==="
    echo ""

    mkdir -p "$output_dir/$language"

    local count=0
    for dataset_dir in "$CODE_DIR"/*/; do
        if [ ! -d "$dataset_dir" ]; then continue; fi

        local dataset_name=$(basename "$dataset_dir")
        info "Scanning $dataset_name..."

        # Look for language-specific directories
        if [ -d "$dataset_dir/$language" ]; then
            info "Found $language subset in $dataset_name"
            # Copy parquet files
            find "$dataset_dir/$language" -name "*.parquet" -exec cp {} "$output_dir/$language/" \; 2>/dev/null || true
            count=$((count + 1))
        fi
    done

    if [ $count -gt 0 ]; then
        log "Extracted $language files: $(find "$output_dir/$language" -type f | wc -l) files"
        log "Output directory: $output_dir/$language"
    else
        warn "No $language-specific data found"
    fi
    echo ""
}

# Verify dataset integrity
verify_datasets() {
    log "=== Verifying Dataset Integrity ==="
    echo ""

    local errors=0

    for dir in "$CODE_DIR"/*/; do
        if [ ! -d "$dir" ]; then continue; fi

        local name=$(basename "$dir")
        local parquet_count=$(find "$dir" -name "*.parquet" 2>/dev/null | wc -l)

        if [ "$parquet_count" -eq 0 ]; then
            warn "$name: No parquet files found"
            errors=$((errors + 1))
            continue
        fi

        # Try to read first parquet file
        local first_parquet=$(find "$dir" -name "*.parquet" -print -quit)
        if [ -z "$first_parquet" ]; then
            error "$name: Cannot find parquet file"
            errors=$((errors + 1))
            continue
        fi

        # Check file size
        if [ ! -s "$first_parquet" ]; then
            error "$name: Parquet file is empty"
            errors=$((errors + 1))
            continue
        fi

        info "$name: OK ($parquet_count parquet files)"
    done

    echo ""
    if [ $errors -eq 0 ]; then
        log "All datasets verified successfully"
    else
        warn "Found $errors issues during verification"
    fi
    echo ""
}

# Show dataset info
show_dataset_info() {
    local dataset=$1

    if [ -z "$dataset" ]; then
        error "Usage: show_dataset_info <dataset_name>"
        echo "Available datasets:"
        ls -d "$CODE_DIR"/*/ 2>/dev/null | xargs -n1 basename
        return 1
    fi

    local dataset_dir="$CODE_DIR/$dataset"

    if [ ! -d "$dataset_dir" ]; then
        error "Dataset not found: $dataset"
        return 1
    fi

    log "=== Dataset Info: $dataset ==="
    echo ""
    echo "Location: $dataset_dir"
    echo "Size: $(du -sh "$dataset_dir")"
    echo "File count: $(find "$dataset_dir" -type f | wc -l)"
    echo ""

    if [ -f "$dataset_dir/metadata.json" ]; then
        echo "Metadata:"
        cat "$dataset_dir/metadata.json"
        echo ""
    fi

    if [ -f "$dataset_dir/README.md" ]; then
        echo "README:"
        head -30 "$dataset_dir/README.md"
        echo ""
    fi

    echo "Files:"
    find "$dataset_dir" -type f ! -name ".*" | head -20 | xargs -I {} basename {}
}

# Cleanup and organize
cleanup_datasets() {
    log "=== Cleanup and Organization ==="
    echo ""

    warn "This will remove empty directories and temporary files"
    echo -n "Continue? (y/n) "
    read -r response
    if [ "$response" != "y" ]; then
        warn "Cleanup cancelled"
        return
    fi

    # Remove empty directories
    log "Removing empty directories..."
    find "$CODE_DIR" -type d -empty -delete

    # Remove .hf lock files
    log "Removing HF lock files..."
    find "$CODE_DIR" -name ".*.lock" -delete 2>/dev/null || true

    # Remove incomplete downloads
    log "Removing incomplete downloads..."
    find "$CODE_DIR" -name "*.incomplete" -delete 2>/dev/null || true

    log "Cleanup complete"
    echo ""
}

# Show help
show_help() {
    cat << 'EOF'
Code Dataset Management Tools

USAGE:
    ./dataset_tools.sh <command> [options]

COMMANDS:
    stats               Show dataset statistics
    list                List all available datasets
    status              Show download status
    extract <lang>      Extract language-specific code (rust|typescript|go)
    verify              Verify dataset integrity
    info <dataset>      Show detailed info about a dataset
    cleanup             Clean up temporary files
    help                Show this help message

EXAMPLES:
    # Show statistics
    ./dataset_tools.sh stats

    # List datasets
    ./dataset_tools.sh list

    # Extract Rust code
    ./dataset_tools.sh extract rust ./extracted

    # Verify all datasets
    ./dataset_tools.sh verify

    # Get info about specific dataset
    ./dataset_tools.sh info the-stack-rust-clean

    # Show download progress
    ./dataset_tools.sh status

LOCATIONS:
    Code directory: /data/datasets/tritter/pretrain/code
    Logs: /data/datasets/tritter/logs
    Data: /data/datasets/tritter/pretrain

EOF
}

# ============================================
# Main Command Handler
# ============================================

main() {
    local command=${1:-help}

    case "$command" in
        stats)
            show_stats
            ;;
        list)
            list_datasets
            ;;
        status)
            download_status
            ;;
        extract)
            extract_language "$2" "$3"
            ;;
        verify)
            verify_datasets
            ;;
        info)
            show_dataset_info "$2"
            ;;
        cleanup)
            cleanup_datasets
            ;;
        help)
            show_help
            ;;
        *)
            error "Unknown command: $command"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
