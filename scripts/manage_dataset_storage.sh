#!/bin/bash
# Dataset Storage Management Script
# Provides compression, archival, and restoration of datasets
#
# Usage:
#   ./manage_dataset_storage.sh status          # Show storage status
#   ./manage_dataset_storage.sh compress <dir>  # Compress a dataset to .tar.zst
#   ./manage_dataset_storage.sh restore <archive> # Restore from archive
#   ./manage_dataset_storage.sh archive-inactive # Archive datasets not used in 30 days
#   ./manage_dataset_storage.sh cleanup         # Remove original after successful archive

set -e

# Configuration
DATA_DIR="/data/datasets/tritter"
ARCHIVE_DIR="/data/datasets/tritter/archives"
MANIFEST="$ARCHIVE_DIR/manifest.json"
ZSTD_LEVEL=19  # Max compression for archival (slower but smallest)
INACTIVE_DAYS=30

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

log() { echo -e "${GREEN}[$(date '+%H:%M:%S')]${NC} $1"; }
warn() { echo -e "${YELLOW}[$(date '+%H:%M:%S')] WARN:${NC} $1"; }
err() { echo -e "${RED}[$(date '+%H:%M:%S')] ERROR:${NC} $1"; }
info() { echo -e "${BLUE}[INFO]${NC} $1"; }

# Ensure archive directory exists
mkdir -p "$ARCHIVE_DIR"

# Initialize manifest if needed
if [ ! -f "$MANIFEST" ]; then
    echo '{"archives": [], "total_saved_bytes": 0}' > "$MANIFEST"
fi

show_status() {
    log "=== Dataset Storage Status ==="
    echo ""

    # Disk usage
    info "Disk Usage:"
    df -h /data | tail -1 | awk '{print "  Total: "$2", Used: "$3", Free: "$4", Use%: "$5}'
    echo ""

    # Dataset sizes
    info "Active Datasets:"
    for dir in "$DATA_DIR"/*/; do
        if [ -d "$dir" ] && [ "$(basename "$dir")" != "archives" ]; then
            name=$(basename "$dir")
            size=$(du -sh "$dir" 2>/dev/null | cut -f1)
            files=$(find "$dir" -type f | wc -l)
            last_access=$(find "$dir" -type f -printf '%T@\n' 2>/dev/null | sort -n | tail -1 | xargs -I{} date -d @{} '+%Y-%m-%d' 2>/dev/null || echo "unknown")
            printf "  %-25s %8s  (%d files, last: %s)\n" "$name" "$size" "$files" "$last_access"
        fi
    done
    echo ""

    # Archives
    if [ -d "$ARCHIVE_DIR" ] && [ "$(ls -A "$ARCHIVE_DIR"/*.tar.zst 2>/dev/null)" ]; then
        info "Archived Datasets:"
        for archive in "$ARCHIVE_DIR"/*.tar.zst; do
            if [ -f "$archive" ]; then
                name=$(basename "$archive" .tar.zst)
                size=$(du -sh "$archive" | cut -f1)
                printf "  %-25s %8s (compressed)\n" "$name" "$size"
            fi
        done
        echo ""
    fi

    # Total
    total=$(du -sh "$DATA_DIR" 2>/dev/null | cut -f1)
    info "Total tritter data: $total"

    # Compression savings
    if [ -f "$MANIFEST" ]; then
        saved=$(python3 -c "import json; print(json.load(open('$MANIFEST'))['total_saved_bytes'])" 2>/dev/null || echo "0")
        if [ "$saved" != "0" ]; then
            saved_human=$(numfmt --to=iec "$saved" 2>/dev/null || echo "${saved}B")
            info "Space saved by compression: $saved_human"
        fi
    fi
}

compress_dataset() {
    local source_dir="$1"
    local dir_name=$(basename "$source_dir")
    local archive_path="$ARCHIVE_DIR/${dir_name}.tar.zst"

    if [ ! -d "$source_dir" ]; then
        err "Directory not found: $source_dir"
        exit 1
    fi

    if [ -f "$archive_path" ]; then
        warn "Archive already exists: $archive_path"
        read -p "Overwrite? [y/N] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi

    log "Compressing $dir_name with zstd level $ZSTD_LEVEL..."

    # Get original size
    original_size=$(du -sb "$source_dir" | cut -f1)

    # Compress with progress
    cd "$(dirname "$source_dir")"
    tar -cf - "$dir_name" | pv -s $(du -sb "$dir_name" | cut -f1) | zstd -$ZSTD_LEVEL -T0 -o "$archive_path"

    # Get compressed size
    compressed_size=$(stat -c%s "$archive_path")
    saved=$((original_size - compressed_size))
    ratio=$(echo "scale=1; $compressed_size * 100 / $original_size" | bc)

    log "Compression complete!"
    info "Original:   $(numfmt --to=iec $original_size)"
    info "Compressed: $(numfmt --to=iec $compressed_size)"
    info "Ratio:      ${ratio}%"
    info "Saved:      $(numfmt --to=iec $saved)"

    # Update manifest
    python3 << EOF
import json
from datetime import datetime

manifest_path = "$MANIFEST"
with open(manifest_path, 'r') as f:
    manifest = json.load(f)

manifest['archives'].append({
    'name': '$dir_name',
    'archive_path': '$archive_path',
    'original_size': $original_size,
    'compressed_size': $compressed_size,
    'compression_ratio': $ratio,
    'created': datetime.now().isoformat(),
    'zstd_level': $ZSTD_LEVEL
})
manifest['total_saved_bytes'] += $saved

with open(manifest_path, 'w') as f:
    json.dump(manifest, f, indent=2)
EOF

    log "Manifest updated"
    echo ""
    warn "Original directory NOT deleted. Run with 'cleanup' after verifying archive."
}

restore_archive() {
    local archive_path="$1"

    if [ ! -f "$archive_path" ]; then
        # Try adding .tar.zst extension
        if [ -f "$ARCHIVE_DIR/$archive_path.tar.zst" ]; then
            archive_path="$ARCHIVE_DIR/$archive_path.tar.zst"
        else
            err "Archive not found: $archive_path"
            exit 1
        fi
    fi

    local dir_name=$(basename "$archive_path" .tar.zst)
    local restore_path="$DATA_DIR/pretrain/$dir_name"

    if [ -d "$restore_path" ]; then
        warn "Directory already exists: $restore_path"
        read -p "Overwrite? [y/N] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
        rm -rf "$restore_path"
    fi

    log "Restoring $dir_name..."

    cd "$DATA_DIR/pretrain"
    pv "$archive_path" | zstd -d | tar -xf -

    log "Restored to $restore_path"
    info "Size: $(du -sh "$restore_path" | cut -f1)"
}

archive_inactive() {
    log "Looking for datasets not accessed in $INACTIVE_DAYS days..."

    for dir in "$DATA_DIR"/pretrain/*/; do
        if [ -d "$dir" ]; then
            name=$(basename "$dir")

            # Skip if already archived
            if [ -f "$ARCHIVE_DIR/${name}.tar.zst" ]; then
                continue
            fi

            # Check last access time
            last_access=$(find "$dir" -type f -printf '%A@\n' 2>/dev/null | sort -n | tail -1)
            if [ -n "$last_access" ]; then
                days_ago=$(echo "scale=0; ($(date +%s) - $last_access) / 86400" | bc)
                if [ "$days_ago" -gt "$INACTIVE_DAYS" ]; then
                    size=$(du -sh "$dir" | cut -f1)
                    info "$name: $size (last accessed $days_ago days ago)"
                    read -p "Archive this dataset? [y/N] " -n 1 -r
                    echo
                    if [[ $REPLY =~ ^[Yy]$ ]]; then
                        compress_dataset "$dir"
                    fi
                fi
            fi
        fi
    done

    log "Inactive dataset scan complete"
}

cleanup_originals() {
    log "Checking for archived datasets that can be cleaned up..."

    for archive in "$ARCHIVE_DIR"/*.tar.zst; do
        if [ -f "$archive" ]; then
            name=$(basename "$archive" .tar.zst)

            # Check multiple possible locations
            for location in "$DATA_DIR/pretrain/$name" "$DATA_DIR/$name"; do
                if [ -d "$location" ]; then
                    archive_size=$(stat -c%s "$archive")
                    original_size=$(du -sb "$location" | cut -f1)

                    info "Found: $name"
                    info "  Original:   $(numfmt --to=iec $original_size) at $location"
                    info "  Archive:    $(numfmt --to=iec $archive_size)"

                    # Verify archive integrity
                    log "Verifying archive integrity..."
                    if zstd -t "$archive" 2>/dev/null; then
                        info "  Archive OK"
                        read -p "  Delete original directory? [y/N] " -n 1 -r
                        echo
                        if [[ $REPLY =~ ^[Yy]$ ]]; then
                            rm -rf "$location"
                            log "Deleted $location"
                        fi
                    else
                        err "Archive failed integrity check! NOT deleting original."
                    fi
                fi
            done
        fi
    done

    log "Cleanup complete"
}

# Quick compression for processed JSONL files
compress_processed() {
    log "Compressing processed JSONL files..."

    for jsonl in "$DATA_DIR"/processed/*.jsonl; do
        if [ -f "$jsonl" ] && [ ! -f "${jsonl}.zst" ]; then
            name=$(basename "$jsonl")
            size=$(du -sh "$jsonl" | cut -f1)
            log "Compressing $name ($size)..."

            zstd -$ZSTD_LEVEL -T0 "$jsonl"

            new_size=$(du -sh "${jsonl}.zst" | cut -f1)
            log "  $size -> $new_size"

            # Keep original for now
            # rm "$jsonl"
        fi
    done
}

# Main
case "${1:-status}" in
    status)
        show_status
        ;;
    compress)
        if [ -z "$2" ]; then
            err "Usage: $0 compress <directory>"
            exit 1
        fi
        compress_dataset "$2"
        ;;
    restore)
        if [ -z "$2" ]; then
            err "Usage: $0 restore <archive>"
            exit 1
        fi
        restore_archive "$2"
        ;;
    archive-inactive)
        archive_inactive
        ;;
    cleanup)
        cleanup_originals
        ;;
    compress-processed)
        compress_processed
        ;;
    *)
        echo "Usage: $0 {status|compress|restore|archive-inactive|cleanup|compress-processed}"
        exit 1
        ;;
esac
