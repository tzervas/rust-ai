# Code Datasets - Quick Start Guide

## TL;DR

Download code datasets for Tritter training:

```bash
# Run the full download script (all sources)
bash /home/kang/Documents/projects/rust-ai/scripts/download_code_datasets.sh

# Or download specific datasets manually
cd /data/datasets/tritter/pretrain

# 1. CodeParrot GitHub Code (1TB, all languages)
/home/kang/.local/bin/hf download codeparrot/github-code \
  --repo-type dataset --local-dir code/codeparrot-github-code \
  --local-dir-use-symlinks False

# 2. StarCoder Rust (~20GB)
/home/kang/.local/bin/hf download bigcode/starcoderdata \
  --repo-type dataset --include "rust/*" \
  --local-dir code/starcoderdata --local-dir-use-symlinks False

# 3. The Stack - Rust Clean (10GB, highest quality Rust)
/home/kang/.local/bin/hf download ammarnasr/the-stack-rust-clean \
  --repo-type dataset --local-dir code/the-stack-rust-clean \
  --local-dir-use-symlinks False
```

## Available Datasets

| Dataset | Size | Files | Languages | Auth Required |
|---------|------|-------|-----------|---------------|
| CodeParrot GitHub | 1TB | 115M | 30 (Rust ✓, TS ✓, Go ✓) | No |
| StarCoder | 786GB | Multi | 86 (Rust ✓, TS ✓, Go ✓) | TOU only |
| Stack Rust Clean | 10GB | 993k | Rust only | No |
| CodeSearchNet | Variable | Multi | Go, Python, Java, etc | No |
| CodeParrot Clean | Variable | 5.17M | Multi | No |

## Quick Downloads

### Option 1: Start Small (Quality First)
- Time: 2-4 hours
- Size: ~40GB
- Best for: Quick testing, quality over quantity

```bash
# High-quality Rust
/home/kang/.local/bin/hf download ammarnasr/the-stack-rust-clean \
  --repo-type dataset \
  --local-dir /data/datasets/tritter/pretrain/code/rust-clean \
  --local-dir-use-symlinks False

# StarCoder TypeScript
/home/kang/.local/bin/hf download bigcode/starcoderdata \
  --repo-type dataset --include "typescript/*" \
  --local-dir /data/datasets/tritter/pretrain/code/starcoderdata \
  --local-dir-use-symlinks False

# StarCoder Go
/home/kang/.local/bin/hf download bigcode/starcoderdata \
  --repo-type dataset --include "go/*" \
  --local-dir /data/datasets/tritter/pretrain/code/starcoderdata \
  --local-dir-use-symlinks False
```

### Option 2: Balanced (Recommended)
- Time: 4-8 hours
- Size: ~50-100GB
- Best for: Most use cases

```bash
# Run complete script (does this + fallbacks)
bash /home/kang/Documents/projects/rust-ai/scripts/download_code_datasets.sh
```

### Option 3: Maximum Coverage
- Time: 8+ hours
- Size: 1TB+
- Best for: Comprehensive training data

```bash
# Run complete script (includes CodeParrot 1TB)
bash /home/kang/Documents/projects/rust-ai/scripts/download_code_datasets.sh
```

## Where Data Goes

```
/data/datasets/tritter/pretrain/
├── code/
│   ├── codeparrot-github-code/      ← CodeParrot (1TB if downloaded)
│   ├── starcoderdata/               ← StarCoder (language subsets)
│   │   ├── rust/
│   │   ├── typescript/
│   │   └── go/
│   ├── the-stack-rust-clean/        ← Rust-specific
│   ├── codesearchnet/               ← Go/others
│   ├── codeparrot-clean/            ← General cleanup
│   ├── README.md                    ← Dataset info
│   └── STACK_V2_AUTH.md             ← Instructions for Stack v2
└── logs/
    └── download_code_datasets_*.log  ← Download logs
```

## Verify Downloads

```bash
# Check sizes
du -sh /data/datasets/tritter/pretrain/code/*

# Count parquet files
find /data/datasets/tritter/pretrain/code -name "*.parquet" | wc -l

# Check logs
tail -100 /data/datasets/tritter/logs/download_code_datasets_*.log
```

## Stack v2 (If Accessible)

Stack v2 is better quality (deduplicated) but requires accepting Terms of Use:

```bash
# Steps:
# 1. Visit: https://huggingface.co/datasets/bigcode/the-stack-v2-dedup
# 2. Click "Accept Terms"
# 3. Run script from: /data/datasets/tritter/pretrain/code/STACK_V2_AUTH.md
```

## Processing After Download

```bash
# List what you have
python3 << 'EOF'
import os
import glob

code_dir = "/data/datasets/tritter/pretrain/code"
for dataset_dir in sorted(glob.glob(f"{code_dir}/*/")):
    name = os.path.basename(dataset_dir.rstrip("/"))
    parquets = glob.glob(f"{dataset_dir}**/*.parquet", recursive=True)
    print(f"{name}: {len(parquets)} files")
EOF
```

## Troubleshooting

### Download Stalled
```bash
# Kill and restart
pkill -f "hf download"

# Check connection
ping huggingface.co

# Run specific download again
/home/kang/.local/bin/hf download codeparrot/github-code \
  --repo-type dataset --local-dir /data/datasets/tritter/pretrain/code/codeparrot \
  --local-dir-use-symlinks False
```

### Out of Space
```bash
# Check available space
df -h /data/datasets/tritter/

# Check current size
du -sh /data/datasets/tritter/pretrain/code/

# Delete largest dataset if needed
rm -rf /data/datasets/tritter/pretrain/code/codeparrot-github-code
```

### Authentication Issues
- CodeParrot/Stack Clean: No auth needed
- StarCoder: Accepts Terms of Use automatically
- Stack v2: Visit page and click "Accept" button first

## Recommended Dataset Mix for Training

### For 100M parameter model:
- 40% CodeParrot GitHub (filtered for Rust/TS/Go)
- 30% StarCoder (language subsets)
- 20% Stack Rust Clean (high quality)
- 10% CodeSearchNet Go (specialization)

### For Rust-Focused Training:
- 60% Stack Rust Clean
- 30% StarCoder Rust
- 10% CodeParrot GitHub (Rust filtered)

### For Multi-Language (Rust + TS + Go):
- 30% Stack Rust Clean (Rust)
- 25% StarCoder TypeScript
- 25% StarCoder Go
- 20% CodeParrot GitHub (multi-language)

## Performance Notes

**Download Speed**:
- Depends on your connection
- Typical: 10-50MB/s from HF
- 1TB = 6-27 hours at these speeds

**Storage**:
- Make sure `/data` has enough space
- Use `du -sh` to monitor

**CPU**:
- Minimal (just file operations)
- Can run in background

**Network**:
- HF CLI is resumable (safe to pause/restart)
- Can run multiple downloads in parallel (with care)

## Next Steps

1. **Choose your option** (Small, Balanced, or Max)
2. **Run download script** (with appropriate dataset selection)
3. **Monitor** `tail -f /data/datasets/tritter/logs/download_code_datasets_*.log`
4. **Verify** with size/count commands above
5. **Process datasets** (deduplicate, clean, tokenize)
6. **Start training** with combined data

## Resources

- Full guide: `/home/kang/Documents/projects/rust-ai/CODE_DATASET_SOURCES.md`
- Download script: `/home/kang/Documents/projects/rust-ai/scripts/download_code_datasets.sh`
- Data location: `/data/datasets/tritter/pretrain/code/`
- Logs: `/data/datasets/tritter/logs/`
