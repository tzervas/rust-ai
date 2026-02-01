# Public Code Dataset Sources for Tritter Training

## Summary

Research of public, authentication-free code datasets has identified **5 primary sources** containing Rust, TypeScript, and Go code. Stack v2 is partially accessible (requires Terms of Use agreement) but fallback sources are available.

## Primary Public Sources (No Auth Required)

### 1. CodeParrot GitHub Code ⭐ PRIMARY
- **URL**: https://huggingface.co/datasets/codeparrot/github-code
- **Size**: ~1TB
- **Files**: 115 million code files
- **Languages**: 30 languages including:
  - ✓ Rust
  - ✓ TypeScript / JavaScript
  - ✓ Go
  - Plus: Python, Java, C/C++, Shell, Ruby, PHP, etc.
- **Format**: Parquet files
- **License**: Mixed (MIT, Apache, GPL, etc.)
- **Access**: Public - NO authentication required
- **Pros**:
  - Largest public code dataset
  - Includes all target languages
  - Well-documented file structure
  - Tracks original licenses (15 different types)
  - Direct HF CLI support
- **Cons**:
  - Very large (~1TB)
  - Mixed quality
  - No deduplication
- **Download Command**:
  ```bash
  /home/kang/.local/bin/hf download codeparrot/github-code \
    --repo-type dataset \
    --local-dir /data/datasets/tritter/pretrain/code/codeparrot-github-code \
    --local-dir-use-symlinks False
  ```

### 2. BigCode StarCoder Data ⭐ PRIMARY
- **URL**: https://huggingface.co/datasets/bigcode/starcoderdata
- **Size**: 786GB
- **Languages**: 86 programming languages
- **Code Components**:
  - 783GB pure code (86 languages)
  - 54GB GitHub issues
  - 13GB Jupyter notebooks (scripts + text-code pairs)
  - 32GB git commits
- **Format**: Parquet
- **License**: OpenRAIL (open source model license)
- **Access**: Public - requires Terms of Use agreement (no authentication)
- **Subset Availability**:
  - `rust/` - Rust code
  - `typescript/` - TypeScript code
  - `go/` - Go code
  - Plus 82 other language directories
- **Pros**:
  - Comprehensive language coverage
  - Largest open code dataset (250B tokens)
  - Quality filtering applied
  - Jupyter notebooks useful for documentation
  - Git commits show code evolution
- **Cons**:
  - Requires Terms of Use acceptance
  - Large initial download
- **Download Command** (language-specific):
  ```bash
  /home/kang/.local/bin/hf download bigcode/starcoderdata \
    --repo-type dataset \
    --include "rust/*" \
    --local-dir /data/datasets/tritter/pretrain/code/starcoderdata \
    --local-dir-use-symlinks False
  ```

### 3. The Stack - Rust Clean ⭐ PRIMARY (Rust only)
- **URL**: https://huggingface.co/datasets/ammarnasr/the-stack-rust-clean
- **Size**: ~5-10GB (estimated)
- **Files**: 993,000 Rust files
  - Train: 900,000
  - Validation: 50,000
  - Test: 50,000
- **Language**: Rust exclusively (deduplicated)
- **Format**: Parquet
- **License**: OpenRAIL
- **Access**: Public - NO authentication required
- **Columns**:
  - `hexsha`: Git commit hash (40 chars)
  - `size`: File size in bytes
  - `content`: Full source code
  - `avg_line_length`: Average line length
  - `max_line_length`: Maximum line length
  - `alphanum_fraction`: Alphanumeric character ratio
- **Pros**:
  - High-quality Rust-specific dataset
  - Deduplicated and cleaned
  - Balanced train/val/test splits
  - Rich metadata
  - Smallest of primary sources
- **Cons**:
  - Rust only (need other sources for TS/Go)
- **Download Command**:
  ```bash
  /home/kang/.local/bin/hf download ammarnasr/the-stack-rust-clean \
    --repo-type dataset \
    --local-dir /data/datasets/tritter/pretrain/code/the-stack-rust-clean \
    --local-dir-use-symlinks False
  ```

### 4. CodeSearchNet (Go Focus)
- **URL**: https://huggingface.co/datasets/bigcode/codesearchnet
- **Languages**: Go, Python, Java, PHP, Ruby, JavaScript
- **Purpose**: Code search and clone detection benchmark
- **Format**: Parquet
- **License**: MIT
- **Access**: Public - NO authentication required
- **Pros**:
  - Go-specific content
  - High-quality pairs (function + docstring)
  - Small size (good for validation)
- **Cons**:
  - Smaller than other sources
  - Limited language coverage

### 5. CodeParrot Clean (General Purpose)
- **URL**: https://huggingface.co/datasets/codeparrot/codeparrot-clean
- **Size**: Variable
- **Files**: 5.17 million
- **Languages**: Multiple including Rust, TypeScript, Go
- **Format**: Parquet
- **License**: OpenRAIL
- **Access**: Public - NO authentication required
- **Pros**:
  - Well-cleaned and deduplicated
  - Good starting point
- **Cons**:
  - Smaller than primary sources
  - Less comprehensive

## Gated but Accessible (Terms of Use)

### Stack v2 Deduplicated (RECOMMENDED IF ACCESSIBLE)
- **URL**: https://huggingface.co/datasets/bigcode/the-stack-v2-dedup
- **Size**: 3TB total (subsets available)
- **Estimated per language**:
  - Rust: ~50GB
  - TypeScript: ~40GB
  - Go: ~35GB
- **Status**: Gated (requires TOU acceptance, NOT payment/auth token)
- **Pros**:
  - Largest deduplicated dataset
  - Higher quality than Stack v1
  - Per-language separation
  - Most recent version
- **Access Instructions**:
  1. Visit: https://huggingface.co/datasets/bigcode/the-stack-v2-dedup
  2. Accept Terms of Use (green button)
  3. Run script provided in /data/datasets/tritter/pretrain/code/STACK_V2_AUTH.md

## Alternative Fallback Sources

If primary sources become unavailable:

### 1. Swallow Code v2 (147M items)
- **URL**: https://huggingface.co/datasets/tokyotech-llm/swallow-code-v2
- **Size**: Large-scale multi-language
- **Languages**: Multiple (check dataset page)
- **Format**: Parquet
- **Access**: Public

### 2. CodeParrot Code Clones
- **URL**: https://huggingface.co/datasets/codeparrot/codeparrot-code-clones
- **Purpose**: Clone detection
- **Access**: Public

### 3. xCodeEval (Cross-language Evaluation)
- **URL**: https://huggingface.co/datasets/NTU-NLP-sg/xCodeEval
- **Files**: 1M
- **Specialization**: Code generation evaluation
- **Access**: Public

## Recommended Download Strategy

### Option A: Maximum Coverage (Recommended)
1. **Start with**: CodeParrot GitHub Code (all languages)
   - Size: ~1TB
   - Time: 4-8 hours depending on connection

2. **Supplement with**: The Stack Rust Clean (high-quality Rust)
   - Size: ~10GB
   - Time: 30 min - 1 hour

3. **Add if available**: Stack v2 (deduplicated)
   - Size: ~125GB for Rust/TS/Go (sample)
   - Time: 2-4 hours

### Option B: Balanced (Recommended for Medium Resources)
1. **BigCode StarCoder**: Language-specific subsets
   ```bash
   # Rust subset (~15-20GB)
   /home/kang/.local/bin/hf download bigcode/starcoderdata \
     --include "rust/*" --repo-type dataset ...

   # TypeScript subset (~12-15GB)
   /home/kang/.local/bin/hf download bigcode/starcoderdata \
     --include "typescript/*" --repo-type dataset ...

   # Go subset (~10-12GB)
   /home/kang/.local/bin/hf download bigcode/starcoderdata \
     --include "go/*" --repo-type dataset ...
   ```
   Total: ~40-50GB, Time: 2-4 hours

2. **The Stack Rust Clean**: Pure Rust focus
   - Size: ~10GB
   - Complements StarCoder with high-quality dedup

### Option C: Quality-First (Recommended for Quick Start)
1. **The Stack Rust Clean** (Rust)
   - 993k files, deduplicated, ~10GB

2. **StarCoder Go** (Go)
   - ~10-12GB

3. **StarCoder TypeScript** (TypeScript)
   - ~12-15GB

Total: ~35-40GB, Time: 1.5-3 hours

## Download Script

**Location**: `/home/kang/Documents/projects/rust-ai/scripts/download_code_datasets.sh`

**Usage**:
```bash
# Run the complete download script
bash /home/kang/Documents/projects/rust-ai/scripts/download_code_datasets.sh

# Or download specific datasets
/home/kang/.local/bin/hf download codeparrot/github-code \
  --repo-type dataset \
  --local-dir /data/datasets/tritter/pretrain/code/codeparrot-github-code \
  --local-dir-use-symlinks False
```

**Features**:
- Logs all downloads to: `/data/datasets/tritter/logs/download_code_datasets_*.log`
- Skips existing datasets (idempotent)
- Creates metadata.json for each dataset
- Generates README.md with dataset information
- Provides fallback sources if primary fails
- Color-coded output (errors, warnings, success)

## Stack v2 Why It Failed

The original Stack v2 download likely failed because:

1. **Gated Dataset**: Requires accepting Terms of Use at dataset page
2. **Large Size**: 3TB+ requires significant time/bandwidth
3. **No Direct Download**: Can't use HF CLI without TOU acceptance
4. **Session Timeout**: Large downloads may timeout without retry logic

**Solution**: Our script includes retry logic and guides users through TOU acceptance.

## Language-Specific Statistics

Based on HF dataset information:

| Language | CodeParrot | StarCoder | Stack-Clean | CodeSearchNet |
|----------|-----------|-----------|------------|---------------|
| **Rust** | ~15-20% of 1TB | ~20GB | ~10GB ✓ | Small |
| **TypeScript** | ~15-20% of 1TB | ~15GB | N/A | Small |
| **Go** | ~10-15% of 1TB | ~12GB | N/A | ~5GB ✓ |
| **Python** | ~25-30% of 1TB | ~80GB | N/A | ~10GB |

## Estimated Training Data Sizes

For a 100M-1B parameter model:

- **Small dataset** (1-10GB): Quick prototyping
  - Use: Stack Rust Clean + CodeSearchNet Go subset

- **Medium dataset** (40-100GB): Good for fine-tuning
  - Use: StarCoder language subsets (Rust/TS/Go)

- **Large dataset** (200-500GB): Full pre-training
  - Use: CodeParrot GitHub Code (filtered)

- **Extra Large** (500GB+): Comprehensive training
  - Use: CodeParrot + StarCoder + Stack v2 (if TOU accepted)

## Quality Considerations

1. **Deduplication Level**:
   - CodeParrot GitHub: Low (raw GitHub)
   - StarCoder: Medium (some filtering)
   - Stack Rust Clean: High (explicitly cleaned)

2. **License Tracking**:
   - CodeParrot: Tracks 15 license types
   - StarCoder: OpenRAIL license only
   - Stack Clean: OpenRAIL license only

3. **Best for Each Language**:
   - **Rust**: The Stack Rust Clean (best quality)
   - **TypeScript**: StarCoder or CodeParrot GitHub
   - **Go**: CodeSearchNet (domain-specific)

## Next Steps

1. **Run the download script**:
   ```bash
   bash /home/kang/Documents/projects/rust-ai/scripts/download_code_datasets.sh
   ```

2. **Monitor progress**:
   ```bash
   tail -f /data/datasets/tritter/logs/download_code_datasets_*.log
   ```

3. **Verify downloads**:
   ```bash
   du -sh /data/datasets/tritter/pretrain/code/*
   find /data/datasets/tritter/pretrain/code -name "*.parquet" | wc -l
   ```

4. **Process datasets** (create separate script):
   - Extract language-specific files
   - Remove duplicates (use exact match or semantic similarity)
   - Clean/normalize code (formatting, remove invalid files)
   - Tokenize with your tokenizer
   - Create training shards

## References

- [HuggingFace Datasets](https://huggingface.co/datasets)
- [BigCode Initiative](https://www.bigcode-project.org/)
- [StarCoder Paper](https://arxiv.org/abs/2305.06161)
- [The Stack Dataset](https://huggingface.co/datasets/bigcode/the-stack)

## Contact & Troubleshooting

If downloads fail:

1. **Check internet connection**: `ping huggingface.co`
2. **Verify HF CLI**: `/home/kang/.local/bin/hf --version`
3. **Check storage space**: `df -h /data/datasets/tritter/`
4. **Review logs**: `cat /data/datasets/tritter/logs/download_code_datasets_*.log`
5. **Try individual source**: See download commands in this document
6. **Fallback sources**: Use alternatives listed above
