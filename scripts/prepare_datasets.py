#!/usr/bin/env python3
"""
Dataset Preparation and Curation Script for Tritter Training

This script:
1. Validates downloaded parquet files
2. Creates mixed training datasets with specified ratios
3. Samples and shuffles data
4. Outputs JSONL files ready for training

Usage:
    python prepare_datasets.py --output /data/datasets/tritter/processed/train.jsonl
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Iterator
import hashlib

# Try to import optional dependencies
try:
    import pyarrow.parquet as pq
    HAS_PARQUET = True
except ImportError:
    HAS_PARQUET = False
    print("Warning: pyarrow not installed. Install with: pip install pyarrow")

# Data mix ratios (should sum to 1.0)
# Organized by category with subdirectories
DEFAULT_MIX = {
    # Natural language (40%)
    "fineweb-edu-10B": 0.25,      # Educational web content
    "wikipedia": 0.15,             # Wikipedia knowledge base

    # Code (25%)
    "stack-v2-python": 0.10,      # Python code
    "stack-v2-rust": 0.05,        # Rust code
    "stack-v2-typescript": 0.03,  # TypeScript code
    "smollm-python": 0.04,        # SmolLM curated Python
    "triton": 0.03,               # Triton GPU kernels

    # Math & Reasoning (15%)
    "finemath-4plus": 0.08,       # High-quality math
    "openwebmath": 0.04,          # Math from web
    "proof-pile-2": 0.03,         # Mathematical proofs

    # AI/ML Domain (10%)
    "cosmopedia-v2": 0.05,        # Synthetic educational
    "automathtext": 0.05,         # Synthetic math problems

    # Instruction/Alignment (10%)
    "instruction": 0.10,          # Combined instruction datasets
}

# Text column names to try (in order of preference)
TEXT_COLUMNS = [
    "text", "content", "code", "solution",
    "output", "response", "input", "question",
    "document", "body", "abstract"
]


def find_text_column(columns: List[str]) -> Optional[str]:
    """Find the text column in a parquet file."""
    for col in TEXT_COLUMNS:
        if col in columns:
            return col
    # Fallback: first string-like column
    return columns[0] if columns else None


def hash_text(text: str) -> str:
    """Create a hash for deduplication."""
    return hashlib.md5(text.encode('utf-8', errors='ignore')).hexdigest()[:16]


def read_parquet_texts(file_path: Path, max_samples: Optional[int] = None) -> Iterator[str]:
    """Read text from a parquet file."""
    if not HAS_PARQUET:
        return

    try:
        table = pq.read_table(file_path)
        columns = table.column_names
        text_col = find_text_column(columns)

        if text_col is None:
            print(f"  Warning: No text column found in {file_path}")
            return

        texts = table[text_col].to_pylist()

        if max_samples and len(texts) > max_samples:
            texts = random.sample(texts, max_samples)

        for text in texts:
            if text and isinstance(text, str) and len(text.strip()) > 50:
                yield text.strip()

    except Exception as e:
        print(f"  Error reading {file_path}: {e}")


def read_jsonl_texts(file_path: Path, max_samples: Optional[int] = None) -> Iterator[str]:
    """Read text from a JSONL file."""
    try:
        texts = []
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    for col in TEXT_COLUMNS:
                        if col in data and data[col]:
                            texts.append(data[col].strip())
                            break
                except json.JSONDecodeError:
                    continue

        if max_samples and len(texts) > max_samples:
            texts = random.sample(texts, max_samples)

        yield from texts

    except Exception as e:
        print(f"  Error reading {file_path}: {e}")


def collect_dataset_texts(
    dataset_dir: Path,
    max_samples: Optional[int] = None
) -> List[str]:
    """Collect all texts from a dataset directory."""
    texts = []

    # Find all data files
    parquet_files = list(dataset_dir.rglob("*.parquet"))
    jsonl_files = list(dataset_dir.rglob("*.jsonl")) + list(dataset_dir.rglob("*.json"))

    print(f"  Found {len(parquet_files)} parquet, {len(jsonl_files)} jsonl files")

    # Read parquet files
    for pf in parquet_files:
        for text in read_parquet_texts(pf, max_samples=max_samples):
            texts.append(text)
            if max_samples and len(texts) >= max_samples:
                return texts

    # Read JSONL files
    for jf in jsonl_files:
        if jf.name == "metadata.json":
            continue
        for text in read_jsonl_texts(jf, max_samples=max_samples):
            texts.append(text)
            if max_samples and len(texts) >= max_samples:
                return texts

    return texts


def deduplicate_texts(texts: List[str], threshold: float = 0.8) -> List[str]:
    """Simple hash-based deduplication."""
    seen_hashes = set()
    unique_texts = []

    for text in texts:
        h = hash_text(text)
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_texts.append(text)

    removed = len(texts) - len(unique_texts)
    if removed > 0:
        print(f"  Removed {removed} duplicates ({removed/len(texts)*100:.1f}%)")

    return unique_texts


def create_mixed_dataset(
    data_dir: Path,
    mix_ratios: Dict[str, float],
    total_samples: int,
    deduplicate: bool = True
) -> List[Dict]:
    """Create a mixed dataset with specified ratios."""

    all_samples = []

    for dataset_name, ratio in mix_ratios.items():
        dataset_path = data_dir / dataset_name

        if not dataset_path.exists():
            print(f"Skipping {dataset_name} (not found)")
            continue

        target_samples = int(total_samples * ratio)
        print(f"\nProcessing {dataset_name} (target: {target_samples} samples)...")

        texts = collect_dataset_texts(dataset_path, max_samples=target_samples * 2)

        if deduplicate:
            texts = deduplicate_texts(texts)

        # Truncate to target
        if len(texts) > target_samples:
            texts = random.sample(texts, target_samples)

        print(f"  Collected {len(texts)} samples")

        for text in texts:
            all_samples.append({
                "text": text,
                "source": dataset_name,
            })

    # Shuffle
    random.shuffle(all_samples)

    return all_samples


def write_jsonl(samples: List[Dict], output_path: Path):
    """Write samples to JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f"\nWrote {len(samples)} samples to {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Prepare mixed training dataset")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/data/datasets/tritter/pretrain"),
        help="Directory containing downloaded datasets"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/data/datasets/tritter/processed/train.jsonl"),
        help="Output JSONL file path"
    )
    parser.add_argument(
        "--total-samples",
        type=int,
        default=1_000_000,
        help="Total number of samples to include"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--no-dedup",
        action="store_true",
        help="Skip deduplication"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate datasets, don't create output"
    )

    args = parser.parse_args()
    random.seed(args.seed)

    print("=== Tritter Dataset Preparation ===")
    print(f"Data directory: {args.data_dir}")
    print(f"Output: {args.output}")
    print(f"Target samples: {args.total_samples:,}")

    # Validate datasets exist
    print("\n=== Validating Datasets ===")
    available_datasets = {}
    for name in DEFAULT_MIX.keys():
        path = args.data_dir / name
        if path.exists():
            size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
            files = len(list(path.rglob("*.parquet"))) + len(list(path.rglob("*.jsonl")))
            print(f"  {name}: {size/1024/1024:.1f} MB, {files} files")
            available_datasets[name] = DEFAULT_MIX[name]
        else:
            print(f"  {name}: NOT FOUND")

    if args.validate_only:
        print("\n=== Validation Complete ===")
        return

    if not available_datasets:
        print("\nError: No datasets found!")
        return

    # Normalize ratios for available datasets
    total_ratio = sum(available_datasets.values())
    normalized_mix = {k: v/total_ratio for k, v in available_datasets.items()}

    print("\n=== Normalized Mix Ratios ===")
    for name, ratio in normalized_mix.items():
        print(f"  {name}: {ratio*100:.1f}%")

    # Create mixed dataset
    print("\n=== Creating Mixed Dataset ===")
    samples = create_mixed_dataset(
        args.data_dir,
        normalized_mix,
        args.total_samples,
        deduplicate=not args.no_dedup
    )

    # Write output
    write_jsonl(samples, args.output)

    # Summary by source
    print("\n=== Final Mix ===")
    source_counts = {}
    for s in samples:
        source_counts[s["source"]] = source_counts.get(s["source"], 0) + 1

    for source, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        print(f"  {source}: {count:,} ({count/len(samples)*100:.1f}%)")

    print("\n=== Preparation Complete ===")


if __name__ == "__main__":
    main()
