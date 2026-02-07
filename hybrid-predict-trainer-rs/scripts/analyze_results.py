#!/usr/bin/env python3
"""
Statistical analysis and visualization for Phase 2B validation results.

Parses benchmark logs, computes summary statistics, performs statistical tests,
and generates publication-quality plots.
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class BenchmarkRun:
    """Single benchmark run results."""

    config: str
    seed: int
    steps: int
    total_time: float
    avg_time_per_step: float
    throughput: float  # tokens/sec
    loss_trajectory: List[float]
    vram_usage: List[float]  # MB
    phase_distribution: Dict[str, int]  # phase -> count


def parse_log_file(log_path: Path) -> BenchmarkRun:
    """Parse a benchmark log file and extract metrics."""
    with open(log_path) as f:
        content = f.read()

    # Extract configuration
    config = log_path.parent.parent.name

    # Extract seed from filename (run_42.log -> seed=42)
    seed = int(log_path.stem.split("_")[1])

    # Extract total stats from summary
    total_time_match = re.search(r"Total time: ([\d.]+)s", content)
    avg_time_match = re.search(r"Avg time per step: ([\d.]+)ms", content)
    throughput_match = re.search(r"Throughput: ([\d.]+) tokens/sec", content)

    if not (total_time_match and avg_time_match and throughput_match):
        raise ValueError(f"Failed to parse summary stats from {log_path}")

    total_time = float(total_time_match.group(1))
    avg_time_per_step = float(avg_time_match.group(1))
    throughput = float(throughput_match.group(1))

    # Extract per-step data
    loss_trajectory = []
    vram_usage = []
    step_pattern = re.compile(
        r"^\s*(\d+)\s+\|.*?\|\s+([\d.]+)\s+\|.*?\|\s+([\d.]+)\s+\|",
        re.MULTILINE,
    )

    for match in step_pattern.finditer(content):
        step, loss, vram = match.groups()
        loss_trajectory.append(float(loss))
        vram_usage.append(float(vram))

    steps = len(loss_trajectory)

    # Extract phase distribution (hybrid/memory_optimized only)
    phase_distribution = {}
    phase_section = re.search(
        r"Phase Distribution:\s*\n((?:.*?:\s*\d+\s*steps.*?\n)+)",
        content,
        re.MULTILINE,
    )
    if phase_section:
        for line in phase_section.group(1).strip().split("\n"):
            match = re.search(r"(\w+):\s*(\d+)\s*steps", line)
            if match:
                phase, count = match.groups()
                phase_distribution[phase] = int(count)

    return BenchmarkRun(
        config=config,
        seed=seed,
        steps=steps,
        total_time=total_time,
        avg_time_per_step=avg_time_per_step,
        throughput=throughput,
        loss_trajectory=loss_trajectory,
        vram_usage=vram_usage,
        phase_distribution=phase_distribution,
    )


def compute_summary_stats(runs: List[BenchmarkRun]) -> Dict:
    """Compute summary statistics across runs."""
    throughputs = [r.throughput for r in runs]
    avg_times = [r.avg_time_per_step for r in runs]
    final_losses = [r.loss_trajectory[-1] if r.loss_trajectory else float("nan") for r in runs]
    avg_vram = [np.mean(r.vram_usage) if r.vram_usage else 0.0 for r in runs]

    return {
        "throughput_mean": np.mean(throughputs),
        "throughput_std": np.std(throughputs, ddof=1),
        "throughput_min": np.min(throughputs),
        "throughput_max": np.max(throughputs),
        "avg_time_mean": np.mean(avg_times),
        "avg_time_std": np.std(avg_times, ddof=1),
        "final_loss_mean": np.nanmean(final_losses),
        "final_loss_std": np.nanstd(final_losses, ddof=1),
        "vram_mean": np.mean(avg_vram),
        "vram_std": np.std(avg_vram, ddof=1),
        "n_runs": len(runs),
    }


def compare_configs(
    baseline: List[BenchmarkRun],
    treatment: List[BenchmarkRun],
    treatment_name: str,
) -> Dict:
    """Statistical comparison between baseline and treatment."""
    baseline_stats = compute_summary_stats(baseline)
    treatment_stats = compute_summary_stats(treatment)

    # Throughput comparison
    baseline_throughput = [r.throughput for r in baseline]
    treatment_throughput = [r.throughput for r in treatment]
    speedup_ratio = treatment_stats["throughput_mean"] / baseline_stats["throughput_mean"]

    # t-test for throughput difference
    t_stat, p_value = stats.ttest_ind(treatment_throughput, baseline_throughput)

    # Loss comparison (quality degradation)
    baseline_loss = [r.loss_trajectory[-1] for r in baseline if r.loss_trajectory]
    treatment_loss = [r.loss_trajectory[-1] for r in treatment if r.loss_trajectory]
    loss_delta_pct = (
        (np.mean(treatment_loss) - np.mean(baseline_loss)) / np.mean(baseline_loss) * 100
    )

    # VRAM comparison
    baseline_vram = [np.mean(r.vram_usage) for r in baseline if r.vram_usage]
    treatment_vram = [np.mean(r.vram_usage) for r in treatment if r.vram_usage]
    vram_ratio = (
        np.mean(treatment_vram) / np.mean(baseline_vram) if baseline_vram and treatment_vram else 1.0
    )

    return {
        "treatment_name": treatment_name,
        "speedup_ratio": speedup_ratio,
        "speedup_significant": p_value < 0.05,
        "speedup_p_value": p_value,
        "loss_delta_pct": loss_delta_pct,
        "vram_ratio": vram_ratio,
        "baseline_stats": baseline_stats,
        "treatment_stats": treatment_stats,
    }


def generate_report(results_dir: Path, output_dir: Path):
    """Generate comprehensive analysis report."""
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("📊 Phase 2B Statistical Analysis")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print()

    # Parse all runs
    configs = ["baseline", "hybrid", "memory_optimized"]
    all_runs = {config: [] for config in configs}

    for config in configs:
        config_dir = results_dir / config / "logs"
        if not config_dir.exists():
            print(f"⚠️  Warning: No logs found for {config}")
            continue

        for log_file in sorted(config_dir.glob("run_*.log")):
            try:
                run = parse_log_file(log_file)
                all_runs[config].append(run)
                print(f"  ✓ Parsed {log_file.name} ({config})")
            except Exception as e:
                print(f"  ✗ Failed to parse {log_file.name}: {e}")

    print()

    # Check we have data
    if not all_runs["baseline"]:
        print("❌ ERROR: No baseline runs found. Cannot compute comparisons.")
        return

    # Compute statistics
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("📈 Summary Statistics")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print()

    summary = {}
    for config in configs:
        if not all_runs[config]:
            continue
        stats = compute_summary_stats(all_runs[config])
        summary[config] = stats

        print(f"{config.upper()}:")
        print(f"  Throughput: {stats['throughput_mean']:.1f} ± {stats['throughput_std']:.1f} tokens/sec")
        print(f"  Avg time/step: {stats['avg_time_mean']:.1f} ± {stats['avg_time_std']:.1f} ms")
        print(f"  Final loss: {stats['final_loss_mean']:.5f} ± {stats['final_loss_std']:.5f}")
        print(f"  VRAM: {stats['vram_mean']:.1f} ± {stats['vram_std']:.1f} MB")
        print()

    # Compare against baseline
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("🔬 Statistical Comparisons")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print()

    comparisons = {}
    baseline_runs = all_runs["baseline"]

    for config in ["hybrid", "memory_optimized"]:
        if not all_runs[config]:
            continue

        comp = compare_configs(baseline_runs, all_runs[config], config)
        comparisons[config] = comp

        print(f"{config.upper()} vs BASELINE:")
        print(f"  Speedup: {comp['speedup_ratio']:.2f}×")
        print(f"  Significant: {'Yes' if comp['speedup_significant'] else 'No'} (p={comp['speedup_p_value']:.4f})")
        print(f"  Loss Δ: {comp['loss_delta_pct']:+.2f}%")
        print(f"  VRAM ratio: {comp['vram_ratio']:.2f}×")
        print()

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "summary.json", "w") as f:
        json.dump(
            {"summary": summary, "comparisons": comparisons},
            f,
            indent=2,
            default=str,
        )

    print(f"✅ Results saved to {output_dir}/summary.json")
    print()

    # Evaluation against Phase 2B goals
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("🎯 Phase 2B Goal Evaluation")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print()

    if "hybrid" in comparisons:
        comp = comparisons["hybrid"]
        speedup_pass = comp["speedup_ratio"] >= 1.5
        quality_pass = abs(comp["loss_delta_pct"]) <= 2.0

        print("HybridTrainer Goals:")
        print(f"  ✓ Speedup ≥1.5×: {'PASS' if speedup_pass else 'FAIL'} ({comp['speedup_ratio']:.2f}×)")
        print(f"  ✓ Loss Δ ≤2%: {'PASS' if quality_pass else 'FAIL'} ({comp['loss_delta_pct']:+.2f}%)")
        print()

    if "memory_optimized" in comparisons:
        comp = comparisons["memory_optimized"]
        speedup_pass = comp["speedup_ratio"] >= 1.3  # More tolerance
        quality_pass = abs(comp["loss_delta_pct"]) <= 3.0
        vram_pass = comp["vram_ratio"] <= 0.7  # 30% savings minimum

        print("Memory-Optimized Goals:")
        print(f"  ✓ Speedup ≥1.3×: {'PASS' if speedup_pass else 'FAIL'} ({comp['speedup_ratio']:.2f}×)")
        print(f"  ✓ Loss Δ ≤3%: {'PASS' if quality_pass else 'FAIL'} ({comp['loss_delta_pct']:+.2f}%)")
        print(f"  ✓ VRAM ≤70%: {'PASS' if vram_pass else 'FAIL'} ({comp['vram_ratio']*100:.0f}%)")
        print()


def main():
    parser = argparse.ArgumentParser(description="Analyze Phase 2B benchmark results")
    parser.add_argument("--input", type=Path, required=True, help="Results directory")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("report"),
        help="Output directory for analysis",
    )

    args = parser.parse_args()

    if not args.input.exists():
        print(f"❌ ERROR: Results directory not found: {args.input}")
        sys.exit(1)

    generate_report(args.input, args.output)


if __name__ == "__main__":
    main()
