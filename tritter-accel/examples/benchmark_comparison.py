#!/usr/bin/env python3
"""
Performance benchmark comparison for tritter-accel.

This example compares:
1. Float matmul vs ternary matmul at various sizes
2. Gradient compression at different compression ratios
3. Memory usage statistics

Note: Results depend heavily on:
- CPU/GPU hardware
- Whether CUDA feature is enabled
- Matrix dimensions
- NumPy/BLAS configuration

For best results, run with GPU support:
    maturin develop --release --features cuda
"""

import sys
import time
from typing import Callable, List, Tuple

import numpy as np


def benchmark(func: Callable, n_warmup: int = 3, n_runs: int = 10) -> Tuple[float, float]:
    """
    Benchmark a function.

    Returns:
        Tuple of (mean_time_ms, std_time_ms)
    """
    # Warmup
    for _ in range(n_warmup):
        func()

    # Timed runs
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    return np.mean(times), np.std(times)


def main():
    try:
        from tritter_accel import (
            compress_gradients_vsa,
            decompress_gradients_vsa,
            pack_ternary_weights,
            quantize_weights_absmean,
            ternary_matmul,
            version,
            cuda_available_py,
        )
    except ImportError as e:
        print("Error: tritter_accel module not found.")
        print("To install, run: cd tritter-accel && maturin develop --release")
        print(f"Import error: {e}")
        sys.exit(1)

    print(f"tritter-accel version: {version()}")
    print(f"CUDA available: {cuda_available_py()}")
    print("=" * 70)
    print("Performance Benchmark Comparison")
    print("=" * 70)

    np.random.seed(42)

    # ==========================================================================
    # Benchmark 1: Float vs Ternary Matmul
    # ==========================================================================
    print("\n" + "=" * 70)
    print("BENCHMARK 1: Float32 vs Ternary Matrix Multiplication")
    print("=" * 70)

    # Test different matrix sizes
    sizes = [
        (32, 256, 256),     # Small: batch=32, in=256, out=256
        (32, 512, 512),     # Medium: batch=32, in=512, out=512
        (32, 1024, 1024),   # Large: batch=32, in=1024, out=1024
        (64, 2048, 2048),   # XL: batch=64, in=2048, out=2048
        (128, 4096, 4096),  # XXL: batch=128, in=4096, out=4096
    ]

    print(f"\n{'Size (B,In,Out)':>20} {'Float32 (ms)':>14} {'Ternary (ms)':>14} "
          f"{'Speedup':>10} {'Memory':>10}")
    print("-" * 70)

    matmul_results = []

    for batch, in_feat, out_feat in sizes:
        # Create data
        weights = np.random.randn(out_feat, in_feat).astype(np.float32) * 0.1
        input_data = np.random.randn(batch, in_feat).astype(np.float32)

        # Quantize and pack weights
        ternary_weights, scales = quantize_weights_absmean(weights)
        packed_weights, packed_scales = pack_ternary_weights(ternary_weights, scales)
        dequantized = ternary_weights * scales[:, np.newaxis]

        # Benchmark float matmul (numpy)
        def float_op():
            return input_data @ dequantized.T

        float_mean, float_std = benchmark(float_op)

        # Benchmark ternary matmul
        def ternary_op():
            return ternary_matmul(input_data, packed_weights, packed_scales, (out_feat, in_feat))

        ternary_mean, ternary_std = benchmark(ternary_op)

        # Compute speedup and memory ratio
        speedup = float_mean / ternary_mean

        float_bytes = weights.nbytes
        ternary_bytes = packed_weights.nbytes + packed_scales.nbytes
        memory_ratio = float_bytes / ternary_bytes

        size_str = f"({batch},{in_feat},{out_feat})"
        print(f"{size_str:>20} {float_mean:>10.2f}ms {ternary_mean:>10.2f}ms "
              f"{speedup:>10.2f}x {memory_ratio:>10.1f}x")

        matmul_results.append({
            'size': (batch, in_feat, out_feat),
            'float_time': float_mean,
            'ternary_time': ternary_mean,
            'speedup': speedup,
            'memory_ratio': memory_ratio
        })

    # Summary
    avg_speedup = np.mean([r['speedup'] for r in matmul_results])
    avg_memory = np.mean([r['memory_ratio'] for r in matmul_results])
    print("-" * 70)
    print(f"{'Average':>20} {'-':>14} {'-':>14} {avg_speedup:>10.2f}x {avg_memory:>10.1f}x")

    print("\nNotes:")
    print("- Speedup < 1.0 means ternary is slower (expected on CPU vs optimized BLAS)")
    print("- Memory ratio is always ~16x due to 2-bit packing")
    print("- GPU acceleration (--features cuda) provides significant speedup")

    # ==========================================================================
    # Benchmark 2: Gradient Compression at Different Ratios
    # ==========================================================================
    print("\n" + "=" * 70)
    print("BENCHMARK 2: VSA Gradient Compression Performance")
    print("=" * 70)

    gradient_sizes = [100_000, 1_000_000, 10_000_000]
    compression_ratios = [0.5, 0.1, 0.05, 0.01]

    print(f"\n{'Gradient Size':>15} {'Ratio':>8} {'Compress (ms)':>14} "
          f"{'Decompress (ms)':>16} {'Total (ms)':>12} {'Cosine':>8}")
    print("-" * 80)

    compression_results = []
    seed = 42

    for grad_size in gradient_sizes:
        gradients = np.random.randn(grad_size).astype(np.float32) * 0.01

        for ratio in compression_ratios:
            # Benchmark compression
            def compress_op():
                return compress_gradients_vsa(gradients, ratio, seed)

            compress_mean, _ = benchmark(compress_op, n_warmup=2, n_runs=5)
            compressed, seed_out = compress_gradients_vsa(gradients, ratio, seed)

            # Benchmark decompression
            def decompress_op():
                return decompress_gradients_vsa(compressed, grad_size, seed_out)

            decompress_mean, _ = benchmark(decompress_op, n_warmup=2, n_runs=5)
            decompressed = decompress_gradients_vsa(compressed, grad_size, seed_out)

            # Compute cosine similarity
            dot = np.dot(gradients, decompressed)
            norm_g = np.linalg.norm(gradients)
            norm_d = np.linalg.norm(decompressed)
            cosine = dot / (norm_g * norm_d + 1e-8)

            total = compress_mean + decompress_mean

            size_str = f"{grad_size:,}"
            print(f"{size_str:>15} {ratio:>8.2f} {compress_mean:>12.2f}ms "
                  f"{decompress_mean:>14.2f}ms {total:>10.2f}ms {cosine:>8.4f}")

            compression_results.append({
                'grad_size': grad_size,
                'ratio': ratio,
                'compress_time': compress_mean,
                'decompress_time': decompress_mean,
                'cosine': cosine
            })

    print("\nNotes:")
    print("- Lower compression ratio = smaller data but more error")
    print("- Cosine similarity measures gradient direction preservation")
    print("- Times scale roughly linearly with gradient size")

    # ==========================================================================
    # Benchmark 3: Quantization Performance
    # ==========================================================================
    print("\n" + "=" * 70)
    print("BENCHMARK 3: Weight Quantization Performance")
    print("=" * 70)

    weight_shapes = [
        (256, 256),
        (512, 512),
        (1024, 1024),
        (2048, 2048),
        (4096, 4096),
    ]

    print(f"\n{'Shape':>15} {'Quantize (ms)':>14} {'Pack (ms)':>12} "
          f"{'Total (ms)':>12} {'Throughput':>15}")
    print("-" * 70)

    for shape in weight_shapes:
        weights = np.random.randn(*shape).astype(np.float32) * 0.1

        # Benchmark quantization
        def quant_op():
            return quantize_weights_absmean(weights)

        quant_mean, _ = benchmark(quant_op, n_warmup=2, n_runs=5)
        ternary, scales = quantize_weights_absmean(weights)

        # Benchmark packing
        def pack_op():
            return pack_ternary_weights(ternary, scales)

        pack_mean, _ = benchmark(pack_op, n_warmup=2, n_runs=5)

        total = quant_mean + pack_mean

        # Compute throughput in MB/s
        weight_mb = weights.nbytes / (1024 * 1024)
        throughput = weight_mb / (total / 1000)  # MB/s

        shape_str = f"{shape[0]}x{shape[1]}"
        print(f"{shape_str:>15} {quant_mean:>12.2f}ms {pack_mean:>10.2f}ms "
              f"{total:>10.2f}ms {throughput:>12.1f} MB/s")

    print("\nNotes:")
    print("- Throughput measures end-to-end quantization + packing speed")
    print("- Target is >100 MB/s for production use")

    # ==========================================================================
    # Benchmark 4: Memory Usage Summary
    # ==========================================================================
    print("\n" + "=" * 70)
    print("BENCHMARK 4: Memory Usage Summary")
    print("=" * 70)

    print(f"\n{'Data Type':>20} {'Per Element':>15} {'1M Elements':>15} {'Ratio':>10}")
    print("-" * 65)

    n_elements = 1_000_000

    data_types = [
        ("Float32", 4, 1.0),
        ("Float16", 2, 2.0),
        ("Int8", 1, 4.0),
        ("Ternary (2-bit)", 0.25, 16.0),
    ]

    for name, bytes_per, ratio in data_types:
        total_bytes = int(n_elements * bytes_per)
        total_mb = total_bytes / (1024 * 1024)
        print(f"{name:>20} {bytes_per:>12.2f} B {total_mb:>12.2f} MB {ratio:>10.1f}x")

    print("\nNotes:")
    print("- Ternary packing achieves 16x compression vs float32")
    print("- Additional ~1% overhead for scale factors (one per row)")

    # ==========================================================================
    # Summary
    # ==========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\nMatrix Multiplication:")
    print(f"  Average speedup (vs numpy): {avg_speedup:.2f}x")
    print(f"  Memory compression: {avg_memory:.1f}x")

    # Best compression result
    best_compress = min(compression_results, key=lambda x: x['ratio'])
    print(f"\nGradient Compression (ratio={best_compress['ratio']}):")
    print(f"  Cosine similarity: {best_compress['cosine']:.4f}")
    print(f"  Compression time: {best_compress['compress_time']:.2f}ms")

    print("\nRecommendations:")
    print("- Use GPU acceleration (--features cuda) for best matmul performance")
    print("- Choose compression ratio based on accuracy requirements:")
    print("    - 0.1 (10%): Good balance of compression and accuracy")
    print("    - 0.05 (5%): Higher compression, slight accuracy loss")
    print("    - 0.01 (1%): Maximum compression, use with caution")

    print("\n" + "=" * 70)
    print("Benchmarks complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
