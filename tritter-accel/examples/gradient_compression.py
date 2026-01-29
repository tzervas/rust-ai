#!/usr/bin/env python3
"""
VSA gradient compression example for tritter-accel.

This example demonstrates:
1. Creating sample gradients (simulating a training step)
2. Compressing gradients using VSA random projection
3. Decompressing gradients back to original dimension
4. Measuring compression ratio and reconstruction error

VSA (Vector Symbolic Architecture) gradient compression uses random projection
to reduce gradient communication overhead in distributed training. This is
particularly useful for:
- Multi-GPU training (reduce AllReduce bandwidth)
- Federated learning (compress client updates)
- Gradient checkpointing (reduce memory)

The compression is lossy but preserves gradient direction reasonably well.
"""

import sys
import time

import numpy as np


def main():
    try:
        from tritter_accel import (
            compress_gradients_vsa,
            decompress_gradients_vsa,
            version,
        )
    except ImportError as e:
        print("Error: tritter_accel module not found.")
        print("To install, run: cd tritter-accel && maturin develop --release")
        print(f"Import error: {e}")
        sys.exit(1)

    print(f"tritter-accel version: {version()}")
    print("=" * 60)
    print("VSA Gradient Compression Example")
    print("=" * 60)

    # Configuration
    np.random.seed(42)

    # Simulate gradients from a model (e.g., 1M parameters)
    gradient_dim = 1_000_000

    print("\n1. Creating sample gradients...")
    print("-" * 40)

    # Realistic gradient distribution: mostly small values with some larger ones
    # This mimics actual neural network gradients
    gradients = np.random.randn(gradient_dim).astype(np.float32) * 0.01

    # Add some larger gradients (tail of distribution)
    large_idx = np.random.choice(gradient_dim, size=gradient_dim // 100, replace=False)
    gradients[large_idx] *= 10

    print(f"Gradient dimension: {gradient_dim:,}")
    print(f"Gradient dtype: {gradients.dtype}")
    print(f"Gradient range: [{gradients.min():.6f}, {gradients.max():.6f}]")
    print(f"Gradient mean: {gradients.mean():.6f}")
    print(f"Gradient std: {gradients.std():.6f}")
    print(f"Gradient L2 norm: {np.linalg.norm(gradients):.4f}")

    # Test different compression ratios
    compression_ratios = [0.5, 0.1, 0.05, 0.01]

    print("\n2. Testing different compression ratios...")
    print("-" * 40)
    print(f"{'Ratio':>8} {'Compressed':>12} {'Original':>12} {'MSE':>12} {'Cosine':>8} {'Time':>10}")
    print("-" * 70)

    results = []
    seed = 12345  # Use same seed for reproducibility

    for ratio in compression_ratios:
        # Compress
        start = time.perf_counter()
        compressed, returned_seed = compress_gradients_vsa(gradients, ratio, seed)
        compress_time = time.perf_counter() - start

        # Decompress
        start = time.perf_counter()
        decompressed = decompress_gradients_vsa(compressed, gradient_dim, returned_seed)
        decompress_time = time.perf_counter() - start

        total_time = compress_time + decompress_time

        # Compute metrics
        mse = np.mean((gradients - decompressed) ** 2)

        # Cosine similarity (measures direction preservation)
        dot = np.sum(gradients * decompressed)
        norm_g = np.linalg.norm(gradients)
        norm_d = np.linalg.norm(decompressed)
        cosine = dot / (norm_g * norm_d + 1e-8)

        # Compression stats
        original_bytes = gradients.nbytes
        compressed_bytes = compressed.nbytes
        actual_ratio = compressed_bytes / original_bytes

        print(f"{ratio:>8.2f} {compressed_bytes:>12,} {original_bytes:>12,} "
              f"{mse:>12.6f} {cosine:>8.4f} {total_time*1000:>8.2f}ms")

        results.append({
            'ratio': ratio,
            'compressed': compressed,
            'decompressed': decompressed,
            'mse': mse,
            'cosine': cosine,
            'time': total_time
        })

    # Detailed analysis of one compression ratio
    print("\n3. Detailed analysis (ratio=0.1)...")
    print("-" * 40)

    ratio = 0.1
    compressed, seed_out = compress_gradients_vsa(gradients, ratio, seed)
    decompressed = decompress_gradients_vsa(compressed, gradient_dim, seed_out)

    print(f"Original gradient dim:   {gradient_dim:,}")
    print(f"Compressed dim:          {len(compressed):,}")
    print(f"Actual compression:      {len(compressed) / gradient_dim:.4f}")
    print(f"Original bytes:          {gradients.nbytes:,}")
    print(f"Compressed bytes:        {compressed.nbytes:,}")
    print(f"Space savings:           {(1 - compressed.nbytes / gradients.nbytes) * 100:.1f}%")

    # Error analysis
    errors = gradients - decompressed
    print(f"\nReconstruction errors:")
    print(f"  MSE:           {np.mean(errors**2):.8f}")
    print(f"  RMSE:          {np.sqrt(np.mean(errors**2)):.8f}")
    print(f"  MAE:           {np.mean(np.abs(errors)):.8f}")
    print(f"  Max error:     {np.max(np.abs(errors)):.8f}")
    print(f"  Error std:     {np.std(errors):.8f}")

    # Direction preservation
    dot = np.sum(gradients * decompressed)
    norm_g = np.linalg.norm(gradients)
    norm_d = np.linalg.norm(decompressed)
    cosine = dot / (norm_g * norm_d + 1e-8)

    print(f"\nDirection metrics:")
    print(f"  Cosine similarity: {cosine:.6f}")
    print(f"  Angle (degrees):   {np.arccos(np.clip(cosine, -1, 1)) * 180 / np.pi:.2f}")

    # Magnitude preservation
    print(f"\nMagnitude metrics:")
    print(f"  Original L2 norm:      {norm_g:.6f}")
    print(f"  Reconstructed L2 norm: {norm_d:.6f}")
    print(f"  Norm ratio:            {norm_d / norm_g:.6f}")

    # Show that compression is deterministic
    print("\n4. Reproducibility test...")
    print("-" * 40)

    compressed1, _ = compress_gradients_vsa(gradients, 0.1, seed)
    compressed2, _ = compress_gradients_vsa(gradients, 0.1, seed)

    if np.allclose(compressed1, compressed2):
        print("PASS: Same seed produces identical compression")
    else:
        print("FAIL: Non-deterministic compression")

    # Different seed should give different results
    compressed3, _ = compress_gradients_vsa(gradients, 0.1, seed + 1)
    if not np.allclose(compressed1, compressed3):
        print("PASS: Different seeds produce different compressions")
    else:
        print("WARN: Different seeds produced same compression")

    # Demonstrate use case: simulated distributed training
    print("\n5. Simulated distributed training scenario...")
    print("-" * 40)

    n_workers = 4
    worker_gradients = [
        np.random.randn(gradient_dim).astype(np.float32) * 0.01
        for _ in range(n_workers)
    ]

    # Without compression: AllReduce full gradients
    uncompressed_bytes = gradient_dim * 4 * n_workers  # float32 per worker

    # With compression (0.1 ratio): AllReduce compressed gradients
    ratio = 0.1
    compressed_dim = int(gradient_dim * ratio)
    compressed_bytes = compressed_dim * 4 * n_workers

    print(f"Number of workers: {n_workers}")
    print(f"Parameters per worker: {gradient_dim:,}")
    print(f"\nWithout compression:")
    print(f"  AllReduce data: {uncompressed_bytes / 1024 / 1024:.2f} MB")
    print(f"\nWith VSA compression (ratio={ratio}):")
    print(f"  AllReduce data: {compressed_bytes / 1024 / 1024:.2f} MB")
    print(f"  Bandwidth reduction: {uncompressed_bytes / compressed_bytes:.1f}x")

    # Simulate the workflow
    print("\nSimulating workflow:")
    seed = 42
    total_compress_time = 0
    total_decompress_time = 0

    # Each worker compresses their gradients
    compressed_grads = []
    for i, grads in enumerate(worker_gradients):
        start = time.perf_counter()
        comp, _ = compress_gradients_vsa(grads, ratio, seed)
        total_compress_time += time.perf_counter() - start
        compressed_grads.append(comp)

    # Aggregate compressed gradients (simplified: just sum)
    aggregated = np.sum(compressed_grads, axis=0) / n_workers

    # Decompress aggregated gradient
    start = time.perf_counter()
    decompressed_agg = decompress_gradients_vsa(aggregated.astype(np.float32), gradient_dim, seed)
    total_decompress_time = time.perf_counter() - start

    # Compare to true average
    true_average = np.mean(worker_gradients, axis=0)
    agg_cosine = np.dot(decompressed_agg, true_average) / (
        np.linalg.norm(decompressed_agg) * np.linalg.norm(true_average) + 1e-8
    )

    print(f"  Compression time ({n_workers} workers): {total_compress_time*1000:.2f} ms")
    print(f"  Decompression time: {total_decompress_time*1000:.2f} ms")
    print(f"  Aggregated gradient cosine similarity: {agg_cosine:.4f}")

    print("\n" + "=" * 60)
    print("Gradient compression complete!")
    print("=" * 60)
    print("\nKey takeaways:")
    print("- VSA compression can achieve 10-100x reduction")
    print("- Higher compression = more error (but direction preserved)")
    print("- Same seed ensures reproducible compression/decompression")
    print("- Useful for distributed training bandwidth reduction")


if __name__ == "__main__":
    main()
