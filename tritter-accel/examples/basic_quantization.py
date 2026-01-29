#!/usr/bin/env python3
"""
Basic weight quantization example for tritter-accel.

This example demonstrates:
1. Creating sample weights (simulating a neural network layer)
2. Quantizing weights to ternary {-1, 0, +1} using AbsMean scaling
3. Analyzing the ternary distribution
4. Computing compression statistics

The AbsMean quantization method:
- Computes per-row mean absolute value as the scale factor
- Rounds weights to nearest ternary value: sign(w) * round(|w| / scale)
- Clips to {-1, 0, +1}
"""

import sys
from collections import Counter

import numpy as np


def main():
    try:
        from tritter_accel import quantize_weights_absmean, version
    except ImportError as e:
        print("Error: tritter_accel module not found.")
        print("To install, run: cd tritter-accel && maturin develop --release")
        print(f"Import error: {e}")
        sys.exit(1)

    print(f"tritter-accel version: {version()}")
    print("=" * 60)
    print("Basic Weight Quantization Example")
    print("=" * 60)

    # Create sample weights simulating a small linear layer
    # Shape: (out_features=128, in_features=256)
    np.random.seed(42)
    out_features, in_features = 128, 256

    # Simulate weights from a trained model (roughly normal distribution)
    weights = np.random.randn(out_features, in_features).astype(np.float32)
    weights *= 0.1  # Scale down like typical neural network weights

    print(f"\nOriginal weights shape: {weights.shape}")
    print(f"Original weights dtype: {weights.dtype}")
    print(f"Original weights range: [{weights.min():.4f}, {weights.max():.4f}]")
    print(f"Original weights mean: {weights.mean():.4f}")
    print(f"Original weights std: {weights.std():.4f}")

    # Quantize weights to ternary using AbsMean method
    print("\n" + "-" * 40)
    print("Quantizing to ternary...")
    print("-" * 40)

    ternary_weights, scales = quantize_weights_absmean(weights)

    print(f"Ternary weights shape: {ternary_weights.shape}")
    print(f"Ternary weights dtype: {ternary_weights.dtype}")
    print(f"Scales shape: {scales.shape}")
    print(f"Scales range: [{scales.min():.6f}, {scales.max():.6f}]")

    # Analyze ternary distribution
    print("\n" + "-" * 40)
    print("Ternary Distribution Analysis")
    print("-" * 40)

    # Count occurrences of each value
    unique, counts = np.unique(ternary_weights.flatten(), return_counts=True)
    total = counts.sum()

    for val, count in zip(unique, counts):
        pct = 100.0 * count / total
        bar = "#" * int(pct / 2)
        sign = "+" if val > 0 else (" " if val == 0 else "")
        print(f"  {sign}{int(val):2d}: {count:8d} ({pct:5.1f}%) {bar}")

    # Verify all values are ternary
    assert set(np.unique(ternary_weights.flatten())).issubset({-1.0, 0.0, 1.0}), \
        "Quantization produced non-ternary values!"
    print("\nAll values are valid ternary {-1, 0, +1}")

    # Compute compression statistics
    print("\n" + "-" * 40)
    print("Compression Statistics")
    print("-" * 40)

    # Original: float32 = 4 bytes per weight
    original_bytes = weights.size * 4

    # Packed ternary: 2 bits per weight = 4 weights per byte
    packed_bytes = (weights.size + 3) // 4  # ceiling division
    scales_bytes = scales.size * 4  # float32 scales
    compressed_bytes = packed_bytes + scales_bytes

    compression_ratio = original_bytes / compressed_bytes
    memory_reduction = (1 - compressed_bytes / original_bytes) * 100

    print(f"Original size:     {original_bytes:,} bytes ({original_bytes / 1024:.2f} KB)")
    print(f"Packed size:       {packed_bytes:,} bytes (weights)")
    print(f"Scales size:       {scales_bytes:,} bytes")
    print(f"Total compressed:  {compressed_bytes:,} bytes ({compressed_bytes / 1024:.2f} KB)")
    print(f"Compression ratio: {compression_ratio:.2f}x")
    print(f"Memory reduction:  {memory_reduction:.1f}%")

    # Demonstrate dequantization and reconstruction error
    print("\n" + "-" * 40)
    print("Reconstruction Quality")
    print("-" * 40)

    # Dequantize: multiply ternary values by their row scales
    dequantized = ternary_weights * scales[:, np.newaxis]

    # Compute reconstruction error
    mse = np.mean((weights - dequantized) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(weights - dequantized))
    max_error = np.max(np.abs(weights - dequantized))

    # Cosine similarity between original and reconstructed
    dot_product = np.sum(weights * dequantized)
    norm_original = np.sqrt(np.sum(weights ** 2))
    norm_dequant = np.sqrt(np.sum(dequantized ** 2))
    cosine_sim = dot_product / (norm_original * norm_dequant + 1e-8)

    print(f"MSE:               {mse:.6f}")
    print(f"RMSE:              {rmse:.6f}")
    print(f"MAE:               {mae:.6f}")
    print(f"Max error:         {max_error:.6f}")
    print(f"Cosine similarity: {cosine_sim:.6f}")

    # Show example row
    print("\n" + "-" * 40)
    print("Example: First row comparison")
    print("-" * 40)
    print(f"Original (first 10):    {weights[0, :10]}")
    print(f"Ternary (first 10):     {ternary_weights[0, :10].astype(int)}")
    print(f"Scale for row 0:        {scales[0]:.6f}")
    print(f"Dequantized (first 10): {dequantized[0, :10]}")

    print("\n" + "=" * 60)
    print("Quantization complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
