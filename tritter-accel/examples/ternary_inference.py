#!/usr/bin/env python3
"""
Ternary inference example for tritter-accel.

This example demonstrates:
1. Creating random weights and input tensors
2. Quantizing weights to ternary using AbsMean scaling
3. Packing weights into efficient 2-bit representation
4. Running ternary matrix multiplication
5. Comparing output to standard float32 matmul

Key insight: Ternary matmul replaces multiplications with additions/subtractions,
potentially offering significant speedup on hardware that supports it.
"""

import sys
import time

import numpy as np


def float_matmul(input_tensor, weights):
    """Standard float32 matrix multiplication: input @ weights.T"""
    return input_tensor @ weights.T


def main():
    try:
        from tritter_accel import (
            pack_ternary_weights,
            quantize_weights_absmean,
            ternary_matmul,
            version,
        )
    except ImportError as e:
        print("Error: tritter_accel module not found.")
        print("To install, run: cd tritter-accel && maturin develop --release")
        print(f"Import error: {e}")
        sys.exit(1)

    print(f"tritter-accel version: {version()}")
    print("=" * 60)
    print("Ternary Inference Example")
    print("=" * 60)

    # Configuration
    batch_size = 32
    in_features = 512
    out_features = 256

    np.random.seed(42)

    # Create random weights (simulating a trained layer)
    print("\n1. Creating weights and input...")
    print("-" * 40)

    weights = np.random.randn(out_features, in_features).astype(np.float32) * 0.1
    input_tensor = np.random.randn(batch_size, in_features).astype(np.float32)

    print(f"Weights shape: {weights.shape}")
    print(f"Input shape:   {input_tensor.shape}")
    print(f"Expected output shape: ({batch_size}, {out_features})")

    # Step 2: Quantize weights to ternary
    print("\n2. Quantizing weights to ternary...")
    print("-" * 40)

    ternary_weights, scales = quantize_weights_absmean(weights)

    # Verify quantization
    unique_vals = np.unique(ternary_weights)
    print(f"Ternary values: {unique_vals.astype(int)}")
    print(f"Scales range: [{scales.min():.6f}, {scales.max():.6f}]")

    # Step 3: Pack ternary weights
    print("\n3. Packing weights into 2-bit representation...")
    print("-" * 40)

    packed_weights, packed_scales = pack_ternary_weights(ternary_weights, scales)

    original_bytes = weights.nbytes
    packed_bytes = packed_weights.nbytes + packed_scales.nbytes

    print(f"Original size: {original_bytes:,} bytes")
    print(f"Packed size:   {packed_bytes:,} bytes")
    print(f"Compression:   {original_bytes / packed_bytes:.2f}x")

    # Step 4: Run ternary matrix multiplication
    print("\n4. Running ternary matmul...")
    print("-" * 40)

    ternary_output = ternary_matmul(
        input_tensor,
        packed_weights,
        packed_scales,
        (out_features, in_features)
    )

    print(f"Output shape: {ternary_output.shape}")
    print(f"Output dtype: {ternary_output.dtype}")

    # Step 5: Compare to float matmul
    print("\n5. Comparing to float32 matmul...")
    print("-" * 40)

    # Compute float reference using dequantized weights
    dequantized_weights = ternary_weights * scales[:, np.newaxis]
    float_output = float_matmul(input_tensor, dequantized_weights)

    # Compute difference
    diff = np.abs(ternary_output - float_output)
    max_diff = diff.max()
    mean_diff = diff.mean()

    print(f"Max absolute difference:  {max_diff:.6e}")
    print(f"Mean absolute difference: {mean_diff:.6e}")

    # These should be essentially zero (numerical precision only)
    if max_diff < 1e-5:
        print("Result: PASS - Ternary matmul matches float reference")
    else:
        print("Result: WARN - Larger than expected difference")

    # Compare to original (non-quantized) float matmul
    print("\n6. Quantization error analysis...")
    print("-" * 40)

    original_output = float_matmul(input_tensor, weights)

    quant_diff = np.abs(ternary_output - original_output)
    mse = np.mean((ternary_output - original_output) ** 2)

    # Compute cosine similarity
    dot = np.sum(ternary_output * original_output)
    norm_t = np.sqrt(np.sum(ternary_output ** 2))
    norm_o = np.sqrt(np.sum(original_output ** 2))
    cosine_sim = dot / (norm_t * norm_o + 1e-8)

    print(f"Max error vs original:  {quant_diff.max():.6f}")
    print(f"Mean error vs original: {quant_diff.mean():.6f}")
    print(f"MSE vs original:        {mse:.6f}")
    print(f"Cosine similarity:      {cosine_sim:.6f}")

    # Timing comparison
    print("\n7. Performance comparison...")
    print("-" * 40)

    # Warm up
    _ = ternary_matmul(input_tensor, packed_weights, packed_scales, (out_features, in_features))
    _ = float_matmul(input_tensor, dequantized_weights)

    n_iterations = 100

    # Time ternary matmul
    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = ternary_matmul(input_tensor, packed_weights, packed_scales, (out_features, in_features))
    ternary_time = (time.perf_counter() - start) / n_iterations

    # Time float matmul (using numpy, which is highly optimized)
    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = float_matmul(input_tensor, dequantized_weights)
    float_time = (time.perf_counter() - start) / n_iterations

    print(f"Ternary matmul: {ternary_time * 1000:.3f} ms")
    print(f"Float matmul:   {float_time * 1000:.3f} ms")
    print(f"Ratio: {float_time / ternary_time:.2f}x")

    # Note: The Rust CPU implementation may be slower than numpy for small sizes
    # because numpy uses highly optimized BLAS. The real benefit comes from:
    # 1. GPU acceleration (not shown in this example)
    # 2. Memory bandwidth reduction (16x compression)
    # 3. Avoiding multiplications (significant on some hardware)

    # Show sample output
    print("\n8. Sample output values...")
    print("-" * 40)
    print(f"Ternary output [0, :5]: {ternary_output[0, :5]}")
    print(f"Float output [0, :5]:   {float_output[0, :5]}")
    print(f"Original output [0, :5]: {original_output[0, :5]}")

    print("\n" + "=" * 60)
    print("Ternary inference complete!")
    print("=" * 60)
    print("\nKey takeaways:")
    print("- Ternary weights achieve ~16x compression")
    print("- Ternary matmul uses only addition/subtraction")
    print("- Some accuracy loss from quantization is expected")
    print("- Full benefits require GPU acceleration (--features cuda)")


if __name__ == "__main__":
    main()
