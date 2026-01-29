#!/usr/bin/env python3
"""
VSA (Vector Symbolic Architecture) operations example for tritter-accel.

This example demonstrates hyperdimensional computing operations:
1. Creating random ternary vectors with vsa_random()
2. Binding vectors with vsa_bind() (association/composition)
3. Unbinding to recover original vectors
4. Computing similarity with vsa_similarity()
5. Bundling multiple vectors (superposition)

VSA enables symbolic AI operations with high-dimensional vectors:
- Bind: Creates associations (like key-value pairs)
- Bundle: Creates superpositions (like sets/bags)
- Similarity: Measures semantic relatedness

Note: GPU-accelerated VSA requires the cuda feature:
    maturin develop --release --features cuda
"""

import sys

import numpy as np


def main():
    try:
        from tritter_accel import cuda_available_py, version
    except ImportError as e:
        print("Error: tritter_accel module not found.")
        print("To install, run: cd tritter-accel && maturin develop --release")
        print(f"Import error: {e}")
        sys.exit(1)

    print(f"tritter-accel version: {version()}")
    print("=" * 60)
    print("VSA Operations Example")
    print("=" * 60)

    # Check if CUDA VSA ops are available
    cuda_available = cuda_available_py()
    print(f"\nCUDA available: {cuda_available}")

    # Try to import CUDA-specific VSA functions
    try:
        from tritter_accel import (
            vsa_bind,
            vsa_bundle,
            vsa_random,
            vsa_similarity,
            vsa_unbind,
        )
        vsa_available = True
        print("VSA operations available (cuda feature enabled)")
    except ImportError:
        vsa_available = False
        print("VSA operations not available (need --features cuda)")
        print("\nTo enable GPU-accelerated VSA operations:")
        print("  cd tritter-accel")
        print("  maturin develop --release --features cuda")
        print("\nShowing CPU-only simulation instead...")

    if not vsa_available:
        # Demonstrate with numpy as fallback
        print("\n" + "=" * 60)
        print("CPU Simulation (numpy)")
        print("=" * 60)
        demonstrate_vsa_with_numpy()
        return

    # Device selection
    device = "cuda" if cuda_available else "cpu"
    print(f"Using device: {device}")

    # Configuration
    dim = 10000  # High dimensionality is key to VSA

    print("\n1. Creating random ternary vectors...")
    print("-" * 40)

    # Create semantic vectors for concepts
    apple = vsa_random(dim, seed=1, device=device)
    banana = vsa_random(dim, seed=2, device=device)
    red = vsa_random(dim, seed=3, device=device)
    yellow = vsa_random(dim, seed=4, device=device)
    fruit = vsa_random(dim, seed=5, device=device)
    color = vsa_random(dim, seed=6, device=device)

    print(f"Vector dimension: {dim}")
    print(f"Vector dtype: {apple.dtype}")
    print(f"Sample values: {apple[:10]}")

    # Verify vectors are ternary
    unique = np.unique(apple)
    print(f"Unique values: {unique}")
    assert set(unique).issubset({-1, 0, 1}), "Non-ternary values detected!"

    print("\n2. Binding vectors (association)...")
    print("-" * 40)

    # Bind creates associations: "red apple", "yellow banana"
    red_apple = vsa_bind(red, apple, device=device)
    yellow_banana = vsa_bind(yellow, banana, device=device)

    print("Created associations:")
    print("  red + apple -> red_apple")
    print("  yellow + banana -> yellow_banana")

    # Binding property: bound vector is dissimilar to both components
    sim_bound_apple = vsa_similarity(red_apple, apple, metric="cosine", device=device)
    sim_bound_red = vsa_similarity(red_apple, red, metric="cosine", device=device)

    print(f"\nBinding property check:")
    print(f"  similarity(red_apple, apple): {sim_bound_apple:.4f}")
    print(f"  similarity(red_apple, red): {sim_bound_red:.4f}")
    print("  (Both should be near 0 - bound is dissimilar to components)")

    print("\n3. Unbinding to recover associations...")
    print("-" * 40)

    # Query: "What color is the apple?" = unbind(red_apple, apple)
    recovered_color = vsa_unbind(red_apple, apple, device=device)

    # The recovered vector should be similar to red
    sim_to_red = vsa_similarity(recovered_color, red, metric="cosine", device=device)
    sim_to_yellow = vsa_similarity(recovered_color, yellow, metric="cosine", device=device)

    print("Query: What color is the apple?")
    print(f"  similarity(recovered, red): {sim_to_red:.4f}")
    print(f"  similarity(recovered, yellow): {sim_to_yellow:.4f}")
    print(f"  Correctly identifies: {'red' if sim_to_red > sim_to_yellow else 'yellow'}")

    # Query: "What is yellow?" = unbind(yellow_banana, yellow)
    recovered_fruit = vsa_unbind(yellow_banana, yellow, device=device)

    sim_to_apple = vsa_similarity(recovered_fruit, apple, metric="cosine", device=device)
    sim_to_banana = vsa_similarity(recovered_fruit, banana, metric="cosine", device=device)

    print("\nQuery: What is yellow?")
    print(f"  similarity(recovered, apple): {sim_to_apple:.4f}")
    print(f"  similarity(recovered, banana): {sim_to_banana:.4f}")
    print(f"  Correctly identifies: {'banana' if sim_to_banana > sim_to_apple else 'apple'}")

    print("\n4. Computing similarities...")
    print("-" * 40)

    # Self-similarity should be 1.0
    self_sim = vsa_similarity(apple, apple, metric="cosine", device=device)
    print(f"Self-similarity (apple, apple): {self_sim:.4f}")

    # Random vectors should be nearly orthogonal (similarity ~0)
    random_sim = vsa_similarity(apple, banana, metric="cosine", device=device)
    print(f"Random vector similarity (apple, banana): {random_sim:.4f}")

    # Different similarity metrics
    print("\nDifferent similarity metrics for (apple, red):")
    cosine = vsa_similarity(apple, red, metric="cosine", device=device)
    dot = vsa_similarity(apple, red, metric="dot", device=device)
    hamming = vsa_similarity(apple, red, metric="hamming", device=device)

    print(f"  Cosine:  {cosine:.4f}")
    print(f"  Dot:     {dot:.1f}")
    print(f"  Hamming: {hamming:.0f}")

    print("\n5. Bundling vectors (superposition)...")
    print("-" * 40)

    # Bundle creates sets: "fruits" = apple + banana
    fruits_bundle = vsa_bundle([apple, banana], device=device)

    print("Created bundle: fruits = apple + banana")

    # Bundle is similar to both components
    sim_to_apple = vsa_similarity(fruits_bundle, apple, metric="cosine", device=device)
    sim_to_banana = vsa_similarity(fruits_bundle, banana, metric="cosine", device=device)
    sim_to_red = vsa_similarity(fruits_bundle, red, metric="cosine", device=device)

    print(f"\nBundle membership test:")
    print(f"  similarity(fruits, apple): {sim_to_apple:.4f}")
    print(f"  similarity(fruits, banana): {sim_to_banana:.4f}")
    print(f"  similarity(fruits, red): {sim_to_red:.4f}")
    print("  (Apple and banana should be similar, red should not)")

    # Bundle many vectors
    print("\n6. Bundling many vectors...")
    print("-" * 40)

    many_vectors = [vsa_random(dim, seed=i, device=device) for i in range(10, 20)]
    big_bundle = vsa_bundle(many_vectors, device=device)

    # Check similarity to one of the components
    target = many_vectors[5]
    sim = vsa_similarity(big_bundle, target, metric="cosine", device=device)
    print(f"Bundle of 10 vectors:")
    print(f"  similarity to component: {sim:.4f}")

    # Similarity to random non-member
    non_member = vsa_random(dim, seed=999, device=device)
    sim_non = vsa_similarity(big_bundle, non_member, metric="cosine", device=device)
    print(f"  similarity to non-member: {sim_non:.4f}")

    print("\n7. Building a simple knowledge base...")
    print("-" * 40)

    # Create role-filler pairs
    has_color = vsa_random(dim, seed=100, device=device)
    has_shape = vsa_random(dim, seed=101, device=device)

    circle = vsa_random(dim, seed=102, device=device)
    square = vsa_random(dim, seed=103, device=device)

    # Create structured representations
    # "red circle" = bind(has_color, red) + bind(has_shape, circle)
    red_circle = vsa_bundle([
        vsa_bind(has_color, red, device=device),
        vsa_bind(has_shape, circle, device=device)
    ], device=device)

    # "yellow square" = bind(has_color, yellow) + bind(has_shape, square)
    yellow_square = vsa_bundle([
        vsa_bind(has_color, yellow, device=device),
        vsa_bind(has_shape, square, device=device)
    ], device=device)

    print("Knowledge base created:")
    print("  red_circle = has_color:red + has_shape:circle")
    print("  yellow_square = has_color:yellow + has_shape:square")

    # Query: What color is the red_circle?
    query_color = vsa_unbind(red_circle, has_color, device=device)
    sim_red = vsa_similarity(query_color, red, metric="cosine", device=device)
    sim_yellow = vsa_similarity(query_color, yellow, metric="cosine", device=device)

    print(f"\nQuery: What color is red_circle?")
    print(f"  Similarity to red: {sim_red:.4f}")
    print(f"  Similarity to yellow: {sim_yellow:.4f}")
    print(f"  Answer: {'red' if sim_red > sim_yellow else 'yellow'}")

    # Query: What shape is yellow_square?
    query_shape = vsa_unbind(yellow_square, has_shape, device=device)
    sim_circle = vsa_similarity(query_shape, circle, metric="cosine", device=device)
    sim_square = vsa_similarity(query_shape, square, metric="cosine", device=device)

    print(f"\nQuery: What shape is yellow_square?")
    print(f"  Similarity to circle: {sim_circle:.4f}")
    print(f"  Similarity to square: {sim_square:.4f}")
    print(f"  Answer: {'square' if sim_square > sim_circle else 'circle'}")

    print("\n" + "=" * 60)
    print("VSA operations complete!")
    print("=" * 60)
    print("\nKey takeaways:")
    print("- Bind creates associations (key-value pairs)")
    print("- Unbind recovers associated vectors (queries)")
    print("- Bundle creates superpositions (sets/bags)")
    print("- High dimensionality enables near-orthogonal random vectors")
    print("- GPU acceleration provides speedup for large dimensions")


def demonstrate_vsa_with_numpy():
    """Fallback demonstration using pure numpy."""
    print("\nThis demonstrates VSA concepts using numpy (no tritter_accel GPU ops).")

    np.random.seed(42)
    dim = 1000  # Smaller for CPU demo

    def random_ternary(dim):
        return np.random.choice([-1, 0, 1], size=dim, p=[1/3, 1/3, 1/3]).astype(np.int8)

    def bind(a, b):
        # Ternary bind: multiply and clip
        result = a.astype(np.int32) * b.astype(np.int32)
        return np.clip(result, -1, 1).astype(np.int8)

    def unbind(bound, key):
        # Ternary unbind: same as bind for self-inverse keys
        return bind(bound, key)

    def cosine_similarity(a, b):
        dot = np.dot(a.astype(np.float64), b.astype(np.float64))
        norm_a = np.linalg.norm(a.astype(np.float64))
        norm_b = np.linalg.norm(b.astype(np.float64))
        return dot / (norm_a * norm_b + 1e-8)

    # Create vectors
    apple = random_ternary(dim)
    red = random_ternary(dim)
    yellow = random_ternary(dim)

    # Bind
    red_apple = bind(red, apple)

    # Unbind to recover
    recovered = unbind(red_apple, apple)

    # Check similarity
    sim_to_red = cosine_similarity(recovered, red)
    sim_to_yellow = cosine_similarity(recovered, yellow)

    print(f"\nVector dimension: {dim}")
    print(f"Created: apple, red, yellow (random ternary)")
    print(f"Bound: red_apple = bind(red, apple)")
    print(f"Recovered: unbind(red_apple, apple)")
    print(f"\nSimilarity to red: {sim_to_red:.4f}")
    print(f"Similarity to yellow: {sim_to_yellow:.4f}")
    print(f"Correctly identifies: {'red' if sim_to_red > sim_to_yellow else 'yellow'}")


if __name__ == "__main__":
    main()
