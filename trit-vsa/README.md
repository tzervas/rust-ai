# trit-vsa

A high-performance balanced ternary arithmetic library for Rust.

[![Crates.io](https://img.shields.io/crates/v/trit-vsa.svg)](https://crates.io/crates/trit-vsa)
[![Documentation](https://docs.rs/trit-vsa/badge.svg)](https://docs.rs/trit-vsa)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

`trit-vsa` provides core primitives for balanced ternary arithmetic, including:

- **Trit**: Single balanced ternary digit {-1, 0, +1}
- **Tryte3**: 3 trits packed in a byte (values -13 to +13)
- **Word6**: 6 trits packed in a u16 (values -364 to +364)
- **PackedTritVec**: Bitsliced vector storage with SIMD acceleration
- **SparseVec**: COO format for high-sparsity vectors
- **VSA Operations**: Vector Symbolic Architecture (bind, bundle, similarity)

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
trit-vsa = "0.1"
```

## Quick Start

```rust
use trit_vsa::{Trit, Tryte3, PackedTritVec, vsa};

// Basic trit operations
let a = Trit::P;  // +1
let b = Trit::N;  // -1
let product = a * b;  // -1
assert_eq!(product, Trit::N);

// Tryte arithmetic
let x = Tryte3::from_value(5).unwrap();
let y = Tryte3::from_value(3).unwrap();
let (sum, carry) = x + y;
assert_eq!(sum.value() + carry.value() as i32 * 27, 8);

// High-dimensional vectors for VSA
let mut vec_a = PackedTritVec::new(1000);
let mut vec_b = PackedTritVec::new(1000);

// Set some values
for i in 0..500 {
    vec_a.set(i, Trit::P);
    vec_b.set(i, Trit::P);
}

// Compute similarity
let sim = vsa::cosine_similarity(&vec_a, &vec_b);
println!("Similarity: {:.3}", sim);

// Bind vectors (creates association)
let bound = vsa::bind(&vec_a, &vec_b);
let recovered = vsa::unbind(&bound, &vec_b);

// Bundle vectors (superposition)
let bundled = vsa::bundle(&vec_a, &vec_b);
```

## Features

### Core Types

| Type | Storage | Range | Use Case |
|------|---------|-------|----------|
| `Trit` | 2 bits | {-1, 0, +1} | Single digit |
| `Tryte3` | 1 byte | [-13, +13] | Small integers |
| `Word6` | 2 bytes | [-364, +364] | Medium integers |
| `PackedTritVec` | Bitsliced | Any length | Dense vectors |
| `SparseVec` | COO | Any length | Sparse vectors |

### SIMD Acceleration

Enable SIMD optimizations with the `simd` feature:

```toml
[dependencies]
trit-vsa = { version = "0.1", features = ["simd"] }
```

Supports:
- x86_64: AVX2
- aarch64: NEON

### Vector Symbolic Architecture (VSA)

Implements hyperdimensional computing operations:

- **Bind**: Creates associations between vectors
- **Bundle**: Superposition via majority voting
- **Similarity**: Cosine and Hamming metrics

```rust
use trit_vsa::{PackedTritVec, Trit, vsa};

// Create symbol vectors
let dog = PackedTritVec::random(10000);
let cat = PackedTritVec::random(10000);
let animal = PackedTritVec::random(10000);

// Create compound: "dog is an animal"
let dog_animal = vsa::bind(&dog, &animal);

// Query: what is dog?
let query_result = vsa::unbind(&dog_animal, &dog);
let sim = vsa::cosine_similarity(&query_result, &animal);
// sim should be high (close to 1.0)
```

## Performance

Benchmarks on typical hardware show:

| Operation | Dimension | Throughput |
|-----------|-----------|------------|
| Dot product | 10,000 | ~50 million trits/sec |
| Bind | 10,000 | ~40 million trits/sec |
| Bundle (3 vectors) | 10,000 | ~30 million trits/sec |

Run benchmarks:

```bash
cargo bench -p trit-vsa
```

## Documentation

Full API documentation: [docs.rs/trit-vsa](https://docs.rs/trit-vsa)

## References

- Kanerva, P. (2009). "Hyperdimensional Computing"
- Gayler, R. (2003). "Vector Symbolic Architectures"
- Rachkovskij, D. (2001). "Binding and Normalization of Binary Sparse Distributed Representations"

## License

MIT License - see [LICENSE](LICENSE) for details.
