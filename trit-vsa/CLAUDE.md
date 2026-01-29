# trit-vsa Development Guide

Balanced ternary arithmetic library for the rust-ai workspace.

## Architecture Overview

```
trit-vsa/
├── src/
│   ├── lib.rs          # Public API, prelude, crate docs
│   ├── error.rs        # TernaryError enum
│   ├── trit.rs         # Core Trit {N, Z, P} type
│   ├── tryte.rs        # Tryte3 (3 trits in u8)
│   ├── word.rs         # Word6 (6 trits in u16)
│   ├── arithmetic.rs   # Conversion utilities
│   ├── packed.rs       # PackedTritVec (bitsliced)
│   ├── sparse.rs       # SparseVec (COO format)
│   ├── simd/           # SIMD acceleration
│   │   ├── mod.rs      # Feature detection
│   │   ├── avx2.rs     # x86_64 AVX2
│   │   └── neon.rs     # ARM NEON
│   └── vsa/            # Vector Symbolic Architecture
│       ├── mod.rs      # Re-exports
│       ├── bind.rs     # Association operations
│       ├── bundle.rs   # Superposition operations
│       └── similarity.rs # Similarity metrics
├── benches/
│   └── ternary_ops.rs  # Criterion benchmarks
├── examples/           # Usage examples
└── tests/              # Integration tests
```

## Key Design Decisions

### Bitsliced Storage (PackedTritVec)

Trits are stored in two parallel bit planes:
- `plus`: 1 where trit is +1
- `minus`: 1 where trit is -1
- Both 0: trit is 0

This enables efficient SIMD operations via popcount.

### Bind/Unbind Semantics

- `bind(a, b)` = (a - b) mod 3 (subtraction)
- `unbind(bound, key)` = (bound + key) mod 3 (addition)
- Relationship: `unbind(bind(a, b), b) == a`

Note: bind is NOT self-inverse. Always use unbind for recovery.

### Trit Encoding

```
Trit::N (-1) → bits: 0b00
Trit::Z (0)  → bits: 0b01
Trit::P (+1) → bits: 0b10
```

## Development Commands

```bash
# Run all tests
cargo test -p trit-vsa

# Run with verbose output
cargo test -p trit-vsa -- --nocapture

# Check for warnings
cargo clippy -p trit-vsa -- -W clippy::pedantic

# Run benchmarks
cargo bench -p trit-vsa

# Generate docs
cargo doc -p trit-vsa --no-deps --open

# Check SIMD feature
cargo test -p trit-vsa --features simd
```

## Adding New Features

### New Trit Operation

1. Add method to `Trit` in `src/trit.rs`
2. Add corresponding method to `PackedTritVec` in `src/packed.rs`
3. Add corresponding method to `SparseVec` in `src/sparse.rs`
4. Add tests for all three implementations
5. Verify consistency between implementations

### New VSA Operation

1. Create function in appropriate `src/vsa/*.rs` file
2. Add to `src/vsa/mod.rs` re-exports
3. Add to `src/lib.rs` public API
4. Include academic reference if applicable

## Performance Guidelines

- Use `#[inline]` for small, hot functions
- Prefer bitwise operations over arithmetic
- Use `count_ones()` for popcount (LLVM optimizes to hardware instruction)
- Test SIMD paths on both x86_64 and aarch64

## Compatibility

- Minimum Rust: 1.92
- No-std compatible with `alloc`
- Workspace dependencies: none (foundation crate)
- Used by: bitnet-quantize, unsloth-rs (optional)

## Common Pitfalls

1. **Balanced ternary conversion**: Remember to handle negative remainders
2. **Bind vs XOR**: Ternary bind is NOT equivalent to XOR
3. **Sparsity assumption**: SparseVec is only efficient when >90% zeros
4. **SIMD availability**: Always provide scalar fallback
