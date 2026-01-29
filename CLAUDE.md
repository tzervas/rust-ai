# Rust AI Training Stack - Workspace

This workspace contains Rust ports of essential Python AI/ML libraries, filling gaps in the Rust crate ecosystem for LLM fine-tuning and training.

## Subagent Context Management

When spawning subagents via the Task tool:

- **Autocompact at 98%**: Subagents must compact their context when usage reaches 98%
- **Focused prompts**: Provide clear, scoped tasks to minimize context bloat
- **Result summarization**: Subagents should return concise summaries, not full file contents
- **Parallel execution**: Launch independent subagents in parallel to maximize throughput

### Subagent Prompt Template

When spawning exploration or implementation agents, include:
```
Context management: Compact your context automatically at 98% usage.
Focus: [specific task scope]
Output: Return a concise summary of findings/changes, not raw file contents.
```

## Workspace Architecture

```
rust-ai/
├── peft-rs/           # Foundation: PEFT adapters (LoRA, DoRA, AdaLoRA, etc.)
├── qlora-rs/          # 4-bit quantization + QLoRA (depends on peft-rs)
├── unsloth-rs/        # GPU-optimized transformer kernels (CubeCL)
├── axolotl-rs/        # High-level fine-tuning orchestration (uses all above)
├── paste-fork/        # Utility: proc-macro for token pasting
│
│   # Tritter Acceleration Stack
├── trit-vsa/          # Balanced ternary arithmetic + VSA operations
├── bitnet-quantize/   # BitNet b1.58 weight/activation quantization
├── vsa-optim-rs/      # VSA gradient compression + prediction
├── tritter-accel/     # Python bindings for acceleration (PyO3)
└── rust-ai-core/      # Shared GPU dispatch traits (GpuDispatchable)
```

## Dependency Hierarchy

```
peft-rs (v0.4.1)          ← Foundation, no internal deps
    ↓
qlora-rs (v0.3.0)         ← Requires peft-rs for adapter management

unsloth-rs (v0.1.0-alpha) ← GPU kernels, depends on trit-vsa for ternary types

axolotl-rs (v0.1.0)       ← Orchestrates all via optional features
    └── optional: peft-rs, qlora-rs, unsloth-rs, tritter-accel

# Tritter Acceleration Stack
trit-vsa (v0.2.0)         ← Canonical ternary foundation (PackedTritVec, VSA ops)
    ↓
bitnet-quantize (v0.2.0)  ← BitNet quantization, uses trit-vsa
    ↓
vsa-optim-rs (v0.1.0)     ← Gradient compression, uses trit-vsa
    ↓
tritter-accel (v0.1.1)    ← Python bindings, delegates to all above
```

## Cross-Repo Development Rules

### Breaking Change Protocol
When modifying shared interfaces:

1. **peft-rs changes**: Check qlora-rs and axolotl-rs for compatibility
2. **qlora-rs changes**: Check axolotl-rs for compatibility
3. **Trait modifications**: Run `cargo check --workspace` before committing

### Version Coordination
- All crates target Rust 1.92+
- All use candle 0.9.x for tensor operations
- Workspace uses `[patch]` sections for local development

### Testing Strategy
```bash
# Full workspace check
cargo check --workspace

# Test individual crate
cargo test -p peft-rs
cargo test -p qlora-rs
cargo test -p unsloth-rs
cargo test -p axolotl-rs

# Test with features
cargo test -p axolotl-rs --features "peft,qlora,unsloth"

# GPU tests (requires CUDA)
cargo test -p unsloth-rs --features cuda -- --ignored
cargo test -p axolotl-rs --features cuda -- --ignored
```

## Current Status & Goals

| Crate | Version | Status | 1.0 Readiness |
|-------|---------|--------|---------------|
| peft-rs | 0.4.1 | Published on crates.io | ~80% |
| qlora-rs | 0.3.0 | Published on crates.io | ~60% |
| unsloth-rs | 0.1.0-alpha | Early development | ~30% |
| axolotl-rs | 0.1.0 | Framework scaffold | ~20% |
| trit-vsa | 0.2.0 | GPU ops implemented | ~70% |
| bitnet-quantize | 0.2.0 | Core quantization done | ~65% |
| vsa-optim-rs | 0.1.0 | Deterministic trainer done | ~60% |
| tritter-accel | 0.1.1 | Inline impl, needs refactor | ~30% |

### 1.0 Requirements (All Crates)
- [ ] Zero `cargo clippy` warnings
- [ ] 100% public API documentation
- [ ] Comprehensive test coverage (unit + integration)
- [ ] Benchmarks for performance-critical paths
- [ ] CHANGELOG.md with semantic versioning
- [ ] Examples directory with runnable code
- [ ] CI/CD with GitHub Actions

## Build Commands

```bash
# Development build
cargo build --workspace

# Release build
cargo build --workspace --release

# With CUDA support
cargo build --workspace --features cuda --release

# Documentation
cargo doc --workspace --no-deps --open
```

## Common Patterns

### Error Handling
All crates use `thiserror` for error types. Pattern:
```rust
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("description: {0}")]
    Variant(String),
}
```

### Tensor Operations
Use candle for all tensor ops:
```rust
use candle_core::{Device, Tensor, DType};
use candle_nn::VarBuilder;
```

### Configuration
Use serde for serialization:
```rust
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Config { ... }
```

## Known Issues

1. **axolotl-rs build failure**: Missing `PeftLoraConfig` import in model.rs
2. **unsloth-rs warnings**: Unused feature cfg warnings for ternary_cubecl_todo
3. **Workspace Cargo.toml**: Has outdated candle 0.3 in workspace.dependencies (crates use 0.9)
4. **tritter-accel inline code**: Uses inline implementations instead of delegating to sister crates (see SPEC.md)
5. **unsloth-rs/trit-vsa duplication**: Both have ternary implementations; unsloth-rs should depend on trit-vsa

## File Organization Convention

Each crate follows:
```
crate-name/
├── Cargo.toml
├── README.md
├── CLAUDE.md          # Crate-specific Claude Code instructions
├── src/
│   ├── lib.rs         # Public API exports
│   ├── error.rs       # Error types
│   └── ...            # Implementation modules
├── tests/             # Integration tests
├── benches/           # Criterion benchmarks
└── examples/          # Usage examples
```

## When Working Across Repos

1. **Always check workspace compilation first**: `cargo check --workspace`
2. **Run affected tests**: If changing peft-rs, also run qlora-rs tests
3. **Update version bounds**: When bumping versions, update dependent Cargo.tomls
4. **Coordinate releases**: peft-rs → qlora-rs → axolotl-rs order

## Performance Targets

- peft-rs: LoRA forward pass < 1ms overhead on GPU
- qlora-rs: NF4 quantization at > 100MB/s
- unsloth-rs: FlashAttention-comparable throughput
- axolotl-rs: Config parsing < 10ms, training step overhead < 5%

### Tritter Acceleration Targets

- trit-vsa: GPU bind/bundle 10x faster than CPU at 10K+ dimensions
- bitnet-quantize: AbsMean quantization at > 500MB/s
- vsa-optim-rs: 10-100x gradient compression with <5% accuracy loss
- tritter-accel:
  - Ternary matmul 4x faster than FP32 on GPU
  - 16x memory reduction via 2-bit packing
  - Zero-copy Python ↔ Rust transfer
