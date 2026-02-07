---
name: rust-workspace-check
description: Run comprehensive workspace-wide quality checks including compilation, linting, formatting, and tests across all crates in the rust-ai workspace. Use when verifying overall workspace health or before merging branches.
metadata:
  author: tzervas
  version: "1.0"
allowed-tools: Bash(cargo:*) Bash(git:*) Read Glob Grep
---

# Rust Workspace Check

## When to use
- Before merging a branch into staging or main
- After making cross-crate changes
- When asked to verify workspace health
- As part of CI/CD validation

## Steps

### 1. Workspace compilation
```bash
cargo check --workspace
```

### 2. Per-crate clippy
Run clippy on each active workspace member:
```bash
for crate in peft-rs qlora-rs hybrid-predict-trainer-rs training-tools trit-vsa bitnet-quantize vsa-optim-rs tritter-accel tritter-model-rs; do
  echo "=== $crate ==="
  cargo clippy -p "$crate" -- -W clippy::all 2>&1 | tail -5
done
```

### 3. Test suite
```bash
cargo test --workspace
```

### 4. Formatting
```bash
cargo fmt --all -- --check
```

### 5. Report
Summarize results per crate in a table:
| Crate | Check | Clippy | Tests | Fmt |
|-------|-------|--------|-------|-----|

Mark any failures clearly.
