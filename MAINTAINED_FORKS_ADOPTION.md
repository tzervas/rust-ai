# Maintained Forks Adoption Guide

This document provides guidance for adopting the maintained forks of unmaintained crates used in the Rust ML ecosystem.

## Overview

The following crates have been forked and are actively maintained:

| Original Crate | Maintained Fork | crates.io | Repository |
|----------------|-----------------|-----------|------------|
| `paste` | `qlora-paste` | [qlora-paste](https://crates.io/crates/qlora-paste) | [tzervas/qlora-paste](https://github.com/tzervas/qlora-paste) |
| `gemm` | `qlora-gemm` | [qlora-gemm](https://crates.io/crates/qlora-gemm) | [tzervas/qlora-gemm](https://github.com/tzervas/qlora-gemm) |
| `candle-*` | `qlora-candle` | N/A (git dep) | [tzervas/qlora-candle](https://github.com/tzervas/qlora-candle) |

## Why These Forks Exist

### paste -> qlora-paste

The `paste` crate has no active maintainer since 2023:
- No releases addressing compatibility issues
- Outstanding security/compatibility concerns
- Blocks adoption of newer Rust versions in downstream crates

`qlora-paste` v1.0.20 is a drop-in replacement with identical API.

### gemm -> qlora-gemm

The `gemm` crate depends on unmaintained `paste`:
- Transitive dependency on unmaintained crate
- Blocks candle and other ML libraries

`qlora-gemm` v0.20.0 uses `qlora-paste` and maintains API compatibility with `gemm` v0.19.x.

### candle-* -> qlora-candle

HuggingFace's candle depends on `gemm`:
- PR submitted upstream: https://github.com/huggingface/candle/pull/3335
- Until merged, use qlora-candle fork

## Adoption Scenarios

### Scenario 1: Direct paste Dependency

**Before:**
```toml
[dependencies]
paste = "1.0"
```

**After:**
```toml
[dependencies]
qlora-paste = "1.0"
```

**Code changes:**
```rust
// Before
use paste::paste;

// After
use qlora_paste::paste;
```

### Scenario 2: Direct gemm Dependency

**Before:**
```toml
[dependencies]
gemm = "0.19"
```

**After:**
```toml
[dependencies]
qlora-gemm = "0.20"
```

**Code changes:**
```rust
// Before
use gemm::{gemm, Parallelism};

// After
use qlora_gemm::{gemm, Parallelism};
```

### Scenario 3: candle Dependency (Preferred - Git Patch)

Use cargo's patch mechanism to redirect candle to the maintained fork:

```toml
[dependencies]
candle-core = "0.9.2"
candle-nn = "0.9.2"

[patch.crates-io]
candle-core = { git = "https://github.com/tzervas/qlora-candle.git", branch = "use-qlora-gemm" }
candle-nn = { git = "https://github.com/tzervas/qlora-candle.git", branch = "use-qlora-gemm" }
candle-transformers = { git = "https://github.com/tzervas/qlora-candle.git", branch = "use-qlora-gemm" }
```

No code changes required - patch is transparent.

### Scenario 4: Workspace with Multiple Crates

For workspaces, add patches at the workspace root:

```toml
# Cargo.toml (workspace root)
[workspace]
members = ["crate-a", "crate-b"]

[workspace.dependencies]
candle-core = { version = "0.9.2", default-features = false }
candle-nn = { version = "0.9.2", default-features = false }

[patch.crates-io]
# Maintained candle fork with qlora-gemm
candle-core = { git = "https://github.com/tzervas/qlora-candle.git", branch = "use-qlora-gemm" }
candle-nn = { git = "https://github.com/tzervas/qlora-candle.git", branch = "use-qlora-gemm" }
candle-transformers = { git = "https://github.com/tzervas/qlora-candle.git", branch = "use-qlora-gemm" }
```

Member crates use workspace dependencies normally:
```toml
# crate-a/Cargo.toml
[dependencies]
candle-core = { workspace = true }
```

### Scenario 5: Transitive Dependency on paste/gemm

If your crate doesn't directly depend on paste/gemm but a dependency does:

```toml
[patch.crates-io]
# Patch transitive dependencies
paste = { package = "qlora-paste", version = "1.0" }
gemm = { package = "qlora-gemm", version = "0.20" }
```

Note: This only works if the package names match. For renamed packages like qlora-*, you need to patch at the source (e.g., use qlora-candle which already uses qlora-gemm).

## Version Compatibility Matrix

| qlora-paste | Compatible with paste |
|-------------|----------------------|
| 1.0.20 | 1.0.x |

| qlora-gemm | Compatible with gemm |
|------------|---------------------|
| 0.20.0 | 0.19.x |

| qlora-candle | Compatible with candle |
|--------------|----------------------|
| branch: use-qlora-gemm | 0.9.2 |

## Migration Checklist

1. [ ] Identify all direct `paste` dependencies -> replace with `qlora-paste`
2. [ ] Identify all direct `gemm` dependencies -> replace with `qlora-gemm`
3. [ ] Identify all `candle-*` dependencies -> add patch to workspace
4. [ ] Update imports in source files
5. [ ] Run `cargo update` to refresh lockfile
6. [ ] Run `cargo check` to verify compilation
7. [ ] Run tests to verify functionality

## Upstream Status

### candle PR

- PR: https://github.com/huggingface/candle/pull/3335
- Status: Open, Mergeable
- Changes: `gemm` -> `qlora-gemm` in candle-core

Once merged, the patch will no longer be necessary for new candle versions.

## Maintenance Commitment

These forks are maintained by Tyler Zervas (tzervas):
- Email: tz-dev@vectorweight.com
- GitHub: https://github.com/tzervas

Maintenance includes:
- Security patches
- Rust version compatibility
- API compatibility with upstream
- Continued maintenance even if upstream adopts the changes

## rust-ai Workspace Alignment

The rust-ai workspace is fully aligned with these maintained forks:

| Crate | candle Usage | Status |
|-------|--------------|--------|
| peft-rs | workspace dep | Patched via workspace |
| qlora-rs | workspace dep | Patched via workspace |
| unsloth-rs | workspace dep | Patched via workspace |
| axolotl-rs | workspace dep | Patched via workspace |
| bitnet-quantize | workspace dep | Patched via workspace |
| trit-vsa | optional workspace dep | Patched via workspace |
| vsa-optim-rs | workspace dep | Patched via workspace |
| tritter-accel | workspace dep | Patched via workspace |
| rust-ai-core | direct 0.9 | Needs patch if used standalone |

## Troubleshooting

### "duplicate lang item" errors

Ensure only one version of paste/gemm is in the dependency tree:
```bash
cargo tree -i paste
cargo tree -i gemm
```

### Patch not applying

Verify patch section syntax and that the patch URL/path is correct:
```bash
cargo update
cargo tree -i candle-core
```

### Version conflicts

Ensure patch version is compatible with the version requested by dependencies.

## Questions?

Open an issue at:
- https://github.com/tzervas/qlora-paste/issues
- https://github.com/tzervas/qlora-gemm/issues
- https://github.com/tzervas/qlora-candle/issues
