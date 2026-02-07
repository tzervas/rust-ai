---
name: crate-publish
description: Guide the process of publishing a rust-ai workspace crate to crates.io. Covers version bumping, changelog updates, pre-publish checks, and the publish command sequence. Use when preparing a crate release.
metadata:
  author: tzervas
  version: "1.0"
allowed-tools: Bash(cargo:*) Bash(git:*) Read Edit Write Glob Grep
---

# Crate Publishing Guide

## When to use
- Publishing a new version of any workspace crate to crates.io
- Bumping versions after breaking changes
- Coordinating multi-crate releases

## Release order (respect dependencies)
1. `peft-rs` (foundation, no internal deps)
2. `qlora-rs` (depends on peft-rs)
3. `trit-vsa` (ternary foundation)
4. `bitnet-quantize` (depends on trit-vsa)
5. `vsa-optim-rs` (depends on trit-vsa)
6. `tritter-accel` (depends on all above)
7. `hybrid-predict-trainer-rs` (standalone)
8. `training-tools` (standalone)
9. `tritter-model-rs` (depends on trit-vsa, bitnet-quantize)

## Pre-publish checklist
For each crate:

1. **Version bump** in Cargo.toml (follow semver)
2. **Update CHANGELOG.md** with release notes
3. **Update dependent Cargo.tomls** if version bounds changed
4. **Run checks**:
   ```bash
   cargo check -p <crate>
   cargo clippy -p <crate> -- -D warnings
   cargo test -p <crate>
   cargo fmt -p <crate> -- --check
   cargo doc -p <crate> --no-deps
   ```
5. **Dry-run publish**:
   ```bash
   cargo publish -p <crate> --dry-run
   ```
6. **Commit and tag**:
   ```bash
   git commit -S -m "chore(<crate>): release v<version>"
   git tag -s <crate>-v<version> -m "<crate> v<version>"
   ```

## Publish
```bash
cargo publish -p <crate>
git push origin main --tags
```

## Post-publish
- Verify on crates.io
- Update workspace Cargo.toml patch section if needed
- Announce in relevant channels
