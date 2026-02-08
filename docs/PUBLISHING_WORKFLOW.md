# Publishing Workflow (Option C Strategy)

## Overview

This workspace uses **Option C: Workspace Development + Pre-Publish Script**:

- **Local development**: Uses `{ workspace = true }` for DRY dependency management
- **Publishing**: Script converts to explicit versions for standalone compatibility

This gives us the best of both worlds:
1. Clean, maintainable workspace with centralized version management
2. Publishable crates that work standalone without workspace

## Workflow

### Local Development

Work normally with `{ workspace = true }` in crate Cargo.toml files:

```toml
[dependencies]
candle-core = { workspace = true }
serde = { workspace = true, features = ["derive"] }
thiserror = { workspace = true }
```

The workspace root defines versions:

```toml
# Workspace Cargo.toml
[workspace.dependencies]
candle-core = "0.9.2"
serde = { version = "1.0", features = ["derive"] }
thiserror = "2.0"
```

### Pre-Publishing Preparation

Before publishing a crate to crates.io:

```bash
# 1. Prepare publish-ready Cargo.toml
./scripts/prepare-publish.py <crate-name> --dry-run

# Example: peft-rs
./scripts/prepare-publish.py peft-rs --dry-run

# This creates <crate>/Cargo.toml.publish with explicit versions and stripped paths
```

The script will:
1. Extract workspace dependency versions
2. Replace `{ workspace = true }` with explicit versions
3. Strip `path = "../..."` from sister crate deps (crates.io requirement)
4. Preserve features and other configuration
5. Optionally run `cargo publish --dry-run` with `--validate` flag

### Publishing

Once the dry-run succeeds:

```bash
cd <crate-name>

# Swap to publish-ready Cargo.toml
mv Cargo.toml.publish Cargo.toml

# Final check
cargo publish --dry-run

# Publish to crates.io
cargo publish

# Restore workspace version (IMPORTANT!)
git checkout Cargo.toml
```

### Example: Publishing peft-rs

```bash
# 1. Prepare and validate
./scripts/prepare-publish.py peft-rs --validate

# 2. Review changes (if validation passed)
diff peft-rs/Cargo.toml peft-rs/Cargo.toml.publish

# 3. Publish
cd peft-rs
mv Cargo.toml.publish Cargo.toml
cargo publish
git checkout Cargo.toml
cd ..

# 4. Tag release
git tag peft-rs-v1.0.1
git push origin peft-rs-v1.0.1
```

## Dependency Publishing Order

Publish in dependency order (bottom-up):

### Round 1 (Foundation - parallel)
- `peft-rs` (no sister crate deps)
- `trit-vsa` (no sister crate deps)

### Round 2 (First-level - parallel after Round 1)
- `qlora-rs` (depends on peft-rs)
- `bitnet-quantize` (depends on trit-vsa, peft-rs optional)
- `vsa-optim-rs` (depends on trit-vsa)
- `unsloth-rs` (depends on trit-vsa)

### Round 3 (Second-level - parallel after Round 2)
- `tritter-accel` (depends on trit-vsa, bitnet-quantize, vsa-optim-rs)
- `hybrid-predict-trainer-rs` (standalone, optional deps)

### Round 4 (Orchestration - sequential after Round 3)
- `rust-ai-core` (depends on ALL above)
- `tritter-model-rs` (depends on many above)
- `training-tools` (depends on tritter-model-rs, hybrid-predict)

### Round 5 (Top-level - after Round 4)
- `axolotl-rs` (depends on peft-rs, qlora-rs, unsloth-rs, vsa-optim-rs)

**Important**: Wait 1-2 minutes between rounds for crates.io index propagation.

## Sister Crate Dependencies

Crates that depend on other workspace crates use dual path+version specification:

```toml
# In vsa-optim-rs/Cargo.toml
[dependencies]
trit-vsa = { version = "0.3", path = "../trit-vsa" }
```

The `prepare-publish.py` script strips the path and preserves version-only:

```toml
# After prepare-publish.sh
[dependencies]
trit-vsa = "0.3"  # path removed for publishing
```

The workspace `[patch.crates-io]` section ensures local development uses the path version:

```toml
# Workspace Cargo.toml
[patch.crates-io]
trit-vsa = { path = "trit-vsa" }
```

## Version Management

### Updating a Workspace Dependency

```bash
# 1. Update workspace root Cargo.toml
vim Cargo.toml  # Change candle-core = "0.9.2" to "0.9.3"

# 2. Test workspace builds
cargo check --workspace

# 3. Run tests
cargo test --workspace

# 4. Commit
git commit -am "chore: update candle-core to 0.9.3"
```

Individual crates automatically pick up the new version via `{ workspace = true }`.

### Bumping a Crate Version

Before publishing a new crate version:

```bash
# 1. Update crate version
vim peft-rs/Cargo.toml  # version = "1.0.1" → "1.0.2"

# 2. Update CHANGELOG
vim peft-rs/CHANGELOG.md

# 3. Prepare and publish
./scripts/prepare-publish.sh peft-rs --dry-run
cd peft-rs && mv Cargo.toml.publish Cargo.toml
cargo publish
git checkout Cargo.toml

# 4. Commit and tag
git commit -am "chore(peft-rs): release v1.0.2"
git tag peft-rs-v1.0.2
git push origin peft-rs-v1.0.2
```

## Testing

### Before Publishing

Run comprehensive checks:

```bash
# In crate directory
cargo test --all-features
cargo clippy --all-features
cargo doc --no-deps
cargo publish --dry-run
```

### After Publishing

Verify the published crate works standalone:

```bash
# Create test project in /tmp
cd /tmp
cargo new test-peft --lib
cd test-peft

# Add dependency
cargo add peft-rs

# Test it works
cargo build
```

## Troubleshooting

### "workspace version not found" Warning

If `prepare-publish.sh` shows a warning:

```
⚠ dep-name: workspace version not found, keeping as-is
```

This means the dependency uses `{ workspace = true }` but isn't defined in `[workspace.dependencies]`. Add it to the workspace root:

```toml
# Workspace Cargo.toml
[workspace.dependencies]
dep-name = "x.y.z"
```

### Sister Crate Version Mismatches

If `rust-ai-core` declares `trit-vsa = "0.1"` but the actual version is `0.3.0`, publishing will fail. Update the version bound:

```toml
# In rust-ai-core/Cargo.toml
[dependencies]
trit-vsa = { version = "0.3", path = "../trit-vsa" }  # Was "0.1"
```

### Circular Dependencies

If you see errors about circular dependencies, check the publishing order. You cannot publish a crate until its dependencies are on crates.io.

## Best Practices

1. **Always test workspace builds** after adding `{ workspace = true }`
2. **Run dry-run** before every publish: `./scripts/prepare-publish.sh <crate> --dry-run`
3. **Restore after publishing**: `git checkout Cargo.toml` immediately after `cargo publish`
4. **Version bounds**: Use exact versions for sister crates (`"0.3"`), compatible versions for external deps (`"1.0"`)
5. **Commit atomically**: Workspace dep updates in one commit, crate version bumps in separate commits

## See Also

- [Workspace Cargo.toml](../Cargo.toml) - Central version definitions
- [Local CI Script](../scripts/local-ci.sh) - Pre-publish validation
- [Opus Planner Report](../docs/archive/sessions/SESSION_2026-02-07_DEPENDENCY_UNIFICATION.md) - Full unification strategy
