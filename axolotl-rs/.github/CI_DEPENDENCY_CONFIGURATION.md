# Dynamic Sister Project Dependencies in CI

This document explains how to configure sister project dependencies (peft-rs, qlora-rs, unsloth-rs) for CI workflows.

## Overview

The CI workflows support dynamic configuration of sister project versions through multiple methods, allowing you to test your changes against specific branches, tags, or commits of the sister projects.

## Configuration Methods

### Method 1: PR Description (Recommended)

Add dependency specifications to your PR description using the following format:

```
peft-rs: feature-branch
qlora-rs: v1.2.0
unsloth-rs: abc123def456
```

**Example PR Description:**
```markdown
## Changes
- Implements new LoRA feature
- Updates QLoRA integration

## Sister Project Dependencies
peft-rs: feature/new-lora
qlora-rs: main
unsloth-rs: experimental-kernels

Testing against feature branches to validate compatibility.
```

The CI will automatically extract these specifications and use them when fetching dependencies.

### Method 2: Workflow Dispatch

For manual workflow runs, you can specify dependency versions through the GitHub Actions UI:

1. Navigate to **Actions** → **CI** workflow
2. Click **Run workflow**
3. Fill in the inputs:
   - `peft_rs_ref`: Branch, tag, or commit SHA for peft-rs
   - `qlora_rs_ref`: Branch, tag, or commit SHA for qlora-rs
   - `unsloth_rs_ref`: Branch, tag, or commit SHA for unsloth-rs

### Method 3: Environment Variables

For custom CI setups, set these environment variables:

```bash
export PEFT_RS_REF=feature-branch
export QLORA_RS_REF=v1.2.0
export UNSLOTH_RS_REF=main
```

## How It Works

### Workflow Steps

1. **Checkout**: Repository is checked out
2. **Extract Overrides**: PR description is parsed for dependency specifications
3. **Patch Dependencies**: `.github/scripts/patch-dependencies.sh` script modifies `Cargo.toml` to use specified refs
4. **Setup Rust**: Rust toolchain is installed
5. **Fetch Dependencies**: Dependencies are fetched with the patched configuration
6. **Build/Test**: Normal CI operations proceed

### Patch Script

The `.github/scripts/patch-dependencies.sh` script:
- Reads environment variables (`PEFT_RS_REF`, `QLORA_RS_REF`, `UNSLOTH_RS_REF`)
- Parses PR description for override markers
- Modifies `Cargo.toml` in-place to use specified git refs
- Validates the changes before proceeding

### Priority Order

1. PR description specifications (highest priority)
2. Workflow dispatch inputs
3. Environment variables
4. Default values (all default to `main`)

## Use Cases

### Testing Feature Branches

When developing features that span multiple repositories:

```markdown
## Testing Cross-Repo Feature

peft-rs: feature/new-adapter-type
qlora-rs: feature/new-adapter-type
```

### Testing Release Candidates

Before releasing, test against specific versions:

```markdown
## Pre-Release Testing

peft-rs: v1.0.0
qlora-rs: v1.0.0
unsloth-rs: v1.0.0
```

### Debugging CI Failures

Test against a known working commit:

```markdown
## CI Debug

peft-rs: abc123def456
qlora-rs: 789ghi012jkl
```

## Supported Git References

The system supports any valid Git reference:

- **Branches**: `main`, `dev`, `feature/name`
- **Tags**: `v1.0.0`, `v0.5.0-beta`
- **Commit SHAs**: `abc123def456` (full or short)

## Default Behavior

If no overrides are specified, the CI defaults to:
- `peft-rs: main`
- `qlora-rs: main`
- `unsloth-rs: main`

This ensures that PRs without special requirements automatically test against the latest stable versions.

## Verification

To verify which versions are being used, check the CI logs:

```
Patching Cargo.toml with sister project refs:
  peft-rs:    feature-branch
  qlora-rs:   main
  unsloth-rs: v1.0.0
```

## Troubleshooting

### Invalid Git Reference

If you specify an invalid branch/tag/commit, the `cargo fetch` step will fail with:

```
error: failed to get `peft-rs` as a dependency of package `axolotl-rs`
```

**Solution**: Verify the ref exists in the sister project repository.

### Parsing Issues

If the PR description parser doesn't detect your specifications:

1. Ensure the format is exact: `project-name: ref-name`
2. Use one specification per line
3. Avoid extra whitespace

### Permission Issues

The script needs execute permissions. If you encounter:

```
Permission denied: .github/scripts/patch-dependencies.sh
```

**Solution**: Ensure the file has execute permissions in the repository.

## Examples

### Example 1: Simple Feature Testing

**PR Description:**
```markdown
Adds support for DoRA adapters.

peft-rs: feature/dora-support
```

**Result**: Tests against the `feature/dora-support` branch of peft-rs, while qlora-rs and unsloth-rs use `main`.

### Example 2: Multi-Repo Feature

**PR Description:**
```markdown
Implements quantization-aware training.

peft-rs: feature/qat
qlora-rs: feature/qat
unsloth-rs: main
```

**Result**: Tests against feature branches in both peft-rs and qlora-rs.

### Example 3: Specific Commit Testing

**PR Description:**
```markdown
Rollback compatibility test.

peft-rs: a1b2c3d
qlora-rs: e4f5g6h
```

**Result**: Tests against specific commits for regression testing.

## Integration with CI Cache

The Rust cache system is aware of the patched `Cargo.toml`, ensuring:
- Dependencies are cached per unique combination of refs
- Changing dependency refs invalidates the appropriate cache entries
- Parallel jobs with different refs don't interfere

## Local Testing

To test the patching locally:

```bash
export PEFT_RS_REF=my-feature-branch
export QLORA_RS_REF=main
export UNSLOTH_RS_REF=main

bash .github/scripts/patch-dependencies.sh
cargo check
```

This allows you to verify the configuration before pushing to CI.
