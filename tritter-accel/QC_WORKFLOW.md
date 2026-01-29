# Quality Control Workflow

Automated QC procedures for tritter-accel development.

## QC Gates

Each phase must pass all gates before proceeding to the next.

### Gate 1: Compilation
```bash
cargo check -p tritter-accel
cargo check -p tritter-accel --features cuda  # when GPU features added
```
**Pass criteria**: Zero errors

### Gate 2: Linting
```bash
cargo clippy -p tritter-accel -- -D warnings
```
**Pass criteria**: Zero warnings

### Gate 3: Tests
```bash
cargo test -p tritter-accel --lib
cargo test -p tritter-accel --tests  # integration tests
```
**Pass criteria**: All tests pass

### Gate 4: Documentation
```bash
cargo doc -p tritter-accel --no-deps 2>&1 | grep -c "warning"
```
**Pass criteria**: Zero documentation warnings

### Gate 5: Benchmarks (Phase 3+)
```bash
cargo bench -p tritter-accel
```
**Pass criteria**: No performance regressions >10%

## Automated QC Script

```bash
#!/bin/bash
# Run from rust-ai root

set -e  # Exit on first failure

echo "=== QC Gate 1: Compilation ==="
cargo check -p tritter-accel

echo "=== QC Gate 2: Linting ==="
cargo clippy -p tritter-accel -- -D warnings

echo "=== QC Gate 3: Tests ==="
cargo test -p tritter-accel

echo "=== QC Gate 4: Documentation ==="
WARNINGS=$(cargo doc -p tritter-accel --no-deps 2>&1 | grep -c "warning" || true)
if [ "$WARNINGS" -gt 0 ]; then
    echo "Documentation has $WARNINGS warnings"
    exit 1
fi

echo "=== All QC Gates Passed ==="
```

## Phase-Specific QC

### Phase 1: Delegation Refactor
- [ ] All functions delegate to sister crates
- [ ] No inline implementations remain
- [ ] Python API signatures unchanged
- [ ] Existing tests pass

### Phase 2: GPU Acceleration
- [ ] CUDA feature compiles
- [ ] GPU operations callable from Python
- [ ] CPU fallback works when CUDA unavailable
- [ ] GPU tests pass (with --features cuda)

### Phase 3: Production Polish
- [ ] 100% public API documented
- [ ] All benchmarks implemented
- [ ] 5+ Python examples work
- [ ] CI/CD pipeline green

### Phase 4: Ecosystem Integration
- [ ] Integration with axolotl-rs training loop
- [ ] End-to-end training example works
- [ ] Distributed training example works

## Verification Commands

### Check delegation is complete
```bash
# Should return 0 matches (no inline implementations)
grep -r "let.*=.*match.*val.*as.*i8" tritter-accel/src/ | wc -l
```

### Check GPU bindings exposed
```bash
# Should find GPU function exports
grep -r "GpuBind\|GpuBundle\|GpuUnbind" tritter-accel/src/ | wc -l
```

### Verify Python API
```python
import tritter_accel
# All these should exist
assert hasattr(tritter_accel, 'quantize_weights_absmean')
assert hasattr(tritter_accel, 'pack_ternary_weights')
assert hasattr(tritter_accel, 'unpack_ternary_weights')
assert hasattr(tritter_accel, 'ternary_matmul')
assert hasattr(tritter_accel, 'compress_gradients_vsa')
assert hasattr(tritter_accel, 'decompress_gradients_vsa')
assert hasattr(tritter_accel, 'version')
```

## Rollback Procedure

If QC fails:
1. `git stash` or `git checkout .` to revert changes
2. Review failure logs
3. Fix issues in isolation
4. Re-run QC gates

## Continuous Integration

### GitHub Actions Workflow (planned)
```yaml
name: QC
on: [push, pull_request]
jobs:
  qc:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - run: cargo check -p tritter-accel
      - run: cargo clippy -p tritter-accel -- -D warnings
      - run: cargo test -p tritter-accel
      - run: cargo doc -p tritter-accel --no-deps
```
