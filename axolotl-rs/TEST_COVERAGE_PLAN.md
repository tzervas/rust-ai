# Test Coverage Implementation Plan for axolotl-rs

**Project:** axolotl-rs - Rust port of Axolotl LLM fine-tuning toolkit  
**Current Branch:** dev  
**Date:** January 9, 2026  
**GPU:** NVIDIA RTX 5080 (16GB, CUDA 12.0, MPS enabled)  
**Rust:** 1.92.0

## Current Status

- **Commit:** f415336 "feat: initial axolotl-rs scaffold" (Jan 6, 2026)
- **Test Coverage:** ~15-20% (5 tests across 3 modules)
- **Issue:** Missing workspace configuration, project doesn't compile
- **Strategy:** Mock dependencies with feature flags, keep tests in src/ for dev branch

## Execution Strategy

Each task will be implemented in a separate feature branch with conventional commits:
- Branch naming: `test/<module-name>` or `fix/<issue>`
- Commits: `test:`, `feat:`, `fix:`, `chore:`, `docs:`
- PR to dev branch for review before merge

## Task Breakdown

### Task 1: Fix Workspace Configuration
**Branch:** `fix/workspace-setup`  
**Priority:** CRITICAL (blocks all testing)  
**Conventional Commit:** `fix: add workspace manifest and mock dependencies`

**Subtasks:**
1. Create parent `Cargo.toml` workspace manifest at repo root (if not mono-repo, modify existing)
2. Add workspace.package section with common metadata
3. Add workspace.dependencies for shared deps (criterion, tempfile)
4. Create mock crates with feature flags:
   - `src/mocks/peft.rs` - Mock PEFT (Parameter-Efficient Fine-Tuning) adapter types
   - `src/mocks/qlora.rs` - Mock QLoRA quantization stubs
   - `src/mocks/unsloth.rs` - Mock Unsloth optimization stubs
5. Update `Cargo.toml` to use feature flags:
   ```toml
   [features]
   default = ["mock-peft", "mock-qlora"]
   mock-peft = []
   mock-qlora = []
   mock-unsloth = []
   real-peft = []  # For future when crates.io versions available
   real-qlora = []
   real-unsloth = []
   ```
6. Add conditional compilation in code: `#[cfg(feature = "mock-peft")]`
7. Verify: `cargo test --no-run` succeeds

**Expected Files:**
- `Cargo.toml` (workspace manifest or modified)
- `src/mocks/mod.rs`
- `src/mocks/peft.rs`
- `src/mocks/qlora.rs`
- `src/mocks/unsloth.rs`

**Validation:**
```bash
cargo check
cargo test --no-run
cargo test
```

---

### Task 2: Expand config.rs Test Coverage
**Branch:** `test/config-validation`  
**Priority:** HIGH  
**Conventional Commit:** `test: expand config.rs test coverage`

**Current:** 3 tests (serialization, validation, presets)  
**Target:** 15+ tests covering all validation paths

**Subtasks:**
1. Test `LoraSettings` validation:
   - Valid r and alpha values
   - Invalid r (0, negative, too large)
   - Default target_modules
   - Empty/invalid target_modules
2. Test `QuantizationSettings`:
   - All quantization types (4bit, 8bit, none)
   - Compute dtype options
   - Invalid combinations
3. Test `DatasetConfig`:
   - All format types (alpaca, sharegpt, completion, custom)
   - Path validation (missing file)
   - Train/val split ratios (edge cases: 0.0, 1.0, >1.0)
4. Test `TrainingConfig`:
   - Batch size validation (0, negative)
   - Learning rate validation
   - Num_epochs validation
   - Checkpoint_dir creation
5. Test `load_config`:
   - Valid YAML loading
   - Malformed YAML
   - Missing required fields
   - Invalid file path
   - Empty file
6. Test `save_config`:
   - Successful save
   - Invalid path (no permissions)
   - Roundtrip (load -> save -> load)
7. Test preset generation for all models:
   - Existing: llama2-7b
   - Add: mistral-7b
   - Add: phi3-mini
   - Verify all preset fields populated correctly
8. Test error messages are descriptive

**Expected LOC:** ~200-300 test code

**Validation:**
```bash
cargo test config::tests
```

---

### Task 3: Complete dataset.rs Test Suite
**Branch:** `test/dataset-loaders`  
**Priority:** HIGH  
**Conventional Commit:** `test: complete dataset.rs test coverage`

**Current:** 1 test (Alpaca format)  
**Target:** 12+ tests covering all data loaders

**Subtasks:**
1. Test `load_alpaca_dataset`:
   - Valid JSON array (existing)
   - Empty array
   - Malformed JSON
   - Missing required fields (instruction, output)
   - Large dataset (performance)
2. Test `load_sharegpt_dataset`:
   - Valid ShareGPT format with conversations
   - Multi-turn conversations
   - System/user/assistant roles
   - Missing role field
   - Empty conversations
3. Test `load_completion_dataset`:
   - Valid prompt/completion pairs
   - Missing fields
   - Empty strings
4. Test `load_custom_dataset`:
   - Valid custom format with template
   - Template variable substitution
   - Missing template variables
5. Test `load_dataset` dispatcher:
   - Correct format routing
   - Unknown format error
6. Test `Dataset` struct:
   - Length/is_empty
   - Split functionality (train/val)
   - Edge case: split ratio 0.0, 1.0
   - Empty dataset split
7. Test file I/O errors:
   - Non-existent file
   - Invalid permissions
   - Non-UTF8 content
8. Add property-based tests (optional, if time permits):
   - Use `proptest` for dataset validation properties

**Test Data:** Create small JSON fixtures in test code (use string literals or tempfile)

**Expected LOC:** ~300-400 test code

**Validation:**
```bash
cargo test dataset::tests
```

---

### Task 4: Add error.rs Test Coverage
**Branch:** `test/error-handling`  
**Priority:** MEDIUM  
**Conventional Commit:** `test: add error.rs test coverage`

**Current:** 0 tests  
**Target:** 8+ tests for error types and conversions

**Subtasks:**
1. Test `AxolotlError::Config` variant:
   - Error creation with message
   - Display formatting
   - Debug formatting
2. Test `AxolotlError::Dataset` variant:
   - Error creation
   - Display includes context
3. Test `AxolotlError::Model` variant
4. Test `AxolotlError::Training` variant
5. Test `AxolotlError::Io` conversion:
   - From std::io::Error
   - Preserves error message
6. Test `AxolotlError::Yaml` conversion:
   - From serde_yaml::Error
7. Test error chain/source:
   - Verify error source is preserved
8. Test error downcast (if using anyhow features)

**Expected LOC:** ~100-150 test code

**Validation:**
```bash
cargo test error::tests
```

---

### Task 5: Add trainer.rs Test Coverage
**Branch:** `test/trainer-module`  
**Priority:** MEDIUM  
**Conventional Commit:** `test: expand trainer.rs test coverage`

**Current:** 1 test (basic instantiation)  
**Target:** 10+ tests for trainer lifecycle

**Subtasks:**
1. Test `Trainer::new`:
   - Valid construction (existing)
   - Validate config is stored
   - Validate dataset is stored
2. Test checkpoint directory creation:
   - Directory is created if missing
   - Existing directory is reused
   - Permission errors handled
3. Test `train` method (mocked):
   - Progress bar updates
   - Logging at correct steps
   - Epoch counting
   - Batch iteration
4. Test `save_checkpoint`:
   - Checkpoint files created
   - Metadata saved (epoch, step)
   - Naming convention correct
5. Test `load_checkpoint`:
   - Valid checkpoint loading
   - Missing checkpoint handling
   - Corrupted checkpoint error
6. Test resume training:
   - Continues from correct epoch/step
   - State is restored
7. Test evaluation loop (if implemented):
   - Validation data processing
   - Metrics calculation
8. Test early stopping (if implemented)
9. Test with mock model:
   - Use small mock model for integration-style test
   - Verify forward pass called

**Note:** Since model loading is TODO, focus on structure/lifecycle tests. Mock model operations.

**Expected LOC:** ~200-250 test code

**Validation:**
```bash
cargo test trainer::tests
```

---

### Task 6: Add model.rs Test Stubs
**Branch:** `test/model-stubs`  
**Priority:** LOW (module is TODO)  
**Conventional Commit:** `test: add model.rs test stubs for future implementation`

**Current:** 0 tests (module is all TODOs)  
**Target:** Test structure ready for future implementation

**Subtasks:**
1. Create test module structure:
   ```rust
   #[cfg(test)]
   mod tests {
       use super::*;
       
       #[test]
       #[ignore]  // Ignored until load_model is implemented
       fn test_load_model_llama2() {
           todo!("Test loading LLaMA 2 model from HuggingFace")
       }
       
       // More ignored tests...
   }
   ```
2. Add ignored tests for:
   - `load_model` for each model type (llama2, mistral, phi3)
   - `apply_adapter` with LoRA
   - `apply_adapter` with QLoRA
   - `merge_adapter`
   - `download_model` (if download feature enabled)
3. Add doc comments explaining what each test should verify
4. Add TODO comments linking to implementation issues

**Expected LOC:** ~50-100 test stubs

**Validation:**
```bash
cargo test model::tests -- --ignored  # Should all pass (as TODOs)
```

---

### Task 7: Create CLI Integration Tests Skeleton
**Branch:** `test/cli-integration`  
**Priority:** MEDIUM  
**Conventional Commit:** `test: add CLI integration test framework`

**Current:** 0 CLI tests  
**Target:** Integration test structure in `tests/` directory (skeleton for dev, full tests for testing branch)

**Subtasks:**
1. Create `tests/cli_tests.rs`
2. Add helper functions:
   - `run_cli_command(args: &[&str]) -> Result<Output>`
   - `create_test_config(path: &Path) -> Result<()>`
   - `cleanup_test_dir(path: &Path)`
3. Test `validate` command:
   - Valid config file
   - Invalid config file
   - Missing config file
   - Exit codes correct
4. Test `train` command:
   - Mock training run
   - Config validation before train
   - Checkpoint creation
5. Test `merge` command (stub for now):
   - Argument parsing
   - Error on missing adapter
6. Test `init` command:
   - Creates config file
   - Template selection
   - Overwrite behavior
7. Use `assert_cmd` crate for CLI testing:
   - Add to dev-dependencies: `assert_cmd = "2.0"`
   - Add `predicates = "3.0"` for assertions

**Expected LOC:** ~200-300 test code

**Expected Files:**
- `tests/cli_tests.rs`
- `tests/fixtures/` (test config files)

**Validation:**
```bash
cargo test --test cli_tests
```

---

### Task 8: Add Doc Tests to Public APIs
**Branch:** `docs/api-doctests`  
**Priority:** MEDIUM  
**Conventional Commit:** `docs: add documentation tests to public APIs`

**Subtasks:**
1. Add doc examples to `src/lib.rs`:
   - Module overview with usage example
   - Quick start example
2. Add doc tests to `Config` struct:
   - Loading config from YAML
   - Creating config programmatically
   - Saving config
3. Add doc tests to `Dataset`:
   - Loading dataset
   - Accessing dataset items
4. Add doc tests to `Trainer`:
   - Creating trainer
   - Running training (mocked example)
5. Add doc tests to error types:
   - Error handling patterns
6. Verify all doc tests pass:
   ```bash
   cargo test --doc
   ```

**Expected LOC:** ~150-200 lines of doc comments with tests

**Validation:**
```bash
cargo test --doc
cargo doc --no-deps --open  # Verify examples render correctly
```

---

### Task 9: Complete Config Parsing Benchmarks
**Branch:** `bench/config-performance`  
**Priority:** LOW  
**Conventional Commit:** `bench: implement config parsing performance benchmarks`

**Current:** Empty stub in `benches/config_parsing.rs`  
**Target:** Benchmarks for critical paths

**Subtasks:**
1. Benchmark `Config::from_yaml`:
   - Small config (< 1KB)
   - Large config (with many dataset paths)
2. Benchmark `Config::validate`:
   - Valid config
   - Invalid config (error path)
3. Benchmark preset generation:
   - Each model preset
4. Benchmark serialization/deserialization roundtrip
5. Compare YAML vs JSON (if supporting both)
6. Add baseline markers for regression detection

**Expected LOC:** ~100-150 benchmark code

**Validation:**
```bash
cargo bench --bench config_parsing
```

---

### Task 10: Setup CI/CD Pipeline with GPU Tests
**Branch:** `ci/github-actions`  
**Priority:** HIGH  
**Conventional Commit:** `ci: add GitHub Actions workflow with test coverage`

**Subtasks:**
1. Create `.github/workflows/ci.yml`:
   ```yaml
   name: CI
   on: [push, pull_request]
   jobs:
     test:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v4
         - uses: dtolnay/rust-toolchain@stable
         - uses: Swatinem/rust-cache@v2
         - run: cargo test --all-features
         - run: cargo clippy -- -D warnings
         - run: cargo fmt -- --check
     
     coverage:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v4
         - uses: dtolnay/rust-toolchain@stable
           with:
             components: llvm-tools-preview
         - uses: taiki-e/install-action@cargo-llvm-cov
         - run: cargo llvm-cov --all-features --lcov --output-path lcov.info
         - uses: codecov/codecov-action@v3
           with:
             files: lcov.info
   ```
2. Add separate GPU test job (runs on self-hosted runner):
   ```yaml
     gpu-test:
       runs-on: [self-hosted, linux, gpu]
       steps:
         - uses: actions/checkout@v4
         - run: cargo test --features cuda --test gpu_integration
   ```
3. Add pre-commit hooks configuration (`.pre-commit-config.yaml`)
4. Add `.codecov.yml` for coverage configuration:
   ```yaml
   coverage:
     status:
       project:
         default:
           target: 80%
           threshold: 5%
   ```
5. Add CI badge to README.md
6. Setup branch protection rules (require CI pass)

**Expected Files:**
- `.github/workflows/ci.yml`
- `.github/workflows/gpu-tests.yml`
- `.codecov.yml`
- `.pre-commit-config.yaml` (optional)

**Validation:**
- Push branch and verify workflow runs
- Check coverage report on codecov.io

---

## Workflow

### For Each Task:

1. **Create Feature Branch:**
   ```bash
   git checkout dev
   git pull origin dev
   git checkout -b <branch-name>
   ```

2. **Implement Task:**
   - Follow subtasks systematically
   - Write tests incrementally
   - Run tests frequently: `cargo test`
   - Ensure all tests pass

3. **Quality Checks:**
   ```bash
   cargo test                    # All tests pass
   cargo clippy -- -D warnings   # No clippy warnings
   cargo fmt                     # Format code
   cargo doc --no-deps           # Doc builds without errors
   ```

4. **Commit:**
   ```bash
   git add .
   git commit -m "<type>: <description>"
   ```

5. **Push and Create PR:**
   ```bash
   git push origin <branch-name>
   # Create PR on GitHub targeting dev branch
   ```

6. **Review and Merge:**
   - Wait for CI to pass
   - Request review if needed
   - Merge to dev

7. **Clean Up:**
   ```bash
   git checkout dev
   git pull origin dev
   git branch -d <branch-name>
   ```

---

## GPU Test Setup

**Local GPU:** NVIDIA RTX 5080 (16GB, CUDA 12.0)  
**MPS:** Available for multi-process isolation

### Enable MPS for Multi-Agent Testing:
```bash
# Start MPS daemon (run once)
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
nvidia-cuda-mps-control -d

# Verify MPS is running
ps aux | grep mps

# Stop MPS when done
echo quit | nvidia-cuda-mps-control
```

### Test with Small Models:
- Use `microsoft/phi-2` (2.7B params, ~5GB VRAM)
- Use `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (1.1B params, ~2GB VRAM)
- Download via HuggingFace Hub: `hf_hub_download`

---

## Success Criteria

- [ ] All 10 tasks completed
- [ ] Test coverage > 70% (target: 80%)
- [ ] All tests pass: `cargo test --all-features`
- [ ] No clippy warnings: `cargo clippy -- -D warnings`
- [ ] Code formatted: `cargo fmt --check`
- [ ] Documentation builds: `cargo doc --no-deps`
- [ ] CI pipeline green
- [ ] All PRs merged to dev branch

---

## Timeline Estimate

| Task | Priority | Estimated Time | Dependencies |
|------|----------|----------------|--------------|
| 1. Workspace Setup | CRITICAL | 2-3 hours | None |
| 2. Config Tests | HIGH | 4-5 hours | Task 1 |
| 3. Dataset Tests | HIGH | 5-6 hours | Task 1 |
| 4. Error Tests | MEDIUM | 2-3 hours | Task 1 |
| 5. Trainer Tests | MEDIUM | 4-5 hours | Task 1 |
| 6. Model Stubs | LOW | 1-2 hours | Task 1 |
| 7. CLI Tests | MEDIUM | 4-5 hours | Task 1 |
| 8. Doc Tests | MEDIUM | 3-4 hours | Tasks 2-5 |
| 9. Benchmarks | LOW | 2-3 hours | Task 2 |
| 10. CI/CD | HIGH | 3-4 hours | Tasks 2-7 |

**Total:** ~30-40 hours of development work

---

## Notes

- **Dev Branch Strategy:** Keep tests in source files (`#[cfg(test)] mod tests`) for now
- **Testing Branch:** Later, refactor to move all tests to `tests/` directory for organization and performance optimization
- **Mock Dependencies:** Use feature flags to switch between mocks and real crates when ready
- **Small Models:** Use Phi-2 or TinyLlama for actual GPU validation tests
- **Conventional Commits:** Strict adherence for clean git history
- **PR Reviews:** Consider pairing on complex tasks (config, dataset loaders)

---

## Next Steps

1. Review this plan with team
2. Assign tasks to agent(s)
3. Setup MPS for multi-agent GPU sharing
4. Start with Task 1 (workspace setup) - blocking all others
5. Parallelize Tasks 2-7 once Task 1 complete
6. Tasks 8-10 can run in parallel after core tests done

---

**Plan Created:** January 9, 2026  
**Plan Status:** Ready for Agent Execution  
**Target Branch:** dev  
**Future Branch:** testing (for test refinement and optimization)
