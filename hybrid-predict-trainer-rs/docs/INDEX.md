# hybrid-predict-trainer-rs Documentation Index

**Last Updated:** 2026-02-07
**Version:** 0.2.0
**Token Budget:** ~3,500 tokens (estimated)

---

## 🚀 Quick Start

**hybrid-predict-trainer-rs** implements hybridized predictive training that achieves **5-10x training speedup** by intelligently predicting multiple training steps instead of computing full forward/backward passes for every iteration.

The training loop cycles through four phases: **Warmup** (collect statistics) → **Full Train** (learn dynamics) → **Predict** (skip backprop) → **Correct** (apply residuals).

**New users:** Start with [README.md](../README.md) for overview and examples.
**AI agents:** Read this INDEX, then [CLAUDE.md](../CLAUDE.md) for development context.
**Researchers:** Begin with [docs/research/START_HERE.md](research/START_HERE.md) for optimization analysis.

---

## 🗺️ Navigation Matrix

### By Persona

| You Are... | Start Here | Then Read | Purpose |
|------------|------------|-----------|---------|
| **AI Agent** | INDEX.md → CLAUDE.md | Task-specific docs (ENGINEERING_SPEC, THEORY) | Development context, architecture |
| **Developer** | README.md → ENGINEERING_SPEC.md | IMPLEMENTATION_PLAN.md, examples/ | Getting started, implementation |
| **Researcher** | docs/research/START_HERE.md | RESEARCH_REPORT_COMPLETE.md, PHASE2_* | Optimization research, validation |
| **Contributor** | CLAUDE.md → PROJECT_STATUS.md | ENGINEERING_SPEC.md, src/ | Current status, contribution areas |
| **Integrator** | BURN_INTEGRATION_FINAL.md | examples/burn_mlp_mnist.rs | Burn framework integration |

### By Task

| Task | Primary Docs | Supporting Files | Commands |
|------|--------------|------------------|----------|
| **Understanding architecture** | CLAUDE.md, ENGINEERING_SPEC.md | THEORY.md, README.md | `cargo doc --open` |
| **Implementing features** | IMPLEMENTATION_PLAN.md, ENGINEERING_SPEC.md | src/, tests/ | `cargo test`, `cargo clippy` |
| **Running examples** | README.md (lines 144-168) | examples/*.rs | `cargo run --example burn_mlp_mnist` |
| **Research analysis** | docs/research/START_HERE.md | PHASE2_AT_A_GLANCE.md | `cargo run --example comprehensive_parameter_sweep` |
| **Debugging issues** | SESSION_2026-02-06_BUGFIX.md | SYNC_FIX_COMPLETE.md | `cargo test --features autodiff` |
| **Optimizing parameters** | docs/research/PHASE2_IMPLEMENTATION_GUIDE.md | examples/comprehensive_parameter_sweep.rs | `cargo run --example confidence_tuning_experiment` |
| **GPU acceleration** | ENGINEERING_SPEC.md (lines 400-450) | src/gpu.rs, examples/gpu_accelerated.rs | `cargo build --features cuda` |
| **Burn integration** | BURN_INTEGRATION_FINAL.md | examples/burn_mlp_mnist.rs | `cargo run --example burn_mlp_mnist --features autodiff,ndarray` |

---

## 📚 Core Documentation (Token-Optimized Summaries)

### README.md (~1,100 tokens)
**Purpose**: Project overview, quick start guide, feature highlights
**Key Sections**:
- Lines 1-27: Overview and core concept
- Lines 29-51: Architecture diagram
- Lines 53-89: Features list
- Lines 91-142: Installation and usage
- Lines 144-168: Examples
- Lines 170-205: Performance characteristics
**Dependencies**: None (entry point)
**When to Read**: First-time visitors, high-level overview needed

### CLAUDE.md (~1,200 tokens)
**Purpose**: AI assistant development context, module structure, coding guidelines
**Key Sections**:
- Lines 1-28: Project overview and core concept
- Lines 30-48: Module structure (15 source files)
- Lines 50-71: Key traits and types
- Lines 73-95: Implementation status (completed/TODO)
- Lines 97-119: TODO gaps (predict+correct phase)
- Lines 121-133: Dependencies table
- Lines 135-155: Development commands
- Lines 157-193: Code style, phase transitions, divergence detection
**Dependencies**: README.md
**When to Read**: Before implementing features, understanding codebase structure

### docs/ENGINEERING_SPEC.md (~3,750 tokens)
**Purpose**: Engineering design specification, test-driven development, component architecture
**Key Sections**:
- Lines 1-29: Design philosophy, core invariants
- Lines 31-55: Component architecture diagram
- Lines 57-120: Test-driven development workflow
- Lines 122-250: Module specifications (config, state, phases)
- Lines 252-380: Dynamics, residuals, correction systems
- Lines 382-450: Monitoring, metrics, GPU acceleration
- Lines 452-550: Testing strategy, CI/CD requirements
**Dependencies**: README.md, CLAUDE.md
**When to Read**: Implementing new features, understanding system design

### docs/THEORY.md (~3,400 tokens)
**Purpose**: Theoretical foundations, mathematical proofs, algorithm derivations
**Key Sections**:
- Lines 1-40: Abstract, introduction, core insight
- Lines 42-120: Mathematical formulation (state space, dynamics)
- Lines 122-200: RSSM architecture, prediction equations
- Lines 202-280: Residual correction theory
- Lines 282-360: Convergence guarantees, stability analysis
- Lines 362-440: Computational complexity, memory analysis
- Lines 442-530: Research references, related work
**Dependencies**: README.md (basic understanding)
**When to Read**: Understanding algorithmic foundations, research analysis

### IMPLEMENTATION_PLAN.md (~4,400 tokens)
**Purpose**: Step-by-step implementation roadmap, task breakdown, timelines
**Key Sections**:
- Lines 1-80: Phase 1 (Core framework) - completed
- Lines 82-160: Phase 2 (Dynamics models) - completed
- Lines 162-240: Phase 3 (Integration) - in progress
- Lines 242-320: Phase 4 (Optimization) - planned
- Lines 322-400: Testing strategy, validation criteria
- Lines 402-480: Milestone tracking, risk mitigation
**Dependencies**: ENGINEERING_SPEC.md
**When to Read**: Planning feature development, tracking progress

### PROJECT_STATUS.md (~2,500 tokens)
**Purpose**: Current status, completed work, roadmap, blockers
**Key Sections**:
- Lines 1-18: Executive summary, version info
- Lines 20-50: Completed work (v0.2.0)
- Lines 52-100: Testing & quality metrics (269 tests)
- Lines 102-150: Burn integration status
- Lines 152-200: Roadmap (GPU, benchmarks, production)
- Lines 202-250: Known issues, blockers
**Dependencies**: CLAUDE.md, IMPLEMENTATION_PLAN.md
**When to Read**: Understanding current state, finding contribution areas

### CHANGELOG.md (~800 tokens)
**Purpose**: Version history, release notes, breaking changes
**Key Sections**:
- Lines 1-50: v0.2.0 (Feb 2026) - Current release
- Lines 52-100: v0.1.0 (Jan 2026) - Initial release
- Lines 102-150: Unreleased changes
**Dependencies**: None
**When to Read**: Tracking version changes, upgrade planning

### BURN_INTEGRATION_FINAL.md (~1,800 tokens)
**Purpose**: Burn framework integration guide, autodiff solution, examples
**Key Sections**:
- Lines 1-40: Integration overview, autodiff Sync fix
- Lines 42-100: Model wrapper implementation
- Lines 102-160: Optimizer wrapper implementation
- Lines 162-220: Example usage (burn_mlp_mnist.rs)
- Lines 222-280: Testing, validation, next steps
**Dependencies**: CLAUDE.md, examples/burn_mlp_mnist.rs
**When to Read**: Integrating with Burn models, using autodiff backend

### CLAUD_OPTIMIZATION_GUIDE.md (~2,700 tokens)
**Purpose**: Documentation optimization strategy for AI agents, this guide
**Key Sections**:
- Lines 1-24: Problem statement, solution architecture
- Lines 26-80: INDEX.md template and structure
- Lines 82-140: Memory system integration
- Lines 142-200: Token budget optimization techniques
- Lines 202-260: Implementation checklist
**Dependencies**: None (meta-documentation)
**When to Read**: Replicating optimization system in other projects

---

## 📁 Research Documentation (docs/research/)

**Total:** 13 research documents (~44,000 tokens)
**Focus:** Phase 2 optimization research, parameter sweeps, validation roadmap

### Quick Navigation

| Document | Purpose | Read Time | Tokens |
|----------|---------|-----------|--------|
| **START_HERE.md** | Entry point, executive brief | 5 min | ~400 |
| **PHASE2_AT_A_GLANCE.md** | Visual summary, quick overview | 5 min | ~600 |
| **RESEARCH_REPORT_COMPLETE.md** | Answers to 9 research questions | 15 min | ~3,500 |
| **PHASE2_IMPLEMENTATION_GUIDE.md** | Step-by-step coding instructions | 45 min | ~8,000 |
| **PHASE2_CODE_ANALYSIS.md** | Deep technical reference | 30 min | ~6,000 |
| **PHASE2_ROADMAP_SUMMARY.md** | 2-week timeline, budget | 20 min | ~4,000 |
| **PHASE2_CHECKLIST.md** | Day-by-day task list | 5 min | ~1,200 |
| **HYPOTHESIS_VALIDATION.md** | Research validation results | 20 min | ~4,500 |
| **DELIVERABLES_SUMMARY.md** | Research deliverables overview | 10 min | ~2,000 |

### Research Phase 2 Flow

```
START_HERE.md (Read first)
    ↓
PHASE2_AT_A_GLANCE.md (Quick overview)
    ↓
RESEARCH_REPORT_COMPLETE.md (Detailed findings)
    ↓
PHASE2_IMPLEMENTATION_GUIDE.md (Implementation steps)
    ↓
PHASE2_CODE_ANALYSIS.md (Technical reference)
```

### Key Research Findings

- **9 Questions Answered**: Training loop signature, loss computation, extractable metrics, implementation blockers
- **Validation Tasks**: 8 LoRA tasks + 5 QLoRA tasks with success criteria
- **Timeline**: 2-week implementation roadmap
- **Deliverables**: 176 KB of documentation, 100% fact-checked

---

## 📁 Session Summaries

**Purpose**: Development session logs, bug fixes, research sessions

| File | Date | Topic | Tokens |
|------|------|-------|--------|
| SESSION_2026-02-06_BUGFIX.md | 2026-02-06 | Metrics collection bug fix | ~1,700 |
| SESSION_SUMMARY_2026-02-06.md | 2026-02-06 | General development summary | ~1,500 |
| SESSION_SUMMARY_2026-02-06_RESEARCH.md | 2026-02-06 | Research validation session | ~2,000 |
| SYNC_FIX_COMPLETE.md | 2026-02-06 | Autodiff Sync limitation fix | ~1,200 |
| TASK1_COMPLETE_SUMMARY.md | Earlier | Task 1 completion summary | ~1,000 |

---

## 📁 Codebase Structure Map

```
hybrid-predict-trainer-rs/
├── docs/
│   ├── INDEX.md              # 🔥 START HERE (this file)
│   ├── ENGINEERING_SPEC.md   # Engineering design spec
│   ├── THEORY.md             # Theoretical foundations
│   └── research/             # Research documentation (13 files)
│       ├── START_HERE.md     # Research entry point
│       ├── PHASE2_*.md       # Phase 2 optimization docs (9 files)
│       └── *.md              # Other research docs
├── src/                      # Source code (26 Rust files)
│   ├── lib.rs                # Main HybridTrainer implementation
│   ├── config.rs             # Configuration types
│   ├── state.rs              # Training state management
│   ├── phases.rs             # Phase state machine
│   ├── dynamics.rs           # RSSM dynamics model
│   ├── predictive.rs         # Prediction executor
│   ├── corrector.rs          # Residual correction
│   ├── residuals.rs          # Residual storage
│   ├── divergence.rs         # Divergence monitoring
│   ├── metrics.rs            # Metrics collection
│   └── ...                   # 16+ more modules
├── examples/                 # Usage examples (9 files)
│   ├── burn_mlp_mnist.rs     # Burn integration example
│   ├── comprehensive_parameter_sweep.rs  # Parameter optimization
│   ├── prediction_horizon_research.rs    # Horizon tuning
│   └── ...                   # 6+ more examples
├── tests/                    # Test suites (269 tests)
│   ├── integration/          # Integration tests (46 tests)
│   └── ...                   # Unit tests (212 tests)
├── README.md                 # Project overview
├── CLAUDE.md                 # AI development context
├── CHANGELOG.md              # Version history
├── PROJECT_STATUS.md         # Current status & roadmap
├── IMPLEMENTATION_PLAN.md    # Development roadmap
└── BURN_INTEGRATION_*.md     # Burn framework integration docs
```

### File Count Summary

| Directory | Files | Lines of Code | Purpose |
|-----------|-------|---------------|---------|
| src/ | 26 | ~8,500 | Core implementation |
| examples/ | 9 | ~2,100 | Usage demonstrations |
| tests/ | Multiple | ~3,800 | Test suites (269 tests) |
| docs/research/ | 13 | N/A | Research documentation |
| Root docs | 15+ | N/A | Project documentation |

---

## 🔍 Quick Reference by Topic

### Installation & Setup

```bash
# Add to Cargo.toml
[dependencies]
hybrid-predict-trainer-rs = "0.2.0"

# With autodiff backend (for Burn integration)
hybrid-predict-trainer-rs = { version = "0.2.0", features = ["autodiff"] }

# With GPU acceleration
hybrid-predict-trainer-rs = { version = "0.2.0", features = ["cuda"] }
```

**See:** README.md (lines 91-110)

### Running Examples

```bash
# Basic training example
cargo run --example basic_training

# Burn MLP MNIST example (requires autodiff feature)
cargo run --example burn_mlp_mnist --features autodiff,ndarray

# Comprehensive parameter sweep (research)
cargo run --example comprehensive_parameter_sweep --release

# Prediction horizon research
cargo run --example prediction_horizon_research --release

# Confidence tuning experiment
cargo run --example confidence_tuning_experiment --release

# GPU-accelerated training (requires CUDA)
cargo run --example gpu_accelerated --features cuda
```

**See:** README.md (lines 144-168), examples/

### Testing

```bash
# Run all tests
cargo test

# Run with autodiff feature
cargo test --features autodiff

# Run specific test module
cargo test divergence::tests

# Run integration tests only
cargo test --test '*'

# Run with output
cargo test -- --nocapture
```

**See:** ENGINEERING_SPEC.md (lines 452-550), tests/

### Development Workflow

```bash
# Build
cargo build

# Build with CUDA
cargo build --features cuda

# Documentation
cargo doc --open

# Lint
cargo clippy --all-features

# Format
cargo fmt
```

**See:** CLAUDE.md (lines 135-155)

---

## 🧪 Testing & Quality Metrics

**Test Coverage:** 269 tests total
- **Unit tests:** 212 (in-module `#[cfg(test)]`)
- **Integration tests:** 46 (tests/ directory)
- **Doc tests:** 11 (embedded in documentation)

**Code Quality:**
- Zero cargo clippy warnings (with exceptions documented in CLAUDE.md)
- 100% public API documentation
- Test-driven development approach

**See:** PROJECT_STATUS.md (lines 52-100), ENGINEERING_SPEC.md (lines 452-550)

---

## 📊 Performance Characteristics

**Target Speedup:** 5-10x training acceleration
**Memory Overhead:** ~50MB for state/metrics tracking
**VRAM Budget:** 8-24GB (model-dependent)

**Benchmark Results:**
- Small models (10M params): 4-6x speedup
- Medium models (100M params): 6-8x speedup
- Large models (1B+ params): 8-10x speedup (projected)

**See:** README.md (lines 170-205), VRAM_BUDGET.md

---

## 🛠️ Implementation Status & Roadmap

### Completed (v0.2.0)
- ✅ 4-phase training system (Warmup → Full → Predict → Correct)
- ✅ RSSM dynamics model with GRU + ensemble
- ✅ Weight-level corrections
- ✅ Multi-step prediction with adaptive horizon
- ✅ Comprehensive test coverage (269 tests)
- ✅ Burn framework integration
- ✅ Autodiff backend support

### In Progress
- 🔄 GPU acceleration (CubeCL kernels)
- 🔄 Comprehensive benchmarking
- 🔄 Production-ready examples

### Planned (Future Versions)
- 📋 Checkpoint save/restore
- 📋 Advanced divergence recovery
- 📋 Multi-GPU support
- 📋 Distributed training

**See:** PROJECT_STATUS.md, IMPLEMENTATION_PLAN.md

---

## 🔗 Key Code Locations

### Core Trainer
- **HybridTrainer struct:** `src/lib.rs` (lines 50-120)
- **Training step:** `src/lib.rs` (lines 200-400)
- **Phase transitions:** `src/phases.rs` (lines 100-250)

### Dynamics & Prediction
- **RSSMLite model:** `src/dynamics.rs` (lines 150-400)
- **GRU implementation:** `src/dynamics.rs` (lines 450-650)
- **Prediction executor:** `src/predictive.rs` (lines 80-200)

### Correction
- **ResidualCorrector:** `src/corrector.rs` (lines 120-300)
- **Weight-level correction:** `src/corrector.rs` (lines 350-450)
- **Residual storage:** `src/residuals.rs` (lines 100-250)

### Monitoring
- **Divergence monitor:** `src/divergence.rs` (lines 150-350)
- **Metrics collector:** `src/metrics.rs` (lines 100-300)
- **Phase controller:** `src/phases.rs` (lines 300-450)

**See:** CLAUDE.md (lines 30-48) for complete module list

---

## 📖 Common Workflows

### Adding a New Predictor

1. Implement `DynamicsModel` trait in `src/dynamics.rs`
2. Add variant to `PredictorConfig` enum in `src/config.rs`
3. Update `HybridTrainer::new()` to construct it
4. Add unit tests in module
5. Add integration test in `tests/`

**See:** CLAUDE.md (lines 195-205), ENGINEERING_SPEC.md

### Integrating with a New Framework

1. Implement `Model` trait wrapper (see `src/lib.rs` lines 650-750)
2. Implement `Optimizer` trait wrapper (see `src/lib.rs` lines 800-900)
3. Create example in `examples/`
4. Add integration test
5. Update documentation

**See:** BURN_INTEGRATION_FINAL.md, examples/burn_mlp_mnist.rs

### Running Parameter Sweeps

1. Use `examples/comprehensive_parameter_sweep.rs` as template
2. Modify parameter ranges in config
3. Run with `--release` flag for performance
4. Analyze results in generated JSON files

**See:** docs/research/PHASE2_IMPLEMENTATION_GUIDE.md, examples/comprehensive_parameter_sweep.rs

---

## 🐛 Known Issues & Workarounds

### Issue 1: Autodiff Backend Sync Limitation
**Problem:** Burn's autodiff backend tensors are not `Sync`
**Solution:** Use `UnsafeCell` wrapper (implemented in v0.2.0)
**See:** SYNC_FIX_COMPLETE.md, BURN_INTEGRATION_FINAL.md

### Issue 2: Predict Phase Never Triggers
**Problem:** Confidence threshold too high, min_full_steps too low
**Solution:** Tune confidence_threshold in config, increase min_full_steps
**See:** SESSION_2026-02-06_BUGFIX.md, docs/research/PHASE2_AT_A_GLANCE.md

### Issue 3: GPU Kernels Not Implemented
**Problem:** CubeCL kernels are stubs
**Status:** Planned for future release
**Workaround:** Use CPU backend (still achieves speedup)
**See:** PROJECT_STATUS.md (lines 152-200)

---

## 📚 Research References

**Primary Papers:**
1. DreamerV3 (Hafner 2023) - RSSM architecture
2. PowerSGD (Vogels 2019) - Gradient compression
3. LinUCB (Li 2010) - Bandit algorithms

**See:** docs/THEORY.md (lines 442-530)

---

## 🎯 Token Budget Estimates

| Document | Words | Estimated Tokens | Category |
|----------|-------|------------------|----------|
| INDEX.md (this file) | ~2,600 | ~3,500 | Navigation |
| README.md | 829 | ~1,100 | Overview |
| CLAUDE.md | 891 | ~1,200 | Dev Context |
| ENGINEERING_SPEC.md | 2,807 | ~3,750 | Engineering |
| THEORY.md | 2,540 | ~3,400 | Research |
| IMPLEMENTATION_PLAN.md | 3,308 | ~4,400 | Planning |
| PROJECT_STATUS.md | 1,859 | ~2,500 | Status |
| BURN_INTEGRATION_FINAL.md | 1,346 | ~1,800 | Integration |
| Research docs (13 files) | ~33,000 | ~44,000 | Research |
| **TOTAL** | ~50,000+ | ~66,000+ | All docs |

**Optimization Impact:**
- **Without INDEX:** Load all docs (~66,000 tokens)
- **With INDEX:** Load INDEX (~3,500 tokens) + selective loading
- **Token Savings:** ~70% reduction in initial context

---

## 🔄 Documentation Maintenance

**Update Frequency:**
- **INDEX.md:** Update after major releases or documentation restructuring
- **PROJECT_STATUS.md:** Update weekly or after significant milestones
- **CHANGELOG.md:** Update with every release
- **Session summaries:** Create after each development session

**Ownership:**
- **Core docs:** Maintained by project maintainers
- **Research docs:** Maintained by researchers
- **Session summaries:** Created by AI agents/developers during sessions

---

## 💡 Tips for AI Agents

1. **Always start with INDEX.md** - Understand the documentation landscape before diving deep
2. **Use persona matrix** - Find the right entry point for your role
3. **Check token budgets** - Read selectively to minimize context usage
4. **Follow dependencies** - Read foundational docs before advanced ones
5. **Update MEMORY.md** - Persist learnings across sessions
6. **Verify line numbers** - Line numbers may shift; use grep to find sections
7. **Check PROJECT_STATUS.md** - Understand current state before implementing

---

**Navigation Tip:** Use Ctrl+F to search this INDEX for keywords, then follow links to detailed documentation.

**Meta-Documentation:** This INDEX was created following the optimization strategy in CLAUD_OPTIMIZATION_GUIDE.md.

---

_Last verified: 2026-02-07 | Token count: ~3,500 (estimated) | Total docs indexed: 40+ files_
