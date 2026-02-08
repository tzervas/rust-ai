# GPU Coordination & Multi-Session Management Plan

**Date**: 2026-02-07
**Context**: Multiple Claude Code sessions working on GPU-intensive tasks
**Hardware**: NVIDIA RTX 5080 (16GB VRAM)

---

## Current Sessions

### Session 1: hybrid-predict-trainer-rs (THIS SESSION)
- **Task**: Phase 2B validation + VRAM leak investigation
- **VRAM Usage**: 4-14 GB (varies by phase)
- **Model**: Opus/Sonnet mix
- **Status**: Active, VRAM leak identified

### Session 2: self-hosted-ai (VS Code)
- **Task**: Unknown (user hasn't specified details)
- **VRAM Usage**: Unknown
- **Status**: Active (inferred from user message)

---

## GPU Resource Constraints

### Hardware Limits
- **Total VRAM**: 16 GB (RTX 5080)
- **Reserved for system**: ~0.5-1 GB
- **Available for training**: ~15 GB

### Current Risk
- **Session 1 alone**: Uses up to 14 GB
- **If Session 2 also uses GPU**: Potential OOM conflict
- **No coordination**: Both sessions compete for VRAM → crashes

---

## Coordination Strategy

### Option 1: Time-Slicing (Sequential Execution)

**Approach**: Run GPU-intensive tasks one at a time

```bash
# Session 1 (this one)
# 1. Finish current VRAM investigation
# 2. Implement fix
# 3. Run quick validation (5-10 min)
# 4. SIGNAL: "GPU available" to Session 2

# Session 2 (self-hosted-ai)
# Wait for signal, then run GPU tasks
# Signal back when done
```

**Implementation**:
```bash
# Shared lock file approach
LOCK_FILE="/tmp/gpu_lock"

# Before starting GPU work:
while [ -f "$LOCK_FILE" ]; do
  echo "Waiting for GPU (locked by $(cat $LOCK_FILE))"
  sleep 10
done
echo "session-1-hybrid" > $LOCK_FILE

# After finishing GPU work:
rm -f $LOCK_FILE
```

**Pros**: Simple, no conflicts, full VRAM for each task
**Cons**: Serialized (slower total time), manual coordination

### Option 2: VRAM Budget Allocation (Parallel Execution)

**Approach**: Each session gets a VRAM budget

```bash
# Session 1: 8 GB budget (primary training)
# Session 2: 6 GB budget (secondary tasks)
# Reserve: 2 GB for system/spikes

# In Session 1, use smaller batch sizes:
physical_batch = 2  # Instead of 4
# This keeps VRAM under 8 GB
```

**Implementation via CUDA Environment Variables**:
```bash
# Session 1
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:8192"

# Session 2
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:6144"
```

**Pros**: Parallel execution, both sessions can work
**Cons**: Reduced performance per session, fragmentation risk

### Option 3: Checkpointing + Offloading (Memory Management)

**Approach**: Aggressively checkpoint and offload to disk

```rust
// In hybrid-predict-trainer-rs
impl HybridTrainer {
    fn checkpoint_to_disk(&self, path: &Path) -> HybridResult<()> {
        // Save model, optimizer, state to disk
        // Free VRAM after saving
    }

    fn restore_from_disk(&mut self, path: &Path) -> HybridResult<()> {
        // Load back when needed
    }
}

// Checkpoint every N steps:
if step % 10 == 0 {
    trainer.checkpoint_to_disk(&format!("checkpoints/step_{}.ckpt", step))?;
    // Optional: unload model from GPU to free VRAM
}
```

**Pros**: Resilient to crashes, enables VRAM sharing
**Cons**: I/O overhead (~1-2s per checkpoint for 124M model)

### Option 4: Multi-Agent Coordination (Opus-Driven)

**Approach**: Use Opus agent to orchestrate GPU usage

```yaml
# Pseudo-config for Opus orchestrator
agents:
  - name: hybrid-trainer (Sonnet)
    task: Train GPT-2 with HybridTrainer
    vram_budget: 8 GB
    priority: high

  - name: self-hosted-ai (Haiku)
    task: Secondary GPU tasks
    vram_budget: 6 GB
    priority: low

# Opus decides scheduling based on:
# - Current VRAM usage (nvidia-smi query)
# - Task priority
# - Estimated duration
# - Checkpointing availability
```

**Implementation**: Would require custom orchestration script

---

## Recommended Approach: Hybrid (Option 1 + Option 3)

### Phase 1: Immediate (Time-Slicing)
1. **Session 1** (this one):
   - Finish VRAM leak fix (next 30 min)
   - Run quick validation (10 min)
   - Checkpoint state
   - Release GPU (signal via lock file)

2. **Session 2** (self-hosted-ai):
   - Wait for Session 1 to finish
   - Acquire GPU lock
   - Run GPU tasks
   - Signal when done

### Phase 2: Parallel Execution (VRAM Budgets)
1. Implement VRAM budgets (8 GB / 6 GB split)
2. Test both sessions running simultaneously
3. Monitor for OOM or conflicts

### Phase 3: Automated Coordination (Opus Orchestrator)
1. Create Opus agent for GPU scheduling
2. Integrate with both sessions
3. Dynamic VRAM allocation based on workload

---

## Model Assignment Strategy (Per User Request)

### Opus (Complex Planning & Debugging)
- **Use cases**:
  - Architectural decisions (e.g., "should we use delta accumulation?")
  - Root cause analysis (e.g., "why is VRAM growing?")
  - Multi-session coordination planning
  - Trade-off evaluation

- **Current task**: GPU coordination strategy (THIS DOCUMENT)

### Sonnet (Implementation & Execution)
- **Use cases**:
  - Writing code (fixes, features)
  - Running experiments (benchmarks, validation)
  - Integration work
  - Documentation

- **Current tasks**:
  - Implementing VRAM leak fix
  - Running Phase 2B validation
  - Writing analysis documents

### Haiku (Fast Execution & Simple Tasks)
- **Use cases**:
  - Parameter sweeps (60 configs)
  - Simple code generation
  - File operations
  - Quick experiments

- **Future tasks**:
  - 3D parameter sweep execution
  - Batch experiment running

---

## Immediate Action Plan (Next 30 Minutes)

### Task 1: Fix VRAM Leak (Sonnet)
```bash
# Set correction_interval=0 in hybrid example
# Rebuild and test
# Expected VRAM: 3.9 GB → 6-7 GB (vs current 14 GB)
```

### Task 2: Validate Fix (Sonnet)
```bash
# Run 50-step validation
# Monitor VRAM growth
# Document results
```

### Task 3: Signal GPU Available (Manual)
```bash
# Create lock file when done
rm -f /tmp/gpu_lock
echo "GPU available for self-hosted-ai session"
```

### Task 4: Coordinate with Session 2 (User)
User should check self-hosted-ai VS Code session and determine:
1. What GPU tasks are pending?
2. What VRAM budget is needed?
3. Can they wait for Session 1 to finish?

---

## Monitoring & Alerts

### VRAM Monitoring Script
```bash
#!/bin/bash
# monitor_vram.sh - Run in background

while true; do
  VRAM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
  echo "$TIMESTAMP,$VRAM" >> vram_log.csv

  if [ "$VRAM" -gt 14000 ]; then
    echo "⚠️ WARNING: VRAM > 14 GB! Risk of OOM!"
    notify-send "GPU Alert" "VRAM critical: ${VRAM} MB"
  fi

  sleep 10
done
```

### Checkpointing Safety Net
```bash
# Auto-checkpoint every 10 minutes
watch -n 600 'curl -X POST http://localhost:8080/checkpoint'
```

---

## Long-Term Solution: GPU Server Architecture

For **true multi-session coordination**, consider:

1. **GPU Queue System**:
   - SLURM, Kubernetes with GPU scheduling
   - Automatic resource allocation

2. **Model Serving**:
   - Load models on-demand
   - Share GPU across sessions via API

3. **Cloud/Multi-GPU**:
   - Rent additional GPUs for parallel work
   - Or, get a second GPU for the workstation

---

## Status Summary

**Current**: Session 1 (this) monopolizing GPU, no coordination
**Next 30 min**: Fix VRAM leak, validate, release GPU
**Phase 2**: Implement time-slicing with lock file
**Phase 3**: Consider automated orchestration with Opus

**User Action Required**:
1. Confirm self-hosted-ai session GPU usage
2. Decide on time-slicing vs parallel approach
3. Approve implementation plan
