# CLAUDE.md - Training Tools Development Context

## Overview

Training infrastructure for rust-ai, providing:
- Real-time training monitor with TUI
- HuggingFace Hub integration
- Checkpoint management with compression
- Progressive parameter expansion (100M → 500M → 1B)

## Binaries

### `train`
Main training orchestrator with progressive expansion.

```bash
# Train just 100M model
./target/release/train --start 100m --end 100m --steps 10000

# Full progressive training: 100M → 500M → 1B
./target/release/train --start 100m --end 1b --steps 50000

# With CUDA and HuggingFace upload
./target/release/train --start 100m --end 1b --cuda --upload --hf-user myname
```

### `train-monitor`
Real-time TUI for monitoring training runs.

```bash
./target/release/train-monitor --runs-dir ./runs
```

Hotkeys: `q` quit, `j/k` navigate, `r` refresh, `d` toggle details

### `hf-upload`
Upload trained models to HuggingFace Hub.

```bash
./target/release/hf-upload --model ./runs/tritter_100m/model.safetensors \
    --size 100m --user myname
```

## Module Structure

```
training-tools/
├── src/
│   ├── lib.rs              # Public API
│   ├── training_state.rs   # TrainingRun, StepMetrics, RunManager
│   ├── monitor.rs          # TUI training monitor
│   ├── hf.rs               # HuggingFace Hub integration
│   ├── checkpoint_manager.rs # Checkpoint save/load/compress
│   ├── progressive.rs      # Progressive training pipeline
│   └── bin/
│       ├── train.rs        # Training binary
│       ├── train_monitor.rs # Monitor binary
│       └── hf_upload.rs    # Upload binary
```

## Key Types

```rust
// Training run state
TrainingRun {
    run_id, run_name, status,
    current_step, current_loss, current_phase,
    total_forward, total_backward,
    best_loss, best_step,
    ...
}

// Step metrics (written to metrics.jsonl)
StepMetrics {
    step, loss, gradient_norm, phase,
    was_predicted, prediction_error, step_time_ms,
}

// Progressive training stages
TrainingStage { Small100M, Medium500M, Large1B }
```

## Features

- **Gradient Checkpointing**: Enabled by default (every 4 layers)
- **Hybrid Training**: Warmup → Full → Predict → Correct phases
- **Checkpoint Compression**: gzip compression for storage efficiency
- **HuggingFace Upload**: Automatic model card generation

## Performance Notes

- CPU: ~0.2 steps/s for 100M model (development only)
- CUDA: ~5-10x faster (use `--cuda` flag)
- Gradient checkpointing reduces memory ~75% at ~33% compute cost

## Dependencies

- `tritter-model-rs` - Model implementation
- `hybrid-predict-trainer-rs` - Predictive training framework
- `ratatui` / `crossterm` - TUI
- `reqwest` - HTTP for HuggingFace API

## Known Limitations

1. Weight expansion (100M→500M→1B) is placeholder (random init)
2. HuggingFace upload requires `huggingface-cli` installed
3. No real dataset loading (uses random tokens)
4. peft-rs excluded due to safetensors version conflict
