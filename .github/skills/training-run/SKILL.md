---
name: training-run
description: Configure, launch, and monitor a model training run using the rust-ai training stack. Covers hybrid-predict-trainer-rs configuration, tritter-model-rs execution, and training-tools monitoring. Use when setting up or debugging training pipelines.
metadata:
  author: tzervas
  version: "1.0"
allowed-tools: Bash(cargo:*) Read Glob Grep Edit Write
---

# Training Run Management

## When to use
- Setting up a new training configuration
- Launching a training run
- Debugging training issues (loss divergence, OOM, NaN)
- Reviewing training metrics and logs

## Configuration

### hybrid-predict-trainer-rs
The trainer config lives in `HybridTrainerConfig`. Key parameters:
- `warmup_steps`: Baseline statistics collection (default: 100)
- `full_steps`: Full training steps per cycle (default: 20)
- `max_predict_steps`: Max prediction horizon (default: 50)
- `confidence_threshold`: Minimum confidence for predictions (default: 0.85)
- `auto_tuning_config`: Optional health-based adaptive tuning

Example config in `examples/basic_training.rs`.

### Training phases
1. **Warmup** - Collect loss/gradient baselines
2. **Full Train** - Standard training + dynamics learning
3. **Predict** - Skip backward passes using learned dynamics
4. **Correct** - Apply residual corrections

### Divergence signals to watch
- Loss deviation > 3 sigma from EMA
- Gradient norm > 10x baseline (explosion)
- Gradient norm < 0.01x baseline (vanishing)
- NaN/Inf in any tensor
- Prediction error > 20% relative

## Monitoring with training-tools
```bash
cargo run -p training-tools -- --metrics-path <path_to_metrics.json>
```

## Troubleshooting
- **OOM**: Reduce batch size, enable gradient checkpointing
- **Loss NaN**: Check learning rate, reduce warmup steps
- **Slow convergence**: Enable auto-tuning, check LR schedule
