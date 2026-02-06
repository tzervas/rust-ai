---
name: axolotl-config-management
description: Create, validate, and debug YAML training configurations for axolotl-rs fine-tuning
---

# Configuration Management Skill

## When to Use

Invoke when the user asks to:
- Create a new training configuration
- Debug configuration validation errors
- Optimize hyperparameters for a specific model/task
- Convert between configuration formats
- Add support for new configuration options

## Configuration Structure

```yaml
base_model: <model_id>
adapter: lora | qlora | none

lora:
  r: 64
  alpha: 16
  dropout: 0.05
  target_modules: [q_proj, v_proj, ...]

quantization:  # Only for qlora
  bits: 4
  quant_type: nf4
  double_quant: true

dataset:
  path: ./data.jsonl
  format: alpaca | sharegpt | completion | custom
  max_length: 2048

training:
  epochs: 3
  batch_size: 4
  learning_rate: 2e-4
  ...

output_dir: ./outputs
```

## Common Tasks

### Create Config for Model
```bash
axolotl init config.yaml --preset llama2-7b
```

### Validate Config
```bash
axolotl validate config.yaml
```

### Hyperparameter Recommendations

| Model Size | LoRA r | Learning Rate | Batch Size |
|------------|--------|---------------|------------|
| 7B | 64 | 2e-4 | 4 |
| 13B | 32 | 1e-4 | 2 |
| 70B | 16 | 5e-5 | 1 |

## Adding New Config Options

### 1. Add to Config Struct
```rust
// src/config.rs
#[derive(Deserialize)]
pub struct NewSetting {
    #[serde(default)]
    pub option: bool,
}
```

### 2. Add Validation
```rust
fn validate(&self) -> Result<()> {
    // Check new option
}
```

### 3. Update Presets
```rust
fn llama2_7b_preset() -> Self {
    Self {
        new_setting: NewSetting::default(),
        ...
    }
}
```

## Key Files

- `src/config.rs` - Configuration types and validation
- `src/main.rs` - CLI commands
- `examples/configs/` - Example configurations
