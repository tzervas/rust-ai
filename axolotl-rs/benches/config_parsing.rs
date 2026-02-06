use axolotl_rs::config::AxolotlConfig;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use std::io::Write;
use tempfile::NamedTempFile;

// Small YAML config for basic benchmarking
const SMALL_YAML: &str = r#"
base_model: meta-llama/Llama-2-7b-hf
adapter: qlora

lora:
  r: 64
  alpha: 16
  dropout: 0.05
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj

quantization:
  bits: 4
  quant_type: nf4
  double_quant: true
  block_size: 64

dataset:
  path: ./data/train.jsonl
  format: alpaca
  max_length: 2048
  val_split: 0.05

training:
  epochs: 3
  batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 2.0e-4
  lr_scheduler: cosine
  warmup_ratio: 0.03
  weight_decay: 0.0
  max_grad_norm: 1.0
  save_steps: 500
  logging_steps: 10
  gradient_checkpointing: true
  mixed_precision: true

output_dir: ./outputs/llama2-7b-qlora
seed: 42
"#;

// Generate large YAML config with many dataset paths
fn generate_large_yaml(num_datasets: usize) -> String {
    let mut yaml = String::from(
        r#"
base_model: meta-llama/Llama-2-7b-hf
adapter: qlora

lora:
  r: 64
  alpha: 16
  dropout: 0.05
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj

quantization:
  bits: 4
  quant_type: nf4
  double_quant: true
  block_size: 64

dataset:
  path: ./data/train.jsonl
  format: alpaca
  max_length: 2048
  val_split: 0.05

training:
  epochs: 3
  batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 2.0e-4
  lr_scheduler: cosine
  warmup_ratio: 0.03
  weight_decay: 0.0
  max_grad_norm: 1.0
  save_steps: 500
  logging_steps: 10
  gradient_checkpointing: true
  mixed_precision: true

output_dir: ./outputs/llama2-7b-qlora
seed: 42
"#,
    );

    // Add extra comment lines to simulate large config
    yaml.push_str("\n# Additional dataset configurations:\n");
    for i in 0..num_datasets {
        yaml.push_str(&format!("# dataset_{}: ./data/train_{}.jsonl\n", i, i));
    }

    yaml
}

// Invalid YAML for error path benchmarking
const INVALID_YAML: &str = r#"
base_model:
adapter: qlora

lora:
  r: 0
  alpha: 16

dataset:
  path: ""
  format: alpaca

training:
  epochs: 3
  batch_size: 4

output_dir: ./outputs
"#;

fn bench_config_from_yaml(c: &mut Criterion) {
    c.bench_function("config_from_yaml_small", |b| {
        b.iter(|| {
            let config: AxolotlConfig = black_box(serde_yaml::from_str(SMALL_YAML).unwrap());
            black_box(config);
        });
    });
}

fn bench_config_from_yaml_large(c: &mut Criterion) {
    let mut group = c.benchmark_group("config_from_yaml_large");

    for size in [100, 500, 1000].iter() {
        let large_yaml = generate_large_yaml(*size);
        group.bench_with_input(format!("datasets_{}", size), size, |b, _| {
            b.iter(|| {
                let config: AxolotlConfig = black_box(serde_yaml::from_str(&large_yaml).unwrap());
                black_box(config);
            });
        });
    }

    group.finish();
}

fn bench_config_validate(c: &mut Criterion) {
    c.bench_function("config_validate", |b| {
        let config = AxolotlConfig::llama2_7b_preset();
        b.iter(|| {
            let result = black_box(config.validate());
            black_box(result).unwrap();
        });
    });
}

fn bench_config_validate_invalid(c: &mut Criterion) {
    c.bench_function("config_validate_invalid", |b| {
        let mut config = AxolotlConfig::llama2_7b_preset();
        config.base_model = String::new(); // Make invalid
        config.dataset.path = String::new(); // Make invalid
        config.lora.r = 0; // Make invalid

        b.iter(|| {
            let result = black_box(config.validate());
            black_box(result.is_err());
        });
    });
}

fn bench_preset_llama2(c: &mut Criterion) {
    c.bench_function("preset_llama2", |b| {
        b.iter(|| {
            let config = black_box(AxolotlConfig::from_preset("llama2-7b").unwrap());
            black_box(config);
        });
    });
}

fn bench_preset_mistral(c: &mut Criterion) {
    c.bench_function("preset_mistral", |b| {
        b.iter(|| {
            let config = black_box(AxolotlConfig::from_preset("mistral-7b").unwrap());
            black_box(config);
        });
    });
}

fn bench_preset_phi3(c: &mut Criterion) {
    c.bench_function("preset_phi3", |b| {
        b.iter(|| {
            let config = black_box(AxolotlConfig::from_preset("phi3-mini").unwrap());
            black_box(config);
        });
    });
}

fn bench_config_roundtrip(c: &mut Criterion) {
    c.bench_function("config_roundtrip", |b| {
        let config = AxolotlConfig::llama2_7b_preset();

        b.iter(|| {
            // Serialize to YAML
            let yaml = black_box(serde_yaml::to_string(&config).unwrap());
            // Deserialize back
            let restored: AxolotlConfig = black_box(serde_yaml::from_str(&yaml).unwrap());
            black_box(restored);
        });
    });
}

fn bench_config_file_io(c: &mut Criterion) {
    c.bench_function("config_file_io", |b| {
        let config = AxolotlConfig::llama2_7b_preset();

        b.iter(|| {
            // Create temporary file
            let mut temp_file = NamedTempFile::new().unwrap();
            let yaml = serde_yaml::to_string(&config).unwrap();
            temp_file.write_all(yaml.as_bytes()).unwrap();
            temp_file.flush().unwrap();

            // Read back
            let path = temp_file.path();
            let loaded = black_box(AxolotlConfig::from_file(path).unwrap());
            black_box(loaded);
        });
    });
}

criterion_group!(
    benches,
    bench_config_from_yaml,
    bench_config_from_yaml_large,
    bench_config_validate,
    bench_config_validate_invalid,
    bench_preset_llama2,
    bench_preset_mistral,
    bench_preset_phi3,
    bench_config_roundtrip,
    bench_config_file_io,
);
criterion_main!(benches);
