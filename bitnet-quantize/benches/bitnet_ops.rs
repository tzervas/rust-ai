//! Benchmarks for BitNet operations.

use bitnet_quantize::{BitLinear, BitNetConfig};
use candle_core::Device;
use candle_nn::Module;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

fn bench_bitlinear_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("bitlinear_forward");
    let device = Device::Cpu;

    for (out_features, in_features) in [(64, 128), (256, 512), (1024, 4096)].iter() {
        let config = BitNetConfig::default();
        let weight =
            candle_core::Tensor::randn(0.0f32, 1.0, (*out_features, *in_features), &device)
                .unwrap();
        let layer = BitLinear::from_weight(&weight, None, &config).unwrap();

        let input = candle_core::Tensor::randn(0.0f32, 1.0, (4, *in_features), &device).unwrap();

        let label = format!("{}x{}", out_features, in_features);
        group.bench_with_input(BenchmarkId::new("forward", &label), &(), |bench, _| {
            bench.iter(|| black_box(layer.forward(&input).unwrap()))
        });
    }

    group.finish();
}

fn bench_weight_quantization(c: &mut Criterion) {
    let mut group = c.benchmark_group("weight_quantization");
    let device = Device::Cpu;

    for size in [64, 256, 1024].iter() {
        let config = BitNetConfig::default();
        let weight = candle_core::Tensor::randn(0.0f32, 1.0, (*size, *size * 4), &device).unwrap();

        group.bench_with_input(BenchmarkId::new("quantize", size), size, |bench, _| {
            bench.iter(|| black_box(bitnet_quantize::quantize_weights(&weight, &config).unwrap()))
        });
    }

    group.finish();
}

fn bench_activation_quantization(c: &mut Criterion) {
    let mut group = c.benchmark_group("activation_quantization");
    let device = Device::Cpu;

    for (batch, seq, hidden) in [(4, 128, 512), (8, 256, 1024), (16, 512, 2048)].iter() {
        let config = BitNetConfig::default();
        let activations =
            candle_core::Tensor::randn(0.0f32, 1.0, (*batch, *seq, *hidden), &device).unwrap();

        let label = format!("{}x{}x{}", batch, seq, hidden);
        group.bench_with_input(BenchmarkId::new("quantize", &label), &(), |bench, _| {
            bench.iter(|| {
                black_box(bitnet_quantize::quantize_activations(&activations, &config).unwrap())
            })
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_bitlinear_forward,
    bench_weight_quantization,
    bench_activation_quantization
);
criterion_main!(benches);
