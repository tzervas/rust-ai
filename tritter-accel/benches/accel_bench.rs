//! Benchmarks for tritter-accel operations.
//!
//! These benchmarks test the core Rust operations that underlie the Python bindings.
//! Run with: cargo bench -p tritter-accel
//!
//! Performance targets:
//! - quantize_weights_absmean: < 10ms for 1024x1024
//! - pack/unpack_ternary: < 1ms for 1024x1024
//! - ternary_matmul: competitive with float matmul
//! - compress_gradients_vsa: throughput > 100MB/s

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

use bitnet_quantize::{quantize_weights, BitNetConfig};
use candle_core::{Device, Tensor};
use trit_vsa::{vsa, PackedTritVec, Trit};

// ============================================================================
// Helper Functions
// ============================================================================

/// Create a random f32 weight matrix with normal distribution.
fn create_random_weights(rows: usize, cols: usize, seed: u64) -> Vec<f32> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..rows * cols)
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect()
}

/// Create a random ternary vector.
fn create_random_ternary(size: usize, seed: u64) -> PackedTritVec {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut vec = PackedTritVec::new(size);
    for i in 0..size {
        let val: i8 = rng.gen_range(-1..=1);
        let trit = match val {
            1 => Trit::P,
            -1 => Trit::N,
            _ => Trit::Z,
        };
        vec.set(i, trit);
    }
    vec
}

/// Create random gradients for compression benchmarks.
fn create_random_gradients(size: usize, seed: u64) -> Vec<f32> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..size).map(|_| rng.gen_range(-1.0..1.0)).collect()
}

// ============================================================================
// Quantization Benchmarks
// ============================================================================

fn bench_quantize_weights_absmean(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantize_weights_absmean");
    let device = Device::Cpu;
    let config = BitNetConfig::default().with_group_size(64);

    for size in [64, 256, 512, 1024].iter() {
        let weights = create_random_weights(*size, *size, 42);
        let tensor = Tensor::from_vec(weights, (*size, *size), &device).unwrap();

        // Set throughput for MB/s calculation
        group.throughput(Throughput::Bytes((size * size * 4) as u64));

        group.bench_with_input(
            BenchmarkId::new("matrix", format!("{}x{}", size, size)),
            &tensor,
            |b, tensor| {
                b.iter(|| quantize_weights(black_box(tensor), black_box(&config)).unwrap())
            },
        );
    }
    group.finish();
}

fn bench_quantize_weights_group_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantize_weights_group_sizes");
    let device = Device::Cpu;
    let size = 512;

    let weights = create_random_weights(size, size, 42);
    let tensor = Tensor::from_vec(weights, (size, size), &device).unwrap();

    for group_size in [32, 64, 128, 256].iter() {
        let config = BitNetConfig::default().with_group_size(*group_size);

        group.bench_with_input(
            BenchmarkId::new("group_size", group_size),
            &tensor,
            |b, tensor| {
                b.iter(|| quantize_weights(black_box(tensor), black_box(&config)).unwrap())
            },
        );
    }
    group.finish();
}

// ============================================================================
// Packing Benchmarks
// ============================================================================

fn bench_pack_ternary_weights(c: &mut Criterion) {
    let mut group = c.benchmark_group("pack_ternary_weights");

    for size in [64, 256, 512, 1024].iter() {
        // Create ternary values as i8
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let ternary_values: Vec<i8> = (0..size * size)
            .map(|_| rng.gen_range(-1i8..=1i8))
            .collect();

        // Throughput: packing rate
        group.throughput(Throughput::Elements((size * size) as u64));

        group.bench_with_input(
            BenchmarkId::new("matrix", format!("{}x{}", size, size)),
            &ternary_values,
            |b, values| {
                b.iter(|| {
                    let mut packed_vecs: Vec<PackedTritVec> = Vec::with_capacity(*size);
                    for row in 0..*size {
                        let mut packed = PackedTritVec::new(*size);
                        for col in 0..*size {
                            let val = values[row * size + col];
                            let trit = match val {
                                v if v > 0 => Trit::P,
                                v if v < 0 => Trit::N,
                                _ => Trit::Z,
                            };
                            packed.set(col, trit);
                        }
                        packed_vecs.push(packed);
                    }
                    black_box(packed_vecs)
                })
            },
        );
    }
    group.finish();
}

fn bench_unpack_ternary_weights(c: &mut Criterion) {
    let mut group = c.benchmark_group("unpack_ternary_weights");

    for size in [64, 256, 512, 1024].iter() {
        // Pre-create packed vectors
        let mut packed_vecs: Vec<PackedTritVec> = Vec::with_capacity(*size);
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        for _ in 0..*size {
            let mut packed = PackedTritVec::new(*size);
            for col in 0..*size {
                let val: i8 = rng.gen_range(-1..=1);
                let trit = match val {
                    v if v > 0 => Trit::P,
                    v if v < 0 => Trit::N,
                    _ => Trit::Z,
                };
                packed.set(col, trit);
            }
            packed_vecs.push(packed);
        }

        let scales: Vec<f32> = (0..*size).map(|i| 0.1 + (i as f32 * 0.001)).collect();

        group.throughput(Throughput::Elements((size * size) as u64));

        group.bench_with_input(
            BenchmarkId::new("matrix", format!("{}x{}", size, size)),
            &(packed_vecs.clone(), scales.clone()),
            |b, (packed_vecs, scales)| {
                b.iter(|| {
                    let mut weights = vec![0.0f32; size * size];
                    for (row_idx, pvec) in packed_vecs.iter().enumerate() {
                        let scale = scales[row_idx];
                        for col_idx in 0..*size {
                            let value = f32::from(pvec.get(col_idx).value()) * scale;
                            weights[row_idx * size + col_idx] = value;
                        }
                    }
                    black_box(weights)
                })
            },
        );
    }
    group.finish();
}

// ============================================================================
// Matrix Multiplication Benchmarks
// ============================================================================

fn bench_ternary_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("ternary_matmul");

    for size in [64, 128, 256, 512].iter() {
        let batch_size = 4;
        let in_features = *size;
        let out_features = *size;

        // Create packed weights
        let mut weight_vecs: Vec<PackedTritVec> = Vec::with_capacity(out_features);
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        for _ in 0..out_features {
            let mut pvec = PackedTritVec::new(in_features);
            for i in 0..in_features {
                let val: i8 = rng.gen_range(-1..=1);
                let trit = match val {
                    v if v > 0 => Trit::P,
                    v if v < 0 => Trit::N,
                    _ => Trit::Z,
                };
                pvec.set(i, trit);
            }
            weight_vecs.push(pvec);
        }

        let scales: Vec<f32> = (0..out_features).map(|i| 0.1 + (i as f32 * 0.001)).collect();
        let input: Vec<f32> = (0..batch_size * in_features)
            .map(|_| rng.gen_range(-1.0f32..1.0f32))
            .collect();

        // FLOPs for matmul: 2 * batch * in * out
        group.throughput(Throughput::Elements(
            (2 * batch_size * in_features * out_features) as u64,
        ));

        group.bench_with_input(
            BenchmarkId::new("ternary", format!("{}x{}", size, size)),
            &(input.clone(), weight_vecs.clone(), scales.clone()),
            |b, (input, weight_vecs, scales)| {
                b.iter(|| {
                    let mut output = vec![0.0f32; batch_size * out_features];

                    for batch in 0..batch_size {
                        for (o, weight_vec) in weight_vecs.iter().enumerate() {
                            let scale = scales[o];
                            let mut sum = 0.0f32;

                            for i in 0..in_features {
                                let trit = weight_vec.get(i);
                                let x = input[batch * in_features + i];
                                sum += match trit {
                                    Trit::P => x,
                                    Trit::N => -x,
                                    Trit::Z => 0.0,
                                };
                            }

                            output[batch * out_features + o] = sum * scale;
                        }
                    }
                    black_box(output)
                })
            },
        );
    }
    group.finish();
}

fn bench_ternary_vs_float_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul_comparison");

    let size = 256;
    let batch_size = 4;
    let in_features = size;
    let out_features = size;

    let mut rng = ChaCha8Rng::seed_from_u64(42);

    // Setup for ternary matmul
    let mut weight_vecs: Vec<PackedTritVec> = Vec::with_capacity(out_features);
    for _ in 0..out_features {
        let mut pvec = PackedTritVec::new(in_features);
        for i in 0..in_features {
            let val: i8 = rng.gen_range(-1..=1);
            let trit = match val {
                v if v > 0 => Trit::P,
                v if v < 0 => Trit::N,
                _ => Trit::Z,
            };
            pvec.set(i, trit);
        }
        weight_vecs.push(pvec);
    }

    let scales: Vec<f32> = (0..out_features).map(|i| 0.1 + (i as f32 * 0.001)).collect();

    // Setup for float matmul
    let weight_float: Vec<f32> = (0..out_features * in_features)
        .map(|_| rng.gen_range(-1.0f32..1.0f32))
        .collect();

    let input: Vec<f32> = (0..batch_size * in_features)
        .map(|_| rng.gen_range(-1.0f32..1.0f32))
        .collect();

    // Benchmark ternary matmul
    group.bench_function("ternary_256x256", |b| {
        b.iter(|| {
            let mut output = vec![0.0f32; batch_size * out_features];

            for batch in 0..batch_size {
                for (o, weight_vec) in weight_vecs.iter().enumerate() {
                    let scale = scales[o];
                    let mut sum = 0.0f32;

                    for i in 0..in_features {
                        let trit = weight_vec.get(i);
                        let x = input[batch * in_features + i];
                        sum += match trit {
                            Trit::P => x,
                            Trit::N => -x,
                            Trit::Z => 0.0,
                        };
                    }

                    output[batch * out_features + o] = sum * scale;
                }
            }
            black_box(output)
        })
    });

    // Benchmark naive float matmul (for comparison)
    group.bench_function("float_naive_256x256", |b| {
        b.iter(|| {
            let mut output = vec![0.0f32; batch_size * out_features];

            for batch in 0..batch_size {
                for o in 0..out_features {
                    let mut sum = 0.0f32;
                    for i in 0..in_features {
                        sum += input[batch * in_features + i] * weight_float[o * in_features + i];
                    }
                    output[batch * out_features + o] = sum;
                }
            }
            black_box(output)
        })
    });

    group.finish();
}

// ============================================================================
// VSA Gradient Compression Benchmarks
// ============================================================================

fn bench_compress_gradients_vsa(c: &mut Criterion) {
    let mut group = c.benchmark_group("compress_gradients_vsa");

    for &grad_size in &[65536, 262144, 1048576] {
        // 64K, 256K, 1M
        let gradients = create_random_gradients(grad_size, 42);

        group.throughput(Throughput::Bytes((grad_size * 4) as u64));

        for &compression_ratio in &[0.1, 0.01] {
            let compressed_dim =
                ((grad_size as f32 * compression_ratio).ceil() as usize).max(256);

            group.bench_with_input(
                BenchmarkId::new(
                    format!("ratio_{}", compression_ratio),
                    format!("{}K", grad_size / 1024),
                ),
                &gradients,
                |b, gradients| {
                    b.iter(|| {
                        let mut rng = ChaCha8Rng::seed_from_u64(42);
                        let mut compressed = vec![0.0f32; compressed_dim];
                        let scale = 1.0 / (grad_size as f32).sqrt();

                        for &g in gradients.iter() {
                            for c in compressed.iter_mut() {
                                let r: f32 = rng.gen();
                                if r < 0.16 {
                                    *c += g * scale;
                                } else if r < 0.32 {
                                    *c -= g * scale;
                                }
                            }
                        }
                        black_box(compressed)
                    })
                },
            );
        }
    }
    group.finish();
}

fn bench_decompress_gradients_vsa(c: &mut Criterion) {
    let mut group = c.benchmark_group("decompress_gradients_vsa");

    for &original_dim in &[65536, 262144, 1048576] {
        // 64K, 256K, 1M
        let compression_ratio = 0.1;
        let compressed_dim = ((original_dim as f32 * compression_ratio).ceil() as usize).max(256);

        // Create compressed data
        let compressed: Vec<f32> = (0..compressed_dim)
            .map(|i| (i as f32 * 0.001) - 0.5)
            .collect();

        group.throughput(Throughput::Bytes((original_dim * 4) as u64));

        group.bench_with_input(
            BenchmarkId::new("decompress", format!("{}K", original_dim / 1024)),
            &compressed,
            |b, compressed| {
                b.iter(|| {
                    let mut rng = ChaCha8Rng::seed_from_u64(42);
                    let mut gradients = vec![0.0f32; original_dim];
                    let scale = 1.0 / (original_dim as f32).sqrt();

                    for g in gradients.iter_mut() {
                        for &c in compressed.iter() {
                            let r: f32 = rng.gen();
                            if r < 0.16 {
                                *g += c * scale;
                            } else if r < 0.32 {
                                *g -= c * scale;
                            }
                        }
                    }
                    black_box(gradients)
                })
            },
        );
    }
    group.finish();
}

// ============================================================================
// VSA Operations Benchmarks (from trit-vsa)
// ============================================================================

fn bench_vsa_bundle(c: &mut Criterion) {
    let mut group = c.benchmark_group("vsa_bundle");

    for size in [1024, 4096, 16384, 65536].iter() {
        let a = create_random_ternary(*size, 42);
        let b = create_random_ternary(*size, 43);

        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::new("size", size), &(&a, &b), |bench, (a, b)| {
            bench.iter(|| black_box(vsa::bundle(black_box(a), black_box(b))))
        });
    }
    group.finish();
}

fn bench_vsa_bind(c: &mut Criterion) {
    let mut group = c.benchmark_group("vsa_bind");

    for size in [1024, 4096, 16384, 65536].iter() {
        let a = create_random_ternary(*size, 42);
        let b = create_random_ternary(*size, 43);

        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::new("size", size), &(&a, &b), |bench, (a, b)| {
            bench.iter(|| black_box(vsa::bind(black_box(a), black_box(b))))
        });
    }
    group.finish();
}

fn bench_vsa_similarity(c: &mut Criterion) {
    let mut group = c.benchmark_group("vsa_similarity");

    for size in [1024, 4096, 16384, 65536].iter() {
        let a = create_random_ternary(*size, 42);
        let b = create_random_ternary(*size, 43);

        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(
            BenchmarkId::new("cosine", size),
            &(&a, &b),
            |bench, (a, b)| {
                bench.iter(|| black_box(vsa::cosine_similarity(black_box(a), black_box(b))))
            },
        );
    }
    group.finish();
}

fn bench_packed_dot_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("packed_dot_product");

    for size in [1024, 4096, 16384, 65536].iter() {
        let a = create_random_ternary(*size, 42);
        let b = create_random_ternary(*size, 43);

        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::new("size", size), &(&a, &b), |bench, (a, b)| {
            bench.iter(|| black_box(a.dot(black_box(b))))
        });
    }
    group.finish();
}

// ============================================================================
// Criterion Groups
// ============================================================================

criterion_group!(
    name = quantization;
    config = Criterion::default();
    targets = bench_quantize_weights_absmean, bench_quantize_weights_group_sizes
);

criterion_group!(
    name = packing;
    config = Criterion::default();
    targets = bench_pack_ternary_weights, bench_unpack_ternary_weights
);

criterion_group!(
    name = matmul;
    config = Criterion::default();
    targets = bench_ternary_matmul, bench_ternary_vs_float_matmul
);

criterion_group!(
    name = vsa_compression;
    config = Criterion::default();
    targets = bench_compress_gradients_vsa, bench_decompress_gradients_vsa
);

criterion_group!(
    name = vsa_ops;
    config = Criterion::default();
    targets = bench_vsa_bundle, bench_vsa_bind, bench_vsa_similarity, bench_packed_dot_product
);

criterion_main!(quantization, packing, matmul, vsa_compression, vsa_ops);
