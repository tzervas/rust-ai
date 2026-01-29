//! Benchmarks for ternary operations.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use trit_vsa::{PackedTritVec, SparseVec, Trit};

fn bench_packed_dot(c: &mut Criterion) {
    let mut group = c.benchmark_group("packed_dot");

    for size in [64, 256, 1024, 4096, 16384].iter() {
        let mut a = PackedTritVec::new(*size);
        let mut b = PackedTritVec::new(*size);

        // Fill with pattern
        for i in 0..*size {
            a.set(
                i,
                match i % 3 {
                    0 => Trit::P,
                    1 => Trit::N,
                    _ => Trit::Z,
                },
            );
            b.set(
                i,
                match i % 5 {
                    0 | 1 => Trit::P,
                    2 | 3 => Trit::N,
                    _ => Trit::Z,
                },
            );
        }

        group.bench_with_input(BenchmarkId::new("scalar", size), size, |bench, _| {
            bench.iter(|| black_box(a.dot(&b)))
        });
    }

    group.finish();
}

fn bench_sparse_dot(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_dot");

    // Test with varying sparsity
    for (size, nonzero) in [(10000, 100), (10000, 500), (10000, 1000)].iter() {
        let mut a = SparseVec::new(*size);
        let mut b = SparseVec::new(*size);

        // Set sparse values
        for i in 0..*nonzero {
            a.set(i * (size / nonzero), Trit::P);
            b.set(
                i * (size / nonzero),
                if i % 2 == 0 { Trit::P } else { Trit::N },
            );
        }

        let label = format!("size={}_nonzero={}", size, nonzero);
        group.bench_with_input(BenchmarkId::new("sparse", &label), &(), |bench, _| {
            bench.iter(|| black_box(a.dot(&b)))
        });
    }

    group.finish();
}

fn bench_bundle(c: &mut Criterion) {
    let mut group = c.benchmark_group("bundle");

    for size in [256, 1024, 4096].iter() {
        let mut a = PackedTritVec::new(*size);
        let mut b = PackedTritVec::new(*size);

        for i in 0..*size / 2 {
            a.set(i, Trit::P);
            b.set(i, Trit::P);
        }

        group.bench_with_input(BenchmarkId::new("two_vectors", size), size, |bench, _| {
            bench.iter(|| black_box(trit_vsa::vsa::bundle(&a, &b)))
        });
    }

    group.finish();
}

fn bench_bind(c: &mut Criterion) {
    let mut group = c.benchmark_group("bind");

    for size in [256, 1024, 4096].iter() {
        let mut a = PackedTritVec::new(*size);
        let mut b = PackedTritVec::new(*size);

        for i in 0..*size / 2 {
            a.set(i, Trit::P);
            b.set(i, Trit::N);
        }

        group.bench_with_input(BenchmarkId::new("bind_unbind", size), size, |bench, _| {
            bench.iter(|| {
                let bound = trit_vsa::vsa::bind(&a, &b);
                black_box(trit_vsa::vsa::unbind(&bound, &b))
            })
        });
    }

    group.finish();
}

fn bench_cosine_similarity(c: &mut Criterion) {
    let mut group = c.benchmark_group("cosine_similarity");

    for size in [256, 1024, 4096].iter() {
        let mut a = PackedTritVec::new(*size);
        let mut b = PackedTritVec::new(*size);

        for i in 0..*size / 2 {
            a.set(i, Trit::P);
            b.set(i, if i % 2 == 0 { Trit::P } else { Trit::N });
        }

        group.bench_with_input(BenchmarkId::new("packed", size), size, |bench, _| {
            bench.iter(|| black_box(trit_vsa::vsa::cosine_similarity(&a, &b)))
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_packed_dot,
    bench_sparse_dot,
    bench_bundle,
    bench_bind,
    bench_cosine_similarity
);
criterion_main!(benches);
