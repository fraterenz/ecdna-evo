use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ecdna_sim::rate::exprand;
use rand::SeedableRng;
use rand_pcg::Pcg64Mcg;

fn exprand_benchmark(c: &mut Criterion) {
    let mut rng = Pcg64Mcg::seed_from_u64(1u64);
    let mut group = c.benchmark_group("exprand_benchmark");
    group.bench_function("exprand 0.5", |b| {
        b.iter(|| exprand(black_box(0.5), black_box(&mut rng)))
    });
    group.finish();
}

criterion_group!(benches, exprand_benchmark);
criterion_main!(benches);
