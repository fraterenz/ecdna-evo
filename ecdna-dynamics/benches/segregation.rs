use criterion::criterion_group;
use criterion::criterion_main;
use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::Throughput;
use ecdna_dynamics::segregation::BinomialNoNMinusSegregation;
use ecdna_dynamics::segregation::BinomialSegregation;
use ecdna_dynamics::segregation::Segregation;
use rand::SeedableRng;
use rand_pcg::Pcg64Mcg;

fn random_segregation_no_nminus(c: &mut Criterion) {
    let mut group = c.benchmark_group("random_segregation_no_nminus");
    let data = (1..10).map(|ele| 2_usize.pow(ele)).collect::<Vec<usize>>();
    let mut rng = Pcg64Mcg::seed_from_u64(26);
    let segregation = Segregation::Random(
        BinomialNoNMinusSegregation(BinomialSegregation).into(),
    );

    for ecdna_copies in data.iter() {
        group.throughput(Throughput::Bytes(*ecdna_copies as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(ecdna_copies),
            ecdna_copies,
            |b, &ecdna_copies| {
                b.iter(|| {
                    segregation.segregate(ecdna_copies as u16, &mut rng)
                });
            },
        );
    }

    group.finish();
}

fn random_segregation(c: &mut Criterion) {
    let mut group = c.benchmark_group("random_segregation");
    let data = (1..10).map(|ele| 2_usize.pow(ele)).collect::<Vec<usize>>();
    let mut rng = Pcg64Mcg::seed_from_u64(26);
    let segregation = Segregation::Random(BinomialSegregation.into());

    for ecdna_copies in data.iter() {
        group.throughput(Throughput::Bytes(*ecdna_copies as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(ecdna_copies),
            ecdna_copies,
            |b, &ecdna_copies| {
                b.iter(|| {
                    segregation.segregate(ecdna_copies as u16, &mut rng)
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, random_segregation_no_nminus, random_segregation);
criterion_main!(benches);
