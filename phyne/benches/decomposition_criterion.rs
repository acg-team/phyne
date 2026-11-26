use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use phyne::evolutionary_models::EvoModel;
use phyne::substitution_models::{SubstModel, WAG};
use std::hint::black_box;

// cargo bench --bench decomposition_criterion to run
fn decomposition_benchmark(c: &mut Criterion) {
    let model = SubstModel::<WAG>::default();
    let times = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0];

    let mut group = c.benchmark_group("p_decomp");
    for t in times {
        group.bench_with_input(BenchmarkId::from_parameter(t), &t, |b, &t| {
            b.iter(|| model.p(black_box(t)));
        });
    }
}
criterion_group!(bench_decomp, decomposition_benchmark);
criterion_main!(bench_decomp);
