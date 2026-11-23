use criterion::{criterion_group, criterion_main, Criterion};
use phyne::evolutionary_models::EvoModel;
use phyne::substitution_models::{SubstModel, WAG};
use std::hint::black_box;

fn decomposition_benchmark(c: &mut Criterion) {
    let model = SubstModel::<WAG>::default();

    //let times = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0];
    let t = 0.05;
    c.bench_function("p_decomp", |b| b.iter(|| model.p(black_box(t))));
}
criterion_group!(bench_decomp, decomposition_benchmark);
criterion_main!(bench_decomp);
