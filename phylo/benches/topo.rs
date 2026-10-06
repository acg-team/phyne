use std::fmt::Display;
use std::num::NonZero;

use criterion::{criterion_group, criterion_main, Criterion};

use phylo::alignment::MSA;
use phylo::evolutionary_models::FrequencyOptimisation;
use phylo::likelihood::TreeSearchCost;
use phylo::optimisers::{Compatible, SprOptimiser, StopCondition, TopologyOptimiser};
use phylo::pip_model::PIPCost;
use phylo::random::FakeGenerator;
use phylo::substitution_models::{QMatrix, QMatrixMaker, JC69, WAG};

mod helpers;
use helpers::{
    black_box_pip_cost, SequencePaths, AA_EASY_12X73, AA_EASY_6X97, DNA_EASY_5X1000,
    DNA_EASY_8X1252,
};

fn run_fixed_iter_topo<C: TreeSearchCost + Clone + Display + Send + Compatible<SprOptimiser>>(
    cost: C,
) -> anyhow::Result<f64> {
    // Only use the FakeGenerator for deterministic benchmarking
    let mut fake_rng = FakeGenerator::default();
    let topo_opt = TopologyOptimiser::with_stop_condition(
        cost,
        SprOptimiser {},
        &mut fake_rng,
        StopCondition::fixed_iter(NonZero::new(3).unwrap()),
    );
    Ok(topo_opt.run()?.final_cost)
}

fn run_simulated_topo_for_sizes<Q: QMatrix + QMatrixMaker + Default + Send>(
    paths: &SequencePaths,
    group_name: &'static str,
    criterion: &mut Criterion,
) {
    let mut bench_group =
        criterion.benchmark_group(format!("SIMULATED-TOPO-OPTIMISER {group_name}"));
    let mut bench = |id: &str, data: PIPCost<Q, MSA>| {
        bench_group.bench_function(id, |bench| {
            bench.iter_batched(
                // clone because of interior mutability in PIPCost
                || data.clone(),
                run_fixed_iter_topo,
                criterion::BatchSize::SmallInput,
            );
        });
    };
    for (key, path) in paths {
        let cost = black_box_pip_cost::<Q>(path, FrequencyOptimisation::Empirical);
        bench(key, cost);
    }
    bench_group.finish();
}

fn topo_dna(criterion: &mut Criterion) {
    let paths = SequencePaths::from([("5X1000", DNA_EASY_5X1000), ("8X1252", DNA_EASY_8X1252)]);
    run_simulated_topo_for_sizes::<JC69>(&paths, "topology optimiser DNA", criterion);
}

fn topo_aa(criterion: &mut Criterion) {
    let paths = SequencePaths::from([("6X97", AA_EASY_6X97), ("12X73", AA_EASY_12X73)]);
    run_simulated_topo_for_sizes::<WAG>(&paths, "topology optimiser AA", criterion);
}

criterion_group! {
name = dna;
config = helpers::setup_suite().sample_size(15);
targets = topo_dna
}
criterion_group! {
name = aa;
config = helpers::setup_suite().sample_size(15);
targets = topo_aa,
}
criterion_main!(aa, dna);
