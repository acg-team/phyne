use std::fmt::Display;
use std::result::Result::Ok;
use std::time::Duration;

use anyhow::Result;
use criterion::{criterion_group, criterion_main, Criterion};
use log::info;

use phylo::alignment::MSA;
use phylo::evolutionary_models::FrequencyOptimisation;
use phylo::likelihood::{ModelSearchCost, TreeSearchCost};
use phylo::optimisers::{Compatible, ModelOptimiser, SprOptimiser, TopologyOptimiser};
use phylo::pip_model::PIPCost;
use phylo::random::FakeGenerator;
use phylo::substitution_models::{QMatrix, HKY, WAG};
use phylo::tree::Tree;

mod helpers;
use helpers::{
    black_box_raw_pip_cost_with_config, PIPConfig, SequencePaths, AA_EASY_12X73, AA_EASY_6X97,
    DNA_EASY_5X1000, DNA_EASY_8X1252,
};

/// NOTE: This is essentially a snapshot of the JATI binary at the time of creation
/// ANY changes must be manually tracked to accurately measure real-world usage.
///
/// TODO: expose this as part of the rust-phylo library
fn run_optimisation(
    cost: impl TreeSearchCost + ModelSearchCost + Display + Clone + Send + Compatible<SprOptimiser>,
    freq_opt: FrequencyOptimisation,
    max_iterations: usize,
    epsilon: f64,
) -> Result<(f64, Tree)> {
    // Only use the FakeGenerator for deterministic benchmarking
    let mut fake_rng = FakeGenerator::default();
    let mut cost = cost;
    let mut prev_cost = f64::NEG_INFINITY;
    let mut final_cost = TreeSearchCost::cost(&cost);

    let mut iterations = 0;
    while final_cost - prev_cost > epsilon && iterations < max_iterations {
        iterations += 1;
        info!("Iteration: {iterations}");

        prev_cost = final_cost;
        let model_optimiser = ModelOptimiser::new(cost, freq_opt);
        let o = TopologyOptimiser::new(model_optimiser.run()?.cost, SprOptimiser {}, &mut fake_rng)
            .run()
            .unwrap();
        final_cost = o.final_cost;
        cost = o.cost;
    }
    Ok((final_cost, cost.tree().clone()))
}

fn run_for_sizes<Q: QMatrix + Default + Send>(
    paths: &SequencePaths,
    group_name: &'static str,
    criterion: &mut Criterion,
) {
    let mut bench_group = criterion.benchmark_group(group_name);
    let mut bench = |id: &str, data: (PIPConfig, PIPCost<Q, MSA>)| {
        bench_group.bench_function(id, |bench| {
            bench.iter_batched(
                // clone because of interior mutability in PIPCost
                || data.clone(),
                |data| run_optimisation(data.1, data.0.freq_opt, data.0.max_iters, data.0.epsilon),
                criterion::BatchSize::SmallInput,
            );
        });
    };
    for (key, path) in paths {
        let data = black_box_raw_pip_cost_with_config::<Q>(path);
        bench(key, data);
    }
    bench_group.finish();
}

fn pip_inference_dna(criterion: &mut Criterion) {
    let paths = SequencePaths::from([("5X1000", DNA_EASY_5X1000), ("8X1252", DNA_EASY_8X1252)]);
    run_for_sizes::<HKY>(&paths, "Tree-from-MSA DNA", criterion);
}

fn pip_inference_aa(criterion: &mut Criterion) {
    let paths = SequencePaths::from([("6X97", AA_EASY_6X97), ("12X73", AA_EASY_12X73)]);
    run_for_sizes::<WAG>(&paths, "Tree-from-MSA AA", criterion);
}

criterion_group! {
name = dna;
config = helpers::setup_suite().measurement_time(Duration::from_secs(120)).sample_size(10);
targets = pip_inference_dna
}
criterion_group! {
name = aa;
config = helpers::setup_suite().measurement_time(Duration::from_secs(120)).sample_size(10);
targets = pip_inference_aa,
}
criterion_main!(aa, dna);
