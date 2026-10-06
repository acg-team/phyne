use std::fmt::Display;

use criterion::{criterion_group, criterion_main, Criterion};

use itertools::Itertools;
use phylo::alignment::MSA;
use phylo::evolutionary_models::FrequencyOptimisation;
use phylo::likelihood::TreeSearchCost;
use phylo::optimisers::{Compatible, MoveOptimiser, SprOptimiser, TopologyOptimiser};
use phylo::pip_model::PIPCost;
use phylo::random::FakeRng;
use phylo::substitution_models::{QMatrix, QMatrixMaker, JC69, WAG};
use phylo::tree::NodeIdx;

mod helpers;
use helpers::{
    black_box_pip_cost, SequencePaths, AA_EASY_12X73, AA_EASY_27X632, AA_EASY_6X97,
    DNA_EASY_17X2292, DNA_EASY_5X1000, DNA_EASY_8X1252,
};

fn single_spr_cycle<C: TreeSearchCost + Clone + Display + Send + Compatible<SprOptimiser>>(
    mut cost_fn: C,
    prune_locations: &[&NodeIdx],
    spr_optimiser: SprOptimiser,
) -> phylo::Result<f64> {
    TopologyOptimiser::<SprOptimiser, C, FakeRng>::fold_improving_moves(
        &mut cost_fn,
        &spr_optimiser,
        f64::MIN,
        prune_locations,
    )
}

fn find_best_regraft_for_single_spr_move<C: TreeSearchCost + Clone + Display + Send>(
    cost_fn: C,
    prune_location: &NodeIdx,
) -> anyhow::Result<f64> {
    let regraft_optimiser = SprOptimiser {};
    let best_regraft =
        regraft_optimiser.best_move_at_location(f64::MIN, &cost_fn, prune_location)?;
    Ok(best_regraft.cost)
}

fn run_single_spr_cycle_for_sizes<Q: QMatrix + QMatrixMaker + Default + Send>(
    paths: &SequencePaths,
    group_name: &'static str,
    criterion: &mut Criterion,
) {
    let mut bench_group = criterion.benchmark_group(format!("SINGLE-SPR-CYCLE {group_name}"));
    let spr_optimiser = SprOptimiser {};
    let mut bench = |id: &str, data: (PIPCost<Q, MSA>, &[&NodeIdx])| {
        bench_group.bench_function(id, |bench| {
            bench.iter_batched(
                // clone because of interior mutability in PIPCost
                || data.clone(),
                |(cost_fn, prune_locations)| {
                    single_spr_cycle(cost_fn, prune_locations, spr_optimiser.clone())
                },
                criterion::BatchSize::SmallInput,
            );
        });
    };
    let spr_optimiser = SprOptimiser {};
    for (key, path) in paths {
        let cost_fn = black_box_pip_cost::<Q>(path, FrequencyOptimisation::Empirical);
        let prune_locations = spr_optimiser
            .move_locations(&cost_fn)
            .copied()
            .collect_vec();
        let prune_locations_ref = prune_locations.iter().collect_vec();
        bench(key, (cost_fn, &prune_locations_ref));
    }
    bench_group.finish();
}

fn run_find_best_regraft_for_single_spr_move<Q: QMatrix + QMatrixMaker + Default + Send>(
    paths: &SequencePaths,
    group_name: &'static str,
    criterion: &mut Criterion,
) {
    let mut bench_group =
        criterion.benchmark_group(format!("SINGLE-SPR-MOVE-FIND-BEST-REGRAFT {group_name}"));
    let mut bench = |id: &str, data: (PIPCost<Q, MSA>, &NodeIdx)| {
        bench_group.bench_function(id, |bench| {
            bench.iter_batched(
                // clone because of interior mutability in PIPCost
                || data.clone(),
                |(cost_fn, prune_locations)| {
                    find_best_regraft_for_single_spr_move(cost_fn, prune_locations)
                },
                criterion::BatchSize::SmallInput,
            );
        });
    };
    for (key, path) in paths {
        let cost_fn = black_box_pip_cost::<Q>(path, FrequencyOptimisation::Empirical);
        let tree = cost_fn.tree();
        // NOTE: regrafting an early preorder node would mean that a long path along the tree stays in tact
        // and less has to be re-calculated overall. We try to benchmark a likely worst case
        // since all parents have to be re-calculated
        let prune_location = *tree
            .postorder()
            .iter()
            .filter(|&n| n != &tree.root)
            .find(|prune| !cost_fn.tree().node(&tree.root).children.contains(prune))
            .expect("tree should have at least one node not a direct child of root");
        bench(key, (cost_fn, &prune_location));
    }
    bench_group.finish();
}

fn spr_dna(criterion: &mut Criterion) {
    let paths = SequencePaths::from([("5X1000", DNA_EASY_5X1000), ("8X1252", DNA_EASY_8X1252)]);
    let long_running_paths = SequencePaths::from([("17X2292", DNA_EASY_17X2292)]);
    run_single_spr_cycle_for_sizes::<JC69>(&paths, "spr DNA", criterion);
    run_find_best_regraft_for_single_spr_move::<JC69>(&paths, "spr DNA", criterion);
    run_find_best_regraft_for_single_spr_move::<JC69>(&long_running_paths, "spr AA", criterion);
}

fn spr_aa(criterion: &mut Criterion) {
    let paths = SequencePaths::from([("6X97", AA_EASY_6X97), ("12X73", AA_EASY_12X73)]);
    let long_running_paths = SequencePaths::from([("27X632", AA_EASY_27X632)]);
    run_single_spr_cycle_for_sizes::<WAG>(&paths, "spr AA", criterion);
    run_find_best_regraft_for_single_spr_move::<WAG>(&paths, "spr AA", criterion);
    run_find_best_regraft_for_single_spr_move::<WAG>(&long_running_paths, "spr AA", criterion);
}

criterion_group! {
name = dna;
config = helpers::setup_suite();
targets = spr_dna
}
criterion_group! {
name = aa;
config = helpers::setup_suite();
targets = spr_aa,
}
criterion_main!(aa, dna);
