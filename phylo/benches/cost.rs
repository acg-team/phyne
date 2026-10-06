use criterion::{criterion_group, criterion_main, Criterion};

use phylo::alignment::MSA;
use phylo::evolutionary_models::FrequencyOptimisation;
use phylo::likelihood::ModelSearchCost;
use phylo::pip_model::PIPCost;
use phylo::substitution_models::{QMatrix, JC69, WAG};

mod helpers;
use helpers::{
    black_box_pip_cost, SequencePaths, AA_EASY_12X73, AA_EASY_14X165, AA_EASY_27X632,
    AA_EASY_45X223, AA_EASY_6X97, AA_MEDIUM_79X106, DNA_EASY_17X2292, DNA_EASY_33X4455,
    DNA_EASY_46X16250, DNA_EASY_5X1000, DNA_EASY_8X1252, DNA_MEDIUM_128X688,
};

fn run_for_sizes<Q: QMatrix + Default>(
    paths: &SequencePaths,
    group_name: &'static str,
    criterion: &mut Criterion,
) {
    let mut bench_group = criterion.benchmark_group(group_name);
    let mut bench = |id: &str, data: PIPCost<Q, MSA>| {
        bench_group.bench_function(id, |bench| {
            bench.iter_batched(
                // clone because of interior mutability
                || data.clone(),
                |cost_fn| cost_fn.cost(),
                criterion::BatchSize::SmallInput,
            );
        });
    };
    for (key, path) in paths {
        let data = black_box_pip_cost::<Q>(path, FrequencyOptimisation::Empirical);
        bench(key, data);
    }
    bench_group.finish();
}

fn pip_cost_dna_easy(criterion: &mut Criterion) {
    let paths = SequencePaths::from([
        ("5X1000", DNA_EASY_5X1000),
        ("8X1252", DNA_EASY_8X1252),
        ("17X2292", DNA_EASY_17X2292),
        ("33X4455", DNA_EASY_33X4455),
        ("46X16250", DNA_EASY_46X16250),
        ("128X688", DNA_MEDIUM_128X688),
    ]);
    run_for_sizes::<JC69>(&paths, "PIP Cost DNA", criterion);
}

fn pip_cost_aa_easy(criterion: &mut Criterion) {
    let paths = SequencePaths::from([
        ("6X97", AA_EASY_6X97),
        ("12X73", AA_EASY_12X73),
        ("14X165", AA_EASY_14X165),
        ("27X632", AA_EASY_27X632),
        ("45X223", AA_EASY_45X223),
        ("79X106", AA_MEDIUM_79X106),
    ]);
    run_for_sizes::<WAG>(&paths, "PIP Cost AA", criterion);
}

criterion_group! {
name = pip_cost_dna;
config = helpers::setup_suite();
targets = pip_cost_dna_easy,
}
criterion_group! {
name = pip_cost_aa;
config = helpers::setup_suite();
targets = pip_cost_aa_easy,
}
criterion_main!(pip_cost_dna, pip_cost_aa);
