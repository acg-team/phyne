#![allow(dead_code)]
/// this file is essentially a workaround for #[cfg(test)] like behaviour for the benchmarks
/// The dev-dependencies are only available in benchmarks or tests
use std::{collections::HashMap, hint::black_box, path::PathBuf, time::Duration};

use criterion::Criterion;

use phylo::alignment::MSA;
use phylo::evolutionary_models::FrequencyOptimisation;
use phylo::optimisers::ModelOptimiser;
use phylo::phylo_info::{PhyloInfo, PhyloInfoBuilder};
use phylo::pip_model::{PIPCost, PIPCostBuilder, PIPModel};
use phylo::random::FakeGenerator;
use phylo::substitution_models::{QMatrix, QMatrixMaker};

pub type BenchPath = &'static str;
pub type SequencePaths = HashMap<&'static str, BenchPath>;

pub const DNA_EASY_5X1000: &str = "data/sim/GTR/gtr.fasta";
pub const DNA_EASY_8X1252: &str = "data/benchmark-datasets/dna/easy/wickd3b_7705.processed.fasta";
pub const DNA_EASY_17X2292: &str = "data/benchmark-datasets/dna/easy/wickd3a_7771.processed.fasta";
pub const DNA_EASY_33X4455: &str =
    "data/benchmark-datasets/dna/easy/tarvd7_ENSG00000126777.nuc.ord.processed.fasta";
pub const DNA_EASY_46X16250: &str =
    "data/benchmark-datasets/dna/easy/jarvd5a_intron_1521.processed.fasta";

pub const DNA_MEDIUM_128X688: &str =
    "data/benchmark-datasets/dna/medium/misod2b_EOG58W9HG.processed.fasta";

pub const AA_EASY_6X97: &str = "data/benchmark-datasets/aa/easy/nagya1_Cluster9992.aln";
pub const AA_EASY_12X73: &str = "data/benchmark-datasets/aa/easy/nagya1_Cluster6386.aln";
pub const AA_EASY_12X445: &str = "data/benchmark-datasets/aa/easy/wicka3_7004.aln";
pub const AA_EASY_14X165: &str = "data/benchmark-datasets/aa/easy/nagya1_Cluster5493.aln";
pub const AA_EASY_45X223: &str = "data/benchmark-datasets/aa/easy/whela7_Gene_0917.aln";
pub const AA_EASY_27X632: &str = "data/benchmark-datasets/aa/easy/boroa6_OG126_gene166.fasta";

pub const AA_MEDIUM_79X106: &str = "data/benchmark-datasets/aa/medium/strua5_gene1339_23221.aln";
pub const AA_MEDIUM_30X86: &str = "data/benchmark-datasets/aa/medium/nagya1_Cluster3439.aln";

pub fn black_box_deterministic_phylo_info(seq_file: impl Into<PathBuf>) -> PhyloInfo<MSA> {
    // Only use the FakeGenerator for deterministic benchmarking
    let mut fake_rng = FakeGenerator::default();
    black_box(
        PhyloInfoBuilder::new(seq_file.into())
            .build_w_rng(&mut fake_rng)
            .expect("sequence file should be able to build phylo info"),
    )
}

pub fn black_box_pip_cost<Model: QMatrix + Default>(
    path: impl Into<PathBuf>,
    freq_opt: FrequencyOptimisation,
) -> PIPCost<Model, MSA> {
    let info = black_box_deterministic_phylo_info(path);
    let pip_cost = PIPCostBuilder::new(PIPModel::<Model>::default(), info)
        .build()
        .expect("failed to build pip cost optimiser");

    // done for a more 'realistic' setup
    let model_optimiser = ModelOptimiser::new(pip_cost, freq_opt);
    black_box(
        model_optimiser
            .run()
            .expect("model optimiser should pass")
            .cost,
    )
}

#[derive(Clone)]
pub struct PIPConfig {
    pub freq_opt: FrequencyOptimisation,
    pub max_iters: usize,
    pub epsilon: f64,
}
pub fn black_box_raw_pip_cost_with_config<Model: QMatrix + QMatrixMaker + Default>(
    seq_path: impl Into<PathBuf>,
) -> (PIPConfig, PIPCost<Model, MSA>) {
    let info = black_box_deterministic_phylo_info(seq_path);

    let cfg = black_box(PIPConfig {
        freq_opt: FrequencyOptimisation::Empirical,
        epsilon: 1e-2,
        max_iters: 5,
    });

    let pip_cost = black_box(PIPCostBuilder::new(PIPModel::<Model>::default(), info))
        .build()
        .expect("failed to build pip cost optimiser");

    (cfg, pip_cost)
}

pub fn setup_suite() -> Criterion {
    Criterion::default()
        .measurement_time(Duration::from_secs(60))
        .with_profiler(pprof::criterion::PProfProfiler::new(
            997, // taken from the flamegraph repo's default
            pprof::criterion::Output::Flamegraph(None),
        ))
}

/// empty on purpose, there are no benches here but the crate still needs
/// to be runnable otherwise criterion crashes
fn main() {}
