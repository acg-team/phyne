use criterion::{black_box, criterion_group, criterion_main, Criterion};
use std::path::Path;

use phyne::alignment::{MASA, MSA};
use phyne::asr::AncestralSequenceReconstruction;
use phyne::phylo_info::{PhyloInfo, PhyloInfoBuilder};
use phyne::pip_model::{PIPIndels, PIPModel};
use phyne::substitution_models::JC69;

fn pip_asr_benchmark(c: &mut Criterion) {
    c.bench_function("pip_asr_homology_arpip", |b| {
        b.iter(|| {
            let fldr = Path::new("./data/pip/arpip");
            let alignment_file = fldr.join("sim-0_msa_new.fasta");
            let tree_file = fldr.join("sim-0_new.newick");

            let info: PhyloInfo<MSA> =
                PhyloInfoBuilder::with_attrs(black_box(alignment_file), &tree_file)
                    .build()
                    .unwrap();

            let asr =
                PIPIndels::new(&PIPModel::<JC69>::with_default_substitution(&[1.0, 0.1]).unwrap());
            let _reconstruction: PhyloInfo<MASA> = asr.reconstruct_ancestral_seqs(info).unwrap();
        });
    });
}

criterion_group!(benches, pip_asr_benchmark);
criterion_main!(benches);
