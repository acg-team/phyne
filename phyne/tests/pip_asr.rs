use std::path::Path;

use phyne::alignment::{AncestralAlignment, MASA, MSA};
use phyne::asr::AncestralSequenceReconstruction;
use phyne::phylo_info::{PhyloInfo, PhyloInfoBuilder};
use phyne::pip_model::{PIPIndels, PIPModel};
use phyne::substitution_models::JC69;
use phyne::Result;

#[test]
fn homology_against_arpip() {
    // Data from the ARPIP paper, javapip simulation with mu = 0.1
    // sim-0_msa_new.fasta as the input msa
    // sim-0_new.newick as the input tree (changed node ids from "x" -> "i_x")
    // sim-0_inferred.fasta is the result for comparison (concatenated alignment from input and ARPIP reconstruction)
    // Note: this dataset has no insertions that are more likely to be higher than the MRCA.
    let fldr = Path::new("./data/pip/arpip");
    let alignment_file = fldr.join("sim-0_msa_new.fasta");
    let tree_file = fldr.join("sim-0_new.newick");
    let inferred_file = fldr.join("sim-0_inferred.fasta");

    let info: PhyloInfo<MSA> = PhyloInfoBuilder::with_attrs(alignment_file, tree_file.clone())
        .build()
        .unwrap();

    let asr = PIPIndels::<_, MSA, MASA>::new(
        PIPModel::<JC69>::with_default_substitution(&[1.0, 0.1]).unwrap(),
    );

    let result: Result<PhyloInfo<MASA>> = asr.reconstruct_ancestral_seqs(info);
    assert!(result.is_ok());
    let reconstruction = result.unwrap();

    let reconstruction_arpip = PhyloInfoBuilder::with_attrs(inferred_file, tree_file)
        .build_with_ancestors()
        .unwrap();

    for node in reconstruction.tree.internals() {
        let map = reconstruction.msa.ancestral_map(&node.idx);
        let map_arpip = reconstruction_arpip.msa.ancestral_map(&node.idx);
        assert_eq!(map, map_arpip, "Ancestral map mismatch at node {}", node.id);
    }
}
