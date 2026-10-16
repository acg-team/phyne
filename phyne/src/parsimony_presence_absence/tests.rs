use crate::alignment::{Alignment, AncestralAlignment, Sequences, MASA, MSA};
use crate::asr::AncestralSequenceReconstruction;
use crate::parsimony_presence_absence::ParsimonyPresenceAbsence;
use crate::phylo_info::PhyloInfo;
use crate::tree::NodeIdx::{Internal, Leaf};
use crate::{align, record_wo_desc as record, tree};

#[cfg(test)]
fn aligned_seqs_with_ancestors() -> Sequences {
    aligned_seqs_with_ancestors_subset(&["A0", "B1", "C2", "D3", "E4", "I5", "I6", "I7", "R8"])
}

#[cfg(test)]
fn aligned_seqs_with_ancestors_subset(ids: &[&str]) -> Sequences {
    Sequences::new_unchecked(
        [
            record!("A0", b"AA--AAA"),
            record!("B1", b"--A--AA"),
            record!("C2", b"-A-A-A-"),
            record!("D3", b"--A-A--"),
            record!("E4", b"---A---"),
            record!("I5", b"-XX-XXX"), // parent of A0 and B1
            record!("I6", b"--XXX--"), // parent of D3 and E4
            record!("I7", b"-XXXXX-"), // parent of C2 and I6
            record!("R8", b"-XX-XX-"), // parent of I5 and I7
        ]
        .into_iter()
        .filter(|rec| ids.contains(&rec.id()))
        .collect(),
    )
}

#[test]
fn parsimony_presence_absence() {
    // arrange
    let tree = tree!("((A0:1.0, B1:1.0) I5:1.0,(C2:1.0,(D3:1.0, E4:1.0) I6:1.0) I7:1.0) R8:1.0;");
    let aligned_s = aligned_seqs_with_ancestors_subset(&["A0", "B1", "C2", "D3", "E4"]);
    let all_seqs = aligned_seqs_with_ancestors();
    let phylo = MSA::from_aligned(aligned_s.clone(), tree).unwrap();
    let asr = ParsimonyPresenceAbsence {};

    // act
    let ancestral_phylo: PhyloInfo<MASA> = asr.reconstruct_ancestral_seqs(phylo).unwrap();
    let ancestral_msa_len = ancestral_phylo.msa.len();

    // assert
    for node_idx in ancestral_phylo.tree.postorder() {
        let true_map = &align!(all_seqs
            .record_by_id(&ancestral_phylo.tree.node(node_idx).id)
            .seq());
        let msa_map = match node_idx {
            Leaf(_) => ancestral_phylo.msa.leaf_map(node_idx),
            Internal(_) => ancestral_phylo.msa.ancestral_map(node_idx),
        };
        assert_eq!(msa_map, true_map);
    }
    let leaf_seqs = ancestral_phylo.msa.seqs();
    let true_leaf_seqs =
        aligned_seqs_with_ancestors_subset(&["A0", "B1", "C2", "D3", "E4"]).into_gapless();
    assert_eq!(leaf_seqs.s, true_leaf_seqs.s);
    let ancestral_seqs = ancestral_phylo.msa.ancestral_seqs();
    let true_ancestral_seqs =
        aligned_seqs_with_ancestors_subset(&["I5", "I6", "I7", "R8"]).into_gapless();
    assert_eq!(ancestral_seqs.s, true_ancestral_seqs.s);
    assert_eq!(ancestral_msa_len, 7);
    assert_eq!(ancestral_phylo.msa.seq_count(), 5);
}
