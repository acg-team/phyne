use crate::alignment::{Alignment, AncestralAlignment, Sequences, MASA, MSA};
use crate::asr::AncestralSequenceReconstruction;
use crate::parsimony_presence_absence::ParsimonyPresenceAbsence;
use crate::tree::NodeIdx::{Internal, Leaf};
use crate::{align, record, tree};

#[cfg(test)]
fn aligned_seqs_with_ancestors() -> Sequences {
    aligned_seqs_with_ancestors_subset(&["A0", "B1", "C2", "D3", "E4", "I5", "I9", "I7", "I8"])
}

#[cfg(test)]
fn aligned_seqs_with_ancestors_subset(ids: &[&str]) -> Sequences {
    Sequences::new_unchecked(
        [
            record!("A0", Some("seq A0 with 4 nucls"), b"AA--AAA"),
            record!("B1", Some("seq B1 with 3 nucls"), b"--A--AA"),
            record!("C2", Some("seq C2 with 3 nucls"), b"-A-A-A-"),
            record!("D3", Some("seq D3 with 2 nucls"), b"--A-A--"),
            record!("E4", Some("seq E4 with 1 nucls"), b"---A---"),
            record!("I5", None, b"-XX-XXX"),
            record!("I9", None, b"--XXX--"),
            record!("I7", None, b"-XXXXX-"),
            record!("I8", None, b"-XX-XX-"),
        ]
        .into_iter()
        .filter(|rec| ids.contains(&rec.id()))
        .collect(),
    )
}

#[test]
fn parsimony_presence_absence() {
    // arrange
    let tree = tree!("((A0:1.0, B1:1.0) I5:1.0,(C2:1.0,(D3:1.0, E4:1.0) I9:1.0) I7:1.0) I8:1.0;");
    let aligned_s = aligned_seqs_with_ancestors_subset(&["A0", "B1", "C2", "D3", "E4"]);
    let all_seqs = aligned_seqs_with_ancestors();
    let msa = MSA::from_aligned(aligned_s.clone(), &tree).unwrap();
    let asr = ParsimonyPresenceAbsence {};

    // act
    let ancestral_msa: MASA = asr.reconstruct_ancestral_seqs(&msa, &tree).unwrap();
    let ancestral_msa_len = msa.len();

    // assert
    for node_idx in tree.postorder() {
        let true_map = &align!(all_seqs.record_by_id(&tree.node(node_idx).id).seq());
        let msa_map = match node_idx {
            Leaf(_) => ancestral_msa.leaf_map(node_idx),
            Internal(_) => ancestral_msa.ancestral_map(node_idx),
        };
        assert_eq!(msa_map, true_map);
    }
    let leaf_seqs = ancestral_msa.seqs();
    let true_leaf_seqs =
        aligned_seqs_with_ancestors_subset(&["A0", "B1", "C2", "D3", "E4"]).into_gapless();
    assert_eq!(leaf_seqs.s, true_leaf_seqs.s);
    let ancestral_seqs = ancestral_msa.ancestral_seqs();
    let true_ancestral_seqs =
        aligned_seqs_with_ancestors_subset(&["I5", "I9", "I7", "I8"]).into_gapless();
    assert_eq!(ancestral_seqs.s, true_ancestral_seqs.s);
    assert_eq!(ancestral_msa_len, 7);
    assert_eq!(ancestral_msa.seq_count(), 5);
}
