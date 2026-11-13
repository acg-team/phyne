use assert_matches::assert_matches;

use crate::alignment::{Alignment, Sequences, MASA, MSA};
use crate::asr::AncestralSequenceReconstruction;
use crate::parsimony_presence_absence::ParsimonyPresenceAbsence;
use crate::phylo_info::PhyloInfo;
use crate::{record_wo_desc as record, tree, Error, Result};

#[test]
fn reconstruct_ancestral_seqs_invalid_internal_ids() {
    let tree = tree!("((A:1.0, B:1.0)I:1.0,C:1.0)I:1.0;");
    let seqs = Sequences::new_unchecked(vec![
        record!("A", b"AA--AAA"),
        record!("B", b"AA-AAAG"),
        record!("C", b"AAAAA-A"),
    ]);
    let phylo = MSA::from_aligned(seqs, tree).unwrap();
    let asr = ParsimonyPresenceAbsence {};

    let err: Result<PhyloInfo<MASA>> = asr.reconstruct_ancestral_seqs(phylo);

    assert_matches!(
        err,
        Err(Error::Tree(msg)) if msg.contains("duplicate node id (I)")
    );
}

#[test]
fn reconstruct_ancestral_seqs_single_node() {
    let tree = tree!("(A:1.0);");
    let seqs = Sequences::new_unchecked(vec![record!("A", b"AA--AAA")]);
    let phylo = MSA::from_aligned_unchecked(seqs, tree);
    let asr = ParsimonyPresenceAbsence {};

    let err: Result<PhyloInfo<MASA>> = asr.reconstruct_ancestral_seqs(phylo);

    assert_matches!(
        err,
        Err(Error::AncestralAlignment(msg)) if msg.contains("cannot reconstruct ancestral sequences for a tree with only one node.")
    );
}
