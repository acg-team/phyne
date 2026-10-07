use assert_matches::assert_matches;

use crate::alignment::{Alignment, Sequences, MASA, MSA};
use crate::asr::AncestralSequenceReconstruction;
use crate::parsimony_presence_absence::ParsimonyPresenceAbsence;
use crate::{record, tree, Error};

#[test]
fn reconstruct_ancestral_seqs_n_seqs_not_same_as_leaves() {
    let tree = tree!("root:1.0;");
    let seqs =
        Sequences::new_unchecked(vec![record!("root", Some("seq with 4 nucls"), b"AA--AAA")]);
    let msa = MSA::from_aligned(seqs, &tree).unwrap();
    let wrong_tree =
        tree!("((A0:1.0, B1:1.0) I5:1.0,(C2:1.0,(D3:1.0, E4:1.0) I9:1.0) I7:1.0) I8:1.0;");
    let asr = ParsimonyPresenceAbsence {};

    let err = AncestralSequenceReconstruction::<MSA, MASA>::reconstruct_ancestral_seqs(
        &asr,
        &msa,
        &wrong_tree,
    );

    assert_matches!(
        err,
        Err(Error::AncestralAlignment(msg)) if msg.contains("but tree has")
    );
}
