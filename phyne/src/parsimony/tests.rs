use assert_matches::assert_matches;

use crate::alignment::{Aligner, Alignment, Sequences, MSA};
use crate::parsimony::{GapCost, ParsimonyAligner, SimpleScoring, SiteFlag::*};
use crate::{align, record_wo_desc as rec, site, tree};
use crate::{Error, Result};

#[test]
fn align_two_first_outcome() {
    let mismatch = 1.0;
    let gap = GapCost::new(2.0, 0.5);
    let scoring = SimpleScoring::new(mismatch, gap);

    // Leaf sequence representation
    let x_leaf = [
        site!(b"A", NoGap),
        site!(b"A", NoGap),
        site!(b"C", NoGap),
        site!(b"T", NoGap),
    ];
    let y_leaf = [site!(b"A", NoGap), site!(b"C", NoGap)];

    let aligner = ParsimonyAligner::new(scoring);
    let (_info, alignment, score) = aligner.pairwise_align(&x_leaf, 1.0, &y_leaf, 1.0, |l| l - 1);

    assert_eq!(score, 3.5);
    assert_eq!(alignment.len(), 4);
    assert_eq!(alignment.map_x(), &align!(b"0123"));
    assert_eq!(alignment.map_y(), &align!(b"01--"));
}

#[test]
fn align_two_second_outcome() {
    let mismatch = 1.0;
    let gap = GapCost::new(2.0, 0.5);

    let scoring = SimpleScoring::new(mismatch, gap);

    let x_leaf = [
        site!(b"A", NoGap),
        site!(b"A", NoGap),
        site!(b"C", NoGap),
        site!(b"T", NoGap),
    ];
    let y_leaf = [site!(b"A", NoGap), site!(b"C", NoGap)];

    let aligner = ParsimonyAligner::new(scoring);
    let (_info, alignment, score) = aligner.pairwise_align(&x_leaf, 1.0, &y_leaf, 1.0, |_| 0);

    assert_eq!(score, 3.5);
    assert_eq!(alignment.len(), 4);
    assert_eq!(alignment.map_x(), &align!(b"0123"));
    assert_eq!(alignment.map_y(), &align!(b"0--1"));
}

#[test]
fn align_two_on_tree() {
    let mismatch = 1.0;
    let gap = GapCost::new(2.0, 0.5);
    let seqs = Sequences::new_unchecked(vec![rec!("A", b"AACT"), rec!("B", b"AC")]);
    let tree = tree!("(A:1.0, B:1.0):0.0;");
    let scoring = SimpleScoring::new(mismatch, gap);

    let aligner = ParsimonyAligner::new(scoring);
    let (alignment, score): (MSA, _) = aligner.align_with_scores(&seqs, &tree);

    assert_eq!(score[Into::<usize>::into(tree.root)], 3.5);
    let alignment = &alignment.internal_alignments()[&tree.root];
    assert_eq!(alignment.len(), 4);
}

#[test]
fn internal_alignment_first_outcome() {
    let mismatch = 1.0;
    let gap = GapCost::new(2.0, 0.5);
    let scoring = SimpleScoring::new(mismatch, gap);

    let x_leaf = [
        site!(b"A", NoGap),
        site!(b"CA", NoGap),
        site!(b"C", GapOpen),
        site!(b"T", GapOpen),
    ];

    let y_leaf = [site!(b"G", GapOpen), site!(b"A", NoGap)];

    let aligner = ParsimonyAligner::new(scoring);
    let (_info, alignment, score) = aligner.pairwise_align(&x_leaf, 1.0, &y_leaf, 1.0, |l| l - 1);

    assert_eq!(score, 1.0);
    assert_eq!(alignment.map_x(), &align!(b"0123"));
    assert_eq!(alignment.map_y(), &align!(b"01--"));
}

#[test]
fn internal_alignment_second_outcome() {
    let mismatch = 1.0;
    let gap = GapCost::new(2.0, 0.5);
    let scoring = SimpleScoring::new(mismatch, gap);

    let x_leaf = [
        site!(b"A", NoGap),
        site!(b"A", GapOpen),
        site!(b"C", GapOpen),
        site!(b"TC", NoGap),
    ];

    let y_leaf = [site!(b"G", GapOpen), site!(b"A", NoGap)];

    let aligner = ParsimonyAligner::new(scoring);
    let (_info, alignment, score) = aligner.pairwise_align(&x_leaf, 1.0, &y_leaf, 1.0, |_| 0);

    assert_eq!(score, 2.0);
    assert_eq!(alignment.map_x, align!(b"0123"));
    assert_eq!(alignment.map_y, align!(b"0--1"));
}

#[test]
fn internal_alignment_third_outcome() {
    let mismatch = 1.0;
    let gap = GapCost::new(2.0, 0.5);
    let scoring = SimpleScoring::new(mismatch, gap);

    let x_leaf = [
        site!(b"A", NoGap),
        site!(b"A", GapOpen),
        site!(b"C", GapOpen),
        site!(b"TC", NoGap),
    ];

    let y_leaf = [site!(b"G", GapOpen), site!(b"A", NoGap)];

    let aligner = ParsimonyAligner::new(scoring);
    let (_info, alignment, score) = aligner.pairwise_align(&x_leaf, 1.0, &y_leaf, 1.0, |l| l - 1);

    assert_eq!(score, 2.0);
    assert_eq!(alignment.map_x, align!(b"-0123"));
    assert_eq!(alignment.map_y, align!(b"01---"));
}

#[test]
fn align_four_on_tree() {
    let mismatch = 1.0;
    let gap = GapCost::new(2.0, 0.5);

    let seqs = Sequences::new_unchecked(vec![
        rec!("A", b"AACT"),
        rec!("B", b"AC"),
        rec!("C", b"A"),
        rec!("D", b"GA"),
    ]);

    let tree = tree!("((A:1.0, B:1.0):1.0, (C:1.0, D:1.0):1.0);");
    let scoring = SimpleScoring::new(mismatch, gap);

    let aligner = ParsimonyAligner::new(scoring);
    let (alignment, score): (MSA, _) = aligner.align_with_scores(&seqs, &tree);
    // first cherry
    let idx = &tree.by_id("A").parent.unwrap();
    assert_eq!(score[usize::from(idx)], 3.5);
    assert_eq!(alignment.internal_alignments()[idx].len(), 4);

    // second cherry
    let idx = &tree.by_id("C").parent.unwrap();
    assert_eq!(score[usize::from(idx)], 2.0);
    assert_eq!(alignment.internal_alignments()[idx].len(), 2);

    // root, three possible alignments
    let idx = &tree.root;
    assert!(score[usize::from(idx)] == 1.0 || score[usize::from(idx)] == 2.0);
    if score[2] == 1.0 {
        assert_eq!(alignment.internal_alignments()[idx].len(), 4);
    } else {
        assert!(
            alignment.internal_alignments()[idx].len() == 4
                || alignment.internal_alignments()[idx].len() == 5
        );
    }
}

#[test]
fn parsimony_site_debug() {
    assert!(format!("{:?}", site!(b"T", GapOpen)).contains("GapOpen"));
    assert!(format!("{:?}", site!(b"C", GapExt)).contains("GapExt"));
    assert!(format!("{:?}", site!(b"A", NoGap)).contains("NoGap"));
    assert!(format!("{:?}", site!(b"T", GapOpen)).contains("GapOpen"));
    assert!(format!("{:?}", site!(b"G", NoGap)).contains("NoGap"));
    assert!(format!("{:?}", site!(b"-", GapExt)).contains("GapExt"));
}

#[test]
fn try_aligning_aligned_sequences() {
    let scoring = SimpleScoring::new(1.0, GapCost::new(2.0, 0.5));

    let seqs = Sequences::new_unchecked(vec![
        rec!("A", b"A--ACT"),
        rec!("B", b"A--C--"),
        rec!("C", b"AA--CT"),
    ]);
    let tree = tree!("((A:1.0, B:1.0):1.0, C:2.0);");
    let aligner = ParsimonyAligner::new(scoring);
    let alignment: Result<MSA> = aligner.align(&seqs, &tree);

    assert_matches!(
        alignment,
        Err(Error::Alignment(msg)) if msg.contains("sequences must not be already aligned")
    );
}
