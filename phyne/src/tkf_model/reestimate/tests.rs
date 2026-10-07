use approx::assert_relative_eq;
use assert_matches::assert_matches;
use fixedbitset::FixedBitSet;

use crate::alignment::{AncestralAlignment, Sequences, MASA};
use crate::alphabets::Alphabet;
use crate::phylo_info::PhyloInfo;
use crate::random::FakeGenerator;
use crate::tkf_model::reestimate::mapping_from_node_seq;
use crate::tkf_model::tests::setup_test_phylo;
use crate::tkf_model::{EdgeSeqsReestimator, TKF92IndelCostBuilder};
use crate::{record_wo_desc as record, tree, Error};

#[test]
fn tkf_reestimate_without_choice() {
    let tree = tree!("(((A1:2.0,(B2:1.0, C3:1.2)I4:2.0)I5:0.3,D6:2.0)R7:1.0);");
    // The alignment is designed such that for every site the re-estimation will have no choice but
    // to keep the current states, since it must conform to Dollo's principle.
    let msa = MASA::from_aligned_with_ancestral(
        Sequences::new_unchecked(vec![
            record!("A1", b"-----N-----"),
            record!("B2", b"NN----NN-NN"),
            record!("C3", b"--NNN---NNN"),
            record!("I4", b"NNNNN----NN"),
            record!("I5", b"NNNNNN---NN"),
            record!("D6", b"NNNNNN---NN"),
            record!("R7", b"NNNNNN---NN"),
        ]),
        &tree,
    )
    .unwrap();
    let phylo = PhyloInfo { msa, tree };
    let mut cost = TKF92IndelCostBuilder::new(&[0.4, 0.5, 0.8], phylo)
        .build()
        .unwrap();
    let logl = cost.clone().logl();
    println!("Initial logl: {}", logl);
    let rng = &mut FakeGenerator::default();
    let v2_idx = cost.phylo.tree.by_id("I4").idx;
    let mut reestimator = EdgeSeqsReestimator::new(&mut cost, rng);
    assert_relative_eq!(reestimator.reestimate(&v2_idx).unwrap(), logl);
    // some tmp valid are set to false, all valid_for_reestimation should be true
    assert!(!reestimator.cost.model_info.borrow().valid.is_full());
    assert!(reestimator
        .cost
        .model_info
        .borrow()
        .valid_for_reestimation
        .is_full());
}

#[test]
fn tkf_reestimation_fails_for_root() {
    let phylo = setup_test_phylo(Alphabet::dna());
    let mut cost = TKF92IndelCostBuilder::new(&[0.4, 0.5, 0.8], phylo)
        .build()
        .unwrap();
    let rng = &mut FakeGenerator::default();
    let mut reestimator = EdgeSeqsReestimator::new(&mut cost, rng);
    let root_idx = reestimator.cost.phylo.tree.root;

    let err = reestimator.reestimate(&root_idx);
    assert_matches!(err, Err(Error::EdgeSeqsReestimator(msg)) if msg.contains("root"));
}

#[test]
fn tkf_reestimation_fails_for_leaf() {
    let phylo = setup_test_phylo(Alphabet::dna());
    let mut cost = TKF92IndelCostBuilder::new(&[0.4, 0.5, 0.8], phylo)
        .build()
        .unwrap();
    let rng = &mut FakeGenerator::default();
    let mut reestimator = EdgeSeqsReestimator::new(&mut cost, rng);
    let leaf_idx = reestimator.cost.phylo.tree.by_id("A1").idx;

    let err = reestimator.reestimate(&leaf_idx);
    assert_matches!(err, Err(Error::EdgeSeqsReestimator(msg)) if msg.contains("leaf"));
}

#[test]
fn tkf_mapping_from_node_seq() {
    let mut node_seq = FixedBitSet::with_capacity(5);
    node_seq.insert(0);
    node_seq.insert(2);
    node_seq.insert(4);
    let block_lens = [2, 3, 1, 4, 1];
    let seq_len: usize = block_lens.iter().sum();
    let expected_mapping = [
        Some(0),
        Some(1), // first block finished
        None,
        None,
        None,    // second block finished
        Some(2), // third block finished
        None,
        None,
        None,
        None,    // fourth block finished
        Some(3), // fifth block finished
    ];
    let mapping = mapping_from_node_seq(&node_seq, &block_lens, seq_len);
    assert_eq!(mapping, expected_mapping);
}
