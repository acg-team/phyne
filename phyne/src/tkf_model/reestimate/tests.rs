use approx::assert_relative_eq;
use assert_matches::assert_matches;
use fixedbitset::FixedBitSet;

use crate::alignment::{AncestralAlignment, Sequences, MASA};
use crate::alphabets::Alphabet;
use crate::likelihood::TreeSearchCost;
use crate::optimisers::rooted_nni;
use crate::phylo_info::PhyloInfo;
use crate::random::FakeGenerator;
use crate::tkf_model::reestimate::mapping_from_node_seq;
use crate::tkf_model::tests::setup_test_phylo;
use crate::tkf_model::{EdgeSeqsReestimator, TKF92IndelModel, TKFIndelCostBuilder};
use crate::{record_wo_desc as record, tree, Error};

#[test]
fn tkf_reestimate_without_choice() {
    let tree = tree!("(((A1:2.0,(B2:1.0, C3:1.2)I4:2.0)I5:0.3,D6:2.0)R7:1.0);");
    // The alignment is designed such that for every site the re-estimation will have no choice but
    // to keep the current states, since it must conform to Dollo's principle.
    let phylo = MASA::from_aligned_with_ancestral(
        Sequences::new_unchecked(vec![
            record!("A1", b"-----N-----"),
            record!("B2", b"NN----NN-NN"),
            record!("C3", b"--NNN---NNN"),
            record!("I4", b"NNNNN----NN"),
            record!("I5", b"NNNNNN---NN"),
            record!("D6", b"NNNNNN---NN"),
            record!("R7", b"NNNNNN---NN"),
        ]),
        tree,
    )
    .unwrap();
    let tkf_indel_model = TKF92IndelModel::new(&[0.4, 0.5, 0.8]).unwrap();
    let mut cost = TKFIndelCostBuilder::new(tkf_indel_model, phylo).build();
    let logl = cost.clone().logl();
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

/// Reproduces the stale tmp values logl bug: see https://github.com/acg-team/phyne/issues/216.
/// Which is fixed now.
///
/// After an NNI move the parent of `A1` and `C4` changes (since they swap places),
/// so their TKF events change, but their tmp values were not invalidated in `TKFIndelCost::update_tree`.
/// If the re-estimation did not change the ancestral mappings,
/// it assumes nothing changed and returns the before re-estimation logl, which is incorrect.
/// Now, nodes are correctly invalidated after an NNI move so the before re-estimation logl is correct.
#[test]
fn tkf_nni_with_unchanged_mappings_does_not_return_stale_logl() {
    // The branches of A1 and C4 differ on purpose, otherwise their swapped event factors
    // would cancel out and mask the bug.
    let tree = tree!("(((A1:0.5,B2:0.1)I3:1.0,C4:2.0)R5:1.0);");
    let phylo = MASA::from_aligned_with_ancestral(
        Sequences::new_unchecked(vec![
            record!("A1", b"-"),
            record!("B2", b"A"),
            record!("C4", b"-"),
            record!("I3", b"N"),
            record!("R5", b"-"),
        ]),
        tree,
    )
    .unwrap();
    let tkf_indel_model = TKF92IndelModel::new(&[0.4, 0.5, 0.8]).unwrap();
    let mut cost = TKFIndelCostBuilder::new(tkf_indel_model.clone(), phylo.clone()).build();
    let original_masa = cost.phylo.msa.clone();
    assert!(TreeSearchCost::cost(&cost).is_finite());

    let node_idx = phylo.tree.by_id("I3").idx;
    let child_idx = phylo.tree.by_id("A1").idx;
    let new_tree = rooted_nni(&phylo.tree, &node_idx, &child_idx).unwrap();
    // This panicked before the bug was fixed, because the returned DP re-estimation
    // logl did not match the clean cost in the assert after re- estimation.
    cost.update_tree(new_tree);

    // The mappings should not have changed, i.e. we exercise the stale-logl branch.
    let i3_idx = cost.phylo.tree.by_id("I3").idx;
    let r5_idx = cost.phylo.tree.by_id("R5").idx;
    assert_eq!(
        cost.phylo.msa.ancestral_map(&i3_idx),
        original_masa.ancestral_map(&i3_idx)
    );
    assert_eq!(
        cost.phylo.msa.ancestral_map(&r5_idx),
        original_masa.ancestral_map(&r5_idx)
    );

    let new_phylo = PhyloInfo {
        msa: cost.phylo.msa.clone(),
        tree: cost.tree().clone(),
    };
    // Sanity check
    let clean_cost = TKFIndelCostBuilder::new(tkf_indel_model, new_phylo).build();
    assert_relative_eq!(
        TreeSearchCost::cost(&cost),
        TreeSearchCost::cost(&clean_cost),
        epsilon = 1e-10
    );
}

#[test]
fn tkf_reestimation_fails_for_root() {
    let phylo = setup_test_phylo(Alphabet::dna());
    let tkf_indel_model = TKF92IndelModel::new(&[0.4, 0.5, 0.8]).unwrap();
    let mut cost = TKFIndelCostBuilder::new(tkf_indel_model, phylo).build();
    let rng = &mut FakeGenerator::default();
    let mut reestimator = EdgeSeqsReestimator::new(&mut cost, rng);
    let root_idx = reestimator.cost.phylo.tree.root;

    let err = reestimator.reestimate(&root_idx);
    assert_matches!(err, Err(Error::EdgeSeqsReestimator(msg)) if msg.contains("root"));
}

#[test]
fn tkf_reestimation_fails_for_leaf() {
    let phylo = setup_test_phylo(Alphabet::dna());
    let tkf_indel_model = TKF92IndelModel::new(&[0.4, 0.5, 0.8]).unwrap();
    let mut cost = TKFIndelCostBuilder::new(tkf_indel_model, phylo).build();
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
