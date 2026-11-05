use super::*;
use crate::alphabets::Alphabet;
use crate::substitution_models::{HKY, JC69, WAG};
use crate::tree::NodeIdx::*;
use crate::{record_wo_desc, tree};

fn example(a: &[u8], b: &[u8], c: &[u8]) -> PhyloInfo<MSA> {
    let seqs = Sequences::with_alphabet_unchecked(
        vec![
            record!("A", Some("keep description"), a),
            record_wo_desc!("B", b),
            record_wo_desc!("C", c),
        ],
        Alphabet::dna(),
    );
    MSA::from_aligned(seqs, tree!("((A:0.11,B:0.23)I:0.17,C:0.31)R;")).unwrap()
}

// Independent oracle: enumerate every internal-state assignment and multiply
// probabilities directly, rather than using the dynamic-programming recurrence.
fn assignment_probability<Q: QMatrix>(
    phylo: &PhyloInfo<MSA>,
    model: &SubstModel<Q>,
    states: &[usize],
    site: usize,
) -> f64 {
    let mut probability = model.freqs()[states[usize::from(phylo.tree.root)]];
    for node in phylo.tree.iter() {
        if node.idx == phylo.tree.root {
            continue;
        }
        let parent = usize::from(phylo.tree.parent(&node.idx).unwrap());
        let p = model.p(node.blen);
        match node.idx {
            Internal(_) => probability *= p[(states[parent], states[usize::from(node.idx)])],
            Leaf(_) => {
                let encoding = match phylo.msa.leaf_map(&node.idx)[site] {
                    Some(pos) => phylo
                        .msa
                        .alphabet()
                        .char_encoding(phylo.msa.seqs().record_by_id(&node.id).seq()[pos]),
                    None => phylo.msa.alphabet().missing_char_encoding(),
                };
                probability *= (0..model.n())
                    .map(|s| p[(states[parent], s)] * encoding[s])
                    .sum::<f64>();
            }
        }
    }
    probability
}

#[test]
fn agrees_with_exhaustive_search_including_ambiguity_and_gaps() {
    let phylo = example(b"ACGR-N", b"ATGYCT", b"GCGTNA");
    let model = SubstModel::<HKY>::new(&[0.1, 0.2, 0.3, 0.4], &[2.0]).unwrap();
    let fastml = FastML::new(model.clone());
    let (actual, score) = fastml.calc_joint(&phylo);
    let internals = phylo.tree.internals();
    let mut expected_score = 0.0;
    for site in 0..phylo.msa.len() {
        let mut best = 0.0_f64;
        for assignment in 0..4usize.pow(internals.len() as u32) {
            let mut digits = assignment;
            let mut states = vec![0; phylo.tree.len()];
            for node in &internals {
                states[usize::from(node.idx)] = digits % 4;
                digits /= 4;
            }
            best = best.max(assignment_probability(&phylo, &model, &states, site));
        }
        let actual_states: Vec<_> = actual.iter().map(|seq| seq[site]).collect();
        let actual_probability = assignment_probability(&phylo, &model, &actual_states, site);
        assert!((actual_probability - best).abs() < 1e-12);
        expected_score += best.ln();
    }
    assert!((score - expected_score).abs() < 1e-10);
}

#[test]
fn roots_are_chosen_per_site_and_indices_become_letters() {
    let phylo = example(b"ACGTAC", b"ACGTAC", b"ACGTAC");
    let output = FastML::new(SubstModel::<JC69>::default())
        .reconstruct_ancestral_seqs(phylo)
        .unwrap();
    for node in output.tree.internals() {
        assert_eq!(
            output.msa.ancestral_seqs().record_by_id(&node.id).seq(),
            b"ACGTAC"
        );
    }
}

#[test]
fn preserves_leaf_sequences_gaps_and_descriptions() {
    let phylo = example(b"A-RN", b"ACGT", b"ACGT");
    let output = FastML::new(SubstModel::<JC69>::default())
        .reconstruct_ancestral_seqs(phylo.clone())
        .unwrap();
    for leaf in phylo.tree.leaves() {
        let before = phylo.msa.seqs().record_by_id(&leaf.id);
        let after = output.msa.seqs().record_by_id(&leaf.id);
        assert_eq!(after.desc(), before.desc());
        assert_eq!(
            aligned_seq!(output.msa.leaf_map(&leaf.idx), after.seq()),
            aligned_seq!(phylo.msa.leaf_map(&leaf.idx), before.seq())
        );
    }
}

#[test]
fn rejects_mismatched_alphabet() {
    let result = FastML::new(SubstModel::<WAG>::default())
        .reconstruct_ancestral_seqs(example(b"A", b"A", b"A"));
    assert!(matches!(result, Err(Error::Alphabet(_))));
}

#[test]
fn impossible_zero_length_branches_return_error() {
    let phylo = MSA::from_aligned(
        Sequences::new_unchecked(vec![record_wo_desc!("A", b"A"), record_wo_desc!("B", b"C")]),
        tree!("(A:0.0,B:0.0)R;"),
    )
    .unwrap();
    let fastml = FastML::new(SubstModel::<JC69>::default());
    assert_eq!(fastml.calc_joint(&phylo).1, f64::NEG_INFINITY);
    assert!(matches!(
        fastml.reconstruct_ancestral_seqs(phylo),
        Err(Error::AncestralAlignment(_))
    ));
}

#[test]
fn checked_entry_assigns_missing_ids_and_rejects_duplicates() {
    for (newick, should_succeed) in [
        ("((A:0.1,B:0.1):0.1,C:0.1);", true),
        ("((A:0.1,B:0.1)I:0.1,C:0.1)I;", false),
    ] {
        let seqs = Sequences::new_unchecked(vec![
            record_wo_desc!("A", b"A"),
            record_wo_desc!("B", b"A"),
            record_wo_desc!("C", b"A"),
        ]);
        let phylo = MSA::from_aligned(seqs, tree!(newick)).unwrap();
        let result = FastML::new(SubstModel::<JC69>::default()).reconstruct_ancestral_seqs(phylo);
        assert_eq!(result.is_ok(), should_succeed);
    }
}

#[test]
fn protein_reconstruction_uses_protein_symbols() {
    let seqs = Sequences::with_alphabet_unchecked(
        vec![record_wo_desc!("A", b"WREL"), record_wo_desc!("B", b"WREL")],
        Alphabet::protein(),
    );
    let phylo = MSA::from_aligned(seqs, tree!("(A:0.001,B:0.001)R;")).unwrap();
    let result = FastML::new(SubstModel::<WAG>::default())
        .reconstruct_ancestral_seqs(phylo)
        .unwrap();
    assert_eq!(result.msa.ancestral_seqs().record_by_id("R").seq(), b"WREL");
}

#[test]
fn one_site_zero_length_branches_have_finite_score_when_consistent() {
    let phylo = MSA::from_aligned(
        Sequences::new_unchecked(vec![record_wo_desc!("A", b"G"), record_wo_desc!("B", b"G")]),
        tree!("(A:0.0,B:0.0)R;"),
    )
    .unwrap();
    let fastml = FastML::new(SubstModel::<JC69>::default());
    let (states, score) = fastml.calc_joint(&phylo);
    assert_eq!(
        states[usize::from(phylo.tree.root)],
        vec![Alphabet::dna().index(&b'G')]
    );
    assert!((score - 0.25_f64.ln()).abs() < 1e-12);
}

#[test]
fn completely_ambiguous_observations_choose_first_state_on_ties() {
    let phylo = MSA::from_aligned(
        Sequences::with_alphabet_unchecked(
            vec![record_wo_desc!("A", b"N"), record_wo_desc!("B", b"N")],
            Alphabet::dna(),
        ),
        tree!("(A:0.0,B:0.0)R;"),
    )
    .unwrap();
    let (states, score) = FastML::new(SubstModel::<JC69>::default()).calc_joint(&phylo);
    assert_eq!(states[usize::from(phylo.tree.root)], vec![0]);
    assert!((score - 0.25_f64.ln()).abs() < 1e-12);
}

#[test]
fn marginal_probabilities_and_likelihood_match_exhaustive_sum() {
    let phylo = example(b"ACGR-N", b"ATGYCT", b"GCGTNA");
    let model = SubstModel::<HKY>::new(&[0.1, 0.2, 0.3, 0.4], &[2.0]).unwrap();
    let fastml = FastML::new(model.clone()).with_mode(ReconstructionMode::Marginal);
    let result = fastml.marginal_posteriors(&phylo).unwrap();
    let selected = fastml.selected_states(&phylo).unwrap();
    let internals = phylo.tree.internals();
    let mut expected_log_likelihood = 0.0;
    for site in 0..phylo.msa.len() {
        let mut total = 0.0;
        let mut masses = vec![vec![0.0; 4]; phylo.tree.len()];
        for assignment in 0..4usize.pow(internals.len() as u32) {
            let mut digits = assignment;
            let mut states = vec![0; phylo.tree.len()];
            for node in &internals {
                states[usize::from(node.idx)] = digits % 4;
                digits /= 4;
            }
            let weight = assignment_probability(&phylo, &model, &states, site);
            total += weight;
            for node in &internals {
                let idx = usize::from(node.idx);
                masses[idx][states[idx]] += weight;
            }
        }
        expected_log_likelihood += total.ln();
        for node in &internals {
            let idx = usize::from(node.idx);
            for state in 0..4 {
                assert!(
                    (result.probabilities[idx][(state, site)] - masses[idx][state] / total).abs()
                        < 1e-10
                );
            }
            let best = masses[idx].iter().copied().fold(0.0_f64, f64::max);
            assert!((masses[idx][selected[idx][site]] - best).abs() < 1e-12);
        }
        for probabilities in &result.probabilities {
            assert!((probabilities.column(site).iter().sum::<f64>() - 1.0).abs() < 1e-12);
        }
    }
    assert!((result.log_likelihood - expected_log_likelihood).abs() < 1e-10);
}

#[test]
fn marginal_nonroot_uses_evidence_outside_its_subtree() {
    // I's own descendants are completely unknown. C must nevertheless affect I.
    // Treating a normalized inside vector as the posterior would give a uniform result.
    let phylo = example(b"N", b"N", b"A");
    let result = FastML::new(SubstModel::<JC69>::default())
        .marginal_posteriors(&phylo)
        .unwrap();
    let node = phylo
        .tree
        .internals()
        .into_iter()
        .find(|n| n.id == "I")
        .unwrap();
    let idx = usize::from(node.idx);
    let a = Alphabet::dna().index(&b'A');
    assert!(result.probabilities[idx][(a, 0)] > 0.25);
    for state in 0..4 {
        if state != a {
            assert!(result.probabilities[idx][(a, 0)] > result.probabilities[idx][(state, 0)]);
        }
    }
}

#[test]
fn marginal_handles_zero_branches_without_nan_and_rejects_impossible_data() {
    for (b, valid) in [(b'A', true), (b'C', false)] {
        let phylo = MSA::from_aligned(
            Sequences::new_unchecked(vec![record_wo_desc!("A", b"A"), record_wo_desc!("B", &[b])]),
            tree!("(A:0.0,B:0.0)R;"),
        )
        .unwrap();
        let asr =
            FastML::new(SubstModel::<JC69>::default()).with_mode(ReconstructionMode::Marginal);
        let result = asr.marginal_posteriors(&phylo);
        assert_eq!(result.is_ok(), valid);
        if let Ok(result) = result {
            for p in result.probabilities {
                assert!(p.iter().all(|x| x.is_finite()));
            }
        }
        assert_eq!(asr.reconstruct_ancestral_seqs(phylo).is_ok(), valid);
    }
}

#[test]
fn marginal_output_preserves_leaves_and_outputs_selected_letters() {
    let phylo = example(b"ACGR-N", b"ATGYCT", b"GCGTNA");
    let asr = FastML::new(SubstModel::<JC69>::default()).with_mode(ReconstructionMode::Marginal);
    let states = asr.selected_states(&phylo).unwrap();
    let output = asr.reconstruct_ancestral_seqs(phylo.clone()).unwrap();
    for node in output.tree.internals() {
        let expected: Vec<u8> = states[usize::from(node.idx)]
            .iter()
            .map(|&s| Alphabet::dna().symbols()[s])
            .collect();
        assert_eq!(
            output.msa.ancestral_seqs().record_by_id(&node.id).seq(),
            expected.as_slice()
        );
    }
    for leaf in output.tree.leaves() {
        assert_eq!(
            aligned_seq!(
                output.msa.leaf_map(&leaf.idx),
                output.msa.seqs().record_by_id(&leaf.id).seq()
            ),
            aligned_seq!(
                phylo.msa.leaf_map(&leaf.idx),
                phylo.msa.seqs().record_by_id(&leaf.id).seq()
            )
        );
    }
}

#[test]
fn indels_reconstruct_a_block_and_preserve_leaf_gaps_in_both_modes() {
    let phylo = example(b"A--A", b"A--A", b"AAAA");
    for mode in [ReconstructionMode::Joint, ReconstructionMode::Marginal] {
        let asr = FastML::new(SubstModel::<JC69>::default())
            .with_mode(mode)
            .with_indels(IndelModel::new(0.1, 0.1).unwrap());
        let indels = asr.reconstruct_indels(&phylo).unwrap().unwrap();
        assert_eq!(indels.events, vec![GapEvent { start: 1, end: 3 }]);
        let node = phylo
            .tree
            .internals()
            .into_iter()
            .find(|n| n.id == "I")
            .unwrap();
        let idx = usize::from(node.idx);
        assert!(indels.gap_probabilities[idx][0] > 0.5);
        assert_eq!(indels.gaps[idx], vec![false, true, true, false]);
        let output = asr.reconstruct_ancestral_seqs(phylo.clone()).unwrap();
        let record = output.msa.ancestral_seqs().record_by_id("I");
        let aligned = aligned_seq!(output.msa.ancestral_map(&node.idx), record.seq());
        assert_eq!(aligned.as_slice(), b"A--A");
        for leaf in phylo.tree.leaves() {
            assert_eq!(
                aligned_seq!(
                    output.msa.leaf_map(&leaf.idx),
                    output.msa.seqs().record_by_id(&leaf.id).seq()
                ),
                aligned_seq!(
                    phylo.msa.leaf_map(&leaf.idx),
                    phylo.msa.seqs().record_by_id(&leaf.id).seq()
                )
            );
        }
    }
}

#[test]
fn enabling_indels_on_ungapped_input_does_not_change_output() {
    let phylo = example(b"ACGT", b"ACGT", b"ACGT");
    let asr =
        FastML::new(SubstModel::<JC69>::default()).with_indels(IndelModel::new(0.1, 0.2).unwrap());
    let result = asr.reconstruct_indels(&phylo).unwrap().unwrap();
    assert!(result.events.is_empty());
    assert_eq!(result.log_likelihood, 0.0);
    let output = asr.reconstruct_ancestral_seqs(phylo).unwrap();
    for node in output.tree.internals() {
        assert_eq!(
            output.msa.ancestral_seqs().record_by_id(&node.id).seq(),
            b"ACGT"
        );
    }
}

#[test]
fn marginal_protein_posteriors_work_with_twenty_states() {
    let phylo = MSA::from_aligned(
        Sequences::with_alphabet_unchecked(
            vec![record_wo_desc!("A", b"W"), record_wo_desc!("B", b"W")],
            Alphabet::protein(),
        ),
        tree!("(A:0.01,B:0.01)R;"),
    )
    .unwrap();
    let asr = FastML::new(SubstModel::<WAG>::default()).with_mode(ReconstructionMode::Marginal);
    let result = asr.marginal_posteriors(&phylo).unwrap();
    let root = usize::from(phylo.tree.root);
    let w = Alphabet::protein().index(&b'W');
    assert!(result.probabilities[root][(w, 0)] > 0.99);
    let output = asr.reconstruct_ancestral_seqs(phylo).unwrap();
    assert_eq!(output.msa.ancestral_seqs().record_by_id("R").seq(), b"W");
}
// Add this test to src/fastml/tests.rs, declared with `mod tests;`
// inside the fastml module so it can access the private calc_joint method.
// Expected values are calculated by hand, not using assignment_probability.
// This checks the root calculation; keep the multi-internal-node exhaustive
// test to check the non-root recurrence and traceback as well.

#[test]
fn fastml_matches_hand_calculated_probability() {
    use crate::alignment::{Alignment, Sequences, MSA};
    use crate::fastml::FastML;
    use crate::substitution_models::{SubstModel, JC69};

    // For Phyne's normalized JC69 model:
    // P(same)      = 0.25 + 0.75 * exp(-4t/3)
    // P(different) = 0.25 - 0.25 * exp(-4t/3)
    // Choosing exp(-4t/3) = 0.6 gives 0.7 and 0.1 respectively.
    let branch_length = -0.75 * 0.6_f64.ln();
    let newick = format!("(L1:{branch_length:.17},L2:{branch_length:.17})R;");
    let tree = crate::tree!(&newick);

    let sequences = Sequences::new_unchecked(vec![
        crate::record_wo_desc!("L1", b"A"),
        crate::record_wo_desc!("L2", b"C"),
    ]);
    let phylo = MSA::from_aligned(sequences, tree).unwrap();
    let fastml = FastML::new(SubstModel::<JC69>::default());

    let (states, log_score) = fastml.calc_joint(&phylo);

    // Root A: 0.25 * 0.7 * 0.1 = 0.0175
    // Root C: 0.25 * 0.1 * 0.7 = 0.0175
    // Root G: 0.25 * 0.1 * 0.1 = 0.0025
    // Root T: 0.25 * 0.1 * 0.1 = 0.0025
    let expected_probability = 0.0175_f64;
    assert!(
        (log_score - expected_probability.ln()).abs() < 1e-10,
        "Expected log score {}, got {}",
        expected_probability.ln(),
        log_score,
    );

    // A and C tie: either is an optimal root state.
    let root_index = usize::from(phylo.tree.root);
    let root_state = states[root_index][0];
    let root_letter = phylo.msa.alphabet().symbols()[root_state];
    assert!(
        matches!(root_letter, b'A' | b'C'),
        "Expected root A or C, got {}",
        root_letter as char,
    );
}

#[test]
fn shared_pass_sums_leaf_uncertainty_even_in_joint_mode() {
    let phylo = example(b"N", b"N", b"N");
    let asr = FastML::new(SubstModel::<JC69>::default());
    // Only the internal branch R -> I is maximized. Unknown leaves contribute 1.
    // The fixture has branch length 0.17 for I; JC69 P(same) is known analytically.
    let same = 0.25 + 0.75 * (-4.0_f64 * 0.17 / 3.0).exp();
    let (_, joint_score) = asr.calc_joint(&phylo);
    assert!((joint_score - (0.25 * same).ln()).abs() < 1e-12);
    let marginal = asr.marginal_posteriors(&phylo).unwrap();
    assert!(marginal.log_likelihood.abs() < 1e-12);
    for probabilities in marginal.probabilities {
        assert!(probabilities.iter().all(|p| (*p - 0.25).abs() < 1e-12));
    }
}
