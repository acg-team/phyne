//! FastML-specific inference and integration tests.
//! Binary-event coding and rate-model tests belong in indels.rs.
//! Uses with_indels(model, indel_first) and Option<&IndelReconstruction>
//! arguments on calc_joint, selected_states, and marginal_posteriors.
//! Assumes masked marginal columns are zero (no letter distribution).
use super::*;
use crate::alphabets::Alphabet;
use crate::substitution_models::{HKY, JC69};
use crate::tree::NodeIdx::{Internal, Leaf};
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

fn node_index(phylo: &PhyloInfo<MSA>, id: &str) -> usize {
    usize::from(phylo.tree.iter().find(|node| node.id == id).unwrap().idx)
}

fn close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1e-10,
        "expected {expected}, got {actual}"
    );
}

// Construct a fixed mask to isolate character inference from event inference.
// Leaf gaps match the input; only ancestral gaps are assigned by the test.
fn fixed_mask(phylo: &PhyloInfo<MSA>, ancestral_gaps: &[(&str, usize)]) -> IndelReconstruction {
    let mut gaps = vec![vec![false; phylo.msa.len()]; phylo.tree.len()];
    for leaf in phylo.tree.leaves() {
        gaps[usize::from(leaf.idx)] = phylo
            .msa
            .leaf_map(&leaf.idx)
            .iter()
            .map(Option::is_none)
            .collect();
    }
    for &(id, site) in ancestral_gaps {
        gaps[node_index(phylo, id)][site] = true;
    }
    IndelReconstruction {
        events: Vec::new(),
        gap_probabilities: vec![Vec::new(); phylo.tree.len()],
        gaps,
        log_likelihood: 0.0,
    }
}

// Independent oracle: multiply ordinary probabilities for one assignment.
// A present node below a gap starts with the prior, without an incoming branch.
// Absent nodes have no character factor. Leaf uncertainty is summed, not maximized.
fn assignment_probability<Q: QMatrix>(
    phylo: &PhyloInfo<MSA>,
    model: &SubstModel<Q>,
    states: &[usize],
    site: usize,
    mask: Option<&IndelReconstruction>,
) -> f64 {
    let is_gap = |idx: usize| mask.is_some_and(|m| m.gaps[idx][site]);
    let mut probability = 1.0;
    for node in phylo.tree.iter() {
        let idx = usize::from(node.idx);
        if is_gap(idx) {
            continue;
        }
        let parent = phylo.tree.parent(&node.idx).map(usize::from);
        let starts_component = parent.map_or(true, |p| is_gap(p));
        let p = model.p(node.blen);
        let weight = |state: usize| {
            if starts_component {
                model.freqs()[state]
            } else {
                p[(states[parent.unwrap()], state)]
            }
        };
        match node.idx {
            Internal(_) => probability *= weight(states[idx]),
            Leaf(_) => {
                probability *= match phylo.msa.leaf_map(&node.idx)[site] {
                    Some(pos) => {
                        let symbol = phylo.msa.seqs().record_by_id(&node.id).seq()[pos];
                        let encoding = phylo.msa.alphabet().char_encoding(symbol);
                        (0..model.n()).map(|s| weight(s) * encoding[s]).sum::<f64>()
                    }
                    None => 1.0,
                };
            }
        }
    }
    probability
}

// Enumerate only PRESENT internal states: summing assignments to absent nodes
// would incorrectly count the same character assignment more than once.
fn check_against_enumeration(phylo: &PhyloInfo<MSA>, mask: Option<&IndelReconstruction>) {
    let model = SubstModel::<HKY>::new(&[0.1, 0.2, 0.3, 0.4], &[2.0]).unwrap();
    let asr = FastML::new(model.clone());
    let (joint_states, joint_score) = asr.calc_joint(phylo, mask);
    let marginal = asr.marginal_posteriors(phylo, mask).unwrap();
    let mut expected_joint = 0.0;
    let mut expected_marginal = 0.0;
    for site in 0..phylo.msa.len() {
        let active: Vec<usize> = phylo
            .tree
            .internals()
            .iter()
            .map(|node| usize::from(node.idx))
            .filter(|&idx| !mask.is_some_and(|m| m.gaps[idx][site]))
            .collect();
        let mut masses = vec![vec![0.0; model.n()]; phylo.tree.len()];
        let mut total = 0.0;
        let mut best = 0.0_f64;
        for assignment in 0..model.n().pow(active.len() as u32) {
            let mut digits = assignment;
            let mut states = vec![0; phylo.tree.len()];
            for &idx in &active {
                states[idx] = digits % model.n();
                digits /= model.n();
            }
            let weight = assignment_probability(phylo, &model, &states, site, mask);
            total += weight;
            best = best.max(weight);
            for &idx in &active {
                masses[idx][states[idx]] += weight;
            }
        }
        let actual: Vec<usize> = joint_states.iter().map(|seq| seq[site]).collect();
        close(
            assignment_probability(phylo, &model, &actual, site, mask),
            best,
        );
        expected_joint += best.ln();
        expected_marginal += total.ln();
        for &idx in &active {
            for state in 0..model.n() {
                close(
                    marginal.probabilities[idx][(state, site)],
                    masses[idx][state] / total,
                );
            }
        }
        for idx in 0..phylo.tree.len() {
            let column = marginal.probabilities[idx].column(site);
            assert!(column.iter().all(|p| p.is_finite() && *p >= 0.0));
            if mask.is_some_and(|m| m.gaps[idx][site]) {
                // Convention: an absent position has no letter distribution.
                assert!(column.iter().all(|p| *p == 0.0));
            } else {
                close(column.iter().sum(), 1.0);
            }
        }
    }
    close(joint_score, expected_joint);
    close(marginal.log_likelihood, expected_marginal);
}

#[test]
fn ordinary_joint_and_marginal_match_exhaustive_assignments() {
    check_against_enumeration(&example(b"ACGR-N", b"ATGYCT", b"GCGTNA"), None);
}

#[test]
fn indel_first_components_match_exhaustive_assignments() {
    let phylo = example(b"ACGR-N", b"ATGYCT", b"GCGTNA");
    // Different boundaries per site: internal gap, root gap, both absent,
    // a present leaf below a gap, and a completely uncut site (site 5).
    let mask = fixed_mask(
        &phylo,
        &[("I", 0), ("R", 1), ("R", 2), ("I", 2), ("I", 3), ("R", 4)],
    );
    check_against_enumeration(&phylo, Some(&mask));
}

#[test]
fn all_present_mask_matches_no_mask() {
    let phylo = example(b"ACGN", b"ATGR", b"GCGT");
    let mask = fixed_mask(&phylo, &[]);
    let asr = FastML::new(SubstModel::<JC69>::default());
    let (plain, plain_score) = asr.calc_joint(&phylo, None);
    let (masked, masked_score) = asr.calc_joint(&phylo, Some(&mask));
    assert_eq!(plain, masked);
    close(plain_score, masked_score);
    let plain = asr.marginal_posteriors(&phylo, None).unwrap();
    let masked = asr.marginal_posteriors(&phylo, Some(&mask)).unwrap();
    close(plain.log_likelihood, masked.log_likelihood);
    for (a, b) in plain.probabilities.iter().zip(&masked.probabilities) {
        for (&a, &b) in a.iter().zip(b.iter()) {
            close(a, b);
        }
    }
}

#[test]
fn gap_boundary_blocks_outside_evidence() {
    let phylo = example(b"N", b"N", b"A");
    let mask = fixed_mask(&phylo, &[("R", 0)]);
    let asr = FastML::new(SubstModel::<JC69>::default());
    let plain = asr.marginal_posteriors(&phylo, None).unwrap();
    let cut = asr.marginal_posteriors(&phylo, Some(&mask)).unwrap();
    let idx = node_index(&phylo, "I");
    let a = Alphabet::dna().index(&b'A');
    assert!(plain.probabilities[idx][(a, 0)] > 0.25);
    for state in 0..4 {
        close(cut.probabilities[idx][(state, 0)], 0.25);
    }
    // The unknown component contributes 1; the isolated A leaf contributes 0.25.
    close(cut.log_likelihood, 0.25_f64.ln());

    let model = SubstModel::<HKY>::new(&[0.1, 0.2, 0.3, 0.4], &[2.0]).unwrap();
    let asr = FastML::new(model);
    let (states, _) = asr.calc_joint(&phylo, Some(&mask));
    // I's descendants are unknown; its restarted component chooses the
    // largest prior, independently of the observed A in the other component.
    // The restarted component chooses state 3, whose prior is largest (0.4).
    assert_eq!(states[idx][0], 3);
}

#[test]
fn zero_length_conflict_is_impossible_until_gap_separates_components() {
    let phylo = MSA::from_aligned(
        Sequences::with_alphabet_unchecked(
            vec![record_wo_desc!("A", b"A"), record_wo_desc!("B", b"C")],
            Alphabet::dna(),
        ),
        tree!("(A:0.0,B:0.0)R;"),
    )
    .unwrap();
    let asr = FastML::new(SubstModel::<JC69>::default());
    assert_eq!(asr.calc_joint(&phylo, None).1, f64::NEG_INFINITY);
    assert!(asr.marginal_posteriors(&phylo, None).is_err());
    let mask = fixed_mask(&phylo, &[("R", 0)]);
    // Two isolated observed leaves, each with prior probability 0.25.
    let expected = (0.25_f64 * 0.25).ln();
    close(asr.calc_joint(&phylo, Some(&mask)).1, expected);
    close(
        asr.marginal_posteriors(&phylo, Some(&mask))
            .unwrap()
            .log_likelihood,
        expected,
    );
}

#[test]
fn indel_first_flag_routes_mask_and_preserves_output() {
    let phylo = example(b"A--A", b"A--A", b"AAAA");
    for mode in [ReconstructionMode::Joint, ReconstructionMode::Marginal] {
        for indel_first in [false, true] {
            let asr = FastML::new(SubstModel::<JC69>::default())
                .with_mode(mode)
                .with_indels(IndelModel::new(0.1, 0.1).unwrap(), indel_first);
            let mask = asr.reconstruct_indels(&phylo).unwrap().unwrap();
            assert!(mask.gaps[node_index(&phylo, "I")][1]);
            let forward = if indel_first { Some(&mask) } else { None };
            // Independently verify the inference used by this public route.
            check_against_enumeration(&phylo, forward);
            let states = asr.selected_states(&phylo, forward).unwrap();
            let output = asr.reconstruct_ancestral_seqs(phylo.clone()).unwrap();
            let unchecked = asr.reconstruct_ancestral_seqs_unchecked(phylo.clone());
            for node in phylo.tree.internals() {
                let idx = usize::from(node.idx);
                let expected: Vec<u8> = (0..phylo.msa.len())
                    .map(|site| {
                        if mask.gaps[idx][site] {
                            GAP
                        } else {
                            phylo.msa.alphabet().symbols()[states[idx][site]]
                        }
                    })
                    .collect();
                for result in [&output, &unchecked] {
                    let record = result.msa.ancestral_seqs().record_by_id(&node.id);
                    assert_eq!(
                        aligned_seq!(result.msa.ancestral_map(&node.idx), record.seq()),
                        expected
                    );
                }
            }
            for leaf in phylo.tree.leaves() {
                let before = phylo.msa.seqs().record_by_id(&leaf.id);
                let after = output.msa.seqs().record_by_id(&leaf.id);
                assert_eq!(before.desc(), after.desc());
                assert_eq!(
                    aligned_seq!(output.msa.leaf_map(&leaf.idx), after.seq()),
                    aligned_seq!(phylo.msa.leaf_map(&leaf.idx), before.seq())
                );
            }
        }
    }
}

#[test]
fn no_gap_events_leave_both_indel_settings_unchanged() {
    let phylo = example(b"ACGT", b"ACGT", b"ACGT");
    for mode in [ReconstructionMode::Joint, ReconstructionMode::Marginal] {
        for indel_first in [false, true] {
            let asr = FastML::new(SubstModel::<JC69>::default())
                .with_mode(mode)
                .with_indels(IndelModel::new(0.1, 0.2).unwrap(), indel_first);
            let mask = asr.reconstruct_indels(&phylo).unwrap().unwrap();
            assert!(mask.events.is_empty());
            let output = asr.reconstruct_ancestral_seqs(phylo.clone()).unwrap();
            for node in output.tree.internals() {
                assert_eq!(
                    output.msa.ancestral_seqs().record_by_id(&node.id).seq(),
                    b"ACGT"
                );
            }
        }
    }
}
