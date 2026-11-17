use super::*;
use crate::alphabets::Alphabet;
use crate::substitution_models::{HKY, JC69};

fn example(a: &[u8], b: &[u8], c: &[u8], i: &[u8], r: &[u8]) -> PhyloInfo<MASA> {
    let records = vec![
        record!("A", Some("keep me"), a),
        record!("B", None, b),
        record!("C", None, c),
        record!("I", None, i),
        record!("R", None, r),
    ];
    MASA::from_aligned_with_ancestral(
        Sequences::with_alphabet_unchecked(records, Alphabet::dna()),
        crate::tree!("((A:0.01,B:0.01)I:0.01,C:0.01)R;"),
    )
    .unwrap()
}

fn ancestor(phylo: &PhyloInfo<MASA>, id: &str) -> Vec<u8> {
    let node = phylo.tree.iter().find(|node| node.id == id).unwrap();
    let record = phylo.msa.ancestral_seqs().record_by_id(id);
    aligned_seq!(phylo.msa.ancestral_map(&node.idx), record.seq())
}

fn close(actual: f64, expected: f64) {
    assert!((actual - expected).abs() < 1e-10, "{actual} != {expected}");
}

#[test]
fn joint_and_marginal_match_all_sixteen_assignments() {
    let phylo = example(b"A", b"C", b"G", b"X", b"X");
    let model = SubstModel::<HKY>::new(&[0.1, 0.2, 0.3, 0.4], &[2.0]).unwrap();
    let transitions = precompute_log_transitions(&model, &phylo.tree);
    let (states, score) = joint::infer(
        &phylo,
        model.freqs().as_slice(),
        &transitions,
        observations(&phylo),
        None,
    );
    let posterior = marginal::infer(
        &phylo,
        model.freqs().as_slice(),
        &transitions,
        observations(&phylo),
        None,
    );
    let root = usize::from(phylo.tree.root);
    let inner = usize::from(phylo.tree.iter().find(|n| n.id == "I").unwrap().idx);
    let alphabet = Alphabet::dna();
    let (a, c, g) = (
        alphabet.index(&b'A'),
        alphabet.index(&b'C'),
        alphabet.index(&b'G'),
    );
    let p = model.p(0.01);
    let mut weights = [[0.0; 4]; 4];
    for r in 0..4 {
        for i in 0..4 {
            weights[r][i] = model.freqs()[r] * p[(r, i)] * p[(i, a)] * p[(i, c)] * p[(r, g)];
        }
    }
    let total: f64 = weights.iter().flatten().sum();
    let best = weights.iter().flatten().copied().fold(0.0, f64::max);
    close(score, best.ln());
    close(weights[states[root][0]][states[inner][0]], best);
    close(posterior.log_likelihood, total.ln());
    for state in 0..4 {
        close(
            posterior.probabilities[root][(state, 0)],
            weights[state].iter().sum::<f64>() / total,
        );
        close(
            posterior.probabilities[inner][(state, 0)],
            weights.iter().map(|row| row[state]).sum::<f64>() / total,
        );
    }
}

#[test]
fn gap_conditioning_blocks_evidence_but_both_variants_keep_output_gaps() {
    let phylo = example(b"N", b"N", b"T", b"X", b"-");
    let model = SubstModel::<HKY>::new(&[0.1, 0.2, 0.3, 0.4], &[2.0]).unwrap();
    for mode in [ReconstructionMode::Joint, ReconstructionMode::Marginal] {
        let cut = FastML::new(model.clone())
            .with_mode(mode)
            .reconstruct_ancestral_seqs(phylo.clone())
            .unwrap();
        let connected = FastMLNoIndelsDuringInference::new(model.clone())
            .with_mode(mode)
            .reconstruct_ancestral_seqs(phylo.clone())
            .unwrap();
        // With no evidence in I's component, its most probable state is the largest prior.
        assert_eq!(ancestor(&cut, "I"), vec![Alphabet::dna().symbols()[3]]);
        assert_eq!(ancestor(&connected, "I"), b"T");
        assert_eq!(ancestor(&cut, "R"), b"-");
        assert_eq!(ancestor(&connected, "R"), b"-");
    }
}

#[test]
fn placeholders_are_ignored_and_leaf_records_are_preserved() {
    let first = example(b"A-C", b"ATC", b"ATC", b"A-A", b"AAA");
    let second = example(b"A-C", b"ATC", b"ATC", b"T-T", b"TTT");
    for mode in [ReconstructionMode::Joint, ReconstructionMode::Marginal] {
        let asr = FastML::new(SubstModel::<JC69>::default()).with_mode(mode);
        let output = asr.reconstruct_ancestral_seqs(first.clone()).unwrap();
        let other = asr.reconstruct_ancestral_seqs(second.clone()).unwrap();
        assert_eq!(ancestor(&output, "I"), b"A-C");
        assert_eq!(ancestor(&output, "R"), b"ATC");
        for id in ["I", "R"] {
            assert_eq!(ancestor(&output, id), ancestor(&other, id));
        }
        for leaf in first.tree.leaves() {
            assert_eq!(
                output.msa.seqs().record_by_id(&leaf.id),
                first.msa.seqs().record_by_id(&leaf.id)
            );
            assert_eq!(
                output.msa.leaf_map(&leaf.idx),
                first.msa.leaf_map(&leaf.idx)
            );
        }
    }
}

#[test]
fn indel_event_posterior_matches_binary_formula_and_output_mask() {
    let phylo = MSA::from_aligned(
        Sequences::new_unchecked(vec![
            record!("A", None, b"A--A"),
            record!("B", None, b"AAAA"),
        ]),
        crate::tree!("(A:0.1,B:0.8)R;"),
    )
    .unwrap();
    let model = IndelModel::new(0.2, 0.8).unwrap();
    let result = indels::reconstruct(&phylo, model);
    let root = usize::from(phylo.tree.root);
    // A has the event, B does not. Sum the two possible root states.
    let change_a = 1.0 - (-0.1_f64).exp();
    let change_b = 1.0 - (-0.8_f64).exp();
    let absent = 0.8 * (0.2 * change_a) * (1.0 - 0.2 * change_b);
    let present = 0.2 * (1.0 - 0.8 * change_a) * (0.8 * change_b);
    let probability = present / (absent + present);
    assert_eq!(result.events, vec![GapEvent { start: 1, end: 3 }]);
    close(result.gap_probabilities[root][0], probability);
    close(result.log_likelihood, (absent + present).ln());
    let output: PhyloInfo<MASA> = model.reconstruct_ancestral_seqs(phylo).unwrap();
    assert_eq!(
        ancestor(&output, "R"),
        if probability > 0.5 { b"X--X" } else { b"XXXX" }
    );
}

#[test]
fn indel_step_with_no_events_produces_placeholders() {
    let phylo = MSA::from_aligned(
        Sequences::new_unchecked(vec![record!("A", None, b"AC"), record!("B", None, b"GT")]),
        crate::tree!("(A:0.1,B:0.2)R;"),
    )
    .unwrap();
    let output: PhyloInfo<MASA> = IndelModel::new(0.2, 0.8)
        .unwrap()
        .reconstruct_ancestral_seqs(phylo)
        .unwrap();
    assert_eq!(ancestor(&output, "R"), b"XX");
}
