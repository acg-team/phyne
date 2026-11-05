//! Shared node-first upward traversal for joint and marginal reconstruction.
use super::ReconstructionMode;
use crate::evolutionary_models::ProbabilityMatrix;
use crate::tree::{NodeIdx::Leaf, Tree};
use nalgebra::DMatrix;

// These are intermediate scores, not posterior probabilities. Joint parent_scores
// keep the best internal assignment; marginal parent_scores sum over assignments.
pub(super) struct Upward {
    pub subtree_scores: Vec<DMatrix<f64>>,
    pub(super) conditioned_scores: Vec<DMatrix<f64>>,
    pub choices: Option<Vec<DMatrix<usize>>>,
}

pub(super) fn upward(
    tree: &Tree,
    observations: Vec<DMatrix<f64>>,
    transitions: &[ProbabilityMatrix],
    mode: ReconstructionMode,
) -> Upward {
    let root = usize::from(tree.root);
    let k = observations[root].nrows();
    let sites = observations[root].ncols();
    //could potentially use matrix multiplication for marginal implementation, though we'd have to operate in probability space
    //which risks underflow and would require rescaling.
    //Joint implementation takes the maximum, so matrix operation is not applicable.

    //stores entries for entire tree if memory requirement is too high, could adapt to only store currently processed nodes.
    //subtree_scores[idx] contains scores of the subtree rooted at node idx, for each state and site.
    let mut subtree_scores = observations;
    for values in &mut subtree_scores {
        values.apply(|p| *p = (*p).ln());
    }

    //conditioned_scores[idx] contains scores of the subtree rooted at node idx, for each state and site, conditioned on the parent state.
    let mut conditioned_scores = vec![DMatrix::zeros(0, 0); tree.len()];
    let mut choices =
        (mode == ReconstructionMode::Joint).then(|| vec![DMatrix::zeros(k, sites); tree.len()]);

    // Each node processes its whole sequence in both modes.
    for node in tree.postorder() {
        let idx = usize::from(node);
        for child in tree.children(node) {
            subtree_scores[idx] += &conditioned_scores[usize::from(child)];
        }
        if *node == tree.root {
            continue;
        }
        let p = &transitions[idx];
        conditioned_scores[idx] = DMatrix::from_element(k, sites, f64::NEG_INFINITY);
        for site in 0..sites {
            for parent in 0..k {
                let scores =
                    (0..k).map(|state| p[(parent, state)] + subtree_scores[idx][(state, site)]);
                // Even joint mode sums uncertain LEAF states: only ancestral
                // states are maximized. This preserves ambiguity handling.
                if mode == ReconstructionMode::Marginal || matches!(node, Leaf(_)) {
                    conditioned_scores[idx][(parent, site)] = log_sum_exp(scores);
                } else {
                    for (state, score) in scores.enumerate() {
                        if score > conditioned_scores[idx][(parent, site)] {
                            conditioned_scores[idx][(parent, site)] = score;
                            choices.as_mut().expect("Joint choices")[idx][(parent, site)] = state;
                        }
                    }
                }
            }
        }
        // Joint traceback only needs choices and the root's inside scores.
        // Releasing other inside matrices keeps the shared pass memory-efficient.
        if mode == ReconstructionMode::Joint {
            subtree_scores[idx] = DMatrix::zeros(0, 0);
        }
    }
    Upward {
        subtree_scores,
        conditioned_scores,
        choices,
    }
}

pub(crate) fn log_sum_exp(values: impl Iterator<Item = f64> + Clone) -> f64 {
    let maximum = values.clone().fold(f64::NEG_INFINITY, f64::max);
    if maximum == f64::NEG_INFINITY {
        return maximum;
    }
    maximum + values.map(|v| (v - maximum).exp()).sum::<f64>().ln()
}
