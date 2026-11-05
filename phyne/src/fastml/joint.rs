//! Joint reconstruction: maximize ancestral assignments and trace choices back.

use crate::evolutionary_models::ProbabilityMatrix;
use crate::tree::{NodeIdx::Leaf, Tree};

use super::{log_sum_exp, upward, ReconstructionMode};

pub(super) fn infer(
    tree: &Tree,
    observations: Vec<ProbabilityMatrix>,
    transitions: &[ProbabilityMatrix],
    prior: &[f64], //character frequences at the root
) -> (Vec<Vec<usize>>, f64) {
    let root = usize::from(tree.root);
    let n_char = prior.len();
    let n_sites = observations[root].ncols();

    //upward pass
    let upward = upward(tree, observations, transitions, ReconstructionMode::Joint);
    let choices = upward.choices.as_ref().expect("Joint pass stores choices");

    // Select the root state at each site from the completed upward scores.
    let mut ideal_chars = vec![vec![0usize; n_sites]; tree.len()];
    let mut total_score = 0.0;
    for site in 0..n_sites {
        let scores =
            (0..n_char).map(|state| prior[state].ln() + upward.subtree_scores[root][(state, site)]);
        if matches!(tree.root, Leaf(_)) {
            total_score += log_sum_exp(scores);
        } else {
            let mut best = f64::NEG_INFINITY;
            for (state, score) in scores.enumerate() {
                if score > best {
                    best = score;
                    ideal_chars[root][site] = state;
                }
            }
            total_score += best;
        }
    }
    // Downward pass: leaf output is copied from the input, so only trace ancestors.
    for node_idx in tree.preorder() {
        if *node_idx == tree.root || matches!(node_idx, Leaf(_)) {
            continue;
        }
        let idx = usize::from(node_idx);
        let parent = usize::from(
            tree.parent(node_idx)
                .expect("Non-root node must have a parent"),
        );
        for site in 0..n_sites {
            let parent_state = ideal_chars[parent][site];
            ideal_chars[idx][site] = choices[idx][(parent_state, site)];
        }
    }
    (ideal_chars, total_score)
}
