//! Joint reconstruction: maximize ancestral assignments and trace choices back.

use crate::evolutionary_models::ProbabilityMatrix;
use crate::fastml::IndelReconstruction;
use crate::tree::{NodeIdx::Leaf, Tree};

use super::{log_sum_exp, upward, ReconstructionMode};

pub(super) fn infer(
    tree: &Tree,
    observations: Vec<ProbabilityMatrix>,
    transitions: &[ProbabilityMatrix],
    prior: &[f64], //character frequences at the root
    indels_forward: Option<&IndelReconstruction>,
) -> (Vec<Vec<usize>>, f64) {
    let root = usize::from(tree.root);
    let n_char = prior.len();
    let n_sites = observations[root].ncols();

    //upward pass
    let upward = upward(
        tree,
        observations,
        transitions,
        ReconstructionMode::Joint,
        indels_forward,
    );
    let choices = upward.choices.as_ref().expect("Joint pass stores choices");

    // Select the root state at each site from the completed upward scores.
    let mut ideal_chars = vec![vec![0usize; n_sites]; tree.len()];
    let mut total_score = 0.0;
    for node_idx in tree.preorder() {
        let idx = usize::from(node_idx);
        let parent = tree.parent(node_idx).map(usize::from);

        for site in 0..n_sites {
            let is_gap = |node: usize| indels_forward.is_some_and(|indels| indels.gaps[node][site]);

            // An absent character has no letter to reconstruct.
            if is_gap(idx) {
                continue;
            }

            // A present node starts a new component if it is the tree root
            // or its parent has no character at this site.
            let starts_component = match parent {
                None => true,
                Some(parent) => is_gap(parent),
            };

            if starts_component {
                let scores = (0..n_char)
                    .map(|state| prior[state].ln() + upward.subtree_scores[idx][(state, site)]);

                if matches!(node_idx, Leaf(_)) {
                    // Preserve the observed leaf; sum compatible leaf states.
                    total_score += log_sum_exp(scores);
                } else {
                    let mut best = f64::NEG_INFINITY;

                    for (state, score) in scores.enumerate() {
                        if score > best {
                            best = score;
                            ideal_chars[idx][site] = state;
                        }
                    }

                    total_score += best;
                }
            } else if !matches!(node_idx, Leaf(_)) {
                let parent = parent.expect("Non-root node needs a parent");
                let parent_state = ideal_chars[parent][site];

                ideal_chars[idx][site] = choices[idx][(parent_state, site)];
            }
        }
    }
    (ideal_chars, total_score)
}
