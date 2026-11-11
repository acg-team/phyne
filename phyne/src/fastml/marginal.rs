//! Marginal reconstruction: upward sums, outside messages, and posteriors.
use nalgebra::DMatrix;

use crate::evolutionary_models::ProbabilityMatrix;
use crate::tree::{NodeIdx::Internal, Tree};
use crate::Result;

use super::{log_sum_exp, upward, ReconstructionMode, Upward};

/// Conditional state probabilities using observations from the entire tree.
#[derive(Debug)]
pub struct MarginalReconstruction {
    /// Indexed by node index, then `(state, site)`. Each column sums to one.
    /// For letters these are conditional on the substitution model; inferred
    /// indel masks do not change or renormalize these letter probabilities.
    pub probabilities: Vec<ProbabilityMatrix>,
    /// Log likelihood of the observations, summing over states at every node.
    /// This is NOT the joint assignment score returned by `calc_joint`.
    pub log_likelihood: f64,
}

pub(super) fn infer(
    tree: &Tree,
    observations: Vec<ProbabilityMatrix>,
    transitions: &[ProbabilityMatrix],
    prior: &[f64],
) -> Result<MarginalReconstruction> {
    let k = prior.len();
    let sites = observations[usize::from(tree.root)].ncols();
    let Upward {
        mut subtree_scores,
        conditioned_scores,
        ..
    } = upward(
        tree,
        observations,
        transitions,
        ReconstructionMode::Marginal,
    );

    let root = usize::from(tree.root);

    // outside[v](s,x) includes the root prior and all data outside v's subtree.
    let mut outside = vec![DMatrix::from_element(k, sites, f64::NEG_INFINITY); tree.len()];
    for site in 0..sites {
        for state in 0..k {
            outside[root][(state, site)] = prior[state].ln();
        }
    }

    //incorporate information of parents, sibling and self for all children of a node

    for node_idx in tree.preorder() {
        if let Internal(parent) = node_idx {
            let children = tree.children(node_idx);
            let left = usize::from(children[0]);
            let right = usize::from(children[1]);

            for (child, sibling) in [(left, right), (right, left)] {
                let p = &transitions[child];
                let context = &outside[*parent] + &conditioned_scores[sibling];

                for site in 0..sites {
                    for child_state in 0..k {
                        outside[child][(child_state, site)] =
                            log_sum_exp((0..k).map(|parent_state| {
                                context[(parent_state, site)] + p[(parent_state, child_state)]
                            }));
                    }
                }
            }
        }
    }

    //turn marginal score into probabilities that sum to 1; normalizer takes role of log marginal likelihood
    let mut normalizers = vec![0.0; sites];
    for site in 0..sites {
        normalizers[site] = log_sum_exp(
            (0..k).map(|state| prior[state].ln() + subtree_scores[root][(state, site)]),
        );
    }

    for node_idx in tree.preorder() {
        let idx = usize::from(node_idx);

        subtree_scores[idx] += outside[idx].clone();
        for site in 0..sites {
            for state in 0..k {
                subtree_scores[idx][(state, site)] =
                    (subtree_scores[idx][(state, site)] - normalizers[site]).exp();
            }
        }
    }
    Ok(MarginalReconstruction {
        probabilities: subtree_scores,
        log_likelihood: normalizers.iter().sum(),
    })
}
