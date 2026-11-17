//! Marginal reconstruction: upward sums, outside messages, and posteriors.
use nalgebra::DMatrix;

use super::{log_sum_exp, upward, ReconstructionMode, Upward};
use crate::evolutionary_models::ProbabilityMatrix;
use crate::fastml::IndelReconstruction;
use crate::tree::{Tree};
use crate::{Error, Result};

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
    indels_forward: Option<&IndelReconstruction>,
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
        indels_forward,
    );

    let root = usize::from(tree.root);
    let log_prior: Vec<f64> = prior.iter().map(|p| p.ln()).collect();

    let is_gap =
        |node: usize, site: usize| indels_forward.is_some_and(|indels| indels.gaps[node][site]);

    let mut outside = vec![DMatrix::from_element(k, sites, f64::NEG_INFINITY); tree.len()];

    // The tree root starts a component wherever it is present.
    for site in 0..sites {
        if !is_gap(root, site) {
            outside[root].column_mut(site).copy_from_slice(&log_prior);
        }
    }

    // Calculate outside scores, restarting below gaps.
    for node_idx in tree.preorder() {
        let parent = usize::from(node_idx);
        let children = tree.children(node_idx);

        if children.is_empty() {
            continue;
        }

        let left = usize::from(&children[0]);
        let right = usize::from(&children[1]);

        for (child, sibling) in [(left, right), (right, left)] {
            let p = &transitions[child];
            let context = &outside[parent] + &conditioned_scores[sibling];

            for site in 0..sites {
                if is_gap(child, site) {
                    continue;
                }

                if is_gap(parent, site) {
                    // This child starts a new component.
                    outside[child].column_mut(site).copy_from_slice(&log_prior);
                    continue;
                }

                for child_state in 0..k {
                    outside[child][(child_state, site)] = log_sum_exp((0..k).map(|parent_state| {
                        context[(parent_state, site)] + p[(parent_state, child_state)]
                    }));
                }
            }
        }
    }

    let mut log_likelihood = 0.0;

    for node_idx in tree.preorder() {
        let idx = usize::from(node_idx);
        let parent = tree.parent(node_idx).map(usize::from);

        // Combine evidence from both directions.
        subtree_scores[idx] += &outside[idx];

        for site in 0..sites {
            if is_gap(idx, site) {
                // No letter distribution at an absent position.
                subtree_scores[idx].column_mut(site).fill(0.0);
                continue;
            }

            // Inside + outside includes exactly this component's evidence.
            let normalizer = log_sum_exp(subtree_scores[idx].column(site).iter().copied());

            if !normalizer.is_finite() {
                return Err(Error::AncestralAlignment(format!(
                    "No finite character likelihood at node {idx}, site {site}"
                )));
            }

            let starts_component = match parent {
                None => true,
                Some(parent) => is_gap(parent, site),
            };

            if starts_component {
                log_likelihood += normalizer;
            }

            subtree_scores[idx]
                .column_mut(site)
                .apply(|value| *value = (*value - normalizer).exp());
        }
    }

    Ok(MarginalReconstruction {
        probabilities: subtree_scores,
        log_likelihood,
    })
}
