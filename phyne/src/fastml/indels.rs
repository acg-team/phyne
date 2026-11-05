//! Simple indel coding plus independent binary CTMC posterior reconstruction.
use std::collections::BTreeSet;

use nalgebra::DMatrix;

use crate::alignment::{Alignment, MSA};
use crate::phylo_info::PhyloInfo;
use crate::{Error, Result};

use super::marginal;

/// Fixed rates per unit of tree branch length. State 0 = gap event absent;
/// state 1 = gap event present. These are rates of binary gap-event evolution,
/// not parameters of a full sequence-length insertion/deletion process.
#[derive(Clone, Copy, Debug)]
pub struct IndelModel {
    gain: f64,
    loss: f64,
}

impl IndelModel {
    /// Rates must be finite and strictly positive, with a finite sum.
    /// A stationary root prior is used: P(gap event) = gain / (gain + loss).
    pub fn new(gap_gain_rate: f64, gap_loss_rate: f64) -> Result<Self> {
        if !gap_gain_rate.is_finite()
            || !gap_loss_rate.is_finite()
            || gap_gain_rate <= 0.0
            || gap_loss_rate <= 0.0
            || !(gap_gain_rate + gap_loss_rate).is_finite()
        {
            return Err(Error::AncestralAlignment(
                "Indel rates must be finite, strictly positive, and have a finite sum".into(),
            ));
        }
        Ok(Self {
            gain: gap_gain_rate,
            loss: gap_loss_rate,
        })
    }

    fn prior(self) -> [f64; 2] {
        let total = self.gain + self.loss;
        [self.loss / total, self.gain / total]
    }

    fn transition(self, time: f64) -> DMatrix<f64> {
        let pi = self.prior();
        // expm1 retains precision on very short branches.
        let change = -(-(self.gain + self.loss) * time).exp_m1();
        DMatrix::from_row_slice(
            2,
            2,
            &[
                1.0 - pi[1] * change,
                pi[1] * change,
                pi[0] * change,
                1.0 - pi[0] * change,
            ],
        )
    }
}

/// A maximal observed gap run, with zero-based, half-open MSA coordinates.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct GapEvent {
    pub start: usize,
    pub end: usize,
}

#[derive(Debug)]
pub struct IndelReconstruction {
    /// One binary character per distinct observed gap interval.
    pub events: Vec<GapEvent>,
    /// Indexed [node][event]: posterior probability of gap-event presence.
    pub gap_probabilities: Vec<Vec<f64>>,
    /// Indexed [node][site]. Internal gaps are called when an event's posterior
    /// is strictly greater than 0.5; overlapping called intervals are unioned.
    /// Exact ties retain a letter. Leaf masks preserve the input mapping.
    pub gaps: Vec<Vec<bool>>,
    /// Likelihood of the separately coded binary characters, not of letters.
    pub log_likelihood: f64,
}

fn gap_runs(present: &[bool]) -> Vec<GapEvent> {
    let mut runs = Vec::new();
    let mut site = 0;
    while site < present.len() {
        if present[site] {
            site += 1;
            continue;
        }
        let start = site;
        while site < present.len() && !present[site] {
            site += 1;
        }
        runs.push(GapEvent { start, end: site });
    }
    runs
}

// Simple indel coding: exact run = present; a different, longer run covering
// the event = unknown/inapplicable; otherwise the particular event is absent.
// Partially overlapping runs remain different, observed binary events.
fn event_encoding(runs: &[GapEvent], event: GapEvent) -> [f64; 2] {
    for run in runs {
        if *run == event {
            return [0.0, 1.0]; // Event present.
        }

        if run.start <= event.start && run.end >= event.end {
            return [1.0, 1.0]; // Enclosing gap: unknown.
        }
    }

    [1.0, 0.0] // Event absent.
}

pub(super) fn reconstruct(
    phylo: &PhyloInfo<MSA>,
    model: IndelModel,
) -> Result<IndelReconstruction> {
    let tree = &phylo.tree;
    let sites = phylo.msa.len();
    let mut runs = vec![Vec::new(); tree.len()];
    let mut gaps = vec![vec![false; sites]; tree.len()];
    let mut events = BTreeSet::new();
    for leaf in tree.leaves() {
        let idx = usize::from(leaf.idx);
        let present: Vec<_> = phylo
            .msa
            .leaf_map(&leaf.idx)
            .iter()
            .map(Option::is_some)
            .collect();
        runs[idx] = gap_runs(&present);
        events.extend(runs[idx].iter().copied());
        gaps[idx] = present.iter().map(|p| !p).collect();
    }
    let events: Vec<_> = events.into_iter().collect();
    let mut observations = vec![DMatrix::from_element(2, events.len(), 1.0); tree.len()];
    for leaf in tree.leaves() {
        let idx = usize::from(leaf.idx);
        for (event_idx, &event) in events.iter().enumerate() {
            observations[idx]
                .column_mut(event_idx)
                .copy_from_slice(&event_encoding(&runs[idx], event));
        }
    }
    let mut transitions = vec![DMatrix::zeros(0, 0); tree.len()];
    for node in tree.iter() {
        transitions[usize::from(node.idx)] = model.transition(node.blen).map(|p| p.ln());
    }
    let posterior = marginal::infer(tree, observations, &transitions, &model.prior())?;
    let gap_probabilities: Vec<Vec<f64>> = posterior
        .probabilities
        .iter()
        .map(|p| p.row(1).iter().copied().collect())
        .collect();
    for node in tree.internals() {
        let idx = usize::from(node.idx);
        for (event_idx, event) in events.iter().enumerate() {
            if gap_probabilities[idx][event_idx] > 0.5 {
                gaps[idx][event.start..event.end].fill(true);
            }
        }
    }
    Ok(IndelReconstruction {
        events,
        gap_probabilities,
        gaps,
        log_likelihood: posterior.log_likelihood,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coding_distinguishes_exact_containing_and_partial_runs() {
        let event = GapEvent { start: 2, end: 4 };
        assert_eq!(event_encoding(&[event], event), [0.0, 1.0]);
        assert_eq!(
            event_encoding(&[GapEvent { start: 1, end: 5 }], event),
            [1.0, 1.0]
        );
        assert_eq!(
            event_encoding(&[GapEvent { start: 3, end: 5 }], event),
            [1.0, 0.0]
        );
        assert_eq!(event_encoding(&[], event), [1.0, 0.0]);
    }

    #[test]
    fn binary_transitions_match_closed_form_and_identity() {
        let model = IndelModel::new(0.2, 0.8).unwrap();
        let p = model.transition(0.7);
        assert!((p[(0, 1)] - 0.2 * (1.0 - (-0.7_f64).exp())).abs() < 1e-14);
        assert!((p[(1, 0)] - 0.8 * (1.0 - (-0.7_f64).exp())).abs() < 1e-14);
        assert_eq!(model.transition(0.0), DMatrix::identity(2, 2));
        assert!(IndelModel::new(0.0, 1.0).is_err());
        assert!(IndelModel::new(f64::NAN, 1.0).is_err());
    }
}

#[cfg(test)]
mod exhaustive_tests {
    use super::*;
    use crate::alignment::Sequences;
    use crate::tree::NodeIdx::{Internal, Leaf};

    #[test]
    fn event_posteriors_match_all_binary_internal_assignments() {
        let phylo = MSA::from_aligned(
            Sequences::new_unchecked(vec![
                crate::record_wo_desc!("A", b"A--A"),
                crate::record_wo_desc!("B", b"A--A"),
                crate::record_wo_desc!("C", b"AAAA"),
            ]),
            crate::tree!("((A:0.1,B:0.2)I:0.3,C:0.4)R;"),
        )
        .unwrap();
        let model = IndelModel::new(0.2, 0.8).unwrap();
        let result = reconstruct(&phylo, model).unwrap();
        let internals = phylo.tree.internals();
        let mut masses = vec![[0.0; 2]; phylo.tree.len()];
        let mut total = 0.0;
        for assignment in 0..2usize.pow(internals.len() as u32) {
            let mut states = vec![0; phylo.tree.len()];
            for (bit, node) in internals.iter().enumerate() {
                states[usize::from(node.idx)] = (assignment >> bit) & 1;
            }
            let mut probability = model.prior()[states[usize::from(phylo.tree.root)]];
            for node in phylo.tree.iter() {
                if node.idx == phylo.tree.root {
                    continue;
                }
                let parent = usize::from(phylo.tree.parent(&node.idx).unwrap());
                let state = match node.idx {
                    Internal(_) => states[usize::from(node.idx)],
                    Leaf(_) => {
                        if node.id == "C" {
                            0
                        } else {
                            1
                        }
                    }
                };
                // Closed-form binary transition, independently of transition().
                let pi = [0.8, 0.2];
                let decay = (-node.blen).exp();
                let transition = pi[state]
                    + if states[parent] == state {
                        (1.0 - pi[state]) * decay
                    } else {
                        -pi[state] * decay
                    };
                probability *= transition;
            }
            total += probability;
            for node in &internals {
                let idx = usize::from(node.idx);
                masses[idx][states[idx]] += probability;
            }
        }
        assert!((result.log_likelihood - total.ln()).abs() < 1e-12);
        for node in internals {
            let idx = usize::from(node.idx);
            assert!((result.gap_probabilities[idx][0] - masses[idx][1] / total).abs() < 1e-12);
        }
    }
}
