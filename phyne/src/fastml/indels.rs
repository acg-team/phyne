//! Simple indel coding plus independent binary CTMC posterior reconstruction.
use std::collections::BTreeSet;

use nalgebra::DMatrix;

use crate::alignment::{Alignment, AncestralAlignment, Sequences};
use crate::evolutionary_models::ProbabilityMatrix;
use crate::phylo_info::PhyloInfo;
use crate::{Error, Result};
use bio::io::fasta::Record;

use crate::alphabets::{AMB_CHAR, GAP};
use crate::asr::AncestralSequenceReconstruction;
use crate::fastml::marginal;
use crate::{aligned_seq, record};

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

    fn transition(self, time: f64) -> ProbabilityMatrix {
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

pub fn reconstruct<A: Alignment>(
    phylo: &PhyloInfo<A>,
    indel_model: IndelModel,
) -> IndelReconstruction {
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
        transitions[usize::from(node.idx)] = indel_model.transition(node.blen).map(|p| p.ln());
    }
    let posterior = marginal::infer(
        phylo,
        &indel_model.prior(),
        &transitions,
        observations,
        None,
    );
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
    IndelReconstruction {
        events,
        gap_probabilities,
        gaps,
        log_likelihood: posterior.log_likelihood,
    }
}

impl<A: Alignment, AA: AncestralAlignment> AncestralSequenceReconstruction<A, AA> for IndelModel {
    // Pipeline inputs are already validated and named upstream.
    fn reconstruct_ancestral_seqs(&self, phylo: PhyloInfo<A>) -> Result<PhyloInfo<AA>> {
        Ok(self.reconstruct_ancestral_seqs_unchecked(phylo))
    }

    fn reconstruct_ancestral_seqs_unchecked(&self, phylo: PhyloInfo<A>) -> PhyloInfo<AA> {
        let mut records = Vec::with_capacity(phylo.tree.len());
        for leaf in phylo.tree.leaves() {
            let original = phylo.msa.seqs().record_by_id(&leaf.id);
            let aligned = aligned_seq!(phylo.msa.leaf_map(&leaf.idx), original.seq());
            records.push(record!(&leaf.id, original.desc(), &aligned));
        }
        records.extend(get_ancestral_records(&phylo, *self));
        let sequences = Sequences::with_alphabet_unchecked(records, phylo.msa.seqs().alphabet());
        AA::from_aligned_with_ancestral_unchecked(sequences, phylo.tree)
    }
}

fn get_ancestral_records<A: Alignment>(phylo: &PhyloInfo<A>, model: IndelModel) -> Vec<Record> {
    let result = reconstruct(phylo, model);
    phylo
        .tree
        .internals()
        .iter()
        .map(|node| {
            let sequence: Vec<u8> = result.gaps[usize::from(node.idx)]
                .iter()
                .map(|&gap| if gap { GAP } else { AMB_CHAR })
                .collect();
            record!(&node.id, None, &sequence)
        })
        .collect()
}

//#[cfg(test)]
//mod tests;
