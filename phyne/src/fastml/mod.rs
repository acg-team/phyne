//TODO: add gamma rates / branch-and-bound
// allow rates to vary among gap events
// fastml also includes optimizing branch lengths
// also offers posterior sequence sampling and lists of alternative ancestral sequences
// implement parsimony_presence_absence implementation for indel handling
// test against FastML paper's example data and results

use nalgebra::DMatrix;

use crate::alignment::{Alignment, AncestralAlignment, Sequences, MASA, MSA};
use crate::alphabets::GAP;
use crate::asr::AncestralSequenceReconstruction;
use crate::evolutionary_models::{EvoModel, ProbabilityMatrix};
use crate::phylo_info::{set_missing_tree_node_ids, PhyloInfo};
use crate::substitution_models::{QMatrix, SubstModel};
use crate::tree::Tree;
use crate::{aligned_seq, record, Error, Result};

mod joint;
mod marginal;
use marginal::MarginalReconstruction;
mod upwards;
use upwards::{log_sum_exp, upward, Upward};
mod indels;
pub use indels::{GapEvent, IndelModel, IndelReconstruction};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ReconstructionMode {
    /// Most probable assignment to all internal nodes together.
    #[default]
    Joint,
    /// Most probable state independently at each node, using full-tree posteriors.
    Marginal,
}

pub struct FastML<Q: QMatrix> {
    model: SubstModel<Q>,
    mode: ReconstructionMode,
}

pub struct FastMLNoIndelsDuringInference<Q: QMatrix> {
    model: SubstModel<Q>,
    mode: ReconstructionMode,
}

impl<Q: QMatrix> FastMLNoIndelsDuringInference<Q> {
    pub fn new(model: SubstModel<Q>) -> Self {
        Self {
            model,
            mode: ReconstructionMode::Joint,
        }
    }

    pub fn with_mode(mut self, mode: ReconstructionMode) -> Self {
        self.mode = mode;
        self
    }
}

impl<Q: QMatrix> FastML<Q> {
    pub fn new(model: SubstModel<Q>) -> Self {
        Self {
            model,
            mode: ReconstructionMode::Joint,
        }
    }

    pub fn with_mode(mut self, mode: ReconstructionMode) -> Self {
        self.mode = mode;
        self
    }
}

fn selected_states<Q: QMatrix, AA: AncestralAlignment>(
    model: &SubstModel<Q>,
    mode: ReconstructionMode,
    phylo: &PhyloInfo<AA>,
    gaps: Option<&[Vec<bool>]>,
) -> Vec<Vec<usize>> {
    match mode {
        ReconstructionMode::Joint => {
            joint::infer(
                phylo,
                model.freqs().as_slice(),
                &precompute_log_transitions(&model, &phylo.tree),
                observations(phylo),
                gaps,
            )
            .0
        }
        ReconstructionMode::Marginal => marginal::infer(
            phylo,
            model.freqs().as_slice(),
            &precompute_log_transitions(&model, &phylo.tree),
            observations(phylo),
            gaps,
        )
        .probabilities
        .iter()
        .map(|p| {
            p.column_iter()
                .map(|column| {
                    let mut best = 0;
                    for state in 1..column.len() {
                        if column[state] > column[best] {
                            best = state;
                        }
                    }
                    best
                })
                .collect()
        })
        .collect(),
    }
}

fn gap_mask<AA: AncestralAlignment>(phylo: &PhyloInfo<AA>) -> Vec<Vec<bool>> {
    let mut gaps = vec![Vec::new(); phylo.tree.len()];
    for node in phylo.tree.iter() {
        let mapping = match node.idx {
            crate::tree::NodeIdx::Leaf(_) => phylo.msa.leaf_map(&node.idx),
            crate::tree::NodeIdx::Internal(_) => phylo.msa.ancestral_map(&node.idx),
        };
        gaps[usize::from(node.idx)] = mapping.iter().map(Option::is_none).collect();
    }
    gaps
}

fn build_alignment<AA: AncestralAlignment>(
    phylo: PhyloInfo<AA>,
    states: Vec<Vec<usize>>,
) -> PhyloInfo<AA> {
    let mut records = Vec::with_capacity(phylo.tree.len());
    for leaf in phylo.tree.leaves() {
        let original = phylo.msa.seqs().record_by_id(&leaf.id);
        let aligned = aligned_seq!(phylo.msa.leaf_map(&leaf.idx), original.seq());
        records.push(record!(&leaf.id, original.desc(), &aligned));
    }
    let alphabet = phylo.msa.seqs().alphabet();
    for node in phylo.tree.internals() {
        let idx = usize::from(node.idx);
        let sequence: Vec<u8> = phylo
            .msa
            .ancestral_map(&node.idx)
            .iter()
            .enumerate()
            .map(|(site, position)| {
                if position.is_none() {
                    GAP
                } else {
                    alphabet.symbols()[states[idx][site]]
                }
            })
            .collect();
        let original = phylo.msa.ancestral_seqs().record_by_id(&node.id);
        records.push(record!(&node.id, original.desc(), &sequence));
    }
    let sequences = Sequences::with_alphabet_unchecked(records, alphabet);
    AA::from_aligned_with_ancestral_unchecked(sequences, phylo.tree)
}

fn precompute_log_transitions<Q: QMatrix>(
    model: &SubstModel<Q>,
    tree: &Tree,
) -> Vec<ProbabilityMatrix> {
    let mut transitions = vec![DMatrix::zeros(0, 0); tree.len()];
    for node in tree.iter() {
        transitions[usize::from(node.idx)] = model.p(node.blen).map(|p| p.ln());
    }
    transitions
}

fn observations<A: Alignment>(phylo: &PhyloInfo<A>) -> Vec<ProbabilityMatrix> {
    let k = phylo.msa.alphabet().len();
    let mut observations = vec![DMatrix::from_element(k, phylo.msa.len(), 1.0); phylo.tree.len()];
    for leaf in phylo.tree.leaves() {
        let idx = usize::from(leaf.idx);
        let record = phylo.msa.seqs().record_by_id(&leaf.id);
        for (site, position) in phylo.msa.leaf_map(&leaf.idx).iter().enumerate() {
            if let Some(position) = position {
                observations[idx]
                    .column_mut(site)
                    .copy_from(phylo.msa.alphabet().char_encoding(record.seq()[*position]));
            }
        }
    }
    observations
}

impl<Q: QMatrix, AA: AncestralAlignment> AncestralSequenceReconstruction<AA, AA> for FastML<Q> {
    // Validation belongs to the preceding pipeline stage for both entry points.
    fn reconstruct_ancestral_seqs(&self, phylo: PhyloInfo<AA>) -> Result<PhyloInfo<AA>> {
        Ok(self.reconstruct_ancestral_seqs_unchecked(phylo))
    }

    fn reconstruct_ancestral_seqs_unchecked(&self, phylo: PhyloInfo<AA>) -> PhyloInfo<AA> {
        let gaps = gap_mask(&phylo);
        let states = selected_states(&self.model, self.mode, &phylo, Some(&gaps));
        build_alignment(phylo, states)
    }
}

//will still apply gaps if there exist in AncestralALignment
impl<Q: QMatrix, AA: AncestralAlignment> AncestralSequenceReconstruction<AA, AA>
    for FastMLNoIndelsDuringInference<Q>
{
    // Validation belongs to the preceding pipeline stage for both entry points.
    fn reconstruct_ancestral_seqs(&self, phylo: PhyloInfo<AA>) -> Result<PhyloInfo<AA>> {
        Ok(self.reconstruct_ancestral_seqs_unchecked(phylo))
    }

    fn reconstruct_ancestral_seqs_unchecked(&self, phylo: PhyloInfo<AA>) -> PhyloInfo<AA> {
        let states = selected_states(&self.model, self.mode, &phylo, None);
        build_alignment(phylo, states)
    }
}
mod tests;
