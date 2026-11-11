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
mod pruning;
use pruning::{log_sum_exp, upward, Upward};
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
    indel_model: Option<IndelModel>,
}

impl<Q: QMatrix> FastML<Q> {
    pub fn new(model: SubstModel<Q>) -> Self {
        Self {
            model,
            mode: ReconstructionMode::Joint,
            indel_model: None,
        }
    }

    pub fn with_mode(mut self, mode: ReconstructionMode) -> Self {
        self.mode = mode;
        self
    }

    pub fn with_indels(mut self, model: IndelModel) -> Self {
        self.indel_model = Some(model);
        self
    }

    fn validate_input(&self, phylo: &PhyloInfo<MSA>) -> Result<()> {
        if phylo.msa.alphabet() != SubstModel::<Q>::alphabet() {
            return Err(Error::Alphabet(
                "FastML model and alignment alphabets differ".into(),
            ));
        }
        for node in phylo.tree.iter() {
            if node.idx != phylo.tree.root && (!node.blen.is_finite() || node.blen < 0.0) {
                return Err(Error::AncestralAlignment(
                    "FastML branch lengths must be finite and nonnegative".into(),
                ));
            }
        }
        Ok(())
    }

    fn calc_joint(&self, phylo: &PhyloInfo<MSA>) -> (Vec<Vec<usize>>, f64) {
        joint::infer(
            &phylo.tree,
            self.observations(phylo),
            &self.precompute_log_transitions(&phylo.tree),
            self.model.freqs().as_slice(),
        )
    }

    pub fn marginal_posteriors(&self, phylo: &PhyloInfo<MSA>) -> Result<MarginalReconstruction> {
        self.validate_input(phylo)?;
        marginal::infer(
            &phylo.tree,
            self.observations(phylo),
            &self.precompute_log_transitions(&phylo.tree),
            self.model.freqs().as_slice(),
        )
    }

    fn observations(&self, phylo: &PhyloInfo<MSA>) -> Vec<ProbabilityMatrix> {
        let k = phylo.msa.alphabet().len();
        let mut observations =
            vec![DMatrix::from_element(k, phylo.msa.len(), 1.0); phylo.tree.len()];
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

    pub fn reconstruct_indels(
        &self,
        phylo: &PhyloInfo<MSA>,
    ) -> Result<Option<IndelReconstruction>> {
        self.validate_input(phylo)?;
        self.indel_model
            .map(|model| indels::reconstruct(phylo, model))
            .transpose()
    }

    // One probability transition matrix per node; pruning converts these to logs.
    fn precompute_log_transitions(&self, tree: &Tree) -> Vec<ProbabilityMatrix> {
        let mut transitions =
            vec![DMatrix::from_element(self.model.freqs().len(), tree.len(), 0.0); tree.len()];
        for node in tree.iter() {
            transitions[usize::from(node.idx)] = self.model.p(node.blen).map(|p| p.ln());
        }
        transitions
    }

    fn selected_states(&self, phylo: &PhyloInfo<MSA>) -> Result<Vec<Vec<usize>>> {
        match self.mode {
            ReconstructionMode::Joint => {
                let (states, score) = self.calc_joint(phylo);
                if !score.is_finite() {
                    return Err(Error::AncestralAlignment("No finite FastML reconstruction score; check observations and branch lengths".into()));
                }
                Ok(states)
            }
            ReconstructionMode::Marginal => {
                let result = self.marginal_posteriors(phylo)?;
                Ok(result
                    .probabilities
                    .iter()
                    .map(|p| {
                        (0..p.ncols())
                            .map(|site| {
                                let mut best = 0;
                                for state in 1..p.nrows() {
                                    if p[(state, site)] > p[(best, site)] {
                                        best = state;
                                    }
                                }
                                best
                            })
                            .collect()
                    })
                    .collect())
            }
        }
    }

    fn finish(&self, phylo: PhyloInfo<MSA>, states: Vec<Vec<usize>>) -> Result<PhyloInfo<MASA>> {
        let indels = self.reconstruct_indels(&phylo)?;
        Ok(self.build_alignment(
            phylo,
            states,
            indels
                .as_ref()
                .map(|r: &IndelReconstruction| r.gaps.as_slice()),
        ))
    }

    fn build_alignment(
        &self,
        phylo: PhyloInfo<MSA>,
        ideal_chars: Vec<Vec<usize>>,
        gaps: Option<&[Vec<bool>]>,
    ) -> PhyloInfo<MASA> {
        let mut all_records = Vec::with_capacity(phylo.tree.len());
        for leaf in phylo.tree.leaves() {
            let original = phylo.msa.seqs().record_by_id(&leaf.id);
            let aligned = aligned_seq!(phylo.msa.leaf_map(&leaf.idx), original.seq());
            all_records.push(record!(&leaf.id, original.desc(), &aligned));
        }
        let alphabet = phylo.msa.seqs().alphabet();
        //transform into letters
        for node in phylo.tree.internals() {
            let sequence: Vec<u8> = ideal_chars[usize::from(node.idx)]
                .iter()
                .enumerate()
                .map(|(site, &state)| {
                    if gaps.is_some_and(|g| g[usize::from(node.idx)][site]) {
                        GAP
                    } else {
                        alphabet.symbols()[state]
                    }
                })
                .collect();
            all_records.push(record!(&node.id, None, &sequence));
        }
        let all_seqs = Sequences::with_alphabet_unchecked(all_records, alphabet);
        MASA::from_aligned_with_ancestral_unchecked(all_seqs, phylo.tree)
    }
}

impl<Q: QMatrix> AncestralSequenceReconstruction<MSA, MASA> for FastML<Q> {
    fn reconstruct_ancestral_seqs(&self, mut phylo: PhyloInfo<MSA>) -> Result<PhyloInfo<MASA>> {
        self.validate_input(&phylo)?;
        set_missing_tree_node_ids(&mut phylo.tree)?;
        let states = self.selected_states(&phylo)?;
        self.finish(phylo, states)
    }

    /// Requires matching alphabets, unique node IDs, finite nonnegative branches,
    /// and a positive-probability reconstruction. Panics if inference fails.
    fn reconstruct_ancestral_seqs_unchecked(&self, phylo: PhyloInfo<MSA>) -> PhyloInfo<MASA> {
        let states = self
            .selected_states(&phylo)
            .expect("FastML unchecked preconditions");
        self.finish(phylo, states)
            .expect("FastML unchecked preconditions")
    }
}

#[cfg(test)]
mod tests;
