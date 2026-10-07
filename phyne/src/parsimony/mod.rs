use std::default;
use std::fmt::{Debug, Display};

use bio::io::fasta::Record;
use log::info;
use nalgebra::DMatrix;

use crate::aligned_seq;
use crate::alignment::{Aligner, Alignment, InternalAlignments, PairwiseAlignment, Sequences};
use crate::alphabets::ParsimonySet;
use crate::evolutionary_models::EvoModel;
use crate::phylo_info::PhyloInfo;
use crate::tree::{
    NodeIdx::{Internal as Int, Leaf},
    Tree,
};

pub mod cost;
pub use cost::*;
pub mod scoring;
use scoring::*;
pub mod matrices;
pub(crate) use matrices::*;
pub(crate) mod helpers;
pub(crate) use helpers::*;

pub(crate) type CostMatrix = DMatrix<f64>;

pub trait ParsimonyModel: EvoModel + Debug + Display + Clone {
    fn scoring(&self, time: f64, diagonals: &DiagonalZeros, rounding: &Rounding) -> CostMatrix;
}

pub struct ParsimonyAligner<PS: ParsimonyScoring> {
    pub scoring: PS,
}

impl<PS: ParsimonyScoring + Clone, A: Alignment> Aligner<A> for ParsimonyAligner<PS> {
    fn align_unchecked(&self, seqs: &Sequences, tree: &Tree) -> A {
        self.align_with_scores(seqs, tree).0
    }
}

impl default::Default for ParsimonyAligner<SimpleScoring> {
    fn default() -> Self {
        ParsimonyAligner {
            scoring: SimpleScoring::new(1.0, GapCost::new(2.5, 0.5)),
        }
    }
}

impl<'a, PS: ParsimonyScoring + Clone> ParsimonyAligner<PS> {
    pub fn new(scoring: PS) -> ParsimonyAligner<PS> {
        ParsimonyAligner { scoring }
    }

    pub fn align_with_scores<A: Alignment>(
        &self,
        seqs: &'a Sequences,
        tree: &'a Tree,
    ) -> (A, Vec<f64>) {
        info!("Starting the IndelMAP alignment");

        let order = tree.postorder();

        let mut node_info = vec![Vec::<ParsimonySite>::new(); tree.len()];
        let mut alignments = InternalAlignments::with_capacity(tree.internals().len());
        let mut scores = vec![0.0; tree.len()];

        for &node_idx in order {
            info!("Processing {}{}", node_idx, tree.node(&node_idx).id);
            match node_idx {
                Int(idx) => {
                    let child = tree.children(&node_idx);
                    let x_idx = child[0];
                    let y_idx = child[1];

                    let x_info = &node_info[usize::from(child[0])];
                    let x_blen = tree.node(&child[0]).blen;

                    let y_info = &node_info[usize::from(child[1])];
                    let y_blen = tree.node(&child[1]).blen;

                    info!("Aligning sequences at nodes {x_idx} and {y_idx}");
                    let (info, alignment, score) =
                        self.pairwise_align(x_info, x_blen, y_info, y_blen, rng_len);
                    node_info[idx] = info;
                    alignments.insert(node_idx, alignment);
                    scores[idx] = score;
                    info!("Alignment complete with score {score}");
                }
                Leaf(idx) => {
                    let pars_sets = seqs
                        .record_by_id(tree.node_id(&node_idx))
                        .seq()
                        .iter()
                        .map(|c| seqs.alphabet().parsimony_set(c))
                        .collect::<Vec<_>>();
                    node_info[idx] = pars_sets
                        .into_iter()
                        .map(|set: &ParsimonySet| ParsimonySite::leaf(set.clone()))
                        .collect();
                    info!("Processed leaf node");
                }
            }
        }
        info!("Finished IndelMAP alignment");

        // TODO: to avoid having a fn new(tree, seqs, internal_alignments, leaf_maps) for the Alignment trait,
        // we instead get the aligned Sequences and then create the alignment from it. This discards the internal
        // alignments and rebuilds them when calling `from_aligned`, which might not be ideal.
        // See issue #79 https://github.com/acg-team/rust-phylo/issues/79
        let leaf_maps = PhyloInfo::<A>::compile_leaf_map(&tree.root, &alignments, seqs, tree);
        let aligned_seqs = Sequences::with_alphabet_unchecked(
            leaf_maps
                .iter()
                .map(|(idx, map)| {
                    let rec = seqs.record_by_id(tree.node_id(idx));
                    let aligned_seq = aligned_seq!(map, rec.seq());
                    Record::with_attrs(rec.id(), rec.desc(), &aligned_seq)
                })
                .collect(),
            seqs.alphabet(),
        );
        let alignment = A::from_aligned_unchecked(aligned_seqs, tree);

        (alignment, scores)
    }

    fn pairwise_align(
        &self,
        x_info: &[ParsimonySite],
        x_blen: f64,
        y_info: &[ParsimonySite],
        y_blen: f64,
        rng: fn(usize) -> usize,
    ) -> (Vec<ParsimonySite>, PairwiseAlignment, f64) {
        let mut pars_mats =
            ParsimonyAlignmentMatrices::new(x_info, x_blen, y_info, y_blen, &self.scoring, rng);

        pars_mats.fill_matrices();
        pars_mats.traceback()
    }
}

#[cfg(test)]
#[cfg_attr(coverage, coverage(off))]
mod tests;
