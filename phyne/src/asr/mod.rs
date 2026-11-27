//! Ancestral sequence reconstruction module.
//!
//! Provides the `AncestralSequenceReconstruction` trait for reconstructing ancestral sequences
//! given a multiple sequence alignment and a phylogenetic tree.
//! If the provided alignment already contains ancestral sequences, they will be ignored and reconstructed anew.
//!
//! # Examples
//!
//! ```rust
//! use phyne::alignment::{Alignment, AncestralAlignment, MASA};
//! use phyne::asr::AncestralSequenceReconstruction;
//! use phyne::parsimony_presence_absence::ParsimonyPresenceAbsence;
//! use phyne::phylo_info::{PhyloInfo, PhyloInfoBuilder};
//!
//! # use phyne::Result;
//! # fn main() -> Result<()> {
//! let input = PhyloInfoBuilder::with_attrs(
//!     "./examples/data/sequences_DNA_small.fasta",
//!     "./examples/data/tree_with_ancestral_ids.newick",
//! )
//! .build()?;
//!
//! let result: PhyloInfo<MASA> = AncestralSequenceReconstruction::reconstruct_ancestral_seqs(
//!     &ParsimonyPresenceAbsence {},
//!     input.clone()
//! )?;
//!
//! assert_eq!(result.msa.seq_count(), input.msa.seq_count());
//! assert_eq!(
//!     result.msa.seq_count() + result.msa.ancestral_seqs().len(),
//!     input.tree.len()
//! );
//! assert_eq!(result.msa.len(), input.msa.len());
//! # Ok(()) }
//! ```
use crate::alignment::{Alignment, AncestralAlignment};
use crate::phylo_info::{set_missing_tree_node_ids, PhyloInfo};
use crate::{bail, Result};

/// Trait for ancestral sequence reconstruction.
pub trait AncestralSequenceReconstruction<A: Alignment, AA: AncestralAlignment> {
    /// Checks if all node IDs (leaves and internal) in the tree are unique and calls
    /// [`Self::reconstruct_ancestral_seqs_unchecked`].
    fn reconstruct_ancestral_seqs(&self, mut phylo: PhyloInfo<A>) -> Result<PhyloInfo<AA>> {
        if phylo.tree.len() == 1 {
            bail!(
                AncestralAlignment,
                "cannot reconstruct ancestral sequences for a tree with only one node."
            )
        }

        set_missing_tree_node_ids(&mut phylo.tree)?;
        Ok(self.reconstruct_ancestral_seqs_unchecked(phylo))
    }

    /// Reconstructs ancestral sequences without any checks. Is called by
    /// [`Self::reconstruct_ancestral_seqs`].
    fn reconstruct_ancestral_seqs_unchecked(&self, phylo: PhyloInfo<A>) -> PhyloInfo<AA>;
}

#[cfg(test)]
#[cfg_attr(coverage, coverage(off))]
mod tests;
