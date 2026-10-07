use crate::alignment::{Alignment, AncestralAlignment};
use crate::phylo_info::validate_taxa_ids;
use crate::tree::Tree;
use crate::{bail, Result};

/// Trait for ancestral sequence reconstruction.
pub trait AncestralSequenceReconstruction<A: Alignment, AA: AncestralAlignment> {
    /// Checks if inputs are compatible and calls [`Self::reconstruct_ancestral_seqs_unchecked`].
    /// Checks:
    ///  - if number of sequences in the alignment matches the number of leaves in the tree
    ///  - if node IDs in the tree are unique ([`Tree::node_ids_are_unique`])
    ///  - if the sequence IDs in the alignment match the taxa IDs in the tree ([`validate_taxa_ids`])
    ///
    /// Only overwrite this method if absolutely necessary. The default implementation
    /// ensures that prerequisites are met. Overwriting and not ensuring these checks
    /// may lead to unexpected panics or wrong results.
    fn reconstruct_ancestral_seqs(&self, leaf_alignment: &A, tree: &Tree) -> Result<AA> {
        if leaf_alignment.seq_count() != tree.n {
            bail!(
                AncestralAlignment,
                "alignment has {} sequences, but tree has {} leaves",
                leaf_alignment.seq_count(),
                tree.n
            );
        }
        tree.node_ids_are_unique()?;
        validate_taxa_ids(tree, leaf_alignment.seqs())?;
        Ok(self.reconstruct_ancestral_seqs_unchecked(leaf_alignment, tree))
    }

    /// Reconstructs ancestral sequences without any checks. Is called by
    /// [`Self::reconstruct_ancestral_seqs`].
    fn reconstruct_ancestral_seqs_unchecked(&self, leaf_alignment: &A, tree: &Tree) -> AA;
}

#[cfg(test)]
#[cfg_attr(coverage, coverage(off))]
mod tests;
