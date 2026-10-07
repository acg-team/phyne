use crate::alignment::{Alignment, Sequences};
use crate::phylo_info::validate_taxa_ids;
use crate::tree::Tree;
use crate::{bail, Result};

/// Trait for aligning sequences, optionally using a phylogenetic tree as guidance.
pub trait Aligner<A: Alignment> {
    /// Checks if inputs are compatible and calls [`Self::align_unchecked`].
    /// Checks:
    ///  - if sequences are not already aligned
    ///  - if the sequence IDs in the alignment match the taxa IDs in the tree ([`validate_taxa_ids`])
    fn align(&self, seqs: &Sequences, tree: &Tree) -> Result<A> {
        if seqs.aligned {
            bail!(Alignment, "sequences must not be already aligned");
        }
        validate_taxa_ids(tree, seqs)?;
        Ok(self.align_unchecked(seqs, tree))
    }
    /// Aligns sequences. Is called by [`Self::align`].
    fn align_unchecked(&self, seqs: &Sequences, tree: &Tree) -> A;
}
