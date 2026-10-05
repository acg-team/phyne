use std::fmt::Display;

use crate::alignment::AncestralAlignment;
use crate::likelihood::{ModelSearchCost, ParamRange, TreeSearchCost};
use crate::substitution_models::{FreqVector, QMatrix, SubstitutionCost};
use crate::tree::Tree;
use crate::Result;

pub mod tkf91;
pub use tkf91::*;
pub mod tkf92;
pub use tkf92::*;
pub mod reestimate;
pub use reestimate::*;
pub mod tkf_indel;
pub use tkf_indel::*;

#[derive(Clone, Debug)]
pub struct TKFCost<Q: QMatrix + Display, T: TKFModel, AA: AncestralAlignment> {
    // TODO: if we have just the sum of the two costs like this, we need to keep track of the
    // phylo (which is tree and alignment) twice, which might be too big of a downside, since the
    // cost is copied often. Alternatively we could implement the substitution cost inside the
    // tkf92 cost, which would duplicate some code.
    // See issue #152 https://github.com/acg-team/rust-phylo/issues/152
    indel_cost: TKFIndelCost<T, AA>,
    subst_cost: SubstitutionCost<Q, AA>,
}

impl<Q: QMatrix, T: TKFModel, AA: AncestralAlignment> Display for TKFCost<Q, T, AA> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} and {}",
            self.indel_cost.model, self.subst_cost.model.qmatrix
        )
    }
}

impl<Q: QMatrix, T: TKFModel, AA: AncestralAlignment> ModelSearchCost for TKFCost<Q, T, AA> {
    fn cost(&self) -> f64 {
        ModelSearchCost::cost(&self.subst_cost) + ModelSearchCost::cost(&self.indel_cost)
    }

    fn param_count(&self) -> usize {
        self.indel_cost.model.params().len() + self.subst_cost.model.qmatrix.params().len()
    }

    fn param(&self, idx: usize) -> f64 {
        let num_params_indel_model = self.indel_cost.param_count();
        if idx < num_params_indel_model {
            return self.indel_cost.param(idx);
        }
        let idx = idx - num_params_indel_model;
        self.subst_cost.param(idx)
    }

    fn set_param(&mut self, idx: usize, value: f64) {
        let num_params_indel_model = self.indel_cost.param_count();
        if idx < num_params_indel_model {
            self.indel_cost.set_param(idx, value);
            return;
        }
        let idx = idx - num_params_indel_model;
        self.subst_cost.set_param(idx, value);
    }

    fn param_range(&self, idx: usize) -> ParamRange {
        let num_params_indel_model = self.indel_cost.param_count();
        if idx < num_params_indel_model {
            return self.indel_cost.param_range(idx);
        }
        let idx = idx - num_params_indel_model;
        self.subst_cost.param_range(idx)
    }

    fn set_freqs(&mut self, freqs: FreqVector) -> Result<()> {
        self.subst_cost.set_freqs(freqs)
    }

    fn empirical_freqs(&self) -> FreqVector {
        self.subst_cost.info.freqs()
    }

    fn freqs(&self) -> &FreqVector {
        self.subst_cost.freqs()
    }
}

impl<Q: QMatrix, T: TKFModel, AA: AncestralAlignment> TreeSearchCost for TKFCost<Q, T, AA> {
    fn cost(&self) -> f64 {
        TreeSearchCost::cost(&self.subst_cost) + TreeSearchCost::cost(&self.indel_cost)
    }

    fn update_tree(&mut self, tree: Tree) {
        self.indel_cost.update_tree(tree.clone());
        self.subst_cost.update_tree(tree.clone());
    }

    fn tree(&self) -> &Tree {
        self.indel_cost.tree()
    }
}

impl<Q: QMatrix, T: TKFModel, AA: AncestralAlignment> TKFCost<Q, T, AA> {
    /// Returns a reference to the multiple ancestral sequence alignment.
    ///
    /// # Example
    /// ```rust
    /// use phylo::phylo_info::PhyloInfo;
    /// use phylo::substitution_models::{SubstModel, dna_models::GTR};
    /// use phylo::alignment::{Alignment, AncestralAlignment, Sequences, MASA};
    /// use phylo::tkf_model::TKF92CostBuilder;
    /// use phylo::{tree, record_wo_desc as record};
    ///
    /// # use phylo::Result;
    /// # fn main() -> Result<()> {
    /// let tree = tree!("(((A1:2.0,B2:2.0)I3:0.3,C4:2.0)R5:1.0);");
    /// let msa = MASA::from_aligned_with_ancestral(
    ///     Sequences::new(
    ///         vec![
    ///             record!("A1", b"--GTGGA---"),
    ///             record!("B2", b"-------NNA"),
    ///             record!("I3", b"--T-------"),
    ///             record!("C4", b"AGG-------"),
    ///             record!("R5", b"--A-------"),
    ///            ],
    ///     )?,
    ///   &tree,
    /// )?;
    /// let phylo = PhyloInfo { msa, tree };
    /// let subst_model = SubstModel::<GTR>::default();
    /// let cost = TKF92CostBuilder::new(&[0.4, 0.5, 0.8], subst_model, phylo).build()?;
    /// assert_eq!(cost.masa().seqs().len(), 3);
    /// assert_eq!(cost.masa().ancestral_seqs().len(), 2);
    /// # Ok(()) }
    pub fn masa(&self) -> &impl AncestralAlignment {
        &self.indel_cost.phylo.msa
    }

    /// Returns the TKF blocks of the alignment, see [`TKFModel::get_blocks`].
    ///
    /// # Example
    /// ```rust
    /// use phylo::phylo_info::PhyloInfo;
    /// use phylo::substitution_models::{SubstModel, dna_models::GTR};
    /// use phylo::alignment::{Alignment, AncestralAlignment, Sequences, MASA};
    /// use phylo::tkf_model::TKF92CostBuilder;
    /// use phylo::{tree, record_wo_desc as record};
    ///
    /// # use phylo::Result;
    /// # fn main() -> Result<()> {
    /// let tree = tree!("(((A1:2.0,B2:2.0)I3:0.3,C4:2.0)R5:1.0);");
    /// let msa = MASA::from_aligned_with_ancestral(
    ///     Sequences::new(
    ///         vec![
    ///             record!("A1", b"--GTGGA---"),
    ///             record!("B2", b"-------NNA"),
    ///             record!("I3", b"--T-------"),
    ///             record!("C4", b"AGG-------"),
    ///             record!("R5", b"--A-------"),
    ///            ],
    ///     )?,
    ///   &tree,
    /// )?;
    /// let phylo = PhyloInfo { msa, tree };
    /// let subst_model = SubstModel::<GTR>::default();
    /// let cost = TKF92CostBuilder::new(&[0.4, 0.5, 0.8], subst_model, phylo).build()?;
    /// // under the TKF92 model the blocks are the positions in the alignment where there is
    /// // a sequence that changes from gap to non-gap or vice versa (always including the last
    /// // position of the alignment).
    /// assert_eq!(cost.blocks(), vec![2, 3, 7, 10]);
    /// # Ok(()) }
    pub fn blocks(&self) -> Vec<usize> {
        self.indel_cost.model.get_blocks(&self.indel_cost.phylo.msa)
    }
}

#[cfg(test)]
#[cfg_attr(coverage, coverage(off))]
mod tests;

#[cfg(test)]
#[cfg_attr(coverage, coverage(off))]
mod tkf92_additional_blocks;

#[cfg(test)]
pub use tkf92_additional_blocks::*;

#[cfg(test)]
#[cfg_attr(coverage, coverage(off))]
mod tkf92_fixed_fragmentation;

#[cfg(test)]
pub use tkf92_fixed_fragmentation::*;

#[cfg(test)]
mod tkf_numerical_tests;
