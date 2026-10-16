use crate::alignment::{surviving_columns_mask, AncestralAlignment};
use crate::phylo_info::PhyloInfo;
use crate::tkf_model::simulate_msa::Fragmentation;

/// Result of a TKF simulation (indels only, or indels + substitutions) including fragmentation.
///
/// # Example
/// ```
/// use phyne::alignment::{Alignment, MASA};
/// use phyne::random::DefaultGenerator;
/// use phyne::tkf_model::simulate_msa::TKFIndelMSASimulator;
/// use phyne::tkf_model::TKF92IndelModel;
/// use phyne::tree;
/// # use phyne::Result;
///
/// # fn main() -> Result<()> {
/// let tree = tree!("((A:1.0,B:1.0)I:1.0,C:1.0)R;");
/// let max_ins_len = Some(50);
/// let simulator = TKFIndelMSASimulator::new(TKF92IndelModel::default(), tree, max_ins_len)?;
/// let mut rng = DefaultGenerator::new(123);
/// let result = simulator.simulate_with_fragments::<MASA, _>(&mut rng);
///
/// assert_eq!(result.phylo.msa.seq_count(), 3);
/// assert!(!result.fragmentation.is_empty());
/// # Ok(()) }
/// ```
pub struct TKFSimulationResult<AA: AncestralAlignment> {
    /// The simulated PhyloInfo containing the ancestral alignment and the tree.
    pub phylo: PhyloInfo<AA>,
    /// The fragmentation (right-exclusive boundaries of fragments) in sync with the alignment in
    /// `phylo`.
    pub fragmentation: Fragmentation,
}

impl<AA: AncestralAlignment> TKFSimulationResult<AA> {
    /// Removes columns where the ancestral character goes extinct (i.e., all leaf sequences
    /// have a gap) from the alignment and keeps the fragmentation in sync.
    ///
    /// # Example
    /// ```
    /// use phyne::alignment::{Alignment, MASA};
    /// use phyne::random::{RandomGenerator, FakeGenerator, FakeRng};
    /// use phyne::tkf_model::simulate_msa::{RootLength, TKFIndelMSASimulator};
    /// use phyne::tkf_model::TKF91IndelModel;
    /// use phyne::tree;
    /// # use phyne::Result;
    ///
    /// # fn main() -> Result<()> {
    /// let tree = tree!("(A:1.0,B:1.0)R;");
    /// let max_ins_len = None;
    /// let mut simulator = TKFIndelMSASimulator::new(TKF91IndelModel::default(), tree, max_ins_len)?;
    /// simulator.root_length(RootLength::Defined(2));
    /// // Deterministic values for the fake RNG: they make one root character survive to the
    /// // leaves while the other is deleted on all branches, i.e. it goes extinct.
    /// let values = vec![0.0, 0.0, 0.6, 0.6, 0.0];
    /// let mut rng = FakeGenerator { seed: 0, rng: FakeRng::from_f64_values(values) };
    /// let mut result = simulator.simulate_with_fragments::<MASA, _>(&mut rng);
    /// assert_eq!(result.phylo.msa.len(), 2);
    ///
    /// result.remove_extinct_columns();
    ///
    /// // the extinct column was removed
    /// assert_eq!(result.phylo.msa.len(), 1);
    /// // the fragmentation is still in sync with the alignment
    /// assert_eq!(result.fragmentation.right_exclusive_boundaries(), [1]);
    /// # Ok(()) }
    /// ```
    pub fn remove_extinct_columns(&mut self) {
        let keep_col_mask = surviving_columns_mask(&self.phylo.msa);
        self.phylo.msa.remove_extinct_columns();
        self.fragmentation.remove_cols(&keep_col_mask);
        self.fragmentation
            .fragmentation_works_with_ancestral_alignment(&self.phylo.msa);
    }
}

#[cfg(test)]
#[cfg_attr(coverage, coverage(off))]
mod private_tests {
    use crate::alignment::{Alignment, Sequences, MASA};
    use crate::tree::Tree;
    use crate::{record_wo_desc, tree, Record};

    use super::*;

    /// Builds a [`TKFSimulationResult`] from raw records. The fragmentation is chosen to have
    /// one fragment per column (`[1, 2, ..., n]`), which is always consistent with the alignment
    /// and lets us check that [`TKFSimulationResult::remove_extinct_columns`] keeps it in sync.
    fn result_from(
        records: Vec<Record>,
        tree: Tree,
        fragmentation: Fragmentation,
    ) -> TKFSimulationResult<MASA> {
        let phylo =
            MASA::from_aligned_with_ancestral(Sequences::new(records).unwrap(), tree).unwrap();
        TKFSimulationResult {
            phylo,
            fragmentation,
        }
    }

    #[test]
    fn remove_extinct_columns_no_extinction() {
        let tree = tree!("((A:1.0,B:1.0)I:1.0,C:1.0)R:1.0;");
        let fragmentation = Fragmentation::new(vec![1, 3, 4]);
        let mut result = result_from(
            vec![
                record_wo_desc!("A", b"ACGT"),
                record_wo_desc!("B", b"ACGT"),
                record_wo_desc!("C", b"ACGT"),
                record_wo_desc!("I", b"ACGT"),
                record_wo_desc!("R", b"ACGT"),
            ],
            tree,
            fragmentation,
        );

        // no column goes extinct in the leaves, so nothing is removed
        assert_eq!(result.phylo.msa.len(), 4);
        result.remove_extinct_columns();
        assert_eq!(result.phylo.msa.len(), 4);
        assert_eq!(
            result.fragmentation.right_exclusive_boundaries(),
            [1, 3, 4].as_slice()
        );
    }

    #[test]
    fn remove_extinct_columns_keeps_fragmentation_in_sync() {
        let tree = tree!("((A:1.0,B:1.0)I:1.0,C:1.0)R:1.0;");
        let fragmentation = Fragmentation::new(vec![1, 2, 5, 6, 7, 9]);
        let mut result = result_from(
            vec![
                record_wo_desc!("A", b"A-TTT----"),
                record_wo_desc!("B", b"C-CTT----"),
                record_wo_desc!("C", b"G-CTT--GG"),
                record_wo_desc!("I", b"T-CTTAAGG"),
                record_wo_desc!("R", b"AAATTTGGG"),
            ],
            tree,
            fragmentation,
        );

        result.remove_extinct_columns();

        assert_eq!(result.phylo.msa.len(), 6);
        // the fragmented columns are dropped together with the extinct columns
        assert_eq!(
            result.fragmentation.right_exclusive_boundaries(),
            [1, 4, 6].as_slice()
        );
    }

    #[test]
    fn remove_extinct_columns_all_extinct() {
        let tree = tree!("((A:1.0,B:1.0)I:1.0,C:1.0)R:1.0;");
        let fragmentation = Fragmentation::new(vec![3, 4]);
        let mut result = result_from(
            vec![
                record_wo_desc!("A", b"----"),
                record_wo_desc!("B", b"----"),
                record_wo_desc!("C", b"----"),
                record_wo_desc!("I", b"ACGT"),
                record_wo_desc!("R", b"ACGT"),
            ],
            tree,
            fragmentation,
        );

        result.remove_extinct_columns();

        assert_eq!(result.phylo.msa.len(), 0);
        assert!(result.fragmentation.is_empty());
    }
}
