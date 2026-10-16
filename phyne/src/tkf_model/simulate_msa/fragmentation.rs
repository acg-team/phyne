use fixedbitset::FixedBitSet;

use crate::alignment::AncestralAlignment;
use crate::bug;
use crate::tkf_model::blocks_of_alignment;

/// The fragmentation of an alignment.
///
/// Under the TKF model, an alignment is naturally partitioned into fragments.
/// This struct stores the right-exclusive boundaries of those fragments.
/// There is no public constructor: a `Fragmentation` is obtained from a
/// simulation result (see [`TKFSimulationResult`](`crate::tkf_model::simulate_msa::TKFSimulationResult`))
/// and is always kept in sync with the alignment.
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
/// // The maximum number of inserted links (=fragments) in a single insertion event.
/// let max_ins_len = Some(50);
/// let simulator = TKFIndelMSASimulator::new(TKF92IndelModel::default(), tree, max_ins_len)?;
/// let mut rng = DefaultGenerator::new(123);
/// let result = simulator.simulate_with_fragments::<MASA, _>(&mut rng);
///
/// // the fragments partition the columns of the alignment
/// let boundaries = result.fragmentation.right_exclusive_boundaries();
/// assert_eq!(boundaries.last().copied(), Some(result.phylo.msa.len()));
/// # Ok(()) }
/// ```
#[derive(Clone, Debug)]
pub struct Fragmentation {
    right_exclusive_boundaries: Vec<usize>,
}

impl Fragmentation {
    pub(super) fn new(fragments: Vec<usize>) -> Self {
        Self::is_increasing(&fragments);
        Fragmentation {
            right_exclusive_boundaries: fragments,
        }
    }

    /// Panics if the fragment boundaries are not strictly increasing,
    /// since this indicates a critical failure/bug.
    fn is_increasing(fragments: &[usize]) {
        if !fragments.windows(2).all(|w| w[0] < w[1]) {
            bug!(format!(
                "Fragmentation is not strictly increasing, got {fragments:?}"
            ));
        }
    }

    /// Returns the right-exclusive boundaries of the fragments, i.e., for `n` fragments the
    /// indices `f_0 < ... < f_{n-1}` such that fragment `i` spans columns `[f_{i-1}, f_i)`.
    /// Note that the first fragment starts at column 0, so `f_{-1} = 0` is implicit and
    /// therefore `f_0 > 0` and `f_{n-1} = alignment.len()`.
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
    /// // The maximum number of inserted links (=fragments) in a single insertion event.
    /// let max_ins_len = Some(50);
    /// let simulator = TKFIndelMSASimulator::new(TKF92IndelModel::default(), tree, max_ins_len)?;
    /// let mut rng = DefaultGenerator::new(123);
    /// let result = simulator.simulate_with_fragments::<MASA, _>(&mut rng);
    ///
    /// let boundaries = result.fragmentation.right_exclusive_boundaries();
    /// // the first fragment starts at column 0, so the first boundary is at least 1
    /// assert!(boundaries[0] >= 1);
    /// // boundaries are strictly increasing and the last one is the alignment length
    /// assert!(boundaries.windows(2).all(|w| w[0] < w[1]));
    /// assert_eq!(boundaries.last().copied(), Some(result.phylo.msa.len()));
    /// # Ok(()) }
    /// ```
    pub fn right_exclusive_boundaries(&self) -> &[usize] {
        &self.right_exclusive_boundaries
    }

    /// Returns the number of fragments.
    ///
    /// # Example
    /// ```
    /// use phyne::alignment::{Alignment, MASA};
    /// use phyne::random::DefaultGenerator;
    /// use phyne::tkf_model::simulate_msa::TKFIndelMSASimulator;
    /// use phyne::tkf_model::TKF91IndelModel;
    /// use phyne::tree;
    /// # use phyne::Result;
    ///
    /// # fn main() -> Result<()> {
    /// let tree = tree!("((A:1.0,B:1.0)I:1.0,C:1.0)R;");
    /// let max_ins_len = None;
    /// let simulator = TKFIndelMSASimulator::new(TKF91IndelModel::default(), tree, max_ins_len)?;
    /// let mut rng = DefaultGenerator::new(123);
    /// let result = simulator.simulate_with_fragments::<MASA, _>(&mut rng);
    ///
    /// // in TKF91 every residue is its own fragment, so there is one fragment per column
    /// assert_eq!(result.fragmentation.len(), result.phylo.msa.len());
    /// # Ok(()) }
    /// ```
    pub fn len(&self) -> usize {
        self.right_exclusive_boundaries.len()
    }

    /// Returns whether the fragmentation is empty, i.e., whether the alignment has no columns.
    ///
    /// # Example
    /// ```
    /// use phyne::alignment::MASA;
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
    /// assert!(!result.fragmentation.is_empty());
    /// # Ok(()) }
    /// ```
    pub fn is_empty(&self) -> bool {
        self.right_exclusive_boundaries.is_empty()
    }

    /// Panics if the fragmentation is not consistent with the given ancestral alignment,
    /// since this indicates a critical failure/bug.
    ///
    /// # Panics
    ///
    /// * If the fragmentation is empty, the alignment must also be empty.
    /// * The last fragment boundary must be equal to the alignment length.
    /// * All fragment boundaries must be present in the alignment's block boundaries.
    pub(super) fn fragmentation_works_with_ancestral_alignment<AA: AncestralAlignment>(
        &self,
        masa: &AA,
    ) {
        let incompatible_msg = "Fragmentation and alignment are incompatible:";
        if self.is_empty() {
            if masa.len() == 0 {
                return;
            } else {
                bug!(format!(
                    "{incompatible_msg} fragmentation is empty but alignment has length {}",
                    masa.len()
                ));
            }
        }
        let last_frag_boundary = self.right_exclusive_boundaries.last().unwrap();
        if last_frag_boundary != &masa.len() {
            bug!(format!(
                "{incompatible_msg} fragmentation ({last_frag_boundary}) does not match alignment length ({})",
                masa.len()
            ));
        }

        // get the blocks of the alignment, these indices must also be in the fragmentation
        let blocks = blocks_of_alignment(masa);
        for block in blocks {
            if !self.right_exclusive_boundaries.contains(&block) {
                bug!(format!(
                    "{incompatible_msg} fragmentation does not contain block border {block}"
                ));
            }
        }
    }

    /// When [AncestralAlignment::remove_extinct_columns](`crate::alignment::AncestralAlignment::remove_extinct_columns`)
    /// is called on a simulated alignment generated by either
    /// [TKFMSASimulator::simulate](`crate::tkf_model::simulate_msa::TKFMSASimulator::simulate_with_fragments`) or
    /// [TKFIndelMSASimulator::simulate](`crate::tkf_model::simulate_msa::TKFIndelMSASimulator::simulate_with_fragments`),
    /// the corresponding fragmentation goes out of sync with the alignment. This function removes the
    /// columns that were removed from the alignment from the fragmentation, so that it is in sync again.
    /// Is used in [`TKFSimulationResult::remove_extinct_columns`](`crate::tkf_model::simulate_msa::TKFSimulationResult::remove_extinct_columns`).
    /// The mask is a [`FixedBitSet`] over the original alignment columns, where a set bit indicates a
    /// column that was kept and an unset bit a column that was removed.
    pub(super) fn remove_cols(&mut self, keep_col_mask: &FixedBitSet) {
        let incompatible_msg = "Fragmentation and keep_col_mask are incompatible:";
        if self.is_empty() {
            if keep_col_mask.is_empty() {
                return;
            } else {
                bug!(format!(
                    "{incompatible_msg} fragmentation is empty but keep_col_mask has length {}",
                    keep_col_mask.len()
                ));
            };
        }
        let last_frag_boundary = self.right_exclusive_boundaries.last().unwrap();
        if last_frag_boundary != &keep_col_mask.len() {
            bug!(format!(
                "{incompatible_msg} fragmentation ({last_frag_boundary}) does not match alignment length ({})",
                keep_col_mask.len()
            ));
        }
        let mut new_fragmentation = Vec::with_capacity(self.len());
        let mut prev_border = 0;
        let mut removed_so_far = 0;
        for frag_border in self.right_exclusive_boundaries.iter() {
            let fragment_range = prev_border..*frag_border;
            if !keep_col_mask.contains_any_in_range(fragment_range.clone()) {
                removed_so_far += fragment_range.len();
                // not keep ie remove
            } else if keep_col_mask.contains_all_in_range(fragment_range) {
                // keep
                new_fragmentation.push(*frag_border - removed_so_far);
            } else {
                bug!(format!(
                    "{incompatible_msg} mask does not match fragmentation"
                ));
            }
            prev_border = *frag_border;
        }
        Self::is_increasing(&new_fragmentation);
        self.right_exclusive_boundaries = new_fragmentation;
    }
}

#[cfg(test)]
#[cfg_attr(coverage, coverage(off))]
mod private_tests {

    use super::*;
    use fixedbitset::FixedBitSet;
    use rstest::*;

    #[rstest]
    #[case(vec![])]
    #[case(vec![0])]
    #[case(vec![1, 3, 5])]
    fn fragmentation_init_valid(#[case] frag: Vec<usize>) {
        let fragmentation = Fragmentation::new(frag.clone());
        assert_eq!(fragmentation.right_exclusive_boundaries(), &frag);
    }

    #[rstest]
    #[case(vec![3, 2, 5])]
    #[case(vec![3, 3, 5])]
    #[should_panic(expected = "Fragmentation is not strictly increasing")]
    fn fragmentation_init_invalid(#[case] frag: Vec<usize>) {
        Fragmentation::new(frag);
    }

    #[test]
    fn remove_cols_from_fragmentation_no_removals() {
        let mut fragmentation = Fragmentation::new(vec![3, 6, 9]);
        let keep_cols_mask = (0..9).collect::<FixedBitSet>();
        fragmentation.remove_cols(&keep_cols_mask);
        assert_eq!(fragmentation.right_exclusive_boundaries(), vec![3, 6, 9]);
    }

    #[test]
    fn remove_cols_from_fragmentation_all_removals() {
        let mut fragmentation = Fragmentation::new(vec![3, 6, 9]);
        let keep_cols_mask = FixedBitSet::with_capacity(9);
        fragmentation.remove_cols(&keep_cols_mask);
        assert_eq!(
            fragmentation.right_exclusive_boundaries(),
            Vec::<usize>::new()
        );
    }

    #[test]
    fn remove_cols_from_fragmentation_first_removed() {
        let mut fragmentation = Fragmentation::new(vec![3, 7, 12]);
        let keep_cols_mask = (3..12).collect::<FixedBitSet>();
        fragmentation.remove_cols(&keep_cols_mask);
        assert_eq!(fragmentation.right_exclusive_boundaries(), vec![4, 9]);
    }

    #[test]
    fn remove_cols_from_fragmentation_middle_removed() {
        let mut fragmentation = Fragmentation::new(vec![3, 7, 12]);
        let keep_cols_mask = (0..3).chain(7..12).collect::<FixedBitSet>();
        fragmentation.remove_cols(&keep_cols_mask);
        assert_eq!(fragmentation.right_exclusive_boundaries(), vec![3, 8]);
    }

    #[test]
    fn remove_cols_from_fragmentation_last_removed() {
        let mut fragmentation = Fragmentation::new(vec![3, 7, 12]);
        let mut keep_cols_mask = FixedBitSet::with_capacity(12);
        keep_cols_mask.extend(0..7);
        fragmentation.remove_cols(&keep_cols_mask);
        assert_eq!(fragmentation.right_exclusive_boundaries(), vec![3, 7]);
    }

    #[test]
    fn remove_cols_from_fragmentation_larger_example() {
        let mut fragmentation = Fragmentation::new(vec![1, 3, 6, 10, 15, 21]);
        let keep_cols_mask = (3..6).chain(15..21).collect::<FixedBitSet>();
        fragmentation.remove_cols(&keep_cols_mask);
        assert_eq!(fragmentation.right_exclusive_boundaries(), vec![3, 9]);
    }

    #[test]
    #[should_panic(expected = "fragmentation is empty but keep_col_mask has length")]
    fn remove_cols_from_fragmentation_empty_fragmentation() {
        let mut fragmentation = Fragmentation::new(vec![]);
        let keep_cols_mask = (0..3).collect::<FixedBitSet>();
        fragmentation.remove_cols(&keep_cols_mask);
    }

    #[test]
    #[should_panic(expected = "does not match alignment length")]
    fn remove_cols_from_fragmentation_mismatched_lens() {
        let mut fragmentation = Fragmentation::new(vec![3, 6, 9]);
        let keep_cols_mask = (0..8).collect::<FixedBitSet>(); // length 8 instead of 9
        fragmentation.remove_cols(&keep_cols_mask);
    }

    #[rstest]
    #[case((0..1).chain(2..9).collect::<FixedBitSet>())]
    #[case((2..7).chain(8..9).collect::<FixedBitSet>())]
    #[should_panic(expected = "mask does not match fragmentation")]
    fn remove_cols_from_fragmentation_mask_does_not_match(#[case] keep_cols_mask: FixedBitSet) {
        let mut fragmentation = Fragmentation::new(vec![3, 6, 9]);
        fragmentation.remove_cols(&keep_cols_mask);
    }
}
