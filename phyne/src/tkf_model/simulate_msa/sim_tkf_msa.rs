use bio::io::fasta::Record;
use rand::{Rng, SeedableRng};

use crate::alignment::{AlignmentSimulation, AncestralAlignment, Sequences};
use crate::alphabets::GAP;
use crate::phylo_info::{set_missing_tree_node_ids, PhyloInfo};
use crate::random::RandomGenerator;
use crate::substitution_models::{QMatrix, SubstModel, SubstitutionSimulator};
use crate::tkf_model::simulate_msa::sim_tkf_indel_msa::{FragmentSampler, TKFIndelMSASimulator};
use crate::tkf_model::simulate_msa::{ExpectedRootLength, RootLength, TKFSimulationResult};
use crate::tkf_model::TKFModel;
use crate::tree::{NodeIdx::Internal, NodeIdx::Leaf, Tree};
use crate::{bug, record_wo_desc as record, Result};

/// Simulates a full TKF process: first indels then substitutions.
///
/// First [indel events](TKFIndelMSASimulator) are simulated to obtain an ancestral alignment
/// that represents homology paths, then simulates [substitutions](`SubstitutionSimulator`) on them.
/// Note, that the alignment might contain columns where the character goes extinct, i.e., all leaf
/// sequences have a gap in that column. You may want to call
/// [`TKFSimulationResult::remove_extinct_columns`](`TKFSimulationResult::remove_extinct_columns`).
///
/// # Example
/// ```
/// use phyne::alignment::{Alignment, AlignmentSimulation, AncestralAlignment, MASA};
/// use phyne::random::DefaultGenerator;
/// use phyne::substitution_models::{dna_models::JC69, SubstModel};
/// use phyne::tkf_model::simulate_msa::{RootLength, TKFMSASimulator};
/// use phyne::tkf_model::TKF92IndelModel;
/// use phyne::tree;
/// # use phyne::Result;
///
/// # fn main() -> Result<()> {
/// let tree = tree!("((A:1.0,B:1.0)I:1.0,C:1.0)R;");
/// // The maximum number of inserted links (=fragments) in a single insertion event.
/// let max_ins_len = Some(50);
/// let mut simulator = TKFMSASimulator::new(
///     TKF92IndelModel::default(),
///     SubstModel::<JC69>::default(),
///     tree,
///     max_ins_len
/// )?;
/// // Fix the root sequence to exactly 100 characters.
/// let root_len = 100;
/// simulator.root_length(RootLength::Defined(root_len));
/// let mut rng = DefaultGenerator::new(123);
/// let phylo = simulator.simulate_ancestral_alignment::<MASA>(&mut rng);
/// assert_eq!(phylo.msa.seq_count(), 3);
/// let root_map = phylo.msa.ancestral_map(&phylo.tree.root);
/// assert_eq!(root_map.iter().filter(|site| site.is_some()).count(), root_len);
/// # Ok(()) }
/// ```
pub struct TKFMSASimulator<T>
where
    T: TKFModel + FragmentSampler + ExpectedRootLength,
{
    indel_sim: TKFIndelMSASimulator<T>,
    subst_sim: SubstitutionSimulator,
}

impl<T> TKFMSASimulator<T>
where
    T: TKFModel + FragmentSampler + ExpectedRootLength,
{
    /// Create a new [`TKFMSASimulator`] with the given indel model, substitution model, tree and
    /// max insertion length (i.e., the max number of inserted links (=fragments) in a single event;
    /// since fragments can consist of multiple characters the number of inserted characters can be
    /// longer than this max length).
    ///
    /// # Errors
    /// * If the tree node IDs cannot be completed (see
    ///   [`set_missing_tree_node_ids`](`crate::phylo_info::set_missing_tree_node_ids`)).
    pub fn new<Q: QMatrix>(
        indel_model: T,
        subst_model: SubstModel<Q>,
        mut tree: Tree,
        max_insertion_length: Option<usize>,
    ) -> Result<Self> {
        set_missing_tree_node_ids(&mut tree)?;
        let indel_sim = TKFIndelMSASimulator::new(indel_model, tree.clone(), max_insertion_length)?;
        // This 'alignment_length' is never used as it's always overwritten in the call to
        // 'simulate_with_fragments'. There, we first simulate an indel only alignment. Then we use
        // its length to simulate the substitutions.
        let dummy_len = 0;
        let subst_sim = SubstitutionSimulator::new(subst_model, tree, dummy_len)?;
        Ok(Self {
            indel_sim,
            subst_sim,
        })
    }

    /// Sets the constraint for the [`RootLength`] used for the simulation.
    ///
    /// # Example
    /// ```
    /// use phyne::alignment::{AlignmentSimulation, AncestralAlignment, MASA};
    /// use phyne::random::DefaultGenerator;
    /// use phyne::substitution_models::{dna_models::JC69, SubstModel};
    /// use phyne::tkf_model::simulate_msa::{RootLength, TKFMSASimulator};
    /// use phyne::tkf_model::TKF92IndelModel;
    /// use phyne::tree;
    /// # use phyne::Result;
    ///
    /// # fn main() -> Result<()> {
    /// let tree = tree!("((A:1.0,B:1.0)I:1.0,C:1.0)R;");
    /// // The maximum number of inserted links (=fragments) in a single insertion event.
    /// let max_ins_len = Some(50);
    /// let mut simulator = TKFMSASimulator::new(
    ///     TKF92IndelModel::default(),
    ///     SubstModel::<JC69>::default(),
    ///     tree,
    ///     max_ins_len,
    /// )?;
    /// let root_len = 100;
    /// simulator.root_length(RootLength::Defined(root_len));
    ///
    /// let mut rng = DefaultGenerator::new(123);
    /// let phylo = simulator.simulate_ancestral_alignment::<MASA>(&mut rng);
    /// let root_map = phylo.msa.ancestral_map(&phylo.tree.root);
    /// assert_eq!(root_map.iter().filter(|site| site.is_some()).count(), root_len);
    /// # Ok(()) }
    /// ```
    pub fn root_length(&mut self, root_length: RootLength) {
        self.indel_sim.root_length(root_length);
    }

    /// Simulates the full TKF process (indels and substitutions) and returns
    /// the [fragmentation](`crate::tkf_model::Fragmentation`).
    ///
    /// # Example
    /// ```
    /// use phyne::alignment::{Alignment, MASA};
    /// use phyne::random::DefaultGenerator;
    /// use phyne::substitution_models::{dna_models::JC69, SubstModel};
    /// use phyne::tkf_model::simulate_msa::TKFMSASimulator;
    /// use phyne::tkf_model::TKF92IndelModel;
    /// use phyne::tree;
    /// # use phyne::Result;
    ///
    /// # fn main() -> Result<()> {
    /// let tree = tree!("((A:1.0,B:1.0)I:1.0,C:1.0)R;");
    /// // The maximum number of inserted links (=fragments) in a single insertion event.
    /// let max_ins_len = Some(50);
    /// let simulator = TKFMSASimulator::new(
    ///     TKF92IndelModel::default(),
    ///     SubstModel::<JC69>::default(),
    ///     tree,
    ///     max_ins_len,
    /// )?;
    /// let mut rng = DefaultGenerator::new(123);
    /// let result = simulator.simulate_with_fragments::<MASA, _>(&mut rng);
    ///
    /// assert_eq!(result.phylo.msa.seq_count(), 3);
    /// assert!(!result.fragmentation.is_empty());
    /// # Ok(()) }
    /// ```
    pub fn simulate_with_fragments<AA: AncestralAlignment, R: Rng + SeedableRng>(
        &self,
        rng: &mut RandomGenerator<R>,
    ) -> TKFSimulationResult<AA> {
        // First, indel simulation
        let TKFSimulationResult {
            phylo: indel_phylo,
            fragmentation,
        } = self.indel_sim.simulate_with_fragments::<AA, _>(rng);

        // Second, substitution semulation
        let aln_len = indel_phylo.msa.len();
        let subst_phylo: PhyloInfo<AA> = self
            .subst_sim
            .simulate_ancestral_alignment_with_length(aln_len, rng);

        // Third, mask the substitution msa with gaps from the indel msa
        let phylo = self.mask_substitution_msa(indel_phylo.msa, subst_phylo.msa);

        TKFSimulationResult {
            phylo,
            fragmentation,
        }
    }

    /// Masks the substitution-only MSA with the gaps of the indel-only MSA, i.e., keeps the
    /// substitution character where the indel MSA has a character and inserts a gap
    /// otherwise, and builds the final ancestral alignment from the combined records.
    ///
    /// # Panics
    /// * If the combined sequences cannot be turned into an ancestral alignment (should never
    ///   happen since they are built from the simulation), or if the resulting alignment is
    ///   shorter than the indel MSA (there should be no all-gap columns).
    fn mask_substitution_msa<AA: AncestralAlignment>(
        &self,
        indel_msa: AA,
        subst_msa: AA,
    ) -> PhyloInfo<AA> {
        // Construct a sequences vector including ancestral and leaf records
        let mut combined_records: Vec<Record> = Vec::new();
        for node in self.indel_sim.tree().preorder() {
            let id = self.indel_sim.tree().node(node).id.clone();
            // get mask_seq (from indel msa) and subst_seq (from substitution msa)
            let mask_mapping = match node {
                Internal(_) => indel_msa.ancestral_map(node),
                Leaf(_) => indel_msa.leaf_map(node),
            };
            let subst_seq = match node {
                Leaf(_) => subst_msa.seqs().record_by_id(&id).seq(),
                Internal(_) => subst_msa.ancestral_seqs().record_by_id(&id).seq(),
            };
            debug_assert!(
                mask_mapping.len() == subst_seq.len(),
                "Mask and substitution sequences must be the same length"
            );
            // apply mask: if mask is a gap, put a gap; otherwise keep the subst character
            let final_seq: Vec<u8> = mask_mapping
                .iter()
                .zip(subst_seq.iter())
                .map(|(mask, subst_char)| if mask.is_some() { *subst_char } else { GAP })
                .collect();

            combined_records.push(record!(&id, &final_seq));
        }

        // Construct the final ancestral MSA from the combined records
        let seqs = Sequences::new(combined_records)
            .unwrap_or_else(|e| bug!("Failed to create Sequences from combined records", e));
        let phylo = AA::from_aligned_with_ancestral(seqs, self.indel_sim.tree().clone())
            .unwrap_or_else(|e| {
                bug!(
                    "Failed to create ancestral alignment from combined records",
                    e
                )
            });
        // calling 'from_aligned_with_ancestral' removes all-gap cols. There should be none,
        // so asserting here that the msa did indeed not shrink.
        let aln_len = indel_msa.len();
        if phylo.msa.len() != aln_len {
            bug!("Final MSA length should match the indel MSA, but it does not")
        }
        phylo
    }
}

impl<T, R> AlignmentSimulation<R> for TKFMSASimulator<T>
where
    T: TKFModel + FragmentSampler + ExpectedRootLength,
    R: Rng + SeedableRng,
{
    fn simulate_ancestral_alignment<AA: AncestralAlignment>(
        &self,
        rng: &mut RandomGenerator<R>,
    ) -> PhyloInfo<AA> {
        self.simulate_with_fragments(rng).phylo
    }

    fn tree(&self) -> &Tree {
        self.indel_sim.tree()
    }
}

#[cfg(test)]
#[cfg_attr(coverage, coverage(off))]
mod private_tests {
    use hashbrown::HashSet;

    use crate::alignment::{Alignment, AncestralAlignment, MASA};
    use crate::phylo_info::PhyloInfo;
    use crate::random::DefaultGenerator;
    use crate::substitution_models::{SubstModel, GTR, JC69};
    use crate::tkf_model::{TKF91IndelModel, TKF92IndelModel};
    use crate::tree;

    use super::*;

    /// In the [`tkf92_simulation`] test, only A <-> T and G <-> C transitions are allowed.
    /// Checks that the simulation respects the mutation constraints of the GTR model used.
    /// So each column must contain at most two unique characters, which must be a valid pair.
    fn check_mutation_constraints(msa: &MASA, tree: &Tree) {
        for col_idx in 0..msa.len() {
            let mut col_chars = HashSet::new();
            for node_idx in tree.preorder() {
                let node_id = tree.node_id(node_idx);
                let seq = match node_idx {
                    Leaf(_) => msa.seqs().record_by_id(node_id).seq(),
                    Internal(_) => msa.ancestral_seqs().record_by_id(node_id).seq(),
                };
                let map = match node_idx {
                    Internal(_) => msa.ancestral_map(node_idx),
                    Leaf(_) => msa.leaf_map(node_idx),
                };

                if let Some(pos) = map[col_idx] {
                    col_chars.insert(seq[pos]);
                }
            }
            assert!(
                col_chars.len() <= 2,
                "Column {} has too many unique characters: {:?}",
                col_idx,
                col_chars
            );
            // Verify specific pairings if multiple characters exist
            if col_chars.len() == 2 {
                let chars: Vec<u8> = col_chars.into_iter().collect();
                let c1 = chars[0];
                let c2 = chars[1];
                let valid_pair = matches!(
                    (c1, c2),
                    (b'A', b'G') | (b'G', b'A') | (b'T', b'C') | (b'C', b'T')
                );
                assert!(
                    valid_pair,
                    "Invalid mutation pair in column {}: {} and {}",
                    col_idx, c1 as char, c2 as char
                );
            }
        }
    }

    #[test]
    fn tkf92_simulation() {
        let tree = tree!(
            "((((A:1.0,B:1.0)I1:0.5,C:1.5)I2:0.5,D:2.0)I3:0.5,((E:1.0,F:1.0)I4:0.5,G:1.5)I5:0.5)R;"
        );

        let freqs = [0.25, 0.25, 0.25, 0.25];
        let params = [1.0, 0.0, 0.0, 0.0, 0.0];
        let subst_model = SubstModel::<GTR>::new(&freqs, &params).unwrap();
        let lambda = 0.19;
        let mu = 0.2;
        let r = 0.8;
        let tkf_model = TKF92IndelModel::new(lambda, mu, r);

        let max_ins_len = Some(50);
        let simulator =
            TKFMSASimulator::new(tkf_model, subst_model, tree.clone(), max_ins_len).unwrap();
        let mut rng = DefaultGenerator::new(123);

        let msa = simulator.simulate_ancestral_alignment::<MASA>(&mut rng).msa;

        assert_eq!(msa.seq_count() + msa.ancestral_seqs().len(), 13);
        assert_eq!(msa.seq_count(), 7); // A, B, C, D, E, F, G
        assert!(msa.len() > 1);

        check_mutation_constraints(&msa, &tree);

        let phylo = PhyloInfo {
            msa: msa.clone(),
            tree,
        };
        assert!(
            phylo.check_dollos_constraint().is_ok(),
            "Simulated alignment must satisfy Dollo's constraint (no re-gain of characters)"
        );
    }

    #[test]
    fn tkf91_simulation_fixed_root_length() {
        let tree = tree!("(A:1.0,B:1.0)R;");
        let subst_model = SubstModel::<JC69>::default();
        let tkf_model = TKF91IndelModel::new(0.1, 0.2);

        let max_ins_len = Some(50);
        let mut simulator =
            TKFMSASimulator::new(tkf_model, subst_model, tree.clone(), max_ins_len).unwrap();
        simulator.root_length(RootLength::Defined(100));

        let mut rng = DefaultGenerator::new(123);
        let msa = simulator.simulate_ancestral_alignment::<MASA>(&mut rng).msa;
        let root_map = msa.ancestral_map(&tree.root);
        assert_eq!(root_map.iter().filter(|s| s.is_some()).count(), 100);
    }

    #[test]
    fn tkf92_simulation_fixed_root_length() {
        let tree = tree!("(A:1.0,B:1.0)R;");
        let subst_model = SubstModel::<JC69>::default();
        let tkf_model = TKF92IndelModel::new(0.1, 0.2, 0.8);
        let max_ins_len = Some(50);
        let mut simulator =
            TKFMSASimulator::new(tkf_model, subst_model, tree.clone(), max_ins_len).unwrap();
        simulator.root_length(RootLength::Defined(100));

        let mut rng = DefaultGenerator::new(123);
        let msa = simulator.simulate_ancestral_alignment::<MASA>(&mut rng).msa;
        let root_map = msa.ancestral_map(&tree.root);
        assert_eq!(root_map.iter().filter(|s| s.is_some()).count(), 100);
    }

    #[test]
    fn tkf_msa_masking_matches_indel_gaps() {
        // Simulating the full TKF process (indels + substitutions) with the same seed as an
        // indel-only simulation must place gaps at exactly the same positions: the substitution
        // simulation only fills in characters where the indel MSA has a character.
        let tree = tree!(
            "((((A:1.0,B:1.0)I1:0.5,C:1.5)I2:0.5,D:2.0)I3:0.5,((E:1.0,F:1.0)I4:0.5,G:1.5)I5:0.5)R;"
        );
        let subst_model = SubstModel::<JC69>::default();
        let tkf_model = TKF92IndelModel::new(0.19, 0.2, 0.8);
        let max_ins_len = Some(50);
        let seed = 123;

        let indel_sim =
            TKFIndelMSASimulator::new(tkf_model.clone(), tree.clone(), max_ins_len).unwrap();
        let mut indel_rng = DefaultGenerator::new(seed);
        let indel_msa = indel_sim
            .simulate_with_fragments::<MASA, _>(&mut indel_rng)
            .phylo
            .msa;

        let full_sim =
            TKFMSASimulator::new(tkf_model, subst_model, tree.clone(), max_ins_len).unwrap();
        let mut full_rng = DefaultGenerator::new(seed);
        let full_msa = full_sim
            .simulate_ancestral_alignment::<MASA>(&mut full_rng)
            .msa;

        assert_eq!(indel_msa.len(), full_msa.len());
        for node_idx in tree.preorder() {
            let (indel_map, full_map) = match node_idx {
                Internal(_) => (
                    indel_msa.ancestral_map(node_idx),
                    full_msa.ancestral_map(node_idx),
                ),
                Leaf(_) => (indel_msa.leaf_map(node_idx), full_msa.leaf_map(node_idx)),
            };
            assert_eq!(indel_map.len(), full_map.len());
            let same_gaps = indel_map
                .iter()
                .zip(full_map.iter())
                .all(|(indel_site, full_site)| indel_site.is_some() == full_site.is_some());
            assert!(
                same_gaps,
                "Gap pattern of node '{}' differs between the indel and the full simulation",
                tree.node_id(node_idx)
            );
        }
    }
}
