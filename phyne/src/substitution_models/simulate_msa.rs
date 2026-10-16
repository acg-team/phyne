//! Simulation of alignments and ancestral alignments by substitutions only.
//!
//! This module provides the [`SubstitutionSimulator`] which evolves characters down a
//! [`Tree`](`crate::tree::Tree`) under a substitution model, producing an alignment without any gaps.
//!
//! # Example
//! ```
//! use phyne::alignment::{Alignment, AlignmentSimulation, MASA};
//! use phyne::random::DefaultGenerator;
//! use phyne::substitution_models::{dna_models::JC69, SubstModel, SubstitutionSimulator};
//! use phyne::tree;
//! # use phyne::Result;
//!
//! # fn main() -> Result<()> {
//! let tree = tree!("((A:1.0,B:1.0)I:1.0,C:1.0)R;");
//! let seed = 123;
//! let alignment_length = 20;
//! let simulator = SubstitutionSimulator::new(
//!     SubstModel::<JC69>::default(),
//!     tree,
//!     alignment_length,
//! )?;
//! let mut rng = DefaultGenerator::new(seed);
//! let msa: MASA = simulator.simulate_ancestral_alignment(&mut rng).msa;
//! assert_eq!(msa.seq_count(), 3);
//! assert_eq!(msa.len(), alignment_length);
//! # Ok(()) }
//! ```

use bio::io::fasta::Record;
use hashbrown::HashMap;
use log::warn;
use rand::{distr::weighted::WeightedIndex, Rng, SeedableRng};

use crate::alignment::{AlignmentSimulation, AncestralAlignment, Sequences};
use crate::alphabets::Alphabet;
use crate::phylo_info::{set_missing_tree_node_ids, PhyloInfo};
use crate::random::RandomGenerator;
use crate::record_wo_desc as record;
use crate::substitution_models::{QMatrix, SubstModel};
use crate::tree::{NodeIdx, Tree};
use crate::{bug, Result, MAX_BLEN};

/// Simulates substitutions on a fixed-length ancestral alignment.
///
/// The root sequence is sampled from the substitution model's stationary distribution and
/// evolved down the tree along the branches. The simulation produces no gaps.
#[derive(Debug, Clone)]
pub struct SubstitutionSimulator {
    tree: Tree,
    alphabet: Alphabet,
    root_dist: WeightedIndex<f64>,
    /// Probability distributions for each branch (NodeIdx) and each parent character (index in the Vec).
    p_weighted: HashMap<NodeIdx, Vec<WeightedIndex<f64>>>,
    alignment_length: usize,
}

fn unwrap_weighted_index(slice: &[f64]) -> WeightedIndex<f64> {
    WeightedIndex::new(slice).unwrap_or_else(|e| bug!("Getting WeightedIndex from slice failed", e))
}

impl SubstitutionSimulator {
    /// Create a new SubstitutionSimulator with the given substitution model, tree and
    /// alignment length.
    ///
    /// A zero `alignment_length` is accepted, but simulation will emit a warning and return an empty alignment.
    ///
    /// # Example
    /// ```
    /// use phyne::alignment::{Alignment, AlignmentSimulation, MASA};
    /// use phyne::random::DefaultGenerator;
    /// use phyne::substitution_models::{dna_models::JC69, SubstModel, SubstitutionSimulator};
    /// use phyne::tree;
    /// # use phyne::Result;
    ///
    /// # fn main() -> Result<()> {
    /// let tree = tree!("((A:1.0,B:1.0)I:1.0,C:1.0)R;");
    /// let msa_len = 20;
    /// let simulator = SubstitutionSimulator::new(SubstModel::<JC69>::default(), tree, msa_len)?;
    /// let mut rng = DefaultGenerator::new(123);
    /// let phylo = simulator.simulate_ancestral_alignment::<MASA>(&mut rng);
    /// assert_eq!(phylo.msa.seq_count(), 3);
    /// assert_eq!(phylo.msa.len(), msa_len);
    /// # Ok(()) }
    /// ```
    ///
    /// # Errors
    /// * If the tree node IDs cannot be completed (see
    ///   [`set_missing_tree_node_ids`](`crate::phylo_info::set_missing_tree_node_ids`)).
    pub fn new<Q: QMatrix>(
        model: SubstModel<Q>,
        mut tree: Tree,
        alignment_length: usize,
    ) -> Result<Self> {
        set_missing_tree_node_ids(&mut tree)?;
        let root_dist = unwrap_weighted_index(model.qmatrix.freqs().as_slice());

        let mut p_weighted = HashMap::with_capacity(tree.len());
        for idx in tree.preorder().iter().skip(1) {
            let blen = tree.node(idx).blen;
            let qmat = model.qmatrix.q();
            let p = if blen > MAX_BLEN {
                (qmat * MAX_BLEN).exp()
            } else {
                (qmat * blen).exp()
            };

            let mut column_dists = Vec::with_capacity(p.ncols());
            for col in 0..p.ncols() {
                let column = p.column(col);
                let weighted_index = unwrap_weighted_index(column.as_slice());
                column_dists.push(weighted_index);
            }
            p_weighted.insert(*idx, column_dists);
        }

        let alphabet = *Q::alphabet();

        Ok(Self {
            tree,
            alphabet,
            root_dist,
            p_weighted,
            alignment_length,
        })
    }

    /// Sets the alignment length for the simulation.
    ///
    /// A zero `alignment_length` is accepted, but simulation will emit a warning and return an empty alignment.
    ///
    /// # Example
    /// ```
    /// use phyne::alignment::{Alignment, AlignmentSimulation, MASA};
    /// use phyne::random::DefaultGenerator;
    /// use phyne::substitution_models::{dna_models::JC69, SubstModel, SubstitutionSimulator};
    /// use phyne::tree;
    /// # use phyne::Result;
    ///
    /// # fn main() -> Result<()> {
    /// let tree = tree!("((A:1.0,B:1.0)I:1.0,C:1.0)R;");
    /// let msa_len = 20;
    /// let mut simulator =
    ///     SubstitutionSimulator::new(SubstModel::<JC69>::default(), tree, msa_len)?;
    /// let new_msa_len = 100;
    /// simulator.alignment_length(new_msa_len);
    ///
    /// let mut rng = DefaultGenerator::new(123);
    /// let phylo = simulator.simulate_ancestral_alignment::<MASA>(&mut rng);
    /// assert_eq!(phylo.msa.len(), new_msa_len);
    /// # Ok(()) }
    /// ```
    pub fn alignment_length(&mut self, length: usize) {
        self.alignment_length = length;
    }

    fn simulate_root_sequence<R: Rng + SeedableRng>(
        &self,
        alignment_length: usize,
        rng: &mut RandomGenerator<R>,
    ) -> Vec<usize> {
        (0..alignment_length)
            .map(|_| rng.sample(&self.root_dist))
            .collect()
    }

    fn simulate_child_sequence<R: Rng + SeedableRng>(
        &self,
        parent_seq: &[usize],
        node_idx: NodeIdx,
        rng: &mut RandomGenerator<R>,
    ) -> Vec<usize> {
        let column_dists = self.p_weighted.get(&node_idx).unwrap();
        parent_seq
            .iter()
            .map(|&parent_state| rng.sample(&column_dists[parent_state]))
            .collect()
    }

    fn sequences_to_records(&self, sequences: &HashMap<NodeIdx, Vec<usize>>) -> Vec<Record> {
        sequences
            .iter()
            .map(|(node_idx, seq)| {
                let id = self.tree.node_id(node_idx);
                let char_seq: Vec<u8> = seq.iter().map(|&s| self.alphabet.symbols()[s]).collect();
                record!(id, &char_seq)
            })
            .collect()
    }

    fn simulate_sequences<R: Rng + SeedableRng>(
        &self,
        alignment_length: usize,
        rng: &mut RandomGenerator<R>,
    ) -> Sequences {
        let mut sequences = HashMap::with_capacity(self.tree.len());

        sequences.insert(
            self.tree.root,
            self.simulate_root_sequence(alignment_length, rng),
        );

        for node_idx in self.tree.preorder().iter().skip(1) {
            let parent_idx = self.tree.parent(node_idx).unwrap();
            let parent_seq = sequences.get(&parent_idx).unwrap();
            sequences.insert(
                *node_idx,
                self.simulate_child_sequence(parent_seq, *node_idx, rng),
            );
        }

        let records = self.sequences_to_records(&sequences);

        Sequences::new(records)
            .unwrap_or_else(|e| bug!("Creating Sequences from simulated records failed", e))
    }

    fn zero_length_phylo_info<AA: AncestralAlignment>(&self) -> PhyloInfo<AA> {
        let records = self
            .tree
            .preorder()
            .iter()
            .map(|&node_idx| {
                let id = self.tree.node_id(&node_idx);
                record!(id, b"")
            })
            .collect();
        let seqs = Sequences::new(records).unwrap();
        AA::from_aligned_with_ancestral(seqs, self.tree.clone()).unwrap()
    }

    pub(crate) fn simulate_ancestral_alignment_with_length<
        R: Rng + SeedableRng,
        AA: AncestralAlignment,
    >(
        &self,
        alignment_length: usize,
        rng: &mut RandomGenerator<R>,
    ) -> PhyloInfo<AA> {
        if alignment_length == 0 {
            warn!("Alignment length is set to 0, will produce an empty alignment");
            return self.zero_length_phylo_info();
        }
        let seqs = self.simulate_sequences(alignment_length, rng);
        AA::from_aligned_with_ancestral(seqs, self.tree.clone()).unwrap_or_else(|e| {
            bug!(
                "Creating AncestralAlignment from simulated Sequences failed",
                e
            )
        })
    }
}

impl<R> AlignmentSimulation<R> for SubstitutionSimulator
where
    R: Rng + SeedableRng,
{
    fn simulate_ancestral_alignment<AA: AncestralAlignment>(
        &self,
        rng: &mut RandomGenerator<R>,
    ) -> PhyloInfo<AA> {
        self.simulate_ancestral_alignment_with_length(self.alignment_length, rng)
    }

    fn tree(&self) -> &Tree {
        &self.tree
    }
}

#[cfg(test)]
#[cfg_attr(coverage, coverage(off))]
mod private_tests {

    use crate::alignment::{Alignment, MASA};
    use crate::random::{DefaultGenerator, FakeGenerator, FakeRng};
    use crate::substitution_models::{dna_models::JC69, SubstModel};
    use crate::tree;

    use super::*;

    #[test]
    fn test_substitution_simulator() {
        let model = SubstModel::<JC69>::default();
        let tree = tree!("((A:2.0,B:1.2)AB:1.3,(C:2.0,D:2.0)CD:2.0)R;");
        let values = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
        let mut rng = RandomGenerator::from_rng(FakeRng::from_f64_values(values));
        let alignment_length = 50;
        let simulator = SubstitutionSimulator::new(model, tree.clone(), alignment_length).unwrap();

        let alignment: MASA = simulator.simulate_ancestral_alignment(&mut rng).msa;

        assert_eq!(alignment.len(), alignment_length);
        assert_eq!(
            alignment.seq_count() + alignment.ancestral_seqs().len(),
            tree.len()
        );
        // no gaps in this simulation, so all sequences should have the same length as the alignment
        for seq in alignment.seqs() {
            assert_eq!(seq.seq().len(), alignment.len());
        }
    }

    #[test]
    fn test_substitution_simulator_zero_length() {
        let model = SubstModel::<JC69>::default();
        let tree = tree!("((A:2.0,B:1.2)AB:1.3,(C:2.0,D:2.0)CD:2.0)R;");
        let mut rng = FakeGenerator::default();

        let assert_empty = |alignment: &MASA| {
            assert_eq!(alignment.len(), 0);
            assert_eq!(
                alignment.seq_count() + alignment.ancestral_seqs().len(),
                tree.len()
            );
            for seq in alignment.seqs() {
                assert!(seq.seq().is_empty());
            }
            for seq in alignment.ancestral_seqs() {
                assert!(seq.seq().is_empty());
            }
            // every node in the tree must have exactly one sequence with that id
            let ids: std::collections::HashSet<&str> = alignment
                .seqs()
                .into_iter()
                .chain(alignment.ancestral_seqs())
                .map(|seq| seq.id())
                .collect();
            assert_eq!(ids.len(), tree.len());
            for node_idx in tree.preorder() {
                assert!(ids.contains(tree.node_id(node_idx)));
            }
        };

        // zero length passed directly to the constructor
        let simulator = SubstitutionSimulator::new(model.clone(), tree.clone(), 0).unwrap();
        let alignment: MASA = simulator.simulate_ancestral_alignment(&mut rng).msa;
        assert_empty(&alignment);

        // zero length set after construction
        let mut simulator = SubstitutionSimulator::new(model, tree.clone(), 50).unwrap();
        simulator.alignment_length(0);
        let alignment: MASA = simulator.simulate_ancestral_alignment(&mut rng).msa;
        assert_empty(&alignment);
    }

    #[test]
    fn test_reproducibility() {
        // GTR with same parameters to ensure reproducibility across RNGs
        let model = SubstModel::<JC69>::default();
        let tree = tree!("((A:0.5,B:0.5)AB:0.7,(C:0.6,D:0.6)CD:0.6)R;");

        let mut rng1 = DefaultGenerator::new(42);
        let mut rng2 = DefaultGenerator::new(42);

        let simulator1 = SubstitutionSimulator::new(model.clone(), tree.clone(), 100).unwrap();
        let simulator2 = SubstitutionSimulator::new(model, tree.clone(), 100).unwrap();

        let alignment1: MASA = simulator1.simulate_ancestral_alignment(&mut rng1).msa;
        let alignment2: MASA = simulator2.simulate_ancestral_alignment(&mut rng2).msa;

        assert_eq!(
            alignment1.to_string(),
            alignment2.to_string(),
            "Same seed should produce identical alignments"
        );
    }
}
