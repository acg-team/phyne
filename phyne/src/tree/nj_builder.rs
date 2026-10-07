use anyhow::Context;
use log::{debug, info};
use nalgebra::{DMatrix, DVector};
use rand::distr::weighted::WeightedIndex;
use rand::{Rng, SeedableRng};

use crate::alignment::Sequences;
use crate::evolutionary_distances::EvolutionaryDistance;
use crate::random::RandomGenerator;
use crate::tree::nj_matrices::DistanceMatrix;
use crate::tree::tree_builder::TreeBuilder;
use crate::tree::{NodeIdx, Tree};
use crate::Result;

#[derive(Clone, Debug, PartialEq)]
pub enum Strategy {
    ArgMax,
    SoftmaxUniform(f64),
}

pub struct NJTreeBuilder<D: EvolutionaryDistance> {
    randomise: Strategy,
    distance_function: D,
}

impl<D: EvolutionaryDistance> TreeBuilder for NJTreeBuilder<D> {
    /// Builds a tree from the given sequences using the Neighbor Joining algorithm.
    /// This first computes the distance matrix using the provided distance function,
    /// and then constructs the tree using the NJ algorithm.
    ///
    /// # Example
    /// ```rust
    /// use phylo::alignment::Sequences;
    /// use phylo::evolutionary_distances::LevenshteinDNACorrected;
    /// use phylo::io::read_sequences;
    /// use phylo::random::DefaultGenerator;
    /// use phylo::tree::{NJTreeBuilder, TreeBuilder};
    ///
    /// # use phylo::Result;
    /// # fn main() -> Result<()> {
    /// let sequences = Sequences::new(read_sequences("./data/sequences_DNA1.fasta")?)?;
    /// let mut rng = DefaultGenerator::default();
    /// let nj_builder = NJTreeBuilder::new(LevenshteinDNACorrected {});
    /// let tree = nj_builder.build(&sequences, &mut rng)?;
    /// assert_eq!(tree.len(), 7);
    /// assert_eq!(tree.leaves().len(), 4);
    /// # Ok(()) }
    /// ```
    fn build(
        self,
        sequences: &Sequences,
        rng: &mut RandomGenerator<impl Rng + SeedableRng>,
    ) -> Result<Tree> {
        let distances = self.compute_distance_matrix(sequences);
        self.build_from_distances(distances, sequences, rng)
    }
}

impl<D: EvolutionaryDistance> NJTreeBuilder<D> {
    /// Creates a Neighbor Joining Tree Builder with ArgMax strategy, which uses argmax to minimise the tree length.
    /// This implements the classic NJ algorithm, where ties (multiple pairs of nodes with the smallest distance) are
    /// broken uniformly at random.
    ///
    /// # Example
    /// ```rust
    /// use phylo::alignment::Sequences;
    /// use phylo::evolutionary_distances::LevenshteinDNACorrected;
    /// use phylo::io::read_sequences;
    /// use phylo::random::DefaultGenerator;
    /// use phylo::tree::{  NJTreeBuilder, TreeBuilder};
    ///
    /// # use phylo::Result;
    /// # fn main() -> Result<()> {
    /// let sequences = Sequences::new(read_sequences("./data/sequences_DNA1.fasta")?)?;
    /// let mut rng = DefaultGenerator::default();
    /// let tree = NJTreeBuilder::new(LevenshteinDNACorrected {}).build(&sequences, &mut rng)?;
    /// assert_eq!(tree.len(), 7);
    /// assert_eq!(tree.leaves().len(), 4);
    /// # Ok(()) }
    /// ```
    pub fn new(distance_function: D) -> Self {
        info!("Creating NJTreeBuilder with the traditional strategy of selecting the best pair of nodes to join that minimise
        tree length and breaking ties uniformly at random.");
        Self {
            randomise: Strategy::ArgMax,
            distance_function,
        }
    }

    /// Creates a Neighbor Joining Tree Builder with Softmax strategy, introduces stochasticity to tree building.
    /// This uses SoftMax to select the next pair of nodes to join based on their contribution to minimising the tree length
    /// (`delta_tree_len`). Nodes that contribute more to minimising the tree length have a higher probability of being selected.
    /// Temperature can be between 0.0 and 1.0 and interpolates between the uniform and softmax distributions to select
    /// the next pair of nodes to join. A temperature of 0.0 is fully uniform, while a temperature of 1.0 is fully softmax.
    ///
    /// # Example
    /// ```rust
    /// use phylo::alignment::Sequences;
    /// use phylo::evolutionary_distances::LevenshteinDNACorrected;
    /// use phylo::io::read_sequences;
    /// use phylo::random::DefaultGenerator;
    /// use phylo::tree::{NJTreeBuilder, TreeBuilder};
    ///
    /// # use phylo::Result;
    /// # fn main() -> Result<()> {
    /// let sequences = Sequences::new(read_sequences("./data/sequences_DNA1.fasta")?)?;
    /// let mut rng = DefaultGenerator::default();
    /// let nj_builder = NJTreeBuilder::new_with_softmax(LevenshteinDNACorrected {}, 0.5);
    /// let tree = nj_builder.build(&sequences, &mut rng)?;
    /// assert_eq!(tree.len(), 7);
    /// assert_eq!(tree.leaves().len(), 4);
    /// # Ok(()) }
    /// ```
    pub fn new_with_softmax(distance_function: D, temperature: f64) -> Self {
        info!("Creating NJTreeBuilder with the softmax strategy of selecting the best pair of nodes to join to increase stochasticity
        in tree building. Using temperature {temperature} to interpolate between uniform and softmax distributions.");
        if temperature > 1.0 {
            debug!("Temperature should not be greater than 1.0 (set to {temperature}), clamping to 1.0");
        } else if temperature < 0.0 {
            debug!(
                "Temperature should not be less than 0.0 (set to {temperature}), clamping to 0.0"
            );
        }
        let t = temperature.clamp(0.0, 1.0);
        Self {
            randomise: Strategy::SoftmaxUniform(t),
            distance_function,
        }
    }

    /// Computes the softmax distribution from the delta tree lengths without temperature
    /// scaling. As the delta tree lengths are negative, they are inverted before computing
    /// the distribution, so that smaller (more negative) delta tree lengths have higher probabilities.
    fn softmax_from_deltas(mut delta_tree_len: DVector<f64>) -> DVector<f64> {
        // Invert deltas for softmax, most negative delta should have the highest probability
        delta_tree_len.scale_mut(-1.0);
        // Avoid copying the matrix by mutating in place
        for element in delta_tree_len.iter_mut() {
            *element = element.exp();
        }
        delta_tree_len.unscale_mut(delta_tree_len.sum());
        delta_tree_len
    }

    /// Computes the softmax probabilities from the delta tree lengths with temperature
    /// scaling, interpolating linearly between uniform and softmax distributions.
    /// A temperature of 0.0 results in a uniform distribution, while a temperature of 1.0
    /// results in a pure softmax distribution.
    fn softmax(delta_tree_len: DVector<f64>, temperature: f64) -> DVector<f64> {
        debug_assert!(
            !delta_tree_len.is_empty(),
            "The input vector must not be empty."
        );

        let mut exp_mat = Self::softmax_from_deltas(delta_tree_len);
        let uniform_weight = 1.0 / exp_mat.len() as f64;

        // Interpolated probabilities, temp = 0.0 is the uniform distribution, temp = 1.0 is the softmax of distances
        // Avoid copying the matrix by mutating in place
        for element in exp_mat.iter_mut() {
            *element = ((1.0 - temperature) * uniform_weight) + ((temperature) * *element);
        }
        exp_mat
    }

    /// Computes a vector indicating the position(s) of the minimum value(s) in the input vector,
    /// i.e., a vector with 1 at indices of the minimum, and 0 elsewhere.
    /// Used to uniformly select among the minimum delta tree lengths.
    fn argmin(mut delta_tree_len: DVector<f64>) -> DVector<f64> {
        debug_assert!(
            !delta_tree_len.is_empty(),
            "The input vector must not be empty."
        );

        let min_value = delta_tree_len.argmin().1;
        for elem in delta_tree_len.iter_mut() {
            if *elem == min_value {
                *elem = 1.0;
            } else {
                *elem = 0.0;
            }
        }
        delta_tree_len
    }

    /// Builds a tree from a given distance matrix using the Neighbor Joining algorithm.
    /// TODO: does not actually need the sequences, only used to create the tree with
    /// correct leaf ids, should be refactored @junniest
    fn build_from_distances(
        self,
        mut distances: DistanceMatrix,
        sequences: &Sequences,
        rng: &mut RandomGenerator<impl Rng + SeedableRng>,
    ) -> Result<Tree> {
        let n = distances.distances.ncols();
        let mut tree = Tree::new(sequences)?;
        let root_idx = usize::from(&tree.root);
        for cur_idx in n..=root_idx {
            let delta_lengths = distances.compute_delta_tree_length();

            let distribution = match self.randomise {
                Strategy::SoftmaxUniform(t) => Self::softmax(delta_lengths, t),
                Strategy::ArgMax => Self::argmin(delta_lengths),
            };

            let index = rng.sample(
                &WeightedIndex::new(distribution.iter())
                    .context("Failed to generate WeightedIndex from distribution")?,
            );

            let (i, j) = lower_triangle_index(index);
            let idx_new = cur_idx;
            let (blen_i, blen_j) = distances.branch_lengths(i, j, cur_idx == root_idx);

            tree.add_parent(
                idx_new,
                &distances.idx[i],
                &distances.idx[j],
                blen_i,
                blen_j,
            );

            distances = distances
                .add_merge_node(idx_new)
                .recompute_new_node_distances(i, j)
                .remove_merged_nodes(i, j);
        }

        tree.finalise();
        Ok(tree)
    }

    /// Computes the distance matrix for the given sequences using the provided distance function.
    fn compute_distance_matrix(&self, sequences: &Sequences) -> DistanceMatrix {
        let nseqs = sequences.len();
        let mut distances = DMatrix::zeros(nseqs, nseqs);
        for i in 0..nseqs {
            for j in (i + 1)..nseqs {
                let dist = self.distance_function.dist(&sequences[i], &sequences[j]);
                distances[(i, j)] = dist;
                distances[(j, i)] = dist;
            }
        }
        DistanceMatrix {
            idx: (0..nseqs).map(NodeIdx::Leaf).collect(),
            distances,
        }
    }
}

/// Converts a linear index `k` in the lower triangle of a matrix (excluding the diagonal)
/// to its corresponding `(i, j)` coordinates, both 0-based.
fn lower_triangle_index(k: usize) -> (usize, usize) {
    let p = ((1 + 8 * k).isqrt() - 1) / 2;
    let i = p + 1;
    let j = k - p * (p + 1) / 2;
    (i, j)
}

#[cfg(test)]
#[cfg_attr(coverage, coverage(off))]
mod tests {
    use approx::assert_relative_eq;
    use assert_matches::assert_matches;
    use nalgebra::{dmatrix, dvector};
    use rstest::rstest;

    use crate::evolutionary_distances::{
        LevenshteinDNACorrected as LDNACorr, LevenshteinProteinCorrected,
    };
    use crate::random::{FakeGenerator, FakeRng};
    use crate::tree::Node;
    use crate::tree::NodeIdx::{self, Internal as I, Leaf as L};
    use crate::{record_wo_desc as record, tree};

    use super::*;

    #[cfg(test)]
    fn is_unique<T: std::cmp::Eq + std::hash::Hash>(vec: &[T]) -> bool {
        let set: std::collections::HashSet<_> = vec.iter().collect();
        set.len() == vec.len()
    }

    #[test]
    fn compute_distance_matrix_close() {
        let sequences = Sequences::new_unchecked(vec![
            record!("A0", b"C"),
            record!("B1", b"A"),
            record!("C2", b"AA"),
            record!("D3", b"A"),
            record!("E4", b"CC"),
        ]);

        let mat = NJTreeBuilder::new(LDNACorr {}).compute_distance_matrix(&sequences);
        let true_mat = dmatrix![
        0.0, 26.728641210756745, 26.728641210756745, 26.728641210756745, 0.8239592165010822;
        26.728641210756745, 0.0, 0.8239592165010822, 0.0, 26.728641210756745;
        26.728641210756745, 0.8239592165010822, 0.0, 0.8239592165010822, 26.728641210756745;
        26.728641210756745, 0.0, 0.8239592165010822, 0.0, 26.728641210756745;
        0.8239592165010822, 26.728641210756745, 26.728641210756745, 26.728641210756745, 0.0];
        assert_eq!(mat.distances, true_mat);
    }

    #[test]
    fn compute_distance_matrix_far() {
        let sequences = Sequences::new_unchecked(vec![
            record!("A0", b"AAAAAAAAAAAAAAAAAAAA"),
            record!("B1", b"AAAAAAAAAAAAAAAAAAAA"),
            record!("C2", b"AAAAAAAAAAAAAAAAAAAAAAAAA"),
            record!("D3", b"CAAAAAAAAAAAAAAAAAAA"),
        ]);
        let mat = NJTreeBuilder::new(LDNACorr {}).compute_distance_matrix(&sequences);
        let true_mat = dmatrix![
        0.0, 0.0, 0.2326161962278796, 0.051744653615213576;
        0.0, 0.0, 0.2326161962278796, 0.051744653615213576;
        0.2326161962278796, 0.2326161962278796, 0.0, 0.28924686060898847;
        0.051744653615213576, 0.051744653615213576, 0.28924686060898847, 0.0];
        assert_eq!(mat.distances, true_mat);
    }

    #[test]
    fn nj_tree_original_paper() {
        // Compare against the original paper tree
        // https://academic.oup.com/mbe/article/4/4/406/1029664
        let nj_distances = DistanceMatrix {
            idx: (0..8).map(NodeIdx::Leaf).collect(),
            distances: dmatrix![
                0.0, 7.0, 8.0, 11.0, 13.0, 16.0, 13.0, 17.0;
                7.0, 0.0, 5.0, 8.0, 10.0, 13.0, 10.0, 14.0;
                8.0, 5.0, 0.0, 5.0, 7.0, 10.0, 7.0, 11.0;
                11.0, 8.0, 5.0, 0.0, 8.0, 11.0, 8.0, 12.0;
                13.0, 10.0, 7.0, 8.0, 0.0, 5.0, 6.0, 10.0;
                16.0, 13.0, 10.0, 11.0, 5.0, 0.0, 9.0, 13.0;
                13.0, 10.0, 7.0, 8.0, 6.0, 9.0, 0.0, 8.0;
                17.0, 14.0, 11.0, 12.0, 10.0, 13.0, 8.0, 0.0;
            ],
        };
        let sequences =
            Sequences::new_unchecked((1..=8).map(|i| record!(&i.to_string(), b"")).collect());
        let mut rng = FakeGenerator::default();
        let nj_tree = NJTreeBuilder::new(LDNACorr {})
            .build_from_distances(nj_distances, &sequences, &mut rng)
            .unwrap();
        let correct_tree =
            tree!("((8:6,7:2):0.5,((5:1,6:4):2,(4:3,(3:1,(1:5,2:2):2):1):2):0.5):0.0;");
        assert_eq!(nj_tree.length, correct_tree.length);
        for leaf in nj_tree.leaves() {
            assert_eq!(leaf.blen, correct_tree.by_id(&leaf.id).blen);
        }
    }

    #[test]
    fn nj_correct_2() {
        // NJ based on example from https://www.tenderisthebyte.com/blog/2022/08/31/neighbor-joining-trees/#neighbor-joining-trees
        let nj_distances = DistanceMatrix {
            idx: (0..4).map(NodeIdx::Leaf).collect(),
            distances: dmatrix![
                0.0, 4.0, 5.0, 10.0;
                4.0, 0.0, 7.0, 12.0;
                5.0, 7.0, 0.0, 9.0;
                10.0, 12.0, 9.0, 0.0],
        };
        let sequences = Sequences::new_unchecked(vec![
            record!("A", b""),
            record!("B", b""),
            record!("C", b""),
            record!("D", b""),
        ]);
        let mut rng = FakeGenerator::default();
        let tree = NJTreeBuilder::new(LDNACorr {})
            .build_from_distances(nj_distances, &sequences, &mut rng)
            .unwrap();
        assert_eq!(tree.by_id("A").blen, 1.0);
        assert_eq!(tree.by_id("B").blen, 3.0);
        assert_eq!(tree.by_id("C").blen, 2.0);
        assert_eq!(tree.by_id("D").blen, 7.0);
        assert_eq!(tree.node(&I(4)).blen, 1.0);
        assert_eq!(tree.node(&I(5)).blen, 1.0);
        assert_eq!(tree.len(), 7);
        assert_eq!(tree.postorder.len(), 7);
        assert!(is_unique(&tree.postorder));
        assert_eq!(tree.preorder.len(), 7);
        assert!(is_unique(&tree.preorder));
    }

    #[test]
    fn protein_nj_correct() {
        // NJ based on example sequences from "./data/sequences_protein1.fasta"
        let nj_distances = DistanceMatrix {
            idx: (0..4).map(NodeIdx::Leaf).collect(),
            distances: dmatrix![
                0.0, 0.0, 0.0, 0.2;
                0.0, 0.0, 0.0, 0.2;
                0.0, 0.0, 0.0, 0.2;
                0.2, 0.2, 0.2, 0.0],
        };
        let sequences = Sequences::new_unchecked(vec![
            record!("A0", b""),
            record!("B1", b""),
            record!("C2", b""),
            record!("D3", b""),
        ]);

        // The distance function parameter is required by NJTreeBuilder but not used in this test,
        // since distances are provided directly via build_from_distances.
        // The choice of distance function here does not affect the test outcome.
        let tree = NJTreeBuilder::new(LevenshteinProteinCorrected {})
            .build_from_distances(nj_distances, &sequences, &mut FakeGenerator::default())
            .unwrap();
        assert_eq!(tree.len(), 7);
        assert_eq!(tree.postorder.len(), 7);
        assert!(is_unique(&tree.postorder));
        assert_eq!(tree.preorder.len(), 7);
        assert!(is_unique(&tree.preorder));
    }

    #[test]
    fn nj_correct_wiki_example() {
        // NJ based on example from https://en.wikipedia.org/wiki/Neighbor_joining
        let nj_distances = DistanceMatrix {
            idx: (0..5).map(NodeIdx::Leaf).collect(),
            distances: dmatrix![
                0.0, 5.0, 9.0, 9.0, 8.0;
                5.0, 0.0, 10.0, 10.0, 9.0;
                9.0, 10.0, 0.0, 8.0, 7.0;
                9.0, 10.0, 8.0, 0.0, 3.0;
                8.0, 9.0, 7.0, 3.0, 0.0],
        };
        let sequences = Sequences::new_unchecked(vec![
            record!("a", b""),
            record!("b", b""),
            record!("c", b""),
            record!("d", b""),
            record!("e", b""),
        ]);
        let tree = NJTreeBuilder::new(LDNACorr {})
            .build_from_distances(nj_distances, &sequences, &mut FakeGenerator::default())
            .unwrap();
        assert_eq!(tree.by_id("a").blen, 2.0);
        assert_eq!(tree.by_id("b").blen, 3.0);
        assert_eq!(tree.by_id("c").blen, 4.0);
        assert_eq!(tree.by_id("d").blen, 2.0);
        assert_eq!(tree.by_id("e").blen, 1.0);
        assert_eq!(tree.node(&I(5)).blen, 3.0);
        assert_eq!(tree.node(&I(6)).blen, 1.0);
        assert_eq!(tree.node(&I(7)).blen, 1.0);
        assert_eq!(tree.len(), 9);
        assert_eq!(tree.postorder.len(), 9);
        assert!(is_unique(&tree.postorder));
        assert_eq!(tree.preorder.len(), 9);
        assert!(is_unique(&tree.preorder));
    }

    #[test]
    fn nj_correct() {
        let nj_distances = DistanceMatrix {
            idx: (0..5).map(NodeIdx::Leaf).collect(),
            distances: dmatrix![
                0.0, 5.0, 9.0, 9.0, 8.0;
                5.0, 0.0, 10.0, 10.0, 9.0;
                9.0, 10.0, 0.0, 8.0, 7.0;
                9.0, 10.0, 8.0, 0.0, 3.0;
                8.0, 9.0, 7.0, 3.0, 0.0],
        };
        let sequences = Sequences::new_unchecked(vec![
            record!("A0", b""),
            record!("B1", b""),
            record!("C2", b""),
            record!("D3", b""),
            record!("E4", b""),
        ]);
        let nj_tree = NJTreeBuilder::new(LDNACorr {})
            .build_from_distances(nj_distances, &sequences, &mut FakeGenerator::default())
            .unwrap();
        let nodes = vec![
            Node::new_leaf(0, Some(I(5)), 2.0, "A0".to_string()),
            Node::new_leaf(1, Some(I(5)), 3.0, "B1".to_string()),
            Node::new_leaf(2, Some(I(7)), 4.0, "C2".to_string()),
            Node::new_leaf(3, Some(I(6)), 2.0, "D3".to_string()),
            Node::new_leaf(4, Some(I(6)), 1.0, "E4".to_string()),
            Node::new_internal(5, Some(I(7)), vec![L(1), L(0)], 3.0, "".to_string()),
            Node::new_internal(6, Some(I(8)), vec![L(4), L(3)], 1.0, "".to_string()),
            Node::new_internal(7, Some(I(8)), vec![I(5), L(2)], 1.0, "".to_string()),
            Node::new_internal(8, None, vec![I(7), I(6)], 0.0, "".to_string()),
        ];
        assert_eq!(nj_tree.root, I(8));
        assert_eq!(nj_tree.nodes, nodes);
    }

    #[test]
    fn nj_correct_web_example() {
        let nj_distances = DistanceMatrix {
            idx: (0..4).map(NodeIdx::Leaf).collect(),
            distances: dmatrix![
                    0.0, 4.0, 5.0, 10.0;
                    4.0, 0.0, 7.0, 12.0;
                    5.0, 7.0, 0.0, 9.0;
                    10.0, 12.0, 9.0, 0.0],
        };
        let sequences = Sequences::new_unchecked(vec![
            record!("A0", b""),
            record!("B1", b""),
            record!("C2", b""),
            record!("D3", b""),
        ]);

        let nj_tree = NJTreeBuilder::new(LDNACorr {})
            .build_from_distances(nj_distances, &sequences, &mut FakeGenerator::default())
            .unwrap();
        let nodes = vec![
            Node::new_leaf(0, Some(I(4)), 1.0, "A0".to_string()),
            Node::new_leaf(1, Some(I(4)), 3.0, "B1".to_string()),
            Node::new_leaf(2, Some(I(5)), 2.0, "C2".to_string()),
            Node::new_leaf(3, Some(I(5)), 7.0, "D3".to_string()),
            Node::new_internal(4, Some(I(6)), vec![L(0), L(1)], 1.0, "".to_string()),
            Node::new_internal(5, Some(I(6)), vec![L(3), L(2)], 1.0, "".to_string()),
            Node::new_internal(6, None, vec![I(4), I(5)], 0.0, "".to_string()),
        ];

        assert_eq!(nj_tree.root, I(6));
        assert_eq!(nj_tree.nodes, nodes);
    }

    #[test]
    fn nj_builder_correct_creation() {
        let builder = NJTreeBuilder::new(LDNACorr {});
        assert_matches!(builder.randomise, Strategy::ArgMax);
        let builder = NJTreeBuilder::new(LDNACorr {});
        assert_matches!(builder.randomise, Strategy::ArgMax);
        let builder = NJTreeBuilder::new_with_softmax(LDNACorr {}, 0.0);
        assert_matches!(builder.randomise, Strategy::SoftmaxUniform(t) if t == 0.0);
        let builder = NJTreeBuilder::new_with_softmax(LDNACorr {}, 1.0);
        assert_matches!(builder.randomise, Strategy::SoftmaxUniform(t) if t == 1.0);
        let builder = NJTreeBuilder::new_with_softmax(LDNACorr {}, 0.5);
        assert_matches!(builder.randomise, Strategy::SoftmaxUniform(t) if t == 0.5);
        let builder = NJTreeBuilder::new_with_softmax(LDNACorr {}, 1.5);
        assert_matches!(builder.randomise, Strategy::SoftmaxUniform(t) if t == 1.0);
        let builder = NJTreeBuilder::new_with_softmax(LDNACorr {}, -1.5);
        assert_matches!(builder.randomise, Strategy::SoftmaxUniform(t) if t == 0.0);
    }

    #[test]
    fn lower_triangle_index_conversion() {
        assert_eq!(lower_triangle_index(0), (1, 0));
        assert_eq!(lower_triangle_index(1), (2, 0));
        assert_eq!(lower_triangle_index(2), (2, 1));
        assert_eq!(lower_triangle_index(3), (3, 0));
        assert_eq!(lower_triangle_index(5), (3, 2));
        assert_eq!(lower_triangle_index(6), (4, 0));
        assert_eq!(lower_triangle_index(9), (4, 3));
        assert_eq!(lower_triangle_index(10), (5, 0));
        assert_eq!(lower_triangle_index(14), (5, 4));
        assert_eq!(lower_triangle_index(15), (6, 0));
        assert_eq!(lower_triangle_index(20), (6, 5));
        assert_eq!(lower_triangle_index(21), (7, 0));
        assert_eq!(lower_triangle_index(23), (7, 2));
        assert_eq!(lower_triangle_index(28), (8, 0));
    }

    #[test]
    fn delta_tree_length() {
        let nj_distances = DistanceMatrix {
            idx: (0..5).map(NodeIdx::Leaf).collect(),
            distances: dmatrix![
                0.0, 5.0, 9.0, 9.0, 8.0;
                5.0, 0.0, 10.0, 10.0, 9.0;
                9.0, 10.0, 0.0, 8.0, 7.0;
                9.0, 10.0, 8.0, 0.0, 3.0;
                8.0, 9.0, 7.0, 3.0, 0.0],
        };
        let q = nj_distances.compute_delta_tree_length();
        assert_eq!(
            q,
            dvector![-50.0, -38.0, -38.0, -34.0, -34.0, -40.0, -34.0, -34.0, -40.0, -48.0]
        )
    }

    #[test]
    fn softmax() {
        let softmax =
            NJTreeBuilder::<LDNACorr>::softmax_from_deltas(dvector![-1.3, -5.1, -2.2, -0.7, -1.1]);
        assert_eq!(
            softmax,
            dvector![
                0.020190464732580685,
                0.9025376890165726,
                0.04966052987196013,
                0.011080761983386346,
                0.01653055439550022
            ]
        );
        assert_relative_eq!(softmax.sum(), 1.0, epsilon = 1e-10);

        // Example values from https://medium.com/@hunter-j-phillips/a-simple-introduction-to-softmax-287712d69bac
        let softmax = NJTreeBuilder::<LDNACorr>::softmax_from_deltas(dvector![-5.0, -7.0, -10.0]);
        assert_relative_eq!(softmax, dvector![0.006, 0.047, 0.946], epsilon = 1e-3);
        assert_relative_eq!(softmax.sum(), 1.0, epsilon = 1e-10);

        let softmax = NJTreeBuilder::<LDNACorr>::softmax_from_deltas(dvector![-1.0, -2.0, -3.0]);
        assert_relative_eq!(softmax, dvector![0.0900, 0.2447, 0.6652], epsilon = 1e-4);
        assert_relative_eq!(softmax.sum(), 1.0, epsilon = 1e-10);

        let softmax = NJTreeBuilder::<LDNACorr>::softmax_from_deltas(dvector![-4.0, -5.0, -6.0]);
        assert_relative_eq!(softmax, dvector![0.0900, 0.2447, 0.6652], epsilon = 1e-4);
        assert_relative_eq!(softmax.sum(), 1.0, epsilon = 1e-10);

        // Example values from https://ai.gopubby.com/the-softmax-activation-function-work-with-keras-8f674b4481a5
        let softmax =
            NJTreeBuilder::<LDNACorr>::softmax_from_deltas(dvector![-2.0, -4.3, -1.2, 3.1]);
        assert_relative_eq!(
            softmax,
            dvector![0.087492, 0.872661, 0.039313, 0.000533],
            epsilon = 1e-6
        );
        assert_relative_eq!(softmax.sum(), 1.0, epsilon = 1e-5);
    }

    #[test]
    fn softmax_w_temp_regular() {
        let delta_tree_length = dvector![-1.3, -5.1, -2.2, -0.7, -1.1];
        let softmax_w_temp = NJTreeBuilder::<LDNACorr>::softmax(delta_tree_length.clone(), 1.0);
        let softmax_regular = NJTreeBuilder::<LDNACorr>::softmax_from_deltas(delta_tree_length);
        assert_eq!(softmax_w_temp, softmax_regular);
        assert_eq!(softmax_w_temp.sum(), 1.0);
    }

    #[test]
    fn softmax_w_temp_uniform() {
        let delta_tree_length = dvector![-1.3, -5.1, -2.2, -0.7, -1.1];
        let softmax_vector = NJTreeBuilder::<LDNACorr>::softmax(delta_tree_length, 0.0);
        assert_eq!(softmax_vector, dvector![0.2, 0.2, 0.2, 0.2, 0.2]);
        assert_eq!(softmax_vector.sum(), 1.0);
    }

    #[rstest]
    #[case(0.5)]
    #[case(0.75)]
    #[case(0.35)]
    #[case(0.1)]
    #[case(0.99)]
    fn softmax_w_temp_between(#[case] temp: f64) {
        let delta_tree_length = dvector![-1.3, -5.1, -2.2, -0.7, -1.1];
        let softmax_temp = NJTreeBuilder::<LDNACorr>::softmax(delta_tree_length.clone(), temp);

        let softmax = NJTreeBuilder::<LDNACorr>::softmax(delta_tree_length.clone(), 1.0);
        let softmax_uniform = NJTreeBuilder::<LDNACorr>::softmax(delta_tree_length, 0.0);

        assert_relative_eq!(softmax_temp.sum(), 1.0);
        assert_eq!(
            softmax_temp,
            (softmax_uniform.scale(1.0 - temp) + softmax.scale(temp))
        );
    }

    #[test]
    fn nj_tree_original_paper_fake_softmax() {
        // Compare against the original paper tree
        // https://academic.oup.com/mbe/article/4/4/406/1029664
        let nj_distances = DistanceMatrix {
            idx: (0..8).map(NodeIdx::Leaf).collect(),
            distances: dmatrix![
                0.0, 7.0, 8.0, 11.0, 13.0, 16.0, 13.0, 17.0;
                7.0, 0.0, 5.0, 8.0, 10.0, 13.0, 10.0, 14.0;
                8.0, 5.0, 0.0, 5.0, 7.0, 10.0, 7.0, 11.0;
                11.0, 8.0, 5.0, 0.0, 8.0, 11.0, 8.0, 12.0;
                13.0, 10.0, 7.0, 8.0, 0.0, 5.0, 6.0, 10.0;
                16.0, 13.0, 10.0, 11.0, 5.0, 0.0, 9.0, 13.0;
                13.0, 10.0, 7.0, 8.0, 6.0, 9.0, 0.0, 8.0;
                17.0, 14.0, 11.0, 12.0, 10.0, 13.0, 8.0, 0.0;
            ],
        };
        let sequences =
            Sequences::new_unchecked((1..=8).map(|i| record!(&i.to_string(), b"")).collect());

        // FakeRng will return values that will select the same pairs as in the original paper
        let mut rng = RandomGenerator::from_rng(FakeRng::from_f64_values(vec![
            0.01, 0.01, 0.01, 0.5, 0.4, 0.0, 0.0,
        ]));
        let nj_tree = NJTreeBuilder::new_with_softmax(LDNACorr {}, 1.0)
            .build_from_distances(nj_distances, &sequences, &mut rng)
            .unwrap();
        let correct_tree =
            tree!("((8:6,7:2):0.5,((5:1,6:4):2,(4:3,(3:1,(1:5,2:2):2):1):2):0.5):0.0;");
        assert_eq!(nj_tree.length, correct_tree.length);
        for leaf in nj_tree.leaves() {
            assert_eq!(leaf.blen, correct_tree.by_id(&leaf.id).blen);
        }
        assert_eq!(nj_tree.robinson_foulds(&correct_tree), 0);
    }

    #[test]
    fn nj_builder_fake_softmax() {
        let nj_distances = DistanceMatrix {
            idx: (0..4).map(NodeIdx::Leaf).collect(),
            distances: dmatrix![
                    0.0, 4.0, 5.0, 10.0;
                    4.0, 0.0, 7.0, 12.0;
                    5.0, 7.0, 0.0, 9.0;
                    10.0, 12.0, 9.0, 0.0],
        };
        let sequences = Sequences::new_unchecked(vec![
            record!("A0", b""),
            record!("B1", b""),
            record!("C2", b""),
            record!("D3", b""),
        ]);

        let nj_softmax_tree = NJTreeBuilder::new_with_softmax(LDNACorr {}, 1.0)
            .build_from_distances(
                nj_distances.clone(),
                &sequences,
                &mut RandomGenerator::from_rng(FakeRng::from_f64_values(vec![1.0])),
            )
            .unwrap();

        let tree = NJTreeBuilder::new(LDNACorr {})
            .build_from_distances(
                nj_distances,
                &sequences,
                &mut FakeGenerator::from_rng(FakeRng::from_f64_values(vec![1.0])),
            )
            .unwrap();

        // In this case both end up the same same length
        assert_eq!(nj_softmax_tree.length, tree.length);
        // Different rooting, but rf distance 0
        assert_eq!(nj_softmax_tree.robinson_foulds(&tree), 0);
        assert_eq!(nj_softmax_tree, tree)
    }

    #[test]
    fn nj_builder_softmax() {
        let nj_distances = DistanceMatrix {
            idx: (0..5).map(NodeIdx::Leaf).collect(),
            distances: dmatrix![
                0.0, 5.0, 9.0, 9.0, 8.0;
                5.0, 0.0, 10.0, 10.0, 9.0;
                9.0, 10.0, 0.0, 8.0, 7.0;
                9.0, 10.0, 8.0, 0.0, 3.0;
                8.0, 9.0, 7.0, 3.0, 0.0],
        };
        let sequences = Sequences::new_unchecked(vec![
            record!("A0", b""),
            record!("B1", b""),
            record!("C2", b""),
            record!("D3", b""),
            record!("E4", b""),
        ]);
        let mut rng = FakeGenerator::default();
        let nj_uniform_tree = NJTreeBuilder::new_with_softmax(LDNACorr {}, 0.0)
            .build_from_distances(nj_distances.clone(), &sequences, &mut rng)
            .unwrap();

        let nj_softmax_tree = NJTreeBuilder::new_with_softmax(LDNACorr {}, 1.0)
            .build_from_distances(nj_distances.clone(), &sequences, &mut rng)
            .unwrap();

        let nj_tree = NJTreeBuilder::new(LDNACorr {})
            .build_from_distances(nj_distances, &sequences, &mut rng)
            .unwrap();

        // Since FakeRng always returns 0, both softmax trees should be the same
        assert_eq!(nj_uniform_tree.nodes, nj_softmax_tree.nodes);

        // The softmax/uniform trees should be longer than the original NJ since it does not pick the optimal pair
        // Only checking one because the trees are the same
        assert!(nj_uniform_tree.length > nj_tree.length);
        // Different resulting topologies
        assert!(nj_uniform_tree.robinson_foulds(&nj_tree) > 0);
    }
}
