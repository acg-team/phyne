use bio::io::fasta::Record;
use fixedbitset::FixedBitSet;
use itertools::Itertools;

use crate::alignment::{Alignment, AncestralAlignment, Sequences};
use crate::alphabets::GAP;
use crate::asr::AncestralSequenceReconstruction;
use crate::phylo_info::PhyloInfo;
use crate::pip_model::PIPModel;
use crate::substitution_models::QMatrix;
use crate::tree::{
    NodeIdx::{Internal, Leaf},
    Tree,
};
use crate::{aligned_seq, record};

// Homology reconstruction under PIP: the model's deletion rate `mu` is used to compare a
// deletion on the incoming branch of a wholly deleted clade with deletions on both child branches.
impl<Q: QMatrix> PIPModel<Q> {
    /// Probability of a deletion on the branch above each node, indexed by node index.
    fn deletion_probabilities_at_nodes(&self, tree: &Tree) -> Vec<f64> {
        tree.iter()
            .map(|node| -(-self.mu() * node.blen).exp_m1())
            .collect()
    }

    fn deletion_probabilities_at_children(&self, tree: &Tree) -> Vec<f64> {
        let deletion_probs = self.deletion_probabilities_at_nodes(tree);
        tree.iter()
            .map(|node| {
                node.children
                    .iter()
                    .map(|child| deletion_probs[usize::from(child)])
                    .product1::<f64>()
                    .unwrap_or(0.0)
                    * (1.0 - deletion_probs[usize::from(node.idx)])
            })
            .collect()
    }

    fn delete_at_node_or_children(&self, tree: &Tree) -> FixedBitSet {
        let deletion_at_node: Vec<f64> = self.deletion_probabilities_at_nodes(tree);
        let deletion_at_children = self.deletion_probabilities_at_children(tree);
        let mut delete_at_node_or_children = FixedBitSet::with_capacity(tree.len());
        for node in tree.iter() {
            let idx = usize::from(node.idx);
            delete_at_node_or_children.set(idx, deletion_at_node[idx] > deletion_at_children[idx]);
        }
        delete_at_node_or_children
    }

    /// Returns for every node (indexed by node index) a bitset over alignment columns that marks
    /// where the character is present at the node. Traverses the tree once in each direction for
    /// all sites.
    fn reconstruct_presence(&self, alignment: &impl Alignment, tree: &Tree) -> Vec<FixedBitSet> {
        let len = alignment.len();

        let delete_at_node = self.delete_at_node_or_children(tree);

        // has_data[node]: columns where some leaf below the node has a character
        let has_data = character_maps(alignment, tree, len);

        let mut present = vec![FixedBitSet::with_capacity(len); tree.len()];
        for node_idx in tree.preorder() {
            let idx = usize::from(node_idx);
            // A present leaf without an ancestral homolog is an insertion on its terminal
            // branch; otherwise its character survived until the leaf.
            if let Leaf(_) = node_idx {
                present[idx] = has_data[idx].clone();
                continue;
            }

            let children = tree.children(node_idx);
            let mut both_children = has_data[usize::from(children[0])].clone();
            both_children.intersect_with(&has_data[usize::from(children[1])]);

            let inherited = match tree.node(node_idx).parent {
                Some(parent) => present[usize::from(parent)].clone(),
                None => FixedBitSet::with_capacity(len),
            };

            // Independent of the site: is a deletion at this node more probable than
            // deletions at both children?
            let delete_here = delete_at_node[idx];

            // Inherited columns with data below always survive, those without data survive
            // unless the deletion is placed at this node. Other columns are inserted here
            // iff both children have data.
            let mut row = if delete_here {
                let mut surviving = inherited;
                surviving.intersect_with(&has_data[idx]);
                surviving
            } else {
                inherited
            };
            row.union_with(&both_children);
            present[idx] = row;
        }
        present
    }
}

fn character_maps(alignment: &impl Alignment, tree: &Tree, len: usize) -> Vec<FixedBitSet> {
    let mut has_data = vec![FixedBitSet::with_capacity(len); tree.len()];
    for node_idx in tree.postorder() {
        let idx = usize::from(node_idx);
        match node_idx {
            Leaf(_) => {
                for (site, c) in alignment.leaf_map(node_idx).iter().enumerate() {
                    has_data[idx].set(site, c.is_some());
                }
            }
            Internal(_) => {
                for child in tree.children(node_idx) {
                    let child_data = has_data[usize::from(child)].clone();
                    has_data[idx].union_with(&child_data);
                }
            }
        }
    }
    has_data
}

impl<Q: QMatrix, A: Alignment, AA: AncestralAlignment> AncestralSequenceReconstruction<A, AA>
    for PIPModel<Q>
{
    fn reconstruct_ancestral_seqs_unchecked(&self, phylo: PhyloInfo<A>) -> PhyloInfo<AA> {
        let mut aligned_leaf_records = Vec::new();
        let tree = &phylo.tree;
        let alignment = &phylo.msa;
        for leaf in tree.leaves() {
            let unaligned_record: &Record = alignment.seqs().record_by_id(&leaf.id);
            debug_assert_eq!(unaligned_record.id(), leaf.id);
            aligned_leaf_records.push(record!(
                &leaf.id,
                unaligned_record.desc(),
                &aligned_seq!(alignment.leaf_map(&leaf.idx), unaligned_record.seq())
            ));
        }

        let present = self.reconstruct_presence(alignment, tree);
        let mut ancestral_records = Vec::<Record>::with_capacity(tree.len() - tree.n);
        for node_idx in tree.postorder() {
            if let Internal(_) = node_idx {
                let row = &present[usize::from(node_idx)];
                let seq: Vec<u8> = (0..row.len())
                    .map(|site| if row[site] { b'X' } else { GAP })
                    .collect();
                ancestral_records.push(record!(tree.node_id(node_idx), None, &seq));
            }
        }
        aligned_leaf_records.append(&mut ancestral_records);
        let all_seqs =
            Sequences::with_alphabet_unchecked(aligned_leaf_records, alignment.seqs().alphabet());
        AA::from_aligned_with_ancestral_unchecked(all_seqs, phylo.tree)
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use approx::assert_relative_eq;
    use fixedbitset::FixedBitSet;
    use lazy_static::lazy_static;

    use crate::alignment::{Alignment, AncestralAlignment, Sequences, MASA, MSA};
    use crate::asr::AncestralSequenceReconstruction;
    use crate::phylo_info::PhyloInfo;
    use crate::pip_model::PIPModel;
    use crate::substitution_models::JC69;
    use crate::tree::{NodeIdx::Leaf, Tree};
    use crate::{record_wo_desc as record, tree};

    lazy_static! {
        static ref TREE_DELETION_AT_I2: Tree =
            tree!("(((A:1.0,B:1.0)I2:2.0,C:1.0)I1:1.0,D:1.0)I0:0.0;");
        static ref TREE_DELETION_AT_AB: Tree =
            tree!("(((A:1.0,B:1.0)I2:0.1,C:1.0)I1:1.0,D:1.0)I0:0.0;");
        static ref TREE_DIFF_BLENS: Tree =
            tree!("(((A:0.5,B:2.5)I2:1.2,C:1.4)I1:0.6,D:5.0)I0:0.0;");
    }

    fn pip(mu: f64) -> PIPModel<JC69> {
        PIPModel::<JC69>::with_default_substitution(&[0.5, mu]).unwrap()
    }

    fn reconstruct_on_single_site(tree: Tree, mu: f64) -> PhyloInfo<MASA> {
        let sequences = Sequences::new_unchecked(vec![
            record!("A", b"-"),
            record!("B", b"-"),
            record!("C", b"A"),
            record!("D", b"A"),
        ]);
        let phylo = MSA::from_aligned(sequences, tree).unwrap();
        pip(mu).reconstruct_ancestral_seqs(phylo).unwrap()
    }

    #[rstest]
    #[case(TREE_DELETION_AT_I2.clone(), 1.0, &[0.0, 0.6321205588285577, 0.8646647167633873, 0.6321205588285577, 0.6321205588285577, 0.6321205588285577, 0.6321205588285577])]
    #[case(TREE_DELETION_AT_AB.clone(), 1.0, &[0.0, 0.6321205588285577, 0.09516258196404048, 0.6321205588285577, 0.6321205588285577, 0.6321205588285577, 0.6321205588285577])]
    #[case(TREE_DIFF_BLENS.clone(), 1.5, &[0.0, 0.5934303402594008, 0.8347011117784134, 0.5276334472589853, 0.9764822541439909, 0.877543571747018, 0.9994469156298522])]
    fn deletion_probability_correct(#[case] tree: Tree, #[case] mu: f64, #[case] expected: &[f64]) {
        let pip = pip(mu);
        let deletion_at_node = pip.deletion_probabilities_at_nodes(&tree);

        for node_idx in tree.preorder() {
            assert_relative_eq!(
                deletion_at_node[usize::from(node_idx)],
                expected[usize::from(node_idx)],
            );
        }
    }

    #[rstest]
    #[case(TREE_DELETION_AT_I2.clone(), 1.0, &[0.39957640089372803, 0.20107272845569987, 0.054076785389618985, 0.0, 0.0, 0.0, 0.0])]
    #[case(TREE_DELETION_AT_AB.clone(), 1.0, &[0.39957640089372803, 0.022129502489732, 0.36155167889278234, 0.0, 0.0, 0.0, 0.0])]
    #[case(TREE_DIFF_BLENS.clone(), 1.5, &[0.5931021232134318, 0.2978068256820055, 0.08516606975398619, 0.0, 0.0, 0.0, 0.0])]
    fn deletion_probability_at_children_correct(
        #[case] tree: Tree,
        #[case] mu: f64,
        #[case] expected: &[f64],
    ) {
        let pip = pip(mu);
        let deletion_at_node = pip.deletion_probabilities_at_children(&tree);

        for node_idx in tree.preorder() {
            assert_relative_eq!(
                deletion_at_node[usize::from(node_idx)],
                expected[usize::from(node_idx)],
            );
        }
    }

    #[rstest]
    // It is better to delete at the node than at its children for all nodes except the root
    #[case(TREE_DELETION_AT_I2.clone(), 1.0, FixedBitSet::from_iter([1, 2, 3, 4, 5, 6]))]
    // It is better to delete at the node than at its children for all nodes except I2 and the root
    #[case(TREE_DELETION_AT_AB.clone(), 1.0, FixedBitSet::from_iter([1, 3, 4, 5, 6]))]
    // It is better to delete at the node than at its children for all nodes except the root
    #[case(TREE_DIFF_BLENS.clone(), 1.5, FixedBitSet::from_iter([1, 2, 3, 4, 5, 6]))]
    fn delete_at_node_or_children(
        #[case] tree: Tree,
        #[case] mu: f64,
        #[case] expected: FixedBitSet,
    ) {
        let pip = pip(mu);
        let delete_at_node_or_children = pip.delete_at_node_or_children(&tree);

        assert!(delete_at_node_or_children.ones().eq(expected.ones()));
    }

    #[test]
    fn deletion_on_parent_branch_when_more_probable() {
        let tree = TREE_DELETION_AT_I2.clone();
        let ancestral = reconstruct_on_single_site(tree.clone(), 1.0).msa;

        assert!(ancestral.ancestral_map(&tree.by_id("I1").idx)[0].is_none());
        assert!(ancestral.ancestral_map(&tree.by_id("I2").idx)[0].is_some());
        assert!(ancestral.ancestral_map(&tree.by_id("I3").idx)[0].is_some());
        for id in ["A", "B"] {
            let leaf = tree.by_id(id);
            assert!(ancestral.leaf_map(&leaf.idx)[0].is_none());
        }
    }

    #[test]
    fn deletions_on_child_branches_when_more_probable() {
        let tree = TREE_DELETION_AT_AB.clone();
        let phylo = reconstruct_on_single_site(tree.clone(), 1.0);
        let ancestral = phylo.msa;
        let tree = &phylo.tree;

        assert!(ancestral.ancestral_map(&tree.by_id("I1").idx)[0].is_some());
        assert!(ancestral.ancestral_map(&tree.by_id("I2").idx)[0].is_some());
        assert!(ancestral.ancestral_map(&tree.by_id("I3").idx)[0].is_some());
        for id in ["A", "B"] {
            let leaf = tree.by_id(id);
            assert!(ancestral.leaf_map(&leaf.idx)[0].is_none());
        }
    }

    #[test]
    fn present_leaves_without_ancestors_inserted_on_leaves() {
        let tree = tree!("((A:1.0,B:1.0)I:1.0,C:1.0)R:0.0;");
        let sequences = Sequences::new_unchecked(vec![
            record!("A", b"A"),
            record!("B", b"-"),
            record!("C", b"-"),
        ]);
        let phylo = MSA::from_aligned(sequences, tree.clone()).unwrap();
        let ancestral: MASA = pip(1.0).reconstruct_ancestral_seqs(phylo).unwrap().msa;

        assert!(ancestral.leaf_map(&tree.by_id("A").idx)[0].is_some());
        assert!(ancestral.ancestral_map(&tree.by_id("I").idx)[0].is_none());
        assert!(ancestral.ancestral_map(&tree.by_id("R").idx)[0].is_none());
    }

    #[test]
    fn single_leaf_tree_reconstructs_presence() {
        let tree = tree!("A:0.0;");
        let sequences = Sequences::new_unchecked(vec![record!("A", b"A")]);
        let phylo = MSA::from_aligned(sequences, tree.clone()).unwrap();
        let ancestral: MASA = pip(1.0).reconstruct_ancestral_seqs(phylo).unwrap().msa;

        assert!(ancestral.leaf_map(&tree.root)[0].is_some());
        assert!(matches!(tree.root, Leaf(_)));
    }
}
