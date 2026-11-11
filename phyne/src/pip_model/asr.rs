use std::marker::PhantomData;

use bio::io::fasta::Record;
use fixedbitset::FixedBitSet;
use nalgebra::DVector;

use crate::alignment::{mapping_to_bits, Alignment, AncestralAlignment, Sequences};
use crate::alphabets::{AMB_CHAR, GAP};
use crate::asr::AncestralSequenceReconstruction;
use crate::phylo_info::PhyloInfo;
use crate::pip_model::PIPModel;
use crate::record;
use crate::substitution_models::QMatrix;
use crate::tree::NodeIdx::{self, Internal, Leaf};

pub struct PIPIndels<Q, A, AA>
where
    Q: QMatrix,
    A: Alignment,
    AA: AncestralAlignment,
{
    model: PIPModel<Q>,
    phantom: PhantomData<(A, AA)>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PIPAsrInfo {
    /// Log deletion probabilities (delta(v)) for each node, indexed by node (single value per branch).
    deletion_probs: Vec<f64>,
    /// Log pure survival probabilities (zeta(v)) for each node, indexed by node (single value per branch), computed as ln(1 - exp(-mu * b(v))).
    pure_survival_probs: Vec<f64>,
    /// Log insertion and survival along branch probabilities (iota(v) * beta(v)), indexed by node (single value per branch).
    insert_and_survive_probs: Vec<f64>,
    /// Log deletion/survival probabilities for each node, indexed by node and column.
    f: Vec<DVector<f64>>,
    /// Log insertion probabilities for each node, indexed by node and column.
    p: Vec<DVector<f64>>,
    /// Set G: the set of nodes where the character is extinct in the subtree
    extinct_subtree: Vec<FixedBitSet>,
    /// Set A: the set of nodes where the character could be an insertion
    ancestor_subtree: Vec<FixedBitSet>,
    /// Deletion locations for each character in the alignment, indexed by node and column.
    deletion_locations: Vec<FixedBitSet>,
    /// Insertion location for each character in the alignment, indexed by column (only one insertion is allowed per column).
    insertion_location: Vec<usize>,
    /// Presence for each character in the alignment, indexed by node and column.
    presence: Vec<FixedBitSet>,
}

/// Log probability of a deletion on the branch above each node, indexed by node index.
/// ln(delta(v)) = ln(1 - exp(-mu * b(v)))
fn deletion_probabilities<Q: QMatrix>(
    pip: &PIPModel<Q>,
    info: &PhyloInfo<impl Alignment>,
) -> Vec<f64> {
    info.tree
        .iter()
        .map(|node| (-(-pip.mu() * node.blen).exp_m1()).ln())
        .collect()
}

/// Log probability of pure survival on the branch above each node, indexed by node index.
/// ln(zeta(v)) = ln(exp(-mu * b(v))) = -mu * b(v)
fn pure_survival_probabilities<Q: QMatrix>(
    pip: &PIPModel<Q>,
    info: &PhyloInfo<impl Alignment>,
) -> Vec<f64> {
    info.tree.iter().map(|node| -pip.mu() * node.blen).collect()
}

/// Combined log probability of insertion and survival on the branch above each node, indexed by node index.
/// ln(iota(v)) + ln(beta(v)))
fn insert_and_survive_probabilities<Q: QMatrix>(
    pip: &PIPModel<Q>,
    info: &PhyloInfo<impl Alignment>,
) -> Vec<f64> {
    let common_factor = 1.0 / (info.tree.length + 1.0 / pip.mu());
    info.tree
        .iter()
        .map(|node| {
            let numerator = if node.idx == info.tree.root {
                1.0 / pip.mu()
            } else {
                -(-pip.mu() * node.blen).exp_m1() / pip.mu()
            };
            numerator.ln() + common_factor.ln()
        })
        .collect::<Vec<f64>>()
}

impl PIPAsrInfo {
    pub fn new<A: Alignment>(model: &PIPModel<impl QMatrix>, info: &PhyloInfo<A>) -> Self {
        let node_count = info.tree.len();
        let msa_length = info.msa.len();
        PIPAsrInfo {
            insert_and_survive_probs: insert_and_survive_probabilities(model, info),
            deletion_probs: deletion_probabilities(model, info),
            pure_survival_probs: pure_survival_probabilities(model, info),
            f: vec![DVector::<f64>::zeros(msa_length); node_count],
            p: vec![DVector::<f64>::zeros(msa_length); node_count],
            extinct_subtree: vec![FixedBitSet::with_capacity(msa_length); node_count],
            ancestor_subtree: vec![FixedBitSet::with_capacity(msa_length); node_count],
            deletion_locations: vec![FixedBitSet::with_capacity(node_count); msa_length],
            insertion_location: vec![usize::from(info.tree.root); msa_length],
            presence: vec![FixedBitSet::with_capacity(msa_length); node_count],
        }
    }
}

impl<Q: QMatrix, A: Alignment, AA: AncestralAlignment> AncestralSequenceReconstruction<A, AA>
    for PIPIndels<Q, A, AA>
{
    fn reconstruct_ancestral_seqs_unchecked(&self, phylo: PhyloInfo<A>) -> PhyloInfo<AA> {
        let present = self.reconstruct_presence(&phylo);

        let mut ancestral_records = Vec::<Record>::with_capacity(phylo.tree.len() - phylo.tree.n);

        for node_idx in phylo.tree.postorder() {
            if let Internal(idx) = node_idx {
                let mut seq = vec![GAP; phylo.msa.len()];
                for site in present[*idx].ones() {
                    seq[site] = AMB_CHAR;
                }
                ancestral_records.push(record!(phylo.tree.node_id(node_idx), None, &seq));
            }
        }

        let ancestral_seqs =
            Sequences::with_alphabet_unchecked(ancestral_records, phylo.msa.alphabet());

        AA::from_msa_with_ancestral_unchecked(phylo, ancestral_seqs)
    }
}

impl<Q: QMatrix, A: Alignment, AA: AncestralAlignment> PIPIndels<Q, A, AA> {
    pub fn new(model: PIPModel<Q>) -> Self {
        Self {
            model,
            phantom: PhantomData,
        }
    }

    /// Returns for every node (indexed by node index) a bitset over alignment columns that marks
    /// where the character is present at the node. Traverses the tree once in each direction for
    /// all sites.
    fn reconstruct_presence(&self, phylo: &PhyloInfo<A>) -> Vec<FixedBitSet> {
        let mut tmp = PIPAsrInfo::new(&self.model, phylo);
        // bottom up pass to fill in deletion and insertion sets
        for node_idx in phylo.tree.postorder() {
            match node_idx {
                Internal(_) => {
                    self.set_internal_values(phylo, node_idx, &mut tmp);
                }
                Leaf(_) => {
                    self.set_leaf_values(phylo, node_idx, &mut tmp);
                }
            };
        }

        // top down pass to determine character presence at each node
        for node_idx in phylo.tree.preorder() {
            match node_idx {
                Internal(_) => {
                    self.set_internal_presence(phylo, node_idx, &mut tmp);
                }
                Leaf(idx) => {
                    // Leaf node character presence is defined by the observations
                    tmp.presence[*idx] = tmp.ancestor_subtree[*idx].clone();
                }
            };
        }

        tmp.presence
    }

    fn set_leaf_values(&self, phylo: &PhyloInfo<A>, node_idx: &NodeIdx, tmp: &mut PIPAsrInfo) {
        let idx = usize::from(node_idx);
        let has_character = mapping_to_bits(phylo.msa.leaf_map(node_idx));

        tmp.extinct_subtree[idx] = has_character.clone();
        tmp.extinct_subtree[idx].toggle_range(..);

        tmp.ancestor_subtree[idx] = has_character;

        tmp.f[idx].fill(tmp.pure_survival_probs[idx]);
        tmp.p[idx].fill(tmp.insert_and_survive_probs[idx]);

        for col in tmp.extinct_subtree[idx].ones() {
            // extinct at leaf means potential deletion set
            tmp.p[idx][col] = f64::NEG_INFINITY;
            tmp.f[idx][col] = tmp.deletion_probs[idx];
            tmp.deletion_locations[col].set(idx, true);
        }

        for col in tmp.ancestor_subtree[idx].ones() {
            // present at leaf means potential insertion set (A)
            tmp.insertion_location[col] = idx;
        }
    }

    fn set_internal_values(&self, phylo: &PhyloInfo<A>, node_idx: &NodeIdx, tmp: &mut PIPAsrInfo) {
        let idx = usize::from(node_idx);
        let children: Vec<usize> = phylo
            .tree
            .children(node_idx)
            .iter()
            .map(usize::from)
            .collect();

        // Precompute potential values for f and p. They might be overwritten by deletion and insertion processing
        tmp.f[idx] = tmp.f[children[0]].clone() + tmp.f[children[1]].clone();
        tmp.f[idx].add_scalar_mut(tmp.pure_survival_probs[idx]);

        tmp.p[idx] = tmp.f[children[0]].clone() + tmp.f[children[1]].clone();
        tmp.p[idx].add_scalar_mut(tmp.insert_and_survive_probs[idx]);

        self.deletion_at_node_or_children(idx, &children, tmp);
        self.potential_insertion_at_node(idx, &children, tmp);
    }

    fn deletion_at_node_or_children(&self, idx: usize, children: &[usize], tmp: &mut PIPAsrInfo) {
        let mut both_children_extinct = tmp.extinct_subtree[children[0]].clone();
        both_children_extinct.intersect_with(&tmp.extinct_subtree[children[1]]);
        tmp.extinct_subtree[idx] = both_children_extinct;

        for col in tmp.extinct_subtree[idx].ones() {
            // this subtree is extinct, so no insertion can occur here
            tmp.p[idx][col] = f64::NEG_INFINITY;
            if tmp.f[idx][col] < tmp.deletion_probs[idx] {
                // deletion at node is more probable than deletion in children
                tmp.f[idx][col] = tmp.deletion_probs[idx];
                tmp.deletion_locations[col].set(idx, true);
                tmp.deletion_locations[col].set(children[0], false);
                tmp.deletion_locations[col].set(children[1], false);
            }
        }
    }

    fn potential_insertion_at_node(&self, idx: usize, children: &[usize], tmp: &mut PIPAsrInfo) {
        let ch1 = tmp.ancestor_subtree[children[0]].clone();
        let ch2 = tmp.ancestor_subtree[children[1]].clone();

        tmp.ancestor_subtree[idx] = ch1.clone();
        tmp.ancestor_subtree[idx].union_with(&ch2);

        for col in tmp.ancestor_subtree[idx].ones() {
            if ch1[col] && ch2[col] {
                // both children potentially in set A, therefore neither can be an insertion
                tmp.insertion_location[col] = idx;
            } else if ch1[col] {
                // only child 1 is in set A (potential insertion locations)
                if tmp.p[children[0]][col] >= tmp.p[idx][col] {
                    // insertion in child 1 is more likely
                    tmp.p[idx][col] = tmp.p[children[0]][col];
                } else {
                    tmp.insertion_location[col] = idx;
                }
            } else if ch2[col] {
                // only child 2 is in set A (potential insertion locations)
                if tmp.p[children[1]][col] >= tmp.p[idx][col] {
                    // insertion in child 2 is more likely
                    tmp.p[idx][col] = tmp.p[children[1]][col];
                } else {
                    tmp.insertion_location[col] = idx;
                }
            }
        }
    }

    fn set_internal_presence(
        &self,
        phylo: &PhyloInfo<A>,
        node_idx: &NodeIdx,
        tmp: &mut PIPAsrInfo,
    ) {
        let idx = usize::from(node_idx);

        // mark columns as present if they are inherited from the parent
        if let Some(parent_idx) = phylo.tree.node(node_idx).parent {
            let present_in_parent = tmp.presence[usize::from(parent_idx)].clone();
            tmp.presence[idx].union_with(&present_in_parent);
        }

        // mark columns as present if the insertion occurred at this node
        for col in tmp.ancestor_subtree[idx].ones() {
            if tmp.insertion_location[col] == idx {
                tmp.presence[idx].set(col, true);
            }
        }

        // mark columns as absent if the deletion occurred at this node
        for col in tmp.extinct_subtree[idx].ones() {
            if tmp.deletion_locations[col][idx] {
                tmp.presence[idx].set(col, false);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use approx::assert_relative_eq;
    use fixedbitset::FixedBitSet;
    use lazy_static::lazy_static;
    use nalgebra::dvector;

    use crate::alignment::{Alignment, AncestralAlignment, Sequences, MASA, MSA};
    use crate::asr::AncestralSequenceReconstruction;
    use crate::phylo_info::PhyloInfo;
    use crate::pip_model::{
        asr::{deletion_probabilities, insert_and_survive_probabilities, PIPAsrInfo},
        PIPIndels, PIPModel,
    };
    use crate::substitution_models::JC69;
    use crate::tree::{NodeIdx::*, Tree};
    use crate::{record_wo_desc as record, tree};

    lazy_static! {
        static ref TREE_DELETION_AT_I2: Tree =
            tree!("(((A:1.0,B:1.0)I2:2.0,C:1.0)I1:1.0,D:1.0)I0:0.0;");
        static ref TREE_DELETION_AT_AB: Tree =
            tree!("(((A:1.0,B:1.0)I2:0.1,C:1.0)I1:1.0,D:1.0)I0:0.0;");
        static ref TREE_DIFF_BLENS: Tree =
            tree!("(((A:0.5,B:2.5)I2:1.2,C:1.4)I1:0.6,D:5.0)I0:0.0;");
        static ref TREE_BALANCED: Tree = tree!("((A:1.0,B:1.0)I2:1.0,(C:1.0,D:1.0)I1:1.0)I0:0.0;");
    }

    fn pip(mu: f64) -> PIPModel<JC69> {
        PIPModel::<JC69>::with_default_substitution(&[0.5, mu]).unwrap()
    }

    fn setup_test_info(tree: Tree) -> PhyloInfo<MSA> {
        let sequences = Sequences::new_unchecked(vec![
            record!("A", b"-"),
            record!("B", b"-"),
            record!("C", b"A"),
            record!("D", b"A"),
        ]);
        MSA::from_aligned_unchecked(sequences, tree)
    }

    fn reconstruct_on_single_site(tree: Tree, mu: f64) -> PhyloInfo<MASA> {
        let phylo = setup_test_info(tree);
        let asr = PIPIndels::<_, MSA, MASA>::new(pip(mu));

        asr.reconstruct_ancestral_seqs(phylo).unwrap()
    }

    #[rstest]
    #[case(TREE_DELETION_AT_I2.clone(), 1.0, &[0.0, 0.6321205588285577, 0.8646647167633873, 0.6321205588285577, 0.6321205588285577, 0.6321205588285577, 0.6321205588285577])]
    #[case(TREE_DELETION_AT_AB.clone(), 1.0, &[0.0, 0.6321205588285577, 0.09516258196404048, 0.6321205588285577, 0.6321205588285577, 0.6321205588285577, 0.6321205588285577])]
    #[case(TREE_DIFF_BLENS.clone(), 1.5, &[0.0, 0.5934303402594008, 0.8347011117784134, 0.5276334472589853, 0.9764822541439909, 0.877543571747018, 0.9994469156298522])]
    fn deletion_probability_correct(#[case] tree: Tree, #[case] mu: f64, #[case] expected: &[f64]) {
        let pip = pip(mu);
        let phylo = setup_test_info(tree);
        let deletion_at_node = deletion_probabilities(&pip, &phylo);
        let expected = expected.iter().map(|x| x.ln()).collect::<Vec<_>>();

        assert_relative_eq!(
            deletion_at_node.as_slice(),
            expected.as_slice(),
            epsilon = 1e-10
        );
    }

    #[test]
    fn deletion_on_parent_branch_when_more_probable() {
        let tree = TREE_DELETION_AT_I2.clone();
        let ancestral = reconstruct_on_single_site(tree.clone(), 1.0).msa;

        assert!(ancestral.ancestral_map(&tree.by_id("I2").idx)[0].is_none());
        assert!(ancestral.ancestral_map(&tree.by_id("I1").idx)[0].is_some());
        assert!(ancestral.ancestral_map(&tree.by_id("I0").idx)[0].is_some());
        for id in ["A", "B"] {
            let leaf = tree.by_id(id);
            assert!(ancestral.leaf_map(&leaf.idx)[0].is_none());
        }
    }

    #[rstest]
    #[case(TREE_DELETION_AT_I2.clone())]
    #[case(TREE_DELETION_AT_AB.clone())]
    #[case(TREE_DIFF_BLENS.clone())]
    fn insertion_location_at_root_single_site(#[case] tree: Tree) {
        let sequences = Sequences::new_unchecked(vec![
            record!("A", b"-"),
            record!("B", b"-"),
            record!("C", b"A"),
            record!("D", b"A"),
        ]);
        let phylo = MSA::from_aligned(sequences, tree.clone()).unwrap();

        let asr = PIPIndels::<_, MSA, MASA>::new(pip(1.0));
        let mut tmp = PIPAsrInfo::new(&asr.model, &phylo);

        for node_idx in tree.postorder() {
            match node_idx {
                Internal(_) => asr.set_internal_values(&phylo, node_idx, &mut tmp),
                Leaf(_) => asr.set_leaf_values(&phylo, node_idx, &mut tmp),
            }
        }
        assert_eq!(tmp.insertion_location, vec![0]);
    }

    #[test]
    fn deletions_on_child_branches_when_more_probable() {
        let tree = TREE_DELETION_AT_AB.clone();
        let phylo = reconstruct_on_single_site(tree.clone(), 1.0);
        let ancestral = phylo.msa;
        let tree = &phylo.tree;

        assert!(ancestral.ancestral_map(&tree.by_id("I2").idx)[0].is_some());
        assert!(ancestral.ancestral_map(&tree.by_id("I1").idx)[0].is_some());
        assert!(ancestral.ancestral_map(&tree.by_id("I0").idx)[0].is_some());
        for id in ["A", "B"] {
            let leaf = tree.by_id(id);
            assert!(ancestral.leaf_map(&leaf.idx)[0].is_none());
        }
    }

    #[rstest]
    #[case(TREE_DELETION_AT_I2.clone(), 1.0,  &[0.125, 0.07901506985356971, 0.10808308959542341, 0.07901506985356971,
                                                0.07901506985356971, 0.07901506985356971, 0.07901506985356971])]
    #[case(TREE_DELETION_AT_AB.clone(), 1.0, &[0.1639344262295082, 0.10362632111943569, 0.015600423272793523, 0.10362632111943569,
                                               0.10362632111943569, 0.10362632111943569, 0.10362632111943569])]
    #[case(TREE_DIFF_BLENS.clone(), 1.5, &[0.05617977528089888, 0.0333387831606405, 0.04689332088642773, 0.029642328497695807,
                                           0.05485855360359499, 0.049300200659944844, 0.056148703125272606])]
    fn insert_and_survive_probs(#[case] tree: Tree, #[case] mu: f64, #[case] expected: &[f64]) {
        let pip = pip(mu);
        let phylo = setup_test_info(tree);
        let insert_and_survive_probabilities = insert_and_survive_probabilities(&pip, &phylo);
        let expected = expected.iter().map(|x| x.ln()).collect::<Vec<_>>();

        assert_relative_eq!(
            insert_and_survive_probabilities.as_slice(),
            expected.as_slice(),
            epsilon = 1e-10
        );
    }

    #[test]
    fn insertion_at_i5() {
        let tree =
            tree!("(((L3:1.0,L4:1.0)I2:1.0,(L6:0.001,L7:1.0)I5:10000.0)I1:1.0,L8:1.0)I0:0.0;");
        let sequences = Sequences::new_unchecked(vec![
            record!("L3", b"-"),
            record!("L4", b"-"),
            record!("L6", b"X"),
            record!("L7", b"-"),
            record!("L8", b"-"),
        ]);

        let asr = PIPIndels::<_, MSA, MASA>::new(pip(1.0));

        let phylo = asr
            .reconstruct_ancestral_seqs(MSA::from_aligned(sequences, tree).unwrap())
            .unwrap();
        let ancestral = phylo.msa;
        let tree = &phylo.tree;

        // not present at I0 (root)
        assert!(ancestral.ancestral_map(&tree.by_id("I0").idx)[0].is_none());
        // not present at I1 (not inserted yet)
        assert!(ancestral.ancestral_map(&tree.by_id("I1").idx)[0].is_none());
        // not present at I2 (extinct subtree)
        assert!(ancestral.ancestral_map(&tree.by_id("I2").idx)[0].is_none());
        // present at I5 (inserted)
        assert!(ancestral.ancestral_map(&tree.by_id("I5").idx)[0].is_some());

        for id in ["L3", "L4", "L7", "L8"] {
            let leaf = tree.by_id(id);
            assert!(ancestral.leaf_map(&leaf.idx)[0].is_none());
        }
    }

    #[test]
    fn insertion_higher_than_leaf() {
        let tree = tree!("((L2:0.001,L3:1.0)I1:10000.0,L4:5.0)I0:0.0;");
        let sequences = Sequences::new_unchecked(vec![
            record!("L2", b"A"),
            record!("L3", b"-"),
            record!("L4", b"-"),
        ]);

        let asr = PIPIndels::<_, MSA, MASA>::new(pip(1.0));

        let phylo = asr
            .reconstruct_ancestral_seqs(MSA::from_aligned(sequences, tree).unwrap())
            .unwrap();
        let ancestral = phylo.msa;
        let tree = &phylo.tree;

        // not present at I0 (root)
        assert!(ancestral.ancestral_map(&tree.by_id("I0").idx)[0].is_none());
        // present at I1 (inserted)
        assert!(ancestral.ancestral_map(&tree.by_id("I1").idx)[0].is_some());
        // present at L2 (inherited, observed)
        assert!(ancestral.leaf_map(&tree.by_id("L2").idx)[0].is_some());

        // not present at L3, L4 (deleted, observed)
        for id in ["L3", "L4"] {
            let leaf = tree.by_id(id);
            assert!(ancestral.leaf_map(&leaf.idx)[0].is_none());
        }
    }

    #[test]
    fn insertion_at_i2() {
        let tree = tree!("(((L3:0.001,L4:1.0)I2:1000,L5:0.10)I1:1.0,L6:5.0)I0:0.0;");
        let sequences = Sequences::new_unchecked(vec![
            record!("L3", b"A"),
            record!("L4", b"-"),
            record!("L5", b"-"),
            record!("L6", b"-"),
        ]);

        let asr = PIPIndels::<_, MSA, MASA>::new(pip(1.0));

        let phylo = asr
            .reconstruct_ancestral_seqs(MSA::from_aligned(sequences, tree).unwrap())
            .unwrap();
        let ancestral = phylo.msa;
        let tree = &phylo.tree;

        // not present at I0 (root)
        assert!(ancestral.ancestral_map(&tree.by_id("I0").idx)[0].is_none());
        // not present at I1
        assert!(ancestral.ancestral_map(&tree.by_id("I1").idx)[0].is_none());
        // present at I2 (inserted)
        assert!(ancestral.ancestral_map(&tree.by_id("I2").idx)[0].is_some());
        // present at L3 (inherited, observed)
        assert!(ancestral.leaf_map(&tree.by_id("L3").idx)[0].is_some());

        // not present at L4, L5, L6 (deleted, observed)
        for id in ["L4", "L5", "L6"] {
            let leaf = tree.by_id(id);
            assert!(ancestral.leaf_map(&leaf.idx)[0].is_none());
        }
    }

    #[test]
    fn insertion_at_i1() {
        let tree = tree!("(((L3:0.001,L4:1.0)I2:0.001,L5:0.10)I1:10000.0,L6:4.998)I0:0.0;");
        let sequences = Sequences::new_unchecked(vec![
            record!("L3", b"A"),
            record!("L4", b"-"),
            record!("L5", b"-"),
            record!("L6", b"-"),
        ]);

        let asr = PIPIndels::<_, MSA, MASA>::new(pip(1.0));

        let phylo = asr
            .reconstruct_ancestral_seqs(MSA::from_aligned(sequences, tree).unwrap())
            .unwrap();
        let ancestral = phylo.msa;
        let tree = &phylo.tree;

        // not present at I0 (root)
        assert!(ancestral.ancestral_map(&tree.by_id("I0").idx)[0].is_none());
        // inserted at I1
        assert!(ancestral.ancestral_map(&tree.by_id("I1").idx)[0].is_some());
        // present at I2 (inherited)
        assert!(ancestral.ancestral_map(&tree.by_id("I2").idx)[0].is_some());
        // present at L3 (inherited, observed)
        assert!(ancestral.leaf_map(&tree.by_id("L3").idx)[0].is_some());

        // not present at L4, L5, L6 (deleted, observed)
        for id in ["L4", "L5", "L6"] {
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
        let asr = PIPIndels::<_, MSA, MASA>::new(pip(1.0));
        let ancestral: MASA = asr.reconstruct_ancestral_seqs(phylo).unwrap().msa;

        assert!(ancestral.leaf_map(&tree.by_id("A").idx)[0].is_some());
        assert!(ancestral.ancestral_map(&tree.by_id("I").idx)[0].is_none());
        assert!(ancestral.ancestral_map(&tree.by_id("R").idx)[0].is_none());
    }

    #[test]
    fn single_leaf_tree_reconstructs_presence() {
        let tree = tree!("A:0.0;");
        let sequences = Sequences::new_unchecked(vec![record!("A", b"A")]);
        let phylo = MSA::from_aligned(sequences, tree.clone()).unwrap();
        let asr = PIPIndels::<_, MSA, MASA>::new(pip(1.0));
        let ancestral: MASA = asr.reconstruct_ancestral_seqs(phylo).unwrap().msa;

        assert!(ancestral.leaf_map(&tree.root)[0].is_some());
        assert!(matches!(tree.root, Leaf(_)));
    }

    fn setup_test_info_for_tmp_values() -> PhyloInfo<MSA> {
        let tree = tree!("((A2:1,B3:1)I1:1,(C5:1,D6:1)I4:1)I0:0;");
        let sequences = Sequences::new_unchecked(vec![
            record!("A2", b"XXX--X"),
            record!("B3", b"X-X--X"),
            record!("C5", b"X---X-"),
            record!("D6", b"--XXX-"),
        ]);
        MSA::from_aligned(sequences, tree.clone()).unwrap()
    }

    #[test]
    fn tmp_start_values() {
        let phylo = setup_test_info_for_tmp_values();

        let asr = PIPIndels::<_, MSA, MASA>::new(pip(1.0));
        let tmp = PIPAsrInfo::new(&asr.model, &phylo);

        let delta = (-(-1.0_f64).exp_m1()).ln();
        let beta = delta - (phylo.tree.length + 1.0).ln();

        let expected_deletion_probs =
            vec![f64::NEG_INFINITY, delta, delta, delta, delta, delta, delta];
        let expected_ins_survive_probs = [beta - delta, beta, beta, beta, beta, beta, beta];

        assert_eq!(tmp.deletion_probs, expected_deletion_probs);
        assert_relative_eq!(
            tmp.insert_and_survive_probs.as_slice(),
            expected_ins_survive_probs.as_slice()
        );
    }

    #[test]
    fn intermediate_values_at_leaves() {
        let phylo = setup_test_info_for_tmp_values();

        let asr = PIPIndels::<_, MSA, MASA>::new(pip(1.0));
        let mut tmp = PIPAsrInfo::new(&asr.model, &phylo);

        for node in phylo.tree.leaves() {
            asr.set_leaf_values(&phylo, &node.idx, &mut tmp);
        }

        let expected_extinct = [
            FixedBitSet::new(),
            FixedBitSet::new(),
            FixedBitSet::from_iter([3, 4]),
            FixedBitSet::from_iter([1, 3, 4]),
            FixedBitSet::new(),
            FixedBitSet::from_iter([1, 2, 3, 5]),
            FixedBitSet::from_iter([0, 1, 5]),
        ];

        let expected_ancestor = [
            FixedBitSet::new(),
            FixedBitSet::new(),
            FixedBitSet::from_iter([0, 1, 2, 5]),
            FixedBitSet::from_iter([0, 2, 5]),
            FixedBitSet::new(),
            FixedBitSet::from_iter([0, 4]),
            FixedBitSet::from_iter([2, 3, 4]),
        ];

        let delta = (-(-1.0_f64).exp_m1()).ln();
        let zeta = -1.0;
        let iota_beta = delta - (phylo.tree.length + 1.0).ln();
        let neg_inf = f64::NEG_INFINITY;
        let expected_f = [
            dvector![],
            dvector![],
            dvector![zeta, zeta, zeta, delta, delta, zeta],
            dvector![zeta, delta, zeta, delta, delta, zeta],
            dvector![],
            dvector![zeta, delta, delta, delta, zeta, delta],
            dvector![delta, delta, zeta, zeta, zeta, delta],
        ];

        let expected_p = [
            dvector![],
            dvector![],
            dvector![iota_beta, iota_beta, iota_beta, neg_inf, neg_inf, iota_beta],
            dvector![iota_beta, neg_inf, iota_beta, neg_inf, neg_inf, iota_beta],
            dvector![],
            dvector![iota_beta, neg_inf, neg_inf, neg_inf, iota_beta, neg_inf],
            dvector![neg_inf, neg_inf, iota_beta, iota_beta, iota_beta, neg_inf],
        ];

        for n in phylo.tree.leaves() {
            let idx = usize::from(n.idx);
            assert_eq!(
                tmp.extinct_subtree[idx].difference_count(&expected_extinct[idx]),
                0,
            );
            assert_eq!(
                tmp.ancestor_subtree[idx].difference_count(&expected_ancestor[idx]),
                0,
            );
            assert_relative_eq!(tmp.f[idx], expected_f[idx]);
            assert_relative_eq!(tmp.p[idx], expected_p[idx]);
        }
    }

    #[test]
    fn intermediate_values_at_internals() {
        let phylo = setup_test_info_for_tmp_values();

        let asr = PIPIndels::<_, MSA, MASA>::new(pip(1.0));
        let mut tmp = PIPAsrInfo::new(&asr.model, &phylo);

        for node_idx in phylo.tree.postorder() {
            match node_idx {
                Internal(_) => asr.set_internal_values(&phylo, node_idx, &mut tmp),
                Leaf(_) => asr.set_leaf_values(&phylo, node_idx, &mut tmp),
            }
        }

        let expected_extinct = [
            FixedBitSet::new(),
            FixedBitSet::from_iter([1, 3, 4]),
            FixedBitSet::new(),
            FixedBitSet::new(),
            FixedBitSet::from_iter([1, 5]),
            FixedBitSet::new(),
            FixedBitSet::new(),
        ];

        let expected_ancestor = [
            FixedBitSet::from_iter([0, 1, 2, 3, 4, 5]),
            FixedBitSet::from_iter([0, 1, 2, 5]),
            FixedBitSet::new(),
            FixedBitSet::new(),
            FixedBitSet::from_iter([0, 2, 3, 4]),
            FixedBitSet::new(),
            FixedBitSet::new(),
        ];

        let delta = (-(-1.0_f64).exp_m1()).ln();
        let zeta = -1.0;

        let f_i1 = dvector![
            3.0 * zeta,
            2.0 * zeta + delta,
            3.0 * zeta,
            delta,
            delta,
            3.0 * zeta
        ];
        let f_i4 = dvector![
            2.0 * zeta + delta,
            delta,
            2.0 * zeta + delta,
            2.0 * zeta + delta,
            3.0 * zeta,
            delta
        ];

        let f_i0 = f_i1.clone() + f_i4.clone();

        let expected_f = [
            f_i0,
            f_i1.clone(),
            dvector![],
            dvector![],
            f_i4.clone(),
            dvector![],
            dvector![],
        ];

        let iota_beta = delta - (phylo.tree.length + 1.0).ln();

        let neg_inf = f64::NEG_INFINITY;
        let p_i1 = dvector![
            iota_beta + 2.0 * zeta,
            iota_beta,
            iota_beta + 2.0 * zeta,
            neg_inf,
            neg_inf,
            iota_beta + 2.0 * zeta
        ];
        let p_i4 = dvector![
            iota_beta,
            neg_inf,
            iota_beta,
            iota_beta,
            iota_beta + 2.0 * zeta,
            neg_inf
        ];

        let mut p_i0 = f_i1 + f_i4;
        p_i0.add_scalar_mut(-(phylo.tree.length + 1.0).ln());
        p_i0[1] = p_i1[1];
        p_i0[3] = p_i4[3];
        p_i0[4] = p_i4[4];
        p_i0[5] = p_i1[5];

        let expected_p = [
            p_i0,
            p_i1,
            dvector![],
            dvector![],
            p_i4,
            dvector![],
            dvector![],
        ];

        assert_eq!(tmp.insertion_location, vec![0, 2, 0, 6, 4, 1]);

        for n in phylo.tree.internals() {
            let idx = usize::from(n.idx);

            assert_eq!(
                tmp.extinct_subtree[idx].difference_count(&expected_extinct[idx]),
                0,
            );
            assert_eq!(
                tmp.ancestor_subtree[idx].difference_count(&expected_ancestor[idx]),
                0,
            );

            assert_relative_eq!(tmp.f[idx], expected_f[idx]);
            assert_relative_eq!(tmp.p[idx], expected_p[idx]);
        }
    }

    #[test]
    fn final_reconstruction() {
        let phylo = setup_test_info_for_tmp_values();
        let ancestral_seqs = Sequences::new_unchecked(vec![
            record!("I0", b"X-X---"),
            record!("I1", b"X-X--X"),
            record!("I4", b"X-X-X-"),
        ]);

        let asr = PIPIndels::<_, MSA, MASA>::new(pip(1.0));
        let reconstruction = asr.reconstruct_ancestral_seqs_unchecked(phylo.clone());

        let expected = MASA::from_msa_with_ancestral_unchecked(phylo.clone(), ancestral_seqs);

        for node in phylo.tree.internals() {
            let map = reconstruction.msa.ancestral_map(&node.idx);
            assert_eq!(map, expected.msa.ancestral_map(&node.idx));
        }
    }

    #[test]
    fn tiny_probabilities() {
        let tree = tree!("((A:40,B:40)I:0.001,C:1)R:0;");
        let seqs = Sequences::new_unchecked(vec![
            record!("A", b"X"),
            record!("B", b"X"),
            record!("C", b"-"),
        ]);

        let phylo = MSA::from_aligned(seqs, tree.clone()).unwrap();
        let asr = PIPIndels::<_, MSA, MASA>::new(pip(1.0));
        let reconstruction = asr.reconstruct_ancestral_seqs_unchecked(phylo);

        // The probability of insertion at I is 632 times lower than insertion at R, thus the
        // character must be inserted in R and deleted in C
        assert!(reconstruction.msa.ancestral_map(&tree.by_id("I").idx)[0].is_some());
        assert!(reconstruction.msa.ancestral_map(&tree.by_id("R").idx)[0].is_some());
    }
}
