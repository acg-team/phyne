use std::fmt::Display;

use crate::likelihood::TreeSearchCost;
use crate::optimisers::{optimise_branch, MoveCostInfo, MoveOptimiser};
use crate::tree::{
    NodeIdx::{self, Leaf},
    Tree,
};
use crate::{bail, Result};

#[derive(Clone)]
pub struct NniOptimiser {}

impl NniOptimiser {
    pub fn new() -> Self {
        NniOptimiser {}
    }
}

impl Default for NniOptimiser {
    fn default() -> Self {
        NniOptimiser::new()
    }
}

impl MoveOptimiser for NniOptimiser {
    fn move_locations<'a, C: TreeSearchCost + Display + Send + Clone + Display>(
        &self,
        cost: &'a C,
    ) -> impl Iterator<Item = &'a NodeIdx> {
        cost.tree()
            .preorder()
            .iter()
            .filter(|&n| *n != cost.tree().root && !matches!(n, Leaf(_)))
    }

    fn best_move_at_location<C>(
        &self,
        base_cost: f64,
        cost: &C,
        node_idx: &NodeIdx,
    ) -> Result<MoveCostInfo>
    where
        C: TreeSearchCost + Display + Send + Clone,
    {
        let mut max_cost_info = None;
        let mut max_cost = f64::MIN;
        for child_idx in &cost.tree().node(node_idx).children {
            let move_cost_info =
                calc_nni_cost_with_blen_opt(node_idx, child_idx, base_cost, cost.clone())?;
            if move_cost_info.cost > max_cost {
                max_cost = move_cost_info.cost;
                max_cost_info = Some(move_cost_info);
            }
        }
        if let Some(info) = max_cost_info {
            Ok(info)
        } else {
            bail!(TreeMove, "at least one NNI move should be possible")
        }
    }
}

impl Display for NniOptimiser {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "NNI")
    }
}

fn calc_nni_cost_with_blen_opt<C: TreeSearchCost + Clone + Display>(
    node_idx: &NodeIdx,
    child_idx: &NodeIdx,
    base_cost: f64,
    mut cost_fn: C,
) -> Result<MoveCostInfo> {
    let mut new_tree = rooted_nni(cost_fn.tree(), node_idx, child_idx)?;
    cost_fn.update_tree(new_tree.clone());
    let mut move_cost = cost_fn.cost();
    if cost_fn.blen_optimisation() && move_cost <= base_cost {
        let blen_opt = optimise_branch(&cost_fn, node_idx)?;
        if blen_opt.final_cost > move_cost {
            move_cost = blen_opt.final_cost;
            new_tree.set_blen(node_idx, blen_opt.value);
        }
    }
    Ok(MoveCostInfo::new(move_cost, new_tree))
}

pub(crate) fn rooted_nni(tree: &Tree, node_idx: &NodeIdx, child_idx: &NodeIdx) -> Result<Tree> {
    if node_idx == &tree.root {
        bail!(
            TreeMove,
            "for rooted NNI the node must not be the root of the tree"
        );
    }
    if matches!(node_idx, Leaf(_)) {
        bail!(TreeMove, "for rooted NNI the node must not be a leaf");
    }
    if tree.node(child_idx).parent.is_none() || tree.node(child_idx).parent.unwrap() != *node_idx {
        bail!(
            TreeMove,
            "the node {node_idx} must be the parent of the {child_idx}"
        );
    }

    Ok(rooted_nni_unchecked(tree, node_idx, child_idx))
}

/// ```text
///            |
///       -- parent --
///       |          |
///   --node--      sibling
///   |      |
///   .    child
/// ```
/// Swapping child with sibling.
fn rooted_nni_unchecked(tree: &Tree, node_idx: &NodeIdx, child_idx: &NodeIdx) -> Tree {
    let mut new_tree = tree.clone();
    let sibling = tree.node(&tree.sibling(node_idx).unwrap());
    let parent = tree.node(&tree.node(node_idx).parent.unwrap());
    let node = tree.node(node_idx);

    new_tree.dirty.set(usize::from(node_idx), true);

    {
        let parent = new_tree.node_mut(&tree.node(node_idx).parent.unwrap());
        parent.children.retain(|c| c == node_idx);
        parent.children.push(*child_idx);
    };

    {
        let child = new_tree.node_mut(child_idx);
        child.parent = Some(parent.idx);
    };

    {
        let node = new_tree.node_mut(node_idx);
        node.children.retain(|c| c != child_idx);
        node.children.push(sibling.idx);
    };

    {
        let sibling = new_tree.node_mut(&sibling.idx);
        sibling.parent = Some(node.idx);
    };

    new_tree.compute_preorder();
    new_tree.compute_postorder();
    debug_assert_eq!(new_tree.postorder().len(), new_tree.preorder().len());
    debug_assert_eq!(new_tree.postorder().len(), tree.postorder().len());
    new_tree
}

#[cfg(test)]
#[cfg_attr(coverage, coverage(off))]
mod private_nni_tests {

    use assert_matches::assert_matches;

    use super::*;
    use crate::alignment::{Alignment, Sequences, MSA};
    use crate::phylo_info::PhyloInfo;
    use crate::substitution_models::{SubstModel, SubstitutionCostBuilder as SCB, JC69};
    use crate::tree::Tree;
    use crate::{record_wo_desc as record, tree, Error};

    #[cfg(test)]
    fn compare_trees(tree: &Tree, true_tree: Tree) {
        assert_eq!(tree.root, true_tree.root);
        for node_idx in tree.preorder() {
            let current = tree.node(node_idx);
            let current_id = &current.id;
            assert_eq!(current.blen, true_tree.by_id(current_id).blen);
            if node_idx == &tree.root {
                continue;
            }
            let true_parent = true_tree.by_id(current_id);
            let parent = tree.by_id(current_id);
            assert_eq!(parent.id, true_parent.id);
        }
    }
    #[test]
    fn nni_in_middle_of_tree() {
        // arrange
        let tree = tree!("((((A:1.0,B:1.0)F:1.0,C:2.0)G:1.0,D:3.0)H:1.0,E:4.0)I:1.0;");
        let true_tree_after_nni =
            tree!("(((A:1.0,B:1.0)F:1.0,(D:3.0,C:2.0)G:1.0)H:1.0,E:4.0)I:1.0;");
        let node_id = "G";
        let child_id = "F";

        // act
        let new_tree =
            rooted_nni(&tree, &tree.by_id(node_id).idx, &tree.by_id(child_id).idx).unwrap();

        // assert
        compare_trees(&new_tree, true_tree_after_nni);
        let dirty_nodes: Vec<_> = tree
            .postorder()
            .iter()
            .filter(|&x| new_tree.dirty[usize::from(x)])
            .collect();
        assert_eq!(dirty_nodes.len(), 1);
        assert_eq!(tree.node(dirty_nodes.first().unwrap()).id, node_id);
    }

    #[test]
    fn nni_at_parent_of_leaf() {
        // arrange
        let tree = tree!("((((A:1.0,B:1.0)F:1.0,C:2.0)G:1.0,D:3.0)H:1.0,E:4.0)I:1.0;");
        let true_tree_after_nni =
            tree!("((((C:2.0,B:1.0)F:1.0,A:1.0)G:1.0,D:3.0)H:1.0,E:4.0)I:1.0;");
        let node_id = "F";
        let child_id = "A";

        // act
        let new_tree =
            rooted_nni(&tree, &tree.by_id(node_id).idx, &tree.by_id(child_id).idx).unwrap();

        // assert
        compare_trees(&new_tree, true_tree_after_nni);
        let dirty_nodes: Vec<_> = tree
            .postorder()
            .iter()
            .filter(|&x| new_tree.dirty[usize::from(x)])
            .collect();
        assert_eq!(dirty_nodes.len(), 1);
        assert_eq!(tree.node(dirty_nodes.first().unwrap()).id, node_id);
    }

    #[test]
    fn nni_node_is_root() {
        let tree = tree!("((((A:1.0,B:1.0)F:1.0,C:2.0)G:1.0,D:3.0)H:1.0,E:4.0)I:1.0;");
        let node_id = "I";

        let err = rooted_nni(&tree, &tree.by_id(node_id).idx, &Leaf(0));

        assert_matches!(
            err,
            Err(Error::TreeMove(msg)) if msg.contains("root")
        );
    }

    #[test]
    fn nni_node_is_leaf() {
        let tree = tree!("((((A:1.0,B:1.0)F:1.0,C:2.0)G:1.0,D:3.0)H:1.0,E:4.0)I:1.0;");
        let node_id = "A";

        let err = rooted_nni(&tree, &tree.by_id(node_id).idx, &Leaf(0));

        assert_matches!(
            err,
            Err(Error::TreeMove(msg)) if msg.contains("leaf")
        );
    }

    #[test]
    fn nni_child_is_invalid() {
        let tree = tree!("((((A:1.0,B:1.0)F:1.0,C:2.0)G:1.0,D:3.0)H:1.0,E:4.0)I:1.0;");
        let node_id = "G";
        let child_id = "A";

        let err = rooted_nni(&tree, &tree.by_id(node_id).idx, &tree.by_id(child_id).idx);

        assert_matches!(
            err,
            Err(Error::TreeMove(msg)) if msg.contains("must be the parent")
        );
    }

    #[test]
    fn no_nnis_possible() {
        let tree = tree!("(((A0:1.0,B1:1.0)I1:1.0,C2:1.0)I2:1.0);");
        let seqs = Sequences::new_unchecked(vec![
            record!("A0", b"AAAA"),
            record!("B1", b"---A"),
            record!("C2", b"AA--"),
        ]);
        let msa = MSA::from_aligned(seqs, &tree).unwrap();
        let info = PhyloInfo { msa, tree };
        let node_id = "A0";

        let cost = SCB::new(SubstModel::<JC69>::default(), info)
            .build()
            .unwrap();

        let err =
            NniOptimiser::new().best_move_at_location(0.0, &cost, &cost.tree().by_id(node_id).idx);
        assert_matches!(
            err,
            Err(Error::TreeMove(msg)) if msg.contains("at least one NNI move should be possible")
        );
    }

    #[test]
    fn nni_possible() {
        let tree = tree!("(((A0:1.0,B1:1.0)I1:1.0,(C2:1.0,D3:1.0))I2:1.0);");
        let seqs = Sequences::new_unchecked(vec![
            record!("A0", b"AAAA"),
            record!("B1", b"---A"),
            record!("C2", b"AA--"),
            record!("D3", b"AA--"),
        ]);
        let msa = MSA::from_aligned(seqs, &tree).unwrap();
        let info = PhyloInfo { msa, tree };
        let node_id = "I1";

        let cost = SCB::new(SubstModel::<JC69>::default(), info)
            .build()
            .unwrap();

        let res =
            NniOptimiser::new().best_move_at_location(0.0, &cost, &cost.tree().by_id(node_id).idx);
        assert!(res.is_ok());
    }
}
