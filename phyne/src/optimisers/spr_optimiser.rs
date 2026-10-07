use std::fmt::Display;

use approx::relative_eq;
use itertools::Itertools;
use log::{debug, info};

use crate::likelihood::TreeSearchCost;
use crate::optimisers::move_optimiser::{MoveCostInfo, MoveOptimiser};
use crate::optimisers::optimise_branch;
use crate::tree::{NodeIdx, Tree};
use crate::{bail, Result};

#[derive(Clone)]
pub struct SprOptimiser {}

impl SprOptimiser {
    pub fn new() -> Self {
        SprOptimiser {}
    }
}

impl Default for SprOptimiser {
    fn default() -> Self {
        SprOptimiser::new()
    }
}

impl MoveOptimiser for SprOptimiser {
    fn move_locations<'a, C: TreeSearchCost + Display + Send + Clone + Display>(
        &self,
        cost: &'a C,
    ) -> impl Iterator<Item = &'a NodeIdx> {
        cost.tree()
            .preorder()
            .iter()
            .filter(|&n| *n != cost.tree().root)
    }

    fn best_move_at_location<C: TreeSearchCost + Clone + Display + Send>(
        &self,
        base_cost: f64,
        cost: &C,
        node_idx: &NodeIdx,
    ) -> Result<MoveCostInfo> {
        self.find_max_cost_regraft_for_prune(base_cost, cost, node_idx)
    }
}

impl Display for SprOptimiser {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SPR")
    }
}

impl SprOptimiser {
    fn available_regraft_locations<'a>(
        &self,
        tree: &'a Tree,
        prune_location: &NodeIdx,
    ) -> impl Iterator<Item = &'a NodeIdx> {
        let all_locations = tree.preorder();
        let prune_subtrees = tree.preorder_subroot(prune_location);
        let sibling = tree.sibling(prune_location).unwrap();
        let parent = tree.node(prune_location).parent.unwrap();
        all_locations.iter().filter(move |&node| {
            *node != sibling
                && *node != parent
                && *node != tree.root
                && !prune_subtrees.contains(node)
        })
    }

    fn find_max_cost_regraft_for_prune<C: TreeSearchCost + Clone + Display + Send>(
        &self,
        base_cost: f64,
        cost: &C,
        prune_location: &NodeIdx,
    ) -> Result<MoveCostInfo> {
        let tree = cost.tree();
        if tree.children(&tree.root).contains(prune_location) {
            // due to topology change the current node may have become the direct child of root
            // so the move is no longer possible
            return Ok(MoveCostInfo::new(base_cost, tree.clone()));
        }

        let regraft_locations = self
            .available_regraft_locations(tree, prune_location)
            .copied()
            .collect_vec();

        info!("Node {prune_location:?}: trying to regraft");
        calc_best_regraft_cost(base_cost, *prune_location, regraft_locations, cost)
    }
}

cfg_if::cfg_if! {
if #[cfg(feature="par-regraft")] {
fn calc_best_regraft_cost<C: TreeSearchCost + Clone + Display + Send>(
    base_cost: f64,
    prune_location: NodeIdx,
    regraft_locations: Vec<NodeIdx>,
    cost: &C,
) -> Result<MoveCostInfo> {
    use rayon::prelude::*;
    let cost_funcs = vec![cost.clone(); regraft_locations.len()];
    regraft_locations
        .into_par_iter()
        .zip_eq(cost_funcs)
        .map(move |(regraft, cost_fn)| {
            calc_spr_cost_with_blen_opt(prune_location, regraft, base_cost, cost_fn.clone())
        })
        .try_reduce_with(|left, right| Ok(if left.cost > right.cost {left} else {right}))
        .unwrap_or_else(|| Ok(MoveCostInfo::new(base_cost, cost.tree().clone())))
}
} else if #[cfg(feature="par-regraft-chunk")] {
/// NOTE: seems to be faster than full on parallel for few taxa
fn calc_best_regraft_cost<C: TreeSearchCost + Clone + Display + Send>(
    base_cost: f64,
    prune_location: NodeIdx,
    regraft_locations: Vec<NodeIdx>,
    cost: &C,
) -> Result<MoveCostInfo> {
    use rayon::prelude::*;
    // TODO: determine better factor (maybe dynamically)
    const CHUNK_SIZE: usize = 2;
    let cost_funcs = vec![cost.clone(); regraft_locations.len().div_ceil(CHUNK_SIZE)];

    regraft_locations
        .par_chunks(CHUNK_SIZE)
        .zip_eq(cost_funcs)
        .map(move |(regrafts, cost_func)| -> Result<_> {
            let mut max: Option<MoveCostInfo> = None;
            let mut max_cost = f64::MIN;
            for regraft_result in regrafts.iter().map(move |regraft| {
                calc_spr_cost_with_blen_opt(prune_location, *regraft, base_cost, cost_func.clone())
            }) {
                match regraft_result {
                    Ok(regraft_info) if regraft_info.cost > max_cost => {
                        max_cost = regraft_info.cost;
                        max = Some(regraft_info);
                    },
                    Ok(_) => {}
                    Err(error) => return Err(error),
                }
            }
            Ok(max.expect("at least one regraft location"))
        })
        .try_reduce_with(|left, right| Ok(if left.cost > right.cost {left} else {right}))
        .unwrap_or_else(|| Ok(MoveCostInfo::new(base_cost, cost.tree().clone())))
}
} else if #[cfg(feature="par-regraft-manual")] {
fn calc_best_regraft_cost<C: TreeSearchCost + Clone + Display + Send>(
    base_cost: f64,
    prune_location: NodeIdx,
    regraft_locations: Vec<NodeIdx>,
    cost: &C,
) -> Result<MoveCostInfo> {
    if regraft_locations.is_empty() {
        return Ok(MoveCostInfo::new(base_cost, cost.tree().clone()));
    }
    #[derive(Clone)]
    struct RecursiveForkJoinRegrafter<C: TreeSearchCost + Clone + Display + Send> {
        cost_fn: C,
        prune_location: NodeIdx,
        base_cost: f64,
    }
    /// NOTE: by being recursive these tasks can be stored solely on the stack
    /// using rayon::scope might look simpler but incurs overhead by having to manage
    /// tasks on the heap
    fn regraft_recursive<C: TreeSearchCost + Clone + Display + Send>(state: RecursiveForkJoinRegrafter<C>, regraft_locations: &[NodeIdx]) -> Result<MoveCostInfo> {
        if regraft_locations.len() == 1 {
            return calc_spr_cost_with_blen_opt(state.prune_location, regraft_locations[0], state.base_cost, state.cost_fn);
        }
        let (left_locations, right_locations) = regraft_locations.split_at(regraft_locations.len() / 2);
        let r2 = state.clone();
        match rayon::join(move || regraft_recursive(state, left_locations), move ||regraft_recursive(r2, right_locations)) {
            (Ok(left), Ok(right)) => Ok(if left.cost > right.cost {left} else {right}) ,
            (Err(error), _) | (_, Err(error))   => Err(error),
        }
    }
    regraft_recursive(RecursiveForkJoinRegrafter { cost_fn: cost.clone(), prune_location, base_cost }, &regraft_locations)
}
} else {
fn calc_best_regraft_cost<C: TreeSearchCost + Clone + Display + Send>(
    base_cost: f64,
    prune_location: NodeIdx,
    regraft_locations: Vec<NodeIdx>,
    cost: &C,
) -> Result<MoveCostInfo> {
    let mut max = None;
    let mut max_cost = f64::MIN;
    for regraft in regraft_locations.into_iter().map(move |regraft| {
        calc_spr_cost_with_blen_opt(prune_location, regraft, base_cost, cost.clone())
    }) {
        match regraft {
            Ok(regraft_info) if regraft_info.cost > max_cost => {
                max_cost = regraft_info.cost;
                max = Some(regraft_info);
            },
            Ok(_) => {}
            Err(error) => return Err(error),
        }
    }
    Ok(max.unwrap_or_else(|| MoveCostInfo::new(base_cost, cost.tree().clone())))
}
}}

/// for evo models with branch length optimisation enabled (disabled for parsimony),
/// if the move doesn't result in improvement over `base_cost`
/// the blen of the regrafted branch is optimised to check if an
/// improvement could still be reached
fn calc_spr_cost_with_blen_opt<C: TreeSearchCost + Clone + Display>(
    prune_location: NodeIdx,
    regraft: NodeIdx,
    base_cost: f64,
    mut cost_fn: C,
) -> Result<MoveCostInfo> {
    let mut new_tree = rooted_spr(cost_fn.tree(), &prune_location, &regraft)?;

    cost_fn.update_tree(new_tree.clone());

    let mut move_cost = cost_fn.cost();
    // Branch length optimisation is skipped if the model does not support branch lengths
    // or if the move_cost is already better since we apply the move as is.
    if cost_fn.blen_optimisation() && move_cost <= base_cost {
        // reoptimise branch length at the regraft location
        let blen_opt = optimise_branch(&cost_fn, &regraft)?;
        if blen_opt.final_cost > move_cost {
            move_cost = blen_opt.final_cost;
            new_tree.set_blen(&regraft, blen_opt.value);
        }
    }
    debug!("    Regraft to {regraft:?} w best cost {move_cost}");
    Ok(MoveCostInfo::new(move_cost, new_tree))
}

fn rooted_spr(tree: &Tree, prune_idx: &NodeIdx, regraft_idx: &NodeIdx) -> Result<Tree> {
    // Pruned node must have a parent, it is the one being reattached
    if prune_idx == &tree.root {
        bail!(TreeMove, "cannot prune the root node");
    }
    // Prune and regraft nodes must be different
    if prune_idx == regraft_idx {
        bail!(TreeMove, "prune and regraft nodes must be different");
    }
    // Regraft node cannot be a subtree of the prune node as that will break the tree
    // into disconnected pieces
    if tree.is_subtree(regraft_idx, prune_idx) {
        bail!(
            TreeMove,
            "regraft node cannot be a subtree of the prune node"
        );
    }

    let prune = tree.node(prune_idx);
    // Cannot prune direct child of the root node, otherwise branch lengths are undefined
    if tree.node(&prune.parent.unwrap()).parent.is_none() {
        bail!(TreeMove, "cannot prune direct child of the root node");
    }
    let regraft = tree.node(regraft_idx);
    // Regrafted node must have a parent, the prune parent is attached to that branch
    if regraft.parent.is_none() {
        bail!(TreeMove, "cannot regraft to the root node");
    }
    if regraft.parent == prune.parent {
        bail!(
            TreeMove,
            "prune and regraft nodes must have different parents"
        );
    }

    Ok(rooted_spr_unchecked(tree, prune_idx, regraft_idx))
}

/// text
/// before:                       |
///                     -----prune_grpar-----
///                     |                   |
///               --prune_par--       --regraft_par--
///               |           |       |             |
///          prune_sib      prune     △             |
///                                              regraft
///
/// after:                        |
///                     -----prune_grpar-----
///                     |                   |
///                     |             --regraft_par--
///                prune_sib          |             |
///                                   △        --prune_par--
///                                            |           |
///                                          prune      regraft
///
fn rooted_spr_unchecked(tree: &Tree, prune_idx: &NodeIdx, regraft_idx: &NodeIdx) -> Tree {
    let prune = tree.node(prune_idx);
    let prune_sib = tree.node(&tree.sibling(&prune.idx).unwrap());
    let prune_par = tree.node(&prune.parent.unwrap());
    let prune_grpar = tree.node(&prune_par.parent.unwrap());
    let regraft = tree.node(regraft_idx);
    let regraft_par = tree.node(&regraft.parent.unwrap());

    let mut new_tree = tree.clone();

    {
        new_tree.dirty.set(usize::from(prune_sib.idx), true);
        new_tree.dirty.set(usize::from(prune_par.idx), true);
    }

    {
        // Sibling of pruned node connects to common parent, branch length is updated
        let prune_sib = new_tree.node_mut(&prune_sib.idx);
        prune_sib.parent = prune_par.parent;
        prune_sib.blen += prune_par.blen;
    };

    {
        // Pruned node's parent is removed from its parent's children, pruned nodes sibling is added
        let prune_grpar = new_tree.node_mut(&prune_grpar.idx);
        prune_grpar.children.retain(|&x| x != prune_par.idx);
        prune_grpar.children.push(prune_sib.idx);
    };

    {
        // Regrafted branch is split in two, parent of regrafted node is now pruned node's parent
        let regraft = new_tree.node_mut(&regraft.idx);
        regraft.parent = Some(prune_par.idx);
        regraft.blen /= 2.0;
    }

    {
        // Regrafted node is removed from its parent's children, pruned node's parent is added
        let prune_par = new_tree.node_mut(&prune_par.idx);
        prune_par.children.retain(|&x| x != prune_sib.idx);
        prune_par.children.push(regraft.idx);
        prune_par.blen = regraft.blen / 2.0;
        prune_par.parent = regraft.parent;
    }

    {
        // Regrafted node's parent's children are updated
        let regraft_par = new_tree.node_mut(&regraft_par.idx);
        regraft_par.children.retain(|&x| x != regraft.idx);
        regraft_par.children.push(prune_par.idx);
    }

    // Tree height should not have changed
    debug_assert!(relative_eq!(
        new_tree.length,
        new_tree.nodes.iter().map(|node| node.blen).sum(),
        epsilon = 1e-10
    ));

    new_tree.compute_postorder();
    new_tree.compute_preorder();
    debug_assert_eq!(new_tree.postorder().len(), tree.postorder().len());
    new_tree
}

#[cfg(test)]
#[cfg_attr(coverage, coverage(off))]
mod private_spr_tests {
    use approx::assert_relative_eq;
    use assert_matches::assert_matches;

    use crate::optimisers::spr_optimiser::{rooted_spr, rooted_spr_unchecked};
    use crate::tree;
    use crate::Error;

    #[test]
    fn spr_siblings() {
        let tree = tree!("(((A:1.0,B:1.0)E:5.1,(C:3.0,D:4.0)F:6.2)G:7.3);");
        let res = rooted_spr(&tree, &tree.idx("A"), &tree.idx("B"));
        assert_matches!(res, Err(Error::TreeMove(msg)) if msg.contains("prune and regraft nodes must have different parents"));
    }

    #[test]
    fn spr_prune_root_or_children() {
        let tree = tree!("(((A:1.0,B:1.0)E:5.1,(C:3.0,D:4.0)F:6.2)G:7.3);");
        assert_matches!(rooted_spr(&tree, &tree.idx("G"), &tree.idx("B")),
            Err(Error::TreeMove(msg)) if msg.contains("cannot prune the root node"));
        assert_matches!(rooted_spr(&tree, &tree.idx("E"), &tree.idx("F")),
            Err(Error::TreeMove(msg)) if msg.contains("cannot prune direct child of the root node"));
        assert_matches!(rooted_spr(&tree, &tree.idx("F"), &tree.idx("E")),
            Err(Error::TreeMove(msg)) if msg.contains("cannot prune direct child of the root node"));
    }

    #[test]
    #[should_panic]
    fn spr_prune_root_unchecked() {
        let tree = tree!("(((A:1.0,B:1.0)E:5.1,(C:3.0,D:4.0)F:6.2)G:7.3);");
        rooted_spr_unchecked(&tree, &tree.idx("G"), &tree.idx("B"));
    }

    #[test]
    #[should_panic]
    fn spr_prune_root_child_unchecked() {
        let tree = tree!("(((A:1.0,B:1.0)E:5.1,(C:3.0,D:4.0)F:6.2)G:7.3);");
        rooted_spr_unchecked(&tree, &tree.idx("F"), &tree.idx("B"));
    }

    #[test]
    fn spr_regraft_root() {
        let tree = tree!("(((A:1.0,B:1.0)E:5.1,(C:3.0,D:4.0)F:6.2)G:7.3);");
        let root = tree.root;
        assert_matches!(rooted_spr(&tree, &tree.idx("A"), &root),
            Err(Error::TreeMove(msg)) if msg.contains("cannot regraft to the root node"));
    }

    #[test]
    fn spr_regraft_same() {
        let tree = tree!("(((A:1.0,B:1.0)E:5.1,(C:3.0,D:4.0)F:6.2)G:7.3);");
        assert_matches!(rooted_spr(&tree, &tree.idx("C"), &tree.idx("C")),
               Err(Error::TreeMove(msg)) if msg.contains("prune and regraft nodes must be different"));
    }

    #[test]
    #[should_panic]
    fn spr_regraft_root_unchecked() {
        let tree = tree!("(((A:1.0,B:1.0)E:5.1,(C:3.0,D:4.0)F:6.2)G:7.3);");
        rooted_spr_unchecked(&tree, &tree.idx("B"), &tree.idx("G"));
    }

    #[test]
    fn spr_regraft_subtree() {
        let tree = tree!("((((A:1.0,B:1.0)E:5.1,(C:3.0,D:4.0)F:6.2)G:7.3,H:1.0)K:1.0);");
        assert_matches!(rooted_spr(&tree, &tree.idx("E"), &tree.idx("B")),
            Err(Error::TreeMove(msg)) if msg.contains("regraft node cannot be a subtree of the prune node"));
    }

    #[test]
    #[should_panic]
    fn spr_regraft_subtree_unchecked() {
        let tree = tree!("((((A:1.0,B:1.0)E:5.1,(C:3.0,D:4.0)F:6.2)G:7.3,H:1.0)K:1.0);");
        rooted_spr_unchecked(&tree, &tree.idx("E"), &tree.idx("B"));
    }

    #[test]
    #[should_panic]
    fn spr_regraft_siblings() {
        let tree = tree!("((((A:1.0,B:1.0)E:5.1,(C:3.0,D:4.0)F:6.2)G:7.3,H:1.0)K:1.0);");
        rooted_spr_unchecked(&tree, &tree.idx("A"), &tree.idx("B"));
    }

    #[test]
    fn spr_simple_valid() {
        let tree = tree!("(((A:1.0,B:1.0)E:5.1,(C:3.0,D:4.0)F:6.2)G:7.3);");
        let new_tree = rooted_spr(&tree, &tree.idx("A"), &tree.idx("C")).unwrap();
        assert_eq!(new_tree.len(), tree.len());
        assert_relative_eq!(new_tree.length, tree.length);
        let prune_sib = new_tree.node(&tree.idx("B"));
        assert_eq!(prune_sib.blen, 6.1);
        assert_eq!(prune_sib.parent, Some(tree.idx("G")));
        let prune_gpar = new_tree.node(&tree.idx("G"));
        assert!([tree.idx("F"), tree.idx("B")].contains(&prune_gpar.children[0]));
        assert!([tree.idx("B"), tree.idx("F")].contains(&prune_gpar.children[1]));
        assert_eq!(prune_gpar.blen, 7.3);
        let regraft_par = new_tree.node(&tree.idx("F"));
        assert_eq!(regraft_par.blen, 6.2);
        assert!([tree.idx("E"), tree.idx("D")].contains(&regraft_par.children[0]));
        assert!([tree.idx("E"), tree.idx("D")].contains(&regraft_par.children[1]));
        let prune = new_tree.node(&tree.idx("A"));
        assert_eq!(prune.blen, 1.0);
        assert_eq!(prune.parent, Some(tree.idx("E")));
        let prune_par = new_tree.node(&tree.idx("E"));
        assert_eq!(prune_par.blen, 1.5);
        assert!([tree.idx("A"), tree.idx("C")].contains(&prune_par.children[0]));
        assert!([tree.idx("A"), tree.idx("C")].contains(&prune_par.children[1]));
        let regraft_sib = new_tree.node(&tree.idx("D"));
        assert_eq!(regraft_sib.blen, 4.0);
        assert_eq!(regraft_sib.parent, Some(tree.idx("F")));
    }

    #[test]
    fn spr_broken() {
        let tree = tree!("(((A:1.0,B:1.0)E:2.0,(C:1.0,D:1.0)F:2.0)G:3.0);");
        let new_tree = rooted_spr(&tree, &tree.idx("A"), &tree.idx("C")).unwrap();
        let ng = new_tree.node(&tree.idx("G"));
        assert!([tree.idx("F"), tree.idx("B")].contains(&ng.children[0]));
        assert!([tree.idx("F"), tree.idx("B")].contains(&ng.children[1]));
        let nf = new_tree.node(&tree.idx("F"));
        assert!([tree.idx("E"), tree.idx("D")].contains(&nf.children[0]));
        assert!([tree.idx("E"), tree.idx("D")].contains(&nf.children[1]));
        assert_eq!(nf.parent, Some(tree.idx("G")));
        let ne = new_tree.node(&tree.idx("E"));
        assert!([tree.idx("A"), tree.idx("C")].contains(&ne.children[0]));
        assert!([tree.idx("A"), tree.idx("C")].contains(&ne.children[1]));
        assert_eq!(ne.parent, Some(tree.idx("F")));
        assert_eq!(new_tree.len(), tree.len());
        assert_relative_eq!(new_tree.length, tree.length);
    }
}
