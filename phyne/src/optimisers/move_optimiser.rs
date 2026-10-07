use std::fmt::Display;

use crate::likelihood::TreeSearchCost;
use crate::tree::{NodeIdx, Tree};
use crate::Result;

/// This is the result of a move operation in a phylogenetic tree, i.e. the result of
/// a call to [`MoveOptimiser::best_move_at_location`].
#[derive(Debug, Clone)]
pub struct MoveCostInfo {
    pub cost: f64,
    // The tree contains a tree.dirty FixedBitSet which should mark all the nodes affected by the move,
    // and the call to [`TreeSearchCost::update_tree`] should update its internal node states.
    pub tree: Tree,
}

impl MoveCostInfo {
    pub fn new(cost: f64, tree: Tree) -> Self {
        MoveCostInfo { cost, tree }
    }
}

/// The MoveOptimiser trait is used to perform tree moves in a phylogenetic tree.
/// The method [`Self::move_locations`] returns an iterator over all locations in the tree where a move can
/// be performed. The method [`Self::best_move_at_location`] calculates the best move at a given location
/// in the tree.
///
/// The [`Display`] implementation should return the type of move that is being performed,
/// e.g. `NNI`, `SPR`, etc.
pub trait MoveOptimiser: Clone + Display {
    /// Returns an iterator over all locations in the tree where a move can be performed.
    /// E.g. for (rooted) SPR, this would return all nodes except the root.
    fn move_locations<'a, C: TreeSearchCost + Display + Send + Clone + Display>(
        &self,
        cost: &'a C,
    ) -> impl Iterator<Item = &'a NodeIdx>;

    /// Calculates the best move at the given location in the tree.
    /// E.g. for SPR, the given location `node_idx` is the prune location. Then, all
    /// possible regraft locations are iterated over, and the one with the best cost is
    /// returned.
    fn best_move_at_location<C: TreeSearchCost + Display + Send + Clone + Display>(
        &self,
        base_cost: f64,
        cost: &C,
        node_idx: &NodeIdx,
    ) -> Result<MoveCostInfo>;
}
