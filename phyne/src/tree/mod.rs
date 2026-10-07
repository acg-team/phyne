use std::fmt::{Debug, Display};

use fixedbitset::FixedBitSet;
use hashbrown::HashSet;
use inc_stats::Percentiles;

use crate::alignment::Sequences;
use crate::parsimony::Rounding;
use crate::tree::NodeIdx::{Internal as Int, Leaf};
use crate::{bail, Result};

pub mod nj_builder;
pub use nj_builder::NJTreeBuilder;
mod nj_matrices;
pub mod tree_builder;
pub use tree_builder::TreeBuilder;
pub mod tree_parser;

mod tree_node;
pub use tree_node::*;

#[derive(PartialEq, Clone, Copy, PartialOrd, Eq, Ord, Hash)]
pub enum NodeIdx {
    Internal(usize),
    Leaf(usize),
}

impl Display for NodeIdx {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Int(idx) => write!(f, "internal node {idx}"),
            Leaf(idx) => write!(f, "leaf node {idx}"),
        }
    }
}

impl Debug for NodeIdx {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Int(idx) => write!(f, "Int({idx})"),
            Leaf(idx) => write!(f, "Leaf({idx})"),
        }
    }
}

impl From<&NodeIdx> for usize {
    fn from(node_idx: &NodeIdx) -> usize {
        Self::from(*node_idx)
    }
}

impl From<NodeIdx> for usize {
    fn from(node_idx: NodeIdx) -> usize {
        match node_idx {
            Int(idx) => idx,
            Leaf(idx) => idx,
        }
    }
}

/// A rooted binary phylogenetic tree.
#[derive(Debug, Clone, PartialEq)]
pub struct Tree {
    pub root: NodeIdx,
    pub(crate) nodes: Vec<Node>,
    postorder: Vec<NodeIdx>,
    preorder: Vec<NodeIdx>,
    /// The number of leaves in the tree.
    pub n: usize,
    /// The sum of all branch lengths of the tree.
    pub length: f64,
    pub(crate) dirty: FixedBitSet,
}

impl Display for Tree {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_newick())
    }
}

impl Tree {
    pub(crate) fn new(sequences: &Sequences) -> Result<Self> {
        let n = sequences.len();
        if n == 0 {
            bail!(Sequence, "no sequences provided, aborting");
        }
        let root = if n == 1 { Leaf(0) } else { Int(2 * n - 2) };
        Ok(Self {
            root,
            postorder: Vec::new(),
            preorder: Vec::new(),
            nodes: (0..n)
                .zip(sequences.into_iter().map(|seq| seq.id().to_string()))
                .map(|(idx, id)| Node::new_leaf(idx, None, 0.0, id))
                .collect(),
            n,
            length: 0.0,
            dirty: FixedBitSet::with_capacity(2 * n - 1),
        })
    }

    /// Marks all nodes in the tree as clean (not dirty).
    pub fn clean(&mut self) {
        self.dirty.set_range(0..self.dirty.len(), false);
    }

    /// Marks all nodes in the tree as dirty (in case of disruptive changes to the tree structure
    /// which mean that e.g. cached information for the likelihood computation is not valid anymore).
    pub fn dirty(&mut self) {
        self.dirty.set_range(0..self.dirty.len(), true);
    }

    pub fn robinson_foulds(&self, other: &Tree) -> usize {
        let mut dist = 0;
        let common_leaves = self.common_leaf_set(other);
        let parts = self.filtered_partitions(&common_leaves);
        let other_parts: Vec<HashSet<String>> = other.filtered_partitions(&common_leaves);
        for part in parts.iter() {
            if !other_parts.contains(part) {
                dist += 1;
            }
        }
        dist
    }

    fn filtered_partitions(&self, common_leaves: &HashSet<String>) -> Vec<HashSet<String>> {
        self.partitions()
            .iter()
            .map(|set| set.intersection(common_leaves).cloned().collect())
            .filter(|set: &HashSet<String>| !set.is_empty())
            .collect()
    }

    fn common_leaf_set(&self, other: &Tree) -> HashSet<String> {
        self.leaf_ids()
            .intersection(&other.leaf_ids())
            .cloned()
            .collect()
    }

    pub fn partitions(&self) -> Vec<HashSet<String>> {
        let mut partitions = Vec::new();
        let all_leaves: HashSet<String> =
            self.leaves().iter().map(|node| node.id.clone()).collect();
        // skip root because it is a trivial partition
        for node in self.preorder.iter().skip(1) {
            let partition = self
                .preorder_subroot(node)
                .iter()
                .filter(|idx| matches!(idx, Leaf(_)))
                .map(|idx| self.node_id(idx).to_string())
                .collect();
            let other: HashSet<String> = all_leaves.difference(&partition).cloned().collect();
            if partitions.contains(&partition) {
                continue;
            }
            partitions.push(partition.clone());
            partitions.push(other.clone());
        }
        partitions
    }

    pub(crate) fn is_subtree(&self, query: &NodeIdx, node: &NodeIdx) -> bool {
        let order = self.preorder_subroot(node);
        order.contains(query)
    }

    pub fn sibling(&self, node_idx: &NodeIdx) -> Option<NodeIdx> {
        let parent = self.nodes[usize::from(node_idx)].parent?;
        let siblings = &self.nodes[usize::from(&parent)].children;
        siblings
            .iter()
            .find(|&sibling| sibling != node_idx)
            .cloned()
    }

    pub fn iter(&self) -> impl Iterator<Item = &Node> {
        self.nodes.iter()
    }

    pub fn children(&self, node_idx: &NodeIdx) -> &Vec<NodeIdx> {
        &self.nodes[usize::from(node_idx)].children
    }

    pub fn node(&self, node_idx: &NodeIdx) -> &Node {
        &self.nodes[usize::from(node_idx)]
    }

    pub fn by_id(&self, id: &str) -> &Node {
        self.nodes.iter().find(|node| node.id == id).unwrap()
    }

    pub fn node_mut(&mut self, node_idx: &NodeIdx) -> &mut Node {
        &mut self.nodes[usize::from(node_idx)]
    }

    /// Returns the number of nodes in the tree.
    // TODO: use an alternative name to avoid confusion with length = sum of branch lengths
    // possibly: size, n_nodes, num_nodes,
    // See issue #82 https://github.com/acg-team/rust-phylo/issues/82
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub fn preorder(&self) -> &Vec<NodeIdx> {
        &self.preorder
    }

    pub fn postorder(&self) -> &Vec<NodeIdx> {
        &self.postorder
    }

    pub fn to_newick(&self) -> String {
        format!("({});", self.to_newick_subroot(self.root))
    }

    fn to_newick_subroot(&self, node_idx: NodeIdx) -> String {
        match node_idx {
            NodeIdx::Leaf(idx) => {
                let node = &self.nodes[idx];
                format!("{}:{}", node.id, node.blen)
            }
            NodeIdx::Internal(idx) => {
                let node = &self.nodes[idx];
                let children_newick: Vec<String> = node
                    .children
                    .iter()
                    .map(|&child_idx| self.to_newick_subroot(child_idx))
                    .collect();
                format!("({}){}:{}", children_newick.join(","), node.id, node.blen)
            }
        }
    }

    pub fn node_id(&self, node_idx: &NodeIdx) -> &str {
        &self.nodes[usize::from(node_idx)].id
    }

    pub(crate) fn add_parent(
        &mut self,
        parent_idx: usize,
        idx_i: &NodeIdx,
        idx_j: &NodeIdx,
        blen_i: f64,
        blen_j: f64,
    ) {
        self.nodes.push(Node::new_internal(
            parent_idx,
            None,
            vec![*idx_i, *idx_j],
            0.0,
            "".to_string(),
        ));
        self.add_parent_to_child(idx_i, &Int(parent_idx), blen_i);
        self.add_parent_to_child(idx_j, &Int(parent_idx), blen_j);
    }

    pub(crate) fn add_parent_to_child(&mut self, idx: &NodeIdx, parent_idx: &NodeIdx, blen: f64) {
        self.nodes[usize::from(idx)].parent = Some(*parent_idx);
        self.nodes[usize::from(idx)].blen = blen;
    }

    pub(crate) fn compute_postorder(&mut self) {
        let mut order = Vec::<NodeIdx>::with_capacity(self.nodes.len());
        let mut stack = Vec::<NodeIdx>::with_capacity(self.nodes.len());
        let mut cur_root = self.root;
        stack.push(cur_root);
        while !stack.is_empty() {
            cur_root = stack.pop().unwrap();
            order.push(cur_root);
            if let Int(idx) = cur_root {
                stack.push(self.nodes[idx].children[0]);
                stack.push(self.nodes[idx].children[1]);
            }
        }
        order.reverse();
        self.postorder = order;
    }

    pub(crate) fn compute_preorder(&mut self) {
        self.preorder = self.preorder_subroot(&self.root);
    }

    pub fn preorder_subroot(&self, subroot_idx: &NodeIdx) -> Vec<NodeIdx> {
        let mut order = Vec::<NodeIdx>::with_capacity(self.nodes.len());
        let mut stack = Vec::<NodeIdx>::with_capacity(self.nodes.len());
        let mut cur_root = *subroot_idx;
        stack.push(cur_root);
        while !stack.is_empty() {
            cur_root = stack.pop().unwrap();
            order.push(cur_root);
            if let Int(idx) = cur_root {
                for child in self.nodes[idx].children.iter().rev() {
                    stack.push(*child);
                }
            }
        }
        order
    }

    /// Returns the ids of all leaves in the tree as a HashSet.
    ///
    /// # Examples
    /// ```
    /// use hashbrown::HashSet;
    ///
    /// use phylo::tree::tree_parser::from_newick;
    ///
    /// let trees = from_newick("(A:1.0,(B:1.0,C:1.0)E:2.0)F:1.0;").unwrap();
    /// let leaf_ids = trees[0].leaf_ids();
    /// let expected = HashSet::from_iter(["A", "B", "C"].map(String::from));
    /// assert_eq!(leaf_ids, expected);
    /// ```
    pub fn leaf_ids(&self) -> HashSet<String> {
        self.nodes
            .iter()
            .filter(|n| matches!(n.idx, Leaf(_)))
            .map(|n| n.id.clone())
            .collect()
    }

    pub fn try_idx(&self, id: &str) -> Result<NodeIdx> {
        let node = self.nodes.iter().find(|node| node.id == id);
        if let Some(node) = node {
            return Ok(node.idx);
        }
        bail!(Tree, "no node with id {id} found in the tree");
    }

    #[cfg(test)]
    pub(crate) fn idx(&self, id: &str) -> NodeIdx {
        self.nodes.iter().find(|node| node.id == id).unwrap().idx
    }

    pub fn set_blen(&mut self, node_idx: &NodeIdx, blen: f64) {
        debug_assert!(blen >= 0.0);
        let idx = usize::from(node_idx);
        let old_blen = self.nodes[idx].blen;
        self.length += blen - old_blen;
        self.nodes[idx].blen = blen;
        self.dirty.set(idx, true);
    }

    pub fn parent(&self, node_idx: &NodeIdx) -> Option<NodeIdx> {
        self.nodes[usize::from(node_idx)].parent
    }

    pub fn leaves(&self) -> Vec<&Node> {
        self.nodes
            .iter()
            .filter(|&x| matches!(x.idx, Leaf(_)))
            .collect()
    }

    pub fn internals(&self) -> Vec<&Node> {
        self.nodes
            .iter()
            .filter(|&x| matches!(x.idx, Int(_)))
            .collect()
    }

    pub fn node_ids_are_unique(&self) -> Result<()> {
        let mut seen = HashSet::new();
        for node_idx in self.postorder() {
            if !seen.insert(self.node_id(node_idx)) {
                bail!(
                    Tree,
                    "duplicate node id ({}) found in the tree",
                    self.node_id(node_idx)
                );
            }
        }
        Ok(())
    }

    fn finalise(&mut self) {
        self.n = self.leaves().len();
        debug_assert_eq!(self.nodes.len(), self.n * 2 - 1);

        self.compute_postorder();
        self.compute_preorder();
        self.length = self.nodes.iter().map(|n| n.blen).sum();
        self.dirty = FixedBitSet::with_capacity(self.n * 2 - 1);
    }
}

pub fn percentiles(lengths: &[f64], categories: u32) -> Vec<f64> {
    percentiles_rounded(lengths, categories, &Rounding::none())
}

pub fn percentiles_rounded(lengths: &[f64], categories: u32, rounding: &Rounding) -> Vec<f64> {
    let lengths: Percentiles<f64> = lengths.iter().collect();
    let percentiles: Vec<f64> = (1..(categories + 1))
        .map(|cat| 1.0 / ((categories + 1) as f64) * (cat as f64))
        .collect();
    let mut values = lengths.percentiles(percentiles).unwrap().unwrap();
    if rounding.yes() {
        values.iter_mut().for_each(|len| {
            *len = (*len * (10.0_f64.powf(rounding.digits as f64))).round()
                / (10.0_f64.powf(rounding.digits as f64))
        });
    }
    values
}

#[cfg(test)]
#[cfg_attr(coverage, coverage(off))]
mod tests;
