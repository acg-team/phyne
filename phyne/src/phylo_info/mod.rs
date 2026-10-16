use bio::io::fasta::Record;
use hashbrown::{HashMap, HashSet};

use crate::alignment::{
    Alignment, AncestralAlignment, InternalAlignments, Mapping, SeqMaps, Sequences,
};
use crate::evolutionary_models::FreqVector;
use crate::tree::tree_parser::from_newick;
use crate::tree::{
    NodeIdx::{self, Internal, Leaf},
    Tree,
};
use crate::{aligned_seq, bail, bug, record, Result};

mod phyloinfo_builder;
pub use phyloinfo_builder::*;

/// The PhyloInfo struct contains all the information needed for phylogenetic inference.
///
/// The struct can be built using the PhyloInfoBuilder from at least a fasta sequence file.
/// At the moment the sequences need to be aligned.
/// The struct also holds the leaf sequence character encodings.
///
/// # TODO:
/// * Ensure encoding matches model.
/// * Add support for unaligned sequences.
#[derive(Debug, Clone)]
pub struct PhyloInfo<A: Alignment> {
    /// Multiple sequence alignment
    pub msa: A,
    /// Phylogenetic tree
    pub tree: Tree,
}

impl<A: Alignment> PhyloInfo<A> {
    /// Compiles a representation of the alignment in a vector of fasta records.
    /// The alignment is compiled from the subtree rooted at `subroot`.
    /// If `subroot` is None, the whole alignment is compiled.
    /// Bails if the tree does not contain the subroot or does not match the alignment.
    pub fn compile_alignment(&self, subroot_opt: Option<&NodeIdx>) -> Result<Sequences> {
        let subroot = subroot_opt.unwrap_or(&self.tree.root);
        let maps = if subroot == &self.tree.root {
            self.msa.leaf_maps().clone()
        } else {
            Self::compile_leaf_map(
                subroot,
                self.msa.internal_alignments(),
                self.msa.seqs(),
                &self.tree,
            )
        };
        let mut records = Vec::with_capacity(maps.len());
        for (idx, map) in &maps {
            let rec = self.msa.seqs().record_by_id(self.tree.node_id(idx));
            let aligned_seq = aligned_seq!(map, rec.seq());
            records.push(Record::with_attrs(rec.id(), rec.desc(), &aligned_seq));
        }

        Sequences::with_alphabet(records, self.msa.seqs().alphabet)
    }

    pub(crate) fn compile_leaf_map(
        sub_root: &NodeIdx,
        internal_alignments: &InternalAlignments,
        seqs: &Sequences,
        tree: &Tree,
    ) -> SeqMaps {
        let msa_len = match sub_root {
            Internal(_) => internal_alignments[sub_root].len(),
            Leaf(_) => seqs.record_by_id(tree.node_id(sub_root)).seq().len(),
        };
        let mut stack = HashMap::<NodeIdx, Mapping>::with_capacity(tree.len());
        stack.insert(*sub_root, (0..msa_len).map(Some).collect());
        let mut leaf_map = SeqMaps::with_capacity(tree.n);
        for idx in &tree.preorder_subroot(sub_root) {
            match idx {
                Internal(_) => {
                    let parent = &stack[idx];
                    let childs = tree.children(idx);
                    let map_x = &internal_alignments[idx].map_x;
                    let map_y = &internal_alignments[idx].map_y;
                    let x = Self::map_child(parent, map_x);
                    let y = Self::map_child(parent, map_y);
                    stack.insert(childs[0], x);
                    stack.insert(childs[1], y);
                }
                Leaf(_) => {
                    leaf_map.insert(*idx, stack[idx].clone());
                }
            }
        }
        leaf_map
    }

    fn map_child(parent: &Mapping, child: &Mapping) -> Mapping {
        parent
            .iter()
            .map(|site| {
                if let Some(idx) = site {
                    child[*idx]
                } else {
                    None
                }
            })
            .collect::<Mapping>()
    }

    /// Returns the empirical frequencies of the symbols in the sequences.
    /// The frequencies are calculated from the unaligned sequences.
    /// If an unambiguous character is not present in the sequences, its frequency is artificially
    /// set to at least one.
    ///
    /// # Example
    /// ```
    /// use phyne::frequencies;
    /// use phyne::phylo_info::PhyloInfoBuilder;
    /// use phyne::evolutionary_models::FreqVector;
    /// # use phyne::Result;
    ///
    /// # fn main() -> Result<()> {
    /// let info = PhyloInfoBuilder::with_attrs(
    ///     "./examples/data/sequences_DNA1.fasta",
    ///     "./examples/data/tree_diff_branch_lengths_2.newick")
    /// .build()?;
    /// let freqs = info.freqs();
    /// assert_eq!(freqs, frequencies!(&[1.25, 2.25, 4.25, 1.25]).scale(1.0 / 9.0));
    /// assert_eq!(freqs.sum(), 1.0);
    /// # Ok(()) }
    /// ```
    pub fn freqs(&self) -> FreqVector {
        let alphabet = self.msa.alphabet();
        let mut freqs = FreqVector::zeros(alphabet.len());
        for &char in alphabet.symbols().iter().chain(alphabet.ambiguous()) {
            let count = self
                .msa
                .seqs()
                .into_iter()
                .map(|rec| rec.seq().iter().filter(|&c| c == &char).count())
                .sum::<usize>() as f64;
            let mut char_freq = alphabet.char_encoding(char).clone();
            char_freq.scale_mut(1.0 / char_freq.sum());
            freqs += char_freq.scale(count);
        }
        for char in alphabet.symbols().iter() {
            let idx = alphabet.index(char);
            if freqs[idx] == 0.0 {
                freqs[idx] = 1.0;
            }
        }
        freqs.scale_mut(1.0 / freqs.sum());
        freqs
    }
}

impl<AA: AncestralAlignment> PhyloInfo<AA> {
    /// Checks whether the alignment satisfies the Dollo constraint,
    /// i.e., each alignment column must be introduced exactly once
    /// in the phylogenetic tree.
    ///
    /// Returns `Ok(())` if all columns satisfy this constraint, or an error
    /// identifying the first violating column otherwise.
    pub fn check_dollos_constraint(&self) -> Result<()> {
        let mut violating_cols = Vec::new();
        for col_idx in 0..self.msa.len() {
            let mut num_insertions = 0;
            for node in self.tree.postorder() {
                if *node == self.tree.root {
                    if self.msa.ancestral_map(node)[col_idx].is_some() {
                        num_insertions += 1;
                    }
                } else {
                    let parent = self.tree.parent(node).unwrap();
                    let parent_site = self.msa.ancestral_map(&parent)[col_idx];
                    let node_site = match node {
                        Internal(_) => self.msa.ancestral_map(node)[col_idx],
                        Leaf(_) => self.msa.leaf_map(node)[col_idx],
                    };
                    if parent_site.is_none() && node_site.is_some() {
                        num_insertions += 1;
                    }
                }
            }
            if num_insertions != 1 {
                violating_cols.push((col_idx, num_insertions));
            }
        }
        if !violating_cols.is_empty() {
            let mut msg = String::from("Dollo constraint violated in columns:\n");
            for (col_idx, num_insertions) in violating_cols {
                msg.push_str(&format!(
                    "  Column {col_idx}: {num_insertions} insertions\n"
                ));
            }
            bail!(AncestralAlignment, msg);
        }
        Ok(())
    }

    /// Converts an ancestral alignment (with sequences for all tree nodes) into an alignment
    /// that contains only the leaf sequences, discarding the ancestral sequences. Columns that
    /// become all-gap are removed.
    ///
    /// # Example
    /// ```
    /// use phyne::alignment::{Alignment, AncestralAlignment, MASA, MSA, Sequences};
    /// use phyne::phylo_info::PhyloInfo;
    /// use phyne::{record_wo_desc, tree};
    /// # use phyne::Result;
    ///
    /// # fn main() -> Result<()> {
    /// let tree = tree!("((A:1.0,B:1.0)I:1.0,C:1.0)R:1.0;");
    /// let records = vec![
    ///     record_wo_desc!("A", b"A---C-T"),
    ///     record_wo_desc!("B", b"---CG-T"),
    ///     record_wo_desc!("C", b"A-----T"),
    ///     record_wo_desc!("I", b"A-ACG-T"),
    ///     record_wo_desc!("R", b"A-ACG-T"),
    /// ];
    /// let phylo = MASA::from_aligned_with_ancestral(Sequences::new(records)?, tree)?;
    ///
    /// // columns 2 and 5 are only gaps, so they are removed during the creation
    /// assert_eq!(phylo.msa.len(), 5);
    ///
    /// let phylo: PhyloInfo<MSA> = phylo.discard_ancestral_seqs();
    ///
    /// assert_eq!(phylo.msa.seq_count(), 3);
    /// // column 3 is only gaps in the leaf sequences, so it is removed during the transformation
    /// assert_eq!(phylo.msa.len(), 4);
    /// assert_eq!(phylo.msa.seqs().record_by_id("A").seq(), b"ACT");
    /// # Ok(()) }
    /// ```
    pub fn discard_ancestral_seqs<A: Alignment>(self) -> PhyloInfo<A> {
        let mut leaf_records = Vec::with_capacity(self.msa.leaf_maps().len());
        for (node_idx, map) in self.msa.leaf_maps() {
            let id = self.tree.node_id(node_idx);
            let unaligned_record = self.msa.seqs().record_by_id(id);
            let desc = unaligned_record.desc();
            let aligned_seq = aligned_seq!(map, unaligned_record.seq());
            let new_record = record!(id, desc, &aligned_seq);
            leaf_records.push(new_record);
        }
        let mut seqs = Sequences::with_alphabet_unchecked(leaf_records, self.msa.seqs().alphabet());
        seqs.remove_gap_cols();
        A::from_aligned_unchecked(seqs, self.tree)
    }

    /// Removes leaves whose aligned sequence consists only of gaps from both the tree and the
    /// alignment, collapsing single-child nodes that arise.
    ///
    /// # Example
    /// ```
    /// use phyne::alignment::{Alignment, AncestralAlignment, MASA, Sequences};
    /// use phyne::phylo_info::PhyloInfo;
    /// use phyne::{record_wo_desc, tree};
    /// # use phyne::Result;
    ///
    /// # fn main() -> Result<()> {
    /// let tree = tree!("(((A:1.0,B:2.0)I:3.0,C:4.0)R:5.0);");
    /// let records = vec![
    ///     record_wo_desc!("A", b"GTGGA---"),
    ///     record_wo_desc!("B", b"-----NNA"),
    ///     record_wo_desc!("I", b"T-------"),
    ///     record_wo_desc!("C", b"--------"),
    ///     record_wo_desc!("R", b"A-------"),
    /// ];
    /// let phylo = MASA::from_aligned_with_ancestral(Sequences::new(records)?, tree)?;
    ///
    /// // C consists only of gaps, so it is pruned along with its parent R,
    /// // and the root collapses to I.
    /// let phylo = phylo.prune_empty_leaves()?;
    /// let PhyloInfo { msa, tree } = phylo;
    ///
    /// assert_eq!(msa.seq_count(), 2); // A, B
    /// assert_eq!(msa.ancestral_seqs().len(), 1); // I
    /// assert_eq!(tree.leaves().len(), 2);
    /// assert_eq!(tree.node_id(&tree.root), "I");
    /// assert_eq!(tree.by_id("A").blen, 1.0);
    /// assert_eq!(tree.by_id("B").blen, 2.0);
    /// assert_eq!(tree.by_id("I").blen, 8.0); // R's branch (5.0) + C's branch (4.0) transfers to I
    /// assert!(tree.try_idx("C").is_err());
    /// assert!(tree.try_idx("R").is_err());
    /// # Ok(()) }
    /// ```
    ///
    /// # Errors
    /// * If pruning would leave fewer than two sequences with characters, which is not a
    ///   meaningful tree.
    pub fn prune_empty_leaves(self) -> Result<Self> {
        // Collect the leaves whose aligned sequence consists only of gaps. For each of them
        // the parent node is also removed from the MASA when the tree is pruned.
        let leaves_to_remove = self.all_gap_leaves();
        if leaves_to_remove.is_empty() {
            return Ok(self);
        }
        if self.tree.n - leaves_to_remove.len() < 2 {
            bail!(
                Alignment,
                "pruning empty leaves would leave fewer than two sequences with characters which is not a meaningful tree"
            );
        }

        let aligned = self.aligned_seqs();
        let pruned_tree = prune_empty_leaves_from_tree(self.tree, &leaves_to_remove)
            .unwrap_or_else(|e| bug!("Pruning empty leaves failed", e));
        let original_len = self.msa.len();
        let phylo: PhyloInfo<AA> = rebuild_masa(&aligned, pruned_tree)
            .unwrap_or_else(|e| bug!("Rebuilding MASA after pruning empty leaves failed", e));
        debug_assert_eq!(phylo.msa.len(), original_len);
        // Pruning only removes rows, not columns, so the fragmentation stays in sync.
        Ok(phylo)
    }

    /// Returns the leaves whose aligned sequence consists only of gaps.
    fn all_gap_leaves(&self) -> HashSet<NodeIdx> {
        self.msa
            .leaf_maps()
            .iter()
            .filter(|(_, leaf_map)| leaf_map.iter().all(Option::is_none))
            .map(|(leaf_idx, _)| *leaf_idx)
            .collect()
    }

    /// Returns all sequences (leaf and ancestral) in aligned form, mirroring
    /// [`Display`](`std::fmt::Display`) for [`crate::alignment::MASA`].
    /// Node ids are resolved through the tree, since `idx_to_id` is private to the alignment module.
    fn aligned_seqs(&self) -> Sequences {
        let both_maps = self
            .msa
            .leaf_maps()
            .iter()
            .chain(self.msa.ancestral_maps().iter());
        let records = both_maps
            .map(|(node_idx, seq_map)| {
                let id = self.tree.node_id(node_idx);
                let record = match node_idx {
                    Internal(_) => self.msa.ancestral_seqs().record_by_id(id),
                    Leaf(_) => self.msa.seqs().record_by_id(id),
                };
                let aligned = crate::aligned_seq!(seq_map, record.seq());
                crate::record!(id, record.desc(), &aligned)
            })
            .collect();
        Sequences::with_alphabet_unchecked(records, self.msa.seqs().alphabet())
    }
}

/// Removes the given leaves from the tree, collapsing any single-child nodes that arise
/// and transferring their branch length to the surviving child. Also collapses the root if
/// it ends up with a single child.
///
/// After calling this function, make sure to call [`rebuild_masa`] to keep the alignment in sync
/// with the pruned tree.
///
/// # Errors
/// * If any of the given nodes is not a leaf.
/// * If removing yields a tree with fewer than two leaves.
/// * Something went wrong while removing a leaf or collapsing the root, which should not happen less there is a bug.
fn prune_empty_leaves_from_tree(tree: Tree, leaves_to_remove: &HashSet<NodeIdx>) -> Result<Tree> {
    let mut pruned = tree.clone();
    for leaf_idx in leaves_to_remove {
        remove_leaf(&mut pruned, leaf_idx)?;
    }
    collapse_root(&mut pruned);
    rebuild_tree(&pruned)
}

/// Deletes a leaf from the tree, collapsing its parent (transferring the parent's branch
/// length to the surviving child). The root, if left with a single child, is collapsed by
/// [`collapse_root`]. Relies on the tree being binary.
///
/// After calling this function, make sure to call [`collapse_root`] and then [`rebuild_tree`] to
/// get a clean tree with correct internal fields and indices.
///
/// # Errors
/// * If the given node is not a leaf.
/// * If the given leaf is the only node of the tree.
fn remove_leaf(tree: &mut Tree, leaf_idx: &NodeIdx) -> Result<()> {
    if !matches!(leaf_idx, Leaf(_)) {
        bail!(
            Tree,
            "cannot prune node '{}', it is not a leaf",
            tree.node(leaf_idx).id
        );
    }
    let parent = match tree.parent(leaf_idx) {
        Some(parent) => parent,
        None => bail!(
            Tree,
            "cannot prune leaf '{}', it is the only node of the tree",
            tree.node(leaf_idx).id
        ),
    };
    // Disconnect the leaf from its parent
    tree.node_mut(&parent)
        .children
        .retain(|child| child != leaf_idx);

    // In a binary tree the parent of a pruned leaf is left with exactly one child;
    // collapse it and transfer its branch length to the surviving child.
    if parent != tree.root {
        debug_assert_eq!(tree.node(&parent).children.len(), 1);
        let grandparent = tree.node(&parent).parent.unwrap();
        let child = tree.node(&parent).children[0];
        let parent_blen = tree.node(&parent).blen;
        {
            let child_node = tree.node_mut(&child);
            child_node.blen += parent_blen;
            child_node.parent = Some(grandparent);
        }
        {
            let grandparent_node = tree.node_mut(&grandparent);
            grandparent_node.children.retain(|c| *c != parent);
            grandparent_node.children.push(child);
        }
    }
    Ok(())
}

/// Collapses the root if it has only a single child, transferring its branch length to that child
/// which then becomes the new root. Loops until the root has more than one child or is a leaf.
fn collapse_root(tree: &mut Tree) {
    loop {
        let children = tree.children(&tree.root).clone();
        match children.len() {
            1 => {
                let child = children[0];
                let root_blen = tree.node(&tree.root).blen;
                {
                    let child_node = tree.node_mut(&child);
                    child_node.blen += root_blen;
                    child_node.parent = None;
                }
                tree.root = child;
            }
            _ => break,
        }
    }
}

/// Rebuilds a clean tree from a (possibly mutated) tree, dropping any nodes that are no
/// longer connected to the root. Serialization to Newick and re-parsing re-computes all
/// indices, leaf ids, and traversals.
///
/// You should call [`rebuild_masa`] after this to keep the alignment in sync with the rebuilt tree.
fn rebuild_tree(pruned: &Tree) -> Result<Tree> {
    let newick = pruned.to_newick();
    // TODO: this does reallocation, could also work on the vecs in place
    let mut trees = from_newick(&newick)?;
    debug_assert_eq!(trees.len(), 1);
    Ok(trees.pop().unwrap())
}

/// Rebuilds an ancestral alignment that only contains the nodes of the new (pruned) tree,
/// keeping the aligned sequences of the remaining nodes. Columns are left untouched.
fn rebuild_masa<AA: AncestralAlignment>(
    aligned: &Sequences,
    new_tree: Tree,
) -> Result<PhyloInfo<AA>> {
    // We first must get the aligned seqs since just by using the new_tree and the original masa
    // we cannot have access to the mappings as they are indexed through idx of the original tree.
    let records: Vec<Record> = aligned
        .into_iter()
        .filter(|record| new_tree.try_idx(record.id()).is_ok())
        .cloned()
        .collect();
    let seqs = Sequences::with_alphabet(records, aligned.alphabet())?;
    validate_ids_with_ancestors(&new_tree, &seqs)?;
    Ok(AA::from_aligned_with_ancestral_unchecked(seqs, new_tree))
}

#[cfg(test)]
#[cfg_attr(coverage, coverage(off))]
mod tests;
