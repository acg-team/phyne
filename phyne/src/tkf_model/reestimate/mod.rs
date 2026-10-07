use fixedbitset::FixedBitSet;
use log::info;
use rand::{Rng, SeedableRng};

use crate::alignment::{AncestralAlignment, Mapping};
use crate::likelihood::TreeSearchCost;
use crate::phylo_info::PhyloInfo;
use crate::random::RandomGenerator;
use crate::tkf_model::reestimate::cache::{
    possible_assignments_of_edge, possible_del_or_not, prev_compatible_del_or_not,
};
use crate::tkf_model::{ln_i1, Event, TKFIndelCost, TKFIndelModelInfo, TKFModel};
use crate::tree::NodeIdx::{self, Internal, Leaf};
use crate::{bail, Result};

mod cache;

/// Size of the dynamic programming column: 2 (assignments) * 2 (deletion or not) ^ 5 (edges) = 128, see [`QuartetEdges`].
const DP_COL_SIZE: usize = 128;
const BACKTRACKING_INVALID: usize = DP_COL_SIZE + 1;
/// #{v1, v2, t2, t3, t4}, see [`QuartetEdges`].
const N_EDGES_IN_QUARTET: usize = 5;

/// Assignment of chars present/absent at `v1` and `v2` (see [`QuartetEdges`])
/// and at current [block](`super::TKFModel::get_blocks`).
type EdgeAssignment = (bool, bool);
// type EdgeAssignmentPossibilities = Vec<EdgeAssignment>;
type EdgeAssignmentPossibilities = &'static [EdgeAssignment];
/// Represents whether chars are present or absent for every [block](`super::TKFModel::get_blocks`)
/// for a given [node](`crate::tree::Node`).
type NodeSeq = FixedBitSet;
/// Represent whether the previous event on each edge in the [quartet](`QuartetEdges`) was a deletion or not.
type QuartetDelOrNot = [bool; N_EDGES_IN_QUARTET];
type QuartetDelOrNotPossibilities = Vec<QuartetDelOrNot>;
type QuartetEvents = [Event; N_EDGES_IN_QUARTET];

/// ```text
///       t1
///       |
///       v1
///      /  \
///     /    \
///    v2    t2
///   / \
///  /   \
/// t3   t4
/// ```
/// The ancestral wild card sequences for `v1` and `v2` are re-estimated. The assignments in the
/// dynamic programming are for `(v1, v2)`.
/// The edges in this sketch are directed downwards.
#[derive(Clone)]
struct QuartetEdges {
    edges: [NodeIdx; N_EDGES_IN_QUARTET],
    /// `t1_mapping` is `None` if `v1` is the root since then `t1` does not exist
    t1_mapping: Option<Mapping>,
    t2_mapping: Mapping,
    t3_mapping: Mapping,
    t4_mapping: Mapping,
}

impl QuartetEdges {
    fn default() -> Self {
        QuartetEdges {
            edges: [NodeIdx::Leaf(0); N_EDGES_IN_QUARTET],
            t1_mapping: None,
            t2_mapping: vec![],
            t3_mapping: vec![],
            t4_mapping: vec![],
        }
    }

    /// Panics if `v2` is the root or has no sibling.
    fn new(v2: &NodeIdx, cost: &TKFIndelCost<impl TKFModel, impl AncestralAlignment>) -> Self {
        let phylo = &cost.phylo;
        let tree = &cost.phylo.tree;
        let v1 = tree.node(v2).parent.unwrap();
        let t2 = tree.sibling(v2).unwrap();
        let children_of_v2 = &tree.node(v2).children;
        let t3 = children_of_v2[0];
        let t4 = children_of_v2[1];
        let t2_mapping = get_map_from_any_node(&phylo.msa, &t2).clone();
        let t3_mapping = get_map_from_any_node(&phylo.msa, &t3).clone();
        let t4_mapping = get_map_from_any_node(&phylo.msa, &t4).clone();
        let t1_mapping = if v1 != phylo.tree.root {
            let t1_idx = phylo.tree.node(&v1).parent.unwrap();
            Some(get_map_from_any_node(&phylo.msa, &t1_idx).clone())
        } else {
            None
        };

        QuartetEdges {
            edges: [v1, *v2, t2, t3, t4],
            t2_mapping,
            t3_mapping,
            t4_mapping,
            t1_mapping,
        }
    }

    fn edges(&self) -> &[NodeIdx; N_EDGES_IN_QUARTET] {
        &self.edges
    }

    fn v1(&self) -> &NodeIdx {
        &self.edges[0]
    }

    fn v2(&self) -> &NodeIdx {
        &self.edges[1]
    }

    fn t1_has_char(&self, site: usize) -> bool {
        // in case the t1_mapping is None, v1 is the root and t1 does not exist
        // therefore it cannot have a character
        self.t1_mapping
            .as_ref()
            .is_some_and(|mapping| mapping[site].is_some())
    }

    fn t2_has_char(&self, site: usize) -> bool {
        self.t2_mapping[site].is_some()
    }

    fn t3_has_char(&self, site: usize) -> bool {
        self.t3_mapping[site].is_some()
    }

    fn t4_has_char(&self, site: usize) -> bool {
        self.t4_mapping[site].is_some()
    }
}

#[derive(Debug, PartialEq)]
struct BackTrackingResult {
    v1_bitset: FixedBitSet,
    v2_bitset: FixedBitSet,
    logl: f64,
}

/// Reestimator for indel points in the ancestral alignment at an internal neighbouring node pair.
/// Calling [`EdgeSeqsReestimator::reestimate`] will re-estimate the ancestral sequences
/// of the node that is passed as argument and its parent node under the [`TKFModel`]
/// maximum likelihood criterion. The re-estimation of indel points can remove characters or add new
/// characters ([wild cards](`crate::alphabets::AMB_CHAR`)) to the ancestral sequences.
/// Can be used as an ASR refinement method if repeatedly called on all internal nodes (i.e. edges),
/// see the [example](#example) below.
/// It is also used after an NNI move was applied during tree inference to fix the
/// ancestral sequences of the affected nodes, see
/// [`crate::tkf_model::TKFIndelCost::update_tree`].
///
/// # Example
/// ```rust
/// # use phylo::Result;
/// # fn main() -> Result<()> {
/// use phylo::phylo_info::PhyloInfoBuilder;
/// use phylo::random::DefaultGenerator;
/// use phylo::tkf_model::{EdgeSeqsReestimator, TKF92IndelCostBuilder};
/// use phylo::tree::NodeIdx::{Internal, Leaf};
/// // Re-estimation for ASR refinement.
/// // The alignment below includes ancestral sequences for which the indel points
/// // will be refined.
/// let tree = "data/tkf/reestimate/tree.newick";
/// let msa = "data/tkf/reestimate/masa.fasta";
/// let phylo = PhyloInfoBuilder::with_attrs(msa, tree).build_with_ancestors()?;
/// let lambda = 0.9;
/// let mu = 1.0;
/// let r = 0.5;
/// let mut tkf92_indel_cost = TKF92IndelCostBuilder::new(&[lambda, mu, r], phylo)
///     .build()?;
/// let mut rng = DefaultGenerator::default();
/// let mut reestimator = EdgeSeqsReestimator::new(&mut tkf92_indel_cost, &mut rng);
///
/// for node in reestimator.phylo().tree.postorder().clone() {
///     if node == reestimator.phylo().tree.root {
///         continue;
///     }
///     match node {
///         Internal(_) => {
///             let new_cost = reestimator.reestimate(&node)?;
///             println!("Re-estimated sequences at node {node}, cost after re-estimation: {new_cost}",);
///         }
///         Leaf(_) => {}
///     }
/// }
/// println!(
///     "The re-estimated ancestral MSA is {}",
///     reestimator.phylo().msa
/// );
/// println!("Re-estimation complete.");
/// # Ok(()) }
/// ```
pub struct EdgeSeqsReestimator<'a, T: TKFModel, AA: AncestralAlignment, R: Rng + SeedableRng> {
    dp_table: Vec<[f64; DP_COL_SIZE]>,
    backtracking_table: Vec<[usize; DP_COL_SIZE]>,
    cost: &'a mut TKFIndelCost<T, AA>,
    quartet_edges: QuartetEdges,
    rng: &'a mut RandomGenerator<R>,
}

impl<'a, T, AA, R> EdgeSeqsReestimator<'a, T, AA, R>
where
    T: TKFModel,
    AA: AncestralAlignment,
    R: Rng + SeedableRng,
{
    /// Creates a new [`EdgeSeqsReestimator`] for the provided [`TKFIndelCost`].
    /// The reestimator can then be repeatedly used to [re-estimate](`EdgeSeqsReestimator::reestimate`)
    /// ancestral wild card sequences for different internal node pairs.
    pub fn new(
        cost: &'a mut TKFIndelCost<T, AA>,
        rng: &'a mut RandomGenerator<R>,
    ) -> EdgeSeqsReestimator<'a, T, AA, R> {
        let num_blocks = cost.model_info.borrow().blocks.len();
        EdgeSeqsReestimator {
            dp_table: vec![[f64::NEG_INFINITY; DP_COL_SIZE]; num_blocks],
            backtracking_table: vec![[BACKTRACKING_INVALID; DP_COL_SIZE]; num_blocks],
            cost,
            quartet_edges: QuartetEdges::default(),
            rng,
        }
    }

    pub fn phylo(&self) -> &PhyloInfo<AA> {
        &self.cost.phylo
    }

    /// Reestimate ancestral wildcard sequences at `v2_idx` and its parent.
    ///
    /// This method reestimates under the maximum [TKF](`TKFModel`) likelihood criterion the ancestral wildcard sequences
    /// associated with the given internal node `v2_idx` and its parent,
    /// while keeping all other sequences and tree fixed. See also
    /// [`EdgeSeqsReestimator::reestimate_unchecked`].
    ///
    /// # Errors
    /// Reestimation is only defined for non-root internal nodes.
    /// Accordingly, this method will return an error if this is not the case.
    ///
    /// # Returns
    /// On success, returns the resulting ln likelihood of the MASA given the tree after reestimation.
    pub fn reestimate(&mut self, v2_idx: &NodeIdx) -> Result<f64> {
        let v2_id = self.cost.phylo.tree.node(v2_idx).id.clone();
        if v2_idx == &self.cost.phylo.tree.root {
            bail!(
                EdgeSeqsReestimator,
                "reestimation can't be performed for the root '{v2_id}'"
            );
        }
        if let Leaf(_) = v2_idx {
            bail!(
                EdgeSeqsReestimator,
                "reestimation can't be performed for leaf node '{v2_id}'"
            );
        }
        Ok(self.reestimate_unchecked(v2_idx))
    }

    /// Reestimate ancestral wildcard sequences at `v2_idx` and its parent.
    ///
    /// This method reestimates under the maximum TKF likelihood criterion the ancestral wildcard sequences
    /// associated with the given internal node `v2_idx` and its parent,
    /// while keeping all other sequences and the tree fixed. In contrast to
    /// [`EdgeSeqsReestimator::reestimate`], this method does not perform any
    /// validity checks on `v2_idx`.
    ///
    /// # Panics
    /// Reestimation is only defined for non-root internal nodes that have
    /// a sibling. If this condition is violated, this method panics.
    ///
    /// # Returns
    /// Returns the resulting ln likelihood of the MASA given the tree after reestimation.
    pub fn reestimate_unchecked(&mut self, v2_idx: &NodeIdx) -> f64 {
        if !self
            .cost
            .model_info
            .borrow()
            .valid_for_reestimation
            .is_full()
        {
            info!("Reestimation can only be performed on a cost with valid_for_reestimation internal nodes tmp values. Making them valid now.");
            self.cost.logl();
        }

        // When re-estimating ancestral wild card sequences the tmp values
        // of the model info for all nodes in the quartet, but also for all nodes along the
        // path to the root are invalidated. However, not all tmp values are needed for
        // re-estimation if the tree does not change between re-estimation calls.
        // Therefore, we recompute the tmp values for only the quartet nodes and the root,
        // making them usable for further re-estimation calls. Node flags are still set to
        // false, such that tmp values are properly recomputed when the logl is called.
        self.prepare_for_dp(v2_idx);
        self.fill_dp_table();
        let backtrack_res = self.backtrack();
        self.set_invalid();
        self.update_mappings(&backtrack_res);
        debug_assert!(self.cost.phylo.check_dollos_constraint().is_ok());
        self.make_valid_for_further_reestimate_calls();
        backtrack_res.logl
    }

    /// Resets the DP and backtracking tables. Initialises the [`QuartetEdges`]. Removes the old
    /// quartet contributions from the root aggregated values.
    fn prepare_for_dp(&mut self, v2_idx: &NodeIdx) {
        let num_blocks = self.cost.model_info.borrow().blocks.len();
        for row in &mut self.dp_table {
            row.fill(f64::NEG_INFINITY);
        }
        for row in &mut self.backtracking_table {
            row.fill(BACKTRACKING_INVALID);
        }
        self.quartet_edges = QuartetEdges::new(v2_idx, self.cost);
        for edge in self.quartet_edges.edges() {
            self.cost.reset_cache(edge);
        }
        for block_id in 0..num_blocks {
            self.remove_old_quartet_event_factor_from_root(block_id);
            self.remove_old_quartet_eta_from_root(block_id);
        }
    }

    fn remove_old_quartet_event_factor_from_root(&self, block_id: usize) {
        let root_id = usize::from(self.cost.phylo.tree.root);
        let mut model_info = self.cost.model_info.borrow_mut();
        for node in self.quartet_edges.edges() {
            let x = model_info.ln_node_event_factor[(usize::from(*node), block_id)];
            model_info.ln_subtree_event_factor[(root_id, block_id)] -= x;
        }
    }

    fn remove_old_quartet_eta_from_root(&self, block_id: usize) {
        let root_id = usize::from(self.cost.phylo.tree.root);
        let mut model_info = self.cost.model_info.borrow_mut();
        for node in self.quartet_edges.edges() {
            let eta = model_info.node_eta[(usize::from(*node), block_id)];
            model_info.subtree_eta[(root_id, block_id)] -= eta;
        }
    }

    /// Sets the valid flags of the model info to `false` for all edges (i.e. nodes) in the
    /// [quartet](`QuartetEdges`). This ensures that the next time the logl is computed,
    /// the tmp values for these nodes are recomputed.
    fn set_invalid(&mut self) {
        let mut model_info = self.cost.model_info.borrow_mut();
        for edge in self.quartet_edges.edges() {
            model_info.valid.set(usize::from(*edge), false);
        }
    }

    fn update_mappings(&mut self, backtrack_res: &BackTrackingResult) {
        let block_lengths = &self.cost.model_info.borrow().block_lengths;
        let seq_len = self.cost.phylo.msa.len();
        let v1_mapping = mapping_from_node_seq(&backtrack_res.v1_bitset, block_lengths, seq_len);
        let v2_mapping = mapping_from_node_seq(&backtrack_res.v2_bitset, block_lengths, seq_len);
        let msa = &mut self.cost.phylo.msa;
        assert!(v1_mapping.len() == msa.len());
        assert!(v2_mapping.len() == msa.len());
        // The expect() are never get triggered, unless something is seriously wrong with the algo.
        msa.update_ancestral_map(self.quartet_edges.v1(), v1_mapping)
            .expect("Failed to update ancestral map for v1");
        msa.update_ancestral_map(self.quartet_edges.v2(), v2_mapping)
            .expect("Failed to update ancestral map for v2");
    }

    /// Updates the tmp values of the model info such that there are valid for further
    /// [re-estimation](`EdgeSeqsReestimator::reestimate`) calls.
    /// Assumes that the new mappings were already updated in the msa, see
    /// [`AncestralAlignment::update_ancestral_map`].
    fn make_valid_for_further_reestimate_calls(&mut self) {
        let num_blocks = self.cost.model_info.borrow().blocks.len();
        self.cost
            .model_info
            .borrow_mut()
            .previous_event_deletion
            .clear();
        let root_id = usize::from(self.cost.phylo.tree.root);
        for block_id in 0..num_blocks {
            for edge in self.quartet_edges.edges() {
                let event = self.cost.determine_event(edge, block_id);
                let ln_node_event_factor = self.cost.ln_event_factor(edge, event);
                let node_eta = self.cost.eta_for_non_root(edge, event);
                let mut model_info = self.cost.model_info.borrow_mut();
                if let Some(val) = self.cost.updated_previous_is_deletion(event) {
                    model_info
                        .previous_event_deletion
                        .set(usize::from(*edge), val);
                }
                model_info.ln_node_event_factor[(usize::from(edge), block_id)] =
                    ln_node_event_factor;
                model_info.node_eta[(usize::from(edge), block_id)] = node_eta;
                model_info.ln_subtree_event_factor[(root_id, block_id)] += ln_node_event_factor;
                model_info.subtree_eta[(root_id, block_id)] += node_eta;
            }
        }
        let mut model_info = self.cost.model_info.borrow_mut();
        for edge in self.quartet_edges.edges() {
            model_info
                .valid_for_reestimation
                .set(usize::from(edge), true);
        }
    }

    fn fill_dp_table(&mut self) {
        let n_blocks = self.cost.model_info.borrow().blocks.len();
        for block_id in 0..n_blocks {
            let mut found_at_least_one = false;
            let site = self.cost.model_info.borrow().blocks[block_id] - 1;
            for assignment in self.possible_assignments(site) {
                let events = self.event_for_assignment(assignment, block_id);
                let ln_event_prob = self.ln_integrated_root_event_prob(&events, block_id);
                let is_first_block = block_id == 0;

                for q_del_or_not in possible_del_or_not(
                    &events,
                    is_first_block,
                    &self.quartet_edges,
                    &self.cost.tree().root,
                ) {
                    let dp_index = bools_to_index(assignment, q_del_or_not);
                    if block_id == 0 {
                        self.dp_table[block_id][dp_index] = ln_event_prob;
                        found_at_least_one = true;
                        // Since we are at the first position, the `del_or_not` does not have a
                        // meaning, so we can just skip all other `del_or_not` combinations.
                        continue;
                    }
                    let Some((max_prev, argmax)) =
                        self.max_over_previous(q_del_or_not, &events, block_id)
                    else {
                        continue;
                    };

                    self.backtracking_table[block_id][dp_index] = argmax;
                    let root_id = usize::from(self.cost.phylo.tree.root);
                    // collect eta that corresponds to nodes outside of the quartet
                    let eta_for_block =
                        self.cost.model_info.borrow().subtree_eta[(root_id, block_id)];
                    self.dp_table[block_id][dp_index] = max_prev + eta_for_block + ln_event_prob;
                    found_at_least_one = true;
                }
            }
            // TODO: perhaps instead return any valid assignment that is compatible with Dollo's
            // constraint, and return -infinity in the reassignment method. If we have -infinity here,
            // then any valid assignment is -infinity.
            // See issue #153 https://github.com/acg-team/rust-phylo/issues/153
            assert!(
                found_at_least_one,
                "No valid assignments found for block_id = {block_id}, due to -inf logl"
            );
        }
    }

    /// Finds the max over previous `assignments` and `del_or_not` that lead to the provided
    /// `del_or_not`. Since if we have [`Event::Nothing`] we have to pass through the previous `del_or_not`.
    /// May return [`None`] since we consider all possible `del_or_not` that are
    /// compatible with the `current_events` even though some of these might
    /// not be reached since the previous possible assignment might not produce
    /// these `del_or_not` scenarios. See the implementation of [`EdgeSeqsReestimator::fill_dp_table`].
    ///
    /// # Arguments
    /// Takes a mutable reference to self to be able to use the random generator to break ties.
    // TODO: More sophisticated filtering could be done, but might add more complexity and is perhaps not worth it.
    // See issue #151 https://github.com/acg-team/rust-phylo/issues/151
    fn max_over_previous(
        &mut self,
        current_del_or_not: &QuartetDelOrNot,
        current_events: &QuartetEvents,
        block_id: usize,
    ) -> Option<(f64, usize)> {
        let mut max = f64::NEG_INFINITY;
        let mut argmaxes = Vec::new();

        // TODO: instead of recalculating the possible assignments it could be reused from the previous block
        // See issue #151 https://github.com/acg-team/rust-phylo/issues/151
        let previous_block = block_id - 1;
        let model_info = self.cost.model_info.borrow();
        let site = model_info.blocks[previous_block] - 1;
        for prev_assignment in self.possible_assignments(site) {
            // TODO: here it is not checked whether the `prev_del_or_not` matches the `prev_assignment`
            // which will lead to -inf which is then skipped.
            // See issue #151 https://github.com/acg-team/rust-phylo/issues/151
            for prev_del_or_not in prev_compatible_del_or_not(current_events, current_del_or_not) {
                let prev_dp_index = bools_to_index(prev_assignment, prev_del_or_not);
                let prev_gamma = self.dp_table[block_id - 1][prev_dp_index];
                if prev_gamma == f64::NEG_INFINITY {
                    continue;
                }
                let current =
                    prev_gamma + self.quartet_eta(current_events, prev_del_or_not, &model_info);
                if current > max {
                    max = current;
                    argmaxes.clear();
                    argmaxes.push(prev_dp_index);
                }
                if current == max {
                    argmaxes.push(prev_dp_index);
                }
            }
        }
        if argmaxes.is_empty() {
            debug_assert!(max == f64::NEG_INFINITY);
            None
        } else {
            let argmax = argmaxes[self.rng.random_range(0..argmaxes.len())];
            Some((max, argmax))
        }
    }

    fn quartet_eta(
        &self,
        events: &QuartetEvents,
        prev_events: &[bool],
        model_info: &TKFIndelModelInfo,
    ) -> f64 {
        for i in 0..N_EDGES_IN_QUARTET {
            if events[i] == Event::Insertion && prev_events[i] {
                let edge = &self.quartet_edges.edges()[i];
                return model_info.eta[usize::from(edge)];
            }
        }
        0.0
    }

    /// Computes the integrated event probability for the quartet given the events
    fn ln_integrated_root_event_prob(&self, events: &QuartetEvents, block_id: usize) -> f64 {
        let root_id = usize::from(self.cost.phylo.tree.root);
        let model_info = self.cost.model_info.borrow();
        let block_len = model_info.block_lengths[block_id];
        let mut x = model_info.ln_subtree_event_factor[(root_id, block_id)];
        x += self.ln_quartet_event_factor(events);
        self.cost.model.block_prob(x, block_len)
    }

    /// Computes the sum of ln event factor values for the nodes in the quartet for the provided events
    /// which correspond to an assignment of characters at `v1` and `v2` that is currently considered
    /// in the dynamic programming.
    fn ln_quartet_event_factor(&self, events: &QuartetEvents) -> f64 {
        let mut quartet_event_factor = 0.0;
        let model_info = self.cost.model_info.borrow();
        // Here it is assumed that the cache is already updated for all nodes in the quartet,
        // see `EdgeSeqsReestimator::prepare_for_dp`.
        for (i, node) in self.quartet_edges.edges().iter().enumerate() {
            let node_id = usize::from(*node);
            quartet_event_factor += match events[i] {
                Event::Insertion => model_info.ln_insertion[node_id],
                Event::Deletion => model_info.ln_n0[node_id],
                Event::Homolog => model_info.ln_h1[node_id],
                Event::Nothing => 0.0,
            };
        }
        quartet_event_factor
    }

    /// Based on whether there are chars at the "leaves" of the quartet finds
    /// all possible [assignment for v1, assignment for v2] combinations that
    /// follow Dollo's principle.
    fn possible_assignments(&self, site: usize) -> EdgeAssignmentPossibilities {
        let t1_has_char = self.quartet_edges.t1_has_char(site);
        let t2_has_char = self.quartet_edges.t2_has_char(site);
        let t3_has_char = self.quartet_edges.t3_has_char(site);
        let t4_has_char = self.quartet_edges.t4_has_char(site);
        possible_assignments_of_edge(t1_has_char, t2_has_char, t3_has_char, t4_has_char)
    }

    fn event_for_assignment(&self, assignment: &EdgeAssignment, block_id: usize) -> QuartetEvents {
        let site = self.cost.model_info.borrow().blocks[block_id] - 1;
        let mut events = [Event::Nothing; N_EDGES_IN_QUARTET];
        let v1_has_char = assignment.0;
        let v2_has_char = assignment.1;
        // edge (t1 = pa(v1) -> v1)
        events[0] = event_for_edge(v1_has_char, self.quartet_edges.t1_has_char(site));
        // edge (v1 = pa(v2) -> v2)
        events[1] = event_for_edge(v2_has_char, v1_has_char);
        // edge (v1 = pa(t2) -> t2)
        events[2] = event_for_edge(self.quartet_edges.t2_has_char(site), v1_has_char);
        // edge (v2 = pa(t3) -> t3)
        events[3] = event_for_edge(self.quartet_edges.t3_has_char(site), v2_has_char);
        // edge (v2 = pa(t4) -> t4)
        events[4] = event_for_edge(self.quartet_edges.t4_has_char(site), v2_has_char);
        events
    }

    fn backtrack(&mut self) -> BackTrackingResult {
        // prepare
        let n_blocks = self.cost.model_info.borrow().blocks.len();
        let mut v1_bitset = FixedBitSet::with_capacity(n_blocks);
        let mut v2_bitset = FixedBitSet::with_capacity(n_blocks);

        // start from the last block
        let (last_max, last_argmax) = self.max_of_last_col();
        let (assignment, _quartet_del_or_not) = index_to_bools(last_argmax);
        v1_bitset.set(n_blocks - 1, assignment.0);
        v2_bitset.set(n_blocks - 1, assignment.1);
        let mut came_from = self.backtracking_table[n_blocks - 1][last_argmax];
        // go back the path
        for block_id in (0..(n_blocks - 1)).rev() {
            if came_from == BACKTRACKING_INVALID {
                unreachable!("Backtracking table contains invalid value at block_id = {block_id}");
            }
            let (assignment, _) = index_to_bools(came_from);
            v1_bitset.set(block_id, assignment.0);
            v2_bitset.set(block_id, assignment.1);
            if block_id > 0 {
                came_from = self.backtracking_table[block_id][came_from];
            }
        }
        BackTrackingResult {
            v1_bitset,
            v2_bitset,
            logl: last_max + self.const_per_alignment(),
        }
    }

    /// Finds the maximum value in the last column of the DP table and its index.
    /// If there are multiple maxima, one is chosen at random.
    /// This is used to start the backtracking.
    ///
    /// # Arguments
    /// Takes a mutable reference to self to be able to use the random generator to break ties.
    fn max_of_last_col(&mut self) -> (f64, usize) {
        let n_blocks = self.cost.model_info.borrow().blocks.len();
        let mut max = f64::NEG_INFINITY;
        let mut max_indices = Vec::new();
        for (index, &value) in self.dp_table[n_blocks - 1].iter().enumerate() {
            if value > max {
                max = value;
                max_indices.clear();
                max_indices.push(index);
            } else if value == max {
                max_indices.push(index);
            }
        }
        let max_index = max_indices[self.rng.random_range(0..max_indices.len())];
        (max, max_index)
    }

    /// Computes the constant part of the log likelihood that is independent of the alignment and
    /// only depends on the tree and model parameters.
    fn const_per_alignment(&self) -> f64 {
        let l = self.cost.model.lambda();
        let m = self.cost.model.mu();
        let mut const_per_alignment: f64 = (1.0 - l / m).ln();
        let nodes = self.cost.phylo.tree.preorder().iter().skip(1); // skip root
        let model_info = self.cost.model_info.borrow();
        for node in nodes {
            const_per_alignment += ln_i1(l, model_info.ln_beta[usize::from(node)]);
        }
        const_per_alignment
    }
}

#[inline]
fn mapping_from_node_seq(node_seq: &NodeSeq, block_lens: &[usize], seq_len: usize) -> Mapping {
    debug_assert!(
        block_lens.iter().sum::<usize>() == seq_len,
        "Block lengths do not sum up to the sequence length."
    );
    let mut mapping = Vec::with_capacity(seq_len);
    let mut count = 0;
    for (i, &block_len) in block_lens.iter().enumerate() {
        for _ in 0..block_len {
            if node_seq.contains(i) {
                mapping.push(Some(count));
                count += 1;
            } else {
                mapping.push(None);
            }
        }
    }
    mapping
}

/// Converts the provided assignment and quartet del_or_not combination
/// into a unique index for the DP table. The DP algorithm calculates probabilities
/// for such an assignment and quartet del_or_not combination. To store these
/// results in a flat array, we need to convert the combination of booleans
/// into a unique index.
/// Is the inverse of [`index_to_bools`].
fn bools_to_index(assignment: &EdgeAssignment, q_del_or_not: &QuartetDelOrNot) -> usize {
    // Iterate over all booleans (i.e., assignment and del_or_not concatenated):
    // first shift the index to the left by 1 (multiply by 2)
    // then add 1 if the boolean is true, or 0 if it is false
    [assignment.0, assignment.1]
        .iter()
        .chain(q_del_or_not.iter())
        .fold(0, |index, &b| (index << 1) | (b as usize))
}

/// Converts the provided index as used by the DP table into the corresponding `assignment` and
/// quartet `del_or_not` combination.
/// Is the inverse of [`bools_to_index`].
/// During backtracking we only need to know the assignment.
fn index_to_bools(index: usize) -> (EdgeAssignment, QuartetDelOrNot) {
    let mut bits = index;
    let mut q_del_or_not = [false; N_EDGES_IN_QUARTET];

    // extract del_or_not booleans first
    for i in (0..N_EDGES_IN_QUARTET).rev() {
        q_del_or_not[i] = (bits & 1) != 0;
        bits >>= 1;
    }
    // then extract assignment booleans
    let assignment = ((bits & 2) != 0, (bits & 1) != 0);

    (assignment, q_del_or_not)
}

#[inline]
fn event_for_edge(node_has_char: bool, parent_has_char: bool) -> Event {
    match (node_has_char, parent_has_char) {
        (true, true) => Event::Homolog,
        (true, false) => Event::Insertion,
        (false, true) => Event::Deletion,
        (false, false) => Event::Nothing,
    }
}

fn get_map_from_any_node<'a, AA: AncestralAlignment>(
    msa: &'a AA,
    node: &'a NodeIdx,
) -> &'a Mapping {
    match node {
        Internal(_) => msa.ancestral_map(node),
        Leaf(_) => msa.leaf_map(node),
    }
}

#[cfg(test)]
#[cfg_attr(coverage, coverage(off))]
mod private_tests {
    use std::path::Path;

    use approx::assert_relative_eq;
    use rstest::rstest;

    use crate::alignment::{Alignment, Sequences, MASA};
    use crate::alphabets::Alphabet;
    use crate::phylo_info::PhyloInfoBuilder;
    use crate::random::{DefaultGenerator, FakeGenerator, FakeRng};
    use crate::tkf_model::{tests::setup_test_phylo, EdgeSeqsReestimator, TKF92IndelCostBuilder};
    use crate::{record_wo_desc as record, tree};

    use super::*;

    #[test]
    fn tkf_index_to_bools_and_back() {
        for i in 0..DP_COL_SIZE {
            let (assignment, del_or_not) = index_to_bools(i);
            let j = bools_to_index(&assignment, &del_or_not);
            assert_eq!(i, j);
        }
    }

    #[test]
    fn tkf_get_map_from_any_node() {
        let phylo = setup_test_phylo(Alphabet::dna());
        let msa = &phylo.msa;
        let leaf_node = phylo.tree.by_id("A1").idx;
        let internal_node = phylo.tree.by_id("I3").idx;

        let leaf_map = get_map_from_any_node(msa, &leaf_node);
        let expected_leaf_map = msa.leaf_map(&leaf_node);
        assert_eq!(leaf_map, expected_leaf_map);

        let internal_map = get_map_from_any_node(msa, &internal_node);
        let expected_internal_map = msa.ancestral_map(&internal_node);
        assert_eq!(internal_map, expected_internal_map);
    }

    #[rstest]
    #[case(true, true, Event::Homolog)]
    #[case(true, false, Event::Insertion)]
    #[case(false, true, Event::Deletion)]
    #[case(false, false, Event::Nothing)]
    fn tkf_event_for_edge(
        #[case] node_has_char: bool,
        #[case] parent_has_char: bool,
        #[case] expected: Event,
    ) {
        let result = event_for_edge(node_has_char, parent_has_char);
        assert_eq!(result, expected);
    }

    #[cfg(test)]
    fn confirm_quartet_edges_for_setup_test_phylo<T, AA, R>(
        reestimator: &EdgeSeqsReestimator<T, AA, R>,
    ) where
        T: TKFModel,
        AA: AncestralAlignment,
        R: Rng + SeedableRng,
    {
        assert_eq!(
            reestimator.quartet_edges.edges()[0],
            reestimator.phylo().tree.by_id("R5").idx
        ); // v1
        assert_eq!(
            reestimator.quartet_edges.edges()[1],
            reestimator.phylo().tree.by_id("I3").idx
        ); // v2
        assert_eq!(
            reestimator.quartet_edges.edges()[2],
            reestimator.phylo().tree.by_id("C4").idx
        ); // t2
        assert_eq!(
            reestimator.quartet_edges.edges()[3],
            reestimator.phylo().tree.by_id("A1").idx
        ); // t3
        assert_eq!(
            reestimator.quartet_edges.edges()[4],
            reestimator.phylo().tree.by_id("B2").idx
        ); // t4
    }

    #[rstest]
    // for every block test one of the four possible assignments. In these tests we don't care about Dollo's principle.
    #[case::first_block(0, (false, false), [Event::Nothing, Event::Nothing, Event::Insertion, Event::Nothing, Event::Nothing])]
    #[case::second_block(1, (true, false), [Event::Insertion, Event::Deletion, Event::Homolog, Event::Insertion, Event::Nothing])]
    #[case::thrid_block(2, (true, true), [Event::Insertion, Event::Homolog, Event::Deletion, Event::Homolog, Event::Deletion])]
    #[case::forth_block(3, (false, true), [Event::Nothing, Event::Insertion, Event::Nothing, Event::Deletion, Event::Homolog])]
    fn tkf_event_for_assignment(
        #[case] block_id: usize,
        #[case] assignment: EdgeAssignment,
        #[case] expected_events: QuartetEvents,
    ) {
        let phylo = setup_test_phylo(Alphabet::dna());
        let mut cost = TKF92IndelCostBuilder::new(&[0.4, 0.5, 0.8], phylo)
            .build()
            .unwrap();
        let rng = &mut FakeGenerator::default();
        let v2_idx = cost.phylo.tree.by_id("I3").idx;
        let mut reestimator = EdgeSeqsReestimator::new(&mut cost, rng);
        reestimator.prepare_for_dp(&v2_idx);
        confirm_quartet_edges_for_setup_test_phylo(&reestimator);
        let events = reestimator.event_for_assignment(&assignment, block_id);
        assert_eq!(events, expected_events);
    }

    #[test]
    fn tkf_backtrack() {
        let phylo = setup_test_phylo(Alphabet::dna());
        // the parameters here do not matter for the backtracking test
        let mut cost = TKF92IndelCostBuilder::new(&[0.4, 0.5, 0.8], phylo)
            .build()
            .unwrap();
        // FakeRng such that we can test tie-breaking (max value in last column) in backtracking
        let rng = &mut FakeGenerator::from_rng(FakeRng::from_f64_values(vec![0.1, 0.2]));

        let mut reestimator = EdgeSeqsReestimator::new(&mut cost, rng);
        // backtracking_table dimensions num_blocks = 4, DP_COL_SIZE = 128
        reestimator.backtracking_table[1][32] = 13;
        // a path that splits in the middle
        reestimator.backtracking_table[2][110] = 32;
        reestimator.backtracking_table[2][20] = 32;
        reestimator.backtracking_table[3][85] = 110;
        reestimator.backtracking_table[3][42] = 20;
        // to select the argmax at the end of backtracking
        let dp_last_col_logl = -5.0;
        reestimator.dp_table[3][85] = dp_last_col_logl;
        reestimator.dp_table[3][42] = dp_last_col_logl;

        // the 85 is selected here
        let backtrack_res = reestimator.backtrack();
        let indices = [13, 32, 110, 85];
        let mut expected_v1_bitset = FixedBitSet::with_capacity(4);
        let mut expected_v2_bitset = FixedBitSet::with_capacity(4);
        for (i, &idx) in indices.iter().enumerate() {
            let (first, second) = index_to_bools(idx).0;
            if first {
                expected_v1_bitset.insert(i);
            }
            if second {
                expected_v2_bitset.insert(i);
            }
        }
        assert_eq!(backtrack_res.v1_bitset, expected_v1_bitset);
        assert_eq!(backtrack_res.v2_bitset, expected_v2_bitset);
        assert_eq!(
            backtrack_res.logl,
            reestimator.const_per_alignment() + dp_last_col_logl
        );

        // the 42 is selected here
        let backtrack_res = reestimator.backtrack();
        let indices = [13, 32, 20, 42];
        let mut expected_v1_bitset = FixedBitSet::with_capacity(4);
        let mut expected_v2_bitset = FixedBitSet::with_capacity(4);
        for (i, &idx) in indices.iter().enumerate() {
            let (first, second) = index_to_bools(idx).0;
            if first {
                expected_v1_bitset.insert(i);
            }
            if second {
                expected_v2_bitset.insert(i);
            }
        }
        assert_eq!(backtrack_res.v1_bitset, expected_v1_bitset);
        assert_eq!(backtrack_res.v2_bitset, expected_v2_bitset);
        assert_eq!(
            backtrack_res.logl,
            reestimator.const_per_alignment() + dp_last_col_logl
        );
    }

    #[test]
    fn tkf_const_per_alignment() {
        let tree = tree!("(((A1:2.0,B2:2.0)I3:0.3,C4:2.0)R5:1.0);");
        let msa = MASA::from_aligned_with_ancestral(
            Sequences::new_unchecked(vec![
                record!("A1", b""),
                record!("B2", b""),
                record!("I3", b""),
                record!("C4", b""),
                record!("R5", b""),
            ]),
            &tree,
        )
        .unwrap();
        let phylo = PhyloInfo { msa, tree };
        let mut cost = TKF92IndelCostBuilder::new(&[0.4, 0.5, 0.8], phylo)
            .build()
            .unwrap();
        let logl = cost.logl(); // must be called to initialize the model_info, which is
                                // needed for const_per_alignment().
        let rng = &mut DefaultGenerator::default();
        let reestimator = EdgeSeqsReestimator::new(&mut cost, rng);
        assert_eq!(reestimator.const_per_alignment(), logl);
    }

    #[test]
    fn tkf_remove_and_add_back_quartet() {
        let phylo = setup_test_phylo(Alphabet::dna());
        let mut cost = TKF92IndelCostBuilder::new(&[0.4, 0.5, 0.8], phylo)
            .build()
            .unwrap();

        let rng = &mut DefaultGenerator::default();
        let mut reestimator = EdgeSeqsReestimator::new(&mut cost, rng);
        let original_logl = reestimator.cost.logl();
        assert_relative_eq!(original_logl, reestimator.cost.logl_from_root_model_info());
        let dummy_v2_idx = reestimator.cost.phylo.tree.by_id("I3").idx;
        reestimator.prepare_for_dp(&dummy_v2_idx);
        assert_ne!(reestimator.cost.logl_from_root_model_info(), original_logl);
        reestimator.make_valid_for_further_reestimate_calls();
        assert_relative_eq!(reestimator.cost.logl_from_root_model_info(), original_logl);
    }

    #[test]
    #[cfg_attr(feature = "ci_coverage", ignore)]
    fn tkf_remove_and_add_back_quartet_large_tree() {
        let dir = Path::new("data/tkf/reestimate/");
        let msa = dir.join("masa.fasta");
        let tree = dir.join("tree.newick");
        let phylo = PhyloInfoBuilder::with_attrs(msa, tree)
            .build_with_ancestors()
            .unwrap();

        let mut cost = TKF92IndelCostBuilder::new(&[1.0, 2.0, 0.3], phylo)
            .build()
            .unwrap();
        let rng = &mut DefaultGenerator::default();
        let mut reestimator = EdgeSeqsReestimator::new(&mut cost, rng);
        let original_logl = reestimator.cost.logl();
        assert_eq!(
            original_logl,
            reestimator.cost.logl_from_root_model_info(),
            "before removing quartet"
        );
        let v2_idx = reestimator.cost.phylo.tree.by_id("N312").idx;
        reestimator.prepare_for_dp(&v2_idx);
        assert_ne!(reestimator.cost.logl_from_root_model_info(), original_logl);
        reestimator.make_valid_for_further_reestimate_calls();
        assert_eq!(
            reestimator.cost.logl_from_root_model_info(),
            original_logl,
            "after adding back quartet"
        );
    }

    #[test]
    #[cfg_attr(feature = "ci_coverage", ignore)]
    fn tkf_remove_and_add_back_quartet_large_tree_child_of_root() {
        let dir = Path::new("data/tkf/reestimate/");
        let msa = dir.join("masa.fasta");
        let tree = dir.join("tree.newick");
        let phylo = PhyloInfoBuilder::with_attrs(msa, tree)
            .build_with_ancestors()
            .unwrap();

        let mut cost = TKF92IndelCostBuilder::new(&[1.0, 2.0, 0.3], phylo)
            .build()
            .unwrap();
        let rng = &mut DefaultGenerator::default();
        let mut reestimator = EdgeSeqsReestimator::new(&mut cost, rng);
        let original_logl = reestimator.cost.logl();
        assert_eq!(
            original_logl,
            reestimator.cost.logl_from_root_model_info(),
            "before removing quartet"
        );
        let v2_idx = reestimator.cost.phylo.tree.by_id("N380").idx;
        reestimator.prepare_for_dp(&v2_idx);
        assert_ne!(reestimator.cost.logl_from_root_model_info(), original_logl);
        reestimator.make_valid_for_further_reestimate_calls();
        assert_eq!(
            reestimator.cost.logl_from_root_model_info(),
            original_logl,
            "after adding back quartet"
        );
    }
}

#[cfg(test)]
#[cfg_attr(coverage, coverage(off))]
mod tests;

#[cfg(test)]
#[cfg_attr(coverage, coverage(off))]
mod brute_force_ancestors_tests;
