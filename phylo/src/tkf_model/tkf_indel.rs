use std::cell::RefCell;
use std::fmt::Display;

use approx::assert_relative_eq;
use fixedbitset::FixedBitSet;
use itertools::Itertools;
use lazy_static::lazy_static;
use nalgebra::{DMatrix, DVector};

use crate::alignment::AncestralAlignment;
use crate::likelihood::{ModelSearchCost, ParamRange, TreeSearchCost};
use crate::phylo_info::PhyloInfo;
use crate::random::FakeGenerator;
use crate::substitution_models::FreqVector;
use crate::tkf_model::reestimate::EdgeSeqsReestimator;
use crate::tree::NodeIdx::{self, Internal, Leaf};
use crate::tree::Tree;
use crate::{bail, Result, REPORT_ISSUES_URL};

lazy_static! {
    pub(super) static ref DUMMY_FREQS: DVector<f64> = DVector::<f64>::zeros(0);
}

pub(super) static DEFAULT_LAMBDA: f64 = 1.0;
pub(super) static DEFAULT_MU: f64 = 1.1;
pub(super) static DEFAULT_LAMBDA_MU_RATIO: f64 = 0.9;
pub(super) static DEFAULT_R: f64 = 0.5;

/// For the function [u]: only if the time is shorter than this threshold, we use the Taylor approximation.
static SHORT_TIME_U: f64 = 1e-5;
/// For the function [u]: the critical threshold used for condition 4. Only necessary for x86 CPUs.
static X86_CRITICAL_THRESHOLD_U: f64 = 1e-11;
/// For the function [ln_n0]: threshold for numerical issues
static NUMERICAL_ISSUE_N0_THRESHOLD: f64 = 1e-12;

/// Events that can happen on a branch in the TKF model.
#[derive(Copy, Clone, Debug, PartialEq)]
pub(super) enum Event {
    Insertion,
    Deletion,
    Homolog,
    Nothing,
}

/// Trait for TKF indel models (i.e., [TKF91IndelModel](`crate::tkf_model::TKF91IndelModel`),
/// [TKF92IndelModel](`crate::tkf_model::TKF92IndelModel`)).
#[allow(clippy::upper_case_acronyms)]
pub trait TKFModel: Clone + Display {
    // TODO: it might be better for model optimisation to have parameter lambda and scale s = mu/lambda,
    // because of the constraint that mu > lambda.
    // See issue #152 https://github.com/acg-team/rust-phylo/issues/152
    fn lambda(&self) -> f64;
    fn mu(&self) -> f64;
    /// [TKF91](crate::tkf_model::tkf91) has 2 parameters: `lambda` and `mu`, [TKF92](crate::tkf_model::tkf92)
    /// has 3 parameters: `lambda`, `mu` and `r`.
    /// The parameter `r` in [TKF92](crate::tkf_model::tkf92) is used to model the length distribution of inserted segments,
    /// i.e., in [`super::TKF92IndelModel::ln_insertion_factor_at_non_root`] and
    /// [`super::TKF92IndelModel::ln_insertion_factor_at_root`].
    fn params(&self) -> &[f64];
    fn set_param(&mut self, idx: usize, value: f64);
    fn param_range(&self, idx: usize) -> ParamRange;
    /// Returns the factor corresponding to an insertion event at the root.
    fn ln_insertion_factor_at_root(&self) -> f64;
    /// Returns the factor corresponding to an insertion event at a non-root node.
    fn ln_insertion_factor_at_non_root(&self, ln_beta: f64) -> f64;
    /// Given the subtree event factor for the root (i.e., the tree event factor)
    /// and the block length, returns the ln probability of the [block](`TKFModel::get_blocks`) under the model.
    fn block_prob(&self, ln_tree_event_factor: f64, block_len: usize) -> f64;
    /// For every block (i.e., an alignment slice) as determined by this method and factors
    /// corresponding to the evolutionary events in this block [`TKFModel::block_prob`] computes
    /// the ln probability of the block under the model.
    fn get_blocks<AA: AncestralAlignment>(&self, msa: &AA) -> Vec<usize>;
}

// TODO: link our paper once it is published. For now see original TKF92 paper: https://doi.org/10.1007/bf00163848
/// This struct holds intermediate values for the computation of the ln likelihood
/// of an ancestral alignment and tree under a TKF indel model, i.e., without substitutions.
/// The intermediate values are needed for re-alignment, which is not implemented yet.
/// See issue #150 https://github.com/acg-team/rust-phylo/issues/150
#[derive(Clone, Debug)]
pub(super) struct TKFIndelModelInfo {
    /// ln_node_event_factor[(node, block)] = the ln probability factor for the event
    /// on the edge above <node> for the block with id <block>.
    /// See [`TKFIndelCost`] and
    /// [`TKFIndelCost::ln_event_factor`].
    pub(super) ln_node_event_factor: DMatrix<f64>,
    /// ln_subtree_event_factor[(node, block)] = the sum of the ln event probability factors
    /// for all edges in the subtree rooted in <node> for the block with id <block>,
    /// including the edge above <node>.
    /// See [`TKFIndelCost::set_node_values`].
    pub(super) ln_subtree_event_factor: DMatrix<f64>,

    /// node_eta[(node, block)] = eta if the current event is an
    /// insertion and the previous one was a deletion, 0 otherwise.
    /// See [`TKFIndelCost::eta_for_non_root`].
    pub(super) node_eta: DMatrix<f64>,
    /// subtree_eta[(node, block)] = sum of node_eta for all nodes in the subtree rooted in <node>
    /// for the block with id <block>. Since we only have one insertion per column, at most one
    /// node in the subtree can contribute to this sum.
    pub(super) subtree_eta: DMatrix<f64>,

    /// ln_beta[node] = ln(beta(node.blen)), precomputed for each node.
    /// See [`ln_beta`] function.
    pub(super) ln_beta: Vec<f64>,
    /// ln_n0[node] = ln(n0(node.blen)), precomputed for each node.
    /// See [`ln_n0`] function.
    pub(super) ln_n0: Vec<f64>,
    /// ln_h1[node] = ln(h1(node.blen)), precomputed for each node.
    /// See [`ln_h1`] function.
    pub(super) ln_h1: Vec<f64>,
    /// ln_insertion[node], precomputed for each node.
    /// See [`TKFModel::ln_insertion_factor_at_root`] and [`TKFModel::ln_insertion_factor_at_non_root`].
    pub(super) ln_insertion: Vec<f64>,
    /// eta[node] = n1/ (n0 * lambda * beta(node.blen)), precomputed for each node.
    /// See [`eta`] function.
    pub(super) eta: Vec<f64>,

    /// The right exclusive interval borders of the blocks.
    /// See [`TKFModel::get_blocks`].
    pub(super) blocks: Vec<usize>,
    /// The lengths of the blocks.
    /// See [`get_block_lengths`].
    pub(super) block_lengths: Vec<usize>,

    /// previous_event_deletion[node] = true if the last event was a deletion for a that <node>.
    /// See [`TKFIndelCost::determine_event`].
    pub(super) previous_event_deletion: FixedBitSet,

    /// valid[node] = true if the intermediate values for that <node> are valid.
    pub(super) valid: FixedBitSet,
    /// valid_for_reestimation[node] = true if the intermediate values can be used for re-estimation.
    /// Since for re-estimation we don't need the subtree values for the internal nodes except the
    /// root. So if many re-estimations are done for a fixed tree and model, we can save time by not
    /// recomputing subtree values for internal nodes that are not the root.
    pub(super) valid_for_reestimation: FixedBitSet,
}

impl TKFIndelModelInfo {
    pub(super) fn new<AA: AncestralAlignment, T: TKFModel>(
        model: &T,
        phylo: &PhyloInfo<AA>,
    ) -> TKFIndelModelInfo {
        let blocks = model.get_blocks(&phylo.msa);
        let block_lengths = get_block_lengths(&blocks);
        let n_blocks = blocks.len();
        let n_nodes = phylo.tree.len();
        TKFIndelModelInfo {
            ln_node_event_factor: DMatrix::<f64>::zeros(n_nodes, n_blocks),
            ln_subtree_event_factor: DMatrix::<f64>::zeros(n_nodes, n_blocks),
            node_eta: DMatrix::<f64>::zeros(n_nodes, n_blocks),
            subtree_eta: DMatrix::<f64>::zeros(n_nodes, n_blocks),
            ln_beta: vec![0.0; n_nodes],
            ln_n0: vec![0.0; n_nodes],
            ln_h1: vec![0.0; n_nodes],
            ln_insertion: vec![0.0; n_nodes],
            eta: vec![0.0; n_nodes],
            blocks,
            block_lengths,
            previous_event_deletion: FixedBitSet::with_capacity(n_nodes),
            valid: FixedBitSet::with_capacity(n_nodes),
            valid_for_reestimation: FixedBitSet::with_capacity(n_nodes),
        }
    }
}

/// Computes the ln likelihood of an [ancestral alignment](`AncestralAlignment`)
/// and tree under a [TKF](`TKFModel`) indel model, i.e., without substitutions.
#[derive(Debug)]
pub struct TKFIndelCost<T: TKFModel, AA: AncestralAlignment> {
    pub(super) model: T,
    pub(super) phylo: PhyloInfo<AA>,
    pub(super) model_info: RefCell<TKFIndelModelInfo>,
}

impl<T: TKFModel, AA: AncestralAlignment + Clone> Clone for TKFIndelCost<T, AA> {
    fn clone(&self) -> Self {
        TKFIndelCost {
            model: self.model.clone(),
            phylo: self.phylo.clone(),
            model_info: RefCell::new(self.model_info.borrow().clone()),
        }
    }
}

impl<T: TKFModel, AA: AncestralAlignment> Display for TKFIndelCost<T, AA> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.model)
    }
}

impl<T: TKFModel, AA: AncestralAlignment> TKFIndelCost<T, AA> {
    pub(super) fn set_all_nodes(&self) {
        for node_idx in self.phylo.tree.postorder() {
            match node_idx {
                Internal(_) => {
                    if self.phylo.tree.root == *node_idx {
                        self.set_root();
                    } else {
                        self.set_non_root(node_idx);
                    }
                }
                Leaf(_) => {
                    self.set_non_root(node_idx);
                }
            };
        }
    }

    pub(super) fn logl(&self) -> f64 {
        self.set_all_nodes();
        self.logl_from_root_model_info()
    }

    pub(super) fn logl_from_root_model_info(&self) -> f64 {
        let lambda = self.model.lambda();
        let mu = self.model.mu();
        let root_id = usize::from(self.phylo.tree.root);
        let mut logl = (1.0 - lambda / mu).ln();
        let model_info = self.model_info.borrow();
        // for every node except the root
        for node in self.phylo.tree.preorder().iter().skip(1) {
            logl += ln_i1(lambda, model_info.ln_beta[usize::from(node)]);
        }
        for block_id in 0..model_info.blocks.len() {
            let block_len = model_info.block_lengths[block_id];
            logl += model_info.subtree_eta[(root_id, block_id)];
            let tree_event_factor = model_info.ln_subtree_event_factor[(root_id, block_id)];
            logl += self.model.block_prob(tree_event_factor, block_len);
        }
        logl
    }

    fn set_root(&self) {
        let root_idx = &self.phylo.tree.root;
        let root_id = usize::from(root_idx);
        if self.model_info.borrow().valid[root_id] {
            return;
        }
        self.reset_cache(root_idx);
        let n_blocks = self.model_info.borrow().blocks.len();
        for block_id in 0..n_blocks {
            let event = self.determine_event(root_idx, block_id);
            let node_event_factor = self.ln_event_factor(root_idx, event);
            let node_eta = 0.0;
            self.set_node_values(root_idx, block_id, node_event_factor, node_eta);
        }
        let mut model_info = self.model_info.borrow_mut();
        model_info.valid.set(root_id, true);
        model_info.valid_for_reestimation.set(root_id, true);
    }

    fn set_non_root(&self, node_idx: &NodeIdx) {
        let node_id = usize::from(node_idx);
        if self.model_info.borrow().valid[node_id] {
            return;
        }
        self.reset_cache(node_idx);
        let n_blocks = self.model_info.borrow().blocks.len();
        for block_id in 0..n_blocks {
            if block_id == 0 {
                self.model_info
                    .borrow_mut()
                    .previous_event_deletion
                    .set(usize::from(node_idx), false);
            }
            let event = self.determine_event(node_idx, block_id);
            let node_event_factor = self.ln_event_factor(node_idx, event);
            let node_eta = self.eta_for_non_root(node_idx, event);
            self.set_node_values(node_idx, block_id, node_event_factor, node_eta);
            if let Some(val) = self.updated_previous_is_deletion(event) {
                self.model_info
                    .borrow_mut()
                    .previous_event_deletion
                    .set(usize::from(node_idx), val);
            }
        }
        let mut model_info = self.model_info.borrow_mut();
        if let Some(parent_idx) = self.phylo.tree.parent(node_idx) {
            model_info.valid.set(usize::from(parent_idx), false);
        }
        model_info.valid.set(node_id, true);
        model_info.valid_for_reestimation.set(node_id, true);
    }

    pub(super) fn updated_previous_is_deletion(&self, event: Event) -> Option<bool> {
        match event {
            Event::Deletion => Some(true),
            Event::Insertion | Event::Homolog => Some(false),
            Event::Nothing => None,
        }
    }

    pub(super) fn reset_cache(&self, node_idx: &NodeIdx) {
        let node_id = usize::from(node_idx);
        let lambda = self.model.lambda();
        let mu = self.model.mu();
        let blen = self.phylo.tree.node(node_idx).blen;
        let ln_beta = ln_beta(lambda, mu, blen);
        let mut model_info = self.model_info.borrow_mut();

        if node_idx != &self.phylo.tree.root {
            // these four don't need to be set for the root, since these events cannot happen at the root
            model_info.ln_beta[node_id] = ln_beta;
            model_info.ln_n0[node_id] = ln_n0(mu, ln_beta);
            model_info.ln_h1[node_id] = ln_h1(lambda, mu, ln_beta, blen);
            model_info.eta[node_id] = eta(lambda, mu, ln_beta, blen);
        }

        model_info.ln_insertion[node_id] = if node_idx == &self.phylo.tree.root {
            self.model.ln_insertion_factor_at_root()
        } else {
            self.model.ln_insertion_factor_at_non_root(ln_beta)
        };
        model_info.previous_event_deletion.set(node_id, false);
        model_info.valid.set(node_id, false);
    }

    fn set_node_values(
        &self,
        node_idx: &NodeIdx,
        block_id: usize,
        node_event_factor: f64,
        node_eta: f64,
    ) {
        let node_id = usize::from(node_idx);
        let mut model_info = self.model_info.borrow_mut();
        model_info.ln_node_event_factor[(node_id, block_id)] = node_event_factor;
        model_info.node_eta[(node_id, block_id)] = node_eta;
        let mut substree_event_factor = node_event_factor;
        let mut subtree_eta = node_eta;
        for child in &self.phylo.tree.node(node_idx).children {
            let child_id = usize::from(child);
            substree_event_factor += model_info.ln_subtree_event_factor[(child_id, block_id)];
            subtree_eta += model_info.subtree_eta[(child_id, block_id)];
        }
        model_info.ln_subtree_event_factor[(node_id, block_id)] = substree_event_factor;
        model_info.subtree_eta[(node_id, block_id)] = subtree_eta;
    }

    /// Determines the event that happened on the edge above `node_idx` for the given `block_id`
    /// based on the [ancestral alignment](`AncestralAlignment`).
    pub(super) fn determine_event(&self, node_idx: &NodeIdx, block_id: usize) -> Event {
        // the presence or absence of characters is the same for all sites in a block
        // so we can just check the last site of the block
        let site = self.model_info.borrow().blocks[block_id] - 1;

        let parent_is_gap = match self.phylo.tree.node(node_idx).parent {
            Some(parent_idx) => match parent_idx {
                Internal(_) => self.phylo.msa.ancestral_map(&parent_idx)[site].is_none(),
                _ => unreachable!("The parent of a node cannot be a leaf."),
            },
            None => true, // root has no parent, so we treat the position as a gap, then if there is
                          // a character at the root the event will be determined as an insertion,
                          // which is correct under the TKF model
        };

        let current_is_gap = match node_idx {
            Internal(_) => self.phylo.msa.ancestral_map(node_idx)[site].is_none(),
            Leaf(_) => self.phylo.msa.leaf_map(node_idx)[site].is_none(),
        };
        if !parent_is_gap && current_is_gap {
            Event::Deletion
        } else if !parent_is_gap && !current_is_gap {
            Event::Homolog
        } else if parent_is_gap && !current_is_gap {
            Event::Insertion
        } else {
            Event::Nothing
        }
    }

    pub(super) fn ln_event_factor(&self, node_idx: &NodeIdx, event: Event) -> f64 {
        let node_id = usize::from(node_idx);
        match event {
            Event::Deletion => self.model_info.borrow().ln_n0[node_id],
            Event::Homolog => self.model_info.borrow().ln_h1[node_id],
            Event::Insertion => self.model_info.borrow().ln_insertion[node_id],
            Event::Nothing => 0.0,
        }
    }

    /// Returns eta if the current event is an insertion and the previous one was a deletion, 0 otherwise.
    /// See [`eta`] function.
    /// Since there can't be a deletion at the root (it has no parent),
    /// this function is only for non-root nodes.
    pub(super) fn eta_for_non_root(&self, node_idx: &NodeIdx, event: Event) -> f64 {
        let model_info = self.model_info.borrow();
        if matches!(event, Event::Insertion)
            && model_info.previous_event_deletion[usize::from(node_idx)]
        {
            model_info.eta[usize::from(node_idx)]
        } else {
            0.0
        }
    }
}

impl<T: TKFModel, AA: AncestralAlignment> ModelSearchCost for TKFIndelCost<T, AA> {
    fn cost(&self) -> f64 {
        self.logl()
    }

    fn param_count(&self) -> usize {
        self.model.params().len()
    }

    fn param(&self, idx: usize) -> f64 {
        self.model.params()[idx]
    }

    fn set_param(&mut self, idx: usize, value: f64) {
        self.model.set_param(idx, value);
        self.model_info.borrow_mut().valid.clear();
    }

    /// Returns the valid range for a model parameter [min, max], inclusive.
    /// Assumes that current parameter values are valid.
    fn param_range(&self, idx: usize) -> ParamRange {
        self.model.param_range(idx)
    }

    fn set_freqs(&mut self, _: FreqVector) -> Result<()> {
        bail!(
            EvolutionaryModel,
            "The TKF indel model does not support setting frequencies"
        );
    }

    fn empirical_freqs(&self) -> FreqVector {
        // At the time of writing this, this method is only used to set the frequencies of
        // the model, but the TKF92IndelCost does not have frequencies.
        self.phylo.freqs()
    }

    fn freqs(&self) -> &FreqVector {
        &DUMMY_FREQS
    }
}

impl<T: TKFModel, AA: AncestralAlignment> TreeSearchCost for TKFIndelCost<T, AA> {
    fn cost(&self) -> f64 {
        self.logl()
    }

    fn update_tree(&mut self, tree: Tree) {
        let mut dirty_nodes = vec![];
        let mut model_info = self.model_info.borrow_mut();
        for idx in tree.dirty.ones() {
            model_info.valid.set(idx, false);
            model_info.valid_for_reestimation.set(idx, false);
            dirty_nodes.push(idx);
        }
        drop(model_info);

        let update_due_to_nni = dirty_nodes.len() == 1 && {
            // check if children of the dirty node are different than before
            let mut previous_children = self.phylo.tree.nodes[dirty_nodes[0]]
                .children
                .iter()
                .collect_vec();
            let mut new_children = tree.nodes[dirty_nodes[0]].children.iter().collect_vec();
            previous_children.sort();
            new_children.sort();

            previous_children != new_children
        };
        self.phylo.tree = tree;
        if update_due_to_nni {
            let v2 = self.tree().nodes[dirty_nodes[0]].idx;
            // TODO: For now we use the FakeGenerator to have deterministic behavior, but in the future
            // we should use a proper RNG here. But pass the RNG from outside and not
            // create a new one each time, see issue #142 https://github.com/acg-team/rust-phylo/issues/142
            let rng = &mut FakeGenerator::default();
            let mut reestimator = EdgeSeqsReestimator::new(self, rng);
            let dp_logl = reestimator.reestimate_unchecked(&v2);
            assert_relative_eq!(dp_logl, self.logl(), epsilon = 1e-10);
        }
        self.phylo.tree.clean();
    }

    fn tree(&self) -> &Tree {
        &self.phylo.tree
    }
}

/// Returns `ln(1 - exp(x))` in a numerically stable way.
///
/// This function handles two regions to avoid precision loss or underflow:
/// 1. For `x < -ln(2)`, it uses `ln(1 - exp(x))` via `f64::ln_1p(-exp(x))`.
/// 2. For `x >= -ln(2)`, it uses `ln(-(exp(x) - 1))` via `ln(-exp_m1(x))`.
///
/// # Arguments
/// * `x` - must be non-positive (`x <= 0.0`). Usually a ln-probability value.
pub(crate) fn ln1mexp(x: f64) -> f64 {
    assert!(
        x <= 0.0,
        "ln1mexp is only defined for x <= 0 but x is {x}. \
        Please report this at {REPORT_ISSUES_URL}."
    );
    if x < -std::f64::consts::LN_2 {
        // x is small, therefore exp(x) is close to 0, so we use the stable formula for ln
        (-x.exp()).ln_1p()
    } else {
        // x might be close to 0, therefore we use the stable formula for exp
        (-x.exp_m1()).ln()
    }
}

/// Returns the value of `ln(beta(t))` for a branch of length/time `t`.
/// See the TKF papers.
pub(super) fn ln_beta(lambda: f64, mu: f64, time: f64) -> f64 {
    let expo = (lambda - mu) * time;
    let term1 = ln1mexp(expo);
    let ln_ratio = expo + lambda.ln() - mu.ln();
    let term2 = mu.ln() + ln1mexp(ln_ratio);
    term1 - term2
}

/// Returns the ln probability factor of a character being inserted to the right of the immortal link
/// along a branch of length `time`, i.e., at the very left of the sequence.
/// The `time` is also implicitly included in `beta`.
/// It is called `p''_1` in the TKF papers.
#[inline]
pub(super) fn ln_i1(lambda: f64, ln_beta: f64) -> f64 {
    let x = ln_beta + lambda.ln(); // ln(lambda * beta)
    ln1mexp(x)
}

/// Returns the ln probability factor of a homologous character surviving along a branch of length `time`.
/// The `time` is also implicitly included in `beta`.
/// It is called `p_1` in the TKF papers.
#[inline]
pub(super) fn ln_h1(lambda: f64, mu: f64, ln_beta: f64, time: f64) -> f64 {
    -mu * time + ln_i1(lambda, ln_beta)
}

/// Returns the ln probability factor of a character being deleted along a branch of length `time`.
/// It is called `p'_0` in the TKF papers.
/// The `time` is implicitly included in `beta`.
#[inline]
pub(super) fn ln_n0(mu: f64, ln_beta: f64) -> f64 {
    let x = mu.ln() + ln_beta;
    if x > 0.0 {
        assert!(
            x < NUMERICAL_ISSUE_N0_THRESHOLD,
            "ln_n0 ({}) is much larger than 0 but should be \
            at most slightly larger than 0 which may happen due to numerical issues. \
            Please report this at {REPORT_ISSUES_URL}.",
            x
        );
        0.0
    } else {
        x
    }
}

/// Returns the ln of the `n1 / (n0 * lambda * beta)`.
/// This is used in the case where an insertion follows a deletion,
/// since the event factors included `n0` for the deletion and `lambda * beta` for the insertion
/// but under the TKF model they are not independent and instead `n1` should be used.
/// `Eta` corrects for that.
pub(super) fn eta(l: f64, m: f64, ln_beta: f64, t: f64) -> f64 {
    if t == 0.0 {
        // In the limit of t -> 0 eta approaches -ln(2).
        // Even for very small t the calculation below is stable, just not for exactly t = 0
        return -std::f64::consts::LN_2;
    }
    u(l, m, t) + ln_i1(l, ln_beta) - ln_n0(m, ln_beta) - l.ln() - ln_beta
}

/// Returns ln(1 - e^{-m*t} - m*beta).
/// Is only used in [`crate::tkf_model::tkf_indel::eta`] and was just extracted to make the code cleaner.
pub(super) fn u(l: f64, m: f64, t: f64) -> f64 {
    // See https://github.com/MattesMrzik/tkf_mathematica for how this was found.
    let critical_condition_1 = (-l * t).exp() == 1.0;
    let critical_condition_2 = (-m * t).exp() == 1.0;
    let critical_condition_3 = ((l - m) * t).exp() == 1.0;
    // The tkf_numerical_test passed on macOS (M-series) with a threshold of 0.
    // However, the same tests failed on Linux x86. Using this threshold ensures they pass on both platforms.
    let critical_condition_4 =
        ((m - l) - m * (-l * t).exp() + l * (-m * t).exp()).abs() <= X86_CRITICAL_THRESHOLD_U;

    let critical_condition = critical_condition_1
        || critical_condition_2
        || critical_condition_3
        || critical_condition_4;

    if critical_condition && t < SHORT_TIME_U {
        // using the Taylor expansion around t = 0 and for small times t
        return l.ln() + m.ln() + 2.0 * t.ln() - 2.0f64.ln() + (-(l + 4.0 * m) * t / 3.0).ln_1p();
    }

    let term1 = (l - m) * t;
    let term2 = ((m - l) - m * (-l * t).exp() + l * (-m * t).exp()).ln();
    let term3 = -(m - l * ((l - m) * t).exp()).ln();
    term1 + term2 + term3
}

/// Given the right exclusive block borders, returns the lengths of the blocks.
/// For example, given [3, 5, 8], the block lengths are [3, 2, 3].
pub(super) fn get_block_lengths(blocks: &[usize]) -> Vec<usize> {
    let mut block_lens = vec![0; blocks.len()];
    for (i, block) in blocks.iter().enumerate() {
        block_lens[i] = if i == 0 {
            *block
        } else {
            block - blocks[i - 1]
        };
    }
    block_lens
}

#[cfg(test)]
#[cfg_attr(coverage, coverage(off))]
mod private_tests {

    use super::*;
    use crate::alphabets::Alphabet;
    use crate::tkf_model::tests::setup_test_phylo;
    use crate::tkf_model::TKF91IndelCostBuilder;
    use crate::tkf_model::TKF92IndelCostBuilder;
    use crate::tkf_model::TKFModel;

    #[cfg(test)]
    fn validate_lambda_mu(l: f64, m: f64, l_expected: f64, m_expected: f64) {
        let cost = TKF91IndelCostBuilder::new(&[l, m], setup_test_phylo(Alphabet::dna()))
            .build()
            .unwrap();
        assert_eq!(cost.model.lambda(), l_expected);
        assert_eq!(cost.model.mu(), m_expected);
        let cost = TKF92IndelCostBuilder::new(&[l, m, 0.1], setup_test_phylo(Alphabet::dna()))
            .build()
            .unwrap();
        assert_eq!(cost.model.lambda(), l_expected);
        assert_eq!(cost.model.mu(), m_expected);
    }

    #[cfg(test)]
    fn validate_r(r: f64, r_expected: f64) {
        let cost = TKF92IndelCostBuilder::new(&[1.0, 2.0, r], setup_test_phylo(Alphabet::dna()))
            .build()
            .unwrap();
        assert_eq!(cost.model.r(), r_expected);
    }

    #[test]
    fn tkf_validate_params_for_builder() {
        validate_lambda_mu(-1.0, -2.0, DEFAULT_LAMBDA, DEFAULT_MU);
        validate_lambda_mu(0.0, 2.0, DEFAULT_LAMBDA_MU_RATIO * 2.0, 2.0);
        validate_lambda_mu(2.0, -0.1, 2.0, 2.0 / DEFAULT_LAMBDA_MU_RATIO);
        validate_lambda_mu(2.0, 1.9999, 2.0, 2.0 / DEFAULT_LAMBDA_MU_RATIO);
        validate_lambda_mu(1.2, 1.21, 1.2, 1.21);
        validate_r(-0.5, DEFAULT_R);
        validate_r(0.0, DEFAULT_R);
        validate_r(1.0, DEFAULT_R);
        validate_r(1.5, DEFAULT_R);
        validate_r(0.1, 0.1);
    }
}
