use std::cell::RefCell;
use std::fmt::{Debug, Display};
use std::marker::{PhantomData, Sized};
use std::ops::Mul;
use std::vec;

use hashbrown::HashMap;
use lazy_static::lazy_static;
use nalgebra::{DMatrix, DVector};

use crate::alignment::{Alignment, Mapping};
use crate::alphabets::{Alphabet, GAP};
use crate::evolutionary_models::EvoModel;
use crate::likelihood::{ModelSearchCost, ParamRange, TreeSearchCost, PARAM_RANGE_POSITIVE};
use crate::phylo_info::PhyloInfo;
use crate::substitution_models::{FreqVector, QMatrix, QMatrixMaker, SubstMatrix};
use crate::tree::{
    NodeIdx::{self, Internal as Int, Leaf},
    Tree,
};
use crate::{bail, Result};

/// (2.0 * PI).ln() / 2.0 = 0.9189385332046727;
pub static SHIFT: f64 = 0.9189385332046727;

lazy_static! {
    pub static ref MINLOGPROB: f64 = (f64::MIN_POSITIVE).ln();
}

fn log_factorial_shifted(n: usize) -> f64 {
    // An approximation using Stirling's formula, minus constant log(sqrt(2*PI)).
    // Has an absolute error of 3e-4 for n=2, 3e-6 for n=10, 3e-9 for n=100.
    let n = n as f64;
    n * n.ln() - n + n.ln() / 2.0 + 1.0 / (12.0 * n) + SHIFT
}

#[derive(Debug, Clone, PartialEq)]
pub struct PIPModel<Q: QMatrix> {
    pub(crate) subst_q: Q,
    q: SubstMatrix,
    freqs: FreqVector,
    params: Vec<f64>,
}

const DEFAULT_PIP_PARAM: f64 = 1.5;
const PIP_PARAM_N: usize = 2;

fn pip_q(q: &mut SubstMatrix, subst_q: &SubstMatrix, mu: f64) {
    let n = subst_q.ncols();
    q.view_mut((0, 0), (n, n)).copy_from(subst_q);
    q.fill_column(n, mu);
    for i in 0..(n + 1) {
        q[(i, i)] -= mu;
    }
}

impl<Q: QMatrix> PIPModel<Q> {
    fn from_substitution_with_params(subst_q: Q, pip_parameters: Vec<f64>) -> Self {
        let n = subst_q.n();
        let freqs = subst_q.freqs().clone().insert_row(n, 0.0);
        let mut q = SubstMatrix::zeros(n + 1, n + 1);
        pip_q(&mut q, subst_q.q(), pip_parameters[1]);
        let mut params = pip_parameters.clone();
        params.extend(subst_q.params());
        PIPModel {
            subst_q,
            q,
            freqs,
            params,
        }
    }

    fn lambda(&self) -> f64 {
        self.params[0]
    }

    fn mu(&self) -> f64 {
        self.params[1]
    }

    fn param_range(&self, _: usize) -> ParamRange {
        PARAM_RANGE_POSITIVE
    }
}

impl<Q: QMatrix + Default> Default for PIPModel<Q> {
    fn default() -> Self {
        let subst_q = Q::default();
        let params = vec![DEFAULT_PIP_PARAM; PIP_PARAM_N];
        Self::from_substitution_with_params(subst_q, params)
    }
}

impl<Q: QMatrix + Default> PIPModel<Q> {
    pub fn with_default_substitution(params: &[f64]) -> Result<Self> {
        // Only 2 parameters are expected as the substitution model uses defaults
        if params.len() != PIP_PARAM_N {
            bail!(
                EvolutionaryModel,
                ParameterCount,
                "PIP",
                PIP_PARAM_N,
                params.len()
            );
        }

        let subst_q = Q::default();
        Ok(Self::from_substitution_with_params(
            subst_q,
            params[..PIP_PARAM_N].to_vec(),
        ))
    }
}

impl<Q: QMatrix + QMatrixMaker> PIPModel<Q> {
    pub fn new(frequencies: &[f64], params: &[f64]) -> Result<Self>
    where
        Self: Sized,
    {
        // At least 2 parameters are expected as substitution models can vary in the number of additional parameters
        if params.len() < PIP_PARAM_N {
            bail!(
                EvolutionaryModel,
                ParameterCount,
                "PIP",
                PIP_PARAM_N,
                params.len()
            );
        }

        let subst_q = Q::create(frequencies, &params[PIP_PARAM_N..])?;
        Ok(Self::from_substitution_with_params(
            subst_q,
            params[..PIP_PARAM_N].to_vec(),
        ))
    }
}

// TODO: where is this ever used?
// See issue #119 https://github.com/acg-team/rust-phylo/issues/119
impl<Q: QMatrix> EvoModel for PIPModel<Q> {
    fn p(&self, time: f64) -> SubstMatrix {
        (self.q().clone() * time).exp()
    }

    fn q(&self) -> &SubstMatrix {
        &self.q
    }

    fn rate(&self, i: u8, j: u8) -> f64 {
        let n = self.subst_q.n();
        if i == GAP {
            self.q[(n, 0)]
        } else if j == GAP {
            self.q[(0, n)]
        } else {
            self.subst_q.rate(i, j)
        }
    }

    fn freqs(&self) -> &FreqVector {
        &self.freqs
    }

    fn set_freqs(&mut self, pi: FreqVector) -> Result<()> {
        self.subst_q.set_freqs(pi)?;
        self.freqs
            .view_mut((0, 0), self.subst_q.freqs().shape())
            .copy_from(self.subst_q.freqs());
        pip_q(&mut self.q, self.subst_q.q(), self.params[1]);
        Ok(())
    }

    fn params(&self) -> &[f64] {
        &self.params
    }

    fn set_param(&mut self, param: usize, value: f64) {
        match param {
            0 => {
                self.params[0] = value;
            }
            1 => {
                self.params[1] = value;
                pip_q(&mut self.q, self.subst_q.q(), self.params[1]);
            }
            _ => {
                self.params[param] = value;
                self.subst_q.set_param(param - 2, value);
                pip_q(&mut self.q, self.subst_q.q(), self.params[1]);
            }
        }
    }

    fn n(&self) -> usize {
        self.subst_q.n() + 1
    }

    fn alphabet() -> &'static Alphabet {
        Q::alphabet()
    }
}

impl<Q: QMatrix + Display> Display for PIPModel<Q> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "PIP with [lambda = {:.5}, mu = {:.5}]\n and {}",
            self.lambda(),
            self.mu(),
            self.subst_q
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PIPModelInfo<Q: QMatrix> {
    phantom: PhantomData<Q>,
    surv_ins_weights: Vec<f64>,
    anc: Vec<DMatrix<f64>>,
    ftilde: Vec<DMatrix<f64>>,
    f: Vec<DVector<f64>>,
    pnu: Vec<DVector<f64>>,
    c0_f1: Vec<f64>,
    c0_pnu: Vec<f64>,
    valid: Vec<bool>,
    models: Vec<SubstMatrix>,
    models_valid: Vec<bool>,
    leaf_seq_info: HashMap<NodeIdx, DMatrix<f64>>,
}

impl<Q: QMatrix> PIPModelInfo<Q> {
    pub fn new<M: Alignment>(info: &PhyloInfo<M>, model: &PIPModel<Q>) -> Result<Self> {
        let n = model.q().nrows();
        let node_count = info.tree.len();
        let msa_length = info.msa.len();
        let mut leaf_seq_info = HashMap::with_capacity(info.tree.leaves().len());
        for node in info.tree.leaves() {
            let seq = info.msa.seqs().record_by_id(&node.id).seq().to_vec();
            let alignment_map = info.msa.leaf_map(&node.idx);
            let mut leaf_seq_w_gaps = DMatrix::<f64>::zeros(n, msa_length);
            for (i, mut site_info) in leaf_seq_w_gaps.column_iter_mut().enumerate() {
                if let Some(c) = alignment_map[i] {
                    let encoding = info
                        .msa
                        .alphabet()
                        .char_encoding(seq[c])
                        .clone()
                        .insert_row(n - 1, 0.0);
                    site_info.copy_from(&encoding);
                } else {
                    site_info.fill_row(n - 1, 1.0);
                }
                site_info.scale_mut((1.0) / site_info.sum());
            }
            leaf_seq_info.insert(node.idx, leaf_seq_w_gaps);
        }
        Ok(PIPModelInfo {
            phantom: PhantomData,
            ftilde: vec![DMatrix::<f64>::zeros(n, msa_length); node_count],
            surv_ins_weights: vec![0.0; node_count],
            f: vec![DVector::<f64>::zeros(msa_length); node_count],
            pnu: vec![DVector::<f64>::zeros(msa_length); node_count],
            c0_f1: vec![0.0; node_count],
            c0_pnu: vec![0.0; node_count],
            anc: vec![DMatrix::<f64>::zeros(msa_length, 3); node_count],
            valid: vec![false; node_count],
            models: vec![SubstMatrix::zeros(n, n); node_count],
            models_valid: vec![false; node_count],
            leaf_seq_info,
        })
    }
}

pub struct PIPCostBuilder<Q: QMatrix, A: Alignment> {
    model: PIPModel<Q>,
    info: PhyloInfo<A>,
}

impl<Q: QMatrix, A: Alignment> PIPCostBuilder<Q, A> {
    pub fn new(model: PIPModel<Q>, info: PhyloInfo<A>) -> Self {
        PIPCostBuilder { model, info }
    }

    pub fn build(self) -> Result<PIPCost<Q, A>> {
        if self.info.msa.alphabet() != Q::alphabet() {
            bail!(Alphabet, "alphabet mismatch between model and alignment");
        }

        let tmp = RefCell::new(PIPModelInfo::new(&self.info, &self.model).unwrap());
        Ok(PIPCost {
            model: self.model,
            info: self.info,
            tmp,
        })
    }
}

#[derive(Debug, Clone)]
pub struct PIPCost<Q: QMatrix, A: Alignment> {
    pub(crate) model: PIPModel<Q>,
    pub(crate) info: PhyloInfo<A>,
    tmp: RefCell<PIPModelInfo<Q>>,
}

impl<Q: QMatrix, A: Alignment> TreeSearchCost for PIPCost<Q, A> {
    fn cost(&self) -> f64 {
        self.logl()
    }

    fn update_tree(&mut self, tree: Tree) {
        self.info.tree = tree;
        for idx in self.info.tree.dirty.ones() {
            self.tmp.borrow_mut().valid[idx] = false;
            self.tmp.borrow_mut().models_valid[idx] = false;
        }
        self.info.tree.clean();
    }

    fn tree(&self) -> &Tree {
        &self.info.tree
    }
}

impl<Q: QMatrix, M: Alignment> ModelSearchCost for PIPCost<Q, M> {
    fn cost(&self) -> f64 {
        self.logl()
    }

    fn param_count(&self) -> usize {
        self.model.params.len()
    }

    fn param(&self, param: usize) -> f64 {
        self.model.params()[param]
    }

    fn set_param(&mut self, param: usize, value: f64) {
        self.model.set_param(param, value);
        self.tmp.borrow_mut().models_valid.fill(false);
    }

    fn param_range(&self, param: usize) -> ParamRange {
        self.model.param_range(param)
    }

    fn set_freqs(&mut self, freqs: FreqVector) -> Result<()> {
        self.model.set_freqs(freqs)?;
        self.tmp.borrow_mut().models_valid.fill(false);
        Ok(())
    }

    fn empirical_freqs(&self) -> FreqVector {
        self.info.freqs()
    }

    fn freqs(&self) -> &FreqVector {
        self.model.freqs()
    }
}

impl<Q: QMatrix + Display, M: Alignment> Display for PIPCost<Q, M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.model)
    }
}

impl<Q: QMatrix, M: Alignment> PIPCost<Q, M> {
    fn logl(&self) -> f64 {
        for node_idx in self.info.tree.postorder() {
            match node_idx {
                Int(_) => {
                    if self.info.tree.root == *node_idx {
                        self.set_root(&self.info.tree, node_idx);
                    } else {
                        self.set_internal(&self.info.tree, node_idx);
                    }
                }
                Leaf(_) => {
                    self.set_leaf(&self.info.tree, node_idx, self.info.msa.leaf_map(node_idx));
                }
            };
        }

        let tmp = self.tmp.borrow();
        let root_idx = usize::from(&self.info.tree.root);
        let msa_length = self.info.msa.len();

        // In certain scenarios (e.g. a completely unrelated sequence, see data/p105.msa.fa)
        // individual column probabilities become too close to 0.0 (become subnormal)
        // and the log likelihood becomes -Inf. This is mathematically reasonable, but during branch
        // length optimisation BrentOpt cannot handle it and proposes NaN branch lengths.
        // This is a workaround that sets the probability to the smallest possible positive float,
        // which is equivalent to restricting the log likelihood to f64::MIN.
        tmp.pnu[root_idx]
            .map(|x| {
                if x == 0.0 || x.is_subnormal() {
                    *MINLOGPROB
                } else {
                    x.ln()
                }
            })
            .sum()
            + tmp.c0_pnu[root_idx]
            - log_factorial_shifted(msa_length)
    }

    fn set_root(&self, tree: &Tree, node_idx: &NodeIdx) {
        self.set_model(tree, node_idx);
        let idx = usize::from(node_idx);
        if !self.tmp.borrow().valid[idx] {
            let mut tmp = self.tmp.borrow_mut();

            tmp.surv_ins_weights[idx] = self.model.lambda() / self.model.mu();
            tmp.anc[idx].fill_column(0, 1.0);
            drop(tmp);

            self.set_ftilde(tree, node_idx);
            self.set_ancestors(tree, node_idx);
            self.set_pnu(tree, node_idx);
            self.set_c0(tree, node_idx);

            let mut tmp = self.tmp.borrow_mut();
            tmp.valid[idx] = true;
        }
    }

    fn set_internal(&self, tree: &Tree, node_idx: &NodeIdx) {
        self.set_model(tree, node_idx);
        let idx = usize::from(node_idx);
        let node = tree.node(node_idx);
        if !self.tmp.borrow().valid[idx] {
            let mut tmp = self.tmp.borrow_mut();

            tmp.surv_ins_weights[idx] =
                Self::survival_insertion_weight(self.model.lambda(), self.model.mu(), node.blen);
            drop(tmp);

            self.set_ftilde(tree, node_idx);
            self.set_ancestors(tree, node_idx);
            self.set_pnu(tree, node_idx);
            self.set_c0(tree, node_idx);

            let mut tmp = self.tmp.borrow_mut();
            if let Some(parent_idx) = tree.parent(node_idx) {
                tmp.valid[usize::from(parent_idx)] = false;
            }
            tmp.valid[idx] = true;
        }
    }

    fn set_leaf(&self, tree: &Tree, node_idx: &NodeIdx, leaf_map: &Mapping) {
        self.set_model(tree, node_idx);
        let idx = usize::from(node_idx);
        let node = tree.node(node_idx);
        if !self.tmp.borrow().valid[idx] {
            let mut tmp = self.tmp.borrow_mut();
            tmp.surv_ins_weights[idx] =
                Self::survival_insertion_weight(self.model.lambda(), self.model.mu(), node.blen);
            for (i, c) in leaf_map.iter().enumerate() {
                if c.is_some() {
                    tmp.anc[idx][(i, 0)] = 1.0;
                }
            }

            tmp.ftilde[idx] = tmp.leaf_seq_info.get(node_idx).unwrap().clone();

            tmp.f[idx] = tmp.ftilde[idx]
                .tr_mul(self.model.freqs())
                .component_mul(&tmp.anc[idx].column(0));

            tmp.pnu[idx] = tmp.f[idx].clone() * tmp.surv_ins_weights[idx];
            tmp.c0_f1[idx] = -1.0;
            tmp.c0_pnu[idx] = -tmp.surv_ins_weights[idx];
            if let Some(parent_idx) = tree.parent(node_idx) {
                tmp.valid[usize::from(parent_idx)] = false;
            }
            tmp.valid[idx] = true;
        }
    }

    /// Dependent:
    /// - tmp.pnu of both children
    /// - tmp.anc of this node
    /// - tmp.f of this node
    /// - tmp.surv_ins_weights of this node
    ///
    /// Modifies:
    /// - tmp.pnu of this node
    fn set_pnu(&self, tree: &Tree, node_idx: &NodeIdx) {
        let children: Vec<usize> = tree.children(node_idx).iter().map(usize::from).collect();
        let idx = usize::from(node_idx);

        let tmp = self.tmp.borrow();
        let x_pnu = tmp.pnu[children[0]].clone();
        let y_pnu = tmp.pnu[children[1]].clone();
        let ancestors = tmp.anc[idx].clone();

        drop(tmp);

        let mut tmp = self.tmp.borrow_mut();

        // TODO: why clone?
        tmp.pnu[idx] =
            tmp.f[idx].clone().component_mul(&ancestors.column(0)) * tmp.surv_ins_weights[idx];
        tmp.pnu[idx] +=
            ancestors.column(1).component_mul(&x_pnu) + ancestors.column(2).component_mul(&y_pnu);
    }

    /// Dependent:
    /// - tmp.models for both children
    /// - tmp.ftilde for both children
    ///
    /// Modifies:
    /// - tmp.ftilde for this node
    /// - tmp.f for this node
    fn set_ftilde(&self, tree: &Tree, node_idx: &NodeIdx) {
        let children = tree.children(node_idx);
        let idx = usize::from(node_idx);
        let x_idx = usize::from(children[0]);
        let y_idx = usize::from(children[1]);
        let mut tmp = self.tmp.borrow_mut();
        tmp.ftilde[idx] = (&tmp.models[x_idx])
            .mul(&tmp.ftilde[x_idx])
            .component_mul(&(&tmp.models[y_idx]).mul(&tmp.ftilde[y_idx]));
        tmp.f[idx] = tmp.ftilde[idx].tr_mul(self.model.freqs());
    }

    fn set_model(&self, tree: &Tree, node_idx: &NodeIdx) {
        let idx = usize::from(node_idx);
        let node = tree.node(node_idx);
        if !self.tmp.borrow().valid[idx] || !self.tmp.borrow().models_valid[idx] {
            let mut tmp = self.tmp.borrow_mut();
            tmp.models[idx] = self.model.p(node.blen);
            tmp.models_valid[idx] = true;
            tmp.valid[idx] = false;
        }
    }

    /// Dependent:
    /// - tmp.anc of both children
    ///
    /// Modifies:
    /// - tmp.anc of this node
    fn set_ancestors(&self, tree: &Tree, node_idx: &NodeIdx) {
        let idx = usize::from(node_idx);
        let children: Vec<usize> = tree.children(node_idx).iter().map(usize::from).collect();
        let mut tmp = self.tmp.borrow_mut();

        // TODO: unnecessary clone of whole matrix
        let x_anc = tmp.anc[children[0]].clone();
        let y_anc = tmp.anc[children[1]].clone();

        tmp.anc[idx].set_column(1, &x_anc.column(0));
        tmp.anc[idx].set_column(2, &y_anc.column(0));
        tmp.anc[idx].set_column(0, &(x_anc.column(0) + y_anc.column(0)));
        for i in 0..tmp.anc[idx].nrows() {
            debug_assert!((0.0..=2.0).contains(&tmp.anc[idx][(i, 0)]));
            if tmp.anc[idx][(i, 0)] == 2.0 {
                tmp.anc[idx].fill_row(i, 0.0);
                tmp.anc[idx][(i, 0)] = 1.0;
            }
        }
    }

    /// Dependent:
    /// - blen of both children
    /// - tmp.c0_f1 of both children
    /// - tmp.c0_pnu of both children
    ///
    /// Modifies:
    /// - tmp.c0_f1 of this node
    /// - tmp.c0_pnu of this node
    fn set_c0(&self, tree: &Tree, node_idx: &NodeIdx) {
        let idx = usize::from(node_idx);
        let children: Vec<usize> = tree.children(node_idx).iter().map(usize::from).collect();
        let x_idx = children[0];
        let y_idx = children[1];

        let x_blen = tree.nodes[x_idx].blen;
        let y_blen = tree.nodes[y_idx].blen;

        let mut tmp = self.tmp.borrow_mut();
        let mu = self.model.mu();
        tmp.c0_f1[idx] = (1.0 + (-mu * x_blen).exp() * tmp.c0_f1[x_idx])
            * (1.0 + (-mu * y_blen).exp() * tmp.c0_f1[y_idx])
            - 1.0;

        tmp.c0_pnu[idx] =
            tmp.surv_ins_weights[idx] * tmp.c0_f1[idx] + tmp.c0_pnu[x_idx] + tmp.c0_pnu[y_idx];
    }

    fn survival_insertion_weight(lambda: f64, mu: f64, b: f64) -> f64 {
        // A function equal to old
        // nu * insertion_probability(tree_length, b, mu) * survival_probability(mu, b)
        lambda / mu * (1.0 - (-b * mu).exp())
    }
}

#[cfg(test)]
#[cfg_attr(coverage, coverage(off))]
mod tests;
