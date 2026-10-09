use std::cell::RefCell;
use std::fmt::{Debug, Display};
use std::marker::{PhantomData, Sized};
use std::ops::Mul;

use hashbrown::HashMap;
use nalgebra::{DMatrix, DVector};

use crate::alignment::Alignment;
use crate::alphabets::Alphabet;
use crate::evolutionary_models::EvoModel;
use crate::likelihood::{ModelSearchCost, ParamRange, TreeSearchCost};
use crate::parsimony::{CostMatrix, DiagonalZeros, ParsimonyModel, Rounding};
use crate::phylo_info::PhyloInfo;
use crate::tree::{
    NodeIdx::{self, Internal, Leaf},
    Tree,
};
use crate::{bail, Result, MAX_BLEN};

pub mod dna_models;
pub use dna_models::*;
pub mod protein_models;
pub use protein_models::*;

pub type SubstMatrix = DMatrix<f64>;
pub type FreqVector = DVector<f64>;

#[macro_export]
macro_rules! frequencies {
    ($slice:expr) => {
        FreqVector::from_column_slice($slice)
    };
}

pub trait QMatrixMaker {
    fn create(frequencies: &[f64], params: &[f64]) -> Result<Self>
    where
        Self: Sized;
}

pub trait QMatrix: Debug + Clone + Display {
    fn q(&self) -> &SubstMatrix;
    fn rate(&self, i: u8, j: u8) -> f64;
    fn params(&self) -> &[f64];
    fn param_count() -> usize;
    fn param(&self, param: usize) -> f64 {
        self.params()[param]
    }
    fn set_param(&mut self, param: usize, value: f64);
    /// Returns the valid range for a model parameter [min, max], inclusive.
    fn param_range(&self, param: usize) -> ParamRange;
    fn freqs(&self) -> &FreqVector;
    fn set_freqs(&mut self, freqs: FreqVector) -> Result<()>;
    fn n(&self) -> usize;
    fn alphabet() -> &'static Alphabet;
}

#[derive(Debug, Clone, PartialEq)]
pub struct SubstModel<Q: QMatrix> {
    pub(crate) qmatrix: Q,
}

impl<Q: QMatrix + Default> Default for SubstModel<Q> {
    fn default() -> Self {
        SubstModel {
            qmatrix: Q::default(),
        }
    }
}

impl<Q: QMatrix + Display> Display for SubstModel<Q> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.qmatrix)
    }
}

impl<Q: QMatrix + QMatrixMaker> SubstModel<Q> {
    pub fn new(frequencies: &[f64], params: &[f64]) -> Result<Self>
    where
        Self: Sized,
    {
        Ok(SubstModel {
            qmatrix: Q::create(frequencies, params)?,
        })
    }
}

impl<Q: QMatrix> EvoModel for SubstModel<Q> {
    fn p(&self, time: f64) -> SubstMatrix {
        // If time > 1e10f64 the matrix exponentiation breaks and stops converging, but before it breaks
        // starting with 1e5f64 it converges to the same result.
        if time > MAX_BLEN {
            (self.q().clone() * MAX_BLEN).exp()
        } else {
            (self.q().clone() * time).exp()
        }
    }

    fn q(&self) -> &SubstMatrix {
        self.qmatrix.q()
    }

    fn rate(&self, i: u8, j: u8) -> f64 {
        self.qmatrix.rate(i, j)
    }

    fn params(&self) -> &[f64] {
        self.qmatrix.params()
    }

    fn set_param(&mut self, param: usize, value: f64) {
        self.qmatrix.set_param(param, value);
    }

    fn freqs(&self) -> &FreqVector {
        self.qmatrix.freqs()
    }

    fn set_freqs(&mut self, pi: FreqVector) -> Result<()> {
        self.qmatrix.set_freqs(pi)
    }

    fn n(&self) -> usize {
        self.qmatrix.n()
    }

    fn alphabet() -> &'static Alphabet {
        Q::alphabet()
    }
}

pub struct SubstitutionCostBuilder<Q: QMatrix, A: Alignment> {
    model: SubstModel<Q>,
    info: PhyloInfo<A>,
}

impl<Q: QMatrix, A: Alignment> SubstitutionCostBuilder<Q, A> {
    pub fn new(model: SubstModel<Q>, info: PhyloInfo<A>) -> Self {
        SubstitutionCostBuilder { model, info }
    }

    pub fn build(self) -> Result<SubstitutionCost<Q, A>> {
        if self.info.msa.alphabet() != Q::alphabet() {
            bail!(Alphabet, "alphabet mismatch between model and alignment");
        }

        let tmp = RefCell::new(SubstModelInfo::new(&self.info, &self.model).unwrap());
        Ok(SubstitutionCost {
            model: self.model,
            info: self.info,
            tmp,
        })
    }
}

#[derive(Debug)]
pub struct SubstitutionCost<Q: QMatrix, A: Alignment> {
    pub(crate) model: SubstModel<Q>,
    pub(crate) info: PhyloInfo<A>,
    tmp: RefCell<SubstModelInfo<Q>>,
}

impl<Q: QMatrix, A: Alignment> Clone for SubstitutionCost<Q, A> {
    fn clone(&self) -> Self {
        SubstitutionCost {
            model: self.model.clone(),
            info: self.info.clone(),
            tmp: RefCell::new(self.tmp.borrow().clone()),
        }
    }
}

impl<Q: QMatrix, A: Alignment> TreeSearchCost for SubstitutionCost<Q, A> {
    fn cost(&self) -> f64 {
        self.logl(&self.info)
    }

    fn update_tree(&mut self, tree: Tree) {
        self.info.tree = tree;
        for idx in self.info.tree.dirty.ones() {
            self.tmp.borrow_mut().node_info_valid[idx] = false;
            self.tmp.borrow_mut().node_models_valid[idx] = false;
        }
        self.info.tree.clean();
    }

    fn tree(&self) -> &Tree {
        &self.info.tree
    }
}

impl<Q: QMatrix, A: Alignment> ModelSearchCost for SubstitutionCost<Q, A> {
    fn cost(&self) -> f64 {
        self.logl(&self.info)
    }

    fn param_count(&self) -> usize {
        Q::param_count()
    }

    fn param(&self, param: usize) -> f64 {
        self.model.qmatrix.param(param)
    }

    fn set_param(&mut self, param: usize, value: f64) {
        self.model.set_param(param, value);
        self.tmp.borrow_mut().node_models_valid.fill(false);
    }

    fn param_range(&self, param: usize) -> ParamRange {
        self.model.qmatrix.param_range(param)
    }

    fn set_freqs(&mut self, freqs: FreqVector) -> Result<()> {
        self.model.set_freqs(freqs)?;
        self.tmp.borrow_mut().node_models_valid.fill(false);
        Ok(())
    }

    fn empirical_freqs(&self) -> FreqVector {
        self.info.freqs()
    }

    fn freqs(&self) -> &FreqVector {
        self.model.freqs()
    }
}

impl<Q: QMatrix, A: Alignment> Display for SubstitutionCost<Q, A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.model)
    }
}

impl<Q: QMatrix, A: Alignment> SubstitutionCost<Q, A> {
    fn logl(&self, info: &PhyloInfo<A>) -> f64 {
        for node_idx in info.tree.postorder() {
            match node_idx {
                Internal(_) => {
                    self.set_internal(&info.tree, node_idx);
                }
                Leaf(_) => {
                    self.set_leaf(&info.tree, node_idx);
                }
            };
        }
        let tmp_values = self.tmp.borrow();
        debug_assert_eq!(info.tree.len(), tmp_values.node_info.len());

        let likelihood = self
            .model
            .freqs()
            .transpose()
            .mul(&tmp_values.node_info[usize::from(&info.tree.root)]);
        drop(tmp_values);

        debug_assert_eq!(likelihood.ncols(), info.msa.len());
        debug_assert_eq!(likelihood.nrows(), 1);
        likelihood.map(|x| x.ln()).sum()
    }

    fn set_internal(&self, tree: &Tree, node_idx: &NodeIdx) {
        let node = tree.node(node_idx);
        let childx_info = self.tmp.borrow().node_info[usize::from(&node.children[0])].clone();
        let childy_info = self.tmp.borrow().node_info[usize::from(&node.children[1])].clone();

        let idx = usize::from(node_idx);

        let mut tmp_values = self.tmp.borrow_mut();
        if !tmp_values.node_info_valid[idx] || !tmp_values.node_models_valid[idx] {
            tmp_values.node_models[idx] = self.model.p(node.blen);
            tmp_values.node_models_valid[idx] = true;
            tmp_values.node_info_valid[idx] = false;
        }
        if tmp_values.node_info_valid[idx] {
            return;
        }
        tmp_values.node_info[idx] =
            (&tmp_values.node_models[idx]).mul(childx_info.component_mul(&childy_info));
        tmp_values.node_info_valid[idx] = true;
        if let Some(parent_idx) = node.parent {
            tmp_values.node_info_valid[usize::from(parent_idx)] = false;
        }
        drop(tmp_values);
    }

    fn set_leaf(&self, tree: &Tree, node_idx: &NodeIdx) {
        let mut tmp_values = self.tmp.borrow_mut();
        let node = tree.node(node_idx);
        let idx = usize::from(node_idx);

        if !tmp_values.node_info_valid[idx] || !tmp_values.node_models_valid[idx] {
            tmp_values.node_models[idx] = self.model.p(node.blen);
            tmp_values.node_models_valid[idx] = true;
            tmp_values.node_info_valid[idx] = false;
        }
        if tmp_values.node_info_valid[idx] {
            return;
        }

        let leaf_seq = tmp_values.leaf_seq_info.get(node_idx).unwrap();
        tmp_values.node_info[idx] = (&tmp_values.node_models[idx]).mul(leaf_seq);
        if let Some(parent_idx) = tree.parent(node_idx) {
            tmp_values.node_info_valid[usize::from(parent_idx)] = false;
        }
        tmp_values.node_info_valid[idx] = true;
        drop(tmp_values);
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SubstModelInfo<Q: QMatrix> {
    phantom: PhantomData<Q>,
    node_info: Vec<DMatrix<f64>>,
    node_info_valid: Vec<bool>,
    node_models: Vec<SubstMatrix>,
    node_models_valid: Vec<bool>,
    leaf_seq_info: HashMap<NodeIdx, DMatrix<f64>>,
}

impl<Q: QMatrix> SubstModelInfo<Q> {
    pub fn new<A: Alignment>(info: &PhyloInfo<A>, model: &SubstModel<Q>) -> Result<Self> {
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
                    site_info.copy_from(info.msa.alphabet().char_encoding(seq[c]));
                } else {
                    site_info.copy_from(info.msa.alphabet().missing_char_encoding());
                }
            }
            leaf_seq_info.insert(node.idx, leaf_seq_w_gaps);
        }
        Ok(SubstModelInfo::<Q> {
            phantom: PhantomData,
            node_info: vec![DMatrix::<f64>::zeros(n, msa_length); node_count],
            node_info_valid: vec![false; node_count],
            node_models: vec![SubstMatrix::zeros(n, n); node_count],
            node_models_valid: vec![false; node_count],
            leaf_seq_info,
        })
    }
}

impl<Q: QMatrix> ParsimonyModel for SubstModel<Q> {
    fn scoring(
        &self,
        time: f64,
        diagonal_zeros: &DiagonalZeros,
        rounding: &Rounding,
    ) -> CostMatrix {
        let mut scores = self.p(time);
        scores.iter_mut().for_each(|x| *x = -(*x).ln());
        if rounding.yes() {
            scores.iter_mut().for_each(|x| {
                *x = (*x * 10.0_f64.powf(rounding.digits as f64)).round()
                    / 10.0_f64.powf(rounding.digits as f64)
            });
        }
        if diagonal_zeros.is_zero() {
            scores.fill_diagonal(0.0);
        }
        scores
    }
}

#[cfg(test)]
#[cfg_attr(coverage, coverage(off))]
mod tests;
