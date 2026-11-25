use std::cell::RefCell;
use std::fmt::{Debug, Display};
use std::marker::{PhantomData, Sized};
use std::ops::Mul;

use faer;
use hashbrown::HashMap;
use nalgebra::DMatrix;

use crate::alignment::Alignment;
use crate::alphabets::Alphabet;
use crate::evolutionary_models::{EvoModel, FreqVector, ProbabilityMatrix, RateMatrix};
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
pub mod simulate_msa;
pub use simulate_msa::*;

pub trait QMatrixMaker {
    fn create(frequencies: &[f64], params: &[f64]) -> Result<Self>
    where
        Self: Sized;
}

pub trait QMatrix: Debug + Clone + Display {
    fn q(&self) -> &RateMatrix;
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

#[derive(Clone, Debug, PartialEq)]
struct Decomposition {
    left: DMatrix<f64>,
    right: DMatrix<f64>,
    eigenvalues: FreqVector,
}

// Cyclic Jacobi rotations for the small symmetric matrices of reversible models.
// Each rotation updates the matrix and its eigenvector basis together.
// rotates coordinate axes until the matrix is diagonal
// preseres accuracacy for low numbers
fn jacobi_eigen(mut a: DMatrix<f64>) -> Option<(DMatrix<f64>, FreqVector)> {
    let n = a.nrows();
    let mut u = DMatrix::<f64>::identity(n, n);
    let scale = a.amax();
    for x in a.iter_mut() {
        *x /= scale;
    }

    //maximum of 64 sweeps
    for _ in 0..64 {
        let mut changed = false;
        for p in 0..n {
            for r in (p + 1)..n {
                let b = a[(p, r)];
                if b.abs() <= f64::EPSILON {
                    continue;
                }
                changed = true;
                let h = (a[(r, r)] - a[(p, p)]) * 0.5;
                let t = b / (h + h.hypot(b).copysign(h));
                let c = 1.0 / 1.0_f64.hypot(t);
                let s = t * c;
                a[(p, p)] -= t * b;
                a[(r, r)] += t * b;
                a[(p, r)] = 0.0;
                a[(r, p)] = 0.0;

                for i in 0..n {
                    if i != p && i != r {
                        let x = a[(i, p)];
                        let y = a[(i, r)];
                        let xp = c * x - s * y;
                        let yr = s * x + c * y;
                        a[(i, p)] = xp;
                        a[(p, i)] = xp;
                        a[(i, r)] = yr;
                        a[(r, i)] = yr;
                    }
                    let x = u[(i, p)];
                    let y = u[(i, r)];
                    u[(i, p)] = c * x - s * y;
                    u[(i, r)] = s * x + c * y;
                }
            }
        }
        if !changed {
            return Some((u, a.diagonal() * scale)); // return if change smaller epsilon
        }
    }
    None // convergence has failed
}

fn decomp<Q: QMatrix>(qmatrix: &Q) -> Option<Decomposition> {
    let freqs = qmatrix.freqs();
    if freqs.iter().any(|&pi| !pi.is_finite() || pi <= 0.0) {
        return None;
    }

    let q = qmatrix.q();
    let sqrt_pi = freqs.map(|pi| pi.sqrt());
    let n = freqs.len();
    let symmetric = DMatrix::from_fn(n, n, |i, j| q[(i, j)] * sqrt_pi[i] / sqrt_pi[j]);

    let matrix = faer::Mat::from_fn(n, n, |i, j| symmetric[(i, j)]);
    let eig =
        faer::linalg::solvers::SelfAdjointEigen::new(matrix.as_ref(), faer::Side::Lower).ok()?;

    let u = eig.U();
    let values = eig.S().column_vector();

    Some(Decomposition {
        left: DMatrix::from_fn(n, n, |i, k| u[(i, k)] / sqrt_pi[i]),
        right: DMatrix::from_fn(n, n, |k, j| u[(j, k)] * sqrt_pi[j]),
        eigenvalues: FreqVector::from_iterator(n, (0..n).map(|k| values[k])),
    })
}

#[derive(Debug, Clone, PartialEq)]
pub struct SubstModel<Q: QMatrix> {
    pub(crate) qmatrix: Q,
    decomposition: Option<Decomposition>,
}

impl<Q: QMatrix + Default> Default for SubstModel<Q> {
    fn default() -> Self {
        let qmatrix = Q::default();
        let decomposition = decomp(&qmatrix);

        Self {
            qmatrix,
            decomposition,
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
        let qmatrix = Q::create(frequencies, params)?;
        let decomposition = decomp(&qmatrix);
        Ok(Self {
            qmatrix,
            decomposition,
        })
    }
}

impl<Q: QMatrix> EvoModel for SubstModel<Q> {
    fn p(&self, time: f64) -> ProbabilityMatrix {
        if time == 0.0 {
            return ProbabilityMatrix::identity(self.n(), self.n());
        }
        let time = if time > MAX_BLEN { MAX_BLEN } else { time };
        // If time > 1e10f64 the matrix exponentiation breaks and stops converging, but before it breaks
        // starting with 1e5f64 it converges to the same result.
        let Some(decomposition) = &self.decomposition else {
            return (self.q().clone() * time).exp();
        };

        // Near the identity, reconstruct P - I to preserve small transitions.
        let near_identity = decomposition
            .eigenvalues
            .iter()
            .all(|&lambda| (lambda * time).abs() <= 0.5);
        let mut scaled_left = decomposition.left.clone();
        for (k, &lambda) in decomposition.eigenvalues.iter().enumerate() {
            let x = lambda * time;
            let factor = if near_identity { x.exp_m1() } else { x.exp() };
            scaled_left.column_mut(k).scale_mut(factor);
        }
        let mut transition = scaled_left * &decomposition.right;
        if near_identity {
            for i in 0..transition.nrows() {
                transition[(i, i)] += 1.0;
            }
        }
        transition
    }

    fn q(&self) -> &RateMatrix {
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
        self.decomposition = decomp(&self.qmatrix);
    }

    fn freqs(&self) -> &FreqVector {
        self.qmatrix.freqs()
    }

    fn set_freqs(&mut self, pi: FreqVector) -> Result<()> {
        self.qmatrix.set_freqs(pi)?;
        self.decomposition = decomp(&self.qmatrix);
        Ok(())
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

        let tmp = RefCell::new(SubstModelInfo::new(&self.info, &self.model)?);
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

        let root = usize::from(&info.tree.root);
        let likelihood = self
            .model
            .freqs()
            .transpose()
            .mul(&tmp_values.node_info[root]);

        debug_assert_eq!(likelihood.ncols(), info.msa.len());
        debug_assert_eq!(likelihood.nrows(), 1);
        likelihood
            .iter()
            .zip(&tmp_values.log_scales[root])
            .map(|(&p, &scale)| p.ln() + scale)
            .sum()
    }

    fn set_internal(&self, tree: &Tree, node_idx: &NodeIdx) {
        let node = tree.node(node_idx);
        let idx = usize::from(node_idx);
        let mut tmp_values = self.tmp.borrow_mut();
        if tmp_values.node_info_valid[idx] && tmp_values.node_models_valid[idx] {
            return;
        }
        if !tmp_values.node_models_valid[idx] {
            tmp_values.node_models[idx] = self.model.p(node.blen);
            tmp_values.node_models_valid[idx] = true;
        }
        let left = usize::from(&node.children[0]);
        let right = usize::from(&node.children[1]);
        let product = tmp_values.node_info[left].component_mul(&tmp_values.node_info[right]);
        let mut partial = &tmp_values.node_models[idx] * product;
        for (site, mut column) in partial.column_iter_mut().enumerate() {
            let scale = column.max();
            let inherited = tmp_values.log_scales[left][site] + tmp_values.log_scales[right][site];
            tmp_values.log_scales[idx][site] = if scale > 0.0 {
                for value in column.iter_mut() {
                    *value /= scale;
                }
                inherited + scale.ln()
            } else {
                // A zero column represents an impossible observation.
                inherited
            };
        }
        tmp_values.node_info[idx] = partial;
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

        if !tmp_values.node_models_valid[idx] {
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
    log_scales: Vec<Vec<f64>>,
    node_models: Vec<RateMatrix>,
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
            log_scales: vec![vec![0.0; msa_length]; node_count],
            node_models: vec![RateMatrix::zeros(n, n); node_count],
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

#[cfg(test)]
mod spectral_regression {
    use super::*;

    #[test]
    fn gtr_extreme_rates_reconstruct_the_generator_and_transition_matrix() {
        let model = SubstModel::<GTR>::new(
            &[0.25, 0.35, 0.3, 0.1],
            &[
                467.46771733483683,
                0.07044148894930388,
                1.053030992576237e-5,
                0.47937347791908513,
                0.2816632355494551,
            ],
        )
        .unwrap();
        let d = model
            .decomposition
            .as_ref()
            .expect("GTR spectral decomposition");
        let mut weighted = d.left.clone();
        for (k, &lambda) in d.eigenvalues.iter().enumerate() {
            weighted.column_mut(k).scale_mut(lambda);
        }
        let residual = (weighted * &d.right - model.q()).amax();
        assert!(residual < 1e-12 * model.q().amax(), "Q residual={residual}");

        let time = 0.056647;
        let direct = (model.q().clone() * time).exp();
        let spectral = model.p(time);
        assert!((spectral[(0, 2)] - 1.9091842082338955e-5).abs() < 1e-14);
        assert!((spectral - direct).amax() < 1e-13);
    }
}
