use std::cell::RefCell;
use std::fmt::Display;

use hashbrown::HashSet;
use log::warn;
use num_enum::{FromPrimitive, IntoPrimitive};

use crate::alignment::AncestralAlignment;
use crate::likelihood::{ParamRange, PARAM_RANGE_UNIT_INTERVAL_EXCLUSIVE};
use crate::phylo_info::PhyloInfo;
use crate::substitution_models::{QMatrix, SubstModel, SubstitutionCostBuilder as SCB};
use crate::tkf_model::{
    validate_lambda_mu, TKFCost, TKFIndelCost, TKFIndelModelInfo, TKFModel, DEFAULT_LAMBDA,
    DEFAULT_MU, DEFAULT_R,
};
use crate::Result;

#[derive(Debug, Eq, PartialEq, FromPrimitive, IntoPrimitive)]
#[repr(usize)]
pub(crate) enum TKF92Parameters {
    Lambda = 0,
    Mu = 1,
    R = 2,
    #[num_enum(catch_all)]
    Invalid(usize),
}

#[derive(Clone, Debug, PartialEq)]
pub struct TKF92IndelModel {
    params: Vec<f64>,
    /// precomputed r.ln()
    ln_r: f64,
    /// precomputed ((1 - r)/r).ln()
    ln_one_minus_r_over_r: f64,
}

impl TKF92IndelModel {
    pub fn r(&self) -> f64 {
        self.params[usize::from(TKF92Parameters::R)]
    }
}

impl Default for TKF92IndelModel {
    fn default() -> Self {
        let r = DEFAULT_R;
        Self {
            params: vec![DEFAULT_LAMBDA, DEFAULT_MU, r],
            ln_r: r.ln(),
            ln_one_minus_r_over_r: (-r).ln_1p() - r.ln(),
        }
    }
}

impl TKFModel for TKF92IndelModel {
    fn lambda(&self) -> f64 {
        self.params[usize::from(TKF92Parameters::Lambda)]
    }

    fn mu(&self) -> f64 {
        self.params[usize::from(TKF92Parameters::Mu)]
    }

    fn params(&self) -> &[f64] {
        &self.params
    }

    fn set_param(&mut self, idx: usize, value: f64) {
        let param = TKF92Parameters::from_primitive(idx);
        match param {
            TKF92Parameters::R => {
                self.params[usize::from(TKF92Parameters::R)] = value;
                self.ln_r = value.ln();
                self.ln_one_minus_r_over_r = (-value).ln_1p() - value.ln();
            }
            _ => {
                self.params[idx] = value;
            }
        };
    }

    fn param_range(&self, idx: usize) -> ParamRange {
        let param = TKF92Parameters::from_primitive(idx);
        match param {
            TKF92Parameters::Lambda => (f64::EPSILON, self.mu() - f64::EPSILON),
            TKF92Parameters::Mu => (self.lambda() + f64::EPSILON, f64::MAX),
            TKF92Parameters::R => PARAM_RANGE_UNIT_INTERVAL_EXCLUSIVE,
            _ => panic!("Invalid parameter index for TKF model: {param:?}"),
        }
    }

    fn ln_insertion_factor_at_root(&self) -> f64 {
        self.lambda().ln() - self.mu().ln() + self.ln_one_minus_r_over_r
    }

    fn ln_insertion_factor_at_non_root(&self, ln_beta: f64) -> f64 {
        // TODO: this lambda.ln() could be cached, see issue #152 https://github.com/acg-team/rust-phylo/issues/152
        self.lambda().ln() + ln_beta + self.ln_one_minus_r_over_r
    }

    fn block_prob(&self, ln_tree_event_factor: f64, block_len: usize) -> f64 {
        // TODO: For the underflow of exp(ln_tree_event_factor):
        // - True underflow (< -745): not a concern, f64 lacks precision at that scale anyway.
        //   (when adding to the other terms)
        // - Near machine epsilon (< -36): the approximation
        //     m * ln(1 + x) approx ln((1 + x)^m) approx ln(1 + m*x) approx m*x
        //   recovers log(m)  bits of precision, at the cost of two linearization
        //   errors. Whether the net gain is positive requires further investigation.
        //   See issue https://github.com/acg-team/rust-phylo/issues/174.
        ln_tree_event_factor
            + (block_len as f64 - 1.0) * (ln_tree_event_factor.exp()).ln_1p()
            + (block_len as f64) * self.ln_r
    }

    fn get_blocks<AA: AncestralAlignment>(&self, msa: &AA) -> Vec<usize> {
        blocks_of_alignment(msa)
    }
}

impl Display for TKF92IndelModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "TKF92 with lambda = {}, mu = {}, r = {}",
            self.lambda(),
            self.mu(),
            self.r(),
        )
    }
}

/// Validates the TKF92 parameter `r`. If it is not valid, it is set to
/// its default value and a warning is logged.
/// Returns valid `r`.
pub(super) fn validate_r(params: &mut [f64]) {
    let r_id = usize::from(TKF92Parameters::R);
    let r = params[r_id];
    if r == 0.0 {
        params[r_id] = DEFAULT_R;
        warn!(
            "Tried to set r to invalid value 0. \
            It must be in (0, 1). Setting r to {}. \
            Hint: r = 0 yields special case: TKF91 model, consider using that instead.",
            params[r_id]
        );
    } else if r <= 0.0 || r >= 1.0 {
        params[r_id] = DEFAULT_R;
        warn!(
            "Tried to set r to invalid value {r}. \
            It must be in (0, 1). Setting r to {}.",
            params[r_id]
        );
    }
}

/// Builder for the cost using the [`TKF92IndelModel`], i.e., without a substitution model.
pub struct TKF92IndelCostBuilder<AA: AncestralAlignment> {
    params: Vec<f64>,
    phylo: PhyloInfo<AA>,
}

impl<AA: AncestralAlignment> TKF92IndelCostBuilder<AA> {
    pub fn new(params: &[f64], phylo: PhyloInfo<AA>) -> Self {
        Self {
            params: params.to_vec(),
            phylo,
        }
    }

    pub fn build(self) -> Result<TKFIndelCost<TKF92IndelModel, AA>> {
        let mut params = self.params;
        if params.len() != 3 {
            warn!(
                "Expected 3 parameters for TKF92 model (lambda, mu, r), but got {}",
                params.len()
            );
            warn!("Falling back to default values");
            params.resize(3, 0.0);
            params[usize::from(TKF92Parameters::Lambda)] = DEFAULT_LAMBDA;
            params[usize::from(TKF92Parameters::Mu)] = DEFAULT_MU;
            params[usize::from(TKF92Parameters::R)] = DEFAULT_R;
        } else {
            validate_lambda_mu(&mut params);
            validate_r(&mut params);
        }
        let r = params[usize::from(TKF92Parameters::R)];
        let model = TKF92IndelModel {
            params,
            ln_r: r.ln(),
            ln_one_minus_r_over_r: ((1.0 - r) / r).ln(),
        };
        let info = TKFIndelModelInfo::new(&model, &self.phylo);
        Ok(TKFIndelCost {
            model,
            phylo: self.phylo.clone(),
            model_info: RefCell::new(info),
        })
    }
}

/// Builder for the TKF92 cost, i.e., with a substitution model.
pub struct TKF92CostBuilder<Q: QMatrix, AA: AncestralAlignment> {
    params: Vec<f64>,
    subst_model: SubstModel<Q>,
    phylo: PhyloInfo<AA>,
}

impl<Q: QMatrix, AA: AncestralAlignment> TKF92CostBuilder<Q, AA> {
    pub fn new(params: &[f64], subst_model: SubstModel<Q>, phylo: PhyloInfo<AA>) -> Self {
        Self {
            params: params.to_vec(),
            subst_model,
            phylo,
        }
    }

    pub fn build(self) -> Result<TKFCost<Q, TKF92IndelModel, AA>> {
        let indel_cost = TKF92IndelCostBuilder::new(&self.params, self.phylo.clone()).build()?;
        let subst_cost = SCB::new(self.subst_model, self.phylo).build()?;
        Ok(TKFCost {
            indel_cost,
            subst_cost,
        })
    }
}

/// Determines the block borders from the alignment. A block border is defined as a
/// position where any sequence changes from gap to non-gap or vice versa. Returns a sorted
/// vector of the right exclusive block borders.
pub(crate) fn blocks_of_alignment<AA: AncestralAlignment>(msa: &AA) -> Vec<usize> {
    let mut blocks: HashSet<usize> = HashSet::new();
    if msa.len() == 0 {
        return vec![];
    }
    for map in msa
        .ancestral_maps()
        .values()
        .chain(msa.leaf_maps().values())
    {
        let mut previous_is_char = map[0].is_some();
        for (i, c) in map.iter().skip(1).enumerate() {
            let current_is_char = c.is_some();
            // whenever there is a change from gap to not gap or vice versa, we have a block border
            if previous_is_char ^ current_is_char {
                blocks.insert(i + 1);
            }
            previous_is_char = current_is_char;
        }
        blocks.insert(map.len());
    }
    let mut blocks: Vec<usize> = blocks.iter().copied().collect();
    blocks.sort();
    blocks
}

#[cfg(test)]
#[cfg_attr(coverage, coverage(off))]
mod private_tests {
    use super::*;

    #[test]
    #[should_panic]
    fn tkf92_param_range_invalid_index() {
        let model = TKF92IndelModel {
            params: vec![0.5, 1.0, 0.3],
            ln_r: 0.0, // cache filled with dummy, since it is not needed here
            ln_one_minus_r_over_r: 0.0, // cache filled with dummy since it is not needed here
        };
        // Use an invalid index
        model.param_range(3);
    }

    #[test]
    fn tkf92_model_fmt() {
        let tkf_indel_model = TKF92IndelModel {
            params: vec![1.1, 2.0, 0.3],
            ln_r: 0.0,                  // cache filled with dummy since it is not printed
            ln_one_minus_r_over_r: 0.0, // cache filled with dummy since it is not printed
        };

        let fmt = format!("{}", tkf_indel_model);

        assert_eq!(fmt, "TKF92 with lambda = 1.1, mu = 2, r = 0.3");
    }

    #[test]
    fn tkf92_indel_set_param() {
        let mut model = TKF92IndelModel {
            params: vec![1.0, 2.0, 0.3],
            ln_r: 0.0,                  // dummy
            ln_one_minus_r_over_r: 0.0, // dummy
        };
        let new_lambda = 1.1;
        model.set_param(usize::from(TKF92Parameters::Lambda), new_lambda);
        assert_eq!(model.lambda(), new_lambda);
        let new_mu = 2.1;
        model.set_param(usize::from(TKF92Parameters::Mu), new_mu);
        assert_eq!(model.mu(), new_mu);
        let new_r = 0.4;
        model.set_param(usize::from(TKF92Parameters::R), new_r);
        assert_eq!(model.r(), new_r);
        assert_eq!(model.ln_r, new_r.ln());
        assert_eq!(model.ln_one_minus_r_over_r, (-new_r).ln_1p() - (new_r).ln());
    }
}
