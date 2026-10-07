use std::cell::RefCell;
use std::fmt::Display;

use log::warn;
use num_enum::{FromPrimitive, IntoPrimitive};

use crate::alignment::AncestralAlignment;
use crate::likelihood::ParamRange;
use crate::phylo_info::PhyloInfo;
use crate::substitution_models::{QMatrix, SubstModel, SubstitutionCostBuilder as SCB};
use crate::tkf_model::{
    TKFCost, TKFIndelCost, TKFIndelModelInfo, TKFModel, DEFAULT_LAMBDA, DEFAULT_LAMBDA_MU_RATIO,
    DEFAULT_MU,
};
use crate::Result;

#[derive(Debug, Eq, PartialEq, FromPrimitive, IntoPrimitive)]
#[repr(usize)]
pub(crate) enum TKF91Parameters {
    Lambda = 0,
    Mu = 1,
    #[num_enum(catch_all)]
    Invalid(usize),
}

#[derive(Clone, Debug, PartialEq)]
pub struct TKF91IndelModel {
    params: Vec<f64>,
}

impl Default for TKF91IndelModel {
    fn default() -> Self {
        Self {
            params: vec![DEFAULT_LAMBDA, DEFAULT_MU],
        }
    }
}

impl TKFModel for TKF91IndelModel {
    fn lambda(&self) -> f64 {
        self.params[usize::from(TKF91Parameters::Lambda)]
    }

    fn mu(&self) -> f64 {
        self.params[usize::from(TKF91Parameters::Mu)]
    }

    fn params(&self) -> &[f64] {
        &self.params
    }

    fn set_param(&mut self, idx: usize, value: f64) {
        self.params[idx] = value;
    }

    fn param_range(&self, idx: usize) -> ParamRange {
        let param = TKF91Parameters::from_primitive(idx);
        match param {
            TKF91Parameters::Lambda => (f64::EPSILON, self.mu() - f64::EPSILON),
            TKF91Parameters::Mu => (self.lambda() + f64::EPSILON, f64::MAX),
            _ => panic!("Invalid parameter index for TKF model: {param:?}"),
        }
    }

    fn ln_insertion_factor_at_root(&self) -> f64 {
        // TODO: this lambda.ln() and mu.ln() could be cached, see issue #152 https://github.com/acg-team/rust-phylo/issues/152
        self.lambda().ln() - self.mu().ln()
    }

    fn ln_insertion_factor_at_non_root(&self, ln_beta: f64) -> f64 {
        self.lambda().ln() + ln_beta
    }

    fn block_prob(&self, ln_tree_event_factor: f64, block_len: usize) -> f64 {
        (block_len as f64) * ln_tree_event_factor
    }

    /// Since TKF91 is a single-residue indel model, each position is its own block.
    fn get_blocks<AA: AncestralAlignment>(&self, msa: &AA) -> Vec<usize> {
        (1..msa.len() + 1).collect()
    }
}

impl Display for TKF91IndelModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "TKF91 with lambda = {}, mu = {}",
            self.lambda(),
            self.mu(),
        )
    }
}

/// Validates the TKF indel parameters lambda and mu. If they are not valid, they are set to
/// default values and a warning is logged.
/// Returns valid (lambda, mu).
pub(super) fn validate_lambda_mu(params: &mut [f64]) {
    let lambda_id = usize::from(TKF91Parameters::Lambda);
    let mu_id = usize::from(TKF91Parameters::Mu);
    let lambda = params[lambda_id];
    let mu = params[mu_id];

    if lambda <= 0.0 && mu <= 0.0 {
        warn!(
            "Both lambda and mu must be positive. Setting lambda to {DEFAULT_LAMBDA} and mu to {DEFAULT_MU}."
        );
        params[lambda_id] = DEFAULT_LAMBDA;
        params[mu_id] = DEFAULT_MU;
    } else if lambda <= 0.0 {
        params[lambda_id] = DEFAULT_LAMBDA_MU_RATIO * mu;
        warn!(
            "Tried to set lambda to invalid value {lambda}. \
            It must be in (0, mu) with mu = {mu}. \
            Setting lambda to {DEFAULT_LAMBDA_MU_RATIO}*mu = {}",
            params[lambda_id]
        );
    } else if mu <= lambda {
        params[mu_id] = lambda / DEFAULT_LAMBDA_MU_RATIO;
        warn!(
            "Tried to set mu to invalid value {mu}. \
            It must be in (lambda, infinity) with lambda = {lambda}. \
            Setting mu to lambda/{DEFAULT_LAMBDA_MU_RATIO} = {}",
            params[mu_id]
        );
    }
}

/// Builder for the cost using the [`TKF91IndelModel`], i.e., without a substitution model.
pub struct TKF91IndelCostBuilder<AA: AncestralAlignment> {
    params: Vec<f64>,
    phylo: PhyloInfo<AA>,
}

impl<AA: AncestralAlignment> TKF91IndelCostBuilder<AA> {
    pub fn new(params: &[f64], phylo: PhyloInfo<AA>) -> Self {
        Self {
            params: params.to_vec(),
            phylo,
        }
    }

    pub fn build(self) -> Result<TKFIndelCost<TKF91IndelModel, AA>> {
        let mut params = self.params;
        if params.len() != 2 {
            warn!(
                "Expected 2 parameters for TKF91 model (lambda, mu), but got {}",
                params.len()
            );
            warn!("Falling back to default values");
            params.resize(2, 0.0);
            params[usize::from(TKF91Parameters::Lambda)] = DEFAULT_LAMBDA;
            params[usize::from(TKF91Parameters::Mu)] = DEFAULT_MU;
        }
        validate_lambda_mu(&mut params);
        let model = TKF91IndelModel { params };
        let info = TKFIndelModelInfo::new(&model, &self.phylo);
        Ok(TKFIndelCost {
            model,
            phylo: self.phylo,
            model_info: RefCell::new(info),
        })
    }
}

/// Builder for the TKF91 cost, i.e., with a substitution model.
pub struct TKF91CostBuilder<Q: QMatrix, AA: AncestralAlignment> {
    params: Vec<f64>,
    subst_model: SubstModel<Q>,
    phylo: PhyloInfo<AA>,
}

impl<Q: QMatrix, AA: AncestralAlignment> TKF91CostBuilder<Q, AA> {
    pub fn new(params: &[f64], subst_model: SubstModel<Q>, phylo: PhyloInfo<AA>) -> Self {
        Self {
            params: params.to_vec(),
            subst_model,
            phylo,
        }
    }

    pub fn build(self) -> Result<TKFCost<Q, TKF91IndelModel, AA>> {
        let indel_cost = TKF91IndelCostBuilder::new(&self.params, self.phylo.clone()).build()?;
        let subst_cost = SCB::new(self.subst_model, self.phylo.clone()).build()?;
        Ok(TKFCost {
            indel_cost,
            subst_cost,
        })
    }
}

#[cfg(test)]
mod private_tests {
    use super::*;

    #[test]
    #[should_panic]
    fn tkf91_param_range_invalid_index() {
        let model = TKF91IndelModel {
            params: vec![0.5, 1.0],
        };
        // Use an invalid index
        model.param_range(2);
    }

    #[test]
    fn tkf91_model_fmt() {
        let tkf_indel_model = TKF91IndelModel {
            params: vec![1.1, 2.0],
        };

        let fmt = format!("{}", tkf_indel_model);

        assert_eq!(fmt, "TKF91 with lambda = 1.1, mu = 2");
    }

    #[test]
    fn tkf91_indel_set_param() {
        let mut model = TKF91IndelModel {
            params: vec![1.0, 2.0],
        };
        let new_lambda = 1.1;
        model.set_param(usize::from(TKF91Parameters::Lambda), new_lambda);
        assert_eq!(model.lambda(), new_lambda);
        let new_mu = 2.1;
        model.set_param(usize::from(TKF91Parameters::Mu), new_mu);
        assert_eq!(model.mu(), new_mu);
    }
}
