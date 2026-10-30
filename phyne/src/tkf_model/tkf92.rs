use std::fmt::Display;

use num_enum::{FromPrimitive, IntoPrimitive};
use rand::{Rng, SeedableRng};
use rand_distr::Geometric;

use crate::alignment::{blocks_of_ancestral_alignment, AncestralAlignment};
use crate::likelihood::{ParamRange, PARAM_RANGE_UNIT_INTERVAL_EXCLUSIVE};
use crate::random::RandomGenerator;
use crate::tkf_model::simulate_msa::{ExpectedRootLength, FragmentSampler};
use crate::tkf_model::{
    valid_lambda_mu, valid_number_tkf_params, valid_r, TKFModel, DEFAULT_LAMBDA, DEFAULT_MU,
    DEFAULT_R,
};
use crate::Result;

pub const TKF92_PARAM_COUNT: usize = 3;

#[derive(Debug, Eq, PartialEq, FromPrimitive, IntoPrimitive)]
#[repr(usize)]
pub(crate) enum TKF92Parameters {
    Lambda = 0,
    Mu = 1,
    R = 2,
    #[num_enum(catch_all)]
    Invalid(usize),
}

pub(crate) const TKF92_HELPER_PARAM_COUNT: usize = 2;

#[derive(Debug, Eq, PartialEq, FromPrimitive, IntoPrimitive)]
#[repr(usize)]
pub(crate) enum TKF92HelperParameters {
    Rate = 0,
    Ratio = 1,
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
    /// original lambda before ratio or rate was set
    before_helper_lambda: f64,
    /// original mu before ratio or rate was set
    before_helper_mu: f64,
}

impl TKF92IndelModel {
    /// Creates a new TKF92 model given parameters: lambda, mu, r.
    /// Validates the parameters and returns an error if they are invalid:
    /// 0 < lambda < mu < infinity, 0 < r < 1.
    pub fn new(params: &[f64]) -> Result<Self> {
        valid_number_tkf_params("TKF92", "lambda, mu, r", params, TKF92_PARAM_COUNT)?;
        let lambda = params[usize::from(TKF92Parameters::Lambda)];
        let mu = params[usize::from(TKF92Parameters::Mu)];
        valid_lambda_mu(lambda, mu)?;
        let r = params[usize::from(TKF92Parameters::R)];
        valid_r(r)?;
        Ok(Self {
            params: params.to_vec(),
            ln_r: r.ln(),
            ln_one_minus_r_over_r: ((1.0 - r) / r).ln(),
            before_helper_lambda: lambda,
            before_helper_mu: mu,
        })
    }

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
            before_helper_lambda: DEFAULT_LAMBDA,
            before_helper_mu: DEFAULT_MU,
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
            TKF92Parameters::Lambda => {
                self.params[idx] = value;
                self.before_helper_lambda = value;
            }
            TKF92Parameters::Mu => {
                self.params[idx] = value;
                self.before_helper_mu = value;
            }
            _ => panic!("Invalid parameter index for TKF model: {param:?}"),
        };
        println!("set param: lambda = {}, mu = {}", self.lambda(), self.mu(),);
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
        // TODO: this lambda.ln() could be cached, see issue #152 https://github.com/acg-team/phyne/issues/152
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
        //   See issue https://github.com/acg-team/phyne/issues/174.
        ln_tree_event_factor
            + (block_len as f64 - 1.0) * (ln_tree_event_factor.exp()).ln_1p()
            + (block_len as f64) * self.ln_r
    }

    fn get_blocks<AA: AncestralAlignment>(&self, msa: &AA) -> Vec<usize> {
        blocks_of_ancestral_alignment(msa)
    }

    fn helper_param_count(&self) -> usize {
        // with that instead of zero the optimisation converges much faster and achieves better logl
        TKF92_HELPER_PARAM_COUNT
        // 0
    }

    fn helper_param(&self, idx: usize) -> f64 {
        let helper_param = TKF92HelperParameters::from_primitive(idx);
        match helper_param {
            TKF92HelperParameters::Rate => (self.lambda() + self.mu()) / 2.0,
            TKF92HelperParameters::Ratio => self.lambda() / self.mu(),
            _ => panic!("Invalid helper parameter index for TKF model: {helper_param:?}"),
        }
    }

    fn set_helper_param(&mut self, idx: usize, value: f64) {
        let param = TKF92HelperParameters::from_primitive(idx);
        match param {
            TKF92HelperParameters::Rate => {
                let lambda = self.before_helper_lambda;
                let mu = self.before_helper_mu;
                self.params[usize::from(TKF92Parameters::Lambda)] =
                    2.0 * value / (mu / lambda + 1.0);
                self.params[usize::from(TKF92Parameters::Mu)] = 2.0 * value / (lambda / mu + 1.0);
            }
            TKF92HelperParameters::Ratio => {
                let lambda = self.before_helper_lambda;
                let mu = self.before_helper_mu;
                let x = (lambda + mu) / 2.0;
                let y = (x * (1.0 - value)) / (1.0 + value);
                self.params[usize::from(TKF92Parameters::Lambda)] = x - y;
                self.params[usize::from(TKF92Parameters::Mu)] = x + y;
            }
            _ => panic!("Invalid helper parameter index for TKF model: {param:?}"),
        }
        println!("set helper: lambda = {}, mu = {}", self.lambda(), self.mu());
    }

    fn helper_param_range(&self, _idx: usize) -> ParamRange {
        let param = TKF92HelperParameters::from_primitive(_idx);
        match param {
            TKF92HelperParameters::Rate => (f64::EPSILON, f64::MAX),
            TKF92HelperParameters::Ratio => (f64::EPSILON, 1.0 - f64::EPSILON),
            _ => panic!("Invalid helper parameter index for TKF model: {param:?}"),
        }
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

impl FragmentSampler for TKF92IndelModel {
    /// Samples fragment length from a geometric distribution parameterised by `r`.
    /// `r` is the probability that the next residue continues the current fragment,
    /// so `1 - r` is the stopping (success) probability of the geometric draw.
    /// Returns `(length, log_probability)`.
    fn sample_fragment_length<R: Rng + SeedableRng>(
        &self,
        rng: &mut RandomGenerator<R>,
    ) -> (usize, f64) {
        let prob_of_success = 1.0 - self.r();
        let geom = Geometric::new(prob_of_success).unwrap();
        let choice = rng.sample(&geom);
        // Geometric PMF: (1-p)^k * p  where k = number of failures before first success
        let log_prob = (choice as f64) * (1.0 - prob_of_success).ln() + prob_of_success.ln();
        (choice as usize + 1, log_prob) // +1: every fragment has at least one character
    }
}

impl ExpectedRootLength for TKF92IndelModel {
    fn expected_root_length(&self) -> f64 {
        (self.lambda() / self.mu()) / ((1.0 - self.lambda() / self.mu()) * (1.0 - self.r()))
    }
}

#[cfg(test)]
#[cfg_attr(coverage, coverage(off))]
mod private_tests {
    use assert_matches::assert_matches;
    use rstest::rstest;

    use crate::Error;

    use super::*;

    #[test]
    fn tkf92_indel_model_new_success() {
        let model = TKF92IndelModel::new(&[0.5, 1.0, 0.3]).unwrap();
        assert_eq!(model.lambda(), 0.5);
        assert_eq!(model.mu(), 1.0);
        assert_eq!(model.r(), 0.3);
        assert_eq!(model.params(), &[0.5, 1.0, 0.3]);
        assert_eq!(model.ln_r, 0.3f64.ln());
        assert_eq!(model.ln_one_minus_r_over_r, ((1.0f64 - 0.3) / 0.3).ln());
    }

    #[rstest]
    #[case::too_few(&[0.5, 1.0])]
    #[case::too_many(&[0.5, 1.0, 0.3, 2.0])]
    fn tkf92_indel_model_new_wrong_param_count(#[case] params: &[f64]) {
        let err = TKF92IndelModel::new(params).unwrap_err();
        assert_matches!(
            err,
            Error::Tkf(msg)
                if msg.contains("TKF92: expected 3 parameters (lambda, mu, r), got")
        );
    }

    #[rstest]
    #[case::zero(&[0.0, 1.0, 0.3])]
    #[case::negative(&[-0.5, 1.0, 0.3])]
    #[case::nan(&[f64::NAN, 1.0, 0.3])]
    #[case::positive_infinite(&[f64::INFINITY, 1.0, 0.3])]
    #[case::negative_infinite(&[f64::NEG_INFINITY, 1.0, 0.3])]
    fn tkf92_indel_model_new_invalid_lambda(#[case] params: &[f64]) {
        let err = TKF92IndelModel::new(params).unwrap_err();
        assert_matches!(
            err,
            Error::Tkf(msg)
                if msg.contains("parameter lambda must be positive, finite and non-nan")
        );
    }

    #[rstest]
    #[case::zero(&[0.5, 0.0, 0.3])]
    #[case::negative(&[0.5, -1.0, 0.3])]
    #[case::nan(&[0.5, f64::NAN, 0.3])]
    #[case::positive_infinite(&[0.5, f64::INFINITY, 0.3])]
    #[case::negative_infinite(&[0.5, f64::NEG_INFINITY, 0.3])]
    fn tkf92_indel_model_new_invalid_mu(#[case] params: &[f64]) {
        let err = TKF92IndelModel::new(params).unwrap_err();
        assert_matches!(
            err,
            Error::Tkf(msg) if msg.contains("parameter mu must be positive, finite and non-nan")
        );
    }

    #[rstest]
    #[case::lambda_eq_mu(&[1.0, 1.0, 0.3])]
    #[case::lambda_gt_mu(&[2.0, 1.0, 0.3])]
    fn tkf92_indel_model_new_lambda_ge_mu(#[case] params: &[f64]) {
        let err = TKF92IndelModel::new(params).unwrap_err();
        assert_matches!(
            err,
            Error::Tkf(msg) if msg.contains("parameter lambda must be less than mu")
        );
    }

    #[rstest]
    #[case::zero(&[0.5, 1.0, 0.0])]
    #[case::negative(&[0.5, 1.0, -0.3])]
    #[case::nan(&[0.5, 1.0, f64::NAN])]
    #[case::positive_infinite(&[0.5, 1.0, f64::INFINITY])]
    #[case::negative_infinite(&[0.5, 1.0, f64::NEG_INFINITY])]
    #[case::eq_one(&[0.5, 1.0, 1.0])]
    #[case::gt_one(&[0.5, 1.0, 1.5])]
    fn tkf92_indel_model_new_invalid_r(#[case] params: &[f64]) {
        let err = TKF92IndelModel::new(params).unwrap_err();
        assert_matches!(
            err,
            Error::Tkf(msg) if msg.contains("parameter r must be in (0, 1)")
        );
    }

    #[test]
    #[should_panic]
    fn tkf92_param_range_invalid_index() {
        let model = TKF92IndelModel {
            params: vec![0.5, 1.0, 0.3],
            ln_r: 0.0, // cache filled with dummy, since it is not needed here
            ln_one_minus_r_over_r: 0.0, // cache filled with dummy since it is not needed here
            before_helper_lambda: 0.0, // cache filled with dummy since it is not needed here
            before_helper_mu: 0.0, // cache filled with dummy since it is not
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
            before_helper_lambda: 0.0,  // cache filled with dummy since it is not needed here
            before_helper_mu: 0.0,      // cache filled with dummy since it is not
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
            before_helper_lambda: 0.0,  // cache filled with dummy since it is not needed here
            before_helper_mu: 0.0,      // cache filled with dummy since it is not
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
