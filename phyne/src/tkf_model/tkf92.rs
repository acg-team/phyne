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

/// Number of parameters in the TKF92 model: lambda, mu, and r.
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

/// By setting this to a value smaller than the number of parameters in [`TKF92SearchParameters`],
/// we can limit the model search to a subset of parameters/axes. For example, if we set
/// this to 2, then only the ratio and rate will be optimized. This is useful for ad hoc testing.
const TKF92_SEARCH_PARAM_COUNT: usize = 3;

/// Defined parameters/axes for the [`TKF92IndelModel`] where the parameter optimisation is
/// performed on. Can be limited by setting [`TKF92_SEARCH_PARAM_COUNT`] to a value smaller than the
/// number of parameters in this enum.
///
/// Searching over derived parameters can be useful for optimisation in case of constraints like lambda < mu,
/// which is assumed so that the equilibrium root length distribution is well-defined.
///
/// Code and tests are in a way that allows for changing the order. In combination with the
/// [`TKF92_SEARCH_PARAM_COUNT`] constant, this allows for ad hoc testing of different search strategies.
#[derive(Debug, Eq, PartialEq, FromPrimitive, IntoPrimitive)]
#[repr(usize)]
enum TKF92SearchParameters {
    Ratio = 0,
    Rate = 1,
    R = 2,
    Lambda = 3,
    Mu = 4,
    #[num_enum(catch_all)]
    Invalid(usize),
}

/// Number of parameters stored internally. That is, both the [usual parameters](`TKF92Parameters`)
/// and [derived parameters](`TKF92SearchParameters`).
const TKF92_INTERNAL_PARAM_COUNT: usize = 5;

#[derive(Clone, Debug, PartialEq)]
pub struct TKF92IndelModel {
    params: Vec<f64>,
    /// precomputed r.ln()
    ln_r: f64,
    /// precomputed ((1 - r)/r).ln()
    ln_one_minus_r_over_r: f64,
}

impl TKF92IndelModel {
    /// Creates a new TKF92 model given parameters: lambda, mu, r.
    /// Validates the parameters and returns an error if they are invalid:
    /// 0 < lambda < mu < infinity, 0 < r < 1.
    pub fn new(params: &[f64]) -> Result<Self> {
        valid_number_tkf_params("TKF92", "lambda, mu, r", params, TKF92_PARAM_COUNT)?;
        let lambda = params[usize::from(TKF92Parameters::Lambda)];
        let mu = params[usize::from(TKF92Parameters::Mu)];
        let r = params[usize::from(TKF92Parameters::R)];
        valid_lambda_mu(lambda, mu)?;
        valid_r(r)?;
        let r = params[usize::from(TKF92Parameters::R)];

        let mut params = vec![0.0; TKF92_INTERNAL_PARAM_COUNT];
        params[usize::from(TKF92SearchParameters::Lambda)] = lambda;
        params[usize::from(TKF92SearchParameters::Mu)] = mu;
        params[usize::from(TKF92SearchParameters::R)] = r;
        params[usize::from(TKF92SearchParameters::Ratio)] = lambda / mu;
        params[usize::from(TKF92SearchParameters::Rate)] = (lambda + mu) / 2.0;
        Ok(Self {
            params,
            ln_r: r.ln(),
            ln_one_minus_r_over_r: ((1.0 - r) / r).ln(),
        })
    }

    pub fn r(&self) -> f64 {
        self.params[usize::from(TKF92SearchParameters::R)]
    }

    /// Recomputes the helper coordinates from the atomic parameters.
    /// Called after an atomic parameter was set.
    fn sync_helpers_with_params(&mut self) {
        let lambda = self.params[usize::from(TKF92SearchParameters::Lambda)];
        let mu = self.params[usize::from(TKF92SearchParameters::Mu)];
        self.params[usize::from(TKF92SearchParameters::Ratio)] = lambda / mu;
        self.params[usize::from(TKF92SearchParameters::Rate)] = (lambda + mu) / 2.0;
    }

    /// Recomputes the atomic parameters from the helper coordinates.
    /// Called after a helper parameter was set.
    fn sync_params_with_helpers(&mut self) {
        let rate = self.params[usize::from(TKF92SearchParameters::Rate)];
        let ratio = self.params[usize::from(TKF92SearchParameters::Ratio)];
        self.params[usize::from(TKF92SearchParameters::Lambda)] =
            2.0 * rate * ratio / (1.0 + ratio);
        self.params[usize::from(TKF92SearchParameters::Mu)] = 2.0 * rate / (1.0 + ratio);
    }
}

impl Default for TKF92IndelModel {
    fn default() -> Self {
        let r = DEFAULT_R;
        let lambda = DEFAULT_LAMBDA;
        let mu = DEFAULT_MU;
        let mut params = vec![0.0; TKF92_INTERNAL_PARAM_COUNT];
        params[usize::from(TKF92SearchParameters::Lambda)] = lambda;
        params[usize::from(TKF92SearchParameters::Mu)] = mu;
        params[usize::from(TKF92SearchParameters::R)] = r;
        params[usize::from(TKF92SearchParameters::Ratio)] = lambda / mu;
        params[usize::from(TKF92SearchParameters::Rate)] = (lambda + mu) / 2.0;
        Self {
            params,
            ln_r: r.ln(),
            ln_one_minus_r_over_r: (-r).ln_1p() - r.ln(),
        }
    }
}

impl TKFModel for TKF92IndelModel {
    fn lambda(&self) -> f64 {
        self.params[usize::from(TKF92SearchParameters::Lambda)]
    }

    fn mu(&self) -> f64 {
        self.params[usize::from(TKF92SearchParameters::Mu)]
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

    fn param_count(&self) -> usize {
        TKF92_SEARCH_PARAM_COUNT
    }

    fn param(&self, idx: usize) -> f64 {
        self.params[idx]
    }

    fn set_param(&mut self, idx: usize, value: f64) {
        let param = TKF92SearchParameters::from_primitive(idx);
        match param {
            TKF92SearchParameters::R => {
                self.params[usize::from(TKF92Parameters::R)] = value;
                self.ln_r = value.ln();
                self.ln_one_minus_r_over_r = (-value).ln_1p() - value.ln();
            }
            TKF92SearchParameters::Lambda | TKF92SearchParameters::Mu => {
                self.params[idx] = value;
                self.sync_helpers_with_params();
            }
            TKF92SearchParameters::Ratio | TKF92SearchParameters::Rate => {
                self.params[idx] = value;
                self.sync_params_with_helpers();
            }
            _ => panic!("Invalid parameter index: {}", idx),
        };
    }

    fn param_range(&self, idx: usize) -> ParamRange {
        let param = TKF92SearchParameters::from_primitive(idx);
        match param {
            TKF92SearchParameters::Lambda => (f64::EPSILON, self.mu() - f64::EPSILON),
            TKF92SearchParameters::Mu => (self.lambda() + f64::EPSILON, f64::MAX),
            TKF92SearchParameters::R => PARAM_RANGE_UNIT_INTERVAL_EXCLUSIVE,
            TKF92SearchParameters::Ratio => (f64::EPSILON, 1.0 - f64::EPSILON),
            TKF92SearchParameters::Rate => (f64::EPSILON, f64::MAX),
            _ => panic!("Invalid parameter index for TKF model: {param:?}"),
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
        assert_eq!(model.ln_r, 0.3f64.ln());
        assert_eq!(model.ln_one_minus_r_over_r, ((1.0f64 - 0.3) / 0.3).ln());
        assert_eq!(model.param(usize::from(TKF92SearchParameters::Ratio)), 0.5);
        assert_eq!(model.param(usize::from(TKF92SearchParameters::Rate)), 0.75);
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
        let model = TKF92IndelModel::default();
        // Use an invalid index
        model.param_range(TKF92_INTERNAL_PARAM_COUNT);
    }

    #[test]
    fn tkf92_model_fmt() {
        let tkf_indel_model = TKF92IndelModel::new(&[1.1, 2.0, 0.3]).unwrap();

        let fmt = format!("{}", tkf_indel_model);

        assert_eq!(fmt, "TKF92 with lambda = 1.1, mu = 2, r = 0.3");
    }

    #[test]
    fn tkf92_setting_params_doesnt_drift() {
        // During prototyping when setting a new ratio or rate the lambda and mu
        // would be used to calculate new values for lambda and mu.
        // This caused parameters to differ slightly when setting rate or ratio multiple times,
        // i.e., setting rate to x_1, x_2, ..., x_n or setting rate straight to x_n would yield
        // different values for lambda and mu.
        // With the current implementation, coordinates systems are separated: to calculate lambda
        // and mu, only ratio and rate are used, and to calculate ratio and rate, only lambda and mu
        // are used.
        let mut model1 = TKF92IndelModel::new(&[1.0, 2.0, 0.3]).unwrap();
        let mut model2 = model1.clone();
        let ratio_idx = usize::from(TKF92SearchParameters::Ratio);
        let rate_idx = usize::from(TKF92SearchParameters::Rate);

        let new_rate = f64::EPSILON;
        let end_rate = 2.0;
        model1.set_param(rate_idx, new_rate);
        model1.set_param(rate_idx, end_rate);
        model2.set_param(rate_idx, end_rate);
        assert_eq!(model1.lambda(), model2.lambda());
        assert_eq!(model1.mu(), model2.mu());

        let new_ratio = 1.0 - f64::EPSILON;
        let end_ratio = 0.5;
        model1.set_param(ratio_idx, new_ratio);
        model1.set_param(ratio_idx, end_ratio);
        model2.set_param(ratio_idx, end_ratio);
        assert_eq!(model1.lambda(), model2.lambda());
        assert_eq!(model1.mu(), model2.mu());
    }

    #[test]
    fn tkf92_set_lambda() {
        let mut model = TKF92IndelModel::new(&[1.0, 2.0, 0.5]).unwrap();
        let new_lambda = 0.5;
        model.set_param(usize::from(TKF92SearchParameters::Lambda), new_lambda);
        assert_eq!(model.lambda(), new_lambda);
        assert_eq!(model.mu(), 2.0);
        assert_eq!(model.r(), 0.5);
        assert_eq!(model.param(usize::from(TKF92SearchParameters::Ratio)), 0.25);
        assert_eq!(model.param(usize::from(TKF92SearchParameters::Rate)), 1.25);
    }

    #[test]
    fn tkf92_set_mu() {
        let mut model = TKF92IndelModel::new(&[1.0, 2.0, 0.5]).unwrap();
        let new_mu = 3.0;
        model.set_param(usize::from(TKF92SearchParameters::Mu), new_mu);
        assert_eq!(model.lambda(), 1.0);
        assert_eq!(model.mu(), new_mu);
        assert_eq!(model.r(), 0.5);
        assert_eq!(
            model.param(usize::from(TKF92SearchParameters::Ratio)),
            1.0 / 3.0
        );
        assert_eq!(model.param(usize::from(TKF92SearchParameters::Rate)), 2.0);
    }

    #[test]
    fn tkf92_set_r() {
        let mut model = TKF92IndelModel::new(&[1.0, 2.0, 0.5]).unwrap();
        let new_r = 0.25;
        model.set_param(usize::from(TKF92SearchParameters::R), new_r);
        assert_eq!(model.lambda(), 1.0);
        assert_eq!(model.mu(), 2.0);
        assert_eq!(model.r(), new_r);
        assert_eq!(model.param(usize::from(TKF92SearchParameters::Ratio)), 0.5);
        assert_eq!(model.param(usize::from(TKF92SearchParameters::Rate)), 1.5);
    }

    #[test]
    fn tkf92_set_ratio() {
        let mut model = TKF92IndelModel::new(&[1.0, 5.0, 0.5]).unwrap();
        let initial_rate = model.param(usize::from(TKF92SearchParameters::Rate));
        let new_ratio = 0.5;
        model.set_param(usize::from(TKF92SearchParameters::Ratio), new_ratio);
        assert_eq!(model.lambda(), 2.0);
        assert_eq!(model.mu(), 4.0);
        assert_eq!(model.r(), 0.5);
        assert_eq!(
            model.param(usize::from(TKF92SearchParameters::Ratio)),
            new_ratio
        );
        assert_eq!(
            model.param(usize::from(TKF92SearchParameters::Rate)),
            initial_rate
        );
    }

    #[test]
    fn tkf92_set_rate() {
        let mut model = TKF92IndelModel::new(&[1.0, 2.0, 0.5]).unwrap();
        let initial_ratio = model.param(usize::from(TKF92SearchParameters::Ratio));
        let new_rate = 3.0;
        model.set_param(usize::from(TKF92SearchParameters::Rate), new_rate);
        assert_eq!(model.lambda(), 2.0);
        assert_eq!(model.mu(), 4.0);
        assert_eq!(model.r(), 0.5);
        assert_eq!(
            model.param(usize::from(TKF92SearchParameters::Ratio)),
            initial_ratio
        );
        assert_eq!(
            model.param(usize::from(TKF92SearchParameters::Rate)),
            new_rate
        );
    }
}
