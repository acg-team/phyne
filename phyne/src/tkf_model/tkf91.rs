use std::fmt::Display;

use num_enum::{FromPrimitive, IntoPrimitive};
use rand::{Rng, SeedableRng};

use crate::alignment::AncestralAlignment;
use crate::likelihood::ParamRange;
use crate::random::RandomGenerator;
use crate::tkf_model::simulate_msa::{ExpectedRootLength, FragmentSampler};
use crate::tkf_model::{
    valid_lambda_mu, valid_number_tkf_params, TKFModel, DEFAULT_LAMBDA, DEFAULT_MU,
};
use crate::Result;

/// Number of parameters in the TKF91 model: lambda and mu.
pub const TKF91_PARAM_COUNT: usize = 2;

#[derive(Debug, Eq, PartialEq, FromPrimitive, IntoPrimitive)]
#[repr(usize)]
pub(crate) enum TKF91Parameters {
    Lambda = 0,
    Mu = 1,
    #[num_enum(catch_all)]
    Invalid(usize),
}

/// By setting this to a value smaller than the number of parameters in [`TKF91SearchParameters`],
/// we can limit the model search to a subset of parameters/axes. For example, if we set
/// this to 2, then only the ratio and rate will be optimized. This is useful for ad hoc testing.
const TKF91_SEARCH_PARAM_COUNT: usize = 2;

/// Defined parameters/axes for the [`TKF91IndelModel`] where the parameter optimisation is
/// performed on. Can be limited by setting [`TKF91_SEARCH_PARAM_COUNT`] to a value smaller than the
/// number of parameters in this enum.
///
/// Searching over derived parameters can be useful for optimisation in case of constraints like lambda < mu,
/// which is assumed so that the equilibrium root length distribution is well-defined.
#[derive(Debug, Eq, PartialEq, FromPrimitive, IntoPrimitive)]
#[repr(usize)]
enum TKF91SearchParameters {
    Ratio = 0,
    Rate = 1,
    Lambda = 2,
    Mu = 3,
    #[num_enum(catch_all)]
    Invalid(usize),
}

/// Number of parameters stored internally. That is, both the [usual parameters](`TKF91Parameters`)
/// and [derived parameters](`TKF91SearchParameters`).
const TKF91_INTERNAL_PARAM_COUNT: usize = 4;

#[derive(Clone, Debug, PartialEq)]
pub struct TKF91IndelModel {
    params: Vec<f64>,
}

impl TKF91IndelModel {
    /// Creates a new TKF91 model given parameters: lambda, mu.
    /// Validates the parameters and returns an error if they are invalid:
    /// 0 < lambda < mu < infinity.
    pub fn new(params: &[f64]) -> Result<Self> {
        valid_number_tkf_params("TKF91", "lambda, mu", params, TKF91_PARAM_COUNT)?;
        let lambda = params[usize::from(TKF91Parameters::Lambda)];
        let mu = params[usize::from(TKF91Parameters::Mu)];
        valid_lambda_mu(lambda, mu)?;

        let mut params = vec![0.0; TKF91_INTERNAL_PARAM_COUNT];
        params[usize::from(TKF91SearchParameters::Ratio)] = lambda / mu;
        params[usize::from(TKF91SearchParameters::Rate)] = (lambda + mu) / 2.0;
        params[usize::from(TKF91SearchParameters::Lambda)] = lambda;
        params[usize::from(TKF91SearchParameters::Mu)] = mu;
        Ok(Self { params })
    }

    /// Recomputes the helper coordinates from the atomic parameters.
    /// Called after an atomic parameter was set.
    fn sync_helpers_with_params(&mut self) {
        let lambda = self.params[usize::from(TKF91SearchParameters::Lambda)];
        let mu = self.params[usize::from(TKF91SearchParameters::Mu)];
        self.params[usize::from(TKF91SearchParameters::Ratio)] = lambda / mu;
        self.params[usize::from(TKF91SearchParameters::Rate)] = (lambda + mu) / 2.0;
    }

    /// Recomputes the atomic parameters from the helper coordinates.
    /// Called after a helper parameter was set.
    fn sync_params_with_helpers(&mut self) {
        let rate = self.params[usize::from(TKF91SearchParameters::Rate)];
        let ratio = self.params[usize::from(TKF91SearchParameters::Ratio)];
        self.params[usize::from(TKF91SearchParameters::Lambda)] =
            2.0 * rate * ratio / (1.0 + ratio);
        self.params[usize::from(TKF91SearchParameters::Mu)] = 2.0 * rate / (1.0 + ratio);
    }
}

impl Default for TKF91IndelModel {
    fn default() -> Self {
        let lambda = DEFAULT_LAMBDA;
        let mu = DEFAULT_MU;
        let mut params = vec![0.0; TKF91_INTERNAL_PARAM_COUNT];
        params[usize::from(TKF91SearchParameters::Ratio)] = lambda / mu;
        params[usize::from(TKF91SearchParameters::Rate)] = (lambda + mu) / 2.0;
        params[usize::from(TKF91SearchParameters::Lambda)] = lambda;
        params[usize::from(TKF91SearchParameters::Mu)] = mu;
        Self { params }
    }
}

impl TKFModel for TKF91IndelModel {
    fn lambda(&self) -> f64 {
        self.params[usize::from(TKF91SearchParameters::Lambda)]
    }

    fn mu(&self) -> f64 {
        self.params[usize::from(TKF91SearchParameters::Mu)]
    }

    fn ln_insertion_factor_at_root(&self) -> f64 {
        // TODO: this lambda.ln() and mu.ln() could be cached, see issue #152 https://github.com/acg-team/phyne/issues/152
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

    fn param_count(&self) -> usize {
        TKF91_SEARCH_PARAM_COUNT
    }

    fn param(&self, idx: usize) -> f64 {
        self.params[idx]
    }

    fn set_param(&mut self, idx: usize, value: f64) {
        let param = TKF91SearchParameters::from_primitive(idx);
        match param {
            TKF91SearchParameters::Lambda | TKF91SearchParameters::Mu => {
                self.params[idx] = value;
                self.sync_helpers_with_params();
            }
            TKF91SearchParameters::Ratio | TKF91SearchParameters::Rate => {
                self.params[idx] = value;
                self.sync_params_with_helpers();
            }
            _ => panic!("Invalid parameter index: {}", idx),
        };
    }

    fn param_range(&self, idx: usize) -> ParamRange {
        let param = TKF91SearchParameters::from_primitive(idx);
        match param {
            TKF91SearchParameters::Lambda => (f64::EPSILON, self.mu() - f64::EPSILON),
            TKF91SearchParameters::Mu => (self.lambda() + f64::EPSILON, f64::MAX),
            TKF91SearchParameters::Ratio => (f64::EPSILON, 1.0 - f64::EPSILON),
            TKF91SearchParameters::Rate => (f64::EPSILON, f64::MAX),
            _ => panic!("Invalid parameter index for TKF model: {param:?}"),
        }
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

impl FragmentSampler for TKF91IndelModel {
    /// In TKF91 every residue is its own independent link, so the fragment
    /// length is always 1 and its log-probability is ln(1) = 0.0.
    fn sample_fragment_length<R: Rng + SeedableRng>(
        &self,
        _rng: &mut RandomGenerator<R>,
    ) -> (usize, f64) {
        (1, 0.0)
    }
}

impl ExpectedRootLength for TKF91IndelModel {
    fn expected_root_length(&self) -> f64 {
        (self.lambda() / self.mu()) / (1.0 - (self.lambda() / self.mu()))
    }
}

#[cfg(test)]
mod private_tests {
    use assert_matches::assert_matches;
    use rstest::rstest;

    use crate::Error;

    use super::*;

    #[test]
    fn tkf91_indel_model_new_success() {
        let model = TKF91IndelModel::new(&[0.5, 1.0]).unwrap();
        assert_eq!(model.lambda(), 0.5);
        assert_eq!(model.mu(), 1.0);
        assert_eq!(model.param_count(), TKF91_SEARCH_PARAM_COUNT);
        assert_eq!(model.param(usize::from(TKF91SearchParameters::Ratio)), 0.5);
        assert_eq!(model.param(usize::from(TKF91SearchParameters::Rate)), 0.75);
    }

    #[rstest]
    #[case::too_few(&[0.5])]
    #[case::too_many(&[0.5, 1.0, 2.0])]
    fn tkf91_indel_model_new_wrong_param_count(#[case] params: &[f64]) {
        let err = TKF91IndelModel::new(params).unwrap_err();
        assert_matches!(
            err,
            Error::Tkf(msg) if msg.contains("TKF91: expected 2 parameters (lambda, mu), got")
        );
    }

    #[rstest]
    #[case::zero(&[0.0, 1.0])]
    #[case::negative(&[-0.5, 1.0])]
    #[case::nan(&[f64::NAN, 1.0])]
    #[case::positive_infinite(&[f64::INFINITY, 1.0])]
    #[case::negative_infinite(&[f64::NEG_INFINITY, 1.0])]
    fn tkf91_indel_model_new_invalid_lambda(#[case] params: &[f64]) {
        let err = TKF91IndelModel::new(params).unwrap_err();
        assert_matches!(
            err,
            Error::Tkf(msg)
                if msg.contains("parameter lambda must be positive, finite and non-nan")
        );
    }

    #[rstest]
    #[case::zero(&[0.5, 0.0])]
    #[case::negative(&[0.5, -1.0])]
    #[case::nan(&[0.5, f64::NAN])]
    #[case::positive_infinite(&[0.5, f64::INFINITY])]
    #[case::negative_infinite(&[0.5, f64::NEG_INFINITY])]
    fn tkf91_indel_model_new_invalid_mu(#[case] params: &[f64]) {
        let err = TKF91IndelModel::new(params).unwrap_err();
        assert_matches!(
            err,
            Error::Tkf(msg) if msg.contains("parameter mu must be positive, finite and non-nan")
        );
    }

    #[rstest]
    #[case::lambda_eq_mu(&[1.0, 1.0])]
    #[case::lambda_gt_mu(&[2.0, 1.0])]
    fn tkf91_indel_model_new_lambda_ge_mu(#[case] params: &[f64]) {
        let err = TKF91IndelModel::new(params).unwrap_err();
        assert_matches!(
            err,
            Error::Tkf(msg) if msg.contains("parameter lambda must be less than mu")
        );
    }

    #[test]
    #[should_panic]
    fn tkf91_param_range_invalid_index() {
        let model = TKF91IndelModel::new(&[0.5, 1.0]).unwrap();
        // Use an invalid index
        model.param_range(TKF91_INTERNAL_PARAM_COUNT);
    }

    #[test]
    fn tkf91_model_fmt() {
        let tkf_indel_model = TKF91IndelModel::new(&[1.1, 2.0]).unwrap();

        let fmt = format!("{}", tkf_indel_model);

        assert_eq!(fmt, "TKF91 with lambda = 1.1, mu = 2");
    }

    #[test]
    fn tkf91_setting_params_doesnt_drift() {
        // During prototyping when setting a new ratio or rate the lambda and mu
        // would be used to calculate new values for lambda and mu.
        // This caused parameters to differ slightly when setting rate or ratio multiple times,
        // i.e., setting rate to x_1, x_2, ..., x_n or setting rate straight to x_n would yield
        // different values for lambda and mu.
        // With the current implementation, coordinates systems are separated: to calculate lambda
        // and mu, only ratio and rate are used, and to calculate ratio and rate, only lambda and mu
        // are used.
        let mut model1 = TKF91IndelModel::new(&[1.0, 2.0]).unwrap();
        let mut model2 = model1.clone();
        let ratio_idx = usize::from(TKF91SearchParameters::Ratio);
        let rate_idx = usize::from(TKF91SearchParameters::Rate);

        let new_rate = f64::EPSILON;
        let end_rate = 2.0;
        model1.set_param(rate_idx, new_rate);
        model1.set_param(rate_idx, end_rate);
        model2.set_param(rate_idx, end_rate);
        assert_eq!(model1.lambda(), model2.lambda());
        assert_eq!(model1.mu(), model2.mu());

        let new_ratio = f64::EPSILON;
        let end_ratio = 0.5;
        model1.set_param(ratio_idx, new_ratio);
        model1.set_param(ratio_idx, end_ratio);
        model2.set_param(ratio_idx, end_ratio);
        assert_eq!(model1.lambda(), model2.lambda());
        assert_eq!(model1.mu(), model2.mu());
    }

    #[test]
    fn tkf91_set_lambda() {
        let mut model = TKF91IndelModel::new(&[1.0, 2.0]).unwrap();
        let new_lambda = 0.5;
        model.set_param(usize::from(TKF91SearchParameters::Lambda), new_lambda);
        assert_eq!(model.lambda(), new_lambda);
        assert_eq!(model.mu(), 2.0);
        assert_eq!(model.param(usize::from(TKF91SearchParameters::Ratio)), 0.25);
        assert_eq!(model.param(usize::from(TKF91SearchParameters::Rate)), 1.25);
    }

    #[test]
    fn tkf91_set_mu() {
        let mut model = TKF91IndelModel::new(&[1.0, 2.0]).unwrap();
        let new_mu = 3.0;
        model.set_param(usize::from(TKF91SearchParameters::Mu), new_mu);
        assert_eq!(model.lambda(), 1.0);
        assert_eq!(model.mu(), new_mu);
        assert_eq!(
            model.param(usize::from(TKF91SearchParameters::Ratio)),
            1.0 / 3.0
        );
        assert_eq!(model.param(usize::from(TKF91SearchParameters::Rate)), 2.0);
    }

    #[test]
    fn tkf91_set_ratio() {
        let mut model = TKF91IndelModel::new(&[1.0, 5.0]).unwrap();
        let initial_rate = model.param(usize::from(TKF91SearchParameters::Rate));
        let new_ratio = 0.5;
        model.set_param(usize::from(TKF91SearchParameters::Ratio), new_ratio);
        assert_eq!(model.lambda(), 2.0);
        assert_eq!(model.mu(), 4.0);
        assert_eq!(
            model.param(usize::from(TKF91SearchParameters::Ratio)),
            new_ratio
        );
        assert_eq!(
            model.param(usize::from(TKF91SearchParameters::Rate)),
            initial_rate
        );
    }

    #[test]
    fn tkf91_set_rate() {
        let mut model = TKF91IndelModel::new(&[1.0, 2.0]).unwrap();
        let initial_ratio = model.param(usize::from(TKF91SearchParameters::Ratio));
        assert_eq!(initial_ratio, 0.5);
        let new_rate = 3.0;
        model.set_param(usize::from(TKF91SearchParameters::Rate), new_rate);
        assert_eq!(model.lambda(), 2.0);
        assert_eq!(model.mu(), 4.0);
        assert_eq!(
            model.param(usize::from(TKF91SearchParameters::Ratio)),
            initial_ratio
        );
        assert_eq!(
            model.param(usize::from(TKF91SearchParameters::Rate)),
            new_rate
        );
    }
}
