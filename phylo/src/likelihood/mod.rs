use crate::evolutionary_models::FreqVector;
use crate::tree::Tree;
use crate::Result;

/// The valid range for a model parameter [min, max], inclusive.
pub type ParamRange = (f64, f64);
/// Parameter is unbounded (i.e. can take any real value).
pub static PARAM_RANGE_UNBOUNDED: ParamRange = (f64::MIN, f64::MAX);
/// Parameter is strictly positive (i.e. greater than zero).
pub static PARAM_RANGE_POSITIVE: ParamRange = (f64::EPSILON, f64::MAX);
/// Parameter is non-negative (i.e. greater than or equal to zero).
pub static PARAM_RANGE_NON_NEGATIVE: ParamRange = (0.0, f64::MAX);
/// Parameter is in the unit interval [0, 1], inclusive.
pub static PARAM_RANGE_UNIT_INTERVAL: ParamRange = (0.0, 1.0);
/// Parameter is in the unit interval (0, 1), exclusive.
pub static PARAM_RANGE_UNIT_INTERVAL_EXCLUSIVE: ParamRange = (f64::EPSILON, 1.0 - f64::EPSILON);
/// Dummy parameter range for models without parameters.
pub static PARAM_RANGE_DUMMY: ParamRange = (0.0, 0.0);

pub trait TreeSearchCost {
    // The optimisers will maximise the cost, so if the cost should be minimised instead, it should be negated.
    // The likelihood or the log-likelihood are maximised, the parsimony score is minimised.
    fn cost(&self) -> f64;
    // update_tree implies that the tree is a valid modification of the existing tree (e.g. an SPR move),
    // and that the dirty_nodes are the nodes that have changed, but this is not enforced by the trait.
    // TODO: enforce this in the trait.
    fn update_tree(&mut self, tree: Tree);
    fn tree(&self) -> &Tree;
    fn blen_optimisation(&self) -> bool {
        true
    }
}

pub trait ModelSearchCost {
    // The cost function definition for model search, e.g. the likelihood of the alignment given the model and the tree.
    // The optimisers will maximise the cost, so if the cost should be minimised instead, it should be negated.
    // The likelihood or the log-likelihood are maximised, the parsimony score is minimised.
    fn cost(&self) -> f64;
    fn param_count(&self) -> usize;
    fn param(&self, param: usize) -> f64;
    fn set_param(&mut self, param: usize, value: f64);
    /// Returns the valid range for a model parameter [min, max], inclusive.
    fn param_range(&self, param: usize) -> ParamRange;
    fn set_freqs(&mut self, freqs: FreqVector) -> Result<()>;
    fn empirical_freqs(&self) -> FreqVector;
    fn freqs(&self) -> &FreqVector;
}

#[cfg(test)]
#[cfg_attr(coverage, coverage(off))]
mod tests;
