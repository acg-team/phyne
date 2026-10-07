use std::num::NonZeroUsize;

use crate::likelihood::{ModelSearchCost, TreeSearchCost};
use crate::DEFAULT_EPSILON;

pub mod blen_optimiser;
pub use blen_optimiser::*;
pub mod model_optimiser;
pub use model_optimiser::*;
pub mod topo_optimiser;
pub use topo_optimiser::*;
pub mod spr_optimiser;
pub use spr_optimiser::*;
pub mod nni_optimiser;
pub use nni_optimiser::*;
pub mod move_optimiser;
pub use move_optimiser::*;

#[derive(Debug, Clone, Copy)]
pub enum StopCondition {
    Epsilon(f64),
    FixedIter(NonZeroUsize),
    MaxIterEpsilon(NonZeroUsize, f64),
    // NOTE: use of `fn(..) -> ..` disallows closures that capture any
    // surrounding variables, for that we would need to allow Boxed Fn
    // trait objects (or introduce a generic parameter which might get tedious)
    Custom(fn(usize, f64) -> bool),
}

impl std::fmt::Display for StopCondition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StopCondition::Epsilon(e) => write!(f, "delta cost < {e}"),
            StopCondition::FixedIter(n) => write!(f, "fixed number of iterations = {}", n.get()),
            StopCondition::MaxIterEpsilon(n, e) => {
                write!(f, "delta cost < {e}, max iterations = {}", n.get())
            }
            StopCondition::Custom(fun) => write!(f, "custom predicate function: {fun:?}"),
        }
    }
}

impl Default for StopCondition {
    fn default() -> Self {
        StopCondition::Epsilon(DEFAULT_EPSILON)
    }
}

impl StopCondition {
    /// Validates whether the optimisation should continue based on the current iteration
    /// number and the change in cost (delta) since the last iteration.
    /// Returns `true` if the optimisation should continue, `false` otherwise
    pub fn should_continue(&self, iteration: usize, delta: f64) -> bool {
        match *self {
            StopCondition::Epsilon(min_delta) => delta > min_delta,
            StopCondition::FixedIter(max) => max.get() > iteration,
            StopCondition::MaxIterEpsilon(max, min_delta) => {
                max.get() > iteration && delta > min_delta
            }
            StopCondition::Custom(pred) => pred(iteration, delta),
        }
    }

    pub fn epsilon(epsilon: f64) -> Self {
        Self::Epsilon(epsilon)
    }

    pub fn fixed_iter(num: NonZeroUsize) -> Self {
        Self::FixedIter(num)
    }

    pub fn max_iter_epsilon(num: NonZeroUsize, epsilon: f64) -> Self {
        Self::MaxIterEpsilon(num, epsilon)
    }

    pub fn max_iter(num: NonZeroUsize) -> Self {
        Self::MaxIterEpsilon(num, DEFAULT_EPSILON)
    }

    pub fn custom(pred: fn(usize, f64) -> bool) -> Self {
        Self::Custom(pred)
    }
}

// Struct for any single value optimisation result, e.g. branch length or evolutionary model parameter value
pub struct SingleValOptResult {
    // final cost after optimisation
    pub final_cost: f64,
    // value of the parameter after optimisation
    pub value: f64,
}

#[derive(Clone, Debug)]
pub struct TreeOptimisationResult<C: TreeSearchCost> {
    pub initial_cost: f64,
    pub final_cost: f64,
    pub iterations: usize,
    #[allow(dead_code)]
    pub(crate) costs: Vec<f64>,
    pub cost: C,
}

#[derive(Clone, Debug)]
pub struct ModelOptimisationResult<C: ModelSearchCost> {
    pub initial_cost: f64,
    pub final_cost: f64,
    pub iterations: usize,
    #[allow(dead_code)]
    pub(crate) costs: Vec<f64>,
    pub cost: C,
}

#[cfg(test)]
#[cfg_attr(coverage, coverage(off))]
mod blen_optimiser_tests;
#[cfg(test)]
#[cfg_attr(coverage, coverage(off))]
mod model_optimiser_tests;
#[cfg(test)]
#[cfg_attr(coverage, coverage(off))]
mod topo_optimiser_tests;

#[cfg(test)]
#[cfg_attr(coverage, coverage(off))]
mod tests {
    use std::num::NonZeroUsize;

    use rstest::rstest;

    use crate::DEFAULT_EPSILON;

    use super::StopCondition;

    #[rstest]
    #[case(0, 0.1, true)]
    #[case(1, 0.02, true)]
    #[case(2, 0.005, false)]
    #[case(3, 0.0, false)]
    #[case(10, 0.000001, false)]
    #[case(10, 0.1, true)]
    fn stop_condition_epsilon(#[case] iters: usize, #[case] delta: f64, #[case] expected: bool) {
        let pred = StopCondition::epsilon(0.01);
        assert_eq!(pred.should_continue(iters, delta), expected);
    }

    #[rstest]
    #[case(0, 0.00000001, true)]
    #[case(1, 0.02, true)]
    #[case(2, 0.000005, true)]
    #[case(3, 0.0, false)]
    #[case(3, 1.0, false)]
    #[case(5, 1e-10, false)]
    fn stop_condition_fixed_iter(#[case] iters: usize, #[case] delta: f64, #[case] expected: bool) {
        let pred = StopCondition::fixed_iter(NonZeroUsize::new(3).unwrap());
        assert_eq!(pred.should_continue(iters, delta), expected);
    }

    #[rstest]
    #[case(0, 0.00000001, false)]
    #[case(1, 0.02, true)]
    #[case(1, 0.009, false)]
    #[case(2, 0.000005, false)]
    #[case(3, 0.0, false)]
    #[case(3, 1.0, false)]
    #[case(5, 1e-10, false)]
    #[case(5, 1e10, false)]
    fn stop_condition_max_iter_epsilon(
        #[case] iters: usize,
        #[case] delta: f64,
        #[case] expected: bool,
    ) {
        let pred = StopCondition::max_iter_epsilon(NonZeroUsize::new(3).unwrap(), 0.01);
        assert_eq!(pred.should_continue(iters, delta), expected);
    }

    #[rstest]
    #[case(0, 0.00000001, false)]
    #[case(1, 0.02, true)]
    #[case(1, 0.009, true)]
    #[case(1, 0.0002, false)]
    #[case(2, 0.000005, false)]
    #[case(3, 0.0, false)]
    #[case(3, 1.0, false)]
    #[case(5, 1e-10, false)]
    #[case(5, 1e10, false)]
    fn stop_condition_max_iter(#[case] iters: usize, #[case] delta: f64, #[case] expected: bool) {
        let pred = StopCondition::max_iter(NonZeroUsize::new(3).unwrap());
        match pred {
            StopCondition::MaxIterEpsilon(_, eps) => {
                assert_eq!(eps, DEFAULT_EPSILON);
            }
            _ => panic!("Expected MaxIterEpsilon variant"),
        }
        assert_eq!(pred.should_continue(iters, delta), expected);
    }

    #[rstest]
    #[case(0, 0.00000001, true)]
    #[case(1, 0.02, true)]
    #[case(2, 0.000005, false)]
    #[case(3, 0.0, false)]
    #[case(3, 1.0, true)]
    #[case(5, 1e-10, false)]
    #[case(5, 1e10, true)]
    fn stop_condition_custom(#[case] iters: usize, #[case] delta: f64, #[case] expected: bool) {
        fn custom(i: usize, d: f64) -> bool {
            !(i >= 2 && d < 0.01)
        }
        let pred = StopCondition::custom(custom);
        assert_eq!(pred.should_continue(iters, delta), expected);
    }

    #[test]
    fn stop_condition_display() {
        let cond = StopCondition::epsilon(0.001);
        assert_eq!(format!("{cond}"), "delta cost < 0.001");
        let cond = StopCondition::fixed_iter(NonZeroUsize::new(5).unwrap());
        assert_eq!(format!("{cond}"), "fixed number of iterations = 5");
        let cond = StopCondition::max_iter_epsilon(NonZeroUsize::new(10).unwrap(), 0.0001);
        assert_eq!(
            format!("{cond}"),
            "delta cost < 0.0001, max iterations = 10"
        );
        let cond = StopCondition::max_iter(NonZeroUsize::new(10).unwrap());
        assert_eq!(
            format!("{cond}"),
            format!("delta cost < {}, max iterations = 10", DEFAULT_EPSILON)
        );
        fn custom(_: usize, _: f64) -> bool {
            true
        }
        let cond = StopCondition::custom(custom);
        assert!(format!("{cond}").starts_with("custom predicate function:"));
    }
}
