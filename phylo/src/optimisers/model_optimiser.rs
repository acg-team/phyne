use std::cell::RefCell;
use std::fmt::Display;

use argmin::core::{CostFunction, Executor, IterState, State};
use argmin::solver::brent::BrentOpt;
use log::{debug, info, warn};

use crate::evolutionary_models::FrequencyOptimisation;
use crate::likelihood::ModelSearchCost;
use crate::optimisers::{ModelOptimisationResult, SingleValOptResult, StopCondition};
use crate::Result;

pub struct ModelOptimiser<C: ModelSearchCost + Display + Clone> {
    pub(crate) stop_condition: StopCondition,
    pub(crate) c: C,
    pub(crate) freq_opt: FrequencyOptimisation,
}

impl<C: ModelSearchCost + Display + Clone> ModelOptimiser<C> {
    pub fn new(cost: C, freq_opt: FrequencyOptimisation) -> Self {
        Self {
            stop_condition: StopCondition::default(),
            c: cost,
            freq_opt,
        }
    }

    pub fn with_stop_condition(
        cost: C,
        freq_opt: FrequencyOptimisation,
        stop_condition: StopCondition,
    ) -> Self {
        Self {
            stop_condition,
            c: cost,
            freq_opt,
        }
    }

    pub fn run(mut self) -> Result<ModelOptimisationResult<C>> {
        info!("Optimising the evolutionary model: {}", self.c);
        info!("Optimisation stopping condition: {}", self.stop_condition);

        let init_cost = self.c.cost();
        info!("Initial cost: {init_cost}");

        let mut curr_cost = self.optimise_frequencies()?;
        // Set previous cost to negative infinity to ensure at least one iteration if frequency optimisation did not change the cost
        let mut prev_cost = f64::NEG_INFINITY;
        let mut iterations = 0;
        let mut delta = curr_cost - prev_cost;
        // Store costs for each iteration, including initial cost before potential frequency optimisation
        let mut costs = vec![init_cost, curr_cost];

        while self.stop_condition.should_continue(iterations, delta) {
            iterations += 1;
            info!("Iteration: {iterations}, current cost: {curr_cost}");
            prev_cost = curr_cost;
            curr_cost = self.single_optimisation_iteration()?;
            debug!("New parameters: {}\n", self.c);
            delta = curr_cost - prev_cost;
            costs.push(curr_cost);
        }

        debug_assert_eq!(curr_cost, self.c.cost());
        info!("Done optimising model parameters");
        info!("Final cost: {curr_cost}, achieved in {iterations} iteration(s)");

        Ok(ModelOptimisationResult::<C> {
            initial_cost: init_cost,
            final_cost: curr_cost,
            iterations,
            costs,
            cost: self.c,
        })
    }

    /// Optimises the stationary frequencies of the model according to the specified strategy, empirical
    /// (from the alignment), maximum likelihood estimated (not implemented yet), or fixed.
    /// Returns the cost after frequency optimisation.
    ///
    /// # Errors
    ///
    /// Returns an error if setting the empirical frequencies fails, in particular when trying to
    /// set unequal frequencies for JC69 or K80.
    fn optimise_frequencies(&mut self) -> Result<f64> {
        match self.freq_opt {
            FrequencyOptimisation::Empirical => {
                info!("Setting stationary frequencies to empirical");
                self.set_empirical_freqs()?;
            }
            FrequencyOptimisation::Estimated => {
                warn!("Stationary frequency estimation not available, falling back on empirical");
                self.set_empirical_freqs()?;
            }
            FrequencyOptimisation::Fixed => {
                info!("Not optimising stationary frequencies");
            }
        }
        let cost = self.c.cost();
        info!("Cost after frequency optimisation: {cost}");
        Ok(cost)
    }

    fn single_optimisation_iteration(&mut self) -> Result<f64> {
        let mut curr_cost = self.c.cost();
        for param in 0..self.c.param_count() {
            let start_value = self.c.param(param);
            debug!("Optimising parameter {param:?} from value {start_value} with cost {curr_cost}");
            let param_opt = self.opt_parameter(param, start_value)?;
            if param_opt.final_cost < curr_cost {
                // Parameter will have been reset by the optimiser, set it back to start value
                self.c.set_param(param, start_value);
                continue;
            }
            self.c.set_param(param, param_opt.value);
            curr_cost = param_opt.final_cost;
            debug!(
                "Optimised parameter {param:?} to value {} with cost {curr_cost}",
                param_opt.value
            );
        }
        Ok(curr_cost)
    }

    fn set_empirical_freqs(&mut self) -> Result<()> {
        let emp_freqs = self.c.empirical_freqs();
        self.c.set_freqs(emp_freqs)?;
        Ok(())
    }

    fn opt_parameter(&self, param: usize, start_value: f64) -> Result<SingleValOptResult> {
        let optimiser = ParamOptimiser {
            cost: RefCell::new(self.c.clone()),
            param,
        };
        let range = self.c.param_range(param);
        let min = range.0;
        let max = range.1.min(start_value * 100.0);
        debug_assert!(min <= start_value && start_value <= max);
        let gss = BrentOpt::new(min, max);
        let res = Executor::new(optimiser, gss)
            .configure(|_| IterState::new().param(start_value).max_iters(500))
            .run()?;
        let cost = -res.state().best_cost;
        Ok(SingleValOptResult {
            value: res.state().best_param.unwrap(),
            final_cost: cost,
        })
    }
}

pub(crate) struct ParamOptimiser<C: ModelSearchCost> {
    pub(crate) cost: RefCell<C>,
    pub(crate) param: usize,
}

impl<C: ModelSearchCost> CostFunction for ParamOptimiser<C> {
    type Param = f64;
    type Output = f64;

    fn cost(&self, value: &f64) -> anyhow::Result<f64> {
        let value = if value.is_nan() || value.is_sign_negative() {
            0.0
        } else {
            *value
        };
        self.cost.borrow_mut().set_param(self.param, value);
        Ok(-self.cost.borrow().cost())
    }

    fn parallelize(&self) -> bool {
        true
    }
}

#[cfg(test)]
#[cfg_attr(coverage, coverage(off))]
mod tests {
    use std::path::Path;

    use assert_matches::assert_matches;

    use crate::likelihood::ModelSearchCost;
    use crate::phylo_info::PhyloInfoBuilder as PIB;
    use crate::pip_model::{PIPCostBuilder as PIPCB, PIPModel};
    use crate::substitution_models::{
        dna_models::GTR, protein_models::WAG, SubstModel, SubstitutionCostBuilder as SCB,
    };

    use super::*;

    #[test]
    fn estimated_freqs_default_to_empirical_dna() {
        let fldr = Path::new("./data/sim");
        let info = PIB::with_attrs(fldr.join("K80/K80.fasta"), fldr.join("tree.newick"))
            .build()
            .unwrap();

        let model = SubstModel::<GTR>::new(&[], &[]);
        let cost = SCB::new(model, info).build().unwrap();
        let mut opt = ModelOptimiser::new(cost.clone(), FrequencyOptimisation::Estimated);
        assert_matches!(opt.freq_opt, FrequencyOptimisation::Estimated);

        assert!(opt.optimise_frequencies().is_ok());
        assert_eq!(opt.c.freqs(), &opt.c.empirical_freqs());
    }

    #[test]
    fn estimated_freqs_default_to_empirical_aa() {
        let fldr = Path::new("./data/phyml_protein_example/");
        let info = PIB::with_attrs(fldr.join("seqs.fasta"), fldr.join("wrong_tree.newick"))
            .build()
            .unwrap();

        let model = SubstModel::<WAG>::new(&[], &[]);
        let cost = SCB::new(model, info).build().unwrap();
        let mut opt = ModelOptimiser::new(cost.clone(), FrequencyOptimisation::Estimated);
        assert_matches!(opt.freq_opt, FrequencyOptimisation::Estimated);

        assert!(opt.optimise_frequencies().is_ok());
        assert_eq!(opt.c.freqs(), &opt.c.empirical_freqs());
    }

    #[test]
    fn estimated_freqs_default_to_empirical_pip_dna() {
        let fldr = Path::new("./data/sim");
        let info = PIB::with_attrs(fldr.join("K80/K80.fasta"), fldr.join("tree.newick"))
            .build()
            .unwrap();

        let model = PIPModel::<GTR>::new(&[], &[]);
        let cost = PIPCB::new(model, info).build().unwrap();
        let mut opt = ModelOptimiser::new(cost.clone(), FrequencyOptimisation::Estimated);
        assert_matches!(opt.freq_opt, FrequencyOptimisation::Estimated);

        assert!(opt.optimise_frequencies().is_ok());
        assert_eq!(opt.c.freqs().view((0, 0), (4, 1)), opt.c.empirical_freqs());
    }

    #[test]
    fn estimated_freqs_default_to_empirical_pip_aa() {
        let fldr = Path::new("./data/phyml_protein_example/");
        let info = PIB::with_attrs(fldr.join("seqs.fasta"), fldr.join("wrong_tree.newick"))
            .build()
            .unwrap();

        let model = PIPModel::<WAG>::new(&[], &[]);
        let cost = PIPCB::new(model, info).build().unwrap();
        let mut opt = ModelOptimiser::new(cost.clone(), FrequencyOptimisation::Estimated);
        assert_matches!(opt.freq_opt, FrequencyOptimisation::Estimated);

        assert!(opt.optimise_frequencies().is_ok());
        assert_eq!(opt.c.freqs().view((0, 0), (20, 1)), opt.c.empirical_freqs());
    }
}
