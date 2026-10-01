use std::cell::RefCell;
use std::fmt::Display;
use std::num::NonZeroU64;

use argmin::core::{CostFunction, Executor, IterState, State};
use argmin::solver::brent::BrentOpt;
use log::{debug, info};

use crate::likelihood::TreeSearchCost;
use crate::optimisers::{SingleValOptResult, StopCondition, TreeOptimisationResult};
use crate::tree::NodeIdx;
use crate::{Result, MAX_BLEN};

pub struct BranchOptimiser<C: TreeSearchCost + Display + Clone> {
    pub(crate) stop_condition: StopCondition,
    pub(crate) max_brent_iters: Option<NonZeroU64>,
    pub(crate) c: C,
}

impl<C: TreeSearchCost + Clone + Display> BranchOptimiser<C> {
    pub fn new(cost: C) -> Self {
        Self {
            stop_condition: StopCondition::default(),
            max_brent_iters: None,
            c: cost,
        }
    }

    pub fn with_stop_condition(cost: C, stop_condition: StopCondition) -> Self {
        Self {
            stop_condition,
            max_brent_iters: None,
            c: cost,
        }
    }

    pub fn max_brent_iters(mut self, iters: NonZeroU64) -> Self {
        self.max_brent_iters = Some(iters);
        self
    }

    pub fn run(mut self) -> Result<TreeOptimisationResult<C>> {
        info!("Optimising branch lengths");
        info!("Optimisation stopping condition: {}", self.stop_condition);
        let init_cost = self.c.cost();

        info!("Initial cost: {init_cost}");
        debug!("Initial tree: \n{}", self.c.tree());

        let mut curr_cost = init_cost;
        let mut prev_cost = f64::NEG_INFINITY;
        let mut iterations = 0;
        let mut delta = curr_cost - prev_cost;
        let mut costs = vec![curr_cost];

        while self.stop_condition.should_continue(iterations, delta) {
            iterations += 1;
            info!("Iteration: {iterations}, current cost: {curr_cost}");
            prev_cost = curr_cost;
            curr_cost = self.single_optimisation_iteration()?;
            delta = curr_cost - prev_cost;
            costs.push(curr_cost);
            debug_assert_eq!(curr_cost, self.c.cost());
        }

        info!("Done optimising branch lengths");
        info!("Final cost: {curr_cost}, achieved in {iterations} iteration(s)");
        Ok(TreeOptimisationResult {
            initial_cost: init_cost,
            final_cost: curr_cost,
            iterations,
            costs,
            cost: self.c,
        })
    }

    /// Performs a single iteration of branch length optimisation over all branches in the tree.
    /// Returns the cost after optimising all branches once.
    fn single_optimisation_iteration(&mut self) -> Result<f64> {
        let mut curr_cost = self.c.cost();
        let mut tree = self.c.tree().clone();
        let nodes: Vec<NodeIdx> = tree.iter().map(|node| node.idx).collect();

        for branch in &nodes {
            if tree.root == *branch {
                continue;
            }
            debug!("Node {branch:?}: optimising branch length");
            let blen_opt = optimise_branch_w_iters(&self.c, branch, self.max_brent_iters)?;
            if blen_opt.final_cost > curr_cost {
                curr_cost = blen_opt.final_cost;
                tree.set_blen(branch, blen_opt.value);
                debug!(
                    "    Optimised to {:.5} with cost {curr_cost:.5}",
                    blen_opt.value
                );
            }
            // The branch length may have changed during the optimisation attempt, so the tree
            // should be reset even if the optimisation was unsuccessful.
            self.c.update_tree(tree.clone());
        }

        Ok(curr_cost)
    }
}

pub(crate) fn optimise_branch<T: TreeSearchCost + Clone>(
    cost: &T,
    branch: &NodeIdx,
) -> Result<SingleValOptResult> {
    optimise_branch_w_iters(cost, branch, None)
}

pub(crate) fn optimise_branch_w_iters<T: TreeSearchCost + Clone>(
    cost: &T,
    branch: &NodeIdx,
    max_iters: Option<NonZeroU64>,
) -> Result<SingleValOptResult> {
    let start_blen = cost.tree().node(branch).blen;
    let (min, max) = if start_blen == 0.0 {
        (0.0, 1.0)
    } else {
        (start_blen * 0.1, MAX_BLEN.min(start_blen * 10.0))
    };
    let optimiser = SingleBranchOptimiser {
        cost: RefCell::new(cost.clone()),
        branch: *branch,
    };
    let gss = BrentOpt::new(min, max);

    let res = match max_iters {
        Some(iters) => Executor::new(optimiser, gss)
            .configure(|_| IterState::new().param(start_blen).max_iters(iters.get()))
            .run()?,
        None => Executor::new(optimiser, gss)
            .configure(|_| IterState::new().param(start_blen))
            .run()?,
    };

    let state = res.state();
    Ok(SingleValOptResult {
        final_cost: -state.best_cost,
        value: state.best_param.unwrap(),
    })
}

pub(crate) struct SingleBranchOptimiser<C: TreeSearchCost> {
    pub(crate) cost: RefCell<C>,
    pub(crate) branch: NodeIdx,
}

impl<C: TreeSearchCost> CostFunction for SingleBranchOptimiser<C> {
    type Param = f64;
    type Output = f64;

    fn cost(&self, value: &f64) -> anyhow::Result<f64> {
        let value = if value.is_nan() || value.is_sign_negative() {
            0.0
        } else {
            *value
        };
        let mut tree = self.cost.borrow().tree().clone();
        tree.set_blen(&self.branch, value);
        self.cost.borrow_mut().update_tree(tree);
        Ok(-self.cost.borrow().cost())
    }

    fn parallelize(&self) -> bool {
        true
    }
}

#[cfg(test)]
#[cfg_attr(coverage, coverage(off))]
mod private_tests {
    use std::path::Path;

    use crate::alignment::MSA;
    use crate::likelihood::TreeSearchCost;
    use crate::phylo_info::{PhyloInfo, PhyloInfoBuilder as PIB};
    use crate::pip_model::{PIPCostBuilder as PIPCB, PIPModel};
    use crate::substitution_models::{
        dna_models::*, protein_models::*, QMatrix, QMatrixMaker, SubstModel,
        SubstitutionCostBuilder as SCB,
    };

    use super::*;

    #[cfg(test)]
    fn dna_test_data() -> PhyloInfo<MSA> {
        let fldr = Path::new("./data/sim/");
        PIB::with_attrs(fldr.join("GTR/gtr.fasta"), fldr.join("wrong_tree.newick"))
            .build()
            .unwrap()
    }

    #[cfg(test)]
    fn aa_test_data() -> PhyloInfo<MSA> {
        let fldr = Path::new("./data/phyml_protein_example/");
        PIB::with_attrs(fldr.join("seqs.fasta"), fldr.join("wrong_tree.newick"))
            .build()
            .unwrap()
    }

    #[cfg(test)]
    fn single_iter_pip_template<Q: QMatrix + QMatrixMaker>(info: PhyloInfo<MSA>) {
        let model = PIPModel::<Q>::new(&[], &[]).unwrap();
        let c = PIPCB::new(model.clone(), info.clone()).build().unwrap();
        let init_cost = c.cost();

        let mut optimiser = BranchOptimiser::new(c.clone());
        let optimised_cost = optimiser.single_optimisation_iteration().unwrap();

        assert!(optimised_cost > init_cost);
        assert_eq!(optimised_cost, optimiser.c.cost());

        // Check that branch lengths changed, topology is the same
        let new_info = optimiser.c.info.clone();
        assert_ne!(new_info.tree.length, info.tree.length);
        assert!(new_info.tree.robinson_foulds(&info.tree) == 0);

        // Check that the cost is the same when recomputed from the new info and same model
        let new_cost = PIPCB::new(model, new_info).build().unwrap();
        assert_eq!(new_cost.cost(), optimised_cost);
    }

    #[test]
    fn single_iter_pip_dna() {
        let info = dna_test_data();
        single_iter_pip_template::<JC69>(info.clone());
        single_iter_pip_template::<K80>(info.clone());
        single_iter_pip_template::<HKY>(info.clone());
        single_iter_pip_template::<TN93>(info.clone());
        single_iter_pip_template::<GTR>(info);
    }

    #[test]
    #[cfg_attr(feature = "ci_coverage", ignore)]
    fn single_iteration_pip_aa() {
        let info = aa_test_data();
        single_iter_pip_template::<WAG>(info.clone());
        single_iter_pip_template::<BLOSUM>(info.clone());
        single_iter_pip_template::<HIVB>(info);
    }

    #[cfg(test)]
    fn single_iter_substitution_template<Q: QMatrix + QMatrixMaker>(info: PhyloInfo<MSA>) {
        let model = SubstModel::<Q>::new(&[], &[]).unwrap();
        let c = SCB::new(model.clone(), info.clone()).build().unwrap();
        let init_cost = c.cost();

        let mut optimiser = BranchOptimiser::new(c.clone());
        let optimised_cost = optimiser.single_optimisation_iteration().unwrap();

        assert!(optimised_cost > init_cost);
        assert_eq!(optimised_cost, optimiser.c.cost());

        // Check that branch lengths changed, topology is the same
        let new_info = optimiser.c.info.clone();
        assert_ne!(new_info.tree.length, info.tree.length);
        assert!(new_info.tree.robinson_foulds(&info.tree) == 0);

        // Check that the cost is the same when recomputed from the new info and same model
        let new_cost = SCB::new(model, new_info).build().unwrap();
        assert_eq!(new_cost.cost(), optimised_cost);
    }

    #[test]
    fn single_iteration_substitution_dna() {
        let info = dna_test_data();
        single_iter_substitution_template::<JC69>(info.clone());
        single_iter_substitution_template::<K80>(info.clone());
        single_iter_substitution_template::<HKY>(info.clone());
        single_iter_substitution_template::<TN93>(info.clone());
        single_iter_substitution_template::<GTR>(info);
    }

    #[test]
    #[cfg_attr(feature = "ci_coverage", ignore)]
    fn single_iteration_substitution_aa() {
        let info = aa_test_data();
        single_iter_substitution_template::<WAG>(info.clone());
        single_iter_substitution_template::<BLOSUM>(info.clone());
        single_iter_substitution_template::<HIVB>(info);
    }
}
